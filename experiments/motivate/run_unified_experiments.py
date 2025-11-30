#!/usr/bin/env python3
"""
Unified Experiment: Combines Exp 1, 2, 3, 4a, and 5.

Phase 1: Dataset Profiling (Exp 1 & 3)
- Runs on real dataset samples.
- Measures detailed latencies (Vision, Projector, Prefill, Decode).
- Generates Latency Distribution (Exp 1) and Component Pie Charts (Exp 3).

Phase 2: Controlled Scaling (Exp 2, 4a, 5)
- Runs on resized dummy images to vary vision tokens.
- Measures FLOPs and Latencies.
- Generates FLOPs vs Latency (Exp 2) and Token vs Latency (Exp 4a) plots.
- Outputs comparison table (Exp 5).
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy import stats
from PIL import Image
from transformers import AutoProcessor

from experiments.motivate.base_experiment import BaseExperiment, Timer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class UnifiedExperiment(BaseExperiment):
    """Unified experiment for Molmo profiling."""

    def measure_detailed_latencies(
        self,
        batch: Dict[str, Any],
        num_runs: int = 5,
        is_generation: bool = False
    ) -> Dict[str, float]:
        """Measure latencies for all components."""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        results = {}
        
        # 1. Measure Vision Encoder (ViT only)
        if "images" in batch and batch["images"] is not None:
            vision_backbone = self.model.vision_backbone
            
            # Warmup only if num_runs > 1
            if num_runs > 1:
                with torch.inference_mode():
                    _ = vision_backbone.encode_image(batch["images"])
            
            # Measure ViT
            latencies_vision_encoder = []
            for _ in range(num_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                start = time.perf_counter()
                with torch.inference_mode():
                    _ = vision_backbone.encode_image(batch["images"])
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                latencies_vision_encoder.append((time.perf_counter() - start) * 1000)
            results["T_vision_encoder"] = np.mean(latencies_vision_encoder)
            
            # 2. Measure Total Vision (ViT + Projector)
            # Warmup only if num_runs > 1
            if num_runs > 1:
                with torch.inference_mode():
                    _ = vision_backbone(batch["images"], batch.get("image_masks"))
            
            # Measure Total Vision
            latencies_vision_total = []
            for _ in range(num_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                start = time.perf_counter()
                with torch.inference_mode():
                    _ = vision_backbone(batch["images"], batch.get("image_masks"))
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                latencies_vision_total.append((time.perf_counter() - start) * 1000)
            results["T_vision_total"] = np.mean(latencies_vision_total)
            
            # Calculate Projector
            results["T_projector"] = max(0.0, results["T_vision_total"] - results["T_vision_encoder"])
            
        else:
            results["T_vision_encoder"] = 0.0
            results["T_vision_total"] = 0.0
            results["T_projector"] = 0.0

        # 3. Measure LLM Prefill
        input_ids = batch["input_ids"]
        
        # Warmup only if num_runs > 1 (for stable averaging), otherwise skip for profiling speed
        if num_runs > 1:
            with torch.inference_mode():
                 _ = self.model(
                    input_ids=input_ids,
                    images=batch.get("images"),
                    image_masks=batch.get("image_masks"),
                    image_input_idx=batch.get("image_input_idx"),
                )
        
        latencies_LLM_prefill = []
        for _ in range(num_runs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    _ = self.model(
                        input_ids=input_ids,
                        images=batch.get("images"),
                        image_masks=batch.get("image_masks"),
                        image_input_idx=batch.get("image_input_idx"),
                    )
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            latencies_LLM_prefill.append((time.perf_counter() - start) * 1000)
        results["T_LLM_prefill"] = np.mean(latencies_LLM_prefill)
        
        # 4. Measure LLM Decode
        if is_generation:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            
            with torch.inference_mode():
                _ = self.model.generate(
                    input_ids=batch["input_ids"],
                    images=batch.get("images"),
                    image_masks=batch.get("image_masks"),
                    image_input_idx=batch.get("image_input_idx"),
                    max_steps=10, # Short generation for profiling
                )
                
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            t_generate_total = (time.perf_counter() - start) * 1000
            
            # T_decode = T_generate_total - T_prefill_step (which is roughly T_vision + T_LLM_prefill)
            results["T_LLM_decode"] = max(0.0, t_generate_total - results["T_vision_total"] - results["T_LLM_prefill"])
            results["T_total"] = t_generate_total
        else:
            results["T_LLM_decode"] = 0.0
            results["T_total"] = results["T_vision_total"] + results["T_LLM_prefill"]
        
        return results

    def run_dataset_profiling(self, dataset_name, split, num_samples):
        """Phase 1: Dataset Profiling (Exp 1 & 3)."""
        with Timer("Phase 1: Dataset Profiling"):
            log.info("Starting Phase 1: Dataset Profiling...")
            
            dataloader = self.build_dataloader(dataset_name, split, batch_size=1, max_steps=num_samples)
            
            results = []
            for batch_idx, batch in enumerate(tqdm(dataloader, total=min(num_samples, len(dataloader)))):
                if batch_idx >= num_samples: break
                
                # Measure latencies
                lats = self.measure_detailed_latencies(batch, num_runs=1, is_generation=True) # Speed up
                
                # Calculate FLOPs
                flops = self.count_flops(batch, output_length=10) # Match max_steps=10
                
                # Calculate Vision Tokens & Crops
                if "images" in batch and batch["images"] is not None:
                    # batch["images"] shape is (B, N_crops, C, H, W)
                    num_crops = batch["images"].shape[1]
                    num_vision_tokens = num_crops * 576
                else:
                    num_crops = 0
                    num_vision_tokens = 0
                
                # Calculate Language Tokens
                num_input_tokens = batch["input_ids"].shape[1]
                num_total_tokens = num_vision_tokens + num_input_tokens

                # Merge all metrics
                res = {
                    "num_crops": num_crops,
                    "num_vision_tokens": num_vision_tokens,
                    "num_input_tokens": num_input_tokens,
                    "num_total_tokens": num_total_tokens,
                    **lats,
                    **flops
                }
                results.append(res)
                
            # Analyze
            avg_results = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
            log.info(f"Average Latencies: {avg_results}")
            
            # Plot Exp 1 (Histogram)
            self.plot_latency_distribution(results, dataset_name, split)
            
            # Plot Exp 3 (Pie Chart)
            self.plot_component_pie_chart(avg_results, dataset_name, split)
            
            # Save Results
            self.save_results({"results": results}, "phase1_dataset_profiling.json")
            
            return results

    def run_scaling_profiling(self, dataset_name, split):
        """Phase 2: Controlled Scaling (Exp 2, 4a, 5)."""
        with Timer("Phase 2: Controlled Scaling"):
            log.info("Starting Phase 2: Controlled Scaling...")
            
            # Resolutions: We will generate them dynamically below
            
            # Processor
            model_path = self.model_path
            if model_path.startswith("hf:"):
                model_path = model_path[3:]
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            base_image = Image.new('RGB', (2000, 2000), color=(100, 150, 200))
            text = "Describe this image."
            
            # Global Warmup (Critical to avoid 336x336 anomaly)
            log.info("Performing global warmup to absorb initialization costs...")
            warmup_img = base_image.resize((336, 336))
            warmup_inputs = processor.process(text=text, images=warmup_img)
            warmup_inputs = {k: v.to(self.device).unsqueeze(0) for k, v in warmup_inputs.items()}
            with torch.inference_mode():
                # Warmup Vision
                _ = self.model.vision_backbone.encode_image(warmup_inputs["images"])
                # Warmup LLM
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    _ = self.model(
                        input_ids=warmup_inputs["input_ids"],
                        images=warmup_inputs.get("images"),
                        image_masks=warmup_inputs.get("image_masks"),
                        image_input_idx=warmup_inputs.get("image_input_idx"),
                    )
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            log.info("Global warmup complete.")
            
            # Generate rectangular images to trigger specific grid sizes (1x1, 1x2, ..., 1x12)
            # This gives us fine-grained control over vision tokens:
            # k=1 -> 1x1 grid (+1 global) = 2 crops
            # k=2 -> 1x2 grid (+1 global) = 3 crops
            # ...
            # k=12 -> 1x12 grid (+1 global) = 13 crops
            max_crops = 12
            
            results = []
            for k in range(1, max_crops + 1):
                w, h = 336, 336 * k
                with Timer(f"Grid 1x{k} ({w}x{h})"):
                    log.info(f"Testing Grid 1x{k} (Resolution {w}x{h})...")
                    img = base_image.resize((w, h))
                    inputs = processor.process(text=text, images=img)
                    inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}
                    
                    # Measure
                    lats = self.measure_detailed_latencies(inputs, num_runs=10)
                    
                    # FLOPs
                    flops = self.count_flops(inputs, output_length=1)
                    
                    # Tokens
                    num_vision_tokens = inputs['images'].shape[1] * 576 
                    num_crops = inputs['images'].shape[1]
                    num_input_tokens = inputs['input_ids'].shape[1]
                    num_total_tokens = num_vision_tokens + num_input_tokens
                    
                    res = {
                        "resolution": f"{w}x{h}",
                        "grid": f"1x{k}",
                        "num_crops": num_crops,
                        "num_vision_tokens": num_vision_tokens,
                        "num_input_tokens": num_input_tokens,
                        "num_total_tokens": num_total_tokens,
                        **lats,
                        **flops
                    }
                    results.append(res)
                
            # Plot Exp 2 (FLOPs vs Latency)
            self.plot_flops_vs_latency(results, dataset_name, split)
            
            # Plot Exp 4a (Tokens vs Latency)
            self.plot_tokens_vs_latency(results, dataset_name, split)
            
            # Save Results
            self.save_results({"results": results}, "phase2_scaling.json")
            
            return results

    def plot_latency_distribution(self, results, dataset_name, split):
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        latencies = [r["T_total"] for r in results]
        plt.figure(figsize=(10, 6))
        plt.hist(latencies, bins=30, alpha=0.7, color='blue')
        plt.title(f"Latency Distribution (Exp 1) - {dataset_name}")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Count")
        plt.savefig(fig_dir / f"exp1_dist_{dataset_name}.png")
        plt.close()

    def plot_component_pie_chart(self, avg_results, dataset_name, split):
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        labels = ["Vision Encoder", "Projector", "LLM Prefill"]
        values = [avg_results["T_vision_encoder"], avg_results["T_projector"], avg_results["T_LLM_prefill"]]
        
        plt.figure(figsize=(8, 8))
        plt.pie(values, labels=labels, autopct='%1.1f%%')
        plt.title(f"Component Latency (Exp 3) - {dataset_name}")
        plt.savefig(fig_dir / f"exp3_pie_{dataset_name}.png")
        plt.close()

    def plot_flops_vs_latency(self, results, dataset_name, split):
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        flops = [r["flops_total"] for r in results]
        lats = [r["T_total"] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(flops, lats)
        plt.title(f"FLOPs vs Latency (Exp 2)")
        plt.xlabel("FLOPs")
        plt.ylabel("Latency (ms)")
        plt.savefig(fig_dir / f"exp2_flops_{dataset_name}.png")
        plt.close()

    def plot_tokens_vs_latency(self, results, dataset_name, split):
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        tokens = [r["num_vision_tokens"] for r in results]
        lats = [r["T_total"] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(tokens, lats, 'o-')
        plt.title(f"Vision Tokens vs Latency (Exp 4a)")
        plt.xlabel("Vision Tokens")
        plt.ylabel("Latency (ms)")
        plt.savefig(fig_dir / f"exp4a_tokens_{dataset_name}.png")
        plt.close()

    def run(self, dataset_name, split, num_samples):
        self.run_dataset_profiling(dataset_name, split, num_samples)
        self.run_scaling_profiling(dataset_name, split)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", default="coco_2014_vqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", default="./results/unified")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--phase1_only", action="store_true", help="Run only Phase 1 (Dataset Profiling)")
    parser.add_argument("--phase2_only", action="store_true", help="Run only Phase 2 (Controlled Scaling)")
    args = parser.parse_args()
    
    # Default to running both if neither is specified
    if not args.phase1_only and not args.phase2_only:
        run_phase1 = True
        run_phase2 = True
    else:
        run_phase1 = args.phase1_only
        run_phase2 = args.phase2_only
    
    exp = UnifiedExperiment(args.model_path, args.device, args.output_dir, args.hf_cache_dir)
    
    if run_phase1:
        exp.run_dataset_profiling(args.dataset, args.split, args.num_samples)
    
    if run_phase2:
        exp.run_scaling_profiling(args.dataset, args.split)

if __name__ == "__main__":
    main()
