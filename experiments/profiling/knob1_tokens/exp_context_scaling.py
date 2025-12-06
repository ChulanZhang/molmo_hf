
import argparse
import logging
import sys
import os
from typing import Dict, List

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())
from experiments.base_experiment import BaseExperiment, Timer

log = logging.getLogger(__name__)

class ContextScalingExperiment(BaseExperiment):
    """
    Profiling Experiment 1: Vision Tokens vs Latency (corresponds to Motivation Exp 3)
    
    Goal: Research the impact of input vision tokens on Prefill latency.
    Method: Resize dummy images to different resolutions to trigger different crop counts.
    """
    
    def run(
        self,
        dataset_name: str = None,
        split: str = "validation",
        num_samples: int = 5000,
        batch_size: int = 1,
        max_new_tokens: int = 128,
        max_crops_list: List[int] = None,
        # Legacy parameters for dummy image mode
        max_grid_size: int = 12,
        num_runs: int = 1,
    ):
        """
        Run context scaling experiment (vision tokens scaling).
        
        Args:
            dataset_name: Dataset name (e.g., "coco_2014_vqa"). If None, uses dummy images.
            split: Dataset split (default: "validation")
            num_samples: Number of samples to measure (default: 5000 for dataset mode, 50 for dummy mode)
            batch_size: Batch size (default: 1 for latency measurement)
            max_new_tokens: Maximum tokens to generate
            max_crops_list: List of max_crops values to test (default: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
            max_grid_size: Maximum grid size (1x12) - for dummy image mode only
            num_runs: Number of runs per resolution - for dummy image mode only
        """
        # Determine mode: dataset mode or dummy image mode
        use_dataset = dataset_name is not None
        
        if use_dataset:
            # Dataset mode: use VQA v2 validation set with different max_crops
            if max_crops_list is None:
                max_crops_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            
            log.info(f"Testing {len(max_crops_list)} max_crops configurations: {max_crops_list}")
            log.info(f"Measuring latency on {num_samples} samples from {dataset_name}/{split}")
            
            results_data = []
            
            for max_crops in max_crops_list:
                log.info(f"=" * 80)
                log.info(f"Testing max_crops={max_crops}")
                log.info(f"=" * 80)
                
                # Temporarily modify model config
                original_max_crops = self.model.config.max_crops
                self.model.config.max_crops = max_crops
                
                # Build dataloader
                dataloader = self.build_dataloader(
                    dataset_name=dataset_name,
                    split=split,
                    batch_size=batch_size,
                    max_steps=num_samples,
                    shuffle=False,
                )
                
                all_latencies = []
                per_sample_results = []
                
                # Warmup
                log.info("Warming up...")
                warmup_batch = next(iter(dataloader))
                for _ in range(self.num_warmup):
                    self.measure_inference_latency(
                        warmup_batch,
                        max_new_tokens=max_new_tokens,
                        measure_components=True,
                        num_runs=1,
                    )
                
                # Measure latency
                log.info(f"Measuring latency for max_crops={max_crops}...")
                for sample_idx, batch in enumerate(tqdm(dataloader, total=min(num_samples, len(dataloader)))):
                    if sample_idx >= num_samples:
                        break
                    
                    metrics = self.measure_inference_latency(
                        batch,
                        max_new_tokens=max_new_tokens,
                        measure_components=True,
                        num_runs=1,
                    )
                    
                    total_latency = metrics.get("T_total", 0.0)
                    all_latencies.append(total_latency)
                    
                    num_crops = batch["images"].shape[1] if "images" in batch and batch["images"] is not None else 0
                    num_vision_tokens = metrics.get("num_vision_tokens", 0)
                    
                    per_sample_results.append({
                        "sample_id": sample_idx,
                        "max_crops": max_crops,
                        "actual_crops": num_crops,
                        "num_vision_tokens": num_vision_tokens,
                        **metrics
                    })
                
                # Restore original max_crops
                self.model.config.max_crops = original_max_crops
                
                # Compute statistics
                stats = self.compute_statistics(all_latencies)
                stats["max_crops"] = max_crops
                stats["num_vision_tokens"] = np.mean([r["num_vision_tokens"] for r in per_sample_results]) if per_sample_results else 0
                stats["per_sample_results"] = per_sample_results
                
                results_data.append(stats)
                
                log.info(f"max_crops={max_crops}: Mean Latency={stats['mean']:.2f}ms, P50={stats['P50']:.2f}ms, "
                        f"Vision Tokens={stats['num_vision_tokens']:.0f}")
        else:
            # Dummy image mode: original behavior
            processor = self.processor
            base_image = Image.new("RGB", (2000, 2000), color=(100, 150, 200))
            text = "Describe this image."

            # Global Warmup
            log.info("Performing global warmup...")
            warmup_img = base_image.resize((336, 336))
            warmup_inputs = processor.process(text=text, images=warmup_img)
            warmup_inputs = {k: v.to(self.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in warmup_inputs.items()}
            if isinstance(warmup_inputs.get("images"), torch.Tensor):
                warmup_inputs["images"] = warmup_inputs["images"].unsqueeze(0)
            if isinstance(warmup_inputs.get("image_masks"), torch.Tensor):
                warmup_inputs["image_masks"] = warmup_inputs["image_masks"].unsqueeze(0)
            if isinstance(warmup_inputs.get("image_input_idx"), torch.Tensor):
                warmup_inputs["image_input_idx"] = warmup_inputs["image_input_idx"].unsqueeze(0)
            
            with torch.inference_mode():
                _ = self.model.model.vision_backbone.encode_image(warmup_inputs["images"])
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    _ = self.model(
                        input_ids=warmup_inputs["input_ids"],
                        images=warmup_inputs.get("images"),
                        image_masks=warmup_inputs.get("image_masks"),
                        image_input_idx=warmup_inputs.get("image_input_idx"),
                    )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            log.info("Global warmup complete.")

            # Calculate exact resolutions for precise tiling control
            crop_window_size = 224  # 16 patches * 14 pixels per patch
            total_margin_pixels = 112  # 8 patches * 14 pixels (4+4 margins)
            
            # Define tiling configurations to cover crop counts
            # Format: (tiling_i, tiling_j, description)
            tiling_configs = [
                (1, 1, "1×1"),   # 2 crops
                (1, 2, "1×2"),   # 3 crops
                (1, 3, "1×3"),   # 4 crops
                (1, 4, "1×4"),   # 5 crops
                (1, 5, "1×5"),   # 6 crops
                (1, 6, "1×6"),   # 7 crops
                (1, 7, "1×7"),   # 8 crops
                (1, 8, "1×8"),   # 9 crops
                (1, 9, "1×9"),   # 10 crops
                (1, 10, "1×10"), # 11 crops
                (1, 11, "1×11"), # 12 crops
                (1, 12, "1×12"), # 13 crops
            ]
            
            # Limit to max_grid_size
            tiling_configs = [tc for tc in tiling_configs if tc[1] <= max_grid_size]
            if num_samples < len(tiling_configs):
                # Sample evenly
                step = len(tiling_configs) // num_samples
                tiling_configs = tiling_configs[::max(1, step)][:num_samples]
            
            results_data = []
            
            log.info(f"Testing {len(tiling_configs)} tiling configurations...")
            
            for tiling_i, tiling_j, desc in tqdm(tiling_configs, desc="Vision Tokens Scaling"):
                # Calculate exact resolution for this tiling
                w = tiling_j * crop_window_size + total_margin_pixels
                h = tiling_i * crop_window_size + total_margin_pixels
                expected_crops = tiling_i * tiling_j + 1
                
                log.info(f"Testing Grid {desc} (Resolution {w}x{h}, expected crops: {expected_crops})...")
                img = base_image.resize((w, h))
                inputs = processor.process(text=text, images=img)
                
                # Ensure batch dimension
                if inputs["input_ids"].ndim == 1:
                    inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
                if "images" in inputs and inputs["images"] is not None:
                    if inputs["images"].ndim == 4:
                        inputs["images"] = inputs["images"].unsqueeze(0)
                if "image_masks" in inputs and inputs["image_masks"] is not None:
                    if inputs["image_masks"].ndim == 2:
                        inputs["image_masks"] = inputs["image_masks"].unsqueeze(0)
                if "image_input_idx" in inputs and inputs["image_input_idx"] is not None:
                    if inputs["image_input_idx"].ndim == 2:
                        inputs["image_input_idx"] = inputs["image_input_idx"].unsqueeze(0)
                
                # Move to device
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Run measurements
                latencies_prefill = []
                per_sample_results = []
                
                # Warmup
                for _ in range(self.num_warmup):
                    self.measure_inference_latency(inputs, max_new_tokens=1, measure_components=True)
                
                for sample_idx in range(num_runs):
                    metrics = self.measure_inference_latency(
                        inputs,
                        max_new_tokens=1,  # Prefill only
                        measure_components=True,
                        num_runs=1,
                    )
                    latencies_prefill.append(metrics["T_LLM_prefill"])
                    
                    # Save per-sample result
                    per_sample_results.append({
                        "sample_id": sample_idx,
                        "resolution": f"{w}x{h}",
                        "grid": desc,
                        "num_crops": inputs["images"].shape[1] if "images" in inputs else 0,
                        "num_vision_tokens": metrics.get("num_vision_tokens", 0),
                        "num_input_text_tokens": metrics.get("num_input_text_tokens", 0),
                        **metrics  # Include all metrics from measure_inference_latency
                    })
                
                stats = self.compute_statistics(latencies_prefill)
                stats["resolution"] = f"{w}x{h}"
                stats["grid"] = desc
                stats["num_crops"] = inputs["images"].shape[1] if "images" in inputs else 0
                stats["num_vision_tokens"] = per_sample_results[0]["num_vision_tokens"] if per_sample_results else 0
                stats["per_sample_results"] = per_sample_results
                results_data.append(stats)
                
                log.info(f"Grid {desc}: Vision Tokens={stats['num_vision_tokens']}, P50 Prefill={stats['P50']:.2f}ms")

        # Save results with per-sample data
        all_samples = []
        summary = []
        
        for config_result in results_data:
            # Extract summary stats (exclude per_sample_results)
            summary_entry = {k: v for k, v in config_result.items() if k != "per_sample_results"}
            summary.append(summary_entry)
            
            # Collect all per-sample results
            if "per_sample_results" in config_result:
                all_samples.extend(config_result["per_sample_results"])
        
        final_results = {
            "summary": summary,
            "all_samples": all_samples
        }
        
        self.save_results(final_results, "exp1_context_scaling_results.json")
        log.info(f"Total samples: {len(all_samples)}")

def main():
    parser = argparse.ArgumentParser(description="Run Context Scaling Experiment (Vision Tokens)")
    parser.add_argument("--model_path", type=str, default="checkpoints",
                        help="Path to model checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="./results/profiling/context_scaling")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset name (e.g., 'coco_2014_vqa'). If None, uses dummy images.")
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split (default: validation)")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of samples to measure (default: 5000 for dataset mode, 12 for dummy mode)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1 for latency measurement)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--max_crops_list", type=int, nargs="+", default=None,
                        help="List of max_crops values to test (default: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])")
    parser.add_argument("--max_grid_size", type=int, default=12,
                        help="Maximum grid size (1x12) - for dummy image mode only")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of runs per resolution - for dummy image mode only")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = ContextScalingExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    experiment.run(
        dataset_name=args.dataset_name,
        split=args.split,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_crops_list=args.max_crops_list,
        max_grid_size=args.max_grid_size,
        num_runs=args.num_runs,
    )

if __name__ == "__main__":
    main()
