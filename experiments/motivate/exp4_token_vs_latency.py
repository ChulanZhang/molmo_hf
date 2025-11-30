#!/usr/bin/env python3
"""
Experiment 4: Token 数量 vs Latency (Vision vs Language)

4A: Vision Tokens vs Latency (固定文本)
4B: Language Tokens vs Latency (固定图像)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

from .base_experiment import BaseExperiment, Timer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TokenVsLatencyExperiment(BaseExperiment):
    """Experiment 4: Analyze token count vs latency."""
    
    def run_4a_vision_tokens(
        self,
        dataset_name: str,
        split: str = "validation",
        num_samples: int = 500,
        max_new_tokens: int = 12,
    ):
        """4A: Vision Tokens vs Latency (Controlled Resizing)."""
        with Timer("Experiment 4A: Vision Tokens vs Latency"):
            log.info("Starting Experiment 4A: Vision Tokens vs Latency (Controlled Resizing)")
            
            # Load processor
            # Use self.processor loaded in BaseExperiment
            processor = self.processor

            # Define target resolutions (scales) to force different crop counts
            # Molmo uses 336x336 crops.
            # 336 -> 1 crop
            # 672 -> 4 crops
            # 1008 -> 9 crops
            # etc.
            resolutions = [
                (336, 336),
                (336*2, 336*2),
                (336*3, 336*3),
                (336*4, 336*4),
                (336*5, 336*5), # Max might be 12 crops depending on config
            ]
            
            # Create a dummy image (or use a real one if available)
            # Using a high-res dummy image to allow downscaling/cropping
            base_image = Image.new('RGB', (2000, 2000), color=(100, 150, 200))
            
            results = []
            text_prompt = "Describe this image."
            
            log.info("Running controlled resolution test...")
            
            # Warmup
            log.info("Warming up...")
            inputs = processor.process(text=text_prompt, images=base_image.resize((336, 336)))
            inputs = {k: v.to(self.device, non_blocking=True).unsqueeze(0) for k, v in inputs.items()}
            # Ensure keys match what model expects (MolmoProcessor usually returns correct keys)
            # But we might need to map 'pixel_values' to 'images' if using standard HF processor
            # The Molmo processor from allenai/MolmoE-1B-0924 returns 'images', 'input_ids', etc.
            
            # Check keys
            log.info(f"Processor output keys: {inputs.keys()}")
            
            self.measure_inference_latency(inputs, max_new_tokens=max_new_tokens)
            
            for width, height in resolutions:
                with Timer(f"Resolution {width}x{height}"):
                    log.info(f"Testing resolution: {width}x{height}")
                    
                    # Resize image
                    resized_image = base_image.resize((width, height))
                    
                    # Process
                    inputs = processor.process(text=text_prompt, images=resized_image)
                    inputs = {k: v.to(self.device, non_blocking=True).unsqueeze(0) for k, v in inputs.items()}
                    
                    # Measure latency
                    # We run multiple times per resolution to get stable numbers
                    for i in range(5): # Run 5 times per resolution
                        latency_results = self.measure_inference_latency(
                            inputs,
                            max_new_tokens=max_new_tokens,
                            measure_components=True,
                        )
                        
                        results.append({
                            "query_id": f"{width}x{height}_{i}",
                            "resolution": f"{width}x{height}",
                            "num_vision_tokens": latency_results.get("num_vision_tokens", 0),
                            "num_input_text_tokens": latency_results.get("num_input_text_tokens", 0),
                            "num_output_tokens": latency_results.get("num_output_tokens", 0),
                            "T_total": latency_results.get("T_total", 0),
                            "T_vision": latency_results.get("T_vision", 0),
                            "T_LLM_prefill": latency_results.get("T_LLM_prefill", 0) if "T_LLM_prefill" in latency_results else 0,
                            "T_LLM_decode": latency_results.get("T_LLM_decode", 0) if "T_LLM_decode" in latency_results else 0,
                        })
            
            # Analyze by vision token count
            self.analyze_vision_tokens(results, dataset_name, split)
            
            return results
    
    def run_4b_language_tokens(
        self,
        dataset_name: str,
        split: str = "validation",
        num_samples: int = 500,
        max_new_tokens_list: List[int] = [12, 32, 64, 128, 256, 512],
    ):
        """4B: Language Tokens vs Latency (fixed image)."""
        with Timer("Experiment 4B: Language Tokens vs Latency"):
            log.info("Starting Experiment 4B: Language Tokens vs Latency")
            
            dataloader = self.build_dataloader(
                dataset_name=dataset_name,
                split=split,
                batch_size=1,
                max_steps=num_samples,
            )
            
            # Use a fixed set of images
            fixed_batches = []
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= min(10, num_samples):  # Use first 10 images
                    break
                fixed_batches.append(batch)
            
            if not fixed_batches:
                log.error("No batches available!")
                return []
            
            # Warmup
            log.info("Warming up...")
            for _ in range(self.num_warmup):
                _ = self.measure_inference_latency(fixed_batches[0], max_new_tokens=max(max_new_tokens_list))
            
            # Collect measurements for different output lengths
            log.info("Collecting measurements for different output lengths...")
            results = []
            
            for max_tokens in max_new_tokens_list:
                with Timer(f"Max Tokens {max_tokens}"):
                    log.info(f"Testing with max_new_tokens={max_tokens}")
                    for batch_idx, batch in enumerate(tqdm(fixed_batches, desc=f"max_tokens={max_tokens}")):
                        # Measure latency
                        latency_results = self.measure_inference_latency(
                            batch,
                            max_new_tokens=max_tokens,
                            measure_components=True,
                        )
                        
                        results.append({
                            "query_id": batch_idx,
                            "max_new_tokens": max_tokens,
                            "num_crops": latency_results.get("num_crops", 0),
                            "num_vision_tokens": latency_results.get("num_vision_tokens", 0),
                            "num_input_tokens": latency_results.get("num_input_text_tokens", 0),
                            "num_total_tokens": latency_results.get("num_vision_tokens", 0) + latency_results.get("num_input_text_tokens", 0),
                            "T_total": latency_results.get("T_total", 0),
                            "T_vision": latency_results.get("T_vision", 0),
                            "T_LLM_prefill": latency_results.get("T_LLM_prefill", 0) if "T_LLM_prefill" in latency_results else 0,
                            "T_LLM_decode": latency_results.get("T_LLM_decode", 0) if "T_LLM_decode" in latency_results else 0,
                            # FLOPs would need to be calculated here if BaseExperiment supports it directly or we call count_flops
                            # Since BaseExperiment has count_flops, let's use it
                        })
                        
                        # Add FLOPs
                        flops = self.count_flops(batch, output_length=max_tokens)
                        results[-1].update(flops)
            
            # Analyze by language token count
            self.analyze_language_tokens(results, dataset_name, split)
            
            return results
    
    def analyze_vision_tokens(self, results: List[Dict], dataset_name: str, split: str):
        """Analyze and plot vision tokens vs latency."""
        vision_tokens = [r["num_vision_tokens"] for r in results]
        latencies = [r["T_total"] for r in results]
        prefill_latencies = [r.get("T_vision", 0) + r.get("T_LLM_prefill", 0) for r in results]
        
        # Group by vision token count
        token_groups = {}
        for r in results:
            v_tokens = r["num_vision_tokens"]
            if v_tokens not in token_groups:
                token_groups[v_tokens] = []
            token_groups[v_tokens].append(r)
        
        # Plot 1: Total Latency
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Scatter plot
        ax1 = axes[0]
        ax1.scatter(vision_tokens, latencies, alpha=0.5, s=20)
        ax1.set_xlabel('Number of Vision Tokens', fontsize=12)
        ax1.set_ylabel('Total Latency (ms)', fontsize=12)
        ax1.set_title(f'Vision Tokens vs Total Latency - {dataset_name}/{split}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Grouped analysis
        ax2 = axes[1]
        token_counts = sorted(token_groups.keys())
        means = [np.mean([r["T_total"] for r in token_groups[tc]]) for tc in token_counts]
        stds = [np.std([r["T_total"] for r in token_groups[tc]]) for tc in token_counts]
        counts = [len(token_groups[tc]) for tc in token_counts]
        
        ax2.errorbar(token_counts, means, yerr=stds, fmt='o-', capsize=5, linewidth=2, markersize=8)
        for tc, mean, count in zip(token_counts, means, counts):
            ax2.annotate(f'n={count}', (tc, mean), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Number of Vision Tokens', fontsize=12)
        ax2.set_ylabel('Mean Latency (ms)', fontsize=12)
        ax2.set_title('Mean Latency by Vision Token Count', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f"exp4a_vision_tokens_vs_latency_{dataset_name}_{split}.png", 
                   dpi=300, bbox_inches='tight')
        log.info(f"Plot saved to {fig_dir / f'exp4a_vision_tokens_vs_latency_{dataset_name}_{split}.png'}")
        plt.close()

        # Plot 2: Prefill Latency (TTFT)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Scatter plot
        ax1 = axes[0]
        ax1.scatter(vision_tokens, prefill_latencies, alpha=0.5, s=20, color='green')
        ax1.set_xlabel('Number of Vision Tokens', fontsize=12)
        ax1.set_ylabel('Prefill Latency (ms)', fontsize=12)
        ax1.set_title(f'Vision Tokens vs Prefill Latency (TTFT) - {dataset_name}/{split}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Grouped analysis
        ax2 = axes[1]
        prefill_means = [np.mean([r.get("T_vision", 0) + r.get("T_LLM_prefill", 0) for r in token_groups[tc]]) for tc in token_counts]
        prefill_stds = [np.std([r.get("T_vision", 0) + r.get("T_LLM_prefill", 0) for r in token_groups[tc]]) for tc in token_counts]
        
        ax2.errorbar(token_counts, prefill_means, yerr=prefill_stds, fmt='o-', capsize=5, linewidth=2, markersize=8, color='green')
        for tc, mean, count in zip(token_counts, prefill_means, counts):
            ax2.annotate(f'n={count}', (tc, mean), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Number of Vision Tokens', fontsize=12)
        ax2.set_ylabel('Mean Prefill Latency (ms)', fontsize=12)
        ax2.set_title('Mean Prefill Latency by Vision Token Count', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f"exp4a_vision_tokens_vs_prefill_{dataset_name}_{split}.png", 
                   dpi=300, bbox_inches='tight')
        log.info(f"Plot saved to {fig_dir / f'exp4a_vision_tokens_vs_prefill_{dataset_name}_{split}.png'}")
        plt.close()
        
        # Save results
        results_dict = {
            "experiment_name": "exp4a_vision_tokens",
            "dataset": dataset_name,
            "split": split,
            "results": results,
            "token_groups": {str(k): len(v) for k, v in token_groups.items()},
        }
        self.save_results(results_dict, f"exp4a_{dataset_name}_{split}.json")
    
    def analyze_language_tokens(self, results: List[Dict], dataset_name: str, split: str):
        """Analyze and plot language tokens vs latency."""
        # Group by max_new_tokens
        groups = {}
        for r in results:
            mt = r["max_new_tokens"]
            if mt not in groups:
                groups[mt] = []
            groups[mt].append(r)
            
        max_tokens_list = sorted(groups.keys())
        prefill_means = [np.mean([r["T_LLM_prefill"] for r in groups[mt]]) for mt in max_tokens_list]
        decode_means = [np.mean([r["T_LLM_decode"] for r in groups[mt]]) for mt in max_tokens_list]
        total_means = [np.mean([r["T_total"] for r in groups[mt]]) for mt in max_tokens_list]
        
        # Plot Stacked Bar Chart
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        # Create bars
        x = np.arange(len(max_tokens_list))
        width = 0.6
        
        p1 = plt.bar(x, prefill_means, width, label='Prefill Latency', color='skyblue', alpha=0.8)
        p2 = plt.bar(x, decode_means, width, bottom=prefill_means, label='Decode Latency', color='orange', alpha=0.8)
        
        # Add labels and title
        plt.xlabel('Max New Tokens', fontsize=12)
        plt.ylabel('Latency (ms)', fontsize=12)
        plt.title(f'Language Generation Latency Breakdown (Exp 4B) - {dataset_name}/{split}', fontsize=14)
        plt.xticks(x, max_tokens_list)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, (p, d, t) in enumerate(zip(prefill_means, decode_means, total_means)):
            plt.text(i, t + 5, f"{t:.1f}ms", ha='center', va='bottom', fontsize=9)
            plt.text(i, p/2, f"{p:.1f}", ha='center', va='center', fontsize=8, color='black')
            if d > 10:
                plt.text(i, p + d/2, f"{d:.1f}", ha='center', va='center', fontsize=8, color='black')
        
        plt.tight_layout()
        plt.savefig(fig_dir / f"exp4b_language_tokens_vs_latency_{dataset_name}_{split}.png", 
                   dpi=300, bbox_inches='tight')
        log.info(f"Plot saved to {fig_dir / f'exp4b_language_tokens_vs_latency_{dataset_name}_{split}.png'}")
        plt.close()
        
        # Save results
        results_dict = {
            "experiment_name": "exp4b_language_tokens",
            "dataset": dataset_name,
            "split": split,
            "results": results,
            "summary": {
                str(mt): {
                    "prefill_mean": float(p),
                    "decode_mean": float(d),
                    "total_mean": float(t)
                } for mt, p, d, t in zip(max_tokens_list, prefill_means, decode_means, total_means)
            }
        }
        self.save_results(results_dict, f"exp4b_{dataset_name}_{split}.json")
    
    def run(
        self,
        dataset_name: str,
        split: str = "validation",
        num_samples: int = 500,
        run_4a: bool = True,
        run_4b: bool = True,
    ):
        """Run both 4A and 4B experiments."""
        all_results = {}
        
        if run_4a:
            log.info("=" * 60)
            log.info("Running Experiment 4A: Vision Tokens vs Latency")
            log.info("=" * 60)
            results_4a = self.run_4a_vision_tokens(dataset_name, split, num_samples)
            all_results["4a"] = results_4a
        
        if run_4b:
            log.info("=" * 60)
            log.info("Running Experiment 4B: Language Tokens vs Latency")
            log.info("=" * 60)
            results_4b = self.run_4b_language_tokens(dataset_name, split, num_samples)
            all_results["4b"] = results_4b
        
        log.info("Experiment 4 completed!")
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Token vs Latency")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco_2014_vqa",
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples",
    )
    parser.add_argument(
        "--run_4a",
        action="store_true",
        help="Run 4A: Vision Tokens vs Latency",
    )
    parser.add_argument(
        "--run_4b",
        action="store_true",
        help="Run 4B: Language Tokens vs Latency",
    )
    parser.add_argument(
        "--run_both",
        action="store_true",
        help="Run both 4A and 4B",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/exp4",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface or HF_HOME env var)",
    )
    
    args = parser.parse_args()
    
    if not (args.run_4a or args.run_4b or args.run_both):
        args.run_both = True  # Default to running both
    
    experiment = TokenVsLatencyExperiment(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        hf_cache_dir=args.hf_cache_dir,
    )
    
    experiment.run(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        run_4a=args.run_4a or args.run_both,
        run_4b=args.run_4b or args.run_both,
    )


if __name__ == "__main__":
    main()

