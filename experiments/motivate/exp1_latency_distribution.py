#!/usr/bin/env python3
"""
Experiment 1: Latency Distribution

Goal: Measure latency distribution on real-world datasets to identify tail latency issues.
Method: Run on VQA v2 validation set with 5000 samples, measure T_total only (fast).
Output: Histogram, CDF curve, P50/P95/P99 statistics.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiments.motivate.base_experiment import BaseExperiment, Timer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class LatencyDistributionExperiment(BaseExperiment):
    """Experiment 1: Latency Distribution Analysis."""

    def run(self, dataset_name: str, split: str, num_samples: int):
        """Run Experiment 1: Latency Distribution."""
        with Timer("Exp 1: Latency Distribution"):
            log.info(f"Starting Exp 1: Latency Distribution (samples={num_samples})...")

            dataloader = self.build_dataloader(
                dataset_name, split, batch_size=1, max_steps=num_samples
            )
            
            # Get actual dataset length (min of requested and available)
            actual_num_samples = getattr(dataloader, 'dataset_length', num_samples)
            if actual_num_samples < num_samples:
                log.info(f"Dataset has {actual_num_samples} samples, limiting to {actual_num_samples} (requested {num_samples})")
            
            dataloader_iter = iter(dataloader)

            # Warmup: Run a few samples to absorb initialization costs (not counted in results)
            log.info(f"Performing warmup ({self.num_warmup} samples, not counted in results)...")
            for _ in range(self.num_warmup):
                try:
                    batch = next(dataloader_iter)
                    # Run warmup measurement (not saved to results)
                    _ = self.measure_inference_latency(
                        batch,
                        max_new_tokens=0,
                        measure_components=False,
                        num_runs=1,
                    )
                except StopIteration:
                    log.warning(f"Dataset exhausted during warmup. Only {_} samples available for warmup.")
                    break
            
            log.info("Warmup complete. Starting actual measurements...")

            results = []
            for i in tqdm(range(actual_num_samples), total=actual_num_samples, desc="Exp 1"):
                # Measure Data Processing Latency
                t0_process = time.perf_counter()
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break
                t_data_processing = (time.perf_counter() - t0_process) * 1000

                # Measure Total Latency Only (Fast)
                lats = self.measure_inference_latency(
                    batch,
                    max_new_tokens=0,  # Prefill only for speed
                    measure_components=False,  # Fast mode
                    num_runs=1,
                )

                # Calculate FLOPs
                flops = self.count_flops(batch, output_length=0)

                # Extract Metadata
                metadata = batch.get("metadata", [{}])[0] if "metadata" in batch else {}

                # Construct result with unified format (use None for non-applicable fields)
                res = {
                    "num_crops": batch["images"].shape[1] if "images" in batch else 0,
                    "num_vision_tokens": lats["num_vision_tokens"],
                    "num_language_tokens": lats["num_input_text_tokens"],
                    "num_total_tokens": lats["num_vision_tokens"] + lats["num_input_text_tokens"],
                    "T_data_processing": t_data_processing,
                    "T_vision_encoder": None,  # Not measured in fast mode
                    "T_vision_total": None,  # Not measured in fast mode
                    "T_projector": None,  # Not measured in fast mode
                    "T_vision": None,  # Not measured in fast mode
                    "T_LLM_prefill": None,  # Not measured in fast mode
                    "T_LLM_decode": None,  # Not applicable (max_new_tokens=0)
                    "T_total": lats["T_total"],
                    **flops,
                    **metadata,  # image_id, example_id, answers, image_size
                    "T_system_total": lats["T_total"] + t_data_processing,
                }
                results.append(res)

            # Analyze and Plot
            self.plot_latency_distribution(results, dataset_name, split)
            self.save_results({"results": results}, "exp1_latency_distribution.json")
            return results

    def plot_latency_distribution(self, results: List[Dict], dataset_name: str, split: str):
        """Plot latency distribution: histogram and CDF."""
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        latencies = [r["T_total"] for r in results]
        stats = self.compute_statistics(latencies)

        # Unified color palette (ggthemes Classic_10 - high saturation, colorblind-friendly)
        # Official Classic_10: https://emilhvitfeldt.github.io/r-color-palettes/discrete/ggthemes/Classic_10/
        colors = {
            'primary': '#1F77B4',      # Deep blue (Classic_10 color 1)
            'secondary': '#FF7F0E',     # Bright orange (Classic_10 color 2)
            'tertiary': '#2CA02C',      # Deep green (Classic_10 color 3)
            'quaternary': '#D62728',   # Deep red (Classic_10 color 4)
            'quinary': '#9467BD',      # Deep purple (Classic_10 color 5)
            'senary': '#8C564B',       # Brown (Classic_10 color 6)
            'septenary': '#E377C2',    # Pink (Classic_10 color 7)
            'octonary': '#7F7F7F',     # Gray (Classic_10 color 8)
            'nonary': '#BCBD22',       # Yellow-green (Classic_10 color 9)
            'denary': '#17BECF',       # Cyan (Classic_10 color 10)
            'histogram': '#1F77B4',   # Deep blue for histograms
            'p50': '#1F77B4',          # Deep blue for P50
            'p95': '#FF7F0E',          # Bright orange for P95
            'p99': '#D62728',          # Deep red for P99
        }
        
        # Plot 1: Histogram (separate figure)
        plt.figure(figsize=(8, 6))
        plt.hist(latencies, bins=50, color=colors['histogram'], edgecolor='black', linewidth=1.5)
        plt.axvline(stats["P50"], color=colors['p50'], linestyle="--", linewidth=2, label=f"P50: {stats['P50']:.1f}ms")
        plt.axvline(stats["P95"], color=colors['p95'], linestyle="--", linewidth=2, label=f"P95: {stats['P95']:.1f}ms")
        plt.axvline(stats["P99"], color=colors['p99'], linestyle="--", linewidth=2, label=f"P99: {stats['P99']:.1f}ms")
        plt.xlabel("Latency (ms)", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.title("Latency Distribution", fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(fig_dir / f"exp1_latency_distribution_histogram_{dataset_name}_{split}.png", dpi=300, bbox_inches="tight")
        log.info(f"Plot saved to {fig_dir / f'exp1_latency_distribution_histogram_{dataset_name}_{split}.png'}")
        plt.close()

        # Plot 2: CDF (separate figure)
        plt.figure(figsize=(8, 6))
        sorted_lats = np.sort(latencies)
        percentiles = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats) * 100
        plt.plot(sorted_lats, percentiles, linewidth=2.5, color=colors['primary'])
        plt.axvline(stats["P50"], color=colors['p50'], linestyle="--", linewidth=2, label=f"P50: {stats['P50']:.1f}ms")
        plt.axvline(stats["P95"], color=colors['p95'], linestyle="--", linewidth=2, label=f"P95: {stats['P95']:.1f}ms")
        plt.axvline(stats["P99"], color=colors['p99'], linestyle="--", linewidth=2, label=f"P99: {stats['P99']:.1f}ms")
        plt.xlabel("Latency (ms)", fontsize=16)
        plt.ylabel("Cumulative Percentage (%)", fontsize=16)
        plt.title("Cumulative Distribution Function", fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(fig_dir / f"exp1_latency_distribution_cdf_{dataset_name}_{split}.png", dpi=300, bbox_inches="tight")
        log.info(f"Plot saved to {fig_dir / f'exp1_latency_distribution_cdf_{dataset_name}_{split}.png'}")
        plt.close()

        # Print Statistics
        log.info("=" * 60)
        log.info("Latency Statistics:")
        log.info(f"  Mean: {stats['mean']:.2f} ms")
        log.info(f"  Std:  {stats['std']:.2f} ms")
        log.info(f"  P50:  {stats['P50']:.2f} ms")
        log.info(f"  P95:  {stats['P95']:.2f} ms")
        log.info(f"  P99:  {stats['P99']:.2f} ms")
        log.info(f"  Min:  {stats['min']:.2f} ms")
        log.info(f"  Max:  {stats['max']:.2f} ms")
        log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Latency Distribution")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="coco_2014_vqa", help="Dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--output_dir", type=str, default="./results/motivation/exp1", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="HuggingFace cache directory")

    args = parser.parse_args()

    experiment = LatencyDistributionExperiment(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        hf_cache_dir=args.hf_cache_dir,
    )

    experiment.run(args.dataset, args.split, args.num_samples)


if __name__ == "__main__":
    main()

