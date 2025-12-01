#!/usr/bin/env python3
"""
Experiment 2: Component Profiling

Goal: Analyze the time/parameter cost of each component (Vision Encoder, Projector, LLM).
Method: Measure detailed component latencies and count parameters.
Output: Pie charts for parameter distribution and latency distribution.
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


class ComponentProfilingExperiment(BaseExperiment):
    """Experiment 2: Component Profiling Analysis."""

    def run(self, dataset_name: str, split: str, num_samples: int):
        """Run Experiment 2: Component Profiling."""
        with Timer("Exp 2: Component Profiling"):
            log.info(f"Starting Exp 2: Component Profiling (samples={num_samples})...")

            # Count Parameters
            log.info("Counting parameters for each component...")
            params = self.count_parameters()
            log.info("=" * 60)
            log.info("Parameter Counts:")
            log.info(f"  Vision Encoder: {params['params_vision_encoder']:,} ({params['params_vision_encoder']/1e6:.2f}M)")
            log.info(f"  Projector:      {params['params_projector']:,} ({params['params_projector']/1e6:.2f}M)")
            if params.get('moe_num_experts', 0) > 0:
                log.info(f"  LLM (Total):    {params['params_llm']:,} ({params['params_llm']/1e6:.2f}M)")
                log.info(f"  LLM (Active):   {params['params_llm_active']:,} ({params['params_llm_active']/1e6:.2f}M)")
                log.info(f"  MoE Config:     {params['moe_num_experts']} experts, top_k={params['moe_top_k']}")
            else:
                log.info(f"  LLM:            {params['params_llm']:,} ({params['params_llm']/1e6:.2f}M)")
            log.info(f"  Total:          {params['params_total']:,} ({params['params_total']/1e6:.2f}M)")
            log.info(f"  Active:         {params['params_active']:,} ({params['params_active']/1e6:.2f}M)")
            log.info("=" * 60)

            dataloader = self.build_dataloader(
                dataset_name, split, batch_size=1, max_steps=num_samples
            )
            dataloader_iter = iter(dataloader)

            results = []
            for i in tqdm(range(num_samples), total=num_samples, desc="Exp 2"):
                # Measure Data Processing Latency
                t0_process = time.perf_counter()
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break
                t_data_processing = (time.perf_counter() - t0_process) * 1000

                # Measure Detailed Components (Slower)
                lats = self.measure_inference_latency(
                    batch,
                    max_new_tokens=0,  # Prefill only
                    measure_components=True,
                    num_runs=1,
                )

                # Calculate FLOPs
                flops = self.count_flops(batch, output_length=0)

                # Extract Metadata
                metadata = batch.get("metadata", [{}])[0] if "metadata" in batch else {}

                # Construct result with unified format
                res = {
                    "num_crops": batch["images"].shape[1] if "images" in batch else 0,
                    "num_vision_tokens": lats["num_vision_tokens"],
                    "num_language_tokens": lats["num_input_text_tokens"],
                    "num_total_tokens": lats["num_vision_tokens"] + lats["num_input_text_tokens"],
                    "T_data_processing": t_data_processing,
                    "T_vision_encoder": lats.get("T_vision_encoder", 0.0),
                    "T_vision_total": lats.get("T_vision_total", 0.0),
                    "T_projector": lats.get("T_projector", 0.0),
                    "T_vision": lats.get("T_vision", 0.0),
                    "T_LLM_prefill": lats.get("T_LLM_prefill", 0.0),
                    "T_LLM_decode": None,  # Not applicable (max_new_tokens=0)
                    "T_total": lats.get("T_total", 0.0),
                    **flops,
                    **metadata,  # image_id, example_id, answers, image_size
                    "T_system_total": lats.get("T_total", 0.0) + t_data_processing,
                }
                results.append(res)

            # Analyze & Plot
            # Filter out None values and non-numeric keys
            numeric_keys = [
                k for k in results[0].keys()
                if isinstance(results[0][k], (int, float, np.number)) and results[0][k] is not None
            ]
            avg_results = {k: np.mean([r[k] for r in results if r[k] is not None]) for k in numeric_keys}

            log.info("=" * 60)
            log.info("Average Latencies:")
            log.info(f"  T_data_processing: {avg_results.get('T_data_processing', 0):.2f} ms")
            log.info(f"  T_vision_encoder:  {avg_results.get('T_vision_encoder', 0):.2f} ms")
            log.info(f"  T_projector:       {avg_results.get('T_projector', 0):.2f} ms")
            log.info(f"  T_LLM_prefill:    {avg_results.get('T_LLM_prefill', 0):.2f} ms")
            log.info(f"  T_total:          {avg_results.get('T_total', 0):.2f} ms")
            log.info(f"  T_system_total:   {avg_results.get('T_system_total', 0):.2f} ms")
            log.info("=" * 60)

            self.plot_component_pie_charts(avg_results, params, dataset_name, split)
            self.save_results(
                {
                    "results": results,
                    "average_latencies": avg_results,
                    "parameters": params,
                },
                "exp2_component_profiling.json",
            )
            return results

    def plot_component_pie_charts(self, avg_results: Dict, params: Dict, dataset_name: str, split: str):
        """Plot pie charts for parameter and latency distributions."""
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Unified color mapping for components (ggthemes Classic_10 - high saturation, colorblind-friendly)
        # Same component uses same color across both plots
        # LLM uses green (previously connector's color) since connector is merged into projector
        # Official Classic_10: https://emilhvitfeldt.github.io/r-color-palettes/discrete/ggthemes/Classic_10/
        component_colors = {
            "Vision Encoder": '#1F77B4',  # Deep blue (Classic_10 color 1)
            "Projector": '#FF7F0E',        # Bright orange (Classic_10 color 2)
            "LLM": '#2CA02C',              # Deep green (Classic_10 color 3, previously connector's color)
            "LLM Prefill": '#2CA02C',      # Deep green (same as LLM)
        }

        # Plot 1: Parameter Distribution (separate figure)
        plt.figure(figsize=(8, 6))
        param_labels = ["Vision Encoder", "Projector", "LLM"]
        param_values = [
            params["params_vision_encoder"],
            params["params_projector"],
            params["params_llm"],
        ]
        # Filter out zero values
        param_data = [(l, v) for l, v in zip(param_labels, param_values) if v > 0]
        param_labels_filtered = [l for l, v in param_data]
        param_values_filtered = [v for l, v in param_data]
        
        # Assign colors based on component mapping
        param_colors = [component_colors[label] for label in param_labels_filtered]
        
        # Calculate percentages
        total_params = sum(param_values_filtered)
        param_percentages = [v / total_params * 100 for v in param_values_filtered]
        
        # Draw pie chart without labels (use legend instead)
        wedges, texts, autotexts = plt.pie(
            param_values_filtered,
            labels=None,  # No labels on pie, use legend instead
            colors=param_colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85,  # Percentages inside the pie
            textprops={'fontsize': 14},
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}  # Smaller white border
        )
        
        # Manual distance mapping to avoid overlap (reference: plotting/parameters_pie_chart.py)
        # Increase distances for Vision Encoder and Projector to avoid overlap
        distance_map = {}
        for label in param_labels_filtered:
            if label == 'Vision Encoder':
                distance_map[label] = 0.90  # Further from center to avoid overlap with Projector
            elif label == 'Projector':
                distance_map[label] = 0.85   # Further from center, avoid overlap with Vision Encoder
            else:  # LLM
                distance_map[label] = 0.85   # Normal distance for large slice
        
        # Set text style and manually adjust positions to avoid overlap
        for i, (autotext, wedge, label) in enumerate(zip(autotexts, wedges, param_labels_filtered)):
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)
            
            # Get the angle of the wedge center
            theta = (wedge.theta2 + wedge.theta1) / 2
            theta_rad = np.deg2rad(theta)
            
            # Get specified distance for this component
            distance = distance_map.get(label, 0.85)
            
            # Calculate new position
            x = distance * np.cos(theta_rad)
            y = distance * np.sin(theta_rad)
            
            # Set new position
            autotext.set_position((x, y))
        
        # Create legend with values for parameters
        legend_labels = [f'{label}: {val/1e6:.2f}M ({pct:.1f}%)' 
                         for label, val, pct in zip(param_labels_filtered, param_values_filtered, param_percentages)]
        legend = plt.legend(wedges, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.5), 
                           ncol=1, fontsize=14, framealpha=0.9)
        legend.get_frame().set_linewidth(1.5)
        
        # Remove title as requested
        plt.tight_layout()
        plt.savefig(fig_dir / f"exp2_component_profiling_parameters_{dataset_name}_{split}.png", dpi=300, bbox_inches="tight")
        log.info(f"Plot saved to {fig_dir / f'exp2_component_profiling_parameters_{dataset_name}_{split}.png'}")
        plt.close()

        # Plot 2: Latency Distribution (separate figure)
        plt.figure(figsize=(8, 6))
        lat_labels = ["Vision Encoder", "Projector", "LLM"]
        lat_values = [
            avg_results.get("T_vision_encoder", 0),
            avg_results.get("T_projector", 0),
            avg_results.get("T_LLM_prefill", 0),
        ]
        # Filter out zero values
        lat_data = [(l, v) for l, v in zip(lat_labels, lat_values) if v > 0]
        lat_labels_filtered = [l for l, v in lat_data]
        lat_values_filtered = [v for l, v in lat_data]

        # Assign colors based on component mapping (same components get same colors)
        lat_colors = [component_colors[label] for label in lat_labels_filtered]
        
        # Calculate percentages
        total_latency = sum(lat_values_filtered)
        lat_percentages = [v / total_latency * 100 for v in lat_values_filtered]
        
        # Draw pie chart without labels (use legend instead)
        wedges, texts, autotexts = plt.pie(
            lat_values_filtered,
            labels=None,  # No labels on pie, use legend instead
            colors=lat_colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85,  # Percentages inside the pie
            textprops={'fontsize': 14},
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}  # Smaller white border
        )
        
        # Manual distance mapping to avoid overlap (reference: plotting/latency_pie_chart.py)
        distance_map = {}
        for label in lat_labels_filtered:
            if label == 'Vision Encoder':
                distance_map[label] = 0.80  # Slightly closer to avoid overlap with Projector
            elif label == 'Projector':
                distance_map[label] = 0.75   # Between Vision Encoder and center, avoid overlap
            else:  # LLM
                distance_map[label] = 0.85   # Normal distance for large slice
        
        # Set text style and manually adjust positions to avoid overlap
        for i, (autotext, wedge, label) in enumerate(zip(autotexts, wedges, lat_labels_filtered)):
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)
            
            # Get the angle of the wedge center
            theta = (wedge.theta2 + wedge.theta1) / 2
            theta_rad = np.deg2rad(theta)
            
            # Get specified distance for this component
            distance = distance_map.get(label, 0.85)
            
            # Calculate new position
            x = distance * np.cos(theta_rad)
            y = distance * np.sin(theta_rad)
            
            # Set new position
            autotext.set_position((x, y))
        
        # Create legend with values for latencies
        legend_labels = [f'{label}: {val:.2f}ms ({pct:.1f}%)' 
                         for label, val, pct in zip(lat_labels_filtered, lat_values_filtered, lat_percentages)]
        legend = plt.legend(wedges, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.5), 
                           ncol=1, fontsize=14, framealpha=0.9)
        legend.get_frame().set_linewidth(1.5)
        
        plt.title("Latency Distribution", fontsize=18)
        plt.tight_layout()
        plt.savefig(fig_dir / f"exp2_component_profiling_latency_{dataset_name}_{split}.png", dpi=300, bbox_inches="tight")
        log.info(f"Plot saved to {fig_dir / f'exp2_component_profiling_latency_{dataset_name}_{split}.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Component Profiling")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="coco_2014_vqa", help="Dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output_dir", type=str, default="./results/motivation/exp2", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="HuggingFace cache directory")

    args = parser.parse_args()

    experiment = ComponentProfilingExperiment(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        hf_cache_dir=args.hf_cache_dir,
    )

    experiment.run(args.dataset, args.split, args.num_samples)


if __name__ == "__main__":
    main()

