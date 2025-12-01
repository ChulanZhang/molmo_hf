#!/usr/bin/env python3
"""
Experiment 3: Vision Tokens vs Latency

Goal: Quantify the cost of adding vision tokens (mainly parallelizable prefill).
Method: Resize dummy images to different resolutions to trigger different grid sizes.
Output: Plots showing vision tokens vs latency (total, prefill, components).
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from experiments.motivate.base_experiment import BaseExperiment, Timer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class VisionTokensVsLatencyExperiment(BaseExperiment):
    """Experiment 3: Vision Tokens vs Latency Analysis."""

    def run(self, max_grid_size: int = 12, num_runs: int = 10, use_hook_for_llm_prefill: bool = False):
        """Run Experiment 3: Vision Tokens vs Latency.
        
        Args:
            max_grid_size: Maximum grid size (1x12)
            num_runs: Number of runs per resolution for stability (default: 10)
            use_hook_for_llm_prefill: If True, use forward hooks to directly measure LLM prefill.
                                      If False (default), use subtraction method.
        """
        with Timer("Exp 3: Vision Tokens vs Latency"):
            log.info(f"Starting Exp 3: Vision Tokens vs Latency (num_runs={num_runs})...")

            processor = self.processor
            base_image = Image.new("RGB", (2000, 2000), color=(100, 150, 200))
            text = "Describe this image."

            # Global Warmup (Critical to avoid initialization costs)
            log.info("Performing global warmup to absorb initialization costs...")
            warmup_img = base_image.resize((336, 336))
            warmup_inputs = processor.process(text=text, images=warmup_img)
            warmup_inputs = {k: v.to(self.device).unsqueeze(0) for k, v in warmup_inputs.items()}
            with torch.inference_mode():
                # Warmup Vision
                _ = self.model.model.vision_backbone.encode_image(warmup_inputs["images"])
                # Warmup LLM
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

            # Generate images with different tiling configurations to cover all possible crop counts
            # To ensure precise control over vision tokens, we calculate exact resolutions
            # based on the tiling algorithm's parameters:
            # - crop_window_size = 224 (16 patches * 14 pixels)
            # - total_margin_pixels = 112 (8 patches * 14 pixels)
            # 
            # We use different tiling configurations to get missing crop counts:
            # - 1×k tiling for most cases (vertical rectangles)
            # - k×1 tiling for some cases (horizontal rectangles) 
            # - 2×k tiling for some cases (rectangular grids)
            #
            # Target crop counts: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13

            # Calculate exact resolutions for precise tiling control
            crop_window_size = 224  # 16 patches * 14 pixels per patch
            total_margin_pixels = 112  # 8 patches * 14 pixels (4+4 margins)
            
            # Define tiling configurations to cover all crop counts
            # Format: (tiling_i, tiling_j, description)
            tiling_configs = [
                (1, 1, "1×1"),   # 2 crops
                (1, 2, "1×2"),   # 3 crops (missing)
                (1, 3, "1×3"),   # 4 crops
                (1, 4, "1×4"),   # 5 crops
                (1, 5, "1×5"),   # 6 crops (missing)
                (1, 6, "1×6"),   # 7 crops
                (1, 7, "1×7"),   # 8 crops
                (1, 8, "1×8"),   # 9 crops (missing)
                (1, 9, "1×9"),   # 10 crops
                (1, 10, "1×10"), # 11 crops
                (1, 11, "1×11"), # 12 crops (missing)
                (1, 12, "1×12"), # 13 crops
            ]
            
            results = []
            for tiling_i, tiling_j, desc in tiling_configs:
                # Calculate exact resolution for this tiling
                w = tiling_j * crop_window_size + total_margin_pixels
                h = tiling_i * crop_window_size + total_margin_pixels
                expected_crops = tiling_i * tiling_j + 1
                
                with Timer(f"Grid {desc} ({w}x{h})"):
                    log.info(f"Testing Grid {desc} (Resolution {w}x{h}, expected crops: {expected_crops})...")
                    img = base_image.resize((w, h))
                    inputs = processor.process(text=text, images=img)
                    inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}

                    # Measure (multiple runs for stability)
                    lats = self.measure_inference_latency(
                        inputs,
                        max_new_tokens=0,  # Prefill only
                        measure_components=True,
                        num_runs=num_runs,
                        use_hook_for_llm_prefill=use_hook_for_llm_prefill,
                    )

                    # FLOPs
                    flops = self.count_flops(inputs, output_length=1)

                    # Tokens
                    num_vision_tokens = lats["num_vision_tokens"]
                    num_crops = inputs["images"].shape[1]
                    num_input_tokens = inputs["input_ids"].shape[1]
                    num_total_tokens = num_vision_tokens + num_input_tokens

                    # Construct result with unified format
                    res = {
                        "resolution": f"{w}x{h}",
                        "grid": desc,
                        "num_crops": num_crops,
                        "num_vision_tokens": num_vision_tokens,
                        "num_language_tokens": num_input_tokens,
                        "num_total_tokens": num_total_tokens,
                        "T_data_processing": None,  # Not applicable (synthetic images)
                        "T_vision_encoder": lats.get("T_vision_encoder", 0.0),
                        "T_vision_total": lats.get("T_vision_total", 0.0),
                        "T_projector": lats.get("T_projector", 0.0),
                        "T_vision": lats.get("T_vision", 0.0),
                        "T_LLM_prefill": lats.get("T_LLM_prefill", 0.0),
                        "T_LLM_decode": None,  # Not applicable (max_new_tokens=0)
                        "T_total": lats.get("T_total", 0.0),
                        **flops,
                        "image_id": None,  # Not applicable (synthetic images)
                        "example_id": None,  # Not applicable (synthetic images)
                        "answers": None,  # Not applicable (synthetic images)
                        "image_size": [w, h],
                        "T_system_total": None,  # Not measured in Exp 3
                    }
                    results.append(res)

            # Plot
            self.plot_vision_tokens_vs_latency(results)
            self.save_results({"results": results}, "exp3_vision_tokens_vs_latency.json")
            return results

    def plot_vision_tokens_vs_latency(self, results: List[Dict]):
        """Plot vision tokens vs latency."""
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        vision_tokens = [r["num_vision_tokens"] for r in results]
        prefill_lats = [
            r.get("T_vision_total", 0) + r.get("T_LLM_prefill", 0) for r in results
        ]
        # Note: T_vision_total = T_vision_encoder + T_projector
        vision_encoder_lats = [r.get("T_vision_encoder", 0) for r in results]
        projector_lats = [r.get("T_projector", 0) for r in results]
        prefill_only_lats = [r.get("T_LLM_prefill", 0) for r in results]

        # Unified color palette (ggthemes Classic_10 - high saturation, colorblind-friendly)
        # Official Classic_10: https://emilhvitfeldt.github.io/r-color-palettes/discrete/ggthemes/Classic_10/
        colors = {
            'primary': '#1F77B4',      # Deep blue (Classic_10 color 1)
            'secondary': '#FF7F0E',    # Bright orange (Classic_10 color 2)
            'tertiary': '#2CA02C',     # Deep green (Classic_10 color 3)
            'quaternary': '#D62728',   # Deep red (Classic_10 color 4)
        }
        
        # Combined plot: Stacked bar chart with Time to First Token line on top
        # The top of the stacked bars = Time to First Token (T_vision_total + T_LLM_prefill)
        plt.figure(figsize=(8, 6))
        x = np.arange(len(vision_tokens))
        width = 0.6
        
        # Stack bars: Vision Encoder at bottom, Projector on top, LLM Prefill on top of that
        # Top of bars = T_vision_encoder + T_projector + T_LLM_prefill = T_vision_total + T_LLM_prefill = prefill_lats
        p1 = plt.bar(x, vision_encoder_lats, width, label="Vision Encoder", color=colors['primary'], edgecolor='black', linewidth=1.0)
        p2 = plt.bar(x, projector_lats, width, bottom=vision_encoder_lats, label="Projector", color=colors['secondary'], edgecolor='black', linewidth=1.0)
        p3 = plt.bar(x, prefill_only_lats, width, bottom=np.array(vision_encoder_lats) + np.array(projector_lats), 
                     label="LLM", color=colors['tertiary'], edgecolor='black', linewidth=1.0)
        
        # Add line showing Time to First Token (should align with top of bars)
        plt.plot(x, prefill_lats, "o-", label="Time to First Token", linewidth=2.5, markersize=8, 
                color=colors['quaternary'], zorder=10)
        
        plt.xlabel("Vision Tokens", fontsize=16)
        plt.ylabel("Latency (ms)", fontsize=16)
        plt.title("Latency Breakdown", fontsize=18)
        plt.xticks(x, vision_tokens, fontsize=14)
        
        # Set Y-axis to show 400ms tick
        ax = plt.gca()
        current_ticks = ax.get_yticks()
        # Add 400 to the ticks if not already present
        if 400 not in current_ticks:
            new_ticks = sorted(list(current_ticks) + [400])
            ax.set_yticks(new_ticks)
        # Set maximum to 401 to ensure 400 is visible
        current_ylim = ax.get_ylim()
        plt.ylim(bottom=current_ylim[0], top=max(401, current_ylim[1]))
        
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / "exp3_vision_tokens_vs_latency_breakdown.png", dpi=300, bbox_inches="tight")
        log.info(f"Plot saved to {fig_dir / 'exp3_vision_tokens_vs_latency_breakdown.png'}")
        plt.close()

        # Print Statistics
        log.info("=" * 60)
        log.info("Vision Token Analysis:")
        log.info(f"  Vision tokens range: {min(vision_tokens)} - {max(vision_tokens)}")
        log.info(f"  Prefill latency range: {min(prefill_lats):.2f} - {max(prefill_lats):.2f} ms")
        log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Vision Tokens vs Latency")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results/motivation/exp3", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="HuggingFace cache directory")
    parser.add_argument("--max_grid_size", type=int, default=12, help="Maximum grid size (1x12)")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs per resolution for stability (default: 10)")
    parser.add_argument("--use_hook_for_llm_prefill", action="store_true", 
                        help="Use forward hooks to directly measure LLM prefill (more accurate, avoids subtraction errors)")

    args = parser.parse_args()

    experiment = VisionTokensVsLatencyExperiment(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        hf_cache_dir=args.hf_cache_dir,
    )

    experiment.run(max_grid_size=args.max_grid_size, num_runs=args.num_runs, 
                   use_hook_for_llm_prefill=args.use_hook_for_llm_prefill)


if __name__ == "__main__":
    main()

