#!/usr/bin/env python3
"""
Experiment 4: Language Tokens vs Latency

Goal: Quantify the cost of adding language tokens (sequential decode).
Method: Force generation of different numbers of output tokens with fixed image.
Output: Plots showing output tokens vs decode latency.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from tqdm import tqdm

from experiments.base_experiment import BaseExperiment, Timer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class LanguageTokensVsLatencyExperiment(BaseExperiment):
    """Experiment 4: Language Tokens vs Latency Analysis."""

    def run(self, dataset_name: str, split: str, num_samples: int, max_new_tokens_list: List[int], use_eos_token: bool = False):
        """Run Experiment 4: Language Tokens vs Latency.
        
        Args:
            dataset_name: Name of the dataset.
            split: Dataset split (train/validation/test).
            num_samples: Number of samples to use.
            max_new_tokens_list: List of max_new_tokens values to test.
            use_eos_token: If True, enable early stopping on EOS token.
                          If False, force generation of exactly max_new_tokens tokens.
        """
        with Timer("Exp 4: Language Tokens vs Latency"):
            log.info("Starting Exp 4: Language Tokens vs Latency...")
            log.info(f"  use_eos_token={use_eos_token} (early stopping: {'enabled' if use_eos_token else 'disabled'})")

            dataloader = self.build_dataloader(
                dataset_name, split, batch_size=1, max_steps=num_samples
            )

            # Use a fixed set of images
            fixed_batches = []
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break
                fixed_batches.append(batch)

            if not fixed_batches:
                log.error("No batches available!")
                return []

            # Warmup (use small token count to avoid long warmup time)
            # Warmup is just to initialize CUDA kernels, doesn't need max tokens
            warmup_tokens = min(16, min(max_new_tokens_list))  # Use small value, at most 16 tokens
            log.info(f"Warming up with {warmup_tokens} tokens (to avoid long warmup)...")
            with tqdm(total=self.num_warmup, desc="Warmup", unit="run", leave=False) as pbar:
                for _ in range(self.num_warmup):
                    _ = self.measure_inference_latency(fixed_batches[0], max_new_tokens=warmup_tokens)
                    pbar.update(1)

            # Collect measurements for different output lengths
            log.info(f"Collecting measurements for {len(max_new_tokens_list)} different output lengths...")
            log.info(f"  Max tokens list: {max_new_tokens_list}")
            log.info(f"  Samples per length: {num_samples}")
            log.info(f"  Total measurements: {len(max_new_tokens_list) * num_samples}")
            
            results = []
            total_tasks = len(max_new_tokens_list) * num_samples
            
            # Outer progress bar for overall experiment progress
            with tqdm(total=total_tasks, desc="Exp 4 Progress", unit="measurement", position=0, leave=True) as outer_pbar:
                for token_idx, max_tokens in enumerate(max_new_tokens_list, 1):
                    with Timer(f"Max Tokens {max_tokens}"):
                        log.info(f"[{token_idx}/{len(max_new_tokens_list)}] Testing with max_new_tokens={max_tokens}")
                        # Inner progress bar for current max_tokens
                        for batch_idx, batch in enumerate(tqdm(
                            fixed_batches, 
                            desc=f"  max_tokens={max_tokens}",
                            unit="sample",
                            position=1,
                            leave=False
                        )):
                            # Measure latency and get generated output
                            # If use_eos_token=True: model will stop early when generating EOS token
                            # If use_eos_token=False: model will generate exactly max_new_tokens tokens
                            latency_results = self.measure_inference_latency(
                                batch,
                                max_new_tokens=max_tokens,
                                measure_components=True,
                                num_runs=1,
                                return_output=True,  # Get generated output for decoding
                                use_eos_token=use_eos_token,  # Use parameter to control early stopping
                            )

                            # Decode generated text
                            generated_text = None
                            generated_text_full = None
                            generated_text_preview = None
                            if "generated_output" in latency_results and latency_results["generated_output"] is not None:
                                output = latency_results["generated_output"]
                                input_ids = batch["input_ids"]
                                input_len = input_ids.shape[1]
                                
                                # Extract only generated tokens (after input)
                                if output.shape[1] > input_len:
                                    generated_tokens = output[:, input_len:]
                                else:
                                    generated_tokens = output
                                
                                # Decode to text
                                if generated_tokens.numel() > 0:
                                    # Decode full generated text
                                    generated_text_full = self.tokenizer.decode(
                                        generated_tokens[0].cpu().tolist(),
                                        skip_special_tokens=False
                                    )
                                    
                                    # Also decode with skip_special_tokens=True for cleaner text
                                    generated_text = self.tokenizer.decode(
                                        generated_tokens[0].cpu().tolist(),
                                        skip_special_tokens=True
                                    )
                                    
                                    # Extract first meaningful part (before potential repetition)
                                    # For VQA, the answer is usually at the beginning
                                    # We'll save both full and first 200 chars for readability
                                    generated_text_preview = generated_text[:200] if len(generated_text) > 200 else generated_text

                            # FLOPs
                            flops = self.count_flops(batch, output_length=max_tokens)

                            # Extract Metadata
                            metadata = batch.get("metadata", [{}])[0] if "metadata" in batch else {}

                            # Construct result with unified format
                            res = {
                                "query_id": batch_idx,
                                "max_new_tokens": max_tokens,
                                "num_crops": batch["images"].shape[1] if "images" in batch else 0,
                                "num_vision_tokens": latency_results.get("num_vision_tokens", 0),
                                "num_language_tokens": latency_results.get("num_input_text_tokens", 0),
                                "num_output_tokens": latency_results.get("num_output_tokens", 0),
                                "num_total_tokens": latency_results.get("num_vision_tokens", 0) + latency_results.get("num_input_text_tokens", 0),
                                "T_data_processing": None,  # Not measured in Exp 4
                                "T_vision_encoder": latency_results.get("T_vision_encoder", 0.0),
                                "T_vision_total": latency_results.get("T_vision_total", 0.0),
                                "T_projector": latency_results.get("T_projector", 0.0),
                                "T_vision": latency_results.get("T_vision", 0.0),
                                "T_LLM_prefill": latency_results.get("T_LLM_prefill", 0.0),
                                "T_LLM_decode": latency_results.get("T_LLM_decode", 0.0),
                                "T_total": latency_results.get("T_total", 0.0),
                                "generated_text": generated_text,  # Clean decoded text (skip special tokens)
                                "generated_text_full": generated_text_full,  # Full decoded text (with special tokens)
                                "generated_text_preview": generated_text_preview,  # First 200 chars for readability
                                **flops,
                                **metadata,  # image_id, example_id, answers, image_size
                                "T_system_total": None,  # Not measured in Exp 4
                            }
                            # Remove generated_output from results (it's a tensor, not JSON serializable)
                            if "generated_output" in latency_results:
                                del latency_results["generated_output"]
                            results.append(res)
                            # Update outer progress bar
                            outer_pbar.update(1)
            
            # Analyze and Plot
            log.info("Generating plots and saving results...")
            suffix = "_with_eos" if use_eos_token else "_no_eos"
            self.plot_language_tokens_vs_latency(results, dataset_name, split, suffix=suffix)
            self.save_results({"results": results, "use_eos_token": use_eos_token}, f"exp4_language_tokens_vs_latency{suffix}.json")
            log.info(f"Exp 4 completed! Total measurements: {len(results)}")
            log.info(f"  Results saved with suffix: {suffix}")
            return results

    def plot_language_tokens_vs_latency(self, results: List[Dict], dataset_name: str, split: str, suffix: str = ""):
        """Plot language tokens vs latency."""
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Group by max_new_tokens
        groups = {}
        for r in results:
            mt = r["max_new_tokens"]
            if mt not in groups:
                groups[mt] = []
            groups[mt].append(r)

        max_tokens_list = sorted(groups.keys())
        decode_means = [np.mean([r["T_LLM_decode"] for r in groups[mt]]) for mt in max_tokens_list]
        decode_stds = [np.std([r["T_LLM_decode"] for r in groups[mt]]) for mt in max_tokens_list]
        total_means = [np.mean([r["T_total"] for r in groups[mt]]) for mt in max_tokens_list]
        output_tokens_means = [np.mean([r["num_output_tokens"] for r in groups[mt]]) for mt in max_tokens_list]

        # Color palette: blue for prefill, orange for decode
        colors = {
            'prefill': '#1F77B4',      # Deep blue for prefill
            'decode': '#FF7F0E',       # Bright orange for decode
            'total': '#D62728',        # Deep red for total latency line
        }
        
        # Combined plot: Stacked Bar Chart (Prefill + Decode) with Total Latency line
        # The top of the stacked bars = Total Latency (Prefill + Decode)
        # Convert latencies from ms to seconds for better visibility
        prefill_means = [np.mean([r["T_LLM_prefill"] + r["T_vision_total"] for r in groups[mt]]) for mt in max_tokens_list]
        prefill_means_sec = [v / 1000.0 for v in prefill_means]
        decode_means_sec = [v / 1000.0 for v in decode_means]
        total_means_sec = [v / 1000.0 for v in total_means]

        plt.figure(figsize=(8, 6))  # Match other experiment figures
        x = np.arange(len(max_tokens_list))
        width = 0.6

        # In log scale, stacked bars need special handling
        # Prefill is relatively constant (~305ms = 0.305s)
        # For log scale stacked bars, we need to use log-space calculations
        # Bottom of prefill bar: 0 (or a small minimum)
        # Height of prefill bar: prefill value
        # Bottom of decode bar: prefill value
        # Height of decode bar: decode value
        # But in log scale, matplotlib handles this automatically if we set bottom correctly
        
        # Use actual prefill values - they are relatively fixed around 0.3s
        p1 = plt.bar(x, prefill_means_sec, width, label="Prefill (Vision + LLM)", color=colors['prefill'], edgecolor='black', linewidth=1.5)
        p2 = plt.bar(x, decode_means_sec, width, bottom=prefill_means_sec, label="Decode (LLM only)", color=colors['decode'], edgecolor='black', linewidth=1.5)

        # Add line showing Total Latency (should align with top of bars)
        plt.errorbar(
            x,
            total_means_sec,
            yerr=None,
            fmt="o-",
            capsize=5,
            linewidth=2.5,
            markersize=8,
            label="Total Latency",
            color=colors['total'],
            zorder=10
        )

        plt.xlabel("Decode Output Tokens", fontsize=16)
        plt.ylabel("Latency (seconds, log scale)", fontsize=16, labelpad=3)  # Reduce distance from axis
        plt.title("Latency Breakdown", fontsize=18)
        # Use integer labels for x-axis
        plt.xticks(x, [int(round(v)) for v in output_tokens_means], fontsize=14)
        
        # Use log scale for better visibility of both prefill and decode
        plt.yscale('log')
        plt.ylim(bottom=0.1, top=1000)  # Start from 0.1 seconds, max 1000 seconds
        
        # Format Y-axis labels to be more readable (show as seconds with 1-2 decimal places)
        def format_seconds(x, pos):
            if x < 1:
                return f'{x:.2f}'
            elif x < 10:
                return f'{x:.1f}'
            elif x < 100:
                return f'{x:.0f}'
            else:
                return f'{x:.0f}'
        
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_seconds))
        
        # Add a tick at 0.3 seconds (300ms) to highlight prefill latency
        ax = plt.gca()
        current_ticks = ax.get_yticks()
        # Add 0.3 to the ticks if not already present
        if 0.3 not in current_ticks:
            new_ticks = sorted(list(current_ticks) + [0.3])
            ax.set_yticks(new_ticks)
        
        # Ensure Y-axis still starts from 0.1 and ends at 1000 after setting ticks
        plt.ylim(bottom=0.1, top=1000)
        
        plt.yticks(fontsize=14)
        
        plt.legend(fontsize=14, loc='upper left')
        plt.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plot_filename = f"exp4_language_tokens_vs_latency_breakdown_{dataset_name}_{split}{suffix}.png"
        plt.savefig(fig_dir / plot_filename, dpi=300, bbox_inches="tight")
        log.info(f"Plot saved to {fig_dir / plot_filename}")
        plt.close()

        # Print Statistics
        log.info("=" * 60)
        log.info("Language Token Analysis:")
        log.info(f"  Output tokens range: {min(output_tokens_means):.0f} - {max(output_tokens_means):.0f}")
        log.info(f"  Decode latency range: {min(decode_means):.2f} - {max(decode_means):.2f} ms")
        log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Language Tokens vs Latency")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="coco_2014_vqa", help="Dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of fixed images to use")
    parser.add_argument("--output_dir", type=str, default="./results/motivation/exp4", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="HuggingFace cache directory")
    parser.add_argument(
        "--max_new_tokens_list",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        help="List of max_new_tokens to test",
    )
    parser.add_argument(
        "--use_eos_token",
        action="store_true",
        help="Enable early stopping on EOS token. If not set, model will generate exactly max_new_tokens tokens.",
    )

    args = parser.parse_args()

    experiment = LanguageTokensVsLatencyExperiment(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        hf_cache_dir=args.hf_cache_dir,
    )

    experiment.run(args.dataset, args.split, args.num_samples, args.max_new_tokens_list, use_eos_token=args.use_eos_token)


if __name__ == "__main__":
    main()

