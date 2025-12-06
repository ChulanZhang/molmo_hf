
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

class OutputTokensScalingExperiment(BaseExperiment):
    """
    Profiling Experiment 4: Output Tokens vs Latency (corresponds to Motivation Exp 4)
    
    Goal: Research the impact of number of output tokens on Decode latency.
    Method: Force generation of different numbers of output tokens with fixed image.
    """
    
    def run(self, num_samples: int = 50, max_new_tokens_list: List[int] = None):
        """
        Run output tokens scaling experiment.
        
        Args:
            num_samples: Number of measurement repetitions per max_new_tokens value.
            max_new_tokens_list: List of max_new_tokens values to test.
                                If None, defaults to [1, 5, 10, 20, 50, 100, 200].
        """
        processor = self.processor
        
        # Standard input
        image = Image.new('RGB', (336, 336), color='blue')
        prompt = "Describe this image."
        inputs = processor.process(text=prompt, images=image)
        
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
        
        # Default max_new_tokens list
        if max_new_tokens_list is None:
            max_new_tokens_list = [1, 5, 10, 20, 50, 100, 200]
        
        log.info(f"Testing max_new_tokens values: {max_new_tokens_list}")
        
        results_data = []
        
        # Warmup (use small token count to avoid long warmup time)
        warmup_tokens = min(16, min(max_new_tokens_list))
        log.info(f"Warming up with {warmup_tokens} tokens...")
        for _ in range(self.num_warmup):
            self.measure_inference_latency(inputs, max_new_tokens=warmup_tokens, measure_components=True)
        
        for max_tokens in tqdm(max_new_tokens_list, desc="Output Tokens Scaling"):
            log.info(f"Testing with max_new_tokens={max_tokens}...")
            
            # Run measurements
            latencies_decode = []
            latencies_total = []
            per_sample_results = []
            
            for sample_idx in range(num_samples):
                metrics = self.measure_inference_latency(
                    inputs,
                    max_new_tokens=max_tokens,
                    measure_components=True,
                    num_runs=1,
                )
                latencies_decode.append(metrics["T_LLM_decode"])
                latencies_total.append(metrics["T_total"])
                
                # Save per-sample result
                per_sample_results.append({
                    "sample_id": sample_idx,
                    "max_new_tokens": max_tokens,
                    "num_output_tokens": metrics.get("num_output_tokens", 0),
                    "num_vision_tokens": metrics.get("num_vision_tokens", 0),
                    "num_input_text_tokens": metrics.get("num_input_text_tokens", 0),
                    **metrics  # Include all metrics from measure_inference_latency
                })
            
            stats_decode = self.compute_statistics(latencies_decode)
            stats_total = self.compute_statistics(latencies_total)
            
            combined_stats = {
                "max_new_tokens": max_tokens,
                "decode": stats_decode,
                "total": stats_total,
                "per_sample_results": per_sample_results
            }
            results_data.append(combined_stats)
            
            log.info(f"Max Tokens {max_tokens}: P50 Decode={stats_decode['P50']:.2f}ms, P50 Total={stats_total['P50']:.2f}ms")
        
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
        
        self.save_results(final_results, "exp4_output_tokens_scaling_results.json")
        log.info(f"Total samples: {len(all_samples)}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile output tokens scaling (corresponds to Motivation Exp 4)"
    )
    parser.add_argument("--model_path", type=str, default="checkpoints",
                        help="Path to model checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="./results/profiling/output_tokens",
                        help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of measurement repetitions per max_new_tokens value")
    parser.add_argument("--max_new_tokens", type=int, nargs="+", default=None,
                        help="List of max_new_tokens values to test. "
                             "If not provided, defaults to [1, 5, 10, 20, 50, 100, 200]")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = OutputTokensScalingExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    experiment.run(
        num_samples=args.num_samples,
        max_new_tokens_list=args.max_new_tokens
    )


if __name__ == "__main__":
    main()

