
import argparse
import logging
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor

import sys
import os
sys.path.append(os.getcwd())
from experiments.motivate.base_experiment import BaseExperiment, Timer

log = logging.getLogger(__name__)

class ContextScalingExperiment(BaseExperiment):
    def run(self, num_samples: int = 100, max_length: int = 1500, step_size: int = 100):
        """
        Run context scaling experiment.
        
        Args:
            num_samples: Number of repetitions per length.
            max_length: Maximum text length to test.
            step_size: Step size for text length.
        """
        # Load processor
        processor = AutoProcessor.from_pretrained(
            "allenai/MolmoE-1B-0924", 
            trust_remote_code=True
        )
        
        # Define lengths to test
        # Start from a small number, go up to max_length
        # We want to test pure text scaling, but Molmo expects an image.
        # We will use a fixed small image (1 crop) to minimize vision impact.
        # 336x336 image -> 1 crop -> ~576 tokens (pooled to 144)
        
        # Actually, let's use a standard image size to be realistic
        image = Image.new('RGB', (336, 336), color='blue')
        
        lengths = list(range(50, max_length + 1, step_size))
        results_data = []
        
        log.info(f"Testing lengths: {lengths}")
        
        for length in tqdm(lengths, desc="Context Scaling"):
            # Construct prompt of approx 'length' tokens
            # "word " is 1 token usually.
            prompt = "word " * length
            
            # Process input
            inputs = processor.process(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Debug shapes
            log.info("Input Shapes:")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    log.info(f"  {k}: {v.shape}")
            
            # Ensure batch dimension
            if inputs["input_ids"].ndim == 1:
                log.info("Adding batch dimension...")
                inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
                if "images" in inputs and inputs["images"] is not None:
                    inputs["images"] = inputs["images"].unsqueeze(0)
                if "image_masks" in inputs and inputs["image_masks"] is not None:
                    inputs["image_masks"] = inputs["image_masks"].unsqueeze(0)
                if "image_input_idx" in inputs and inputs["image_input_idx"] is not None:
                    inputs["image_input_idx"] = inputs["image_input_idx"].unsqueeze(0)
            
            actual_length = inputs["input_ids"].shape[1]
            log.info(f"Target: {length}, Actual Input Tokens: {actual_length}")
            
            # Run measurements
            latencies = []
            for _ in range(self.num_warmup):
                self.measure_inference_latency(inputs, max_new_tokens=1, measure_components=True)
                
            for _ in range(num_samples):
                metrics = self.measure_inference_latency(inputs, max_new_tokens=1, measure_components=True)
                latencies.append(metrics["T_LLM_prefill"])
            
            stats = self.compute_statistics(latencies)
            stats["num_input_tokens"] = actual_length
            stats["target_text_length"] = length
            results_data.append(stats)
            
            log.info(f"Length {actual_length}: P50 Prefill = {stats['P50']:.2f} ms")

        # Save results
        self.save_results(results_data, "exp1_context_scaling_results.json")

def main():
    parser = argparse.ArgumentParser(description="Run Context Scaling Experiment")
    parser.add_argument("--model_path", type=str, default="hf:allenai/MolmoE-1B-0924")
    parser.add_argument("--output_dir", type=str, default="./results/context_scaling")
    parser.add_argument("--num_samples", type=int, default=50) # Lower samples for speed
    parser.add_argument("--max_length", type=int, default=1500)
    parser.add_argument("--step_size", type=int, default=100)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = ContextScalingExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    experiment.run(
        num_samples=args.num_samples,
        max_length=args.max_length,
        step_size=args.step_size
    )

if __name__ == "__main__":
    main()
