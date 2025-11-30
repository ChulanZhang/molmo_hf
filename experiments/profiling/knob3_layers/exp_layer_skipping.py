
import argparse
import logging
import torch
import copy
from PIL import Image
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
from experiments.motivate.base_experiment import BaseExperiment, Timer

log = logging.getLogger(__name__)

class LayerSkippingExperiment(BaseExperiment):
    def run(self, num_samples: int = 100, layers_to_keep: list = [24, 12, 6]):
        """
        Run Layer Skipping experiment.
        
        Args:
            num_samples: Number of repetitions.
            layers_to_keep: List of layer counts to test.
        """
        # Processor
        # Use self.processor loaded in BaseExperiment
        processor = self.processor
        
        # Standard input
        image = Image.new('RGB', (336, 336), color='blue')
        prompt = "Describe this image."
        inputs = processor.process(text=prompt, images=image)
        
        # Ensure batch dimension
        if inputs["input_ids"].ndim == 1:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
            if "images" in inputs and inputs["images"] is not None:
                inputs["images"] = inputs["images"].unsqueeze(0)
            if "image_masks" in inputs and inputs["image_masks"] is not None:
                inputs["image_masks"] = inputs["image_masks"].unsqueeze(0)
            if "image_input_idx" in inputs and inputs["image_input_idx"] is not None:
                inputs["image_input_idx"] = inputs["image_input_idx"].unsqueeze(0)
        
        results_data = []
        
        # Store original blocks to restore later
        original_blocks = copy.copy(self.model.transformer.blocks)
        total_layers = len(original_blocks)
        
        log.info(f"Total available layers: {total_layers}")
        log.info(f"Testing layer counts: {layers_to_keep}")
        
        for n_layers in tqdm(layers_to_keep, desc="Layer Scaling"):
            if n_layers > total_layers:
                log.warning(f"Skipping {n_layers} layers (max {total_layers})")
                continue
                
            log.info(f"Setting model to use first {n_layers} layers...")
            
            # Slice the blocks
            # Note: This is a simplistic way to 'skip' layers by just removing them from the list.
            # For a real 'skipping' implementation without modifying structure, one would need to change forward pass.
            # But modifying the list is effective for measuring latency of a smaller model.
            self.model.transformer.blocks = torch.nn.ModuleList(original_blocks[:n_layers])
            
            # Update config to match (some internal logic might rely on it)
            self.model.config.num_hidden_layers = n_layers
            
            # Run measurements
            latencies_prefill = []
            latencies_decode = []
            
            # Warmup
            for _ in range(self.num_warmup):
                self.measure_inference_latency(inputs, max_new_tokens=10, measure_components=True)
                
            for _ in range(num_samples):
                metrics = self.measure_inference_latency(inputs, max_new_tokens=10, measure_components=True)
                latencies_prefill.append(metrics["T_LLM_prefill"])
                latencies_decode.append(metrics["T_LLM_decode"])
            
            stats_prefill = self.compute_statistics(latencies_prefill)
            stats_decode = self.compute_statistics(latencies_decode)
            
            combined_stats = {
                "num_layers": n_layers,
                "prefill": stats_prefill,
                "decode": stats_decode
            }
            results_data.append(combined_stats)
            
            log.info(f"Layers {n_layers}: P50 Prefill={stats_prefill['P50']:.2f}ms, P50 Decode={stats_decode['P50']:.2f}ms")

        # Restore model
        self.model.transformer.blocks = original_blocks
        self.model.config.num_hidden_layers = total_layers
        
        # Save results
        self.save_results(results_data, "exp_layer_skipping_results.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="hf:allenai/MolmoE-1B-0924")
    parser.add_argument("--output_dir", type=str, default="./results/layer_skipping")
    parser.add_argument("--num_samples", type=int, default=50)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = LayerSkippingExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    # Default MolmoE-1B has 24 layers. Let's test full, half, quarter.
    experiment.run(
        num_samples=args.num_samples,
        layers_to_keep=[24, 18, 12, 6]
    )

if __name__ == "__main__":
    main()
