
import argparse
import logging
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from experiments.motivate.base_experiment import BaseExperiment, Timer

log = logging.getLogger(__name__)

class MoETopKExperiment(BaseExperiment):
    def run(self, num_samples: int = 100, top_k_values: list = [1, 2, 4, 8]):
        """
        Run MoE Top-K scaling experiment.
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
        
        log.info(f"Testing Top-K values: {top_k_values}")
        
        for k in tqdm(top_k_values, desc="Top-K Scaling"):
            log.info(f"Setting Top-K to {k}...")
            
            # === CORRECT APPROACH FOR Molmo-wrapped megablocks ===
            # Molmo.from_checkpoint() loads megablocks dMoE in block.ffn
            # Structure: block.ffn (type: megablocks.layers.dmoe.dMoE)
            #           block.ffn.args (type: megablocks.layers.arguments.Arguments)
            #           block.ffn.args.top_k (int)
            
            # Validate range
            assert 1 <= k <= self.model.config.moe_num_experts, \
                f"top_k must be between 1 and {self.model.config.moe_num_experts}"
            
            # 1. Update config (for consistency)
            self.model.config.moe_top_k = k
            log.info(f"Set model.config.moe_top_k = {k}")
          
            # 2. Update each MoE block's args.top_k
            moe_blocks_found = 0
            for i, block in enumerate(self.model.transformer.blocks):
                # For Molmo-wrapped models: block.ffn is dMoE
                if hasattr(block, 'ffn') and hasattr(block.ffn, 'args'):
                    if hasattr(block.ffn.args, 'top_k'):
                        old_k = block.ffn.args.top_k
                        block.ffn.args.top_k = k
                        moe_blocks_found += 1
                        if moe_blocks_found == 1:  # Log first one
                            log.info(f"Block {i}: Changed top_k from {old_k} to {k}")
            
            log.info(f"Updated {moe_blocks_found} MoE blocks to use top_k={k}")
            
            if moe_blocks_found == 0:
                log.warning("No MoE blocks found! Check model structure.")
            
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
                "top_k": k,
                "prefill": stats_prefill,
                "decode": stats_decode
            }
            results_data.append(combined_stats)
            
            log.info(f"Top-K {k}: P50 Prefill={stats_prefill['P50']:.2f}ms, P50 Decode={stats_decode['P50']:.2f}ms")

        # Save results
        self.save_results(results_data, "exp2_moe_topk_results.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="hf:allenai/MolmoE-1B-0924")
    parser.add_argument("--output_dir", type=str, default="./results/moe_topk")
    parser.add_argument("--num_samples", type=int, default=50)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = MoETopKExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    experiment.run(
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
