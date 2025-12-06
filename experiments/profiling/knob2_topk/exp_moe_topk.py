
import argparse
import logging
import sys
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())
from experiments.base_experiment import BaseExperiment, Timer
from molmo.models.modeling_molmoe import MolmoeSparseMoeBlock

log = logging.getLogger(__name__)

class MoETopKExperiment(BaseExperiment):
    def run(
        self,
        dataset_name: str = None,
        split: str = "validation",
        num_samples: int = 5000,
        batch_size: int = 1,
        max_new_tokens: int = 128,
        top_k_values: list = [1, 2, 4, 8, 16, 32],
    ):
        """
        Run MoE Top-K scaling experiment.
        
        Args:
            dataset_name: Dataset name (e.g., "coco_2014_vqa"). If None, uses dummy images.
            split: Dataset split (default: "validation")
            num_samples: Number of samples to measure (default: 5000 for dataset mode, 100 for dummy mode)
            batch_size: Batch size (default: 1 for latency measurement)
            max_new_tokens: Maximum tokens to generate
            top_k_values: List of top_k values to test
        """
        # Determine mode: dataset mode or dummy image mode
        use_dataset = dataset_name is not None
        
        if use_dataset:
            # Dataset mode: use VQA v2 validation set
            log.info(f"Testing Top-K values: {top_k_values}")
            log.info(f"Measuring latency on {num_samples} samples from {dataset_name}/{split}")
            
            results_data = []
            
            for k in top_k_values:
                log.info(f"=" * 80)
                log.info(f"Setting Top-K to {k}...")
                log.info(f"=" * 80)
                
                # Validate range
                assert 1 <= k <= self.model.config.moe_num_experts, \
                    f"top_k must be between 1 and {self.model.config.moe_num_experts}"
                
                # 1. Update config (for consistency)
                self.model.config.moe_top_k = k
                log.info(f"Set model.config.moe_top_k = {k}")
              
                # 2. Update each MoE block's top_k directly
                moe_blocks_found = 0
                
                # Get transformer blocks - handle both dict and attribute access
                transformer = self.model.model.transformer
                if isinstance(transformer, torch.nn.ModuleDict):
                    # ModuleDict: access via ["blocks"]
                    if "blocks" in transformer:
                        blocks = transformer["blocks"]
                    else:
                        log.error("transformer ModuleDict does not contain 'blocks' key")
                        blocks = []
                elif hasattr(transformer, 'blocks'):
                    # Direct attribute access
                    blocks = transformer.blocks
                else:
                    log.error("transformer does not have 'blocks' attribute or key")
                    blocks = []
                
                log.info(f"Found {len(blocks)} transformer blocks")
                
                for i, block in enumerate(blocks):
                    # For HF Molmo models: block.mlp is MolmoeSparseMoeBlock (PyTorch implementation)
                    # Check if block has mlp with top_k attribute (MoE block)
                    if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
                        mlp_type = type(block.mlp)
                        mlp_type_name = mlp_type.__name__ if hasattr(mlp_type, '__name__') else str(mlp_type)
                        
                        # Check by type name or isinstance (more flexible)
                        is_moe_block = (
                            isinstance(block.mlp, MolmoeSparseMoeBlock) or 
                            'MolmoeSparseMoeBlock' in mlp_type_name or
                            'SparseMoe' in mlp_type_name
                        )
                        
                        if is_moe_block:
                            old_k = block.mlp.top_k
                            block.mlp.top_k = k
                            moe_blocks_found += 1
                            if moe_blocks_found == 1:  # Log first one
                                log.info(f"Block {i}: Changed top_k from {old_k} to {k} (mlp type: {mlp_type_name})")
                
                log.info(f"Updated {moe_blocks_found} MoE blocks to use top_k={k}")
                
                if moe_blocks_found == 0:
                    log.warning("No MoE blocks found! Skipping this configuration.")
                    continue
                
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
                log.info(f"Measuring latency for Top-K={k}...")
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
                    
                    per_sample_results.append({
                        "sample_id": sample_idx,
                        "top_k": k,
                        **metrics
                    })
                
                # Compute statistics
                stats = self.compute_statistics(all_latencies)
                stats["top_k"] = k
                stats["per_sample_results"] = per_sample_results
                
                results_data.append(stats)
                
                log.info(f"Top-K={k}: Mean Latency={stats['mean']:.2f}ms, P50={stats['P50']:.2f}ms")
        else:
            # Dummy image mode: original behavior
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
                
                # Validate range
                assert 1 <= k <= self.model.config.moe_num_experts, \
                    f"top_k must be between 1 and {self.model.config.moe_num_experts}"
                
                # 1. Update config (for consistency)
                self.model.config.moe_top_k = k
                log.info(f"Set model.config.moe_top_k = {k}")
              
                # 2. Update each MoE block's top_k directly
                moe_blocks_found = 0
                
                # Get transformer blocks - handle both dict and attribute access
                transformer = self.model.model.transformer
                if isinstance(transformer, torch.nn.ModuleDict):
                    # ModuleDict: access via ["blocks"]
                    if "blocks" in transformer:
                        blocks = transformer["blocks"]
                    else:
                        log.error("transformer ModuleDict does not contain 'blocks' key")
                        blocks = []
                elif hasattr(transformer, 'blocks'):
                    # Direct attribute access
                    blocks = transformer.blocks
                else:
                    log.error("transformer does not have 'blocks' attribute or key")
                    blocks = []
                
                log.info(f"Found {len(blocks)} transformer blocks")
                
                for i, block in enumerate(blocks):
                    # For HF Molmo models: block.mlp is MolmoeSparseMoeBlock (PyTorch implementation)
                    # Check if block has mlp with top_k attribute (MoE block)
                    if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
                        mlp_type = type(block.mlp)
                        mlp_type_name = mlp_type.__name__ if hasattr(mlp_type, '__name__') else str(mlp_type)
                        
                        # Check by type name or isinstance (more flexible)
                        is_moe_block = (
                            isinstance(block.mlp, MolmoeSparseMoeBlock) or 
                            'MolmoeSparseMoeBlock' in mlp_type_name or
                            'SparseMoe' in mlp_type_name
                        )
                        
                        if is_moe_block:
                            old_k = block.mlp.top_k
                            block.mlp.top_k = k
                            moe_blocks_found += 1
                            if moe_blocks_found == 1:  # Log first one
                                log.info(f"Block {i}: Changed top_k from {old_k} to {k} (mlp type: {mlp_type_name})")
                
                log.info(f"Updated {moe_blocks_found} MoE blocks to use top_k={k}")
                
                if moe_blocks_found == 0:
                    log.warning("No MoE blocks found! Check model structure.")
                    log.warning(f"Model config moe_num_experts: {self.model.config.moe_num_experts}")
                    log.warning(f"Model config block_type: {getattr(self.model.config, 'block_type', 'N/A')}")
                
                # Run measurements
                latencies_prefill = []
                latencies_decode = []
                per_sample_results = []
                
                # Warmup
                for _ in range(self.num_warmup):
                    self.measure_inference_latency(inputs, max_new_tokens=10, measure_components=True)
                    
                for sample_idx in range(num_samples):
                    metrics = self.measure_inference_latency(inputs, max_new_tokens=10, measure_components=True)
                    latencies_prefill.append(metrics["T_LLM_prefill"])
                    latencies_decode.append(metrics["T_LLM_decode"])
                    
                    # Save per-sample result
                    per_sample_results.append({
                        "sample_id": sample_idx,
                        "top_k": k,
                        **metrics  # Include all metrics from measure_inference_latency
                    })
                
                stats_prefill = self.compute_statistics(latencies_prefill)
                stats_decode = self.compute_statistics(latencies_decode)
                
                combined_stats = {
                    "top_k": k,
                    "prefill": stats_prefill,
                    "decode": stats_decode,
                    "per_sample_results": per_sample_results
                }
                results_data.append(combined_stats)
                
                log.info(f"Top-K {k}: P50 Prefill={stats_prefill['P50']:.2f}ms, P50 Decode={stats_decode['P50']:.2f}ms")

        # Save results with per-sample data
        # Format: {"summary": [...], "all_samples": [...]}
        all_samples = []
        summary = []
        
        for config_result in results_data:
            # Extract summary stats
            summary_entry = {
                "top_k": config_result["top_k"],
                "prefill": config_result["prefill"],
                "decode": config_result["decode"]
            }
            summary.append(summary_entry)
            
            # Collect all per-sample results
            all_samples.extend(config_result["per_sample_results"])
        
        final_results = {
            "summary": summary,
            "all_samples": all_samples
        }
        
        self.save_results(final_results, "exp2_moe_topk_results.json")
        log.info(f"Total samples: {len(all_samples)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints",
                        help="Path to model checkpoint directory (local path, not HF hub path)")
    parser.add_argument("--output_dir", type=str, default="./results/moe_topk")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset name (e.g., 'coco_2014_vqa'). If None, uses dummy images.")
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split (default: validation)")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of samples to measure (default: 5000 for dataset mode, 100 for dummy mode)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1 for latency measurement)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--top_k_values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="List of top_k values to test. Default: [1, 2, 4, 8, 16, 32]")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = MoETopKExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    experiment.run(
        dataset_name=args.dataset_name,
        split=args.split,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        top_k_values=args.top_k_values
    )

if __name__ == "__main__":
    main()
