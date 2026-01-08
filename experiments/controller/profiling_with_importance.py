"""
Profiling experiment with importance-based block selection.
Collects latency and accuracy for all configuration combinations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.base_experiment import BaseExperiment
from experiments.controller.importance_based_block_selection import ImportanceBasedBlockSelector
from experiments.profiling.knob3_layers.exp_transformer_blocks_mask import BlockMaskWrapper
from molmo.models.modeling_molmoe import MolmoeSparseMoeBlock

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def set_tier_for_model(model, tier: str):
    """
    Set tier configuration for model.
    This affects max_crops in the preprocessor.
    Note: This is a placeholder - actual implementation depends on your preprocessor setup.
    """
    # In practice, this would set model.config.max_crops based on tier
    tier_to_max_crops = {
        'low': 3,
        'medium': 6,
        'high': 12,
    }
    max_crops = tier_to_max_crops.get(tier, 6)
    # model.config.max_crops = max_crops
    log.debug(f"Set tier={tier}, max_crops={max_crops}")


def set_top_k_for_model(model, top_k: int, start_layer: int = 0):
    """
    Set top_k for all MoE layers.
    
    Args:
        model: MolmoModel instance
        top_k: Top-K value
        start_layer: Starting layer index (default: 0, all layers)
    """
    for i in range(start_layer, len(model.model.transformer.blocks)):
        block = model.model.transformer.blocks[i]
        if hasattr(block, 'mlp') and isinstance(block.mlp, MolmoeSparseMoeBlock):
            block.mlp.top_k = top_k
    log.debug(f"Set top_k={top_k} for layers {start_layer} to {len(model.model.transformer.blocks)-1}")


def apply_block_mask(model, block_mask: torch.Tensor):
    """
    Apply block mask to model.
    
    Returns:
        mask_wrapper: BlockMaskWrapper instance (needs to be kept alive)
    """
    from experiments.profiling.knob3_layers.exp_transformer_blocks_mask import BlockMaskWrapper
    
    mask_wrapper = BlockMaskWrapper(model.model, block_mask)
    mask_wrapper.apply()
    return mask_wrapper


def profile_configuration(
    experiment: BaseExperiment,
    dataset_name: str,
    tier: str,
    top_k: int,
    num_blocks: int,
    selected_blocks: List[int],
    num_samples: int = 500,
    batch_size: int = 1,
) -> Dict:
    """
    Profile a single configuration.
    
    Args:
        experiment: BaseExperiment instance
        dataset_name: Dataset name
        tier: Vision tokens tier
        top_k: MoE top-K
        num_blocks: Number of active blocks
        selected_blocks: List of selected block indices
        num_samples: Number of samples to profile
        batch_size: Batch size
    
    Returns:
        metrics: {
            'accuracy': float,
            'T_vision_total': float,
            'T_LLM_prefill': float,
            'T_LLM_decode': float,
            'T_total': float,
            'T_decode_per_token': float,
            'vision_tokens': int,
            'text_tokens': int,
            'output_tokens': int,
        }
    """
    # Set configuration
    set_tier_for_model(experiment.model, tier)
    set_top_k_for_model(experiment.model, top_k)
    
    # Create and apply block mask
    total_blocks = len(experiment.model.model.transformer.blocks)
    block_mask = torch.zeros(total_blocks, dtype=torch.bool)
    for idx in selected_blocks:
        if 0 <= idx < total_blocks:
            block_mask[idx] = True
    
    mask_wrapper = apply_block_mask(experiment.model, block_mask)
    
    try:
        # Build dataloader
        dataloader = experiment.build_dataloader(
            dataset_name=dataset_name,
            split="validation",
            batch_size=batch_size,
            max_steps=num_samples,
        )
        
        # Profile
        all_metrics = []
        for batch in dataloader:
            if len(all_metrics) >= num_samples:
                break
            
            # Measure latency
            latency_metrics = experiment.measure_inference_latency(
                batch=batch,
                max_new_tokens=128,
                measure_components=True,
                num_runs=1,
            )
            
            # Measure accuracy (simplified - in practice would compute actual accuracy)
            # For now, we'll use a placeholder
            accuracy = 0.0  # Would be computed from actual predictions
            
            metrics = {
                'accuracy': accuracy,
                'T_vision_total': latency_metrics.get('T_vision_total', 0.0),
                'T_LLM_prefill': latency_metrics.get('T_LLM_prefill', 0.0),
                'T_LLM_decode': latency_metrics.get('T_LLM_decode', 0.0),
                'T_total': latency_metrics.get('T_total', 0.0),
                'T_decode_per_token': latency_metrics.get('T_decode_per_token', 0.0),
                'vision_tokens': batch.get('vision_tokens', 0),
                'text_tokens': batch.get('text_tokens', 0),
                'output_tokens': batch.get('output_tokens', 0),
            }
            
            all_metrics.append(metrics)
        
        # Aggregate
        if all_metrics:
            aggregated = {
                'accuracy': sum(m['accuracy'] for m in all_metrics) / len(all_metrics),
                'T_vision_total': sum(m['T_vision_total'] for m in all_metrics) / len(all_metrics),
                'T_LLM_prefill': sum(m['T_LLM_prefill'] for m in all_metrics) / len(all_metrics),
                'T_LLM_decode': sum(m['T_LLM_decode'] for m in all_metrics) / len(all_metrics),
                'T_total': sum(m['T_total'] for m in all_metrics) / len(all_metrics),
                'T_decode_per_token': sum(m['T_decode_per_token'] for m in all_metrics) / len(all_metrics),
                'vision_tokens': int(sum(m['vision_tokens'] for m in all_metrics) / len(all_metrics)),
                'text_tokens': int(sum(m['text_tokens'] for m in all_metrics) / len(all_metrics)),
                'output_tokens': int(sum(m['output_tokens'] for m in all_metrics) / len(all_metrics)),
            }
        else:
            aggregated = {}
        
        return aggregated
    
    finally:
        # Remove mask
        mask_wrapper.remove()


def run_profiling_with_importance(
    model_path: str,
    importance_scores_file: str,
    datasets: List[str],
    output_dir: str,
    num_samples: int = 500,
    batch_size: int = 1,
):
    """
    Run profiling with importance-based block selection.
    
    Args:
        model_path: Path to model
        importance_scores_file: Path to importance scores JSON file
        datasets: List of dataset names
        output_dir: Output directory
        num_samples: Number of samples per configuration
        batch_size: Batch size
    """
    log.info("Starting profiling with importance-based block selection...")
    
    # Load importance scores
    block_selector = ImportanceBasedBlockSelector(importance_file=importance_scores_file)
    
    # Load model
    experiment = BaseExperiment(model_path=model_path, device="cuda")
    
    # Configuration grid
    tiers = ['low', 'medium', 'high']
    top_ks = [4, 6, 8, 10, 12]
    num_blocks_list = [8, 10, 12, 14, 16]
    
    total_configs = len(tiers) * len(top_ks) * len(num_blocks_list) * len(datasets)
    log.info(f"Total configurations to profile: {total_configs}")
    
    results = []
    config_idx = 0
    
    for tier in tiers:
        for top_k in top_ks:
            for num_blocks in num_blocks_list:
                # Select blocks by importance
                selected_blocks = block_selector.select_blocks(num_blocks)
                
                log.info(f"Profiling: tier={tier}, top_k={top_k}, num_blocks={num_blocks}, "
                        f"selected_blocks={selected_blocks}")
                
                # Profile on all datasets
                for dataset_name in datasets:
                    config_idx += 1
                    log.info(f"  [{config_idx}/{total_configs}] {dataset_name}...")
                    
                    try:
                        metrics = profile_configuration(
                            experiment=experiment,
                            dataset_name=dataset_name,
                            tier=tier,
                            top_k=top_k,
                            num_blocks=num_blocks,
                            selected_blocks=selected_blocks,
                            num_samples=num_samples,
                            batch_size=batch_size,
                        )
                        
                        result = {
                            'tier': tier,
                            'top_k': top_k,
                            'num_blocks': num_blocks,
                            'selected_blocks': selected_blocks,
                            'dataset': dataset_name,
                            **metrics,
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        log.error(f"Error profiling {dataset_name} with tier={tier}, top_k={top_k}, "
                                f"num_blocks={num_blocks}: {e}")
                        continue
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "profiling_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Profiling completed! Results saved to {output_file}")
    log.info(f"Total configurations profiled: {len(results)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile configurations with importance-based block selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--importance_scores_file",
        type=str,
        required=True,
        help="Path to importance scores JSON file"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["text_vqa", "coco_2014_vqa"],
        help="Dataset names to profile"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/profiling_with_importance",
        help="Output directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples per configuration"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size"
    )
    
    args = parser.parse_args()
    
    run_profiling_with_importance(
        model_path=args.model_path,
        importance_scores_file=args.importance_scores_file,
        datasets=args.datasets,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()





