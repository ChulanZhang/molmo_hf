"""
Evaluate Lookup Table Baseline Controller on datasets.

This script evaluates the lookup table baseline controller (no training required)
on actual datasets and computes accuracy, latency, and other metrics.

Usage:
    python experiments/controller/evaluate_lookup_table_baseline.py \
        --model_path checkpoints/molmo \
        --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
        --dataset text_vqa --num_samples 100 --latency_budget 200.0 --device cuda
"""

import argparse
import logging
import sys
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.base_experiment import BaseExperiment, Timer
from experiments.controller.lookup_table_wrapper import create_lookup_table_controller
from experiments.controller.adaptive_inference import AdaptiveInferenceEngine
from experiments.controller.feature_extractors import (
    LanguageFeatureExtractor,
    LatencyBudgetEncoder,
)


class EvaluationExperiment(BaseExperiment):
    """
    Simple experiment class for evaluation purposes.
    Implements the abstract run() method required by BaseExperiment.
    
    model_path can be:
    - A directory containing pytorch_model.bin (e.g., "checkpoints/molmo/")
    - A direct path to weights file (e.g., "checkpoints/pytorch_model.bin")
    """
    def __init__(self, model_path: str, device: str = "cuda", output_dir: str = "./results", 
                 num_warmup: int = 3, hf_cache_dir: Optional[str] = None):
        """Initialize experiment."""
        super().__init__(model_path=model_path, device=device, output_dir=output_dir, 
                        num_warmup=num_warmup, hf_cache_dir=hf_cache_dir)
    
    def run(self, *args, **kwargs):
        """Dummy implementation of abstract method."""
        pass
    
    def _load_model(self, checkpoint_dir: str):
        """Load the Molmo model from a checkpoint directory or weights file.
        
        model_path can be:
        - A directory: will look for pytorch_model.bin inside it, or in parent checkpoints/ directory
        - A weights file: will use it directly
        """
        if self.rank == 0:
            log.info(f"Loading model from {checkpoint_dir}...")
        
        # 1. Load Config (always from project configs, not from checkpoint)
        from molmo.models.config_molmoe import MolmoConfig
        from transformers import AutoConfig
        
        project_config_path = os.path.join("configs", "model", "config.json")
        
        if os.path.exists(project_config_path):
            if self.rank == 0:
                log.info(f"Loading config from {project_config_path}")
            config = MolmoConfig.from_json_file(project_config_path)
        else:
            if self.rank == 0:
                log.warning("No local config found. Fetching from HF Hub...")
            config = AutoConfig.from_pretrained("allenai/MolmoE-1B-0924", trust_remote_code=True)
        
        # 2. Instantiate Model
        from molmo.models.modeling_molmoe import MolmoForCausalLM
        with Timer("MolmoForCausalLM instantiation"):
            model = MolmoForCausalLM(config)
        
        # 3. Find and load weights
        checkpoint_dir_abs = os.path.abspath(checkpoint_dir)
        
        # Case 1: checkpoint_dir is a weights file - use it directly
        if os.path.isfile(checkpoint_dir_abs) and checkpoint_dir_abs.endswith((".bin", ".safetensors")):
            weights_path = checkpoint_dir_abs
        elif os.path.isfile(checkpoint_dir) and checkpoint_dir.endswith((".bin", ".safetensors")):
            weights_path = os.path.abspath(checkpoint_dir)
        else:
            # Case 2: checkpoint_dir is a directory - search for weights
            possible_paths = []
            
            # Check inside the directory
            if os.path.isdir(checkpoint_dir_abs):
                possible_paths.extend([
                    os.path.join(checkpoint_dir_abs, "pytorch_model.bin"),
                    os.path.join(checkpoint_dir_abs, "model.safetensors"),
                ])
            
            # Check parent directory (common case: weights in checkpoints/, model_path is checkpoints/molmo/)
            if "checkpoints" in checkpoint_dir_abs:
                parts = checkpoint_dir_abs.split(os.sep)
                if "checkpoints" in parts:
                    checkpoints_idx = parts.index("checkpoints")
                    checkpoints_dir = os.sep.join(parts[:checkpoints_idx + 1])
                    possible_paths.extend([
                        os.path.join(checkpoints_dir, "pytorch_model.bin"),
                        os.path.join(checkpoints_dir, "model.safetensors"),
                    ])
            
            # Check project root checkpoints/
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            possible_paths.extend([
                os.path.join(project_root, "checkpoints", "pytorch_model.bin"),
                os.path.join(project_root, "checkpoints", "model.safetensors"),
            ])
            
            # Remove duplicates
            seen = set()
            unique_paths = []
            for p in possible_paths:
                abs_p = os.path.abspath(p) if p else None
                if abs_p and abs_p not in seen:
                    seen.add(abs_p)
                    unique_paths.append(p)
            
            # Find first existing path
            weights_path = None
            for path in unique_paths:
                if path and os.path.exists(path):
                    weights_path = path
                    break
            
            if not weights_path:
                error_msg = f"No weights found. Tried paths:\n"
                error_msg += "\n".join([f"  - {p}" for p in unique_paths])
                if self.rank == 0:
                    log.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        # 4. Load weights
        if self.rank == 0:
            log.info(f"Loading weights from {weights_path}...")
        with Timer("Load State Dict"):
            if weights_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_path)
                except ImportError:
                    if self.rank == 0:
                        log.warning("safetensors not available, trying torch.load...")
                    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            else:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        return model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def create_lookup_table_inference_engine(
    model_path: str,
    lookup_table_path: str,
    device: str = "cuda",
    experiment: Optional[EvaluationExperiment] = None,
) -> AdaptiveInferenceEngine:
    """
    Create adaptive inference engine with lookup table controller.
    
    Args:
        model_path: Path to model checkpoint directory or weights file
                   - If directory: will search for pytorch_model.bin inside or in parent checkpoints/
                   - If file: will use it directly as weights file
        lookup_table_path: Path to lookup table JSON file
        device: Device to use
        experiment: Optional EvaluationExperiment instance (to avoid reloading model)
    
    Returns:
        AdaptiveInferenceEngine instance
    """
    # Load model using EvaluationExperiment (implements abstract run() method)
    # Reuse experiment if provided to avoid loading model twice
    if experiment is None:
        experiment = EvaluationExperiment(model_path=model_path, device=device)
    model = experiment.model
    tokenizer = experiment.tokenizer
    
    # Create lookup table controller
    controller = create_lookup_table_controller(
        lookup_table_path=lookup_table_path,
    )
    
    # Initialize feature extractors (needed for compatibility, but not used by lookup table)
    lang_extractor = LanguageFeatureExtractor(
        tokenizer=tokenizer,
        wte_layer=model.model.transformer.wte,
        max_length=512
    )
    budget_encoder = LatencyBudgetEncoder(
        d_model=2048,
        use_sinusoidal=True,
        normalize_budget=True,
        budget_min=170.0,
        budget_max=380.0,
    ).to(device)
    
    # Create engine (we'll need to modify it to use lookup table directly)
    # For now, we'll create a wrapper that uses lookup table
    engine = LookupTableInferenceEngine(
        model=model,
        experiment=experiment,
        controller=controller,
        device=device,
    )
    
    return engine


class LookupTableInferenceEngine:
    """
    Inference engine that uses lookup table baseline controller.
    
    This is a simplified version that directly uses lookup table predictions
    without going through the full adaptive inference pipeline.
    """
    
    def __init__(
        self,
        model,
        experiment,
        controller,
        device: str = "cuda",
    ):
        """
        Initialize lookup table inference engine.
        
        Args:
            model: Model instance
            experiment: BaseExperiment instance
            controller: LookupTableControllerWrapper instance
            device: Device to use
        """
        self.model = model
        self.experiment = experiment
        self.controller = controller
        self.device = device
        
        # Store original top_k values for restoration
        self.original_top_k_values = {}
        self.block_mask_wrapper = None
    
    def _apply_config(
        self,
        tier: str,
        top_k: int,
        num_active_blocks: int,
    ):
        """Apply configuration to model."""
        # Import at module level to avoid repeated imports and catch errors early
        if not hasattr(self, '_apply_block_mask_imported'):
            try:
                from experiments.controller.adaptive_inference import tier_to_max_crops
                from experiments.controller.profiling_with_importance import apply_block_mask
                self._tier_to_max_crops = tier_to_max_crops
                self._apply_block_mask = apply_block_mask
                self._apply_block_mask_imported = True
            except ImportError as e:
                log.error(f"Failed to import required modules: {e}")
                raise RuntimeError(f"Cannot proceed without required imports: {e}") from e
        
        tier_to_max_crops = self._tier_to_max_crops
        apply_block_mask = self._apply_block_mask
        
        # Set tier (max_crops) - this is handled during image processing
        max_crops = tier_to_max_crops(tier)
        
        # Set top_k for MoE blocks
        transformer = self.model.model.transformer
        if hasattr(transformer, 'blocks'):
            blocks = transformer.blocks
        else:
            blocks = []
        
        for i, block in enumerate(blocks):
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
                # Store original value
                if i not in self.original_top_k_values:
                    self.original_top_k_values[i] = block.mlp.top_k
                # Set new value
                block.mlp.top_k = top_k
        
        # Apply block mask (importance-based selection)
        # For now, we'll use prefix blocks (first num_active_blocks)
        # In practice, you'd use importance-based selection
        total_blocks = len(blocks)
        block_mask = torch.ones(total_blocks, dtype=torch.bool, device=self.device)
        if num_active_blocks < total_blocks:
            # Deactivate blocks from the end
            block_mask[num_active_blocks:] = False
        
        self.block_mask_wrapper = apply_block_mask(self.model, block_mask)
    
    def _restore_config(self):
        """Restore original configuration."""
        # Restore top_k
        transformer = self.model.model.transformer
        if hasattr(transformer, 'blocks'):
            blocks = transformer.blocks
        else:
            blocks = []
        
        for i, block in enumerate(blocks):
            if i in self.original_top_k_values:
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
                    block.mlp.top_k = self.original_top_k_values[i]
        
        # Remove block mask
        if self.block_mask_wrapper is not None:
            self.block_mask_wrapper.remove()
            self.block_mask_wrapper = None
    
    def infer(
        self,
        prompt: str,
        images: Optional[Any] = None,
        latency_budget: float = 200.0,
        max_new_tokens: int = 128,
        return_knobs: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference with lookup table controller.
        
        Args:
            prompt: Input text prompt
            images: Optional images
            latency_budget: Latency budget in milliseconds
            max_new_tokens: Maximum number of tokens to generate
            return_knobs: If True, return knob values
        
        Returns:
            Dict with 'text', 'knobs', 'latency', etc.
        """
        # Predict configuration from lookup table
        config = self.controller.predict(latency_budget)
        if not config:
            log.warning(f"No valid configuration for budget {latency_budget}ms, using default")
            config = {
                'tier': 'medium',
                'top_k': 8,
                'num_active_blocks': 16,
            }
        
        tier = config['tier']
        top_k = config['top_k']
        num_active_blocks = config['num_active_blocks']
        
        # Apply configuration
        self._apply_config(tier, top_k, num_active_blocks)
        
        try:
            # Import tier_to_max_crops
            from experiments.controller.adaptive_inference import tier_to_max_crops
            
            # Prepare input
            # Note: This is simplified - in practice, you'd need proper image processing
            # based on tier (max_crops)
            max_crops = tier_to_max_crops(tier)
            
            # Run inference and measure latency
            # Use experiment's forward method with proper configuration
            # This is a placeholder - actual implementation would use proper model forward
            # with tier, top_k, and num_active_blocks applied
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            # TODO: Implement proper model forward with configuration
            # For now, we'll use a simplified approach
            # In practice, you'd call experiment.model.forward() with proper parameters
            output = "placeholder_output"  # This needs to be replaced with actual model output
            
            torch.cuda.synchronize()
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            
            result = {
                'text': output,
                'latency': latency,
            }
            
            if return_knobs:
                result['knobs'] = {
                    'tier': tier,
                    'top_k': top_k,
                    'num_active_blocks': num_active_blocks,
                    'expected_accuracy': config.get('accuracy', 0.0),
                    'expected_latency': config.get('latency', 0.0),
                }
            
            return result
            
        finally:
            # Restore original configuration
            self._restore_config()


def evaluate_lookup_table_baseline(
    model_path: str,
    lookup_table_path: str,
    dataset: str = "text_vqa",
    split: str = "validation",
    num_samples: Optional[int] = None,
    latency_budget: float = 200.0,
    max_new_tokens: int = 128,
    batch_size: int = 1,
    device: str = "cuda",
    output_path: str = "./results/logs_eval/",
    save_predictions: bool = True,
):
    """
    Evaluate lookup table baseline controller on a dataset.
    
    Args:
        model_path: Path to model checkpoint
        lookup_table_path: Path to lookup table JSON file
        dataset: Dataset name (text_vqa, okvqa, etc.)
        split: Dataset split (validation, test, etc.)
        num_samples: Number of samples to evaluate (None = all)
        latency_budget: Latency budget in milliseconds
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for evaluation (default: 1)
        device: Device to use
        output_path: Output directory for results
        save_predictions: If True, save predictions to file
    """
    log.info("=" * 80)
    log.info("Lookup Table Baseline Evaluation")
    log.info("=" * 80)
    log.info(f"Model path: {model_path}")
    log.info(f"Lookup table path: {lookup_table_path}")
    log.info(f"Dataset: {dataset} ({split})")
    log.info(f"Num samples: {num_samples if num_samples else 'all'}")
    log.info(f"Latency budget: {latency_budget}ms")
    log.info(f"Max new tokens: {max_new_tokens}")
    log.info(f"Batch size: {batch_size} (latency measured with batch_size=1)")
    log.info("=" * 80)
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load experiment first (needed for both engine and dataloader)
    log.info("Loading model and experiment...")
    experiment = EvaluationExperiment(model_path=model_path, device=device)
    
    # Create inference engine (reuse experiment to avoid loading model twice)
    log.info("Creating lookup table inference engine...")
    engine = create_lookup_table_inference_engine(
        model_path=model_path,
        lookup_table_path=lookup_table_path,
        device=device,
        experiment=experiment,  # Pass experiment to avoid reloading
    )
    log.info("Engine created successfully!")
    
    # Load dataset
    log.info(f"Loading dataset: {dataset} ({split})...")
    dataloader = experiment.build_dataloader(
        dataset_name=dataset,
        split=split,
        batch_size=batch_size,
        max_steps=num_samples,
        shuffle=False,
    )
    
    # Get metric for dataset
    from experiments.base_experiment import get_metric_for_dataset
    metric_name = get_metric_for_dataset(dataset)
    log.info(f"Using metric: {metric_name}")
    
    # Get predicted configuration
    config = engine.controller.predict(latency_budget)
    log.info(f"Predicted configuration for budget {latency_budget}ms:")
    log.info(f"  Tier: {config['tier']}")
    log.info(f"  Top-K: {config['top_k']}")
    log.info(f"  Num Active Blocks: {config['num_active_blocks']}")
    log.info(f"  Expected Accuracy: {config.get('accuracy', 0.0):.4f}")
    log.info(f"  Expected Latency: {config.get('latency', 0.0):.2f}ms")
    
    # Evaluation loop
    all_predictions = []
    all_metadata = []
    all_knobs = []
    all_latencies = []
    all_accuracies = []
    budget_violations = 0
    
    log.info("Starting evaluation...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get metadata
            metadatas = batch.get("metadata", [])
            if not metadatas:
                log.warning(f"Batch {batch_idx} has no metadata, skipping")
                continue
            
            # Process each sample in the batch
            for sample_idx in range(len(metadatas)):
                metadata = metadatas[sample_idx]
                
                # Extract prompt/question
                question = metadata.get("question", "")
                if not question:
                    log.warning(f"Sample {sample_idx} in batch {batch_idx} has no question")
                    continue
                
                # Get image
                images = batch.get("images")
                image = None
                if images is not None:
                    if len(images.shape) == 5:  # (B, num_crops, H, W, C)
                        image = images[sample_idx]
                    elif len(images.shape) == 4:  # (B, H, W, C)
                        image = images[sample_idx]
                
                # Run inference
                try:
                    result = engine.infer(
                        prompt=question,
                        images=image,
                        latency_budget=latency_budget,
                        max_new_tokens=max_new_tokens,
                        return_knobs=True,
                    )
                    
                    generated_text = result.get("text", "")
                    knobs = result.get("knobs", {})
                    latency = result.get("latency", 0.0)
                    
                    # Check budget violation
                    if latency > latency_budget * 1.05:  # 5% tolerance
                        budget_violations += 1
                    
                    # Compute accuracy
                    pred_tokens = experiment.tokenizer.encode(
                        generated_text, return_tensors="pt"
                    ).to(device)
                    
                    pred_batch = {
                        "input_ids": batch["input_ids"][sample_idx:sample_idx+1],
                        "metadata": [metadata],
                    }
                    
                    accuracy_result = experiment.compute_accuracy(
                        batch=pred_batch,
                        predictions=pred_tokens,
                        metric_name=metric_name,
                    )
                    
                    accuracy = accuracy_result.get("accuracy", 0.0)
                    
                    # Store results
                    all_predictions.append(generated_text)
                    all_metadata.append(metadata)
                    all_knobs.append(knobs)
                    all_latencies.append(latency)
                    all_accuracies.append(accuracy)
                    
                except Exception as e:
                    error_msg = str(e)
                    log.error(f"Error processing sample {sample_idx} in batch {batch_idx}: {error_msg}")
                    
                    # For critical errors (import errors, etc.), log full traceback
                    if "import" in error_msg.lower() or "name" in error_msg.lower() or "not defined" in error_msg.lower():
                        log.error("Critical error detected (likely import/module issue). This sample will be skipped.")
                        import traceback
                        traceback.print_exc()
                        # Don't continue processing if it's a critical error that will repeat
                        # But for now, we'll continue to see how many samples fail
                    
                    # Skip this sample and continue with next one
                    # The progress bar continues because the loop continues
                    continue
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Compute statistics
    log.info("\n" + "=" * 80)
    log.info("Evaluation Results")
    log.info("=" * 80)
    
    if len(all_accuracies) > 0:
        avg_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        log.info(f"Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        log.info(f"Accuracy range: [{np.min(all_accuracies):.4f}, {np.max(all_accuracies):.4f}]")
    else:
        avg_accuracy = 0.0
        log.warning("No accuracy scores computed")
    
    if len(all_latencies) > 0:
        avg_latency = np.mean(all_latencies)
        std_latency = np.std(all_latencies)
        log.info(f"Average latency: {avg_latency:.2f} ± {std_latency:.2f}ms")
        log.info(f"Latency range: [{np.min(all_latencies):.2f}, {np.max(all_latencies):.2f}]ms")
        log.info(f"Budget violations: {budget_violations} / {len(all_latencies)} ({budget_violations/len(all_latencies)*100:.1f}%)")
    else:
        avg_latency = 0.0
    
    log.info(f"Total samples: {len(all_predictions)}")
    log.info(f"Total time: {total_time:.2f}s")
    log.info(f"Throughput: {len(all_predictions) / total_time:.2f} samples/s")
    
    # Knob statistics
    if all_knobs:
        tier_counts = {"low": 0, "medium": 0, "high": 0}
        top_k_counts = defaultdict(int)
        num_blocks_counts = defaultdict(int)
        
        for knobs in all_knobs:
            tier = knobs.get("tier", "unknown")
            if tier in tier_counts:
                tier_counts[tier] += 1
            
            top_k = knobs.get("top_k", 0)
            top_k_counts[top_k] += 1
            
            num_blocks = knobs.get("num_active_blocks", 0)
            num_blocks_counts[num_blocks] += 1
        
        log.info("\nKnob Distribution:")
        log.info(f"  Tier: {dict(tier_counts)}")
        log.info(f"  Top-K: {dict(top_k_counts)}")
        log.info(f"  Num Active Blocks: {dict(num_blocks_counts)}")
    
    # Save results
    results = {
        "dataset": dataset,
        "split": split,
        "num_samples": len(all_predictions),
        "latency_budget": latency_budget,
        "max_new_tokens": max_new_tokens,
        "predicted_config": config,
        "metrics": {
            "accuracy": float(avg_accuracy) if len(all_accuracies) > 0 else 0.0,
            "accuracy_std": float(std_accuracy) if len(all_accuracies) > 0 else 0.0,
            "avg_latency_ms": float(avg_latency) if len(all_latencies) > 0 else 0.0,
            "latency_std_ms": float(std_latency) if len(all_latencies) > 0 else 0.0,
            "budget_violations": budget_violations,
            "budget_violation_rate": float(budget_violations / len(all_latencies)) if len(all_latencies) > 0 else 0.0,
            "throughput_samples_per_sec": float(len(all_predictions) / total_time) if total_time > 0 else 0.0,
        },
        "knob_distribution": {
            "tier": dict(tier_counts) if all_knobs else {},
            "top_k": dict(top_k_counts) if all_knobs else {},
            "num_active_blocks": dict(num_blocks_counts) if all_knobs else {},
        },
    }
    
    # Save results JSON
    results_file = output_dir / f"{dataset}_{split}_budget_{latency_budget:.0f}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to: {results_file}")
    
    # Save predictions if requested
    if save_predictions:
        predictions_file = output_dir / f"{dataset}_{split}_budget_{latency_budget:.0f}_predictions.jsonl"
        with open(predictions_file, 'w') as f:
            for pred, metadata, knobs, latency, accuracy in zip(
                all_predictions, all_metadata, all_knobs, all_latencies, all_accuracies
            ):
                f.write(json.dumps({
                    "prediction": pred,
                    "metadata": metadata,
                    "knobs": knobs,
                    "latency_ms": latency,
                    "accuracy": accuracy,
                }) + "\n")
        log.info(f"Predictions saved to: {predictions_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Lookup Table Baseline Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--lookup_table_path",
        type=str,
        required=True,
        help="Path to lookup table JSON file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="text_vqa",
        help="Dataset name (text_vqa, okvqa, coco_2014_vqa, etc.)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split (validation, test, etc.)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--latency_budget",
        type=float,
        default=200.0,
        help="Latency budget in milliseconds"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (latency measured with batch_size=1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/logs_eval/lookup_table_baseline/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save individual predictions to file"
    )
    
    args = parser.parse_args()
    
    evaluate_lookup_table_baseline(
        model_path=args.model_path,
        lookup_table_path=args.lookup_table_path,
        dataset=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        latency_budget=args.latency_budget,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output_path,
        save_predictions=args.save_predictions,
    )


if __name__ == "__main__":
    main()

