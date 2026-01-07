"""
Exp3 Accuracy Sensitivity Analysis V2: Beam Search for Block Combination Exploration
Multi-GPU, measure accuracy with beam search-based layer removal.

Key improvements:
1. Beam search method: Explore block combinations considering interactions
2. Multi-dataset support: Test on multiple datasets to verify consistency
3. Large batch size: Optimize for throughput (no latency concern)
4. Proper metrics: Use dataset-specific evaluation metrics

Beam Search Strategy:
- Step 1: Remove 1 block, test all 15 possibilities, keep top 3 with least impact
- Step 2: For each of top 3, remove another block, test 14 possibilities, keep top 3
- Step 3: Continue until max 4 blocks removed (min 12 blocks remain)
"""

import argparse
import logging
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Set TOKENIZERS_PARALLELISM to avoid warnings
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from torch.utils.data import DistributedSampler

sys.path.append(os.getcwd())
from experiments.base_experiment import BaseExperiment, get_metric_for_dataset
from molmo.torch_util import get_world_size, get_global_rank, get_local_rank

# Monkey patch to fix image_flat shape issue in modeling_molmoe.py
# Bug: image_flat needs to be flattened from (batch_size, num_image * num_patch, d_model)
#      to (batch_size * num_image * num_patch, d_model) before index_add_
# Fix: Patch torch.Tensor.index_add_ to automatically flatten source if needed
# Note: This patch is only applied when source is 3D and dim is 0, matching the bug case
if not hasattr(torch.Tensor, '_original_index_add_'):
    torch.Tensor._original_index_add_ = torch.Tensor.index_add_
    
    def patched_index_add_(self_tensor, dim, index, source, alpha=1):
        # Only apply fix for the specific bug case: 3D source with dim=0
        # This matches the case in modeling_molmoe.py line 1968
        if dim == 0 and source.dim() == 3:
            # Flatten from (batch_size, num_patches, d_model) to (batch_size * num_patches, d_model)
            # This ensures source.size(0) == index.size(0) as required by index_add_
            original_source_shape = source.shape
            source = source.view(-1, source.shape[-1])
            
            # Verify that source and index have compatible sizes
            if source.size(0) != index.size(0):
                # This should not happen in normal operation, but if it does, provide detailed error
                raise RuntimeError(
                    f"index_add_: After flattening 3D source, size mismatch. "
                    f"Original source shape: {original_source_shape}, "
                    f"Flattened source shape: {source.shape}, "
                    f"Index shape: {index.shape}, "
                    f"Index range: [{index.min().item()}, {index.max().item()}]"
                )
            
            # Check for invalid indices (negative or out of bounds)
            if index.numel() > 0:
                max_valid_idx = self_tensor.size(0) - 1
                invalid_mask = (index < 0) | (index > max_valid_idx)
                n_invalid = invalid_mask.sum().item()
                if n_invalid > 0:
                    # Filter out invalid indices and corresponding source elements
                    valid_mask = ~invalid_mask
                    index = index[valid_mask]
                    source = source[valid_mask]
                    if index.numel() == 0:
                        # All indices were invalid, nothing to add
                        return self_tensor
        
        return torch.Tensor._original_index_add_(self_tensor, dim, index, source, alpha=alpha)
    
    torch.Tensor.index_add_ = patched_index_add_

# Import BlockMaskWrapper
sys.path.append(os.path.join(os.path.dirname(__file__)))
from exp_transformer_blocks_mask import BlockMaskWrapper

log = logging.getLogger(__name__)


@dataclass
class BeamState:
    """Represents a state in beam search: which blocks are active"""
    active_blocks: List[int]  # List of active block indices
    removed_blocks: List[int]  # List of removed block indices
    accuracy: float = 0.0  # Accuracy with this configuration
    accuracy_drop: float = 0.0  # Drop from baseline (higher = worse)
    
    def __hash__(self):
        return hash(tuple(sorted(self.active_blocks)))
    
    def __eq__(self, other):
        if not isinstance(other, BeamState):
            return False
        return sorted(self.active_blocks) == sorted(other.active_blocks)


class Exp3SensitivityExperimentV2(BaseExperiment):
    """
    Exp3 Sensitivity V2: Beam search-based block combination exploration.
    
    Stage 1: Sensitivity Analysis (same as V1)
    - Ablate each layer individually to compute initial importance scores
    
    Stage 2: Beam Search Pruning
    - Use beam search to explore block combinations
    - Consider block interactions, not just individual importance
    - Keep top-K candidates at each step
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        output_dir: str = "./results",
        num_warmup: int = 3,
        hf_cache_dir: Optional[str] = None,
    ):
        # Auto-detect distributed environment
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.is_distributed = True
        else:
            self.is_distributed = False
        
        self.rank = get_global_rank()
        self.world_size = get_world_size()
        
        # Set device based on local rank if distributed
        if self.is_distributed:
            local_rank = get_local_rank()
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available but distributed training is enabled")
            num_gpus = torch.cuda.device_count()
            if local_rank >= num_gpus:
                raise RuntimeError(
                    f"local_rank ({local_rank}) is >= num_gpus ({num_gpus}). "
                    f"Please use --nproc-per-node={num_gpus} or fewer."
                )
            device = f"cuda:{local_rank}"
            
            # Stagger device initialization to avoid OOM from concurrent model loading
            # Each rank waits a bit before setting device, with rank 0 going first
            import time
            time.sleep(local_rank * 0.5)  # Stagger by 0.5s per rank
            
            # Clear CUDA cache before setting device to avoid OOM
            # Each rank clears its own cache
            try:
                # Try to set device first (may fail if GPU is already in use)
                torch.cuda.set_device(local_rank)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA error" in str(e):
                    # Clear cache and wait a bit longer
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(2.0 + local_rank * 0.5)  # Wait longer for other processes
                    # Retry
                    torch.cuda.empty_cache()
                    torch.cuda.set_device(local_rank)
                else:
                    raise
            
            # Synchronize after device is set
            torch.cuda.synchronize()
            
            log.info(f"Rank {self.rank} (local_rank {local_rank}) using device {device}")
        
        # Clear cache again before loading model (model loading is memory-intensive)
        if self.is_distributed:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Additional stagger before model loading
            import time
            time.sleep(self.rank * 0.3)
        
        super().__init__(
            model_path=model_path,
            device=device,
            output_dir=output_dir,
            num_warmup=num_warmup,
            hf_cache_dir=hf_cache_dir,
        )
        
        # Clear cache after model is loaded
        if self.is_distributed:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _find_optimal_batch_size(
        self,
        num_active_blocks: int,
        total_blocks: int,
        initial_batch_size: int,
        dataloader_factory,
        max_attempts: int = 5,
    ) -> int:
        """
        Dynamically find the optimal batch size for maximum throughput.
        Since we don't care about latency, we can use large batch sizes.
        """
        # For accuracy-only experiments, we can be more aggressive with batch size
        # But start conservatively to avoid OOM
        current_batch_size = initial_batch_size  # Start with initial batch size
        
        # Scale based on active blocks (fewer blocks = more memory available)
        # But be more conservative to avoid OOM
        ratio = num_active_blocks / total_blocks if total_blocks > 0 else 1.0
        if ratio <= 0.5:
            max_allowed_batch_size = initial_batch_size * 2  # More conservative
        elif ratio <= 0.75:
            max_allowed_batch_size = int(initial_batch_size * 1.5)
        else:
            max_allowed_batch_size = initial_batch_size  # Start conservative
        
        current_batch_size = min(current_batch_size, max_allowed_batch_size)
        
        log.info(f"Finding optimal batch size for {num_active_blocks}/{total_blocks} active blocks "
                f"(starting: {current_batch_size}, max: {max_allowed_batch_size})...")
        
        min_working = None
        max_failing = 0
        
        for attempt in range(max_attempts):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                dataloader = dataloader_factory(current_batch_size)
                test_batch = next(iter(dataloader))
                test_batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                             for k, v in test_batch.items()}
                
                with torch.inference_mode():
                    # Disable autocast to avoid dtype mismatch in index_add_ operation
                    # (image_flat becomes BFloat16 but x_flat remains Float32)
                    # Since we're focusing on accuracy, not latency, this is acceptable
                    # with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        _ = self.model(
                            input_ids=test_batch["input_ids"],
                            images=test_batch.get("images"),
                            image_masks=test_batch.get("image_masks"),
                            image_input_idx=test_batch.get("image_input_idx"),
                        )
                
                min_working = current_batch_size
                log.info(f"✓ Batch size {current_batch_size} works")
                
                # Try to increase if below max (but be conservative to avoid OOM)
                if current_batch_size < max_allowed_batch_size and attempt < max_attempts - 1:
                    # Increase more conservatively (1.2x instead of 1.5x)
                    next_try = min(max_allowed_batch_size, int(current_batch_size * 1.2))
                    if next_try > current_batch_size:
                        current_batch_size = next_try
                        continue
                
                return min_working
                
            except RuntimeError as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "oom" in error_str:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    max_failing = current_batch_size
                    old_batch_size = current_batch_size
                    
                    if min_working is not None:
                        current_batch_size = max(1, (min_working + max_failing) // 2)
                    else:
                        current_batch_size = max(1, current_batch_size // 2)
                    
                    log.warning(f"✗ Batch size {old_batch_size} OOM, trying {current_batch_size}...")
                    
                    if current_batch_size == old_batch_size:
                        break
                elif "cuda error" in error_str or "invalid configuration" in error_str:
                    # CUDA configuration error - likely due to invalid tensor dimensions
                    # Try smaller batch size
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    max_failing = current_batch_size
                    old_batch_size = current_batch_size
                    
                    if min_working is not None:
                        current_batch_size = max(1, (min_working + max_failing) // 2)
                    else:
                        current_batch_size = max(1, current_batch_size // 2)
                    
                    log.warning(f"✗ Batch size {old_batch_size} CUDA config error, trying {current_batch_size}...")
                    
                    if current_batch_size == old_batch_size:
                        break
                else:
                    # Other errors - log and re-raise
                    log.error(f"Unexpected error during batch size optimization: {e}")
                    raise
        
        if min_working is not None:
            return min_working
        else:
            # If we couldn't find a working batch size, try batch_size=1 as last resort
            log.warning(f"Could not find working batch size, trying batch_size=1 as last resort...")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                dataloader = dataloader_factory(1)
                test_batch = next(iter(dataloader))
                test_batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                             for k, v in test_batch.items()}
                
                with torch.inference_mode():
                    # Disable autocast to avoid dtype mismatch in index_add_ operation
                    # (image_flat becomes BFloat16 but x_flat remains Float32)
                    # Since we're focusing on accuracy, not latency, this is acceptable
                    # with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        _ = self.model(
                            input_ids=test_batch["input_ids"],
                            images=test_batch.get("images"),
                            image_masks=test_batch.get("image_masks"),
                            image_input_idx=test_batch.get("image_input_idx"),
                        )
                log.info(f"✓ Batch size 1 works, using it")
                return 1
            except Exception as e:
                log.error(f"Even batch_size=1 failed: {e}")
                log.warning(f"Using {current_batch_size} as fallback (may not work)")
                return max(1, current_batch_size)
    
    def _compute_accuracy_on_subset(
        self,
        dataloader,
        max_samples: Optional[int] = None,
        max_new_tokens: int = 16,
        metric_name: str = "vqa_score",
        mask_applied: bool = False,
    ) -> Tuple[float, List[Dict]]:
        """
        Compute accuracy on a subset of the dataset.
        
        Args:
            max_samples: Maximum number of samples to evaluate. If None, use all samples.
        """
        if max_samples is None:
            # Use all samples
            max_samples_per_rank = None
        elif self.is_distributed:
            max_samples_per_rank = max_samples // self.world_size
            if self.rank == 0:
                max_samples_per_rank += max_samples % self.world_size
        else:
            max_samples_per_rank = max_samples
        
        all_scores = []
        all_predictions = []
        
        # Determine total batches to process
        if max_samples_per_rank is None:
            total_batches = len(dataloader)
        else:
            total_batches = min((max_samples_per_rank + dataloader.batch_size - 1) // dataloader.batch_size, len(dataloader))
        
        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches)):
                try:
                    batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    from transformers import GenerationConfig
                    
                    eos_token_id = self.tokenizer.eos_token_id
                    if eos_token_id is None:
                        eos_token_id = getattr(self.model.config, 'eos_token_id', None)
                    
                    pad_token_id = self.tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = getattr(self.model.config, 'pad_token_id', None)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if hasattr(self.model.model, '_MolmoModel__cache'):
                        self.model.model._MolmoModel__cache.clear()
                    
                    generation_config = GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                    )
                    
                    # Disable autocast to avoid dtype mismatch in index_add_ operation
                    # (image_flat becomes BFloat16 but x_flat remains Float32)
                    # Since we're focusing on accuracy, not latency, this is acceptable
                    outputs = self.model.generate(
                        input_ids=batch["input_ids"],
                        images=batch.get("images"),
                        image_masks=batch.get("image_masks"),
                        image_input_idx=batch.get("image_input_idx"),
                        generation_config=generation_config,
                    )
                    
                    # Move outputs to CPU immediately to avoid CUDA errors during tensor destruction
                    if isinstance(outputs, torch.Tensor):
                        outputs = outputs.cpu()
                    
                    batch_accuracy = self.compute_accuracy(
                        batch=batch,
                        predictions=outputs,
                        metric_name=metric_name,
                    )
                    
                    all_scores.extend([s["score"] for s in batch_accuracy["per_sample_scores"]])
                    all_predictions.extend(batch_accuracy["per_sample_scores"])
                    
                    # Explicitly delete tensors to free memory immediately
                    del outputs
                    if isinstance(batch, dict):
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                del v
                    del batch
                    
                    # Force cleanup after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Stop if we've collected enough samples
                    if max_samples_per_rank is not None and len(all_scores) >= max_samples_per_rank:
                        break
                        
                except RuntimeError as e:
                    error_msg = str(e)
                    if "CUDA" in error_msg or "cuda" in error_msg.lower():
                        if not self.is_distributed or self.rank == 0:
                            log.error(f"CUDA error at batch {batch_idx}: {error_msg}")
                            log.error(f"Error type: {type(e).__name__}")
                        
                        # Try to recover by clearing cache
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            except:
                                pass
                        
                        # Re-raise to let outer error handling catch it
                        raise
                    else:
                        # Re-raise non-CUDA errors
                        raise
                except Exception as e:
                    # Log and re-raise other errors
                    if not self.is_distributed or self.rank == 0:
                        log.error(f"Error at batch {batch_idx}: {e}")
                        log.error(f"Error type: {type(e).__name__}")
                    raise
        
        overall_accuracy = np.mean(all_scores) if all_scores else 0.0
        return overall_accuracy, all_predictions
    
    def _sensitivity_analysis(
        self,
        dataset_name: str,
        split: str,
        batch_size: int,
        max_new_tokens: int,
        num_samples: Optional[int] = None,
        auto_adjust_batch_size: bool = True,
    ) -> Dict[int, float]:
        """
        Stage 1: Sensitivity Analysis (same as V1)
        Compute individual block importance scores.
        """
        if not self.is_distributed or self.rank == 0:
            log.info("=" * 80)
            log.info("Stage 1: Sensitivity Analysis")
            log.info("=" * 80)
        
        metric_name = get_metric_for_dataset(dataset_name)
        
        # Import data loading modules
        from molmo.data import get_dataset_by_name
        from molmo.preprocessors.multimodal_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        from torch.utils.data import ConcatDataset
        
        # Prepare preprocessor first (needed for both single and combined splits)
        mm_preprocessor = MultiModalPreprocessor(
            tokenizer=self.tokenizer,
            crop_mode=self.model.config.crop_mode,
            max_crops=self.model.config.max_crops,
            overlap_margins=self.model.config.overlap_margins,
            image_padding_mask=bool(self.model.config.image_padding_embed),
        )
        
        formatter = DataFormatter(
            prompt_templates=self.model.config.prompt_type,
            message_format=self.model.config.message_formatting,
            system_prompt=self.model.config.system_prompt_kind,
            always_start_with_space=self.model.config.always_start_with_space,
        )
        
        preprocessor = Preprocessor(
            formater=formatter,
            mm_preprocessor=mm_preprocessor,
            for_inference=True,
        )
        
        # Support combining multiple splits (e.g., "train+validation")
        # IMPORTANT: Apply DeterministicDataset to each split BEFORE concatenating
        # because ConcatDataset doesn't have the 'get' method that DeterministicDataset expects
        if "+" in split:
            splits = [s.strip() for s in split.split("+")]
            det_datasets = []
            for s in splits:
                try:
                    ds = get_dataset_by_name(dataset_name, split=s)
                    # Apply DeterministicDataset to each split before concatenating
                    det_ds = DeterministicDataset(ds, preprocessor, seed=42)
                    det_datasets.append(det_ds)
                    if not self.is_distributed or self.rank == 0:
                        log.info(f"Loaded {dataset_name} {s} split: {len(ds)} samples")
                except Exception as e:
                    if not self.is_distributed or self.rank == 0:
                        log.warning(f"Failed to load {dataset_name} {s} split: {e}, skipping")
            if not det_datasets:
                raise ValueError(f"Failed to load any split for {dataset_name}")
            det_dataset = ConcatDataset(det_datasets) if len(det_datasets) > 1 else det_datasets[0]
            if not self.is_distributed or self.rank == 0:
                total_samples = sum(len(ds) for ds in det_datasets) if len(det_datasets) > 1 else len(det_datasets[0])
                log.info(f"Combined dataset: {total_samples} total samples from {len(det_datasets)} split(s)")
        else:
            dataset = get_dataset_by_name(dataset_name, split=split)
            det_dataset = DeterministicDataset(dataset, preprocessor, seed=42)
        
        if self.is_distributed:
            sampler = DistributedSampler(
                det_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                seed=42,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = False
        
        def create_dataloader(bs):
            return torch.utils.data.DataLoader(
                det_dataset,
                batch_size=bs,
                shuffle=shuffle,
                sampler=sampler,
                collate_fn=MMCollator(
                    max_sequence_length=1536,
                    include_metadata=True,
                    pad=True,
                    max_crops=self.model.config.max_crops
                ),
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
            )
        
        total_blocks = len(self.model.model.transformer.blocks)
        
        # Compute baseline accuracy
        if not self.is_distributed or self.rank == 0:
            log.info("Computing baseline accuracy (all blocks active)...")
        
        baseline_mask = torch.ones(total_blocks, dtype=torch.bool)
        baseline_mask_wrapper = BlockMaskWrapper(self.model.model, baseline_mask)
        baseline_mask_wrapper.apply()
        
        try:
            if auto_adjust_batch_size:
                baseline_mask_wrapper.remove()
                try:
                    optimal_batch_size = self._find_optimal_batch_size(
                        num_active_blocks=total_blocks,
                        total_blocks=total_blocks,
                        initial_batch_size=batch_size,
                        dataloader_factory=create_dataloader,
                    )
                    current_batch_size = optimal_batch_size
                finally:
                    baseline_mask_wrapper.apply()
                dataloader = create_dataloader(current_batch_size)
            else:
                current_batch_size = batch_size
                dataloader = create_dataloader(current_batch_size)
            
            baseline_accuracy, _ = self._compute_accuracy_on_subset(
                dataloader,
                max_samples=num_samples,
                max_new_tokens=max_new_tokens,
                metric_name=metric_name,
                mask_applied=True,
            )
        finally:
            baseline_mask_wrapper.remove()
        
        if not self.is_distributed or self.rank == 0:
            log.info(f"Baseline accuracy (all {total_blocks} blocks): {baseline_accuracy:.4f}")
        
        # Ablate each layer individually
        # Now include all blocks (0-15), not just 1-14
        layers_to_explore = list(range(total_blocks))  # All layers 0 to 15
        if not self.is_distributed or self.rank == 0:
            log.info(f"Exploring importance of all layers {layers_to_explore} (including first and last)")
        
        # Load existing results for checkpoint resumption
        importance_scores = {}
        results_dir = Path(self.output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing individual results
        for layer_idx in layers_to_explore:
            result_file = results_dir / f"sensitivity_block_{layer_idx}.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                        importance_scores[layer_idx] = result_data.get("importance_score", None)
                        if importance_scores[layer_idx] is not None:
                            if not self.is_distributed or self.rank == 0:
                                log.info(f"✓ Loaded existing result for block {layer_idx}: "
                                        f"importance_score={importance_scores[layer_idx]:.4f}")
                except Exception as e:
                    if not self.is_distributed or self.rank == 0:
                        log.warning(f"Failed to load result for block {layer_idx}: {e}, will recompute")
                    importance_scores[layer_idx] = None
        
        # Filter out already completed layers
        remaining_layers = [idx for idx in layers_to_explore if idx not in importance_scores or importance_scores[idx] is None]
        
        if not self.is_distributed or self.rank == 0:
            if remaining_layers:
                log.info(f"Resuming: {len(remaining_layers)} blocks remaining, {len(layers_to_explore) - len(remaining_layers)} already completed")
            else:
                log.info(f"All {len(layers_to_explore)} blocks already completed, loading from checkpoints")
                return importance_scores
        
        mask_wrapper = None
        
        try:
            for layer_idx in remaining_layers:
                if not self.is_distributed or self.rank == 0:
                    log.info(f"Ablating layer {layer_idx}/{total_blocks-1}")
                
                block_mask = torch.ones(total_blocks, dtype=torch.bool)
                block_mask[layer_idx] = False
                # No longer force block 0 and last block to be active
                # This allows us to test removing first and last blocks
                
                if mask_wrapper is not None:
                    mask_wrapper.remove()
                
                mask_wrapper = BlockMaskWrapper(self.model.model, block_mask)
                mask_wrapper.apply()
                
                if auto_adjust_batch_size:
                    mask_wrapper.remove()
                    try:
                        optimal_batch_size = self._find_optimal_batch_size(
                            num_active_blocks=total_blocks - 1,
                            total_blocks=total_blocks,
                            initial_batch_size=batch_size,
                            dataloader_factory=create_dataloader,
                        )
                        current_batch_size = optimal_batch_size
                    finally:
                        mask_wrapper.apply()
                    dataloader = create_dataloader(current_batch_size)
                else:
                    current_batch_size = batch_size
                    dataloader = create_dataloader(current_batch_size)
                
                ablated_accuracy, _ = self._compute_accuracy_on_subset(
                    dataloader,
                    max_samples=num_samples,
                    max_new_tokens=max_new_tokens,
                    metric_name=metric_name,
                    mask_applied=True,
                )
                
                importance_score = baseline_accuracy - ablated_accuracy
                importance_scores[layer_idx] = importance_score
                
                # Save individual result immediately for checkpoint resumption
                if not self.is_distributed or self.rank == 0:
                    result_file = results_dir / f"sensitivity_block_{layer_idx}.json"
                    result_data = {
                        "block_idx": layer_idx,
                        "baseline_accuracy": float(baseline_accuracy),
                        "ablated_accuracy": float(ablated_accuracy),
                        "importance_score": float(importance_score),
                        "num_samples": num_samples,
                        "dataset_name": dataset_name,
                        "split": split,
                    }
                    with open(result_file, 'w') as f:
                        json.dump(result_data, f, indent=2)
                    
                    log.info(f"Layer {layer_idx}: Accuracy={ablated_accuracy:.4f}, "
                            f"ΔAcc={importance_score:.4f} (saved to {result_file})")
        
        finally:
            if mask_wrapper is not None:
                mask_wrapper.remove()
        
        return importance_scores
    
    def _beam_search_pruning(
        self,
        baseline_accuracy: float,
        dataset_name: str,
        split: str,
        batch_size: int,
        max_new_tokens: int,
        num_samples: Optional[int] = None,
        beam_width: int = 3,
        max_blocks_to_remove: int = 4,
        min_blocks: int = 12,
        auto_adjust_batch_size: bool = True,
    ) -> List[Dict]:
        """
        Stage 2: Beam Search Pruning
        
        Use beam search to explore block combinations:
        - Step 1: Remove 1 block, test all possibilities (all 16 blocks), keep top-K
        - Step 2: For each top-K, remove another block, test all possibilities, keep top-K
        - Continue until max_blocks_to_remove blocks are removed
        
        Note: All 16 blocks (0-15) can be removed, including the first and last blocks.
        This allows us to fully explore the importance of all blocks.
        """
        if not self.is_distributed or self.rank == 0:
            log.info("=" * 80)
            log.info("Stage 2: Beam Search Pruning")
            log.info("=" * 80)
            log.info(f"Beam width: {beam_width}, Max blocks to remove: {max_blocks_to_remove}")
        
        metric_name = get_metric_for_dataset(dataset_name)
        total_blocks = len(self.model.model.transformer.blocks)
        
        # Import data loading modules
        from molmo.data import get_dataset_by_name
        from molmo.preprocessors.multimodal_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        from torch.utils.data import ConcatDataset
        
        # Prepare preprocessor first (needed for both single and combined splits)
        mm_preprocessor = MultiModalPreprocessor(
            tokenizer=self.tokenizer,
            crop_mode=self.model.config.crop_mode,
            max_crops=self.model.config.max_crops,
            overlap_margins=self.model.config.overlap_margins,
            image_padding_mask=bool(self.model.config.image_padding_embed),
        )
        
        formatter = DataFormatter(
            prompt_templates=self.model.config.prompt_type,
            message_format=self.model.config.message_formatting,
            system_prompt=self.model.config.system_prompt_kind,
            always_start_with_space=self.model.config.always_start_with_space,
        )
        
        preprocessor = Preprocessor(
            formater=formatter,
            mm_preprocessor=mm_preprocessor,
            for_inference=True,
        )
        
        # Support combining multiple splits (e.g., "train+validation")
        # IMPORTANT: Apply DeterministicDataset to each split BEFORE concatenating
        # because ConcatDataset doesn't have the 'get' method that DeterministicDataset expects
        if "+" in split:
            splits = [s.strip() for s in split.split("+")]
            det_datasets = []
            for s in splits:
                try:
                    ds = get_dataset_by_name(dataset_name, split=s)
                    # Apply DeterministicDataset to each split before concatenating
                    det_ds = DeterministicDataset(ds, preprocessor, seed=42)
                    det_datasets.append(det_ds)
                    if not self.is_distributed or self.rank == 0:
                        log.info(f"Loaded {dataset_name} {s} split: {len(ds)} samples")
                except Exception as e:
                    if not self.is_distributed or self.rank == 0:
                        log.warning(f"Failed to load {dataset_name} {s} split: {e}, skipping")
            if not det_datasets:
                raise ValueError(f"Failed to load any split for {dataset_name}")
            det_dataset = ConcatDataset(det_datasets) if len(det_datasets) > 1 else det_datasets[0]
            if not self.is_distributed or self.rank == 0:
                total_samples = sum(len(ds) for ds in det_datasets) if len(det_datasets) > 1 else len(det_datasets[0])
                log.info(f"Combined dataset: {total_samples} total samples from {len(det_datasets)} split(s)")
        else:
            dataset = get_dataset_by_name(dataset_name, split=split)
            det_dataset = DeterministicDataset(dataset, preprocessor, seed=42)
        
        if self.is_distributed:
            sampler = DistributedSampler(
                det_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                seed=42,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = False
        
        def create_dataloader(bs):
            return torch.utils.data.DataLoader(
                det_dataset,
                batch_size=bs,
                shuffle=shuffle,
                sampler=sampler,
                collate_fn=MMCollator(
                    max_sequence_length=1536,
                    include_metadata=True,
                    pad=True,
                    max_crops=self.model.config.max_crops
                ),
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
            )
        
        # Initial state: all blocks active
        initial_state = BeamState(
            active_blocks=list(range(total_blocks)),
            removed_blocks=[],
            accuracy=baseline_accuracy,
            accuracy_drop=0.0,
        )
        
        # Beam: list of BeamState objects
        beam = [initial_state]
        all_results = []
        
        # Track evaluated states to avoid duplicates
        evaluated_states = set()
        evaluated_states.add(tuple(sorted(initial_state.active_blocks)))
        
        # Explorable blocks: all blocks (0-15) can be removed
        # No longer force block 0 and last block to be always active
        explorable_blocks = list(range(total_blocks))
        
        mask_wrapper = None
        
        try:
            for step in range(max_blocks_to_remove):
                if not self.is_distributed or self.rank == 0:
                    log.info("=" * 80)
                    log.info(f"Beam Search Step {step + 1}: Removing {step + 1} block(s)")
                    log.info("=" * 80)
                
                # Generate candidates from current beam
                candidates = []
                seen_in_this_step = set()  # Track candidates in current step to avoid duplicates
                
                for state in beam:
                    # For this state, try removing each remaining explorable block
                    remaining_blocks = [b for b in explorable_blocks if b not in state.removed_blocks]
                    
                    for block_to_remove in remaining_blocks:
                        # Create new state
                        new_active = [b for b in state.active_blocks if b != block_to_remove]
                        new_removed = state.removed_blocks + [block_to_remove]
                        
                        # No longer force block 0 and last block to be always active
                        # Allow exploring all possible combinations
                        new_active = sorted(new_active)
                        
                        # Ensure we don't remove all blocks (keep at least min_blocks)
                        if len(new_active) < min_blocks:
                            continue
                        
                        # Skip if already evaluated in previous steps
                        state_key = tuple(sorted(new_active))
                        if state_key in evaluated_states:
                            continue
                        
                        # Skip if already seen in this step (duplicate from different paths)
                        # This handles the case where removing block 4 then 8 vs removing block 8 then 4
                        # results in the same active_blocks configuration
                        if state_key in seen_in_this_step:
                            if not self.is_distributed or self.rank == 0:
                                log.debug(f"Skipping duplicate candidate: active_blocks={new_active} "
                                        f"(already in candidates for this step)")
                            continue
                        
                        seen_in_this_step.add(state_key)
                        
                        # Check if we've already evaluated this state
                        candidate_state = BeamState(
                            active_blocks=new_active,
                            removed_blocks=sorted(new_removed),
                        )
                        
                        candidates.append((candidate_state, block_to_remove))
                
                if not self.is_distributed or self.rank == 0:
                    log.info(f"Evaluating {len(candidates)} candidate configurations...")
                
                # Evaluate all candidates
                evaluated_candidates = []
                results_dir = Path(self.output_dir)
                results_dir.mkdir(parents=True, exist_ok=True)
                
                for candidate_state, block_to_remove in candidates:
                    num_active = len(candidate_state.active_blocks)
                    state_key = tuple(sorted(candidate_state.active_blocks))
                    
                    # Check if this configuration already exists
                    removed_str = '-'.join(map(str, sorted(candidate_state.removed_blocks)))
                    config_filename = f"beam_search_step{step+1}_blocks{num_active}_removed{removed_str}.json"
                    config_file = results_dir / config_filename
                    
                    if config_file.exists():
                        try:
                            with open(config_file, 'r') as f:
                                existing_result = json.load(f)
                            candidate_state.accuracy = existing_result.get("accuracy", 0.0)
                            candidate_state.accuracy_drop = existing_result.get("accuracy_drop", 0.0)
                            evaluated_states.add(state_key)
                            evaluated_candidates.append(candidate_state)
                            
                            if not self.is_distributed or self.rank == 0:
                                log.info(f"✓ Loaded existing result for {sorted(candidate_state.removed_blocks)}: "
                                        f"Accuracy={candidate_state.accuracy:.4f}, Drop={candidate_state.accuracy_drop:.4f}")
                            continue
                        except Exception as e:
                            if not self.is_distributed or self.rank == 0:
                                log.warning(f"Failed to load existing result for {sorted(candidate_state.removed_blocks)}: {e}, will recompute")
                    
                    try:
                        # Clean up previous mask wrapper and clear CUDA cache
                        if mask_wrapper is not None:
                            mask_wrapper.remove()
                            mask_wrapper = None
                        
                        # Clear CUDA cache before creating new mask
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        # Create mask
                        block_mask = torch.zeros(total_blocks, dtype=torch.bool, device=self.device)
                        for idx in candidate_state.active_blocks:
                            block_mask[idx] = True
                        
                        # Apply mask
                        mask_wrapper = BlockMaskWrapper(self.model.model, block_mask)
                        mask_wrapper.apply()
                        
                        # Small delay to ensure mask is properly applied
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        # Find optimal batch size
                        if auto_adjust_batch_size:
                            mask_wrapper.remove()
                            try:
                                optimal_batch_size = self._find_optimal_batch_size(
                                    num_active_blocks=num_active,
                                    total_blocks=total_blocks,
                                    initial_batch_size=batch_size,
                                    dataloader_factory=create_dataloader,
                                )
                                current_batch_size = optimal_batch_size
                            finally:
                                mask_wrapper.apply()
                            dataloader = create_dataloader(current_batch_size)
                        else:
                            current_batch_size = batch_size
                            dataloader = create_dataloader(current_batch_size)
                        
                        # Compute accuracy
                        accuracy, per_sample_scores = self._compute_accuracy_on_subset(
                            dataloader,
                            max_samples=num_samples,
                            max_new_tokens=max_new_tokens,
                            metric_name=metric_name,
                            mask_applied=True,
                        )
                        
                        accuracy_drop = baseline_accuracy - accuracy
                        candidate_state.accuracy = accuracy
                        candidate_state.accuracy_drop = accuracy_drop
                        
                        # Mark as evaluated
                        evaluated_states.add(state_key)
                        
                        evaluated_candidates.append(candidate_state)
                        
                        # Save result immediately for checkpoint resumption
                        result_entry = {
                            "step": step + 1,
                            "num_active_blocks": num_active,
                            "num_total_blocks": total_blocks,
                            "active_block_indices": sorted(candidate_state.active_blocks),
                            "removed_block_indices": sorted(candidate_state.removed_blocks),
                            "accuracy": float(accuracy),
                            "accuracy_drop": float(accuracy_drop),
                            "num_samples": len(per_sample_scores),
                            "std": float(np.std([s["score"] for s in per_sample_scores])) if per_sample_scores else 0.0,
                            "baseline_accuracy": float(baseline_accuracy),
                        }
                        all_results.append(result_entry)
                        
                        # Save individual configuration result
                        if not self.is_distributed or self.rank == 0:
                            with open(config_file, 'w') as f:
                                json.dump(result_entry, f, indent=2)
                            
                            log.info(f"  Removed {sorted(candidate_state.removed_blocks)}: "
                                    f"Accuracy={accuracy:.4f}, Drop={accuracy_drop:.4f} (saved to {config_filename})")
                        
                        # Clean up mask wrapper immediately after each candidate
                        if mask_wrapper is not None:
                            try:
                                mask_wrapper.remove()
                            except Exception as cleanup_error:
                                if not self.is_distributed or self.rank == 0:
                                    log.warning(f"Error removing mask wrapper: {cleanup_error}")
                            mask_wrapper = None
                        
                        # Clean up after each candidate to prevent memory buildup
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        # Force garbage collection periodically to free Python objects
                        import gc
                        gc.collect()
                        
                        # Small delay to allow CUDA operations to complete
                        time.sleep(0.05)
                    
                    except RuntimeError as e:
                        # Handle CUDA errors specifically
                        error_msg = str(e)
                        is_cuda_error = "CUDA" in error_msg or "cuda" in error_msg.lower() or "unrecognized error code" in error_msg
                        
                        if not self.is_distributed or self.rank == 0:
                            log.error(f"Error evaluating candidate {sorted(candidate_state.removed_blocks)}: {error_msg}")
                            log.error(f"Error type: {type(e).__name__}")
                            if is_cuda_error:
                                log.error("This is a CUDA error - attempting recovery...")
                        
                        # Clean up on error - more aggressive cleanup
                        if mask_wrapper is not None:
                            try:
                                mask_wrapper.remove()
                            except Exception as cleanup_error:
                                if not self.is_distributed or self.rank == 0:
                                    log.warning(f"Error removing mask wrapper: {cleanup_error}")
                            mask_wrapper = None
                        
                        # Aggressive memory cleanup on error
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                # Reset peak memory stats
                                torch.cuda.reset_peak_memory_stats()
                            except Exception as cuda_cleanup_error:
                                if not self.is_distributed or self.rank == 0:
                                    log.warning(f"Error during CUDA cleanup: {cuda_cleanup_error}")
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                        
                        # For CUDA errors, add a longer delay to allow recovery
                        if is_cuda_error:
                            time.sleep(0.5)
                        
                        # Skip this candidate
                        continue
                    except Exception as e:
                        # Handle other errors
                        error_msg = str(e)
                        if not self.is_distributed or self.rank == 0:
                            log.error(f"Error evaluating candidate {sorted(candidate_state.removed_blocks)}: {error_msg}")
                            log.error(f"Error type: {type(e).__name__}")
                        
                        # Clean up on error
                        if mask_wrapper is not None:
                            try:
                                mask_wrapper.remove()
                            except:
                                pass
                            mask_wrapper = None
                        
                        # Memory cleanup
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            except:
                                pass
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                        
                        # Skip this candidate
                        continue
                
                # Remove duplicates before selecting top-K
                # Use a dict to keep only the best (lowest accuracy_drop) for each unique active_blocks
                unique_candidates = {}
                for candidate in evaluated_candidates:
                    state_key = tuple(sorted(candidate.active_blocks))
                    if state_key not in unique_candidates:
                        unique_candidates[state_key] = candidate
                    else:
                        # Keep the one with lower accuracy drop (better)
                        if candidate.accuracy_drop < unique_candidates[state_key].accuracy_drop:
                            unique_candidates[state_key] = candidate
                
                # Convert back to list and sort
                unique_candidates_list = list(unique_candidates.values())
                unique_candidates_list.sort(key=lambda s: s.accuracy_drop)  # Lower drop = better
                
                # Select top-K candidates (lowest accuracy drop = best)
                beam = unique_candidates_list[:beam_width]
                
                if not self.is_distributed or self.rank == 0:
                    num_duplicates = len(evaluated_candidates) - len(unique_candidates_list)
                    if num_duplicates > 0:
                        log.info(f"Removed {num_duplicates} duplicate configuration(s) before selecting top-{beam_width}")
                    log.info(f"Top {beam_width} candidates for next step:")
                    for i, state in enumerate(beam):
                        log.info(f"  {i+1}. Removed {sorted(state.removed_blocks)}: "
                                f"Drop={state.accuracy_drop:.4f}")
                
                # Clean up mask wrapper between steps to prevent memory buildup
                if mask_wrapper is not None:
                    mask_wrapper.remove()
                    mask_wrapper = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        finally:
            # Final cleanup
            if mask_wrapper is not None:
                try:
                    mask_wrapper.remove()
                except:
                    pass
                mask_wrapper = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        return all_results
    
    def run(
        self,
        dataset_name: str = "coco_2014_vqa",
        split: str = "train",  # Default to train set for sensitivity analysis
        batch_size: int = 16,  # Lower default to avoid OOM
        max_new_tokens: int = 16,
        num_samples: Optional[int] = None,
        beam_width: int = 3,
        max_blocks_to_remove: int = 4,
        skip_sensitivity: bool = False,
        importance_scores_file: Optional[str] = None,
        auto_adjust_batch_size: bool = True,
    ):
        """
        Run Exp3 Sensitivity Analysis V2 with beam search.
        
        Args:
            dataset_name: Dataset name
            split: Dataset split
            batch_size: Batch size per GPU (will be optimized for throughput)
            max_new_tokens: Maximum tokens to generate
            num_samples: Total number of samples for evaluation (None = use all samples)
            beam_width: Number of top candidates to keep at each step
            max_blocks_to_remove: Maximum number of blocks to remove (default: 4)
            skip_sensitivity: Skip sensitivity analysis and load from file
            importance_scores_file: Path to saved importance scores
            auto_adjust_batch_size: Automatically optimize batch size
        """
        import json
        
        # Stage 1: Sensitivity Analysis
        if skip_sensitivity and importance_scores_file:
            log.info(f"Loading importance scores from {importance_scores_file}")
            with open(importance_scores_file, 'r') as f:
                saved_data = json.load(f)
                importance_scores = {int(k): float(v) for k, v in saved_data.items()}
        else:
            importance_scores = self._sensitivity_analysis(
                dataset_name=dataset_name,
                split=split,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                num_samples=num_samples,
                auto_adjust_batch_size=auto_adjust_batch_size,
            )
            
            if not self.is_distributed or self.rank == 0:
                importance_file = os.path.join(self.output_dir, "layer_importance_scores.json")
                with open(importance_file, 'w') as f:
                    json.dump(importance_scores, f, indent=2)
                log.info(f"Saved importance scores to {importance_file}")
        
        # Compute baseline accuracy for beam search
        # (We need baseline for computing accuracy drops)
        # We'll compute it in beam search, but for now we can use a placeholder
        # Actually, we should compute it in sensitivity analysis and pass it
        
        # Stage 2: Beam Search Pruning
        # We need baseline accuracy - let's compute it here
        metric_name = get_metric_for_dataset(dataset_name)
        from molmo.data import get_dataset_by_name
        from molmo.preprocessors.multimodal_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        from torch.utils.data import ConcatDataset
        
        # Prepare preprocessor first (needed for both single and combined splits)
        mm_preprocessor = MultiModalPreprocessor(
            tokenizer=self.tokenizer,
            crop_mode=self.model.config.crop_mode,
            max_crops=self.model.config.max_crops,
            overlap_margins=self.model.config.overlap_margins,
            image_padding_mask=bool(self.model.config.image_padding_embed),
        )
        
        formatter = DataFormatter(
            prompt_templates=self.model.config.prompt_type,
            message_format=self.model.config.message_formatting,
            system_prompt=self.model.config.system_prompt_kind,
            always_start_with_space=self.model.config.always_start_with_space,
        )
        
        preprocessor = Preprocessor(
            formater=formatter,
            mm_preprocessor=mm_preprocessor,
            for_inference=True,
        )
        
        # Support combining multiple splits (e.g., "train+validation")
        # IMPORTANT: Apply DeterministicDataset to each split BEFORE concatenating
        # because ConcatDataset doesn't have the 'get' method that DeterministicDataset expects
        if "+" in split:
            splits = [s.strip() for s in split.split("+")]
            det_datasets = []
            for s in splits:
                try:
                    ds = get_dataset_by_name(dataset_name, split=s)
                    # Apply DeterministicDataset to each split before concatenating
                    det_ds = DeterministicDataset(ds, preprocessor, seed=42)
                    det_datasets.append(det_ds)
                    if not self.is_distributed or self.rank == 0:
                        log.info(f"Loaded {dataset_name} {s} split: {len(ds)} samples")
                except Exception as e:
                    if not self.is_distributed or self.rank == 0:
                        log.warning(f"Failed to load {dataset_name} {s} split: {e}, skipping")
            if not det_datasets:
                raise ValueError(f"Failed to load any split for {dataset_name}")
            det_dataset = ConcatDataset(det_datasets) if len(det_datasets) > 1 else det_datasets[0]
            if not self.is_distributed or self.rank == 0:
                total_samples = sum(len(ds) for ds in det_datasets) if len(det_datasets) > 1 else len(det_datasets[0])
                log.info(f"Combined dataset: {total_samples} total samples from {len(det_datasets)} split(s)")
        else:
            dataset = get_dataset_by_name(dataset_name, split=split)
            det_dataset = DeterministicDataset(dataset, preprocessor, seed=42)
        
        if self.is_distributed:
            sampler = DistributedSampler(
                det_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                seed=42,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = False
        
        def create_dataloader(bs):
            return torch.utils.data.DataLoader(
                det_dataset,
                batch_size=bs,
                shuffle=shuffle,
                sampler=sampler,
                collate_fn=MMCollator(
                    max_sequence_length=1536,
                    include_metadata=True,
                    pad=True,
                    max_crops=self.model.config.max_crops
                ),
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
            )
        
        total_blocks = len(self.model.model.transformer.blocks)
        
        # Compute baseline
        if not self.is_distributed or self.rank == 0:
            log.info("Computing baseline accuracy for beam search...")
        
        baseline_mask = torch.ones(total_blocks, dtype=torch.bool)
        baseline_mask_wrapper = BlockMaskWrapper(self.model.model, baseline_mask)
        baseline_mask_wrapper.apply()
        
        try:
            if auto_adjust_batch_size:
                baseline_mask_wrapper.remove()
                try:
                    optimal_batch_size = self._find_optimal_batch_size(
                        num_active_blocks=total_blocks,
                        total_blocks=total_blocks,
                        initial_batch_size=batch_size,
                        dataloader_factory=create_dataloader,
                    )
                    current_batch_size = optimal_batch_size
                finally:
                    baseline_mask_wrapper.apply()
                dataloader = create_dataloader(current_batch_size)
            else:
                current_batch_size = batch_size
                dataloader = create_dataloader(current_batch_size)
            
            baseline_accuracy, _ = self._compute_accuracy_on_subset(
                dataloader,
                max_samples=num_samples,
                max_new_tokens=max_new_tokens,
                metric_name=metric_name,
                mask_applied=True,
            )
        finally:
            baseline_mask_wrapper.remove()
        
        if not self.is_distributed or self.rank == 0:
            log.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        # Run beam search (only if max_blocks_to_remove > 0)
        if max_blocks_to_remove > 0:
            results_data = self._beam_search_pruning(
                baseline_accuracy=baseline_accuracy,
                dataset_name=dataset_name,
                split=split,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                num_samples=num_samples,
                beam_width=beam_width,
                max_blocks_to_remove=max_blocks_to_remove,
                auto_adjust_batch_size=auto_adjust_batch_size,
            )
        else:
            # Skip beam search, only sensitivity analysis
            if not self.is_distributed or self.rank == 0:
                log.info("Skipping beam search (max_blocks_to_remove=0), only doing sensitivity analysis")
            results_data = []
        
        # Gather and merge results if distributed
        if self.is_distributed:
            if self.rank == 0:
                gathered_results = [None] * self.world_size
                dist.gather_object(results_data, gathered_results, dst=0)
                
                # Merge results (group by configuration)
                merged_results = []
                seen_configs = set()
                
                for rank_results in gathered_results:
                    if rank_results is None:
                        continue
                    for result in rank_results:
                        config_key = tuple(sorted(result["active_block_indices"]))
                        if config_key not in seen_configs:
                            seen_configs.add(config_key)
                            merged_results.append(result)
                
                results_data = merged_results
            else:
                dist.gather_object(results_data, None, dst=0)
                dist.destroy_process_group()
                return
        
        # Save final results
        final_results = {
            "summary": results_data,
            "importance_scores": importance_scores,
            "baseline_accuracy": float(baseline_accuracy),
            "config": {
                "dataset_name": dataset_name,
                "split": split,
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "num_samples": num_samples,
                "beam_width": beam_width,
                "max_blocks_to_remove": max_blocks_to_remove,
                "total_blocks": total_blocks,
                "world_size": self.world_size,
            }
        }
        
        self.save_results(final_results, "exp3_accuracy_sensitivity_v2_results.json")
        if not self.is_distributed or self.rank == 0:
            log.info(f"Results saved. Total configurations tested: {len(results_data)}")
        
        if self.is_distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Exp3 Sensitivity V2: Beam Search Block Combination")
    parser.add_argument("--model_path", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results/profiling/exp3_accuracy_sensitivity_v2")
    parser.add_argument("--dataset_name", type=str, default="coco_2014_vqa")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use (default: train). Can combine splits with '+', e.g., 'train+validation'")
    parser.add_argument("--batch_size", type=int, default=16, help="Initial batch size (will be optimized, default: 16, lower if OOM)")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    def parse_num_samples(value):
        if value is None or value == "None" or value == "":
            return None
        return int(value)
    
    parser.add_argument("--num_samples", type=parse_num_samples, default=None,
                       help="Number of samples to evaluate (None = use all samples, default: None)")
    parser.add_argument("--beam_width", type=int, default=3, help="Number of top candidates to keep")
    parser.add_argument("--max_blocks_to_remove", type=int, default=4, help="Maximum blocks to remove")
    parser.add_argument("--skip_sensitivity", action="store_true")
    parser.add_argument("--importance_scores_file", type=str, default=None)
    parser.add_argument("--auto_adjust_batch_size", action="store_true", default=True)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = Exp3SensitivityExperimentV2(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    experiment.run(
        dataset_name=args.dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        beam_width=args.beam_width,
        max_blocks_to_remove=args.max_blocks_to_remove,
        skip_sensitivity=args.skip_sensitivity,
        importance_scores_file=args.importance_scores_file,
        auto_adjust_batch_size=args.auto_adjust_batch_size,
    )


if __name__ == "__main__":
    main()

