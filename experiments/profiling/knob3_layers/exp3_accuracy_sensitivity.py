"""
Exp3 Accuracy Sensitivity Analysis: Layer Importance-based Pruning
Multi-GPU, measure accuracy with importance-based layer removal.

Two-stage approach:
1. Sensitivity Analysis: Ablate each layer individually to compute importance scores
2. Importance-based Pruning: Remove layers from least important to most important
"""

import argparse
import logging
import sys
import os
import time
from typing import Dict, List, Any, Optional, Tuple

# Set TOKENIZERS_PARALLELISM to avoid warnings when using DataLoader with num_workers > 0
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from torch.utils.data import DistributedSampler

sys.path.append(os.getcwd())
from experiments.base_experiment import BaseExperiment
from molmo.torch_util import get_world_size, get_global_rank, get_local_rank

# Import BlockMaskWrapper from the original exp3 in the same directory
sys.path.append(os.path.join(os.path.dirname(__file__)))
from exp_transformer_blocks_mask import BlockMaskWrapper

log = logging.getLogger(__name__)


class Exp3SensitivityExperiment(BaseExperiment):
    """
    Exp3 Sensitivity: Layer importance analysis and importance-based pruning.
    
    Stage 1: Sensitivity Analysis
    - Ablate each layer individually (skip that layer)
    - Measure accuracy drop on subset (5k samples)
    - Compute importance scores: ΔAcc(l) = Acc_full - Acc_without_layer_l
    
    Stage 2: Importance-based Pruning
    - Sort layers by importance (least important first)
    - Remove layers from least to most important
    - Test different numbers of active layers (starting from 8)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        output_dir: str = "./results",
        num_warmup: int = 3,
        hf_cache_dir: Optional[str] = None,
    ):
        # Auto-detect distributed environment (set by torchrun)
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
            torch.cuda.set_device(local_rank)
            log.info(f"Rank {self.rank} (local_rank {local_rank}) using device {device}")
        
        super().__init__(
            model_path=model_path,
            device=device,
            output_dir=output_dir,
            num_warmup=num_warmup,
            hf_cache_dir=hf_cache_dir,
        )
    
    def _estimate_batch_size_for_num_blocks(self, num_active_blocks: int, total_blocks: int, base_batch_size: int) -> int:
        """
        Estimate appropriate batch size for a given number of active blocks.
        
        Memory usage scales with:
        - Transformer layers: num_active_blocks determines how many layers are active
        - Smaller num_active_blocks = fewer active layers = less memory = larger batch size possible
        
        Heuristic: batch_size scales inversely with num_active_blocks (but less aggressively than max_crops)
        """
        # Number of active blocks has less impact on memory than max_crops
        # Use conservative scaling
        ratio = num_active_blocks / total_blocks if total_blocks > 0 else 1.0
        
        if ratio <= 0.25:
            scale_factor = 1.0  # Very few blocks can use full batch size
        elif ratio <= 0.5:
            scale_factor = 0.9
        elif ratio <= 0.75:
            scale_factor = 0.8
        else:
            scale_factor = 0.7  # Most blocks active needs smaller batch size
        
        estimated_batch_size = max(1, int(base_batch_size * scale_factor))
        return estimated_batch_size
    
    def _find_optimal_batch_size(
        self, 
        num_active_blocks: int,
        total_blocks: int,
        initial_batch_size: int,
        dataloader_factory,
        max_attempts: int = 5,
    ) -> int:
        """
        Dynamically find the optimal batch size that doesn't cause OOM or CUDA errors.
        
        Uses binary search: start with estimated size, reduce if error, increase if safe.
        
        Args:
            num_active_blocks: Current number of active blocks
            total_blocks: Total number of blocks
            initial_batch_size: Starting batch size to try (base value)
            dataloader_factory: Function that creates dataloader given batch_size
            max_attempts: Maximum number of attempts to find working batch size
            
        Returns:
            Optimal batch size that works without OOM or CUDA errors
        """
        import torch
        
        ratio = num_active_blocks / total_blocks if total_blocks > 0 else 1.0
        
        # For few active blocks, allow exceeding initial_batch_size to find maximum
        if ratio <= 0.25:
            max_allowed_batch_size = initial_batch_size * 2
        elif ratio <= 0.5:
            max_allowed_batch_size = int(initial_batch_size * 1.5)
        else:
            max_allowed_batch_size = initial_batch_size
        
        # Start with estimated batch size
        estimated_batch_size = self._estimate_batch_size_for_num_blocks(num_active_blocks, total_blocks, initial_batch_size)
        current_batch_size = min(estimated_batch_size, max_allowed_batch_size)
        
        log.info(f"Finding optimal batch size for {num_active_blocks}/{total_blocks} active blocks "
                f"(estimated: {estimated_batch_size}, starting with: {current_batch_size}, max allowed: {max_allowed_batch_size})...")
        
        # Track what we've tried
        min_working = None
        max_failing = 0
        
        # Check if mask is currently applied (for temporarily removing during batch size test)
        # We check by seeing if model.forward has been replaced
        model_has_mask = False
        original_forward = None
        if hasattr(self.model.model, 'forward') and hasattr(self.model.model, '_MolmoModel__cache'):
            # Check if forward has been monkey-patched (indicates mask is applied)
            # This is a heuristic: if forward method is not the original, mask might be applied
            # We'll be more conservative: only test forward pass, not generation, when mask might be applied
            # Actually, better approach: test without generation when mask is applied
            pass
        
        for attempt in range(max_attempts):
            try:
                # Clear cache before each attempt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Create dataloader with current batch size
                dataloader = dataloader_factory(current_batch_size)
                
                # Try to get one batch and move to device
                test_batch = next(iter(dataloader))
                test_batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                             for k, v in test_batch.items()}
                
                # Try a forward pass AND a short generation to check memory
                with torch.inference_mode():
                    with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        # First do a forward pass
                        _ = self.model(
                            input_ids=test_batch["input_ids"],
                            images=test_batch.get("images"),
                            image_masks=test_batch.get("image_masks"),
                            image_input_idx=test_batch.get("image_input_idx"),
                        )
                        
                        # Only test generation if we're not in a masked state
                        # (generation with mask can cause attention bias size mismatches)
                        # For batch size testing, forward pass is usually sufficient
                        # But we can try a simpler generation test without mask complications
                        try:
                            from transformers import GenerationConfig
                            eos_token_id = self.tokenizer.eos_token_id
                            if eos_token_id is None:
                                eos_token_id = getattr(self.model.config, 'eos_token_id', None)
                            
                            pad_token_id = self.tokenizer.pad_token_id
                            if pad_token_id is None:
                                pad_token_id = getattr(self.model.config, 'pad_token_id', None)
                            
                            test_gen_config = GenerationConfig(
                                max_new_tokens=1,  # Use minimal generation to test memory
                                do_sample=False,
                                use_cache=True,
                                eos_token_id=eos_token_id,
                                pad_token_id=pad_token_id,
                            )
                            
                            # Try generating to check memory (with minimal tokens to avoid mask issues)
                            _ = self.model.generate(
                                input_ids=test_batch["input_ids"],
                                images=test_batch.get("images"),
                                image_masks=test_batch.get("image_masks"),
                                image_input_idx=test_batch.get("image_input_idx"),
                                generation_config=test_gen_config,
                            )
                        except RuntimeError as gen_error:
                            # If generation fails due to mask-related issues (size mismatch),
                            # we can still use the batch size if forward pass worked
                            error_str = str(gen_error).lower()
                            if "size" in error_str and "tensor" in error_str and "must match" in error_str:
                                # This is likely a mask-related size mismatch, not an OOM
                                # Forward pass worked, so this batch size should be fine
                                log.info(f"  Generation test skipped due to mask-related size mismatch, "
                                        f"but forward pass succeeded - batch size {current_batch_size} should work")
                            else:
                                # Re-raise other generation errors
                                raise
                
                # If we get here, it worked!
                min_working = current_batch_size
                log.info(f"✓ Batch size {current_batch_size} works for {num_active_blocks}/{total_blocks} active blocks")
                
                # Try to increase if we're below max_allowed and haven't hit limit
                if current_batch_size < max_allowed_batch_size and attempt < max_attempts - 1:
                    next_try = min(max_allowed_batch_size, int(current_batch_size * 1.5))
                    if next_try > current_batch_size:
                        current_batch_size = next_try
                        log.info(f"  Trying larger batch size: {current_batch_size}...")
                        continue
                
                # Found a working size, return it
                return min_working
                
            except RuntimeError as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "oom" in error_str or "invalid configuration" in error_str:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    max_failing = current_batch_size
                    old_batch_size = current_batch_size
                    
                    # Binary search: try midpoint between current and last working (or 1)
                    if min_working is not None:
                        current_batch_size = max(1, (min_working + max_failing) // 2)
                    else:
                        current_batch_size = max(1, current_batch_size // 2)
                    
                    error_type = "OOM" if "out of memory" in error_str or "oom" in error_str else "CUDA config error"
                    log.warning(f"✗ Batch size {old_batch_size} caused {error_type} for {num_active_blocks}/{total_blocks} active blocks, "
                              f"trying {current_batch_size}...")
                    
                    # If we've narrowed down to the same size, stop
                    if current_batch_size == old_batch_size:
                        break
                else:
                    # Other error, re-raise
                    raise
        
        # Return the last working size, or the smallest we tried
        if min_working is not None:
            log.info(f"Using batch size {min_working} for {num_active_blocks}/{total_blocks} active blocks")
            return min_working
        else:
            log.warning(f"Could not find working batch size after {max_attempts} attempts, "
                       f"using {current_batch_size} (may cause OOM)")
            return max(1, current_batch_size)
    
    def _compute_accuracy_on_subset(
        self,
        dataloader,
        max_samples: int = 5000,
        max_new_tokens: int = 16,
        mask_applied: bool = False,
    ) -> Tuple[float, List[Dict]]:
        """
        Compute accuracy on a subset of the dataset.
        
        Args:
            max_samples: Maximum number of samples to process (total across all ranks in distributed mode)
        
        Returns:
            overall_accuracy: Mean accuracy
            per_sample_scores: List of per-sample results
        """
        # In distributed mode, divide max_samples across all ranks
        # Each rank processes max_samples_per_rank samples
        if self.is_distributed:
            max_samples_per_rank = max_samples // self.world_size
            # Add remainder to rank 0
            if self.rank == 0:
                max_samples_per_rank += max_samples % self.world_size
        else:
            max_samples_per_rank = max_samples
        
        all_scores = []
        all_predictions = []
        
        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(dataloader, total=min(max_samples_per_rank // dataloader.batch_size, len(dataloader)))):
                # Move batch to device
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate predictions
                from transformers import GenerationConfig
                
                eos_token_id = self.tokenizer.eos_token_id
                if eos_token_id is None:
                    eos_token_id = getattr(self.model.config, 'eos_token_id', None)
                
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = getattr(self.model.config, 'pad_token_id', None)
                
                vqa_max_tokens = min(max_new_tokens, 16)
                
                # Clear cache before generation to avoid attention bias size mismatches with mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Clear model's internal cache if it exists (for attention bias)
                if hasattr(self.model.model, '_MolmoModel__cache'):
                    self.model.model._MolmoModel__cache.clear()
                
                # Model requires use_cache=True (hardcoded assertion in generate method)
                # We always use use_cache=True, even with mask applied
                # The mask wrapper should handle this correctly when all layers are active
                use_cache_for_generation = True
                
                generation_config = GenerationConfig(
                    max_new_tokens=vqa_max_tokens,
                    do_sample=False,
                    use_cache=use_cache_for_generation,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
                
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    outputs = self.model.generate(
                        input_ids=batch["input_ids"],
                        images=batch.get("images"),
                        image_masks=batch.get("image_masks"),
                        image_input_idx=batch.get("image_input_idx"),
                        generation_config=generation_config,
                    )
                
                # Compute accuracy for this batch
                batch_accuracy = self.compute_accuracy(
                    batch=batch,
                    predictions=outputs,
                    metric_name="vqa_score",
                )
                
                all_scores.extend([s["score"] for s in batch_accuracy["per_sample_scores"]])
                all_predictions.extend(batch_accuracy["per_sample_scores"])
                
                # Stop if we've collected enough samples for this rank
                if len(all_scores) >= max_samples_per_rank:
                    break
        
        overall_accuracy = np.mean(all_scores) if all_scores else 0.0
        return overall_accuracy, all_predictions
    
    def _sensitivity_analysis(
        self,
        dataset_name: str,
        split: str,
        batch_size: int,
        max_new_tokens: int,
        num_samples: int = 5000,
        auto_adjust_batch_size: bool = True,
    ) -> Dict[int, float]:
        """
        Stage 1: Sensitivity Analysis
        
        For each layer l, compute:
        ΔAcc(l) = Acc_full - Acc_without_layer_l
        
        Returns:
            importance_scores: Dict mapping layer_idx -> importance_score (ΔAcc)
        """
        if not self.is_distributed or self.rank == 0:
            log.info("=" * 80)
            log.info("Stage 1: Sensitivity Analysis")
            log.info("=" * 80)
            log.info(f"Computing layer importance by ablating each layer individually")
            if self.is_distributed:
                samples_per_rank = num_samples // self.world_size
                samples_per_rank_rank0 = samples_per_rank + (num_samples % self.world_size)
                log.info(f"Using {num_samples} total samples for sensitivity analysis "
                        f"({samples_per_rank_rank0} for rank 0, {samples_per_rank} for other ranks, {self.world_size} ranks)")
            else:
                log.info(f"Using {num_samples} samples for sensitivity analysis")
        
        # Import data loading modules
        from molmo.data import get_dataset_by_name
        from molmo.preprocessors.multimodal_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        
        dataset = get_dataset_by_name(dataset_name, split=split)
        
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
        
        det_dataset = DeterministicDataset(dataset, preprocessor, seed=42)
        
        # Use DistributedSampler if running in distributed mode
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
        
        dataloader = torch.utils.data.DataLoader(
            det_dataset,
            batch_size=batch_size,
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
        if not self.is_distributed or self.rank == 0:
            log.info(f"Total transformer blocks: {total_blocks}")
        
        if auto_adjust_batch_size and (not self.is_distributed or self.rank == 0):
            log.info(f"Auto-adjusting batch size enabled: will optimize for each configuration")
        
        # Step 1: Compute baseline accuracy (all layers active)
        # Apply mask with all layers active (all 1s) to maintain consistent behavior
        if not self.is_distributed or self.rank == 0:
            log.info("Computing baseline accuracy (all layers active)...")
        
        # Create mask with all layers active (baseline)
        baseline_mask = torch.ones(total_blocks, dtype=torch.bool)
        baseline_mask_wrapper = BlockMaskWrapper(self.model.model, baseline_mask)
        baseline_mask_wrapper.apply()
        
        try:
            if auto_adjust_batch_size:
                # Find optimal batch size for full model
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
                
                # Temporarily remove mask for batch size testing
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
                    # Restore mask after batch size testing
                    baseline_mask_wrapper.apply()
                dataloader = create_dataloader(current_batch_size)
            else:
                current_batch_size = batch_size
                dataloader = torch.utils.data.DataLoader(
                    det_dataset,
                    batch_size=current_batch_size,
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
            
            baseline_accuracy, _ = self._compute_accuracy_on_subset(
                dataloader,
                max_samples=num_samples,
                max_new_tokens=max_new_tokens,
                mask_applied=True,  # Mask is applied (but all layers are active)
            )
        finally:
            # Remove baseline mask before starting ablation
            baseline_mask_wrapper.remove()
        if not self.is_distributed or self.rank == 0:
            log.info(f"Baseline accuracy (all {total_blocks} layers): {baseline_accuracy:.4f}")
        
        # Step 2: Ablate each layer individually
        # Only explore layers 1-14 (indices 1-14), keep layer 0 and layer 15 always active
        layers_to_explore = list(range(1, total_blocks - 1))  # Layers 1 to 14 (indices 1-14)
        if not self.is_distributed or self.rank == 0:
            log.info(f"Exploring importance of layers {layers_to_explore} (keeping layer 0 and layer {total_blocks-1} always active)")
        
        importance_scores = {}
        mask_wrapper = None
        
        try:
            for layer_idx in layers_to_explore:
                if not self.is_distributed or self.rank == 0:
                    log.info(f"=" * 80)
                    log.info(f"Ablating layer {layer_idx}/{total_blocks-1} (layer 0 and {total_blocks-1} always active)")
                    log.info(f"=" * 80)
                
                # Create mask: all layers active except layer_idx
                # Layer 0 and layer (total_blocks-1) are always active
                block_mask = torch.ones(total_blocks, dtype=torch.bool)
                block_mask[layer_idx] = False  # Skip this layer
                # Ensure layer 0 and last layer are always active
                block_mask[0] = True
                block_mask[total_blocks - 1] = True
                
                # Apply mask wrapper
                if mask_wrapper is not None:
                    mask_wrapper.remove()
                
                mask_wrapper = BlockMaskWrapper(self.model.model, block_mask)
                mask_wrapper.apply()
                
                # Adjust batch size if needed (ablating one layer = total_blocks - 1 active)
                if auto_adjust_batch_size:
                    # Temporarily remove mask for batch size testing (to avoid attention bias size issues during generation)
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
                        # Restore mask after batch size testing
                        mask_wrapper.apply()
                    dataloader = create_dataloader(current_batch_size)
                else:
                    current_batch_size = batch_size
                    # Create dataloader directly (no need for create_dataloader function when auto_adjust is disabled)
                    dataloader = torch.utils.data.DataLoader(
                        det_dataset,
                        batch_size=current_batch_size,
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
                
                # Record start time for this ablation
                ablation_start_time = time.time()
                
                # Compute accuracy without this layer
                ablated_accuracy, per_sample_scores = self._compute_accuracy_on_subset(
                    dataloader,
                    max_samples=num_samples,
                    max_new_tokens=max_new_tokens,
                    mask_applied=True,  # Mask is applied
                )
                
                # Record end time
                ablation_end_time = time.time()
                ablation_duration_seconds = ablation_end_time - ablation_start_time
                
                # Importance = accuracy drop when this layer is removed
                # Higher drop = more important
                importance_score = baseline_accuracy - ablated_accuracy
                importance_scores[layer_idx] = importance_score
                
                if not self.is_distributed or self.rank == 0:
                    log.info(f"Layer {layer_idx}: Accuracy without layer = {ablated_accuracy:.4f}, "
                            f"ΔAcc = {importance_score:.4f} (higher = more important)")
                
                # Determine active block indices (all layers except the ablated one)
                active_block_indices = sorted([i for i in range(total_blocks) if i != layer_idx])
                # Ensure layer 0 and last layer are always active (they should already be, but make it explicit)
                if 0 not in active_block_indices:
                    active_block_indices.insert(0, 0)
                if (total_blocks - 1) not in active_block_indices:
                    active_block_indices.append(total_blocks - 1)
                active_block_indices = sorted(active_block_indices)
                
                # Save individual ablation result (each rank saves its own in distributed mode)
                ablation_result = {
                    "summary": [{
                        "ablated_layer": layer_idx,
                        "num_total_blocks": total_blocks,
                        "active_block_indices": active_block_indices,
                        "ablated_block_index": layer_idx,
                        "baseline_accuracy": float(baseline_accuracy),
                        "ablated_accuracy": float(ablated_accuracy),
                        "importance_score": float(importance_score),
                        "num_samples": len(per_sample_scores),
                        "std": float(np.std([s["score"] for s in per_sample_scores])) if per_sample_scores else 0.0,
                        "duration_seconds": float(ablation_duration_seconds),
                        "duration_minutes": float(ablation_duration_seconds / 60.0),
                        "duration_hours": float(ablation_duration_seconds / 3600.0),
                        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ablation_start_time)),
                        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ablation_end_time)),
                    }],
                    "all_samples": per_sample_scores,
                    "config": {
                        "dataset_name": dataset_name,
                        "split": split,
                        "batch_size": current_batch_size,
                        "max_new_tokens": max_new_tokens,
                        "num_samples": num_samples,
                        "ablated_layer": layer_idx,
                        "num_total_blocks": total_blocks,
                        "always_active_layers": [0, total_blocks - 1],
                        "world_size": self.world_size,
                        "rank": self.rank,
                    }
                }
                
                # Save individual ablation result (each rank saves its own)
                filename = f"ablation_layer_{layer_idx}_rank{self.rank}.json"
                self.save_results(ablation_result, filename)
                if not self.is_distributed or self.rank == 0:
                    log.info(f"Saved ablation result for layer {layer_idx} to {filename}")
        
        finally:
            # Restore original forward method
            if mask_wrapper is not None:
                mask_wrapper.remove()
                if not self.is_distributed or self.rank == 0:
                    log.info("Restored original forward method")
        
        # Sort layers by importance (least important first)
        sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1])
        if not self.is_distributed or self.rank == 0:
            log.info("=" * 80)
            log.info("Layer Importance Ranking (least important → most important):")
            log.info("=" * 80)
        for layer_idx, importance in sorted_layers:
            if not self.is_distributed or self.rank == 0:
                log.info(f"Layer {layer_idx}: ΔAcc = {importance:.4f}")
        
        return importance_scores
    
    def _importance_based_pruning(
        self,
        importance_scores: Dict[int, float],
        dataset_name: str,
        split: str,
        batch_size: int,
        max_new_tokens: int,
        num_samples: int = 5000,
        min_layers: int = 8,
        auto_adjust_batch_size: bool = True,
    ):
        """
        Stage 2: Importance-based Pruning
        
        Remove layers from least important to most important.
        Test different numbers of active layers (from min_layers to total_blocks).
        """
        if not self.is_distributed or self.rank == 0:
            log.info("=" * 80)
            log.info("Stage 2: Importance-based Pruning")
            log.info("=" * 80)
        
        total_blocks = len(self.model.model.transformer.blocks)
        
        # Sort layers by importance (least important first)
        # Note: Only layers 1-14 are in importance_scores (layer 0 and 15 are always kept)
        sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1])
        removal_order = [layer_idx for layer_idx, _ in sorted_layers]
        
        if not self.is_distributed or self.rank == 0:
            log.info(f"Removal order for explorable layers (least → most important): {removal_order}")
            log.info(f"Layer 0 and layer {total_blocks-1} are always kept active")
            # Test from all layers (16) down to 10 layers (prune 0 to 6 least important)
            max_prune = total_blocks - 10  # Maximum 6 layers to prune (keep at least 10 layers)
            log.info(f"Testing from {total_blocks} layers (all) down to 10 layers (prune 0 to {max_prune} least important)")
        
        # Import data loading modules
        from molmo.data import get_dataset_by_name
        from molmo.preprocessors.multimodal_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        
        dataset = get_dataset_by_name(dataset_name, split=split)
        
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
        
        det_dataset = DeterministicDataset(dataset, preprocessor, seed=42)
        
        # Use DistributedSampler if running in distributed mode
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
        
        results_data = []
        mask_wrapper = None
        
        # Test from all layers (16) down to min_layers (prune 0 to (16-min_layers) least important)
        # num_active: 16, 15, 14, ..., min_layers
        # num_to_prune: 0, 1, 2, ..., (16-min_layers) (from removal_order)
        max_prune = total_blocks - min_layers  # Maximum layers to prune
        num_active_list = list(range(min_layers, total_blocks + 1))  # From min_layers to 16 layers
        num_active_list.reverse()  # Start from 16, then 15, 14, ..., min_layers
        
        try:
            for num_active in num_active_list:
                if not self.is_distributed or self.rank == 0:
                    log.info("=" * 80)
                    log.info(f"Testing {num_active}/{total_blocks} active layers")
                    log.info("=" * 80)
                
                # Calculate how many layers to prune from removal_order
                # removal_order contains only layers 1-14 (explorable layers)
                # Layer 0 and 15 are always kept, so we have 14 explorable layers
                # To get num_active layers total, we need:
                # - Always keep: layer 0 and layer 15 (2 layers)
                # - Keep from removal_order: (num_active - 2) layers
                # - Prune from removal_order: (14 - (num_active - 2)) = (16 - num_active) layers
                num_to_prune = total_blocks - num_active  # Number of layers to prune (0 to 6)
                
                if num_to_prune == 0:
                    # Keep all layers (16 layers)
                    layers_to_remove = []
                    layers_to_keep = list(range(total_blocks))  # All layers 0-15
                else:
                    # Prune num_to_prune least important layers from removal_order
                    layers_to_remove = removal_order[:num_to_prune]
                    layers_to_keep_from_removal = removal_order[num_to_prune:]
                    # Always add layer 0 and layer 15
                    layers_to_keep = [0] + sorted(layers_to_keep_from_removal) + [total_blocks - 1]
                
                actual_num_active = len(layers_to_keep)
                
                if not self.is_distributed or self.rank == 0:
                    if num_to_prune == 0:
                        log.info(f"Keeping all {actual_num_active} layers (no pruning)")
                    else:
                        log.info(f"Pruning {num_to_prune} least important layers: {layers_to_remove}")
                        log.info(f"Keeping {actual_num_active} layers (including always-active layers 0 and {total_blocks-1}): {sorted(layers_to_keep)}")
                
                # Create mask: activate only the kept layers
                block_mask = torch.zeros(total_blocks, dtype=torch.bool)
                for layer_idx in layers_to_keep:
                    block_mask[layer_idx] = True
                # Ensure layer 0 and last layer are always active
                block_mask[0] = True
                block_mask[total_blocks - 1] = True
                
                # Apply mask wrapper
                if mask_wrapper is not None:
                    mask_wrapper.remove()
                
                mask_wrapper = BlockMaskWrapper(self.model.model, block_mask)
                mask_wrapper.apply()
                
                # Determine batch size for this num_active_blocks
                if auto_adjust_batch_size:
                    # Create a factory function for dataloader creation
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
                    
                    # Temporarily remove mask for batch size testing (to avoid attention bias size issues during generation)
                    mask_wrapper.remove()
                    try:
                        # Find optimal batch size
                        optimal_batch_size = self._find_optimal_batch_size(
                            num_active_blocks=num_active,
                            total_blocks=total_blocks,
                            initial_batch_size=batch_size,
                            dataloader_factory=create_dataloader,
                        )
                        current_batch_size = optimal_batch_size
                    finally:
                        # Restore mask after batch size testing
                        mask_wrapper.apply()
                else:
                    current_batch_size = batch_size
                
                if not self.is_distributed or self.rank == 0:
                    log.info(f"Using batch size: {current_batch_size} for {actual_num_active}/{total_blocks} active layers")
                
                # Create dataloader with optimal batch size
                dataloader = torch.utils.data.DataLoader(
                    det_dataset,
                    batch_size=current_batch_size,
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
                
                # Record start time
                config_start_time = time.time()
                
                # Compute accuracy
                accuracy, per_sample_scores = self._compute_accuracy_on_subset(
                    dataloader,
                    max_samples=num_samples,
                    max_new_tokens=max_new_tokens,
                    mask_applied=True,  # Mask is applied
                )
                
                # Record end time
                config_end_time = time.time()
                config_duration_seconds = config_end_time - config_start_time
                config_duration_minutes = config_duration_seconds / 60.0
                config_duration_hours = config_duration_seconds / 3600.0
                
                result_entry = {
                    "num_active_blocks": actual_num_active,  # Use actual number of active layers
                    "num_total_blocks": total_blocks,
                    "active_block_indices": sorted(layers_to_keep),
                    "removed_block_indices": sorted(layers_to_remove),
                    "always_active_layers": [0, total_blocks - 1],  # Record always-active layers
                    "accuracy": float(accuracy),
                    "num_samples": len(per_sample_scores),
                    "std": float(np.std([s["score"] for s in per_sample_scores])) if per_sample_scores else 0.0,
                    "per_sample_scores": per_sample_scores,
                    "batch_size_used": current_batch_size,  # Record actual batch size used
                    "duration_seconds": float(config_duration_seconds),
                    "duration_minutes": float(config_duration_minutes),
                    "duration_hours": float(config_duration_hours),
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_start_time)),
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_end_time)),
                }
                
                results_data.append(result_entry)
                
                # Don't log here - will log merged results after gathering from all ranks
                
                # Save individual result
                if not self.is_distributed or self.rank == 0:
                    single_result = {
                        "summary": [{
                            "num_active_blocks": actual_num_active,
                            "num_total_blocks": total_blocks,
                            "active_block_indices": sorted(layers_to_keep),
                            "removed_block_indices": sorted(layers_to_remove),
                            "accuracy": float(accuracy),
                            "num_samples": len(per_sample_scores),
                            "std": result_entry["std"],
                            "duration_seconds": float(config_duration_seconds),
                            "duration_minutes": float(config_duration_minutes),
                            "duration_hours": float(config_duration_hours),
                            "start_time": result_entry["start_time"],
                            "end_time": result_entry["end_time"],
                        }],
                        "all_samples": per_sample_scores,
                        "config": {
                            "dataset_name": dataset_name,
                            "split": split,
                            "batch_size": batch_size,
                            "max_new_tokens": max_new_tokens,
                            "num_active_blocks": actual_num_active,
                            "num_total_blocks": total_blocks,
                            "active_block_indices": sorted(layers_to_keep),
                            "removed_block_indices": sorted(layers_to_remove),
                            "world_size": self.world_size,
                            "rank": self.rank,
                        }
                    }
                    
                    if self.is_distributed:
                        filename = f"exp3_accuracy_sensitivity_blocks_{num_active}_rank{self.rank}.json"
                    else:
                        filename = f"exp3_accuracy_sensitivity_blocks_{num_active}.json"
                    
                    self.save_results(single_result, filename)
                    log.info(f"Saved result for {num_active}/{total_blocks} blocks to {filename}")
        
        finally:
            # Restore original forward method
            if mask_wrapper is not None:
                mask_wrapper.remove()
                if not self.is_distributed or self.rank == 0:
                    log.info("Restored original forward method")
        
        return results_data
    
    def run(
        self,
        dataset_name: str = "coco_2014_vqa",
        split: str = "validation",
        batch_size: int = 8,
        max_new_tokens: int = 16,
        num_samples: int = 5000,
        min_layers: int = 8,
        skip_sensitivity: bool = False,
        importance_scores_file: Optional[str] = None,
        auto_adjust_batch_size: bool = False,  # Disabled: use fixed batch_size
    ):
        """
        Run Exp3 Sensitivity Analysis.
        
        Args:
            dataset_name: Dataset name (default: "coco_2014_vqa")
            split: Dataset split (default: "validation")
            batch_size: Batch size per GPU
            max_new_tokens: Maximum tokens to generate
            num_samples: Total number of samples to use for evaluation across all ranks (default: 5000)
                        In distributed mode, this is divided across all ranks
            min_layers: Minimum number of layers to test (default: 8)
            skip_sensitivity: If True, skip sensitivity analysis and load from file
            importance_scores_file: Path to saved importance scores JSON file
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
            
            # Save importance scores
            if not self.is_distributed or self.rank == 0:
                importance_file = os.path.join(self.output_dir, "layer_importance_scores.json")
                with open(importance_file, 'w') as f:
                    json.dump(importance_scores, f, indent=2)
                log.info(f"Saved importance scores to {importance_file}")
        
        # Stage 2: Importance-based Pruning
        results_data = self._importance_based_pruning(
            importance_scores=importance_scores,
            dataset_name=dataset_name,
            split=split,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
            min_layers=min_layers,
            auto_adjust_batch_size=auto_adjust_batch_size,
        )
        
        # Gather results from all ranks if distributed
        if self.is_distributed:
            if self.rank == 0:
                gathered_results = [None] * self.world_size
                dist.gather_object(results_data, gathered_results, dst=0)
                
                # Merge results from all ranks
                merged_results_data = []
                for rank_results in gathered_results:
                    if rank_results is not None:
                        merged_results_data.extend(rank_results)
                
                # Group by num_active_blocks and merge
                final_results_data = []
                seen_configs = set()
                
                for result in merged_results_data:
                    num_active = result["num_active_blocks"]
                    if num_active in seen_configs:
                        continue
                    seen_configs.add(num_active)
                    
                    # Find all results with same num_active_blocks across all ranks
                    all_scores_for_config = []
                    all_predictions_for_config = []
                    max_duration_seconds = 0.0
                    earliest_start_time = None
                    latest_end_time = None
                    active_indices = None
                    removed_indices = None
                    
                    for r in merged_results_data:
                        if r["num_active_blocks"] == num_active:
                            all_scores_for_config.extend([s["score"] for s in r["per_sample_scores"]])
                            all_predictions_for_config.extend(r["per_sample_scores"])
                            if "duration_seconds" in r:
                                max_duration_seconds = max(max_duration_seconds, r["duration_seconds"])
                            if "start_time" in r:
                                if earliest_start_time is None or r["start_time"] < earliest_start_time:
                                    earliest_start_time = r["start_time"]
                            if "end_time" in r:
                                if latest_end_time is None or r["end_time"] > latest_end_time:
                                    latest_end_time = r["end_time"]
                            if active_indices is None and "active_block_indices" in r:
                                active_indices = r["active_block_indices"]
                            if removed_indices is None and "removed_block_indices" in r:
                                removed_indices = r["removed_block_indices"]
                    
                    overall_accuracy = np.mean(all_scores_for_config) if all_scores_for_config else 0.0
                    
                    final_results_data.append({
                        "num_active_blocks": num_active,
                        "num_total_blocks": result["num_total_blocks"],
                        "active_block_indices": active_indices if active_indices is not None else [],
                        "removed_block_indices": removed_indices if removed_indices is not None else [],
                        "accuracy": float(overall_accuracy),
                        "num_samples": len(all_scores_for_config),
                        "std": float(np.std(all_scores_for_config)) if all_scores_for_config else 0.0,
                        "per_sample_scores": all_predictions_for_config,
                        "duration_seconds": float(max_duration_seconds),
                        "duration_minutes": float(max_duration_seconds / 60.0),
                        "duration_hours": float(max_duration_seconds / 3600.0),
                        "start_time": earliest_start_time if earliest_start_time else None,
                        "end_time": latest_end_time if latest_end_time else None,
                    })
                    
                    # Log merged result (only on rank 0)
                    log.info(f"{num_active}/{result['num_total_blocks']} active layers: Accuracy={overall_accuracy:.4f} "
                            f"({len(all_scores_for_config)} samples from all ranks)")
                    log.info(f"{num_active}/{result['num_total_blocks']} active layers: Duration={max_duration_seconds/60.0:.2f} minutes")
                
                results_data = final_results_data
            else:
                dist.gather_object(results_data, None, dst=0)
                dist.destroy_process_group()
                return
        else:
            # Non-distributed mode: log results immediately
            for result_entry in results_data:
                num_active = result_entry["num_active_blocks"]
                total_blocks = result_entry["num_total_blocks"]
                accuracy = result_entry["accuracy"]
                num_samples = result_entry["num_samples"]
                duration_minutes = result_entry["duration_minutes"]
                log.info(f"{num_active}/{total_blocks} active layers: Accuracy={accuracy:.4f} "
                        f"({num_samples} samples)")
                log.info(f"{num_active}/{total_blocks} active layers: Duration={duration_minutes:.2f} minutes")
        
        # Save final results
        summary = []
        all_samples = []
        
        for config_result in results_data:
            summary_entry = {
                "num_active_blocks": config_result["num_active_blocks"],
                "num_total_blocks": config_result["num_total_blocks"],
                "active_block_indices": config_result["active_block_indices"],
                "removed_block_indices": config_result["removed_block_indices"],
                "accuracy": config_result["accuracy"],
                "num_samples": config_result["num_samples"],
                "std": config_result["std"],
            }
            if "duration_seconds" in config_result:
                summary_entry["duration_seconds"] = config_result["duration_seconds"]
                summary_entry["duration_minutes"] = config_result["duration_minutes"]
                summary_entry["duration_hours"] = config_result["duration_hours"]
            if "start_time" in config_result:
                summary_entry["start_time"] = config_result["start_time"]
            if "end_time" in config_result:
                summary_entry["end_time"] = config_result["end_time"]
            summary.append(summary_entry)
            if "per_sample_scores" in config_result:
                all_samples.extend(config_result["per_sample_scores"])
        
        final_results = {
            "summary": summary,
            "all_samples": all_samples,
            "importance_scores": importance_scores,
            "config": {
                "dataset_name": dataset_name,
                "split": split,
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "num_samples": num_samples,
                "min_layers": min_layers,
                "total_blocks": len(self.model.model.transformer.blocks),
                "world_size": self.world_size,
            }
        }
        
        self.save_results(final_results, "exp3_accuracy_sensitivity_results.json")
        log.info(f"Results saved. Total samples: {len(all_samples)}")
        
        # Cleanup distributed process group
        if self.is_distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Exp3 Sensitivity: Layer Importance Analysis")
    parser.add_argument("--model_path", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results/profiling/exp3_accuracy_sensitivity")
    parser.add_argument("--dataset_name", type=str, default="coco_2014_vqa")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=16,
                       help="Maximum tokens to generate (default: 16, optimized for VQA)")
    parser.add_argument("--num_samples", type=int, default=5000,
                       help="Total number of samples to use for evaluation across all ranks (default: 5000). "
                            "In distributed mode, this is divided across all ranks.")
    parser.add_argument("--min_layers", type=int, default=10,
                       help="Minimum number of layers to test (default: 10, i.e., prune at most 6 layers)")
    parser.add_argument("--skip_sensitivity", action="store_true",
                       help="Skip sensitivity analysis and load from file")
    parser.add_argument("--importance_scores_file", type=str, default=None,
                       help="Path to saved importance scores JSON file")
    parser.add_argument("--auto_adjust_batch_size", action="store_true", default=False,
                       help="Automatically adjust batch size for each configuration to avoid OOM (default: False, use fixed batch_size)")
    parser.add_argument("--no_auto_adjust_batch_size", dest="auto_adjust_batch_size", action="store_false",
                       help="Disable automatic batch size adjustment (default)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = Exp3SensitivityExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    experiment.run(
        dataset_name=args.dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        min_layers=args.min_layers,
        skip_sensitivity=args.skip_sensitivity,
        importance_scores_file=args.importance_scores_file,
        auto_adjust_batch_size=args.auto_adjust_batch_size,
    )


if __name__ == "__main__":
    main()

