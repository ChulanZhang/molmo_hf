"""
Exp3 Accuracy Measurement: Transformer Blocks Mask
Multi-GPU, large batch size, measure accuracy on full VQA v2 validation set.
"""

import argparse
import logging
import sys
import os
import time
from typing import Dict, List, Any, Optional

# Set TOKENIZERS_PARALLELISM to avoid warnings when using DataLoader with num_workers > 0
# This happens because tokenizers are used in the main process, then forked to worker processes
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


class Exp3AccuracyExperiment(BaseExperiment):
    """
    Exp3 Accuracy: Measure accuracy for different numbers of active transformer blocks.
    Uses full VQA v2 validation set.
    
    Supports multi-GPU via torchrun. When launched with torchrun, automatically
    detects distributed environment and uses DistributedSampler.
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
            # Check if the device exists
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
                        
                        # Then try a short generation (this uses more memory)
                        from transformers import GenerationConfig
                        eos_token_id = self.tokenizer.eos_token_id
                        if eos_token_id is None:
                            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
                        
                        pad_token_id = self.tokenizer.pad_token_id
                        if pad_token_id is None:
                            pad_token_id = getattr(self.model.config, 'pad_token_id', None)
                        
                        test_gen_config = GenerationConfig(
                            max_new_tokens=16,
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=eos_token_id,
                            pad_token_id=pad_token_id,
                        )
                        
                        # Try generating to check memory
                        _ = self.model.generate(
                            input_ids=test_batch["input_ids"],
                            images=test_batch.get("images"),
                            image_masks=test_batch.get("image_masks"),
                            image_input_idx=test_batch.get("image_input_idx"),
                            generation_config=test_gen_config,
                        )
                
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
    
    def run(
        self,
        dataset_name: str = "coco_2014_vqa",
        split: str = "validation",
        batch_size: int = 8,
        max_new_tokens: int = 16,
        num_active_blocks_list: List[int] = None,
        auto_adjust_batch_size: bool = True,
    ):
        """
        Run Exp3 accuracy measurement.
        
        Args:
            dataset_name: Dataset name (default: "coco_2014_vqa")
            split: Dataset split (default: "validation")
            batch_size: Base batch size per GPU (default: 8). 
                       If auto_adjust_batch_size=True, this is the starting point.
            max_new_tokens: Maximum tokens to generate
            num_active_blocks_list: List of active block counts to test 
                (default: sequential activation from 8 layers minimum [8, 9, 10, ..., total_blocks])
                Each count activates layers 1-N sequentially (e.g., 8 layers = layers 1-8, 9 layers = layers 1-9)
            auto_adjust_batch_size: If True, automatically adjust batch size for each num_active_blocks to avoid OOM
        """
        # Get total number of blocks
        total_blocks = len(self.model.model.transformer.blocks)
        log.info(f"Total transformer blocks: {total_blocks}")
        
        # Determine which block counts to test
        # Strategy: Sequential activation from layer 1, starting from 8 layers minimum
        # This gives: [8, 9, 10, 11, 12, 13, 14, 15, 16] (assuming 16 total blocks)
        # For 8 layers: activate layers 1-8 (indices 0-7)
        # For 9 layers: activate layers 1-9 (indices 0-8)
        # For 10 layers: activate layers 1-10 (indices 0-9), etc.
        min_layers = 8  # Minimum number of layers to activate
        if num_active_blocks_list is None:
            # Start from 8 layers, then increment by 1 up to total_blocks
            num_active_blocks_list = list(range(min_layers, total_blocks + 1))
            # Ensure we don't exceed total_blocks
            num_active_blocks_list = [n for n in num_active_blocks_list if n <= total_blocks]
        else:
            num_active_blocks_list = [n for n in num_active_blocks_list if 1 <= n <= total_blocks]
        
        # Ensure minimum is 8 layers
        num_active_blocks_list = [n for n in num_active_blocks_list if n >= min_layers]
        if not num_active_blocks_list:
            log.error(f"No valid block counts! Minimum is {min_layers} layers.")
            return
        
        log.info(f"Testing {len(num_active_blocks_list)} active block counts: {num_active_blocks_list}")
        log.info(f"Strategy: Sequential activation from layer 1, starting from {min_layers} layers minimum")
        log.info(f"Example: {min_layers} layers = layers 1-{min_layers} (indices 0-{min_layers-1}), "
                f"{min_layers+1} layers = layers 1-{min_layers+1} (indices 0-{min_layers}), etc.")
        log.info(f"Measuring accuracy on full {dataset_name}/{split} set")
        log.info(f"Base batch size per GPU: {batch_size}, Total GPUs: {self.world_size}")
        if auto_adjust_batch_size:
            log.info(f"Auto-adjusting batch size enabled: will optimize for each num_active_blocks")
        else:
            log.info(f"Global batch size: {batch_size * self.world_size}")
        
        results_data = []
        mask_wrapper = None
        
        # Import data loading modules once
        from molmo.data import get_dataset_by_name
        from molmo.preprocessors.multimodal_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        
        dataset = get_dataset_by_name(dataset_name, split=split)
        
        try:
            for num_active in num_active_blocks_list:
                # Sequential activation: always use first N layers (indices 0 to num_active-1)
                # For 8 layers: [0, 1, 2, 3, 4, 5, 6, 7] (layers 1-8)
                # For 9 layers: [0, 1, 2, 3, 4, 5, 6, 7, 8] (layers 1-9)
                # For 10 layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (layers 1-10), etc.
                log.info(f"=" * 80)
                log.info(f"Testing {num_active}/{total_blocks} active blocks (layers 1-{num_active})")
                log.info(f"=" * 80)
                
                # Record start time for this configuration
                config_start_time = time.time()
                
                # Sequential activation: use first num_active layers
                block_indices = list(range(num_active))
                
                block_mask = torch.zeros(total_blocks, dtype=torch.bool)
                for idx in block_indices:
                    block_mask[idx] = True
                
                log.info(f"Active block indices: {block_indices} (layers 1-{num_active})")
                
                # Apply mask wrapper
                if mask_wrapper is not None:
                    mask_wrapper.remove()
                
                mask_wrapper = BlockMaskWrapper(self.model.model, block_mask)
                mask_wrapper.apply()
                
                # Determine batch size for this num_active_blocks
                if auto_adjust_batch_size:
                    # Create a factory function for dataloader creation
                    def create_dataloader(bs):
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
                    
                    # Find optimal batch size
                    optimal_batch_size = self._find_optimal_batch_size(
                        num_active_blocks=num_active,
                        total_blocks=total_blocks,
                        initial_batch_size=batch_size,
                        dataloader_factory=create_dataloader,
                    )
                    current_batch_size = optimal_batch_size
                else:
                    current_batch_size = batch_size
                
                log.info(f"Using batch size: {current_batch_size} for {num_active}/{total_blocks} active blocks")
                
                # Build dataloader with optimal batch size
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
                
                all_scores = []
                all_predictions = []
                
                log.info(f"Measuring accuracy for {num_active}/{total_blocks} active blocks (layers 1-{num_active})...")
                
                with torch.inference_mode():
                    for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
                        # Move batch to device (non_blocking for async transfer)
                        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        # Generate predictions
                        from transformers import GenerationConfig
                        
                        # Get EOS token ID from tokenizer
                        eos_token_id = self.tokenizer.eos_token_id
                        if eos_token_id is None:
                            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
                        
                        # For VQA, answers are typically very short (1-10 tokens)
                        # Based on statistical analysis: max_new_tokens=16 covers 100% of answers
                        vqa_max_tokens = min(max_new_tokens, 16)  # Safe: 100% coverage
                        
                        generation_config = GenerationConfig(
                            max_new_tokens=vqa_max_tokens,
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=eos_token_id,
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
                
                # Record end time and calculate duration
                config_end_time = time.time()
                config_duration_seconds = config_end_time - config_start_time
                config_duration_hours = config_duration_seconds / 3600.0
                config_duration_minutes = config_duration_seconds / 60.0
                
                # Compute overall accuracy
                overall_accuracy = np.mean(all_scores) if all_scores else 0.0
                
                results_data.append({
                    "num_active_blocks": num_active,
                    "num_total_blocks": total_blocks,
                    "config_index": 0,
                    "active_block_indices": block_indices,
                    "accuracy": float(overall_accuracy),
                    "num_samples": len(all_scores),
                    "std": float(np.std(all_scores)) if all_scores else 0.0,
                    "per_sample_scores": all_predictions,
                    "batch_size_used": current_batch_size,  # Record actual batch size used
                    "duration_seconds": float(config_duration_seconds),
                    "duration_minutes": float(config_duration_minutes),
                    "duration_hours": float(config_duration_hours),
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_start_time)),
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_end_time)),
                })
                
                log.info(f"{num_active}/{total_blocks} active blocks (layers 1-{num_active}): "
                        f"Accuracy={overall_accuracy:.4f} ({len(all_scores)} samples)")
                log.info(f"{num_active}/{total_blocks} active blocks (layers 1-{num_active}): "
                        f"Duration={config_duration_minutes:.2f} minutes ({config_duration_hours:.2f} hours)")
                
                # Save result for this configuration immediately (incremental save)
                # This ensures we don't lose progress if the script is interrupted
                # Note: In distributed mode, each rank saves its own partial results, which will be merged later
                if not self.is_distributed or self.rank == 0:
                    single_config_result = {
                        "summary": [{
                            "num_active_blocks": num_active,
                            "num_total_blocks": total_blocks,
                            "config_index": 0,
                            "active_block_indices": block_indices,
                            "accuracy": float(overall_accuracy),
                            "num_samples": len(all_scores),
                            "std": float(np.std(all_scores)) if all_scores else 0.0,
                            "duration_seconds": float(config_duration_seconds),
                            "duration_minutes": float(config_duration_minutes),
                            "duration_hours": float(config_duration_hours),
                            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_start_time)),
                            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_end_time)),
                        }],
                        "all_samples": all_predictions,
                        "config": {
                            "dataset_name": dataset_name,
                            "split": split,
                            "batch_size": batch_size,
                            "max_new_tokens": max_new_tokens,
                            "num_active_blocks": num_active,
                            "num_total_blocks": total_blocks,
                            "config_index": 0,
                            "active_block_indices": block_indices,
                            "batch_size_used": current_batch_size,
                            "world_size": self.world_size,
                            "rank": self.rank,  # Note: This is per-rank result before merging
                            "duration_seconds": float(config_duration_seconds),
                            "duration_minutes": float(config_duration_minutes),
                            "duration_hours": float(config_duration_hours),
                            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_start_time)),
                            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_end_time)),
                        }
                    }
                    
                    # Save individual result file for this configuration
                    # In distributed mode, add rank suffix to avoid conflicts
                    if self.is_distributed:
                        individual_filename = f"exp3_accuracy_results_blocks_{num_active}_rank{self.rank}.json"
                    else:
                        individual_filename = f"exp3_accuracy_results_blocks_{num_active}.json"
                    
                    self.save_results(single_config_result, individual_filename)
                    log.info(f"Saved individual result for {num_active}/{total_blocks} blocks to {individual_filename}")
        
        finally:
            # Always restore original forward method
            if mask_wrapper is not None:
                mask_wrapper.remove()
                log.info("Restored original forward method")
        
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
                
                # Recompute accuracy for each num_active_blocks from merged results
                # Group by (num_active_blocks, config_index) - config_index is always 0 now (sequential activation only)
                final_results_data = []
                seen_configs = set()
                
                for result in merged_results_data:
                    num_active = result["num_active_blocks"]
                    config_idx = result.get("config_index", 0)
                    config_key = (num_active, config_idx)
                    
                    if config_key in seen_configs:
                        continue
                    seen_configs.add(config_key)
                    
                    # Find all results with same (num_active, config_index) across all ranks
                    all_scores_for_config = []
                    all_predictions_for_config = []
                    max_duration_seconds = 0.0
                    earliest_start_time = None
                    latest_end_time = None
                    block_indices = None
                    
                    for r in merged_results_data:
                        if r["num_active_blocks"] == num_active and r.get("config_index", 0) == config_idx:
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
                            if block_indices is None and "active_block_indices" in r:
                                block_indices = r["active_block_indices"]
                    
                    overall_accuracy = np.mean(all_scores_for_config) if all_scores_for_config else 0.0
                    
                    # Calculate duration from collected time info
                    max_duration_minutes = max_duration_seconds / 60.0 if max_duration_seconds > 0 else 0.0
                    max_duration_hours = max_duration_seconds / 3600.0 if max_duration_seconds > 0 else 0.0
                    
                    final_results_data.append({
                        "num_active_blocks": num_active,
                        "num_total_blocks": total_blocks,
                        "active_block_indices": block_indices if block_indices is not None else list(range(num_active)),
                        "accuracy": float(overall_accuracy),
                        "num_samples": len(all_scores_for_config),
                        "std": float(np.std(all_scores_for_config)) if all_scores_for_config else 0.0,
                        "per_sample_scores": all_predictions_for_config,
                        "duration_seconds": float(max_duration_seconds),
                        "duration_minutes": float(max_duration_minutes),
                        "duration_hours": float(max_duration_hours),
                        "start_time": earliest_start_time if earliest_start_time else None,
                        "end_time": latest_end_time if latest_end_time else None,
                    })
                
                results_data = final_results_data
                
                # Save merged results for each configuration (rank 0 only, after merging)
                for merged_result in final_results_data:
                    config_idx = merged_result.get("config_index", 0)
                    num_active = merged_result["num_active_blocks"]
                    
                    merged_config_result = {
                        "summary": [{
                            "num_active_blocks": num_active,
                            "num_total_blocks": total_blocks,
                            "config_index": config_idx,
                            "active_block_indices": merged_result.get("active_block_indices", []),
                            "accuracy": merged_result["accuracy"],
                            "num_samples": merged_result["num_samples"],
                            "std": merged_result["std"],
                            "duration_seconds": merged_result.get("duration_seconds", 0.0),
                            "duration_minutes": merged_result.get("duration_minutes", 0.0),
                            "duration_hours": merged_result.get("duration_hours", 0.0),
                            "start_time": merged_result.get("start_time"),
                            "end_time": merged_result.get("end_time"),
                        }],
                        "all_samples": merged_result["per_sample_scores"],
                        "config": {
                            "dataset_name": dataset_name,
                            "split": split,
                            "batch_size": batch_size,
                            "max_new_tokens": max_new_tokens,
                            "num_active_blocks": num_active,
                            "num_total_blocks": total_blocks,
                            "config_index": config_idx,
                            "active_block_indices": merged_result.get("active_block_indices", []),
                            "world_size": self.world_size,
                            "note": "Merged results from all ranks",
                            "duration_seconds": merged_result.get("duration_seconds", 0.0),
                            "duration_minutes": merged_result.get("duration_minutes", 0.0),
                            "duration_hours": merged_result.get("duration_hours", 0.0),
                            "start_time": merged_result.get("start_time"),
                            "end_time": merged_result.get("end_time"),
                        }
                    }
                    
                    # Save merged result (overwrites individual rank files)
                    if config_idx > 0:
                        individual_filename = f"exp3_accuracy_results_blocks_{num_active}_config{config_idx}.json"
                    else:
                        individual_filename = f"exp3_accuracy_results_blocks_{num_active}.json"
                    
                    self.save_results(merged_config_result, individual_filename)
                    if config_idx > 0:
                        log.info(f"Saved merged result for {num_active}/{total_blocks} blocks (config {config_idx}) to {individual_filename}")
                    else:
                        log.info(f"Saved merged result for {num_active}/{total_blocks} blocks to {individual_filename}")
            else:
                dist.gather_object(results_data, None, dst=0)
                dist.destroy_process_group()
                return
        
        # Save results (only on rank 0)
        summary = []
        all_samples = []
        
        for config_result in results_data:
            summary_entry = {
                "num_active_blocks": config_result["num_active_blocks"],
                "num_total_blocks": config_result["num_total_blocks"],
                "active_block_indices": config_result["active_block_indices"],
                "accuracy": config_result["accuracy"],
                "num_samples": config_result["num_samples"],
                "std": config_result["std"],
            }
            # Add time statistics if available
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
            "config": {
                "dataset_name": dataset_name,
                "split": split,
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "num_active_blocks_list": num_active_blocks_list,
                "total_blocks": total_blocks,
                "world_size": self.world_size,
            }
        }
        
        self.save_results(final_results, "exp3_accuracy_results.json")
        log.info(f"Results saved. Total samples: {len(all_samples)}")
        
        # Cleanup distributed process group
        if self.is_distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Exp3 Accuracy: Transformer Blocks Mask")
    parser.add_argument("--model_path", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results/profiling/exp3_accuracy")
    parser.add_argument("--dataset_name", type=str, default="coco_2014_vqa")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=16,
                       help="Maximum tokens to generate (default: 16, optimized for VQA)")
    parser.add_argument("--num_active_blocks", type=int, nargs="+", default=None,
                       help="List of active block counts to test (default: test all from 8 to total_blocks, sequential activation)")
    parser.add_argument("--auto_adjust_batch_size", action="store_true", default=True,
                       help="Automatically adjust batch size for each num_active_blocks to avoid OOM (default: True)")
    parser.add_argument("--no_auto_adjust_batch_size", dest="auto_adjust_batch_size", action="store_false",
                       help="Disable automatic batch size adjustment")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = Exp3AccuracyExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    experiment.run(
        dataset_name=args.dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_active_blocks_list=args.num_active_blocks,
        auto_adjust_batch_size=args.auto_adjust_batch_size,
    )


if __name__ == "__main__":
    main()

