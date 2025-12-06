"""
Exp2 Accuracy Measurement: MoE Top-K Analysis
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
from molmo.models.modeling_molmoe import MolmoeSparseMoeBlock
from molmo.torch_util import get_world_size, get_global_rank, get_local_rank

log = logging.getLogger(__name__)


class Exp2AccuracyExperiment(BaseExperiment):
    """
    Exp2 Accuracy: Measure accuracy for different MoE Top-K values.
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
    
    def _set_top_k(self, k: int):
        """Set top_k for all MoE blocks."""
        assert 1 <= k <= self.model.config.moe_num_experts, \
            f"top_k must be between 1 and {self.model.config.moe_num_experts}"
        
        self.model.config.moe_top_k = k
        
        transformer = self.model.model.transformer
        if isinstance(transformer, torch.nn.ModuleDict):
            blocks = transformer["blocks"] if "blocks" in transformer else []
        elif hasattr(transformer, 'blocks'):
            blocks = transformer.blocks
        else:
            blocks = []
        
        moe_blocks_found = 0
        for block in blocks:
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
                mlp_type = type(block.mlp)
                mlp_type_name = mlp_type.__name__ if hasattr(mlp_type, '__name__') else str(mlp_type)
                
                is_moe_block = (
                    isinstance(block.mlp, MolmoeSparseMoeBlock) or 
                    'MolmoeSparseMoeBlock' in mlp_type_name or
                    'SparseMoe' in mlp_type_name
                )
                
                if is_moe_block:
                    block.mlp.top_k = k
                    moe_blocks_found += 1
        
        log.info(f"Updated {moe_blocks_found} MoE blocks to use top_k={k}")
        return moe_blocks_found
    
    def _estimate_batch_size_for_top_k(self, top_k: int, base_batch_size: int) -> int:
        """
        Estimate appropriate batch size for a given top_k value.
        
        Memory usage scales with:
        - MoE experts: top_k determines how many experts are active
        - Smaller top_k = fewer active experts = less memory = larger batch size possible
        
        Heuristic: batch_size scales inversely with top_k (but less aggressively than max_crops)
        """
        # Top-K has less impact on memory than max_crops
        # Use more conservative scaling
        if top_k <= 4:
            scale_factor = 1.0  # Small top_k can use full batch size
        elif top_k <= 8:
            scale_factor = 0.9
        elif top_k <= 16:
            scale_factor = 0.8
        elif top_k <= 32:
            scale_factor = 0.7
        else:
            scale_factor = 0.6  # Large top_k needs smaller batch size
        
        estimated_batch_size = max(1, int(base_batch_size * scale_factor))
        return estimated_batch_size
    
    def _find_optimal_batch_size(
        self, 
        top_k: int, 
        initial_batch_size: int,
        dataloader_factory,
        max_attempts: int = 5,
    ) -> int:
        """
        Dynamically find the optimal batch size that doesn't cause OOM or CUDA errors.
        
        Uses binary search: start with estimated size, reduce if error, increase if safe.
        
        Args:
            top_k: Current top_k value
            initial_batch_size: Starting batch size to try (base value)
            dataloader_factory: Function that creates dataloader given batch_size
            max_attempts: Maximum number of attempts to find working batch size
            
        Returns:
            Optimal batch size that works without OOM or CUDA errors
        """
        import torch
        
        # For small top_k, allow exceeding initial_batch_size to find maximum
        # For larger top_k, use initial_batch_size as upper limit
        if top_k <= 4:
            max_allowed_batch_size = initial_batch_size * 2
        elif top_k <= 8:
            max_allowed_batch_size = int(initial_batch_size * 1.5)
        else:
            max_allowed_batch_size = initial_batch_size
        
        # Start with estimated batch size
        estimated_batch_size = self._estimate_batch_size_for_top_k(top_k, initial_batch_size)
        current_batch_size = min(estimated_batch_size, max_allowed_batch_size)
        
        log.info(f"Finding optimal batch size for top_k={top_k} (estimated: {estimated_batch_size}, "
                f"starting with: {current_batch_size}, max allowed: {max_allowed_batch_size})...")
        
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
                log.info(f"✓ Batch size {current_batch_size} works for top_k={top_k}")
                
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
                    log.warning(f"✗ Batch size {old_batch_size} caused {error_type} for top_k={top_k}, "
                              f"trying {current_batch_size}...")
                    
                    # If we've narrowed down to the same size, stop
                    if current_batch_size == old_batch_size:
                        break
                else:
                    # Other error, re-raise
                    raise
        
        # Return the last working size, or the smallest we tried
        if min_working is not None:
            log.info(f"Using batch size {min_working} for top_k={top_k}")
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
        top_k_values: List[int] = None,
        auto_adjust_batch_size: bool = True,
    ):
        """
        Run Exp2 accuracy measurement.
        
        Args:
            dataset_name: Dataset name (default: "coco_2014_vqa")
            split: Dataset split (default: "validation")
            batch_size: Base batch size per GPU (default: 8). 
                       If auto_adjust_batch_size=True, this is the starting point.
            max_new_tokens: Maximum tokens to generate
            top_k_values: List of top_k values to test (default: [4, 8, 12, ..., 64] - 16 groups)
            auto_adjust_batch_size: If True, automatically adjust batch size for each top_k to avoid OOM
        """
        if top_k_values is None:
            # Top-K from 4 to 64, step 4: [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
            top_k_values = list(range(4, 65, 4))  # 16 groups
        
        log.info(f"Testing {len(top_k_values)} Top-K values: {top_k_values}")
        log.info(f"Measuring accuracy on full {dataset_name}/{split} set")
        log.info(f"Base batch size per GPU: {batch_size}, Total GPUs: {self.world_size}")
        if auto_adjust_batch_size:
            log.info(f"Auto-adjusting batch size enabled: will optimize for each top_k")
        else:
            log.info(f"Global batch size: {batch_size * self.world_size}")
        
        results_data = []
        
        # Import data loading modules once
        from molmo.data import get_dataset_by_name
        from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        
        dataset = get_dataset_by_name(dataset_name, split=split)
        
        for k in top_k_values:
            log.info(f"=" * 80)
            log.info(f"Testing Top-K={k}")
            log.info(f"=" * 80)
            
            # Record start time for this configuration
            config_start_time = time.time()
            
            # Set top_k
            moe_blocks_found = self._set_top_k(k)
            if moe_blocks_found == 0:
                log.warning("No MoE blocks found! Skipping this configuration.")
                continue
            
            # Determine batch size for this top_k
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
                    top_k=k,
                    initial_batch_size=batch_size,
                    dataloader_factory=create_dataloader,
                )
                current_batch_size = optimal_batch_size
            else:
                current_batch_size = batch_size
            
            log.info(f"Using batch size: {current_batch_size} for top_k={k}")
            
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
            
            log.info(f"Measuring accuracy for Top-K={k}...")
            
            with torch.inference_mode():
                for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
                    # Move batch to device (non_blocking for async transfer)
                    batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Generate predictions
                    from transformers import GenerationConfig
                    
                    # Get EOS and PAD token IDs from tokenizer
                    eos_token_id = self.tokenizer.eos_token_id
                    if eos_token_id is None:
                        eos_token_id = getattr(self.model.config, 'eos_token_id', None)
                    
                    pad_token_id = self.tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = getattr(self.model.config, 'pad_token_id', None)
                    
                    # For VQA, answers are typically very short (1-10 tokens)
                    # Based on statistical analysis: max_new_tokens=16 covers 100% of answers
                    vqa_max_tokens = min(max_new_tokens, 16)  # Safe: 100% coverage
                    
                    generation_config = GenerationConfig(
                        max_new_tokens=vqa_max_tokens,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,  # Explicitly set to avoid warning
                    )
                    
                    try:
                        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                            outputs = self.model.generate(
                                input_ids=batch["input_ids"],
                                images=batch.get("images"),
                                image_masks=batch.get("image_masks"),
                                image_input_idx=batch.get("image_input_idx"),
                                generation_config=generation_config,
                            )
                    except RuntimeError as e:
                        error_str = str(e).lower()
                        if "invalid configuration argument" in error_str or "cuda error" in error_str:
                            log.error(f"CUDA error at batch {batch_idx}: {e}")
                            log.error(f"This usually means batch_size={batch_size} is too large for the model.")
                            log.error(f"Current: {batch_size} per GPU, {batch_size * self.world_size} global batch size")
                            log.error(f"Solution: Reduce --batch_size (try --batch_size 8 or smaller)")
                            raise RuntimeError(
                                f"CUDA error: batch_size={batch_size} is too large. "
                                f"Please reduce --batch_size (suggested: 8 or smaller per GPU)."
                            ) from e
                        else:
                            # Re-raise other errors
                            raise
                    
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
            
            result_entry = {
                "top_k": k,
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
            }
            
            results_data.append(result_entry)
            
            log.info(f"Top-K={k}: Accuracy={overall_accuracy:.4f} ({len(all_scores)} samples)")
            log.info(f"Top-K={k}: Duration={config_duration_minutes:.2f} minutes ({config_duration_hours:.2f} hours)")
            
            # Save result for this top_k immediately (incremental save)
            # This ensures we don't lose progress if the script is interrupted
            # Note: In distributed mode, each rank saves its own partial results, which will be merged later
            if not self.is_distributed or self.rank == 0:
                single_config_result = {
                    "summary": [{
                        "top_k": result_entry["top_k"],
                        "accuracy": result_entry["accuracy"],
                        "num_samples": result_entry["num_samples"],
                        "std": result_entry["std"],
                        "duration_seconds": result_entry["duration_seconds"],
                        "duration_minutes": result_entry["duration_minutes"],
                        "duration_hours": result_entry["duration_hours"],
                        "start_time": result_entry["start_time"],
                        "end_time": result_entry["end_time"],
                    }],
                    "all_samples": result_entry["per_sample_scores"],
                    "config": {
                        "dataset_name": dataset_name,
                        "split": split,
                        "batch_size": batch_size,
                        "max_new_tokens": max_new_tokens,
                        "top_k": k,
                        "batch_size_used": current_batch_size,
                        "world_size": self.world_size,
                        "rank": self.rank,  # Note: This is per-rank result before merging
                        "duration_seconds": result_entry["duration_seconds"],
                        "duration_minutes": result_entry["duration_minutes"],
                        "duration_hours": result_entry["duration_hours"],
                        "start_time": result_entry["start_time"],
                        "end_time": result_entry["end_time"],
                    }
                }
                
                # Save individual result file for this top_k
                # In distributed mode, add rank suffix to avoid conflicts
                if self.is_distributed:
                    individual_filename = f"exp2_accuracy_results_top_k_{k}_rank{self.rank}.json"
                else:
                    individual_filename = f"exp2_accuracy_results_top_k_{k}.json"
                
                self.save_results(single_config_result, individual_filename)
                log.info(f"Saved individual result for top_k={k} to {individual_filename}")
        
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
                
                # Recompute accuracy for each top_k from merged results
                final_results_data = []
                for k in top_k_values:
                    all_scores_for_config = []
                    all_predictions_for_config = []
                    # Collect time info from all ranks (use max duration as total time)
                    max_duration_seconds = 0.0
                    earliest_start_time = None
                    latest_end_time = None
                    
                    for result in merged_results_data:
                        if result["top_k"] == k:
                            all_scores_for_config.extend([s["score"] for s in result["per_sample_scores"]])
                            all_predictions_for_config.extend(result["per_sample_scores"])
                            # Track time info (use max duration across ranks)
                            if "duration_seconds" in result:
                                max_duration_seconds = max(max_duration_seconds, result["duration_seconds"])
                            if "start_time" in result:
                                if earliest_start_time is None or result["start_time"] < earliest_start_time:
                                    earliest_start_time = result["start_time"]
                            if "end_time" in result:
                                if latest_end_time is None or result["end_time"] > latest_end_time:
                                    latest_end_time = result["end_time"]
                    
                    overall_accuracy = np.mean(all_scores_for_config) if all_scores_for_config else 0.0
                    
                    # Calculate duration from collected time info
                    max_duration_minutes = max_duration_seconds / 60.0 if max_duration_seconds > 0 else 0.0
                    max_duration_hours = max_duration_seconds / 3600.0 if max_duration_seconds > 0 else 0.0
                    
                    final_results_data.append({
                        "top_k": k,
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
                
                # Save merged results for each top_k (rank 0 only, after merging)
                for merged_result in final_results_data:
                    merged_config_result = {
                        "summary": [{
                            "top_k": merged_result["top_k"],
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
                            "top_k": merged_result["top_k"],
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
                    individual_filename = f"exp2_accuracy_results_top_k_{merged_result['top_k']}.json"
                    self.save_results(merged_config_result, individual_filename)
                    log.info(f"Saved merged result for top_k={merged_result['top_k']} to {individual_filename}")
            else:
                dist.gather_object(results_data, None, dst=0)
                dist.destroy_process_group()
                return
        
        # Save results (only on rank 0)
        summary = []
        all_samples = []
        
        for config_result in results_data:
            summary_entry = {
                "top_k": config_result["top_k"],
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
                "top_k_values": top_k_values,
                "world_size": self.world_size,
            }
        }
        
        self.save_results(final_results, "exp2_accuracy_results.json")
        log.info(f"Results saved. Total samples: {len(all_samples)}")
        
        # Cleanup distributed process group
        if self.is_distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Exp2 Accuracy: MoE Top-K Analysis")
    parser.add_argument("--model_path", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results/profiling/exp2_accuracy")
    parser.add_argument("--dataset_name", type=str, default="coco_2014_vqa")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Base batch size per GPU (default: 8). If auto_adjust_batch_size=True, this is the starting point.")
    parser.add_argument("--max_new_tokens", type=int, default=16,
                       help="Maximum tokens to generate (default: 16, optimized for VQA)")
    parser.add_argument("--top_k_values", type=int, nargs="+", default=None,
                       help="List of top_k values to test (default: [4, 8, 12, ..., 64] - 16 groups)")
    parser.add_argument("--auto_adjust_batch_size", action="store_true", default=True,
                       help="Automatically adjust batch size for each top_k to avoid OOM (default: True)")
    parser.add_argument("--no_auto_adjust_batch_size", dest="auto_adjust_batch_size", action="store_false",
                       help="Disable automatic batch size adjustment")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = Exp2AccuracyExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    experiment.run(
        dataset_name=args.dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        top_k_values=args.top_k_values,
        auto_adjust_batch_size=args.auto_adjust_batch_size,
    )


if __name__ == "__main__":
    main()

