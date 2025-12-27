"""
Exp1 Accuracy Measurement: Context Scaling (Vision Tokens)
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
from molmo.torch_util import get_world_size, get_global_rank, get_local_rank, is_distributed

log = logging.getLogger(__name__)


class Exp1AccuracyExperiment(BaseExperiment):
    """
    Exp1 Accuracy: Measure accuracy for different vision token configurations.
    Uses full VQA v2 validation set with different max_crops settings.
    
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
        # Check if RANK and WORLD_SIZE are set (torchrun sets these)
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            if not dist.is_initialized():
                # Initialize process group
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
    
    def _estimate_batch_size_for_max_crops(self, max_crops: int, base_batch_size: int) -> int:
        """
        Estimate appropriate batch size for a given max_crops value.
        
        Memory usage scales roughly with:
        - Vision tokens: proportional to max_crops
        - Attention mask: (seq_len + max_new_tokens)², where seq_len includes vision tokens
        
        Heuristic: batch_size scales inversely with max_crops
        Based on user's test: max_crops=2 works with batch_size=128
        """
        # More conservative scaling based on actual measurements
        # max_crops=2: batch_size=128 works
        # So we scale more aggressively for larger max_crops
        if max_crops <= 1:
            scale_factor = 1.0  # max_crops=1 can use full batch size
        elif max_crops <= 2:
            scale_factor = 1.0
        elif max_crops <= 4:
            scale_factor = 0.8
        elif max_crops <= 6:
            scale_factor = 0.6
        elif max_crops <= 8:
            scale_factor = 0.5
        elif max_crops <= 10:
            scale_factor = 0.4
        elif max_crops <= 12:
            scale_factor = 0.3  # More conservative for max_crops=12
        else:
            scale_factor = 0.25  # Very conservative for max_crops=13
        
        estimated_batch_size = max(1, int(base_batch_size * scale_factor))
        return estimated_batch_size
    
    def _find_optimal_batch_size(
        self, 
        max_crops: int, 
        initial_batch_size: int,
        dataloader_factory,
        max_attempts: int = 5,
    ) -> int:
        """
        Dynamically find the optimal batch size that doesn't cause OOM.
        
        Uses binary search: start with estimated size, reduce if OOM, increase if safe.
        
        Args:
            max_crops: Current max_crops value
            initial_batch_size: Starting batch size to try (base value)
            dataloader_factory: Function that creates dataloader given batch_size
            max_attempts: Maximum number of attempts to find working batch size
            
        Returns:
            Optimal batch size that works without OOM
        """
        import torch
        
        # For small max_crops, allow exceeding initial_batch_size to find maximum
        # For larger max_crops, use initial_batch_size as upper limit
        if max_crops <= 1:
            # max_crops=1 can support very large batch sizes
            max_allowed_batch_size = initial_batch_size * 2  # Allow up to 2x for very small max_crops
        elif max_crops <= 2:
            # max_crops=2 can support much larger batch sizes (e.g., 128)
            max_allowed_batch_size = initial_batch_size * 2  # Allow up to 2x for small max_crops
        elif max_crops <= 4:
            max_allowed_batch_size = int(initial_batch_size * 1.5)  # Allow up to 1.5x
        else:
            max_allowed_batch_size = initial_batch_size  # For larger max_crops, use initial as limit
        
        # Start with estimated batch size
        estimated_batch_size = self._estimate_batch_size_for_max_crops(max_crops, initial_batch_size)
        current_batch_size = min(estimated_batch_size, max_allowed_batch_size)
        
        log.info(f"Finding optimal batch size for max_crops={max_crops} (estimated: {estimated_batch_size}, "
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
                # Generation uses more memory than forward pass alone
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
                        # This is critical because generation allocates KV cache and attention bias
                        from transformers import GenerationConfig
                        eos_token_id = self.tokenizer.eos_token_id
                        if eos_token_id is None:
                            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
                        
                        pad_token_id = self.tokenizer.pad_token_id
                        if pad_token_id is None:
                            pad_token_id = getattr(self.model.config, 'pad_token_id', None)
                        
                        test_gen_config = GenerationConfig(
                            max_new_tokens=16,  # Use the same as actual generation
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=eos_token_id,
                            pad_token_id=pad_token_id,  # Explicitly set to avoid warning
                        )
                        
                        # Try generating to check memory (this is where OOM often occurs)
                        # We generate a few tokens to test the full generation path
                        _ = self.model.generate(
                            input_ids=test_batch["input_ids"],
                            images=test_batch.get("images"),
                            image_masks=test_batch.get("image_masks"),
                            image_input_idx=test_batch.get("image_input_idx"),
                            generation_config=test_gen_config,
                        )
                
                # If we get here, it worked!
                min_working = current_batch_size
                log.info(f"✓ Batch size {current_batch_size} works for max_crops={max_crops}")
                
                # Try to increase if we're below max_allowed and haven't hit limit
                if current_batch_size < max_allowed_batch_size and attempt < max_attempts - 1:
                    # Try increasing by 50% to see if we can use more
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
                    # OOM or CUDA configuration error occurred, reduce batch size
                    # "invalid configuration argument" often indicates batch size too large for attention
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
                    log.warning(f"✗ Batch size {old_batch_size} caused {error_type} for max_crops={max_crops}, "
                              f"trying {current_batch_size}...")
                    
                    # If we've narrowed down to the same size, stop
                    if current_batch_size == old_batch_size:
                        break
                else:
                    # Other error, re-raise
                    raise
        
        # Return the last working size, or the smallest we tried
        if min_working is not None:
            log.info(f"Using batch size {min_working} for max_crops={max_crops}")
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
        max_new_tokens: int = 128,
        max_crops_list: List[int] = None,
        start_from_max_crops: int = None,
        auto_adjust_batch_size: bool = True,
    ):
        """
        Run Exp1 accuracy measurement.
        
        Args:
            dataset_name: Dataset name (default: "coco_2014_vqa")
            split: Dataset split (default: "validation")
            batch_size: Base batch size per GPU (default: 8). 
                       If auto_adjust_batch_size=True, this is the starting point.
            max_new_tokens: Maximum tokens to generate
            max_crops_list: List of max_crops values to test (default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            start_from_max_crops: If provided, start testing from this max_crops value (inclusive)
            auto_adjust_batch_size: If True, automatically adjust batch size for each max_crops to avoid OOM
        """
        # Define max_crops configurations (range: 1 to 12)
        if max_crops_list is None:
            max_crops_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
        # Filter to start from specified max_crops
        if start_from_max_crops is not None:
            max_crops_list = [c for c in max_crops_list if c >= start_from_max_crops]
            log.info(f"Starting from max_crops={start_from_max_crops}, testing: {max_crops_list}")
        
        if not max_crops_list:
            log.error("No max_crops values to test after filtering!")
            return
        
        log.info(f"Testing {len(max_crops_list)} max_crops configurations: {max_crops_list}")
        log.info(f"Measuring accuracy on full {dataset_name}/{split} set")
        log.info(f"Base batch size per GPU: {batch_size}, Total GPUs: {self.world_size}")
        if auto_adjust_batch_size:
            log.info(f"Auto-adjusting batch size enabled: will optimize for each max_crops")
        else:
            log.info(f"Global batch size: {batch_size * self.world_size}")
        
        results_data = []
        
        # Import data loading modules once
        from molmo.data import get_dataset_by_name
        from molmo.preprocessors.multimodal_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        
        dataset = get_dataset_by_name(dataset_name, split=split)
        
        for max_crops in max_crops_list:
            log.info(f"=" * 80)
            log.info(f"Testing max_crops={max_crops}")
            log.info(f"=" * 80)
            
            # Record start time for this configuration
            config_start_time = time.time()
            
            # Temporarily modify model config
            original_max_crops = self.model.config.max_crops
            self.model.config.max_crops = max_crops
            
            # Determine batch size for this max_crops
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
                    max_crops=max_crops,
                    initial_batch_size=batch_size,
                    dataloader_factory=create_dataloader,
                )
                current_batch_size = optimal_batch_size
            else:
                current_batch_size = batch_size
            
            log.info(f"Using batch size: {current_batch_size} for max_crops={max_crops}")
            
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
                num_workers=4,  # Parallel data loading
                pin_memory=True,  # Faster CPU->GPU transfer
                prefetch_factor=2,  # Prefetch batches
                persistent_workers=True,  # Keep workers alive between epochs
            )
            
            all_scores = []
            all_predictions = []
            
            log.info(f"Measuring accuracy for max_crops={max_crops}...")
            
            # Pre-create generation config (optimization: don't recreate every batch)
            from transformers import GenerationConfig
            
            # Get EOS and PAD token IDs from tokenizer (once)
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                eos_token_id = getattr(self.model.config, 'eos_token_id', None)
            
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = getattr(self.model.config, 'pad_token_id', None)
            
            # For VQA, answers are typically very short (1-10 tokens)
            # Based on statistical analysis: P99=5, P99.9=9, max=29
            # max_new_tokens=16 covers 100% of answers (only 95 answers >16 tokens out of 2.1M)
            vqa_max_tokens = min(max_new_tokens, 16)  # Safe: 100% coverage
            
            generation_config = GenerationConfig(
                max_new_tokens=vqa_max_tokens,
                do_sample=False,  # Deterministic generation
                use_cache=True,  # Required for Molmo models
                eos_token_id=eos_token_id,  # Explicitly set EOS token for early stopping
                pad_token_id=pad_token_id,  # Explicitly set to avoid warning
            )
            
            with torch.inference_mode():
                for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
                    # Move batch to device (non_blocking for async transfer)
                    batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        outputs = self.model.generate(
                            input_ids=batch["input_ids"],
                            images=batch.get("images"),
                            image_masks=batch.get("image_masks"),
                            image_input_idx=batch.get("image_input_idx"),
                            generation_config=generation_config,
                        )
                    
                    # Debug: Check actual generated length (only log first batch)
                    if batch_idx == 0:
                        input_len = batch["input_ids"].shape[1]
                        output_len = outputs.shape[1]
                        actual_generated = output_len - input_len
                        log.info(f"Debug - Batch 0: max_new_tokens={vqa_max_tokens}, "
                                f"actual_generated={actual_generated}, "
                                f"eos_token_id={eos_token_id}")
                        if actual_generated >= vqa_max_tokens:
                            log.warning(f"Warning: Generated {actual_generated} tokens (max={vqa_max_tokens}). "
                                      f"May not be stopping early. Check EOS token.")
                    
                    # Compute accuracy for this batch
                    batch_accuracy = self.compute_accuracy(
                        batch=batch,
                        predictions=outputs,
                        metric_name="vqa_score",
                    )
                    
                    all_scores.extend([s["score"] for s in batch_accuracy["per_sample_scores"]])
                    all_predictions.extend(batch_accuracy["per_sample_scores"])
            
            # Restore original max_crops
            self.model.config.max_crops = original_max_crops
            
            # Record end time and calculate duration
            config_end_time = time.time()
            config_duration_seconds = config_end_time - config_start_time
            config_duration_hours = config_duration_seconds / 3600.0
            config_duration_minutes = config_duration_seconds / 60.0
            
            # Compute overall accuracy
            overall_accuracy = np.mean(all_scores) if all_scores else 0.0
            
            result_entry = {
                "max_crops": max_crops,
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
            
            log.info(f"max_crops={max_crops}: Accuracy={overall_accuracy:.4f} ({len(all_scores)} samples)")
            log.info(f"max_crops={max_crops}: Duration={config_duration_minutes:.2f} minutes ({config_duration_hours:.2f} hours)")
            
            # Save result for this max_crops immediately (incremental save)
            # This ensures we don't lose progress if the script is interrupted
            # Note: In distributed mode, only rank 0 will save (after gathering results)
            # For now, each rank saves its own partial results, which will be merged later
            if not self.is_distributed or self.rank == 0:
                single_config_result = {
                    "summary": [{
                        "max_crops": result_entry["max_crops"],
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
                        "max_crops": max_crops,
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
                
                # Save individual result file for this max_crops
                # In distributed mode, add rank suffix to avoid conflicts
                if self.is_distributed:
                    individual_filename = f"exp1_accuracy_results_max_crops_{max_crops}_rank{self.rank}.json"
                else:
                    individual_filename = f"exp1_accuracy_results_max_crops_{max_crops}.json"
                
                self.save_results(single_config_result, individual_filename)
                log.info(f"Saved individual result for max_crops={max_crops} to {individual_filename}")
        
        # Gather results from all ranks if distributed
        if self.is_distributed:
            # Gather all predictions from all ranks to rank 0
            if self.rank == 0:
                gathered_results = [None] * self.world_size
                dist.gather_object(results_data, gathered_results, dst=0)
                
                # Merge results from all ranks
                merged_results_data = []
                for rank_results in gathered_results:
                    if rank_results is not None:
                        merged_results_data.extend(rank_results)
                
                # Recompute accuracy for each max_crops from merged results
                final_results_data = []
                for max_crops in max_crops_list:
                    all_scores_for_config = []
                    all_predictions_for_config = []
                    # Collect time info from all ranks (use max duration as total time)
                    max_duration_seconds = 0.0
                    earliest_start_time = None
                    latest_end_time = None
                    
                    for result in merged_results_data:
                        if result["max_crops"] == max_crops:
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
                        "max_crops": max_crops,
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
                
                # Save merged results for each max_crops (rank 0 only, after merging)
                for merged_result in final_results_data:
                    merged_config_result = {
                        "summary": [{
                            "max_crops": merged_result["max_crops"],
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
                            "max_crops": merged_result["max_crops"],
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
                    individual_filename = f"exp1_accuracy_results_max_crops_{merged_result['max_crops']}.json"
                    self.save_results(merged_config_result, individual_filename)
                    log.info(f"Saved merged result for max_crops={merged_result['max_crops']} to {individual_filename}")
            else:
                dist.gather_object(results_data, None, dst=0)
                # Non-rank-0 processes exit early
                dist.destroy_process_group()
                return
        
        # Save results (only on rank 0)
        summary = []
        all_samples = []
        
        for config_result in results_data:
            summary_entry = {
                "max_crops": config_result["max_crops"],
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
                "max_crops_list": max_crops_list,
                "world_size": self.world_size,
            }
        }
        
        # Save final combined results (all max_crops configurations)
        self.save_results(final_results, "exp1_accuracy_results.json")
        log.info(f"Final combined results saved. Total samples: {len(all_samples)}")
        log.info(f"Individual results for each max_crops are also saved as: exp1_accuracy_results_max_crops_*.json")
        
        # Cleanup distributed process group
        if self.is_distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Exp1 Accuracy: Context Scaling (Vision Tokens)")
    parser.add_argument("--model_path", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results/profiling/exp1_accuracy")
    parser.add_argument("--dataset_name", type=str, default="coco_2014_vqa")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_crops_list", type=int, nargs="+", default=None,
                       help="List of max_crops values to test (default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])")
    parser.add_argument("--start_from_max_crops", type=int, default=None,
                       help="Start testing from this max_crops value (inclusive). Useful for testing larger max_crops first.")
    parser.add_argument("--auto_adjust_batch_size", action="store_true", default=True,
                       help="Automatically adjust batch size for each max_crops to avoid OOM (default: True)")
    parser.add_argument("--no_auto_adjust_batch_size", dest="auto_adjust_batch_size", action="store_false",
                       help="Disable automatic batch size adjustment")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = Exp1AccuracyExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    experiment.run(
        dataset_name=args.dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_crops_list=args.max_crops_list,
        start_from_max_crops=args.start_from_max_crops,
        auto_adjust_batch_size=args.auto_adjust_batch_size,
    )


if __name__ == "__main__":
    main()

