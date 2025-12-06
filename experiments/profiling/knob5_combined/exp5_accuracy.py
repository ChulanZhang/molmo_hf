"""
Exp5 Accuracy Measurement: Combined Control Knobs Analysis
Tests combinations of max_crops, top_k, and num_active_blocks.
Multi-GPU, large batch size, measure accuracy on full VQA v2 validation set.
"""

import argparse
import logging
import sys
import os
import time
from typing import Dict, List, Any, Optional, Tuple
import itertools

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
from molmo.models.modeling_molmoe import MolmoeSparseMoeBlock
from molmo.torch_util import get_world_size, get_global_rank, get_local_rank

# Import BlockMaskWrapper from exp3
sys.path.append(os.path.join(os.path.dirname(__file__), "../knob3_layers"))
from exp_transformer_blocks_mask import BlockMaskWrapper

log = logging.getLogger(__name__)


class Exp5AccuracyExperiment(BaseExperiment):
    """
    Exp5 Accuracy: Measure accuracy for different combinations of:
    - max_crops (vision tokens)
    - top_k (MoE expert selection)
    - num_active_blocks (transformer depth)
    
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
    
    def _generate_sparse_combinations(
        self,
        max_crops_list: List[int],
        top_k_list: List[int],
        num_active_blocks_list: List[int],
        sampling_strategy: str = "stratified",
        max_combinations: int = 50,
    ) -> List[Tuple[int, int, int]]:
        """
        Generate sparse combinations of the three knobs.
        
        Args:
            max_crops_list: List of max_crops values
            top_k_list: List of top_k values
            num_active_blocks_list: List of num_active_blocks values
            sampling_strategy: Strategy for sparse sampling
                - "full": All combinations (no sparsification)
                - "stratified": Select key values from each dimension
                - "boundary": Select min, max, and middle values
                - "lhs": Latin Hypercube Sampling (if available)
                - "orthogonal": Orthogonal design
            max_combinations: Maximum number of combinations to generate
        
        Returns:
            List of (max_crops, top_k, num_active_blocks) tuples
        """
        if sampling_strategy == "full":
            # Full grid search
            combinations = list(itertools.product(max_crops_list, top_k_list, num_active_blocks_list))
            log.info(f"Full grid search: {len(combinations)} combinations")
            return combinations
        
        elif sampling_strategy == "stratified":
            # Stratified sampling: select key values from each dimension
            # For max_crops [2,4,6,8,10,12]: select [2, 6, 12] (min, middle, max)
            # For top_k [4,8,12,16,20,24,28,32]: select [4, 16, 32] (min, middle, max)
            # For blocks [8,10,12,14,16]: select [8, 12, 16] (min, middle, max)
            sparse_max_crops = [max_crops_list[0], max_crops_list[len(max_crops_list)//2], max_crops_list[-1]]
            sparse_top_k = [top_k_list[0], top_k_list[len(top_k_list)//2], top_k_list[-1]]
            sparse_blocks = [num_active_blocks_list[0], num_active_blocks_list[len(num_active_blocks_list)//2], num_active_blocks_list[-1]]
            
            combinations = list(itertools.product(sparse_max_crops, sparse_top_k, sparse_blocks))
            log.info(f"Stratified sampling: {len(combinations)} combinations")
            log.info(f"  max_crops: {sparse_max_crops}, top_k: {sparse_top_k}, blocks: {sparse_blocks}")
            return combinations
        
        elif sampling_strategy == "boundary":
            # Boundary sampling: min, max, and one middle value
            # More comprehensive than stratified
            sparse_max_crops = [max_crops_list[0], max_crops_list[len(max_crops_list)//2], max_crops_list[-1]]
            # For top_k, select more values: min, 1/4, middle, 3/4, max
            sparse_top_k = [
                top_k_list[0],
                top_k_list[len(top_k_list)//4],
                top_k_list[len(top_k_list)//2],
                top_k_list[3*len(top_k_list)//4],
                top_k_list[-1]
            ]
            sparse_blocks = [num_active_blocks_list[0], num_active_blocks_list[len(num_active_blocks_list)//2], num_active_blocks_list[-1]]
            
            combinations = list(itertools.product(sparse_max_crops, sparse_top_k, sparse_blocks))
            log.info(f"Boundary sampling: {len(combinations)} combinations")
            log.info(f"  max_crops: {sparse_max_crops}, top_k: {sparse_top_k}, blocks: {sparse_blocks}")
            return combinations
        
        elif sampling_strategy == "balanced":
            # Balanced sampling: ensure each dimension is well-represented
            # Select values that evenly cover each dimension
            # max_crops: [2, 6, 12] (3 values)
            # top_k: [4, 12, 20, 32] (4 values, including default 8 if close)
            # blocks: [8, 12, 16] (3 values)
            sparse_max_crops = [2, 6, 12]
            sparse_top_k = [4, 12, 20, 32]  # Skip 8, use 12 as close alternative
            sparse_blocks = [8, 12, 16]
            
            combinations = list(itertools.product(sparse_max_crops, sparse_top_k, sparse_blocks))
            log.info(f"Balanced sampling: {len(combinations)} combinations")
            log.info(f"  max_crops: {sparse_max_crops}, top_k: {sparse_top_k}, blocks: {sparse_blocks}")
            return combinations
        
        elif sampling_strategy == "custom_sparse":
            # Custom sparse: user-specified reduction
            # max_crops: [2, 6, 12] (every 2nd value: 2, 6, 12)
            # top_k: [4, 12, 20, 32] (every 2nd value: 4, 12, 20, 32)
            # blocks: [8, 12, 16] (every 2nd value: 8, 12, 16)
            sparse_max_crops = max_crops_list[::2]  # [2, 6, 12]
            sparse_top_k = top_k_list[::2]  # [4, 12, 20, 32]
            sparse_blocks = num_active_blocks_list[::2]  # [8, 12, 16]
            
            combinations = list(itertools.product(sparse_max_crops, sparse_top_k, sparse_blocks))
            log.info(f"Custom sparse (every 2nd value): {len(combinations)} combinations")
            log.info(f"  max_crops: {sparse_max_crops}, top_k: {sparse_top_k}, blocks: {sparse_blocks}")
            return combinations
        
        elif sampling_strategy == "lhs":
            # Latin Hypercube Sampling (simplified version)
            # Randomly sample combinations ensuring each dimension is well-covered
            np.random.seed(42)  # For reproducibility
            n_samples = min(max_combinations, len(max_crops_list) * len(top_k_list) * len(num_active_blocks_list) // 4)
            
            combinations = []
            for _ in range(n_samples):
                max_crops = np.random.choice(max_crops_list)
                top_k = np.random.choice(top_k_list)
                num_blocks = np.random.choice(num_active_blocks_list)
                combinations.append((max_crops, top_k, num_blocks))
            
            # Remove duplicates
            combinations = list(set(combinations))
            log.info(f"Latin Hypercube Sampling: {len(combinations)} unique combinations")
            return combinations
        
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    def _set_max_crops(self, max_crops: int):
        """Set max_crops in model config."""
        self.model.config.max_crops = max_crops
        log.info(f"Set max_crops={max_crops}")
    
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
    
    def _set_active_blocks(self, num_active: int, total_blocks: int) -> Tuple[torch.Tensor, List[int]]:
        """
        Set active transformer blocks (always keep first 4, then randomly select additional).
        
        Returns:
            (block_mask, block_indices)
        """
        num_fixed_blocks = 4
        fixed_indices = list(range(num_fixed_blocks))
        
        if num_active <= num_fixed_blocks:
            block_indices = list(range(num_active))
        else:
            num_additional = num_active - num_fixed_blocks
            remaining_indices = list(range(num_fixed_blocks, total_blocks))
            
            # Use deterministic seed based on num_active for reproducibility
            rng = np.random.RandomState(seed=42 + num_active * 1000)
            additional_indices = rng.choice(
                remaining_indices,
                size=num_additional,
                replace=False
            ).tolist()
            additional_indices = sorted(additional_indices)
            block_indices = fixed_indices + additional_indices
        
        block_mask = torch.zeros(total_blocks, dtype=torch.bool)
        for idx in block_indices:
            block_mask[idx] = True
        
        return block_mask, block_indices
    
    def _estimate_batch_size_for_config(
        self,
        max_crops: int,
        top_k: int,
        num_active_blocks: int,
        total_blocks: int,
        base_batch_size: int,
    ) -> int:
        """
        Estimate appropriate batch size for a given configuration.
        
        Combines heuristics from exp1, exp2, and exp3.
        """
        # Heuristic from exp1: max_crops scaling
        if max_crops <= 1:
            crops_factor = 1.0
        elif max_crops <= 2:
            crops_factor = 1.0
        elif max_crops <= 4:
            crops_factor = 0.8
        elif max_crops <= 6:
            crops_factor = 0.6
        elif max_crops <= 8:
            crops_factor = 0.5
        elif max_crops <= 10:
            crops_factor = 0.4
        elif max_crops <= 12:
            crops_factor = 0.3
        else:
            crops_factor = 0.25
        
        # Heuristic from exp2: top_k scaling (less aggressive)
        if top_k <= 4:
            topk_factor = 1.0
        elif top_k <= 8:
            topk_factor = 0.9
        elif top_k <= 16:
            topk_factor = 0.8
        elif top_k <= 32:
            topk_factor = 0.7
        else:
            topk_factor = 0.6
        
        # Heuristic from exp3: blocks scaling
        ratio = num_active_blocks / total_blocks if total_blocks > 0 else 1.0
        if ratio <= 0.25:
            blocks_factor = 1.0
        elif ratio <= 0.5:
            blocks_factor = 0.9
        elif ratio <= 0.75:
            blocks_factor = 0.8
        else:
            blocks_factor = 0.7
        
        # Combine factors (use minimum to be conservative)
        combined_factor = min(crops_factor, topk_factor, blocks_factor)
        
        estimated_batch_size = max(1, int(base_batch_size * combined_factor))
        return estimated_batch_size
    
    def _find_optimal_batch_size(
        self,
        max_crops: int,
        top_k: int,
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
            max_crops: Current max_crops value
            top_k: Current top_k value
            num_active_blocks: Current num_active_blocks value
            total_blocks: Total number of blocks
            initial_batch_size: Starting batch size to try (base value)
            dataloader_factory: Function that creates dataloader given batch_size
            max_attempts: Maximum number of attempts to find working batch size
            
        Returns:
            Optimal batch size that works without OOM or CUDA errors
        """
        import torch
        
        # Estimate initial batch size
        estimated_batch_size = self._estimate_batch_size_for_config(
            max_crops=max_crops,
            top_k=top_k,
            num_active_blocks=num_active_blocks,
            total_blocks=total_blocks,
            base_batch_size=initial_batch_size,
        )
        
        # Determine max allowed batch size based on configuration
        # For very small configs, allow exceeding initial
        if max_crops <= 2 and top_k <= 4 and num_active_blocks <= total_blocks * 0.5:
            max_allowed_batch_size = initial_batch_size * 2
        elif max_crops <= 4 and top_k <= 8:
            max_allowed_batch_size = int(initial_batch_size * 1.5)
        else:
            max_allowed_batch_size = initial_batch_size
        
        current_batch_size = min(estimated_batch_size, max_allowed_batch_size)
        
        log.info(f"Finding optimal batch size for max_crops={max_crops}, top_k={top_k}, "
                f"num_active_blocks={num_active_blocks}/{total_blocks} "
                f"(estimated: {estimated_batch_size}, starting with: {current_batch_size}, "
                f"max allowed: {max_allowed_batch_size})...")
        
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
                log.info(f"✓ Batch size {current_batch_size} works for "
                        f"max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active_blocks}")
                
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
                    log.warning(f"✗ Batch size {old_batch_size} caused {error_type} for "
                              f"max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active_blocks}, "
                              f"trying {current_batch_size}...")
                    
                    # If we've narrowed down to the same size, stop
                    if current_batch_size == old_batch_size:
                        break
                else:
                    # Other error, re-raise
                    raise
        
        # Return the last working size, or the smallest we tried
        if min_working is not None:
            log.info(f"Using batch size {min_working} for "
                    f"max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active_blocks}")
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
        max_crops_list: List[int] = None,
        top_k_list: List[int] = None,
        num_active_blocks_list: List[int] = None,
        sampling_strategy: str = "balanced",
        auto_adjust_batch_size: bool = True,
    ):
        """
        Run Exp5 accuracy measurement with combined knobs.
        
        Args:
            dataset_name: Dataset name (default: "coco_2014_vqa")
            split: Dataset split (default: "validation")
            batch_size: Base batch size per GPU (default: 8)
            max_new_tokens: Maximum tokens to generate
            max_crops_list: List of max_crops values (default: [2, 4, 6, 8, 10, 12])
            top_k_list: List of top_k values (default: [4, 8, 12, 16, 20, 24, 28, 32])
            num_active_blocks_list: List of num_active_blocks values (default: [8, 10, 12, 14, 16])
            sampling_strategy: Sparse sampling strategy (default: "balanced")
            auto_adjust_batch_size: If True, automatically adjust batch size for each config
        """
        # Default values
        if max_crops_list is None:
            max_crops_list = [2, 4, 6, 8, 10, 12]
        if top_k_list is None:
            top_k_list = [4, 8, 12, 16, 20, 24, 28, 32]
        if num_active_blocks_list is None:
            total_blocks = len(self.model.model.transformer.blocks)
            num_active_blocks_list = list(range(8, total_blocks + 1, 2))  # [8, 10, 12, 14, 16]
            num_active_blocks_list = [n for n in num_active_blocks_list if n <= total_blocks]
        
        # Get total blocks
        total_blocks = len(self.model.model.transformer.blocks)
        log.info(f"Total transformer blocks: {total_blocks}")
        
        # If all lists have only one value, use that single combination directly
        # This allows running a single experiment configuration
        if (len(max_crops_list) == 1 and len(top_k_list) == 1 and 
            len(num_active_blocks_list) == 1):
            combinations = [(max_crops_list[0], top_k_list[0], num_active_blocks_list[0])]
            log.info(f"Single configuration specified: max_crops={max_crops_list[0]}, "
                    f"top_k={top_k_list[0]}, num_active_blocks={num_active_blocks_list[0]}")
        else:
            # Generate sparse combinations
            combinations = self._generate_sparse_combinations(
                max_crops_list=max_crops_list,
                top_k_list=top_k_list,
                num_active_blocks_list=num_active_blocks_list,
                sampling_strategy=sampling_strategy,
            )
        
        log.info(f"Testing {len(combinations)} combinations")
        log.info(f"Sampling strategy: {sampling_strategy}")
        log.info(f"Base batch size per GPU: {batch_size}, Total GPUs: {self.world_size}")
        if auto_adjust_batch_size:
            log.info(f"Auto-adjusting batch size enabled: will optimize for each configuration")
        else:
            log.info(f"Global batch size: {batch_size * self.world_size}")
        
        experiment_start_time = time.time()
        experiment_start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(experiment_start_time))
        log.info(f"Experiment started at: {experiment_start_time_str}")
        
        results_data = []
        mask_wrapper = None
        
        # Import data loading modules once
        from molmo.data import get_dataset_by_name
        from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        
        dataset = get_dataset_by_name(dataset_name, split=split)
        
        try:
            for config_idx, (max_crops, top_k, num_active) in enumerate(combinations):
                log.info(f"=" * 80)
                log.info(f"Configuration {config_idx + 1}/{len(combinations)}: "
                        f"max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active}/{total_blocks}")
                log.info(f"=" * 80)
                
                config_start_time = time.time()
                config_start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_start_time))
                log.info(f"Configuration start time: {config_start_time_str}")
                
                # Set max_crops
                self._set_max_crops(max_crops)
                
                # Set top_k
                self._set_top_k(top_k)
                
                # Set active blocks
                block_mask, block_indices = self._set_active_blocks(num_active, total_blocks)
                log.info(f"Active block indices: {block_indices}")
                
                # Apply block mask
                if mask_wrapper is not None:
                    mask_wrapper.remove()
                mask_wrapper = BlockMaskWrapper(self.model.model, block_mask)
                mask_wrapper.apply()
                
                # Determine batch size for this configuration
                if auto_adjust_batch_size:
                    # Create a factory function for dataloader creation
                    def create_dataloader(bs):
                        mm_preprocessor = MultiModalPreprocessor(
                            tokenizer=self.tokenizer,
                            crop_mode=self.model.config.crop_mode,
                            max_crops=max_crops,
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
                                max_crops=max_crops
                            ),
                            num_workers=4,
                            pin_memory=True,
                            prefetch_factor=2,
                            persistent_workers=True,
                        )
                    
                    # Find optimal batch size dynamically
                    optimal_batch_size = self._find_optimal_batch_size(
                        max_crops=max_crops,
                        top_k=top_k,
                        num_active_blocks=num_active,
                        total_blocks=total_blocks,
                        initial_batch_size=batch_size,
                        dataloader_factory=create_dataloader,
                    )
                    current_batch_size = optimal_batch_size
                else:
                    current_batch_size = batch_size
                
                log.info(f"Using batch size: {current_batch_size}")
                
                # Build dataloader
                mm_preprocessor = MultiModalPreprocessor(
                    tokenizer=self.tokenizer,
                    crop_mode=self.model.config.crop_mode,
                    max_crops=max_crops,  # Use the configured max_crops
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
                
                dataloader = torch.utils.data.DataLoader(
                    det_dataset,
                    batch_size=current_batch_size,
                    shuffle=shuffle,
                    sampler=sampler,
                    collate_fn=MMCollator(
                        max_sequence_length=1536,
                        include_metadata=True,
                        pad=True,
                        max_crops=max_crops  # Use the configured max_crops
                    ),
                    num_workers=4,
                    pin_memory=True,
                    prefetch_factor=2,
                    persistent_workers=True,
                )
                
                all_scores = []
                all_predictions = []
                
                log.info(f"Measuring accuracy for max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active}...")
                
                with torch.inference_mode():
                    for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
                        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        from transformers import GenerationConfig
                        
                        eos_token_id = self.tokenizer.eos_token_id
                        if eos_token_id is None:
                            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
                        
                        pad_token_id = self.tokenizer.pad_token_id
                        if pad_token_id is None:
                            pad_token_id = getattr(self.model.config, 'pad_token_id', None)
                        
                        vqa_max_tokens = min(max_new_tokens, 16)
                        
                        generation_config = GenerationConfig(
                            max_new_tokens=vqa_max_tokens,
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=eos_token_id,
                            pad_token_id=pad_token_id,
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
                                log.error(f"Configuration: max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active}")
                                log.error(f"Batch size: {current_batch_size} per GPU")
                                raise RuntimeError(
                                    f"CUDA error: batch_size={current_batch_size} is too large. "
                                    f"Please reduce --batch_size or enable auto_adjust_batch_size."
                                ) from e
                            else:
                                raise
                        
                        batch_accuracy = self.compute_accuracy(
                            batch=batch,
                            predictions=outputs,
                            metric_name="vqa_score",
                        )
                        
                        all_scores.extend([s["score"] for s in batch_accuracy["per_sample_scores"]])
                        all_predictions.extend(batch_accuracy["per_sample_scores"])
                
                config_end_time = time.time()
                config_end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_end_time))
                config_duration_seconds = config_end_time - config_start_time
                config_duration_minutes = config_duration_seconds / 60.0
                config_duration_hours = config_duration_seconds / 3600.0
                
                log.info(f"Configuration end time: {config_end_time_str}")
                log.info(f"Configuration duration: {config_duration_minutes:.2f} minutes ({config_duration_hours:.2f} hours)")
                
                overall_accuracy = np.mean(all_scores) if all_scores else 0.0
                
                result_entry = {
                    "max_crops": max_crops,
                    "top_k": top_k,
                    "num_active_blocks": num_active,
                    "num_total_blocks": total_blocks,
                    "active_block_indices": block_indices,
                    "accuracy": float(overall_accuracy),
                    "num_samples": len(all_scores),
                    "std": float(np.std(all_scores)) if all_scores else 0.0,
                    "per_sample_scores": all_predictions,
                    "batch_size_used": current_batch_size,
                    "duration_seconds": float(config_duration_seconds),
                    "duration_minutes": float(config_duration_minutes),
                    "duration_hours": float(config_duration_hours),
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_start_time)),
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_end_time)),
                }
                
                results_data.append(result_entry)
                
                log.info(f"Configuration {config_idx + 1}/{len(combinations)}: "
                        f"max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active}: "
                        f"Accuracy={overall_accuracy:.4f} ({len(all_scores)} samples)")
                log.info(f"Duration: {config_duration_minutes:.2f} minutes ({config_duration_hours:.2f} hours)")
                
                # Save result for this configuration immediately (incremental save)
                if not self.is_distributed or self.rank == 0:
                    single_config_result = {
                        "summary": [{
                            "max_crops": result_entry["max_crops"],
                            "top_k": result_entry["top_k"],
                            "num_active_blocks": result_entry["num_active_blocks"],
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
                            "top_k": top_k,
                            "num_active_blocks": num_active,
                            "num_total_blocks": total_blocks,
                            "active_block_indices": block_indices,
                            "batch_size_used": current_batch_size,
                            "world_size": self.world_size,
                            "rank": self.rank,
                            "config_index": config_idx,
                            "sampling_strategy": sampling_strategy,
                        }
                    }
                    
                    # Create descriptive filename with configuration parameters
                    if self.is_distributed:
                        individual_filename = f"exp5_accuracy_results_crops{max_crops}_topk{top_k}_blocks{num_active}_rank{self.rank}.json"
                    else:
                        individual_filename = f"exp5_accuracy_results_crops{max_crops}_topk{top_k}_blocks{num_active}.json"
                    
                    self.save_results(single_config_result, individual_filename)
                    log.info(f"Saved individual result for config {config_idx} (max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active}) to {individual_filename}")
        
        finally:
            if mask_wrapper is not None:
                mask_wrapper.remove()
                if not self.is_distributed or self.rank == 0:
                    log.info("Restored original forward method")
        
        # Gather results from all ranks if distributed
        if self.is_distributed:
            # Add barrier to ensure all ranks are synchronized before gather
            try:
                log.info(f"Rank {self.rank}: Waiting for all ranks to finish before gathering results...")
                dist.barrier()
                log.info(f"Rank {self.rank}: All ranks synchronized, starting gather...")
            except Exception as e:
                log.error(f"Rank {self.rank}: Barrier failed: {e}")
                # Continue anyway - individual rank results are already saved
            
            try:
                if self.rank == 0:
                    gathered_results = [None] * self.world_size
                    dist.gather_object(results_data, gathered_results, dst=0)
                    
                    # Merge results from all ranks
                    merged_results_data = []
                    for rank_results in gathered_results:
                        if rank_results is not None:
                            merged_results_data.extend(rank_results)
                    
                    # Group by configuration and merge
                    config_dict = {}
                    for result in merged_results_data:
                        config_key = (result["max_crops"], result["top_k"], result["num_active_blocks"])
                        if config_key not in config_dict:
                            config_dict[config_key] = {
                                "max_crops": result["max_crops"],
                                "top_k": result["top_k"],
                                "num_active_blocks": result["num_active_blocks"],
                                "num_total_blocks": result["num_total_blocks"],
                                "active_block_indices": result.get("active_block_indices", []),
                                "per_sample_scores": [],
                                "all_scores": [],
                            }
                        
                        config_dict[config_key]["per_sample_scores"].extend(result["per_sample_scores"])
                        config_dict[config_key]["all_scores"].extend([s["score"] for s in result["per_sample_scores"]])
                    
                    # Compute merged statistics
                    final_results_data = []
                    for config_key, config_data in config_dict.items():
                        overall_accuracy = np.mean(config_data["all_scores"]) if config_data["all_scores"] else 0.0
                        
                        final_results_data.append({
                            "max_crops": config_data["max_crops"],
                            "top_k": config_data["top_k"],
                            "num_active_blocks": config_data["num_active_blocks"],
                            "num_total_blocks": config_data["num_total_blocks"],
                            "active_block_indices": config_data["active_block_indices"],
                            "accuracy": float(overall_accuracy),
                            "num_samples": len(config_data["all_scores"]),
                            "std": float(np.std(config_data["all_scores"])) if config_data["all_scores"] else 0.0,
                            "per_sample_scores": config_data["per_sample_scores"],
                        })
                    
                    results_data = final_results_data
                    
                    # Save merged results for each configuration (rank 0 only, after merging)
                    for merged_result in final_results_data:
                        merged_max_crops = merged_result["max_crops"]
                        merged_top_k = merged_result["top_k"]
                        merged_num_active = merged_result["num_active_blocks"]
                        
                        merged_config_result = {
                            "summary": [{
                                "max_crops": merged_max_crops,
                                "top_k": merged_top_k,
                                "num_active_blocks": merged_num_active,
                                "accuracy": merged_result["accuracy"],
                                "num_samples": merged_result["num_samples"],
                                "std": merged_result["std"],
                            }],
                            "all_samples": merged_result["per_sample_scores"],
                            "config": {
                                "dataset_name": dataset_name,
                                "split": split,
                                "batch_size": batch_size,
                                "max_new_tokens": max_new_tokens,
                                "max_crops": merged_max_crops,
                                "top_k": merged_top_k,
                                "num_active_blocks": merged_num_active,
                                "num_total_blocks": total_blocks,
                                "active_block_indices": merged_result.get("active_block_indices", []),
                                "world_size": self.world_size,
                                "note": "Merged results from all ranks",
                            }
                        }
                        
                        # Save merged result (overwrites individual rank files)
                        merged_filename = f"exp5_accuracy_results_crops{merged_max_crops}_topk{merged_top_k}_blocks{merged_num_active}.json"
                        self.save_results(merged_config_result, merged_filename)
                        log.info(f"Saved merged result for max_crops={merged_max_crops}, top_k={merged_top_k}, "
                                f"num_active_blocks={merged_num_active} to {merged_filename}")
                else:
                    dist.gather_object(results_data, None, dst=0)
                    log.info(f"Rank {self.rank}: Successfully sent results to rank 0")
            except Exception as e:
                log.error(f"Rank {self.rank}: Gather operation failed: {e}")
                log.warning(f"Rank {self.rank}: Individual results are already saved, continuing without merge...")
                # Set results_data to empty to avoid errors in final save
                if self.rank == 0:
                    results_data = []
        
        # Save final results
        if not self.is_distributed or self.rank == 0:
            summary = []
            all_samples = []
            
            for config_result in results_data:
                summary_entry = {
                    "max_crops": config_result["max_crops"],
                    "top_k": config_result["top_k"],
                    "num_active_blocks": config_result["num_active_blocks"],
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
                "config": {
                    "dataset_name": dataset_name,
                    "split": split,
                    "batch_size": batch_size,
                    "max_new_tokens": max_new_tokens,
                    "max_crops_list": max_crops_list,
                    "top_k_list": top_k_list,
                    "num_active_blocks_list": num_active_blocks_list,
                    "sampling_strategy": sampling_strategy,
                    "num_combinations": len(combinations),
                    "world_size": self.world_size,
                }
            }
            
            self.save_results(final_results, "exp5_accuracy_results.json")
            log.info(f"Results saved. Total samples: {len(all_samples)}")
            
            experiment_end_time = time.time()
            experiment_end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(experiment_end_time))
            experiment_duration_seconds = experiment_end_time - experiment_start_time
            experiment_duration_minutes = experiment_duration_seconds / 60.0
            experiment_duration_hours = experiment_duration_seconds / 3600.0
            log.info(f"Experiment ended at: {experiment_end_time_str}")
            log.info(f"Total experiment duration: {experiment_duration_minutes:.2f} minutes ({experiment_duration_hours:.2f} hours)")
        
        # Cleanup distributed process group
        if self.is_distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Exp5 Accuracy: Combined Control Knobs Analysis")
    parser.add_argument("--model_path", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results/profiling/exp5_accuracy")
    parser.add_argument("--dataset_name", type=str, default="coco_2014_vqa")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Base batch size per GPU (default: 8)")
    parser.add_argument("--max_new_tokens", type=int, default=16,
                       help="Maximum tokens to generate (default: 16, optimized for VQA)")
    parser.add_argument("--max_crops", type=int, nargs="+", default=None,
                       help="List of max_crops values (default: [2, 4, 6, 8, 10])")
    parser.add_argument("--top_k", type=int, nargs="+", default=None,
                       help="List of top_k values (default: [4, 8, 12, 16, 20, 24, 28, 32])")
    parser.add_argument("--num_active_blocks", type=int, nargs="+", default=None,
                       help="List of num_active_blocks values (default: [8, 10, 12, 14, 16])")
    parser.add_argument("--sampling_strategy", type=str, default="balanced",
                       choices=["full", "stratified", "boundary", "balanced", "custom_sparse", "lhs"],
                       help="Sparse sampling strategy (default: balanced)")
    parser.add_argument("--auto_adjust_batch_size", action="store_true", default=True,
                       help="Automatically adjust batch size for each configuration (default: True)")
    parser.add_argument("--no_auto_adjust_batch_size", dest="auto_adjust_batch_size", action="store_false",
                       help="Disable automatic batch size adjustment")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = Exp5AccuracyExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    experiment.run(
        dataset_name=args.dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_crops_list=args.max_crops,
        top_k_list=args.top_k,
        num_active_blocks_list=args.num_active_blocks,
        sampling_strategy=args.sampling_strategy,
        auto_adjust_batch_size=args.auto_adjust_batch_size,
    )


if __name__ == "__main__":
    main()

