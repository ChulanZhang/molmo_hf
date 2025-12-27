"""
Exp6 Latency and Accuracy Measurement: Combined Control Knobs Analysis
Tests combinations of max_crops, top_k, and num_active_blocks.
Fixed batch_size=1 for accurate per-sample latency measurement.
Measures both latency and accuracy on VQA v2 validation set.
Multi-GPU support via torchrun.
"""

import argparse
import logging
import sys
import os
import time
from pathlib import Path
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
from experiments.base_experiment import BaseExperiment, get_metric_for_dataset
from molmo.models.modeling_molmoe import MolmoeSparseMoeBlock
from molmo.torch_util import get_world_size, get_global_rank, get_local_rank

# Import BlockMaskWrapper from exp3
sys.path.append(os.path.join(os.path.dirname(__file__), "../knob3_layers"))
from exp_transformer_blocks_mask import BlockMaskWrapper

log = logging.getLogger(__name__)


class Exp6LatencyExperiment(BaseExperiment):
    """
    Exp6 Latency and Accuracy: Measure latency and accuracy for different combinations of:
    - max_crops (vision tokens)
    - top_k (MoE expert selection)
    - num_active_blocks (transformer depth)
    
    Uses batch_size=1 for accurate per-sample latency measurement.
    Also computes accuracy (VQA score) for each configuration.
    
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
        dataset_name: str = "coco_2014_vqa",
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
        
        # Adjust output_dir to include dataset name to avoid conflicts between datasets
        # Always add dataset suffix (even for coco_2014_vqa) to ensure clear separation
        base_output_dir = Path(output_dir)
        dataset_suffix = dataset_name.replace("_", "-")
        output_dir = str(base_output_dir.parent / f"{base_output_dir.name}_{dataset_suffix}")
        log.info(f"Output directory adjusted to include dataset name: {output_dir}")
        
        super().__init__(
            model_path=model_path,
            device=device,
            output_dir=output_dir,
            num_warmup=num_warmup,
            hf_cache_dir=hf_cache_dir,
        )
        self.dataset_name = dataset_name
    
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
        Same as exp5.
        """
        if sampling_strategy == "full":
            # Full grid search
            combinations = list(itertools.product(max_crops_list, top_k_list, num_active_blocks_list))
            log.info(f"Full grid search: {len(combinations)} combinations")
            return combinations
        
        elif sampling_strategy == "stratified":
            # Stratified sampling: select key values from each dimension
            sparse_max_crops = [max_crops_list[0], max_crops_list[len(max_crops_list)//2], max_crops_list[-1]]
            sparse_top_k = [top_k_list[0], top_k_list[len(top_k_list)//2], top_k_list[-1]]
            sparse_blocks = [num_active_blocks_list[0], num_active_blocks_list[len(num_active_blocks_list)//2], num_active_blocks_list[-1]]
            
            combinations = list(itertools.product(sparse_max_crops, sparse_top_k, sparse_blocks))
            log.info(f"Stratified sampling: {len(combinations)} combinations")
            log.info(f"  max_crops: {sparse_max_crops}, top_k: {sparse_top_k}, blocks: {sparse_blocks}")
            return combinations
        
        elif sampling_strategy == "boundary":
            # Boundary sampling: min, max, and one middle value
            sparse_max_crops = [max_crops_list[0], max_crops_list[len(max_crops_list)//2], max_crops_list[-1]]
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
            # Select values that evenly cover each dimension from the provided lists
            # Use min, middle, max for each dimension
            if len(max_crops_list) >= 3:
                sparse_max_crops = [max_crops_list[0], max_crops_list[len(max_crops_list)//2], max_crops_list[-1]]
            else:
                sparse_max_crops = max_crops_list
            
            if len(top_k_list) >= 3:
                # Select min, middle, max
                sparse_top_k = [top_k_list[0], top_k_list[len(top_k_list)//2], top_k_list[-1]]
            elif len(top_k_list) == 2:
                sparse_top_k = top_k_list
            else:
                sparse_top_k = top_k_list
            
            if len(num_active_blocks_list) >= 3:
                sparse_blocks = [num_active_blocks_list[0], num_active_blocks_list[len(num_active_blocks_list)//2], num_active_blocks_list[-1]]
            else:
                sparse_blocks = num_active_blocks_list
            
            combinations = list(itertools.product(sparse_max_crops, sparse_top_k, sparse_blocks))
            log.info(f"Balanced sampling: {len(combinations)} combinations")
            log.info(f"  max_crops: {sparse_max_crops}, top_k: {sparse_top_k}, blocks: {sparse_blocks}")
            return combinations
        
        elif sampling_strategy == "custom_sparse":
            # Custom sparse: user-specified reduction
            sparse_max_crops = max_crops_list[::2]
            sparse_top_k = top_k_list[::2]
            sparse_blocks = num_active_blocks_list[::2]
            
            combinations = list(itertools.product(sparse_max_crops, sparse_top_k, sparse_blocks))
            log.info(f"Custom sparse (every 2nd value): {len(combinations)} combinations")
            log.info(f"  max_crops: {sparse_max_crops}, top_k: {sparse_top_k}, blocks: {sparse_blocks}")
            return combinations
        
        elif sampling_strategy == "lhs":
            # Latin Hypercube Sampling (simplified version)
            np.random.seed(42)
            n_samples = min(max_combinations, len(max_crops_list) * len(top_k_list) * len(num_active_blocks_list) // 4)
            
            combinations = []
            for _ in range(n_samples):
                max_crops = np.random.choice(max_crops_list)
                top_k = np.random.choice(top_k_list)
                num_blocks = np.random.choice(num_active_blocks_list)
                combinations.append((max_crops, top_k, num_blocks))
            
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
        Same as exp5.
        
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
    
    def run(
        self,
        dataset_name: str = "coco_2014_vqa",
        split: str = "validation",
        max_new_tokens: int = 16,
        num_samples: Optional[int] = None,
        max_crops_list: List[int] = None,
        top_k_list: List[int] = None,
        num_active_blocks_list: List[int] = None,
        sampling_strategy: str = "balanced",
    ):
        """
        Run Exp6 latency measurement with combined knobs.
        
        Args:
            dataset_name: Dataset name (default: "coco_2014_vqa")
            split: Dataset split (default: "validation")
            max_new_tokens: Maximum tokens to generate
            num_samples: Number of samples to measure (default: None = use all samples)
            max_crops_list: List of max_crops values (default: [2, 4, 6, 8, 10], max 10)
            top_k_list: List of top_k values (default: [4, 8, 12], 3 positions, max 12)
            num_active_blocks_list: List of num_active_blocks values (default: [12, 13, 14, 15, 16], min 12, max 16)
            sampling_strategy: Sparse sampling strategy (default: "balanced")
        """
        # Default values (updated based on exp3 sensitivity results)
        if max_crops_list is None:
            max_crops_list = [2, 4, 6, 8, 10]  # Max 10
        if top_k_list is None:
            top_k_list = [4, 8, 12]  # three options, max 12
        if num_active_blocks_list is None:
            total_blocks = len(self.model.model.transformer.blocks)
            # Minimum 12, maximum 16 layers (based on exp3 sensitivity results)
            num_active_blocks_list = list(range(12, total_blocks + 1))  # [12, 13, 14, 15, 16]
            num_active_blocks_list = [n for n in num_active_blocks_list if n <= total_blocks]
        
        # Get total blocks
        total_blocks = len(self.model.model.transformer.blocks)
        log.info(f"Total transformer blocks: {total_blocks}")
        
        # Generate sparse combinations
        combinations = self._generate_sparse_combinations(
            max_crops_list=max_crops_list,
            top_k_list=top_k_list,
            num_active_blocks_list=num_active_blocks_list,
            sampling_strategy=sampling_strategy,
        )
        
        log.info(f"Testing {len(combinations)} combinations")
        log.info(f"Sampling strategy: {sampling_strategy}")
        log.info(f"=" * 80)
        log.info(f"CRITICAL: Exp6 uses batch_size=1 (FIXED, cannot be changed)")
        log.info(f"  - This is essential for accurate per-sample latency measurement")
        log.info(f"  - DO NOT use auto_adjust_batch_size or any batch size optimization")
        log.info(f"  - Each sample is processed individually to measure its latency")
        log.info(f"=" * 80)
        log.info(f"Batch size: 1 (fixed for accurate per-sample latency measurement)")
        if num_samples is not None:
            log.info(f"Number of samples per rank: {num_samples // self.world_size if self.is_distributed else num_samples}")
            log.info(f"Total samples across all ranks: {num_samples} (limited)")
        else:
            log.info(f"Using ALL samples from dataset (num_samples=None)")
        
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
                # Check if result already exists (only rank 0 checks, then broadcasts to others)
                should_skip = False
                if not self.is_distributed or self.rank == 0:
                    # Check for merged result file (preferred) or individual rank file
                    merged_filename = f"exp6_latency_crops{max_crops}_topk{top_k}_blocks{num_active}.json"
                    merged_filepath = Path(self.output_dir) / merged_filename
                    
                    if merged_filepath.exists() and merged_filepath.stat().st_size > 0:
                        should_skip = True
                        log.info(f"=" * 80)
                        log.info(f"Configuration {config_idx + 1}/{len(combinations)}: "
                                f"max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active}/{total_blocks}")
                        log.info(f"✓ Result file already exists: {merged_filename}")
                        log.info(f"  Skipping this configuration...")
                        log.info(f"=" * 80)
                
                # Broadcast skip decision to all ranks in distributed mode
                if self.is_distributed:
                    skip_tensor = torch.tensor([1 if should_skip else 0], dtype=torch.int, device=self.device)
                    dist.broadcast(skip_tensor, src=0)
                    should_skip = bool(skip_tensor.item())
                
                if should_skip:
                    # Load existing result to include in final summary
                    if not self.is_distributed or self.rank == 0:
                        try:
                            with open(merged_filepath, 'r') as f:
                                existing_data = json.load(f)
                            if "summary" in existing_data and len(existing_data["summary"]) > 0:
                                summary_entry = existing_data["summary"][0]
                                results_data.append({
                                    "max_crops": max_crops,
                                    "top_k": top_k,
                                    "num_active_blocks": num_active,
                                    "num_total_blocks": total_blocks,
                                    "active_block_indices": summary_entry.get("active_block_indices", []),
                                    "accuracy": summary_entry.get("accuracy", 0.0),
                                    "num_samples": summary_entry.get("num_samples", 0),
                                    "latency_total_ms": summary_entry.get("latency_total_ms", {}),
                                    "latency_prefill_ms": summary_entry.get("latency_prefill_ms", {}),
                                    "latency_decode_ms": summary_entry.get("latency_decode_ms", {}),
                                    "per_sample_latencies": existing_data.get("per_sample_latencies", []),
                                })
                        except Exception as e:
                            log.warning(f"Failed to load existing result file {merged_filename}: {e}")
                    continue
                
                if not self.is_distributed or self.rank == 0:
                    log.info(f"=" * 80)
                    log.info(f"Configuration {config_idx + 1}/{len(combinations)}: "
                            f"max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active}/{total_blocks}")
                    log.info(f"=" * 80)
                
                config_start_time = time.time()
                
                # Set max_crops
                self._set_max_crops(max_crops)
                
                # Set top_k
                self._set_top_k(top_k)
                
                # Set active blocks
                block_mask, block_indices = self._set_active_blocks(num_active, total_blocks)
                if not self.is_distributed or self.rank == 0:
                    log.info(f"Active block indices: {block_indices}")
                
                # Apply block mask
                if mask_wrapper is not None:
                    mask_wrapper.remove()
                mask_wrapper = BlockMaskWrapper(self.model.model, block_mask)
                mask_wrapper.apply()
                
                # Build dataloader with batch_size=1
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
                
                # CRITICAL: exp6 uses batch_size=1 for accurate per-sample latency measurement
                # DO NOT change this - it's essential for latency profiling
                dataloader = torch.utils.data.DataLoader(
                    det_dataset,
                    batch_size=1,  # Fixed batch size for accurate latency measurement
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
                
                # Verify batch size is 1 (critical for exp6)
                if not self.is_distributed or self.rank == 0:
                    log.info(f"✓ DataLoader created with batch_size=1 (fixed for latency measurement)")
                    # Verify by checking DataLoader's batch_sampler (without consuming a batch)
                    if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'batch_size'):
                        actual_batch_size = dataloader.batch_sampler.batch_size
                        if actual_batch_size != 1:
                            log.error(f"✗ ERROR: DataLoader batch_size is {actual_batch_size}, expected 1!")
                            raise ValueError(f"DataLoader batch_size must be 1 for exp6, but got {actual_batch_size}")
                        log.info(f"✓ Verified: DataLoader batch_size={actual_batch_size} (correct)")
                    else:
                        # Fallback: check first batch (will be consumed, but that's okay for verification)
                        test_iter = iter(dataloader)
                        test_batch = next(test_iter)
                        actual_batch_size = test_batch["input_ids"].shape[0] if "input_ids" in test_batch else 1
                        if actual_batch_size != 1:
                            log.error(f"✗ ERROR: First batch has batch_size={actual_batch_size}, expected 1!")
                            raise ValueError(f"DataLoader batch_size must be 1 for exp6, but got {actual_batch_size}")
                        log.info(f"✓ Verified: First batch has batch_size={actual_batch_size} (correct)")
                        # Recreate dataloader since we consumed the first batch
                        dataloader = torch.utils.data.DataLoader(
                            det_dataset,
                            batch_size=1,
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
                
                all_latencies_total = []
                all_latencies_prefill = []
                all_latencies_decode = []
                per_sample_latencies = []
                all_scores = []
                all_predictions = []
                
                if not self.is_distributed or self.rank == 0:
                    log.info(f"Measuring latency for max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active}...")
                
                # Warmup
                if len(dataloader) > 0:
                    if not self.is_distributed or self.rank == 0:
                        log.info("Warming up for latency measurement...")
                    warmup_batch = next(iter(dataloader))
                    warmup_batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                   for k, v in warmup_batch.items()}
                    # Warmup: just run generate once
                    from transformers import GenerationConfig
                    eos_token_id = self.tokenizer.eos_token_id
                    if eos_token_id is None:
                        eos_token_id = getattr(self.model.config, 'eos_token_id', None)
                    pad_token_id = self.tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = getattr(self.model.config, 'pad_token_id', None)
                    
                    generation_config = GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                    )
                    
                    for _ in range(self.num_warmup):
                        with torch.inference_mode():
                            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                                _ = self.model.generate(
                                    input_ids=warmup_batch["input_ids"],
                                    images=warmup_batch.get("images"),
                                    image_masks=warmup_batch.get("image_masks"),
                                    image_input_idx=warmup_batch.get("image_input_idx"),
                                    generation_config=generation_config,
                                )
                
                # Measure latency for each sample
                # In distributed mode, each rank processes num_samples // world_size samples
                # If num_samples is None, use all samples
                if num_samples is not None:
                    num_samples_per_rank = num_samples // self.world_size if self.is_distributed else num_samples
                    total_batches = min(num_samples_per_rank, len(dataloader))
                else:
                    num_samples_per_rank = None
                    total_batches = len(dataloader)
                
                if not self.is_distributed or self.rank == 0:
                    if num_samples is not None:
                        log.info(f"Processing {num_samples_per_rank} samples per rank ({num_samples} total)")
                    else:
                        log.info(f"Processing all {len(dataloader)} samples from dataset")
                
                with torch.inference_mode():
                    for sample_idx, batch in enumerate(tqdm(dataloader, total=total_batches)):
                        if num_samples is not None and sample_idx >= num_samples_per_rank:
                            break
                        
                        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        # Measure latency: only prefill and decode (no vision components)
                        # Use hooks to measure prefill during generate, then calculate decode
                        try:
                            from transformers import GenerationConfig
                            
                            eos_token_id = self.tokenizer.eos_token_id
                            if eos_token_id is None:
                                eos_token_id = getattr(self.model.config, 'eos_token_id', None)
                            
                            pad_token_id = self.tokenizer.pad_token_id
                            if pad_token_id is None:
                                pad_token_id = getattr(self.model.config, 'pad_token_id', None)
                            
                            generation_config = GenerationConfig(
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                use_cache=True,
                                eos_token_id=eos_token_id,
                                pad_token_id=pad_token_id,
                            )
                            
                            # Set up hooks to measure prefill time
                            transformer = self.model.model.transformer
                            llm_prefill_time = None
                            prefill_start_time = None
                            
                            def prefill_start_hook(module, input, output):
                                nonlocal prefill_start_time
                                if self.device.type == 'cuda':
                                    torch.cuda.synchronize(self.device)
                                prefill_start_time = time.perf_counter()
                            
                            def prefill_end_hook(module, input, output):
                                nonlocal llm_prefill_time, prefill_start_time
                                if prefill_start_time is not None:
                                    if self.device.type == 'cuda':
                                        torch.cuda.synchronize(self.device)
                                    prefill_end_time = time.perf_counter()
                                    llm_prefill_time = (prefill_end_time - prefill_start_time) * 1000  # Convert to ms
                            
                            # Register hooks on first and last transformer blocks
                            start_hook_handle = None
                            end_hook_handle = None
                            if hasattr(transformer, 'blocks') and len(transformer.blocks) > 0:
                                start_hook_handle = transformer.blocks[0].register_forward_hook(prefill_start_hook)
                                end_hook_handle = transformer.blocks[-1].register_forward_hook(prefill_end_hook)
                            
                            # Measure total time (includes vision + prefill + decode)
                            if self.device.type == 'cuda':
                                torch.cuda.synchronize(self.device)
                            total_start_time = time.perf_counter()
                            
                            with torch.inference_mode():
                                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                                    output = self.model.generate(
                                        input_ids=batch["input_ids"],
                                        images=batch.get("images"),
                                        image_masks=batch.get("image_masks"),
                                        image_input_idx=batch.get("image_input_idx"),
                                        generation_config=generation_config,
                                    )
                            
                            if self.device.type == 'cuda':
                                torch.cuda.synchronize(self.device)
                            total_end_time = time.perf_counter()
                            T_total = (total_end_time - total_start_time) * 1000  # Convert to ms
                            
                            # Remove hooks
                            if start_hook_handle is not None:
                                start_hook_handle.remove()
                            if end_hook_handle is not None:
                                end_hook_handle.remove()
                            
                            # Extract latencies
                            T_prefill = llm_prefill_time if llm_prefill_time is not None else 0.0
                            # Decode = Total - Prefill (we don't separate vision, so this includes vision in decode)
                            # But for LLM decode, we approximate: Total - Prefill ≈ Vision + Decode
                            # Since we only care about LLM decode, we use: Decode ≈ Total - Prefill
                            # (This is an approximation, but vision time is typically small compared to decode)
                            T_decode = max(0.0, T_total - T_prefill)
                            
                            # Count vision tokens (same method as motivation experiments)
                            if "image_input_idx" in batch and batch["image_input_idx"] is not None:
                                # image_input_idx maps vision features to input_ids positions
                                # Valid entries (>=0) represent actual vision tokens used
                                num_vision_tokens = int((batch["image_input_idx"] >= 0).sum().item())
                            elif "images" in batch and batch["images"] is not None:
                                # Fallback: estimate from images shape
                                images = batch["images"]
                                if len(images.shape) == 4:  # (B, num_crops, H, W, C) or similar
                                    # Assume 576 tokens per crop (24x24 patches)
                                    num_crops = images.shape[1]
                                    num_vision_tokens = num_crops * 576
                                elif len(images.shape) == 3:  # (B, num_patches, patch_dim)
                                    num_vision_tokens = int(images.shape[1])
                                else:
                                    num_vision_tokens = 0
                            else:
                                num_vision_tokens = 0
                            
                            # Count other tokens
                            num_input_text_tokens = int(batch["input_ids"].shape[1])
                            num_output_tokens = int(output.shape[1] - batch["input_ids"].shape[1]) if output.shape[1] > batch["input_ids"].shape[1] else 0
                            
                            # Get appropriate metric for dataset
                            metric_name = get_metric_for_dataset(self.dataset_name)
                            batch_accuracy = self.compute_accuracy(
                                batch=batch,
                                predictions=output,
                                metric_name=metric_name,
                            )
                            
                            # Collect scores and predictions
                            batch_scores = [s["score"] for s in batch_accuracy["per_sample_scores"]]
                            all_scores.extend(batch_scores)
                            all_predictions.extend(batch_accuracy["per_sample_scores"])
                            
                            all_latencies_total.append(T_total)
                            all_latencies_prefill.append(T_prefill)
                            all_latencies_decode.append(T_decode)
                            
                            # Store per-sample data with both latency and accuracy
                            # Note: batch_size is fixed to 1, so each batch contains 1 sample
                            for i, pred_score in enumerate(batch_accuracy["per_sample_scores"]):
                                per_sample_latencies.append({
                                    "sample_id": sample_idx + i,
                                    "max_crops": max_crops,
                                    "top_k": top_k,
                                    "num_active_blocks": num_active,
                                    "T_total_ms": T_total,
                                    "T_prefill_ms": T_prefill,
                                    "T_decode_ms": T_decode,
                                    "num_vision_tokens": num_vision_tokens,
                                    "num_language_tokens": num_input_text_tokens,
                                    "num_output_tokens": num_output_tokens,
                                    "num_total_tokens": num_vision_tokens + num_input_text_tokens,
                                    "accuracy": pred_score["score"],
                                })
                        except Exception as e:
                            if not self.is_distributed or self.rank == 0:
                                log.warning(f"Failed to measure latency for sample {sample_idx}: {e}. Continuing.")
                            continue
                
                config_end_time = time.time()
                config_duration_seconds = config_end_time - config_start_time
                config_duration_minutes = config_duration_seconds / 60.0
                config_duration_hours = config_duration_seconds / 3600.0
                
                # Compute latency statistics
                latency_stats_total = {}
                latency_stats_prefill = {}
                latency_stats_decode = {}
                
                if all_latencies_total:
                    latency_stats_total = self.compute_statistics(all_latencies_total)
                    latency_stats_prefill = self.compute_statistics(all_latencies_prefill)
                    latency_stats_decode = self.compute_statistics(all_latencies_decode)
                
                # Compute overall accuracy
                overall_accuracy = np.mean(all_scores) if all_scores else 0.0
                
                result_entry = {
                    "max_crops": max_crops,
                    "top_k": top_k,
                    "num_active_blocks": num_active,
                    "num_total_blocks": total_blocks,
                    "active_block_indices": block_indices,
                    "num_samples": len(all_latencies_total),
                    "batch_size": 1,  # Fixed
                    "accuracy": float(overall_accuracy),
                    "latency_total_ms": latency_stats_total,
                    "latency_prefill_ms": latency_stats_prefill,
                    "latency_decode_ms": latency_stats_decode,
                    "per_sample_latencies": per_sample_latencies,
                    "per_sample_predictions": all_predictions,
                    "duration_seconds": float(config_duration_seconds),
                    "duration_minutes": float(config_duration_minutes),
                    "duration_hours": float(config_duration_hours),
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_start_time)),
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_end_time)),
                }
                
                results_data.append(result_entry)
                
                if not self.is_distributed or self.rank == 0:
                    if latency_stats_total:
                        log.info(f"Configuration {config_idx + 1}/{len(combinations)}: "
                                f"max_crops={max_crops}, top_k={top_k}, num_active_blocks={num_active}")
                        log.info(f"Accuracy: {overall_accuracy:.4f} ({len(all_scores)} samples)")
                        log.info(f"Latency (Total): Mean={latency_stats_total['mean']:.2f}ms, "
                                f"P50={latency_stats_total['P50']:.2f}ms, P95={latency_stats_total['P95']:.2f}ms, "
                                f"P99={latency_stats_total['P99']:.2f}ms")
                        log.info(f"Latency (Prefill): Mean={latency_stats_prefill['mean']:.2f}ms, "
                                f"P50={latency_stats_prefill['P50']:.2f}ms")
                        log.info(f"Latency (Decode): Mean={latency_stats_decode['mean']:.2f}ms, "
                                f"P50={latency_stats_decode['P50']:.2f}ms")
                        log.info(f"Duration: {config_duration_minutes:.2f} minutes ({config_duration_hours:.2f} hours)")
                
                # Save result for this configuration immediately (incremental save)
                if not self.is_distributed or self.rank == 0:
                    single_config_result = {
                        "summary": [{
                            "max_crops": result_entry["max_crops"],
                            "top_k": result_entry["top_k"],
                            "num_active_blocks": result_entry["num_active_blocks"],
                            "num_total_blocks": result_entry["num_total_blocks"],
                            "active_block_indices": result_entry["active_block_indices"],
                            "num_samples": result_entry["num_samples"],
                            "accuracy": result_entry["accuracy"],
                            "latency_total_ms": result_entry["latency_total_ms"],
                            "latency_prefill_ms": result_entry["latency_prefill_ms"],
                            "latency_decode_ms": result_entry["latency_decode_ms"],
                        }],
                        "per_sample_latencies": result_entry["per_sample_latencies"],
                        "per_sample_predictions": result_entry.get("per_sample_predictions", []),
                        "config": {
                            "dataset_name": dataset_name,
                            "split": split,
                            "batch_size": 1,
                            "max_new_tokens": max_new_tokens,
                            "max_crops": max_crops,
                            "top_k": top_k,
                            "num_active_blocks": num_active,
                            "num_total_blocks": total_blocks,
                            "active_block_indices": block_indices,
                            "world_size": self.world_size,
                        }
                    }
                    
                    single_filename = f"exp6_latency_crops{max_crops}_topk{top_k}_blocks{num_active}_rank{self.rank}.json"
                    self.save_results(single_config_result, single_filename)
        
        finally:
            # Always restore original forward method
            if mask_wrapper is not None:
                mask_wrapper.remove()
                if not self.is_distributed or self.rank == 0:
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
                            "per_sample_latencies": [],
                            "per_sample_predictions": [],
                            "all_latencies_total": [],
                            "all_latencies_prefill": [],
                            "all_latencies_decode": [],
                            "all_scores": [],
                        }
                    
                    config_dict[config_key]["per_sample_latencies"].extend(result["per_sample_latencies"])
                    if "per_sample_predictions" in result:
                        config_dict[config_key]["per_sample_predictions"].extend(result["per_sample_predictions"])
                    for lat in result["per_sample_latencies"]:
                        if "T_total_ms" in lat:
                            config_dict[config_key]["all_latencies_total"].append(lat["T_total_ms"])
                        if "T_prefill_ms" in lat:
                            config_dict[config_key]["all_latencies_prefill"].append(lat["T_prefill_ms"])
                        if "T_decode_ms" in lat:
                            config_dict[config_key]["all_latencies_decode"].append(lat["T_decode_ms"])
                        if "accuracy" in lat:
                            config_dict[config_key]["all_scores"].append(lat["accuracy"])
                
                # Compute merged statistics
                final_results_data = []
                for config_key, config_data in config_dict.items():
                    # Compute merged latency statistics
                    latency_stats_total = {}
                    latency_stats_prefill = {}
                    latency_stats_decode = {}
                    
                    if config_data["all_latencies_total"]:
                        latency_stats_total = self.compute_statistics(config_data["all_latencies_total"])
                        latency_stats_prefill = self.compute_statistics(config_data["all_latencies_prefill"])
                        latency_stats_decode = self.compute_statistics(config_data["all_latencies_decode"])
                    
                    # Compute merged accuracy
                    merged_accuracy = np.mean(config_data["all_scores"]) if config_data["all_scores"] else 0.0
                    
                    final_results_data.append({
                        "max_crops": config_data["max_crops"],
                        "top_k": config_data["top_k"],
                        "num_active_blocks": config_data["num_active_blocks"],
                        "num_total_blocks": config_data["num_total_blocks"],
                        "active_block_indices": config_data["active_block_indices"],
                        "num_samples": len(config_data["all_latencies_total"]),
                        "accuracy": float(merged_accuracy),
                        "latency_total_ms": latency_stats_total,
                        "latency_prefill_ms": latency_stats_prefill,
                        "latency_decode_ms": latency_stats_decode,
                        "per_sample_latencies": config_data["per_sample_latencies"],
                        "per_sample_predictions": config_data["per_sample_predictions"],
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
                            "num_samples": merged_result["num_samples"],
                            "accuracy": merged_result["accuracy"],
                            "latency_total_ms": merged_result["latency_total_ms"],
                            "latency_prefill_ms": merged_result["latency_prefill_ms"],
                            "latency_decode_ms": merged_result["latency_decode_ms"],
                        }],
                        "per_sample_latencies": merged_result["per_sample_latencies"],
                        "per_sample_predictions": merged_result.get("per_sample_predictions", []),
                        "config": {
                            "dataset_name": dataset_name,
                            "split": split,
                            "batch_size": 1,
                            "max_new_tokens": max_new_tokens,
                            "max_crops": merged_max_crops,
                            "top_k": merged_top_k,
                            "num_active_blocks": merged_num_active,
                            "num_total_blocks": merged_result["num_total_blocks"],
                            "active_block_indices": merged_result.get("active_block_indices", []),
                            "world_size": self.world_size,
                            "note": "Merged results from all ranks",
                        }
                    }
                    
                    # Save merged result (overwrites individual rank files)
                    merged_filename = f"exp6_latency_crops{merged_max_crops}_topk{merged_top_k}_blocks{merged_num_active}.json"
                    self.save_results(merged_config_result, merged_filename)
                    log.info(f"Saved merged result for max_crops={merged_max_crops}, top_k={merged_top_k}, "
                            f"num_active_blocks={merged_num_active}, accuracy={merged_result['accuracy']:.4f} to {merged_filename}")
            else:
                dist.gather_object(results_data, None, dst=0)
        
        # Save final results
        if not self.is_distributed or self.rank == 0:
            summary = []
            all_samples = []
            
            for config_result in results_data:
                summary_entry = {
                    "max_crops": config_result["max_crops"],
                    "top_k": config_result["top_k"],
                    "num_active_blocks": config_result["num_active_blocks"],
                    "num_samples": config_result["num_samples"],
                    "accuracy": config_result.get("accuracy", 0.0),
                    "latency_total_ms": config_result["latency_total_ms"],
                    "latency_prefill_ms": config_result["latency_prefill_ms"],
                    "latency_decode_ms": config_result["latency_decode_ms"],
                }
                if "duration_seconds" in config_result:
                    summary_entry["duration_seconds"] = config_result["duration_seconds"]
                    summary_entry["duration_minutes"] = config_result["duration_minutes"]
                    summary_entry["duration_hours"] = config_result["duration_hours"]
                if "start_time" in config_result:
                    summary_entry["start_time"] = config_result["start_time"]
                    summary_entry["end_time"] = config_result["end_time"]
                summary.append(summary_entry)
                
                # Collect all per-sample results
                if "per_sample_latencies" in config_result:
                    all_samples.extend(config_result["per_sample_latencies"])
            
            final_results = {
                "summary": summary,
                "all_samples": all_samples,
                "config": {
                    "dataset_name": dataset_name,
                    "split": split,
                    "batch_size": 1,
                    "max_new_tokens": max_new_tokens,
                    "max_crops_list": max_crops_list,
                    "top_k_list": top_k_list,
                    "num_active_blocks_list": num_active_blocks_list,
                    "sampling_strategy": sampling_strategy,
                    "num_combinations": len(combinations),
                    "world_size": self.world_size,
                }
            }
            
            self.save_results(final_results, "exp6_latency_results.json")
            log.info(f"Results saved. Total samples: {len(all_samples)}")
        
        # Cleanup distributed process group
        if self.is_distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Exp6 Latency: Combined Control Knobs Latency Analysis")
    parser.add_argument("--model_path", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results/profiling/exp6_latency")
    parser.add_argument("--dataset_name", type=str, default="coco_2014_vqa")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_new_tokens", type=int, default=16,
                       help="Maximum tokens to generate (default: 16, optimized for VQA)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Total number of samples to measure across all ranks (default: None = use all samples)")
    parser.add_argument("--max_crops", type=int, nargs="+", default=None,
                       help="List of max_crops values (default: [2, 4, 6, 8, 10], max 10)")
    parser.add_argument("--top_k", type=int, nargs="+", default=None,
                       help="List of top_k values (default: [4, 8, 12], three options, max 12)")
    parser.add_argument("--num_active_blocks", type=int, nargs="+", default=None,
                       help="List of num_active_blocks values (default: [12, 13, 14, 15, 16], min 12, max 16)")
    parser.add_argument("--sampling_strategy", type=str, default="balanced",
                       choices=["full", "stratified", "boundary", "balanced", "custom_sparse", "lhs"],
                       help="Sparse sampling strategy (default: balanced)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Determine split based on dataset
    split = args.split
    if args.dataset_name == "tally_qa":
        # TallyQA only has train and test, use test for validation
        split = "test"
        log.info(f"TallyQA dataset: using 'test' split instead of 'validation'")
    
    experiment = Exp6LatencyExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )
    
    # Convert num_samples: None means use all samples
    num_samples = args.num_samples if args.num_samples is not None else None
    
    experiment.run(
        dataset_name=args.dataset_name,
        split=split,
        max_new_tokens=args.max_new_tokens,
        num_samples=num_samples,
        max_crops_list=args.max_crops,
        top_k_list=args.top_k,
        num_active_blocks_list=args.num_active_blocks,
        sampling_strategy=args.sampling_strategy,
    )


if __name__ == "__main__":
    main()

