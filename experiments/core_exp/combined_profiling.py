"""
Combined Profiling: Accuracy and Latency with Vision Tokens Control
Tests combinations of vision tokens (target), MoE top_k, and transformer blocks.

Key features:
1. Vision tokens control: Target vision tokens → calculate max_crops → select tiling
2. Combined accuracy and latency measurement
3. Stage-wise latency breakdown (for E1 analysis)
4. Dataset sampling support (consistent across runs)
5. Optional PyTorch profiler (for detailed operator-level analysis)

Records detailed data for E1, E2, E3 analysis.
"""

import argparse
import logging
import sys
import os
import time
import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import itertools

# Set TOKENIZERS_PARALLELISM to avoid warnings
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from torch.utils.data import DistributedSampler, Subset
from transformers import GenerationConfig

sys.path.append(os.getcwd())
from experiments.base_experiment import BaseExperiment, get_metric_for_dataset
from molmo.models.modeling_molmoe import MolmoeSparseMoeBlock
from molmo.torch_util import get_world_size, get_global_rank, get_local_rank

# Import BlockMaskWrapper
sys.path.append(os.path.join(os.path.dirname(__file__), "../profiling/knob3_layers"))
from exp_transformer_blocks_mask import BlockMaskWrapper

log = logging.getLogger(__name__)


def is_a100_gpu(device: Optional[torch.device] = None) -> bool:
    """
    Detect if the current GPU is an A100 (typically 40GB).
    
    A100-40GB has ~40GB memory, while H100 typically has 80GB.
    We use memory size as the primary indicator since GPU name detection
    can vary across systems.
    
    Args:
        device: CUDA device to check. If None, uses current device.
    
    Returns:
        True if GPU appears to be A100 (40GB), False otherwise.
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        if device is None:
            device = torch.cuda.current_device()
        
        props = torch.cuda.get_device_properties(device)
        total_memory_gb = props.total_memory / (1024 ** 3)
        gpu_name = props.name
        
        # A100-40GB has ~40GB memory, H100 has ~80GB
        # Use 45GB as threshold to distinguish A100 from H100
        is_a100 = total_memory_gb < 45.0
        
        if is_a100:
            log.info(f"Detected A100 GPU: {gpu_name} ({total_memory_gb:.2f} GB)")
        else:
            log.info(f"Detected non-A100 GPU: {gpu_name} ({total_memory_gb:.2f} GB)")
        
        return is_a100
    except Exception as e:
        log.warning(f"Could not detect GPU type: {e}, assuming non-A100")
        return False


def tokens_to_crops(target_tokens: int) -> int:
    """
    Calculate number of crops needed for target vision tokens.
    
    Formula: target_tokens = (num_crops + 1) * 144
    Solve: num_crops = (target_tokens / 144) - 1
    
    Args:
        target_tokens: Target number of vision tokens
    
    Returns:
        Number of crops needed (at least 1)
    """
    num_crops = (target_tokens // 144) - 1
    return max(1, num_crops)  # At least 1 crop


def crops_to_tiling(num_crops: int, aspect_ratio: float = 1.0) -> Tuple[int, int]:
    """
    Find tiling configuration for given number of crops.
    
    Args:
        num_crops: Target number of crops
        aspect_ratio: Image aspect ratio (width/height)
    
    Returns:
        (rows, cols) tiling configuration
    """
    best_tiling = None
    best_match = float('inf')
    
    for i in range(1, num_crops + 1):
        if num_crops % i == 0:
            j = num_crops // i
            tiling = (i, j)
            
            # Calculate tiling aspect ratio
            # crop_window_size = 224, total_margin_pixels = 112
            resized_h = i * 224 + 112
            resized_w = j * 224 + 112
            tiling_aspect = resized_w / resized_h if resized_h > 0 else 1.0
            
            # Select closest to target aspect ratio
            if abs(tiling_aspect - aspect_ratio) < best_match:
                best_match = abs(tiling_aspect - aspect_ratio)
                best_tiling = tiling
    
    return best_tiling if best_tiling else (1, num_crops)


def tiling_to_resolution(tiling: Tuple[int, int], 
                         crop_window_size: int = 224,
                         total_margin_pixels: int = 112) -> Tuple[int, int]:
    """
    Calculate image resolution for given tiling.
    
    Args:
        tiling: (rows, cols) tiling configuration
        crop_window_size: Size of each crop window (default: 224)
        total_margin_pixels: Total margin pixels (default: 112)
    
    Returns:
        (target_h, target_w) resolution
    """
    rows, cols = tiling
    target_h = rows * crop_window_size + total_margin_pixels
    target_w = cols * crop_window_size + total_margin_pixels
    return target_h, target_w


def image_size_to_tiling(
    target_h: int,
    target_w: int,
    crop_window_size: int = 224,
    total_margin_pixels: int = 112,
) -> Tuple[int, int]:
    """
    Infer tiling (rows, cols) from target resized image size.
    Approx inverse of tiling_to_resolution: rows ≈ (H - margin)/224, cols ≈ (W - margin)/224.
    """
    rows = max(1, round((target_h - total_margin_pixels) / crop_window_size))
    cols = max(1, round((target_w - total_margin_pixels) / crop_window_size))
    return rows, cols


def calculate_theoretical_values(
    target_vision_tokens: int,
    original_image_size: Optional[Tuple[int, int]] = None,
    crop_window_size: int = 224,
    total_margin_pixels: int = 112,
) -> Dict[str, Any]:
    """
    Calculate theoretical values from target vision tokens.
    
    Args:
        target_vision_tokens: Target number of vision tokens
        original_image_size: Original image size (width, height) or None
        crop_window_size: Size of each crop window
        total_margin_pixels: Total margin pixels
    
    Returns:
        Dictionary with theoretical values:
        - theoretical_num_crops: Number of crops
        - theoretical_tiling: (rows, cols) tiling configuration
        - theoretical_image_size: (height, width) target image size
        - theoretical_vision_tokens: Theoretical vision tokens (should match target)
    """
    # Step 1: Calculate required crops
    theoretical_num_crops = tokens_to_crops(target_vision_tokens)
    
    # Step 2: Find best tiling for aspect ratio
    if original_image_size is not None:
        orig_w, orig_h = original_image_size
        aspect_ratio = orig_w / orig_h if orig_h > 0 else 1.0
    else:
        aspect_ratio = 1.0  # Default to square
    
    theoretical_tiling = crops_to_tiling(theoretical_num_crops, aspect_ratio)
    
    # Step 3: Calculate target resolution
    theoretical_image_size = tiling_to_resolution(
        theoretical_tiling, crop_window_size, total_margin_pixels
    )
    
    # Step 4: Calculate theoretical vision tokens
    theoretical_vision_tokens = (theoretical_num_crops + 1) * 144
    
    return {
        "theoretical_num_crops": theoretical_num_crops,
        "theoretical_tiling": theoretical_tiling,
        "theoretical_image_size": theoretical_image_size,
        "theoretical_vision_tokens": theoretical_vision_tokens,
    }


def image_size_to_tiling(
    target_h: int,
    target_w: int,
    crop_window_size: int = 224,
    total_margin_pixels: int = 112,
) -> Tuple[int, int]:
    """
    Infer tiling (rows, cols) from target resized image size.
    Inverse of tiling_to_resolution: rows ≈ (H - margin)/224, cols ≈ (W - margin)/224.
    """
    rows = max(1, round((target_h - total_margin_pixels) / crop_window_size))
    cols = max(1, round((target_w - total_margin_pixels) / crop_window_size))
    return rows, cols


def calculate_actual_values(
    batch: Dict[str, torch.Tensor],
    actual_vision_tokens: int,
) -> Dict[str, Any]:
    """
    Calculate actual values from batch after preprocessing.
    
    Args:
        batch: Batch dictionary with image_input_idx and metadata
        actual_vision_tokens: Actual vision tokens (from _calculate_vision_tokens)
    
    Returns:
        Dictionary with actual values:
        - actual_num_crops: Actual number of crops (inferred from vision tokens or image_input_idx)
        - actual_tiling: (rows, cols) tiling configuration (inferred, may be None)
        - actual_image_size: (height, width) actual image size (from metadata if available)
        - actual_vision_tokens: Actual vision tokens
    """
    # Try to infer actual number of crops from image_input_idx
    # image_input_idx has shape [seq_len] and contains indices for vision tokens
    # Each crop (including global image) contributes 144 tokens
    # We can count distinct "crop groups" by analyzing image_input_idx patterns
    actual_num_crops = None
    if "image_input_idx" in batch and batch["image_input_idx"] is not None:
        image_input_idx = batch["image_input_idx"]
        if isinstance(image_input_idx, torch.Tensor):
            # Count valid vision tokens (>= 0)
            valid_indices = image_input_idx[image_input_idx >= 0]
            if len(valid_indices) > 0:
                # Each crop contributes 144 tokens (12×12 grid)
                # So num_crops = (total_valid_tokens / 144) - 1 (subtract 1 for global image)
                estimated_num_crops = max(0, (len(valid_indices) // 144) - 1)
                actual_num_crops = estimated_num_crops
    
    # Fallback: estimate from actual_vision_tokens if image_input_idx not available
    if actual_num_crops is None:
        # Formula: actual_vision_tokens = (actual_num_crops + 1) * 144 (theoretical)
        # But actual may be less due to invalid patches
        # So we estimate: actual_num_crops ≈ (actual_vision_tokens / 144) - 1
        actual_num_crops = max(0, (actual_vision_tokens // 144) - 1)
    
    # Try to get actual image size from metadata
    actual_image_size = None
    if "metadata" in batch and batch["metadata"] is not None:
        metadata = batch["metadata"]
        if isinstance(metadata, dict) and "image_size" in metadata:
            # image_size is stored as (width, height) in metadata
            img_size = metadata["image_size"]
            if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                actual_image_size = (img_size[1], img_size[0])  # Convert to (height, width)
    
    # Infer tiling from num_crops (if we can determine it)
    # This is approximate since we don't have direct access to tiling
    actual_tiling = None
    if actual_num_crops > 0:
        # Try to infer tiling from num_crops
        # We can't know the exact tiling without more information, so we'll try to infer
        # from image aspect ratio if available, or use a reasonable default
        if actual_image_size is not None:
            actual_h, actual_w = actual_image_size
            aspect_ratio = actual_w / actual_h if actual_h > 0 else 1.0
            actual_tiling = crops_to_tiling(actual_num_crops, aspect_ratio)
        else:
            # Default to square tiling (best guess)
            actual_tiling = crops_to_tiling(actual_num_crops, 1.0)
    
    return {
        "actual_num_crops": actual_num_crops,
        "actual_tiling": actual_tiling,
        "actual_image_size": actual_image_size,
        "actual_vision_tokens": actual_vision_tokens,
    }


def crops_to_max_crops(num_crops: int) -> int:
    """
    Convert number of crops to max_crops parameter.
    
    max_crops should be >= num_crops to allow select_tiling to choose the tiling.
    We set max_crops = num_crops to ensure select_tiling selects exactly num_crops.
    
    Args:
        num_crops: Target number of crops
    
    Returns:
        max_crops parameter value
    """
    return num_crops


class CombinedProfilingExperiment(BaseExperiment):
    """
    Combined Profiling: Measure accuracy and latency for different combinations of:
    - Vision tokens (target) → max_crops → tiling
    - MoE top_k
    - Transformer blocks (via importance-based masking)
    
    Records detailed data for E1, E2, E3 analysis.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        output_dir: str = "./results/core_exp",
        num_warmup: int = 3,
        hf_cache_dir: Optional[str] = None,
        dataset_name: str = "coco_2014_vqa",
        seed: int = 66,  # Random seed for reproducibility
    ):
        # Set random seeds for reproducibility (before any random operations)
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Set deterministic algorithms where possible (may have performance impact)
            # Note: Some CUDA operations may still have non-deterministic behavior
            # but this helps ensure reproducibility for most operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Auto-detect distributed environment
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.is_distributed = True
        else:
            self.is_distributed = False
        
        self.rank = get_global_rank()
        self.world_size = get_world_size()
        self.seed = seed
        
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
        
        # Adjust output_dir to include dataset name as subdirectory
        base_output_dir = Path(output_dir)
        dataset_suffix = dataset_name.replace("_", "-")
        output_dir = str(base_output_dir / dataset_suffix)  # Save in subdirectory, not sibling
        log.info(f"Output directory: {output_dir}")
        
        super().__init__(
            model_path=model_path,
            device=device,
            output_dir=output_dir,
            num_warmup=num_warmup,
            hf_cache_dir=hf_cache_dir,
        )
        self.dataset_name = dataset_name
        self.block_mask_wrapper = None
        
        # Auto-detect A100 and enable memory optimization if needed
        # This ensures H100 experiments are not affected
        # Device is set in BaseExperiment.__init__, so we can check it now
        self.is_a100 = is_a100_gpu(self.device)
    
    def _create_generation_config(self, max_new_tokens: int) -> GenerationConfig:
        """
        Create GenerationConfig for model.generate().
        
        Args:
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            GenerationConfig object
        """
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self.model.config, 'pad_token_id', None)
        
        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding (deterministic, no randomness)
            use_cache=True,  # Required by Molmo
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    
    def _calculate_vision_tokens(self, batch: Dict[str, torch.Tensor]) -> int:
        """
        Calculate actual vision tokens from batch.
        
        Uses image_input_idx which already reflects post-pooling token count (144 per crop).
        Formula: Total Vision Tokens = (num_crops + 1) × 144 (theoretical maximum)
        
        Note: Actual vision tokens may be LESS than the theoretical value because:
        - Some patches may be marked as invalid (-100) if they exceed image boundaries
        - Padding may cause some patches to be invalid
        - Tiling configuration may result in partial crops
        Therefore, actual_vision_tokens is NOT necessarily a multiple of 144.
        """
        if "image_input_idx" in batch and batch["image_input_idx"] is not None:
            # Count only valid vision tokens (image_input_idx >= 0)
            # Invalid patches are marked as -100 in image_input_idx
            num_vision_tokens = int((batch["image_input_idx"] >= 0).sum().item())
            return num_vision_tokens
        else:
            # Fallback: estimate from max_crops
            max_crops = self.model.config.max_crops
            num_vision_tokens = (max_crops + 1) * 144
            log.warning(f"Could not get vision tokens from image_input_idx, estimating: {num_vision_tokens}")
            return num_vision_tokens
    
    def _vision_tokens_to_max_crops(self, target_tokens: int) -> int:
        """
        Convert target vision tokens to max_crops parameter.
        
        Args:
            target_tokens: Target number of vision tokens
        
        Returns:
            max_crops parameter value
        """
        num_crops = tokens_to_crops(target_tokens)
        max_crops = crops_to_max_crops(num_crops)
        return max_crops
    
    def _generate_sparse_combinations(
        self,
        vision_tokens_list: List[int],
        top_k_list: List[int],
        num_active_blocks_list: List[int],
        sampling_strategy: str = "balanced",
        max_combinations: int = 50,
        seed: int = 66,  # Random seed for reproducibility
    ) -> List[Tuple[int, int, int]]:
        """
        Generate sparse combinations of the three knobs.
        
        Args:
            vision_tokens_list: List of target vision token values
            top_k_list: List of top_k values
            num_active_blocks_list: List of num_active_blocks values
            sampling_strategy: Strategy for sparse sampling
            max_combinations: Maximum number of combinations
        
        Returns:
            List of (vision_tokens, top_k, num_active_blocks) tuples
        """
        if sampling_strategy == "full":
            combinations = list(itertools.product(vision_tokens_list, top_k_list, num_active_blocks_list))
            log.info(f"Full grid search: {len(combinations)} combinations")
            return combinations
        
        elif sampling_strategy == "balanced":
            # Balanced coverage: 3-4 values from each dimension
            # If list has only 1 value, use it directly to avoid duplicates
            if len(vision_tokens_list) >= 4:
                sparse_vision_tokens = [
                    vision_tokens_list[0],
                    vision_tokens_list[len(vision_tokens_list)//3],
                    vision_tokens_list[2*len(vision_tokens_list)//3],
                    vision_tokens_list[-1]
                ]
            elif len(vision_tokens_list) == 1:
                sparse_vision_tokens = vision_tokens_list
            else:
                sparse_vision_tokens = [vision_tokens_list[0], vision_tokens_list[len(vision_tokens_list)//2], vision_tokens_list[-1]]
                # Remove duplicates while preserving order
                sparse_vision_tokens = list(dict.fromkeys(sparse_vision_tokens))
            
            if len(top_k_list) >= 4:
                sparse_top_k = [
                    top_k_list[0],
                    top_k_list[len(top_k_list)//3],
                    top_k_list[2*len(top_k_list)//3],
                    top_k_list[-1]
                ]
            elif len(top_k_list) == 1:
                sparse_top_k = top_k_list
            else:
                sparse_top_k = [top_k_list[0], top_k_list[len(top_k_list)//2], top_k_list[-1]]
                # Remove duplicates while preserving order
                sparse_top_k = list(dict.fromkeys(sparse_top_k))
            
            if len(num_active_blocks_list) >= 4:
                sparse_blocks = [
                    num_active_blocks_list[0],
                    num_active_blocks_list[len(num_active_blocks_list)//3],
                    num_active_blocks_list[2*len(num_active_blocks_list)//3],
                    num_active_blocks_list[-1]
                ]
            elif len(num_active_blocks_list) == 1:
                sparse_blocks = num_active_blocks_list
            else:
                sparse_blocks = [num_active_blocks_list[0], num_active_blocks_list[len(num_active_blocks_list)//2], num_active_blocks_list[-1]]
                # Remove duplicates while preserving order
                sparse_blocks = list(dict.fromkeys(sparse_blocks))
            
            combinations = list(itertools.product(sparse_vision_tokens, sparse_top_k, sparse_blocks))
            # Remove duplicate combinations (in case of duplicate values in sparse lists)
            combinations = list(dict.fromkeys(combinations))
            log.info(f"Balanced sampling: {len(combinations)} combinations")
            log.info(f"  vision_tokens: {sparse_vision_tokens}, top_k: {sparse_top_k}, blocks: {sparse_blocks}")
            return combinations
        
        elif sampling_strategy == "stratified":
            # Stratified: min, middle, max
            # If list has only 1 value, use it directly to avoid duplicates
            if len(vision_tokens_list) == 1:
                sparse_vision_tokens = vision_tokens_list
            else:
                sparse_vision_tokens = [vision_tokens_list[0], vision_tokens_list[len(vision_tokens_list)//2], vision_tokens_list[-1]]
                sparse_vision_tokens = list(dict.fromkeys(sparse_vision_tokens))
            
            if len(top_k_list) == 1:
                sparse_top_k = top_k_list
            else:
                sparse_top_k = [top_k_list[0], top_k_list[len(top_k_list)//2], top_k_list[-1]]
                sparse_top_k = list(dict.fromkeys(sparse_top_k))
            
            if len(num_active_blocks_list) == 1:
                sparse_blocks = num_active_blocks_list
            else:
                sparse_blocks = [num_active_blocks_list[0], num_active_blocks_list[len(num_active_blocks_list)//2], num_active_blocks_list[-1]]
                sparse_blocks = list(dict.fromkeys(sparse_blocks))
            
            combinations = list(itertools.product(sparse_vision_tokens, sparse_top_k, sparse_blocks))
            # Remove duplicate combinations
            combinations = list(dict.fromkeys(combinations))
            log.info(f"Stratified sampling: {len(combinations)} combinations")
            return combinations
        
        elif sampling_strategy == "boundary":
            # Boundary: min, 25%, 50%, 75%, max
            # If list has only 1 value, use it directly to avoid duplicates
            if len(vision_tokens_list) == 1:
                sparse_vision_tokens = vision_tokens_list
            else:
                sparse_vision_tokens = [
                    vision_tokens_list[0],
                    vision_tokens_list[len(vision_tokens_list)//4],
                    vision_tokens_list[len(vision_tokens_list)//2],
                    vision_tokens_list[3*len(vision_tokens_list)//4],
                    vision_tokens_list[-1]
                ]
                sparse_vision_tokens = list(dict.fromkeys(sparse_vision_tokens))
            
            if len(top_k_list) == 1:
                sparse_top_k = top_k_list
            else:
                sparse_top_k = [
                    top_k_list[0],
                    top_k_list[len(top_k_list)//4],
                    top_k_list[len(top_k_list)//2],
                    top_k_list[3*len(top_k_list)//4],
                    top_k_list[-1]
                ]
                sparse_top_k = list(dict.fromkeys(sparse_top_k))
            
            if len(num_active_blocks_list) == 1:
                sparse_blocks = num_active_blocks_list
            else:
                sparse_blocks = [
                    num_active_blocks_list[0],
                    num_active_blocks_list[len(num_active_blocks_list)//4],
                    num_active_blocks_list[len(num_active_blocks_list)//2],
                    num_active_blocks_list[3*len(num_active_blocks_list)//4],
                    num_active_blocks_list[-1]
                ]
                sparse_blocks = list(dict.fromkeys(sparse_blocks))
            
            combinations = list(itertools.product(sparse_vision_tokens, sparse_top_k, sparse_blocks))
            # Remove duplicate combinations
            combinations = list(dict.fromkeys(combinations))
            log.info(f"Boundary sampling: {len(combinations)} combinations")
            return combinations
        
        elif sampling_strategy == "lhs":
            # Latin Hypercube Sampling
            # Use fixed seed for reproducibility
            rng = np.random.RandomState(seed=seed)
            n_samples = min(max_combinations, len(vision_tokens_list) * len(top_k_list) * len(num_active_blocks_list) // 4)
            
            combinations = []
            for _ in range(n_samples):
                vision_tokens = rng.choice(vision_tokens_list)
                top_k = rng.choice(top_k_list)
                num_blocks = rng.choice(num_active_blocks_list)
                combinations.append((vision_tokens, top_k, num_blocks))
            
            combinations = list(set(combinations))
            log.info(f"Latin Hypercube Sampling: {len(combinations)} unique combinations (seed={seed})")
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
    
    def _set_active_blocks_importance_based(
        self,
        num_active: int,
        total_blocks: int,
        importance_scores: Optional[Dict[int, float]] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Set active transformer blocks using importance-based selection.
        
        If importance_scores provided, select top-K by importance.
        Otherwise, use prefix blocks (first N blocks).
        """
        if importance_scores is not None and len(importance_scores) > 0:
            sorted_blocks = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            block_indices = [idx for idx, _ in sorted_blocks[:num_active]]
            block_indices = sorted(block_indices)
        else:
            block_indices = list(range(min(num_active, total_blocks)))
        
        block_mask = torch.zeros(total_blocks, dtype=torch.bool)
        for idx in block_indices:
            block_mask[idx] = True
        
        return block_mask, block_indices
    
    def _create_sampled_dataset(self, dataset, num_samples: Optional[int], seed: int = 66):
        """
        Create randomly sampled dataset with consistent sampling.
        
        Args:
            dataset: Original dataset
            num_samples: Number of samples to keep (None = use all)
            seed: Random seed for reproducibility
        
        Returns:
            Sampled dataset (Subset or original if num_samples is None)
        """
        if num_samples is None or num_samples >= len(dataset):
            return dataset
        
        # Use deterministic random sampling
        rng = np.random.RandomState(seed=seed)
        indices = rng.choice(len(dataset), size=num_samples, replace=False)
        indices = sorted(indices.tolist())  # Sort for reproducibility
        
        log.info(f"Sampling {num_samples} samples from {len(dataset)} total samples (seed={seed})")
        return Subset(dataset, indices)
    
    def run(
        self,
        dataset_name: str = "coco_2014_vqa",
        split: str = "validation",
        max_new_tokens: int = 16,
        vision_tokens_list: List[int] = None,
        image_size_list: Optional[List[str]] = None,  # New: list of "HxW"
        top_k_list: List[int] = None,
        num_active_blocks_list: List[int] = None,
        sampling_strategy: str = "balanced",
        num_samples: Optional[int] = 1000,  # Default: sample 1000 samples
        num_runs_per_sample: int = 3,  # Number of runs per sample for latency averaging
        importance_scores: Optional[Dict[int, float]] = None,
        use_profiler: bool = False,  # Use PyTorch profiler for detailed analysis
        use_profiler_on_all_samples: bool = False,  # If True, profile all samples; if False, only first sample
        profiler_activities: Optional[List] = None,
        enable_memory_optimization: bool = False,  # Enable memory optimizations for limited GPU memory
    ):
        """
        Run combined profiling experiment.
        
        Args:
            vision_tokens_list: List of target vision token values (e.g., [288, 432, 576, 720, 1008, 1440, 1872])
            num_samples: Number of samples to use from dataset (None = use all)
                        Recommended: 1000-2000 for combined profiling
            num_runs_per_sample: Number of runs per sample for latency averaging (default: 3)
            use_profiler: If True, use PyTorch profiler for detailed operator-level analysis
            use_profiler_on_all_samples: If True, profile all samples (slower but comprehensive).
                                         If False, only profile first sample (faster, default).
            profiler_activities: List of profiler activities (default: [ProfilerActivity.CUDA])
        """
        # Default knob ranges (based on previous exp5/exp6 settings)
        # New primary knob: image_size_list (HxW). If provided, overrides vision_tokens_list.
        if image_size_list:
            # Parse image sizes and derive corresponding vision token targets
            image_specs = []
            for sz in image_size_list:
                if isinstance(sz, str) and "x" in sz:
                    h_str, w_str = sz.lower().split("x")
                    target_h, target_w = int(h_str), int(w_str)
                elif isinstance(sz, (list, tuple)) and len(sz) == 2:
                    target_h, target_w = int(sz[0]), int(sz[1])
                else:
                    raise ValueError(f"Invalid image size format: {sz}. Use 'HxW', e.g., 560x336.")
                rows, cols = image_size_to_tiling(target_h, target_w)
                num_crops = rows * cols
                target_tokens = (num_crops + 1) * 144  # theoretical tokens
                image_specs.append({
                    "target_h": target_h,
                    "target_w": target_w,
                    "rows": rows,
                    "cols": cols,
                    "num_crops": num_crops,
                    "target_tokens": target_tokens,
                })
            # Deduplicate by (target_h, target_w)
            unique_specs = []
            seen = set()
            for spec in image_specs:
                key = (spec["target_h"], spec["target_w"])
                if key not in seen:
                    seen.add(key)
                    unique_specs.append(spec)
            image_specs = unique_specs
            log.info(f"Using image_size_list (primary knob): {[(s['target_h'], s['target_w']) for s in image_specs]}")
        else:
            # Fallback to legacy vision_tokens_list knob
            if vision_tokens_list is None:
                # Common vision token values based on vision_tokens_knob.md
                # Corresponds to: 2, 4, 6, 8, 10 crops (max_crops=16 as upper limit)
                vision_tokens_list = [432, 720, 1008, 1296, 1584]  # 2, 4, 6, 8, 10 crops
            log.info(f"Using vision_tokens_list (legacy knob): {vision_tokens_list}")
        if top_k_list is None:
            top_k_list = [4, 8, 12]  # Based on previous experiments
        if num_active_blocks_list is None:
            num_active_blocks_list = [12, 13, 14, 15, 16]  # Based on previous experiments
        
        # Generate combinations
        if image_size_list:
            # With explicit image sizes, use full product to keep mapping exact
            combinations = []
            for spec in image_specs:
                for top_k in top_k_list:
                    for num_active_blocks in num_active_blocks_list:
                        combinations.append((spec, top_k, num_active_blocks))
            log.info(f"Using image-size knob: {len(combinations)} combinations "
                     f"(image_sizes x top_k x num_active_blocks)")
        else:
            combinations = self._generate_sparse_combinations(
                vision_tokens_list, top_k_list, num_active_blocks_list, sampling_strategy, seed=self.seed
            )
        
        log.info(f"Testing {len(combinations)} configurations")
        log.info(f"Dataset: {dataset_name}/{split}")
        log.info(f"Number of samples: {num_samples if num_samples else 'all'}")
        log.info(f"Runs per sample: {num_runs_per_sample}")
        log.info(f"Use profiler: {use_profiler}")
        if use_profiler:
            log.info(f"Profiler mode: {'all samples' if use_profiler_on_all_samples else 'first sample only'}")
        
        # Auto-enable memory optimization on A100, but respect explicit flag
        # If explicitly disabled, don't auto-enable
        if self.is_a100 and not hasattr(self, '_memory_opt_explicitly_disabled'):
            if enable_memory_optimization is False:
                self._memory_opt_explicitly_disabled = True
            else:
                enable_memory_optimization = True
                log.info("Auto-enabled memory optimization for A100 GPU")
        
        if enable_memory_optimization:
            log.info("Memory optimization enabled for limited GPU memory (A100-40GB)")
        
        # Import data loading modules
        from molmo.data import get_dataset_by_name
        from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        
        dataset = get_dataset_by_name(dataset_name, split=split)
        
        # Apply sampling if specified (before creating dataloader)
        if num_samples is not None:
            # Unwrap DeterministicDataset if needed to get base dataset
            # For now, we'll limit via max_steps in dataloader iteration
            log.info(f"Will limit to {num_samples} samples per rank")
        
        results = []
        
        # Process each configuration
        for config_idx, combo in enumerate(combinations):
            if image_size_list:
                spec, top_k, num_active_blocks = combo
                target_vision_tokens = spec["target_tokens"]
                num_crops = spec["num_crops"]
                theoretical_tiling = (spec["rows"], spec["cols"])
                theoretical_image_size = (spec["target_h"], spec["target_w"])
            else:
                target_vision_tokens, top_k, num_active_blocks = combo
                num_crops = tokens_to_crops(target_vision_tokens)
                theoretical_tiling = None  # will be computed later if needed
                theoretical_image_size = None
            if self.rank == 0:
                log.info(f"\n{'='*60}")
                if image_size_list:
                    log.info(f"Configuration {config_idx+1}/{len(combinations)}: "
                             f"image_size={theoretical_image_size}, tiling={theoretical_tiling}, "
                             f"target_vision_tokens={target_vision_tokens}, num_crops={num_crops}, "
                             f"top_k={top_k}, num_active_blocks={num_active_blocks}")
                else:
                    log.info(f"Configuration {config_idx+1}/{len(combinations)}: "
                             f"vision_tokens={target_vision_tokens}, num_crops={num_crops}, "
                             f"top_k={top_k}, num_active_blocks={num_active_blocks}")
                log.info(f"{'='*60}")
            
            try:
                if image_size_list:
                    max_crops = num_crops
                    log.info(f"Target image size: {theoretical_image_size}, tiling={theoretical_tiling}, "
                             f"num_crops={num_crops}, max_crops={max_crops}, target_tokens={target_vision_tokens}")
                else:
                    # Convert vision tokens to num_crops (actual number of crops needed)
                    num_crops = tokens_to_crops(target_vision_tokens)
                    # Set max_crops to num_crops to ensure select_tiling selects exactly num_crops
                    # Note: For very large images, select_tiling may still choose fewer crops if the image
                    # doesn't need that many, but it won't exceed max_crops
                    max_crops = num_crops
                    log.info(f"Target vision tokens: {target_vision_tokens} → num_crops: {num_crops}, max_crops: {max_crops}")
                
                # Aggressive memory cleanup at start of each configuration on A100
                if enable_memory_optimization and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    # Force garbage collection to free Python objects
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Set configuration
                self._set_max_crops(max_crops)
                self._set_top_k(top_k)
                
                # Set active blocks
                total_blocks = len(self.model.model.transformer.blocks) if hasattr(self.model.model.transformer, 'blocks') else 24
                block_mask, block_indices = self._set_active_blocks_importance_based(
                    num_active_blocks, total_blocks, importance_scores
                )
                
                # Apply block mask
                # Note: BlockMaskWrapper expects MolmoModel (self.model.model), not MolmoForCausalLM (self.model)
                if self.block_mask_wrapper is not None:
                    self.block_mask_wrapper.remove()
                self.block_mask_wrapper = BlockMaskWrapper(self.model.model, block_mask)
                self.block_mask_wrapper.apply()
                
                # Build dataloader (batch_size=1 for accurate per-sample measurement)
                # Set exact_num_crops to force select_tiling to use exactly num_crops
                # This ensures actual_vision_tokens matches target_vision_tokens
                force_full_tokens = False  # restore default: do not count padding tokens
                mm_preprocessor = MultiModalPreprocessor(
                    tokenizer=self.tokenizer,
                    crop_mode=self.model.config.crop_mode,
                    max_crops=max_crops,
                    exact_num_crops=num_crops,  # Force exact number of crops
                    overlap_margins=self.model.config.overlap_margins,
                    image_padding_mask=bool(self.model.config.image_padding_embed),
                    force_full_tokens=force_full_tokens,  # Count padded patches as valid to match target tokens
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
                
                det_dataset = DeterministicDataset(dataset, preprocessor, seed=self.seed)
                
                # Apply sampling if specified
                # Note: We sample after DeterministicDataset to ensure consistent sampling
                # across configurations (same seed)
                if num_samples is not None and num_samples < len(det_dataset):
                    # Create sampled dataset
                    rng = np.random.RandomState(seed=self.seed)  # Fixed seed for reproducibility
                    all_indices = list(range(len(det_dataset)))
                    sampled_indices = rng.choice(all_indices, size=num_samples, replace=False)
                    sampled_indices = sorted(sampled_indices.tolist())  # Sort for reproducibility
                    sampled_dataset = Subset(det_dataset, sampled_indices)
                    log.info(f"Sampled {num_samples} samples from {len(det_dataset)} total (seed={self.seed})")
                    final_dataset = sampled_dataset
                else:
                    final_dataset = det_dataset
                
                if self.is_distributed:
                    sampler = DistributedSampler(
                        final_dataset,
                        num_replicas=self.world_size,
                        rank=self.rank,
                        shuffle=False,
                        seed=self.seed,  # Use consistent seed for reproducibility
                    )
                    shuffle = False
                else:
                    sampler = None
                    shuffle = False
                
                # Memory-optimized dataloader settings for A100-40GB
                # Only apply optimizations if explicitly enabled (auto-enabled on A100)
                # Use environment variables if set, otherwise use defaults
                if enable_memory_optimization:
                    num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", 2))  # Reduced for A100
                    prefetch_factor = int(os.environ.get("DATALOADER_PREFETCH_FACTOR", 1))  # Reduced for A100
                    pin_memory = False  # Disable pin_memory to save memory on A100
                    persistent_workers = False  # Disable to avoid worker state issues on A100
                else:
                    # Default settings for H100 (no optimization)
                    num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", 4))
                    prefetch_factor = int(os.environ.get("DATALOADER_PREFETCH_FACTOR", 2))
                    pin_memory = True  # Enable pin_memory for better performance on H100
                    persistent_workers = (num_workers > 0)  # Enable for H100
                
                dataloader = torch.utils.data.DataLoader(
                    final_dataset,
                    batch_size=1,  # Fixed batch_size=1 for accurate per-sample measurement
                    shuffle=shuffle,
                    sampler=sampler,
                    collate_fn=MMCollator(
                        max_sequence_length=1536,
                        include_metadata=True,
                        pad=True,
                        max_crops=max_crops
                    ),
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                )
                
                # Collect results
                all_scores = []
                all_predictions = []
                per_sample_results = []
                
                # Get metric
                metric_name = get_metric_for_dataset(dataset_name)
                
                # Process samples
                num_processed = 0
                # In distributed mode, each rank processes its share
                # The sampler already handles distribution, so we process all samples from this rank's dataloader
                # If num_samples was specified, the dataset was already sampled, so we process all
                max_samples_per_rank = None  # Process all samples from this rank's dataloader
                
                # Aggressive memory cleanup before warmup on A100
                if enable_memory_optimization and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Warmup
                if len(dataloader) > 0:
                    try:
                        warmup_batch = next(iter(dataloader))
                        # Move to device with error handling
                        try:
                            warmup_batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                           for k, v in warmup_batch.items()}
                        except RuntimeError as e:
                            if "CUDA error" in str(e) or "device-side assert" in str(e) or "out of memory" in str(e).lower():
                                log.error(f"CUDA error during warmup batch transfer: {e}")
                                # Reset CUDA state
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                                    import gc
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                raise
                            else:
                                raise
                        
                        warmup_gen_config = self._create_generation_config(max_new_tokens)
                        for warmup_idx in range(self.num_warmup):
                            with torch.inference_mode():
                                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                                    try:
                                        _ = self.model.generate(
                                            input_ids=warmup_batch["input_ids"],
                                            images=warmup_batch.get("images"),
                                            image_masks=warmup_batch.get("image_masks"),
                                            image_input_idx=warmup_batch.get("image_input_idx"),
                                            generation_config=warmup_gen_config,
                                        )
                                        # Clear cache after each warmup iteration on A100
                                        if enable_memory_optimization and torch.cuda.is_available() and warmup_idx < self.num_warmup - 1:
                                            torch.cuda.empty_cache()
                                    except RuntimeError as e:
                                        if "CUDA error" in str(e) or "device-side assert" in str(e) or "out of memory" in str(e).lower():
                                            log.error(f"CUDA error during warmup generation (iteration {warmup_idx+1}/{self.num_warmup}): {e}")
                                            # Reset CUDA state
                                            if torch.cuda.is_available():
                                                torch.cuda.synchronize()
                                                torch.cuda.empty_cache()
                                                import gc
                                                gc.collect()
                                                torch.cuda.empty_cache()
                                            raise
                                        else:
                                            raise
                        # Clear warmup batch from memory
                        del warmup_batch
                        # Aggressive cleanup after warmup on A100
                        if enable_memory_optimization and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "CUDA error" in str(e) or "device-side assert" in str(e) or "out of memory" in str(e).lower():
                            log.error(f"CUDA error during warmup, skipping this configuration: {e}")
                            # Don't continue with this configuration if warmup fails
                            raise
                        else:
                            raise
                
                # Setup profiler if requested
                # Note: Profiler is created per sample, not per configuration, to avoid memory issues
                # We'll create it inside the loop for each sample
                
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Config {config_idx+1}/{len(combinations)}")):
                    # Process all samples from this rank's dataloader
                    # (sampling was already applied to dataset if num_samples was specified)
                    
                    try:
                        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        # Calculate actual vision tokens (measured from image_input_idx)
                        actual_vision_tokens = self._calculate_vision_tokens(batch)
                        
                        # Get original image size from metadata (if available)
                        original_image_size = None
                        if "metadata" in batch and batch["metadata"] is not None:
                            metadata = batch["metadata"]
                            if isinstance(metadata, dict) and "image_size" in metadata:
                                # image_size is stored as (width, height) in metadata
                                img_size = metadata["image_size"]
                                if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                                    original_image_size = tuple(img_size)  # Keep as (width, height)
                        
                        # Calculate theoretical values (from target_vision_tokens or image size)
                        if image_size_list and theoretical_image_size is not None:
                            theoretical_values = {
                                "theoretical_num_crops": num_crops,
                                "theoretical_tiling": theoretical_tiling,
                                "theoretical_image_size": theoretical_image_size,
                                "theoretical_vision_tokens": target_vision_tokens,
                            }
                        else:
                            theoretical_values = calculate_theoretical_values(
                                target_vision_tokens=target_vision_tokens,
                                original_image_size=original_image_size,
                            )
                        
                        # Calculate actual values (from batch after preprocessing)
                        actual_values = calculate_actual_values(
                            batch=batch,
                            actual_vision_tokens=actual_vision_tokens,
                        )
                        
                        # Measure latency with stage breakdown
                        # Note: Profiler can be used on all samples or just first sample
                        # For detailed analysis, use on all samples (slower but more comprehensive)
                        # For quick profiling, use only on first sample (faster)
                        use_profiler_this_sample = use_profiler and (batch_idx == 0 or use_profiler_on_all_samples)
                        
                        if use_profiler_this_sample:
                            # Use profiler for detailed analysis (only on first sample)
                            from torch.profiler import profile, record_function, ProfilerActivity
                            if profiler_activities is None:
                                profiler_activities = [ProfilerActivity.CUDA]
                            
                            profiler = profile(
                                activities=profiler_activities,
                                record_shapes=False,  # Disable for performance
                                with_stack=False,     # Disable for performance
                                profile_memory=False, # Disable for performance
                            )
                            
                            with profiler:
                                with record_function("model_inference"):
                                    # Generate with profiler
                                    gen_config = self._create_generation_config(max_new_tokens)
                                    with torch.inference_mode():
                                        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                                            output = self.model.generate(
                                                input_ids=batch["input_ids"],
                                                images=batch.get("images"),
                                                image_masks=batch.get("image_masks"),
                                                image_input_idx=batch.get("image_input_idx"),
                                                generation_config=gen_config,
                                            )
                            
                            # Save profiler results
                            profiler_output = profiler.key_averages().table(sort_by="cuda_time_total")
                            profiler_file = Path(self.output_dir) / f"profiler_results_config_{config_idx+1}_sample_{batch_idx}.txt"
                            with open(profiler_file, 'w') as f:
                                f.write(profiler_output)
                            log.info(f"Saved profiler results to {profiler_file}")
                            
                            # Measure latency separately (profiler adds overhead, so measure separately)
                            latency_results = self.measure_inference_latency(
                                batch=batch,
                                max_new_tokens=max_new_tokens,
                                measure_components=True,
                                num_runs=num_runs_per_sample,
                                use_hook_for_llm_prefill=True,
                                use_eos_token=True,
                            )
                        else:
                            # Use manual timing (no profiler, or profiler only on first sample)
                            latency_results = self.measure_inference_latency(
                                batch=batch,
                                max_new_tokens=max_new_tokens,
                                measure_components=True,
                                num_runs=num_runs_per_sample,
                                use_hook_for_llm_prefill=True,
                                use_eos_token=True,
                            )
                            
                            # Generate for accuracy
                            gen_config = self._create_generation_config(max_new_tokens)
                            with torch.inference_mode():
                                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                                    output = self.model.generate(
                                        input_ids=batch["input_ids"],
                                        images=batch.get("images"),
                                        image_masks=batch.get("image_masks"),
                                        image_input_idx=batch.get("image_input_idx"),
                                        generation_config=gen_config,
                                    )
                        
                        # Evaluate accuracy
                        batch_accuracy = self.compute_accuracy(
                            batch=batch,
                            predictions=output,
                            metric_name=metric_name,
                        )
                        
                        all_scores.extend([s["score"] for s in batch_accuracy["per_sample_scores"]])
                        all_predictions.extend(batch_accuracy["per_sample_scores"])
                        
                        # Extract token counts
                        num_input_text_tokens = latency_results.get("num_input_text_tokens", 0)
                        num_output_tokens = latency_results.get("num_output_tokens", 0)
                        
                        # Store per-sample result
                        # Extract prediction and groundtruth from batch_accuracy
                        pred_score = batch_accuracy["per_sample_scores"][0] if batch_accuracy["per_sample_scores"] else {}
                        # Note: metadata already contains question and answers, so we don't save them separately
                        sample_result = {
                            "sample_id": batch_idx,
                            "target_vision_tokens": target_vision_tokens,
                            "target_image_size": theoretical_values.get("theoretical_image_size"),
                            "actual_vision_tokens": actual_vision_tokens,
                            "num_crops": num_crops,  # Target number of crops (from target_vision_tokens or image size)
                            "top_k": top_k,
                            "num_active_blocks": num_active_blocks,
                            "input_text_tokens": num_input_text_tokens,
                            "output_tokens": num_output_tokens,
                            # Theoretical values (from target_vision_tokens or image size)
                            "theoretical_num_crops": theoretical_values["theoretical_num_crops"],
                            "theoretical_tiling": theoretical_values["theoretical_tiling"],
                            "theoretical_image_size": theoretical_values["theoretical_image_size"],
                            "theoretical_vision_tokens": theoretical_values["theoretical_vision_tokens"],
                            # Actual values (from batch after preprocessing)
                            "actual_num_crops": actual_values["actual_num_crops"],
                            "actual_tiling": actual_values["actual_tiling"],
                            "actual_image_size": actual_values["actual_image_size"],
                            # Accuracy and prediction details
                            "accuracy": pred_score.get("score", 0.0),
                            "pred": pred_score.get("pred", ""),  # Prediction text
                            "metadata": pred_score.get("metadata", {}),  # Sample metadata (contains question, answers, image_id, etc.)
                            # Stage latencies (ms)
                            "T_vision_encoder": latency_results.get("T_vision_encoder", 0.0),
                            "T_projector": latency_results.get("T_projector", 0.0),
                            "T_vision_total": latency_results.get("T_vision_total", 0.0),
                            "T_LLM_prefill": latency_results.get("T_LLM_prefill", 0.0),
                            "T_LLM_decode": latency_results.get("T_LLM_decode", 0.0),
                            "T_total": latency_results.get("T_total", 0.0),
                            # Decode per token
                            "T_decode_per_token": latency_results.get("T_LLM_decode", 0.0) / max(num_output_tokens, 1),
                        }
                        
                        per_sample_results.append(sample_result)
                        num_processed += 1
                        
                        # Memory optimization: clear cache periodically and after each sample
                        # Only on A100 to avoid affecting H100 performance
                        if enable_memory_optimization and torch.cuda.is_available():
                            # Clear tensors after they're no longer needed
                            # Note: batch and output are already used above, safe to delete now
                            try:
                                del batch
                                del output
                                del batch_accuracy
                                # More aggressive cleanup on A100: clear every 5 samples
                                if batch_idx % 5 == 0:
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                                    # Force garbage collection every 20 samples
                                    if batch_idx % 20 == 0:
                                        import gc
                                        gc.collect()
                                        torch.cuda.empty_cache()
                            except Exception as e:
                                log.warning(f"Error clearing memory for sample {batch_idx}: {e}")
                        
                    except Exception as e:
                        log.error(f"Error processing sample {batch_idx}: {e}")
                        # Clear memory and reset CUDA state on error
                        # Always handle CUDA errors, but only aggressive cleanup on A100
                        if torch.cuda.is_available():
                            try:
                                # Reset CUDA state if there was a device-side error
                                if "CUDA error" in str(e) or "device-side assert" in str(e) or "out of memory" in str(e).lower():
                                    log.warning(f"CUDA error on sample {batch_idx}, attempting to reset...")
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                                    # Give CUDA a moment to recover
                                    import time
                                    time.sleep(0.5)
                                else:
                                    # Only do aggressive cleanup on A100
                                    if enable_memory_optimization:
                                        if 'batch' in locals():
                                            del batch
                                        torch.cuda.empty_cache()
                            except Exception as cleanup_error:
                                log.warning(f"Error during cleanup: {cleanup_error}")
                        continue
                
                # Calculate aggregate statistics
                if per_sample_results:
                    # Accuracy statistics
                    accuracy_values = [s["accuracy"] for s in per_sample_results]
                    accuracy_mean = float(np.mean(accuracy_values))
                    accuracy_std = float(np.std(accuracy_values))
                    
                    # Latency statistics
                    stage_keys = ["T_vision_encoder", "T_projector", "T_vision_total", 
                                 "T_LLM_prefill", "T_LLM_decode", "T_total", "T_decode_per_token"]
                    
                    aggregate_stats = {}
                    for key in stage_keys:
                        values = [s[key] for s in per_sample_results if key in s]
                        if values:
                            aggregate_stats[f"{key}_mean"] = float(np.mean(values))
                            aggregate_stats[f"{key}_std"] = float(np.std(values))
                            aggregate_stats[f"{key}_p50"] = float(np.percentile(values, 50))
                            aggregate_stats[f"{key}_p95"] = float(np.percentile(values, 95))
                            aggregate_stats[f"{key}_p99"] = float(np.percentile(values, 99))
                    
                    # Vision tokens statistics (actual values)
                    vision_token_values = [s["actual_vision_tokens"] for s in per_sample_results]
                    aggregate_stats["vision_tokens_mean"] = float(np.mean(vision_token_values))
                    aggregate_stats["vision_tokens_std"] = float(np.std(vision_token_values))
                    
                    # Theoretical vs actual comparison
                    theoretical_vision_tokens_values = [s.get("theoretical_vision_tokens", target_vision_tokens) for s in per_sample_results]
                    if theoretical_vision_tokens_values:
                        aggregate_stats["theoretical_vision_tokens_mean"] = float(np.mean(theoretical_vision_tokens_values))
                    
                    # Actual num_crops statistics
                    actual_num_crops_values = [s.get("actual_num_crops", 0) for s in per_sample_results]
                    if actual_num_crops_values:
                        aggregate_stats["actual_num_crops_mean"] = float(np.mean(actual_num_crops_values))
                        aggregate_stats["actual_num_crops_std"] = float(np.std(actual_num_crops_values))
                    
                    # Theoretical num_crops (should be constant)
                    theoretical_num_crops_values = [s.get("theoretical_num_crops", num_crops) for s in per_sample_results]
                    if theoretical_num_crops_values:
                        aggregate_stats["theoretical_num_crops_mean"] = float(np.mean(theoretical_num_crops_values))
                    
                    # Vision tokens mismatch statistics
                    vision_tokens_diff = [
                        s.get("theoretical_vision_tokens", target_vision_tokens) - s.get("actual_vision_tokens", 0)
                        for s in per_sample_results
                    ]
                    if vision_tokens_diff:
                        aggregate_stats["vision_tokens_diff_mean"] = float(np.mean(vision_tokens_diff))
                        aggregate_stats["vision_tokens_diff_std"] = float(np.std(vision_tokens_diff))
                        aggregate_stats["vision_tokens_diff_max"] = float(np.max(vision_tokens_diff))
                        aggregate_stats["vision_tokens_diff_min"] = float(np.min(vision_tokens_diff))
                    
                    # Get theoretical values from first sample (should be consistent across samples)
                    first_sample = per_sample_results[0] if per_sample_results else {}
                    
                    # Store results
                    config_result = {
                        "target_vision_tokens": target_vision_tokens,
                        "target_image_size": first_sample.get("theoretical_image_size", None),
                        "actual_vision_tokens_mean": aggregate_stats.get("vision_tokens_mean", 0.0),
                        "num_crops": num_crops,  # Target number of crops (calculated from target_vision_tokens)
                        "max_crops": max_crops,  # max_crops parameter passed to select_tiling (set to num_crops)
                        "top_k": top_k,
                        "num_active_blocks": num_active_blocks,
                        "num_total_blocks": total_blocks,
                        "active_block_indices": block_indices,
                        "accuracy": accuracy_mean,
                        "accuracy_std": accuracy_std,
                        "num_samples": num_processed,
                        # Theoretical values (from target_vision_tokens, should be consistent across samples)
                        "theoretical_num_crops": first_sample.get("theoretical_num_crops", num_crops),
                        "theoretical_tiling": first_sample.get("theoretical_tiling", None),
                        "theoretical_image_size": first_sample.get("theoretical_image_size", None),
                        "theoretical_vision_tokens": first_sample.get("theoretical_vision_tokens", target_vision_tokens),
                        # Aggregate statistics (includes actual values statistics)
                        "aggregate_stats": aggregate_stats,
                        "per_sample_results": per_sample_results,  # Store all samples (will be merged across ranks)
                    }
                    
                    results.append(config_result)
                    
                    # Save intermediate results (each rank saves its own results for fault tolerance)
                    # These will be merged later and the rank-specific files will be deleted
                    if self.is_distributed:
                        output_file = Path(self.output_dir) / f"combined_profiling_results_{config_idx+1}_rank{self.rank}.json"
                    else:
                        output_file = Path(self.output_dir) / f"combined_profiling_results_{config_idx+1}.json"
                    with open(output_file, 'w') as f:
                        json.dump(config_result, f, indent=2)
                    log.info(f"Rank {self.rank}: Saved intermediate results to {output_file}")
                
                # Memory optimization: clear cache between configurations
                # Only on A100 to avoid affecting H100 performance
                if enable_memory_optimization and torch.cuda.is_available():
                    # Properly shutdown dataloader workers before deleting
                    # This prevents the "can only test a child process" errors
                    try:
                        # Close any active iterator
                        if hasattr(dataloader, '_iterator'):
                            iterator = dataloader._iterator
                            if iterator is not None:
                                try:
                                    iterator._shutdown_workers()
                                except Exception:
                                    pass  # Workers may already be shut down
                        # Also try to access the iterator through iteration
                        # This ensures any pending operations complete
                        try:
                            # Consume any remaining items in the iterator
                            for _ in dataloader:
                                break  # Just consume one to trigger cleanup
                        except (StopIteration, RuntimeError):
                            pass  # Iterator is already exhausted or closed
                    except Exception as e:
                        log.debug(f"Error shutting down dataloader workers: {e}")
                    
                    # Clean up dataloader and dataset references
                    # Use explicit None assignment to help garbage collection
                    try:
                        dataloader = None
                    except Exception as e:
                        log.debug(f"Error clearing dataloader reference: {e}")
                    
                    # Clear dataset references
                    try:
                        final_dataset = None
                        if 'sampled_dataset' in locals():
                            sampled_dataset = None
                        if 'det_dataset' in locals():
                            det_dataset = None
                    except Exception as e:
                        log.debug(f"Error clearing dataset references: {e}")
                    
                    # Aggressive cache cleanup and garbage collection
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        log.info(f"Cleared GPU cache after configuration {config_idx+1}")
                    except Exception as e:
                        log.warning(f"Error clearing GPU cache: {e}")
                
            except Exception as e:
                log.error(f"Error in configuration {config_idx+1}: {e}", exc_info=True)
                # Clear memory and reset CUDA state on error
                # Always handle CUDA errors, but only aggressive cleanup on A100
                if torch.cuda.is_available():
                    try:
                        # Reset CUDA state if there was a device-side error or OOM
                        if "CUDA error" in str(e) or "device-side assert" in str(e) or "out of memory" in str(e).lower():
                            log.warning("CUDA error detected, attempting to reset CUDA state...")
                            torch.cuda.synchronize()
                            # Clear all caches and force garbage collection
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                            # Give CUDA a moment to recover
                            import time
                            time.sleep(1)
                        else:
                            # Only do aggressive cleanup on A100
                            if enable_memory_optimization:
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                torch.cuda.empty_cache()
                    except Exception as cleanup_error:
                        log.warning(f"Error during cleanup: {cleanup_error}")
                continue
        
        # Clean up
        if self.block_mask_wrapper is not None:
            self.block_mask_wrapper.remove()
        
        # Gather results from all ranks if distributed
        if self.is_distributed:
            try:
                if self.rank == 0:
                    gathered_results = [None] * self.world_size
                    dist.gather_object(results, gathered_results, dst=0)
                    
                    # Merge results from all ranks
                    # Group by config (vision_tokens, top_k, num_active_blocks)
                    config_dict = {}
                    for rank_results in gathered_results:
                        if rank_results is not None:
                            for config_result in rank_results:
                                config_key = (
                                    config_result.get("target_vision_tokens"),
                                    config_result.get("top_k"),
                                    config_result.get("num_active_blocks")
                                )
                                if config_key not in config_dict:
                                    config_dict[config_key] = config_result.copy()
                                    config_dict[config_key]["per_sample_results"] = []
                                # Merge per_sample_results
                                config_dict[config_key]["per_sample_results"].extend(
                                    config_result.get("per_sample_results", [])
                                )
                    
                    # Recompute aggregate statistics from merged results
                    merged_results = []
                    config_idx_map = {}  # Map config_key to original config_idx
                    
                    # First pass: build config_idx_map from gathered results
                    for rank_idx, rank_results in enumerate(gathered_results):
                        if rank_results is not None:
                            for orig_config_idx, config_result in enumerate(rank_results):
                                config_key = (
                                    config_result.get("target_vision_tokens"),
                                    config_result.get("top_k"),
                                    config_result.get("num_active_blocks")
                                )
                                if config_key not in config_idx_map:
                                    # Use the first rank's config_idx as the canonical one
                                    config_idx_map[config_key] = orig_config_idx
                    
                    # Second pass: merge and save each config
                    for config_key, config_result in config_dict.items():
                        all_per_sample = config_result["per_sample_results"]
                        if all_per_sample:
                            # Recompute statistics from all samples
                            accuracy_values = [s["accuracy"] for s in all_per_sample]
                            config_result["accuracy"] = float(np.mean(accuracy_values))
                            config_result["accuracy_std"] = float(np.std(accuracy_values))
                            config_result["num_samples"] = len(all_per_sample)
                            
                            # Recompute aggregate stats
                            stage_keys = ["T_vision_encoder", "T_projector", "T_vision_total", 
                                         "T_LLM_prefill", "T_LLM_decode", "T_total", "T_decode_per_token"]
                            aggregate_stats = {}
                            for key in stage_keys:
                                values = [s[key] for s in all_per_sample if key in s]
                                if values:
                                    aggregate_stats[f"{key}_mean"] = float(np.mean(values))
                                    aggregate_stats[f"{key}_std"] = float(np.std(values))
                                    aggregate_stats[f"{key}_p50"] = float(np.percentile(values, 50))
                                    aggregate_stats[f"{key}_p95"] = float(np.percentile(values, 95))
                                    aggregate_stats[f"{key}_p99"] = float(np.percentile(values, 99))
                            
                            vision_token_values = [s["actual_vision_tokens"] for s in all_per_sample]
                            aggregate_stats["vision_tokens_mean"] = float(np.mean(vision_token_values))
                            aggregate_stats["vision_tokens_std"] = float(np.std(vision_token_values))
                            
                            config_result["aggregate_stats"] = aggregate_stats
                            # Store all per_sample_results
                            config_result["per_sample_results"] = all_per_sample
                        
                        merged_results.append(config_result)
                        
                        # Save merged result for this config (one file per config, all ranks merged)
                        config_idx = config_idx_map.get(config_key, len(merged_results))
                        output_file = Path(self.output_dir) / f"combined_profiling_results_{config_idx+1}.json"
                        with open(output_file, 'w') as f:
                            json.dump(config_result, f, indent=2)
                        log.info(f"Saved merged results for config {config_idx+1} to {output_file} (merged from {self.world_size} ranks, {len(all_per_sample)} samples)")
                    
                    results = merged_results
                    log.info(f"Merged results from {self.world_size} ranks: {len(results)} configurations, total samples: {sum(r.get('num_samples', 0) for r in results)}")
                else:
                    dist.gather_object(results, None, dst=0)
                    log.info(f"Rank {self.rank}: Sent results to rank 0")
            except Exception as e:
                log.error(f"Rank {self.rank}: Failed to gather results: {e}")
                if self.rank == 0:
                    log.warning("Gather failed, using rank 0 results only")
        
        # Clean up intermediate rank-specific files (only on rank 0, after all configs are saved)
        if self.rank == 0:
            log.info(f"\n{'='*60}")
            log.info(f"Experiment completed! Results saved:")
            
            # List all saved config files (exclude rank-specific intermediate files)
            config_files = sorted(glob.glob(str(Path(self.output_dir) / "combined_profiling_results_*.json")))
            config_files = [f for f in config_files if "_rank" not in Path(f).name]
            for f in config_files:
                log.info(f"  - {Path(f).name}")
            
            # Clean up intermediate rank-specific files (only in distributed mode)
            if self.is_distributed:
                intermediate_pattern = str(Path(self.output_dir) / "combined_profiling_results_*_rank*.json")
                intermediate_files = glob.glob(intermediate_pattern)
                if intermediate_files:
                    log.info(f"\nCleaning up {len(intermediate_files)} intermediate rank files...")
                    deleted_count = 0
                    for f in intermediate_files:
                        try:
                            Path(f).unlink()
                            deleted_count += 1
                        except Exception as e:
                            log.warning(f"Failed to delete {f}: {e}")
                    log.info(f"Deleted {deleted_count}/{len(intermediate_files)} intermediate files.")
            
            log.info(f"Each configuration saved in separate file: combined_profiling_results_<config_idx>.json")
            log.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Combined Profiling: Accuracy and Latency with Vision Tokens Control")
    parser.add_argument("--model_path", type=str, default="checkpoints", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results/core_exp", help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="coco_2014_vqa", help="Dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="Max new tokens")
    parser.add_argument("--vision_tokens_list", type=int, nargs="+", default=None, 
                       help="List of target vision token values (legacy knob; overridden if image_size_list is provided)")
    parser.add_argument("--image_size_list", type=str, nargs="+", default=None,
                       help="List of target image sizes (HxW, e.g., 560x336 560x784 784x784). "
                            "Primary knob; overrides vision_tokens_list.")
    parser.add_argument("--top_k_list", type=int, nargs="+", default=None, help="List of top_k values")
    parser.add_argument("--num_active_blocks_list", type=int, nargs="+", default=None, help="List of num_active_blocks values")
    parser.add_argument("--sampling_strategy", type=str, default="balanced",
                       choices=["full", "balanced", "stratified", "boundary", "lhs"],
                       help="Sampling strategy for knob combinations")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to use (None = all)")
    parser.add_argument("--num_runs_per_sample", type=int, default=3, help="Number of runs per sample for latency averaging")
    parser.add_argument("--use_profiler", action="store_true", help="Use PyTorch profiler for detailed operator-level analysis")
    parser.add_argument("--use_profiler_on_all_samples", action="store_true", 
                       help="If set, profile all samples (slower but comprehensive). If not set, only profile first sample (faster).")
    parser.add_argument("--seed", type=int, default=66, help="Random seed for reproducibility (default: 66)")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="HF cache directory")
    parser.add_argument("--enable_memory_optimization", action="store_true",
                       help="Enable memory optimizations for limited GPU memory (e.g., A100-40GB). "
                            "Reduces dataloader workers, prefetch, and clears cache more aggressively.")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create experiment
    experiment = CombinedProfilingExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        hf_cache_dir=args.hf_cache_dir,
        seed=args.seed,  # Use seed from command line
    )
    
    # Run experiment
    experiment.run(
        dataset_name=args.dataset_name,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        vision_tokens_list=args.vision_tokens_list,
        top_k_list=args.top_k_list,
        num_active_blocks_list=args.num_active_blocks_list,
        sampling_strategy=args.sampling_strategy,
        num_samples=args.num_samples if args.num_samples > 0 else None,
        num_runs_per_sample=args.num_runs_per_sample,
        use_profiler=args.use_profiler,
        use_profiler_on_all_samples=args.use_profiler_on_all_samples,
    )


if __name__ == "__main__":
    main()

