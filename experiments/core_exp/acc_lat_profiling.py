"""
Combined Profiling: Accuracy and Latency with Tier-Based Vision Token Control
Tests combinations of tier-based vision tokens, MoE top_k, and transformer blocks.

Key features:
1. Tier-based vision token control: Tier (e.g., low/medium/high) → adaptive crop selection per image
2. Combined accuracy and latency measurement
3. Stage-wise latency breakdown (for E1 analysis)
4. Dataset sampling support (consistent across runs)
5. Optional PyTorch profiler (for detailed operator-level analysis)

Records detailed data for E1, E2, E3 analysis.
Each image's selected crops and actual vision tokens are recorded.
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

# ANSI color codes for highlighting key information in log messages
# These work with both colorlog and RichHandler
class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    # Bright colors for highlighting
    BRIGHT_CYAN = '\033[1;36m'
    BRIGHT_MAGENTA = '\033[1;35m'
    BRIGHT_YELLOW = '\033[1;33m'
    BRIGHT_WHITE = '\033[1;37m'
    BRIGHT_GREEN = '\033[1;32m'
    BRIGHT_BLUE = '\033[1;34m'
    # Regular colors
    CYAN = '\033[0;36m'
    YELLOW = '\033[0;33m'
    GREEN = '\033[0;32m'

# Helper functions for colored logging of key information
def log_benchmark_name(dataset_name: str, split: str):
    """Log benchmark name with highlight color"""
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}Running Combined Profiling on {Colors.BRIGHT_MAGENTA}{dataset_name}{Colors.RESET}{Colors.BRIGHT_CYAN} ({split}){Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")

def log_config_info(key: str, value: Any, highlight: bool = False):
    """Log configuration information with color"""
    if highlight:
        log.info(f"{Colors.BRIGHT_YELLOW}{key}:{Colors.RESET} {Colors.BRIGHT_WHITE}{value}{Colors.RESET}")
    else:
        log.info(f"{Colors.CYAN}{key}:{Colors.RESET} {value}")

def log_config_section(title: str):
    """Log configuration section header with color"""
    log.info(f"{Colors.BRIGHT_BLUE}{title}{Colors.RESET}")


# Tier configurations for adaptive crop selection
VISION_TOKEN_TIERS = {
    "low": {
        "name": "low",
        "min_crops": 1,
        "max_crops": 3,
        "preferred_crops": [2, 3],
        "typical_vision_tokens": 432,
        "description": "Small images, simple tasks"
    },
    "medium": {
        "name": "medium",
        "min_crops": 4,
        "max_crops": 8,
        "preferred_crops": [6, 4, 8],
        "typical_vision_tokens": 1008,
        "description": "Medium images, standard tasks"
    },
    "high": {
        "name": "high",
        "min_crops": 9,
        "max_crops": 15,
        "preferred_crops": [12, 9, 15],
        "typical_vision_tokens": 1872,
        "description": "Large images, complex tasks"
    },
}


def is_a100_gpu(device: Optional[torch.device] = None, silent: bool = False) -> bool:
    """
    Detect if the current GPU is an A100 (typically 40GB).
    
    A100-40GB has ~40GB memory, while H100 typically has 80GB.
    We use memory size as the primary indicator since GPU name detection
    can vary across systems.
    
    Args:
        device: CUDA device to check. If None, uses current device.
        silent: If True, don't log the detection result (for non-rank-0 processes).
    
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
        
        # Only log if not silent (allows rank 0 to log while other ranks stay silent)
        if not silent:
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


def _is_retryable_config_error(error: Exception) -> bool:
    """Check if an error is retryable at configuration level"""
    error_str = str(error)
    error_type = type(error).__name__
    
    # Non-retryable errors (configuration/code issues)
    non_retryable_types = [
        'AttributeError',
        'TypeError',
        'ValueError',
        'KeyError',
        'ImportError',
        'ModuleNotFoundError',
        'AssertionError',
    ]
    if error_type in non_retryable_types:
        return False
    
    # Non-retryable error messages
    non_retryable_patterns = [
        'not found',
        'does not exist',
        'invalid',
        'must be',
        'required',
    ]
    for pattern in non_retryable_patterns:
        if pattern.lower() in error_str.lower():
            return False
    
    # Retryable errors (transient issues)
    retryable_patterns = [
        'CUDA error',
        'out of memory',
        'device-side assert',
        'cuda runtime error',
    ]
    for pattern in retryable_patterns:
        if pattern.lower() in error_str.lower():
            return True
    
    # Default: retry for RuntimeError and other exceptions (conservative)
    return isinstance(error, RuntimeError) or error_type not in non_retryable_types


def _generate_config_filename(config_result: Dict[str, Any], dataset_name: str, use_tier: bool = True) -> str:
    """
    Generate a descriptive filename for a configuration result.
    
    Format (tier mode): <task_name>_imgsizetier-<tier_name>_crops<mean>_topk<k>_blocks<n>.json
    
    Args:
        config_result: Configuration result dictionary
        dataset_name: Dataset/task name (e.g., "coco_2014_vqa" -> "coco-2014-vqa")
        use_tier: If True, we're in tier-based mode (default: True)
        
    Returns:
        Filename string
    """
    # Format dataset name: replace underscores with hyphens
    task_name = dataset_name.replace("_", "-")
    
    # Tier-based mode: use tier name and selected crops mean
    tier_name = config_result.get("tier", "unknown")
    selected_crops_mean = int(config_result.get("selected_crops_mean", 0))
    img_size_str = f"tier-{tier_name}_crops{selected_crops_mean}"
    prefix = "imgsize"
    
    # Extract top_k
    top_k = config_result.get("top_k", "unknown")
    
    # Extract num_active_blocks
    num_blocks = config_result.get("num_active_blocks", "unknown")
    
    # Generate filename
    filename = f"{task_name}_{prefix}{img_size_str}_topk{top_k}_blocks{num_blocks}.json"
    return filename


def _merge_config_results(gathered_configs: List[Dict], template_config: Dict) -> Dict:
    """
    Merge configuration results from all ranks.
    
    Args:
        gathered_configs: List of config results from all ranks (may contain None for failed ranks)
        template_config: Template config result (from rank 0) for structure reference
    
    Returns:
        Merged configuration result
    """
    # Collect all per_sample_results from all ranks
    all_per_sample = []
    for rank_config in gathered_configs:
        if rank_config is not None:
            all_per_sample.extend(rank_config.get("per_sample_results", []))
    
    if not all_per_sample:
        # No samples collected, return template with empty results
        return template_config
    
    # Recompute statistics from all samples
    merged_config = template_config.copy()
    
    # Accuracy statistics
    accuracy_values = [s["accuracy"] for s in all_per_sample if "accuracy" in s]
    if accuracy_values:
        merged_config["accuracy"] = float(np.mean(accuracy_values))
        merged_config["accuracy_std"] = float(np.std(accuracy_values))
        merged_config["num_samples"] = len(all_per_sample)
    
    # Recompute latency stats
    stage_keys = ["T_vision_total", "T_LLM_prefill", "T_LLM_decode", "T_total", "T_decode_per_token"]
    latency_stats = {}
    for key in stage_keys:
        values = [s[key] for s in all_per_sample if key in s]
        if values:
            latency_stats[f"{key}_mean"] = float(np.mean(values))
            latency_stats[f"{key}_std"] = float(np.std(values))
            latency_stats[f"{key}_p50"] = float(np.percentile(values, 50))
            latency_stats[f"{key}_p95"] = float(np.percentile(values, 95))
            latency_stats[f"{key}_p99"] = float(np.percentile(values, 99))
    
    # Compute positioned decode latency statistics (per position)
    # Collect all per-step decode times by position
    # Note: EOS token also requires a forward pass, so we include all positions
    positioned_decode_times = {}  # {position: [all_times_across_samples]}
    for sample in all_per_sample:
        decode_per_step = sample.get("T_decode_per_step", [])
        if decode_per_step:
            # decode_per_step is a list of lists: [position][run]
            # Include all positions (including EOS) since each requires a forward pass
            for pos_idx, step_times in enumerate(decode_per_step):
                if step_times:  # step_times is a list of run times for this position
                    if pos_idx not in positioned_decode_times:
                        positioned_decode_times[pos_idx] = []
                    positioned_decode_times[pos_idx].extend(step_times)
    
    # Compute statistics for each position
    positioned_decode_stats = {}
    for pos_idx in sorted(positioned_decode_times.keys()):
        pos_times = positioned_decode_times[pos_idx]
        if pos_times:
            positioned_decode_stats[f"pos_{pos_idx}"] = {
                "mean": float(np.mean(pos_times)),
                "std": float(np.std(pos_times)),
                "p50": float(np.percentile(pos_times, 50)),
                "p95": float(np.percentile(pos_times, 95)),
                "p99": float(np.percentile(pos_times, 99)),
                "count": len(pos_times),  # Number of measurements at this position
            }
    
    if positioned_decode_stats:
        latency_stats["T_decode_per_step_stats"] = positioned_decode_stats
    
    # Vision tokens statistics
    vision_token_values = [s["actual_vision_tokens"] for s in all_per_sample if "actual_vision_tokens" in s]
    vision_tokens_mean = float(np.mean(vision_token_values)) if vision_token_values else 0.0
    vision_tokens_std = float(np.std(vision_token_values)) if vision_token_values else 0.0
    
    # Actual num_crops statistics
    actual_num_crops_list = [s.get("actual_num_crops", 0) for s in all_per_sample]
    selected_crops_mean = float(np.mean(actual_num_crops_list)) if actual_num_crops_list else 0.0
    selected_crops_std = float(np.std(actual_num_crops_list)) if actual_num_crops_list else 0.0
    
    # Target vision tokens statistics
    target_vision_tokens_list = [s.get("target_vision_tokens", 0) for s in all_per_sample]
    target_vision_tokens_mean = float(np.mean(target_vision_tokens_list)) if target_vision_tokens_list else 0.0
    target_vision_tokens_std = float(np.std(target_vision_tokens_list)) if target_vision_tokens_list else 0.0
    
    # Selected crops distribution
    selected_crops_distribution = {}
    for crops in actual_num_crops_list:
        selected_crops_distribution[crops] = selected_crops_distribution.get(crops, 0) + 1
    
    # Update merged_config with merged statistics (new structure)
    merged_config["selected_crops_distribution"] = selected_crops_distribution
    merged_config["selected_crops_mean"] = selected_crops_mean
    merged_config["selected_crops_std"] = selected_crops_std
    merged_config["target_vision_tokens_mean"] = target_vision_tokens_mean
    merged_config["target_vision_tokens_std"] = target_vision_tokens_std
    merged_config["actual_vision_tokens_mean"] = vision_tokens_mean
    merged_config["actual_vision_tokens_std"] = vision_tokens_std
    merged_config["latency_stats"] = latency_stats
    merged_config["per_sample_results"] = all_per_sample
    
    return merged_config


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
        # Only log output directory on rank 0 to reduce clutter
        if self.rank == 0:
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
        # Only log GPU detection on rank 0 to reduce clutter
        self.is_a100 = is_a100_gpu(self.device, silent=(self.rank != 0))
    
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
    
    def _set_max_crops(self, max_crops: int):
        """Set max_crops in model config."""
        self.model.config.max_crops = max_crops
        # Don't log max_crops setting - it's already shown in configuration header
    
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
        
        # Don't log MoE block updates - top_k is already shown in configuration header
        return moe_blocks_found
    
    def _set_active_blocks_importance_based(
        self,
        num_active: int,
        total_blocks: int,
        importance_scores: Optional[Dict[int, float]] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Set active transformer blocks using importance-based selection.
        
        Strategy:
        - Always keep block 0 (first) and block (total_blocks-1) (last)
        - For middle blocks (1 to total_blocks-2), select by importance score
        - Higher importance score = more important = keep it
        - Lower importance score = less important = prune it first
        
        Args:
            num_active: Total number of active blocks desired
            total_blocks: Total number of transformer blocks
            importance_scores: Dict mapping block_idx -> importance_score (only for middle blocks, e.g., 1-14)
        
        Returns:
            (block_mask, block_indices): Boolean mask and list of active block indices
        """
        # Always keep first and last blocks
        first_block = 0
        last_block = total_blocks - 1
        always_keep = {first_block, last_block}
        
        if importance_scores is not None and len(importance_scores) > 0:
            # Select from middle blocks (1 to total_blocks-2) based on importance
            # Higher score = more important = keep it
            # IMPORTANT: Filter out first and last blocks before sorting
            # to avoid selecting them twice (they're always kept)
            middle_blocks_only = [(idx, score) for idx, score in importance_scores.items() 
                                  if idx not in always_keep]
            middle_blocks = sorted(middle_blocks_only, key=lambda x: x[1], reverse=True)
            
            # Calculate how many middle blocks we need
            # num_active total = 2 (first + last) + num_middle
            # If num_active <= 2, only keep first and last
            if num_active <= 2:
                block_indices = sorted(list(always_keep))
            else:
                num_middle_needed = num_active - 2  # Subtract first and last
            
                # Select top-K middle blocks by importance
                selected_middle = [idx for idx, _ in middle_blocks[:num_middle_needed]]
                
                # Combine: always keep + selected middle blocks
                block_indices = sorted(list(always_keep) + selected_middle)
            
            if self.rank == 0:
                log.debug(f"Importance-based selection: keeping blocks {block_indices} "
                         f"(first={first_block}, last={last_block}, "
                         f"middle={selected_middle} from {len(middle_blocks)} candidates)")
        else:
            # Fallback: use prefix blocks (first N blocks)
            block_indices = list(range(min(num_active, total_blocks)))
            if self.rank == 0:
                log.debug(f"Prefix selection: keeping first {num_active} blocks: {block_indices}")
        
        # Ensure we don't exceed total_blocks
        block_indices = [idx for idx in block_indices if 0 <= idx < total_blocks]
        
        # Create boolean mask
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
        tier_list: List[str] = None,  # Required: list of tier names (e.g., ["low", "medium", "high"])
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
        max_config_retries: int = 3,  # Maximum retries per configuration on error
        config_retry_delay: int = 5,  # Delay between retries (seconds)
    ):
        """
        Run combined profiling experiment using tier-based vision token control.
        
        Args:
            tier_list: List of tier names (e.g., ["low", "medium", "high"]). Required.
                      Available tiers: low (1-3 crops), medium (4-8 crops), high (9-15 crops)
            num_samples: Number of samples to use from dataset (None = use all)
                        Recommended: 1000-2000 for combined profiling
            num_runs_per_sample: Number of runs per sample for latency averaging (default: 3)
            use_profiler: If True, use PyTorch profiler for detailed operator-level analysis
            use_profiler_on_all_samples: If True, profile all samples (slower but comprehensive).
                                         If False, only profile first sample (faster, default).
            profiler_activities: List of profiler activities (default: [ProfilerActivity.CUDA])
        """
        # Validate tier_list (required)
        if tier_list is None or len(tier_list) == 0:
            raise ValueError("tier_list is required. Available tiers: low, medium, high")
        
        for tier_name in tier_list:
            if tier_name not in VISION_TOKEN_TIERS:
                raise ValueError(f"Unknown tier name: {tier_name}. Available tiers: {list(VISION_TOKEN_TIERS.keys())}")
        
        # Store as instance variable for use in filename generation
        self.tier_list = tier_list
        
        # Log benchmark name with highlight (only on rank 0 to avoid duplication)
        if self.rank == 0:
            log_benchmark_name(dataset_name, split)
            
            # Concise configuration summary
            log_config_section("Experiment Configuration")
            log_config_info("Dataset", f"{dataset_name}/{split}", highlight=True)
            log_config_info("Tiers", tier_list, highlight=True)
            log_config_info("Samples", f"{num_samples if num_samples else 'all'}")
        
        if top_k_list is None:
            top_k_list = [4, 6, 8]  # Based on previous experiments
        if num_active_blocks_list is None:
            num_active_blocks_list = [12, 13, 14, 15, 16]  # Based on previous experiments
        
        # Generate combinations: tier x top_k x num_active_blocks
        combinations = []
        for tier_name in tier_list:
            tier = VISION_TOKEN_TIERS[tier_name]
            for top_k in top_k_list:
                for num_active_blocks in num_active_blocks_list:
                    combinations.append((tier, top_k, num_active_blocks))
        
        if self.rank == 0:
            log_config_info("Configurations", f"{len(combinations)} (tiers × top_k × blocks)", highlight=True)
        
        # Profiler info only if enabled
        if self.rank == 0 and use_profiler:
            log_config_info("Profiler", 'all samples' if use_profiler_on_all_samples else 'first sample only')
        
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
        from molmo.preprocessors.multimodal_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        
        dataset = get_dataset_by_name(dataset_name, split=split)
        
                # Apply sampling if specified (before creating dataloader)
        # Don't log "Will limit to X samples" - it's redundant with "Sampled X samples" below
        
        results = []
        
        # Process each configuration
        for config_idx, combo in enumerate(combinations):
            # All configurations are tier-based
            tier, top_k, num_active_blocks = combo
            tier_name = tier["name"]
            max_crops = tier["max_crops"]  # Use tier max_crops as upper bound
            
            # Check if result already exists (skip if found)
            # Use glob pattern to find matching files (crops value may vary)
            if self.rank == 0:
                task_name = self.dataset_name.replace("_", "-")
                pattern = f"{task_name}_imgsizetier-{tier_name}_crops*_topk{top_k}_blocks{num_active_blocks}.json"
                matching_files = list(Path(self.output_dir).glob(pattern))
                # Filter out rank-specific files
                matching_files = [f for f in matching_files if "_rank" not in f.name]
                
                if matching_files:
                    log.info(f"{Colors.BRIGHT_YELLOW}Config {config_idx+1}/{len(combinations)}:{Colors.RESET} "
                             f"{Colors.BRIGHT_YELLOW}tier={tier_name}{Colors.RESET}, "
                             f"{Colors.BRIGHT_YELLOW}top_k={top_k}{Colors.RESET}, "
                             f"{Colors.BRIGHT_YELLOW}blocks={num_active_blocks}{Colors.RESET} - "
                             f"{Colors.CYAN}Result already exists, skipping{Colors.RESET}")
                    # Skip this configuration - need to synchronize across ranks
                    if self.is_distributed:
                        # Broadcast skip signal to all ranks
                        skip_signal = torch.tensor([1], device=self.device)  # 1 = skip
                        dist.broadcast(skip_signal, src=0)
                    continue
                elif self.is_distributed:
                    # Broadcast continue signal to all ranks
                    skip_signal = torch.tensor([0], device=self.device)  # 0 = don't skip
                    dist.broadcast(skip_signal, src=0)
            elif self.is_distributed:
                # Other ranks wait for skip signal
                skip_signal = torch.tensor([0], device=self.device)  # 0 = don't skip
                dist.broadcast(skip_signal, src=0)
                if skip_signal.item() == 1:
                    continue  # Skip this configuration
            
            if self.rank == 0:
                # Concise configuration header (no leading newline to avoid extra blank line)
                log.info(f"{Colors.BRIGHT_GREEN}Config {config_idx+1}/{len(combinations)}:{Colors.RESET} "
                         f"{Colors.BRIGHT_YELLOW}tier={tier_name}{Colors.RESET} "
                         f"({Colors.CYAN}{tier['min_crops']}-{tier['max_crops']} crops{Colors.RESET}), "
                         f"{Colors.BRIGHT_YELLOW}top_k={top_k}{Colors.RESET}, "
                         f"{Colors.BRIGHT_YELLOW}blocks={num_active_blocks}{Colors.RESET}")
            
            # Retry loop for configuration
            config_success = False
            for retry_attempt in range(max_config_retries):
                try:
                    # Configuration details are already logged above, skip redundant tier info
                    
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
                
                # Build dataloader (always uses batch_size=1 for per-sample measurement)
                    force_full_tokens = False  # keep padding tokens invalid
                
                    mm_preprocessor = MultiModalPreprocessor(
                    tokenizer=self.tokenizer,
                    crop_mode=self.model.config.crop_mode,
                    max_crops=max_crops,  # Upper bound for crop selection
                    tier=tier,  # Tier configuration for adaptive crop selection
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
                        # Only log sampling info once per configuration (not per tier)
                        if self.rank == 0 and config_idx == 0:
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
                        batch_size=1,  # Always use batch_size=1 for per-sample measurement
                        shuffle=shuffle,
                        sampler=sampler,
                        collate_fn=MMCollator(
                            max_sequence_length=None,  # No truncation: use actual sequence length
                            include_metadata=True,
                            pad=False,  # No padding: use actual length for dynamic seq_len
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
                    
                    # Only show progress bar on rank 0 to avoid clutter
                    # Always use tqdm, but configure it to work in both TTY and non-TTY environments
                    if self.rank == 0:
                        # Use tqdm with default stderr output (not explicitly specified)
                        # This allows tqdm to detect TTY and use single-line mode
                        progress_bar = tqdm(
                            dataloader, 
                            desc=f"Progress {config_idx+1}/{len(combinations)}",
                            mininterval=0.5,  # Update at most every 0.5 seconds
                            dynamic_ncols=False,  # Fixed width to prevent line wrapping
                            ncols=100,  # Fixed width
                            leave=True,  # Keep progress bar after completion so it's visible in terminal
                            ascii=True,  # Use ASCII characters for better compatibility
                        )
                    else:
                        progress_bar = dataloader  # Use dataloader directly without tqdm
                    
                    for batch_idx, batch in enumerate(progress_bar):
                        # Process all samples from this rank's dataloader
                        # (sampling was already applied to dataset if num_samples was specified)
                        
                        try:
                            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                    for k, v in batch.items()}
                            
                            # Calculate actual vision tokens (measured from image_input_idx)
                            actual_vision_tokens = self._calculate_vision_tokens(batch)
                            
                            # Extract values directly from metadata (most accurate, no inference needed)
                            # Since we store tiling and image_size in metadata during preprocessing,
                            # we can directly read them instead of inferring from image_input_idx shape
                            metadata = batch.get("metadata")
                            if isinstance(metadata, list) and len(metadata) > 0:
                                metadata = metadata[0]  # Get first sample's metadata
                            if not isinstance(metadata, dict):
                                metadata = {}
                            
                            # Get tiling from metadata (stored during preprocessing)
                            actual_tiling = None
                            if "tiling" in metadata:
                                tiling_val = metadata["tiling"]
                                if isinstance(tiling_val, (list, tuple)) and len(tiling_val) == 2:
                                    actual_tiling = tuple(tiling_val)
                            
                            # Calculate actual_num_crops from tiling (most accurate)
                            actual_num_crops = 0
                            if actual_tiling is not None:
                                actual_num_crops = actual_tiling[0] * actual_tiling[1]
                            
                            # Get image size from metadata (stored during preprocessing)
                            actual_image_size = None
                            if "image_size" in metadata:
                                img_size = metadata["image_size"]
                                if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                                    # image_size is stored as (width, height) in metadata, convert to (height, width)
                                    actual_image_size = (img_size[1], img_size[0])
                            
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
                            
                            # Extract token counts (three separate values)
                            actual_vision_tokens_from_latency = latency_results.get("actual_vision_tokens", 0)
                            actual_text_tokens = latency_results.get("actual_text_tokens", 0)
                            total_sequence_length = latency_results.get("total_sequence_length", 0)
                            num_output_tokens = latency_results.get("num_output_tokens", 0)
                            num_content_tokens = latency_results.get("num_content_tokens", num_output_tokens)  # Excludes EOS if present
                            ends_with_eos = latency_results.get("ends_with_eos", False)
                            
                            # Store per-sample result
                            # Extract prediction and groundtruth from batch_accuracy
                            pred_score = batch_accuracy["per_sample_scores"][0] if batch_accuracy["per_sample_scores"] else {}
                            # Note: metadata already contains question and answers, so we don't save them separately
                            # Store per-sample result
                            if tier_list:
                                # Tier-based mode: record selected crops per image
                                # Calculate target_vision_tokens from actual_num_crops
                                target_vision_tokens = (actual_num_crops + 1) * 144 if actual_num_crops > 0 else 0
                                sample_result = {
                                "sample_id": batch_idx,
                                "tier": tier_name,
                                "tier_range": {"min_crops": tier["min_crops"], "max_crops": tier["max_crops"]},
                                "top_k": top_k,
                                "num_active_blocks": num_active_blocks,
                                "output_tokens": num_output_tokens,  # Total output tokens (includes EOS if present)
                                "content_tokens": num_content_tokens,  # Content tokens (excludes EOS)
                                "ends_with_eos": ends_with_eos,
                                # Image information (directly from metadata, no inference needed)
                                "actual_image_size": actual_image_size,
                                "actual_num_crops": actual_num_crops,
                                "actual_tiling": actual_tiling,
                                "target_vision_tokens": target_vision_tokens,  # Theoretical: (num_crops + 1) * 144
                                # Token counts (three separate values)
                                "actual_vision_tokens": actual_vision_tokens_from_latency,  # Actual vision tokens from image_input_idx
                                "actual_text_tokens": actual_text_tokens,  # Actual text tokens (total - vision)
                                "total_sequence_length": total_sequence_length,  # Total sequence length (vision + text)
                                # Accuracy and prediction details
                                "accuracy": pred_score.get("score", 0.0),
                                "pred": pred_score.get("pred", ""),  # Prediction text
                                "metadata": pred_score.get("metadata", {}),  # Sample metadata (contains question, answers, image_id, etc.)
                                # Stage latencies (ms)
                                "T_vision_total": latency_results.get("T_vision_total", 0.0),
                                "T_LLM_prefill": latency_results.get("T_LLM_prefill", 0.0),
                                "T_LLM_decode": latency_results.get("T_LLM_decode", 0.0),
                                "T_total": latency_results.get("T_total", 0.0),
                                # Decode per token (average per decode step)
                                # Note: EOS token also requires a forward pass, so we use num_output_tokens
                                # which includes EOS. This accurately reflects per-step decode latency.
                                "T_decode_per_token": latency_results.get("T_LLM_decode", 0.0) / max(num_output_tokens, 1),
                                # Positioned decode latency (per-step latency for each position)
                                # Format: [position][run] - e.g., [[8.5, 8.3], [9.2, 9.0], ...] for 2 runs
                                # Statistics are computed at aggregate level in _merge_config_results()
                                "T_decode_per_step": latency_results.get("T_decode_per_step", []),  # List of lists: [position][run]
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
                            # Log full stack to locate CPU-side index errors
                            log.error(f"Error processing sample {batch_idx}: {e}", exc_info=True)
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
                    else:
                        accuracy_mean = 0.0
                        accuracy_std = 0.0
                    
                    # Latency statistics
                    stage_keys = ["T_vision_total", "T_LLM_prefill", "T_LLM_decode", "T_total", "T_decode_per_token"]
                    
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
                    
                    # Actual num_crops statistics (per-image selection within tier)
                    actual_num_crops_list = [s.get("actual_num_crops", 0) for s in per_sample_results]
                    aggregate_stats["selected_crops_mean"] = float(np.mean(actual_num_crops_list)) if actual_num_crops_list else 0.0
                    aggregate_stats["selected_crops_std"] = float(np.std(actual_num_crops_list)) if actual_num_crops_list else 0.0
                    
                    # Target vision tokens statistics (theoretical for selected crops)
                    target_vision_tokens_list = [s.get("target_vision_tokens", 0) for s in per_sample_results]
                    aggregate_stats["target_vision_tokens_mean"] = float(np.mean(target_vision_tokens_list)) if target_vision_tokens_list else 0.0
                    aggregate_stats["target_vision_tokens_std"] = float(np.std(target_vision_tokens_list)) if target_vision_tokens_list else 0.0
                    
                    # Actual num_crops statistics
                    actual_num_crops_values = [s.get("actual_num_crops", 0) for s in per_sample_results]
                    if actual_num_crops_values:
                        aggregate_stats["actual_num_crops_mean"] = float(np.mean(actual_num_crops_values))
                        aggregate_stats["actual_num_crops_std"] = float(np.std(actual_num_crops_values))
                    
                    # Selected crops distribution (how many images selected each crop count)
                    selected_crops_distribution = {}
                    for crops in actual_num_crops_list:
                        selected_crops_distribution[crops] = selected_crops_distribution.get(crops, 0) + 1
                    
                    # Store results: tier-based mode only
                    # Remove duplicate stats from aggregate_stats (they're already in top level)
                    latency_stats = {}
                    for key in stage_keys:
                        if f"{key}_mean" in aggregate_stats:
                            latency_stats[f"{key}_mean"] = aggregate_stats[f"{key}_mean"]
                            latency_stats[f"{key}_std"] = aggregate_stats[f"{key}_std"]
                            latency_stats[f"{key}_p50"] = aggregate_stats[f"{key}_p50"]
                            latency_stats[f"{key}_p95"] = aggregate_stats[f"{key}_p95"]
                            latency_stats[f"{key}_p99"] = aggregate_stats[f"{key}_p99"]
                    
                    # Remove duplicate stats from aggregate_stats
                    clean_aggregate_stats = {k: v for k, v in aggregate_stats.items() 
                                           if k not in ["vision_tokens_mean", "vision_tokens_std", 
                                                       "selected_crops_mean", "selected_crops_std",
                                                       "target_vision_tokens_mean", "target_vision_tokens_std"]}
                    
                    config_result = {
                        "tier": tier_name,
                        "tier_range": {"min_crops": tier["min_crops"], "max_crops": tier["max_crops"]},
                        "max_crops": max_crops,  # max_crops parameter passed to select_tiling
                        "selected_crops_distribution": selected_crops_distribution,
                        "selected_crops_mean": aggregate_stats["selected_crops_mean"],
                        "selected_crops_std": aggregate_stats["selected_crops_std"],
                        "target_vision_tokens_mean": aggregate_stats["target_vision_tokens_mean"],
                        "target_vision_tokens_std": aggregate_stats["target_vision_tokens_std"],
                        "actual_vision_tokens_mean": aggregate_stats.get("vision_tokens_mean", 0.0),
                        "actual_vision_tokens_std": aggregate_stats.get("vision_tokens_std", 0.0),
                        "top_k": top_k,
                        "num_active_blocks": num_active_blocks,
                        "num_total_blocks": total_blocks,
                        "active_block_indices": block_indices,
                        "num_samples": num_processed,
                        "accuracy": accuracy_mean,
                        "accuracy_std": accuracy_std,
                        "latency_stats": latency_stats,
                        "per_sample_results": per_sample_results,  # Store all samples (will be merged across ranks)
                    }
                    
                    results.append(config_result)
                    
                    # Save intermediate results (each rank saves its own results for fault tolerance)
                    # Generate descriptive filename with control knob info
                    base_filename = _generate_config_filename(config_result, self.dataset_name, use_tier=True)
                    if self.is_distributed:
                        # For rank-specific files, add rank suffix before .json
                        base_name = base_filename.replace(".json", "")
                        output_file = Path(self.output_dir) / f"{base_name}_rank{self.rank}.json"
                    else:
                        output_file = Path(self.output_dir) / base_filename
                    with open(output_file, 'w') as f:
                        json.dump(config_result, f, indent=2)
                    
                    # Immediately merge results from all ranks for this configuration
                    if self.is_distributed:
                        # Gather this config's result from all ranks
                        # Only rank 0 should provide gather_list, other ranks pass None
                        if self.rank == 0:
                            gathered_configs = [None] * self.world_size
                            dist.gather_object(config_result, gathered_configs, dst=0)
                            
                            # Merge results from all ranks for this configuration
                            merged_config = _merge_config_results(gathered_configs, config_result)
                            
                            # Save merged result
                            merged_output_file = Path(self.output_dir) / base_filename
                            with open(merged_output_file, 'w') as f:
                                json.dump(merged_config, f, indent=2)
                            
                            # Clean up rank-specific files for this config immediately
                            base_name = base_filename.replace(".json", "")
                            for rank_idx in range(self.world_size):
                                rank_file = Path(self.output_dir) / f"{base_name}_rank{rank_idx}.json"
                                if rank_file.exists():
                                    try:
                                        rank_file.unlink()
                                    except Exception:
                                        pass  # Silently ignore cleanup errors
                        else:
                            # Other ranks: pass None as gather_list
                            dist.gather_object(config_result, None, dst=0)
                    else:
                        # Non-distributed: file already saved above
                        pass
                    
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
                
                    # Configuration completed successfully
                    config_success = True
                    break  # Exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    error_type = type(e).__name__
                    
                    # Log error
                    if retry_attempt == 0:
                        log.error(f"Error in configuration {config_idx+1}: {e}", exc_info=True)
                    else:
                        log.warning(f"Error in configuration {config_idx+1} (retry {retry_attempt}/{max_config_retries-1}): {e}")
                    
                    # Check if error is retryable
                    is_retryable = _is_retryable_config_error(e)
                    
                    # Clear memory and reset CUDA state on error
                    # Always handle CUDA errors, but only aggressive cleanup on A100
                    if torch.cuda.is_available():
                        try:
                            # Reset CUDA state if there was a device-side error or OOM
                            if "CUDA error" in error_str or "device-side assert" in error_str or "out of memory" in error_str.lower():
                                log.warning("CUDA error detected, attempting to reset CUDA state...")
                                torch.cuda.synchronize()
                                # Clear all caches and force garbage collection
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                torch.cuda.empty_cache()
                                # Give CUDA a moment to recover
                                time.sleep(config_retry_delay)
                            else:
                                # Only do aggressive cleanup on A100
                                if enable_memory_optimization:
                                    torch.cuda.empty_cache()
                                    import gc
                                    gc.collect()
                                    torch.cuda.empty_cache()
                        except Exception as cleanup_error:
                            log.warning(f"Error during cleanup: {cleanup_error}")
                    
                    # Decide whether to retry
                    if not is_retryable:
                        log.error(f"Non-retryable error in configuration {config_idx+1}, skipping: {error_type}")
                        break  # Exit retry loop, skip this configuration
                    elif retry_attempt < max_config_retries - 1:
                        log.warning(f"Retryable error in configuration {config_idx+1}, retrying in {config_retry_delay}s... "
                                  f"(attempt {retry_attempt + 1}/{max_config_retries})")
                        time.sleep(config_retry_delay)
                    else:
                        log.error(f"Configuration {config_idx+1} failed after {max_config_retries} attempts, skipping")
                        break  # Exit retry loop, skip this configuration
                
            # If configuration failed after all retries, continue to next configuration
            if not config_success:
                if self.rank == 0:
                    log.warning(f"Skipping configuration {config_idx+1} due to persistent errors")
                continue
        
        # Clean up
        if self.block_mask_wrapper is not None:
            self.block_mask_wrapper.remove()
        
        # Results are already merged and saved per configuration (immediate merge)
        # No need for final merge step
        
        # Clean up any remaining intermediate rank-specific files (safety check)
        if self.rank == 0:
            # Log completion summary (concise)
            task_name_pattern = self.dataset_name.replace("_", "-")
            config_files = sorted(glob.glob(str(Path(self.output_dir) / f"{task_name_pattern}_imgsizetier-*_topk*_blocks*.json")))
            config_files = [f for f in config_files if "_rank" not in Path(f).name]
            
            log.info(f"{Colors.BRIGHT_GREEN}{'='*60}{Colors.RESET}")
            log.info(f"{Colors.BRIGHT_GREEN}Experiment completed!{Colors.RESET}")
            log.info(f"{Colors.CYAN}Results: {len(config_files)} configuration(s) saved to {self.output_dir}{Colors.RESET}")
            # Clean up intermediate rank-specific files silently
            if self.is_distributed:
                task_name_pattern = self.dataset_name.replace("_", "-")
                intermediate_pattern = str(Path(self.output_dir) / f"{task_name_pattern}_imgsizetier-*_topk*_blocks*_rank*.json")
                intermediate_files = glob.glob(intermediate_pattern)
                for f in intermediate_files:
                    try:
                        Path(f).unlink()
                    except Exception:
                        pass  # Silently ignore cleanup errors
            
            log.info(f"{Colors.BRIGHT_GREEN}{'='*60}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="Combined Profiling: Accuracy and Latency with Vision Tokens Control")
    parser.add_argument("--model_path", type=str, default="checkpoints", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results/core_exp", help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="coco_2014_vqa", help="Dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="Max new tokens")
    parser.add_argument("--tier_list", type=str, nargs="+", required=True,
                       help="List of tier names (required). Available tiers: low, medium, high. "
                            "Examples: --tier_list low medium high")
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
    parser.add_argument("--max_config_retries", type=int, default=3,
                       help="Maximum retries per configuration on error (default: 3)")
    parser.add_argument("--config_retry_delay", type=int, default=5,
                       help="Delay between configuration retries in seconds (default: 5)")
    parser.add_argument("--importance_scores_file", type=str, default=None,
                       help="Path to JSON file containing importance scores for block selection")
    
    args = parser.parse_args()
    
    # Setup logging with colors
    # Default: Use colorlog (user has installed it)
    # Fallback: Use RichHandler (already in project)
    try:
        import colorlog
        handler = colorlog.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)  # Only show INFO and above by default
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Root logger: INFO and above
        root_logger.handlers = []
        root_logger.addHandler(handler)
        
        # Set third-party loggers to INFO to reduce noise
        logging.getLogger('PIL').setLevel(logging.INFO)
        logging.getLogger('PIL.PngImagePlugin').setLevel(logging.INFO)
        logging.getLogger('PIL.Image').setLevel(logging.INFO)
        
        # Only our own loggers use DEBUG
        logging.getLogger('experiments').setLevel(logging.DEBUG)
        logging.getLogger('molmo').setLevel(logging.DEBUG)
    except (ImportError, Exception):
        # Fallback to RichHandler (already in project)
        from molmo.util import SafeRichHandler
        handler = SafeRichHandler()
        handler.setLevel(logging.INFO)  # Only show INFO and above by default
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Root logger: INFO and above
        root_logger.handlers = []
        root_logger.addHandler(handler)
        
        # Set third-party loggers to INFO to reduce noise
        logging.getLogger('PIL').setLevel(logging.INFO)
        logging.getLogger('PIL.PngImagePlugin').setLevel(logging.INFO)
        logging.getLogger('PIL.Image').setLevel(logging.INFO)
        
        # Only our own loggers use DEBUG
        logging.getLogger('experiments').setLevel(logging.DEBUG)
        logging.getLogger('molmo').setLevel(logging.DEBUG)
    
    # Our own logger can use DEBUG
    log.setLevel(logging.DEBUG)
    
    # Create experiment
    experiment = CombinedProfilingExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        hf_cache_dir=args.hf_cache_dir,
        seed=args.seed,  # Use seed from command line
    )
    
    # Load importance scores if provided
    importance_scores = None
    if hasattr(args, 'importance_scores_file') and args.importance_scores_file:
        import json
        log.info(f"Loading importance scores from {args.importance_scores_file}")
        with open(args.importance_scores_file, 'r') as f:
            saved_data = json.load(f)
            importance_scores = {int(k): float(v) for k, v in saved_data.items()}
        log.info(f"Loaded importance scores for {len(importance_scores)} blocks")
    
    # Run experiment
    experiment.run(
        dataset_name=args.dataset_name,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        tier_list=args.tier_list,
        top_k_list=args.top_k_list,
        num_active_blocks_list=args.num_active_blocks_list,
        sampling_strategy=args.sampling_strategy,
        num_samples=args.num_samples if args.num_samples > 0 else None,
        num_runs_per_sample=args.num_runs_per_sample,
        importance_scores=importance_scores,
        use_profiler=args.use_profiler,
        use_profiler_on_all_samples=args.use_profiler_on_all_samples,
        max_config_retries=args.max_config_retries,
        config_retry_delay=args.config_retry_delay,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up distributed process group to avoid warnings
        if dist.is_initialized():
            dist.destroy_process_group()

