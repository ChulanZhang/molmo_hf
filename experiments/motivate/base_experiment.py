"""
Base class for motivational study experiments.
Provides common functionality for model loading, data loading, and latency measurement.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

# ANSI Colors
class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Timer:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        print(f"{Color.CYAN}[DEBUG] Starting: {self.name}...{Color.ENDC}")
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        print(f"{Color.CYAN}[DEBUG] Finished: {self.name} in {self.end - self.start:.2f}s{Color.ENDC}")

# Ensure MOLMO_DATA_DIR is set before importing olmo modules
# This must happen at module level to ensure it's set before any data loading
if "MOLMO_DATA_DIR" not in os.environ:
    # Try to use default path if not set
    default_path = "/anvil/projects/x-cis250705/data/vlm/molmo"
    if os.path.exists(default_path):
        os.environ["MOLMO_DATA_DIR"] = default_path
        log = logging.getLogger(__name__)
        log.info(f"MOLMO_DATA_DIR not set, using default: {default_path}")
    else:
        log = logging.getLogger(__name__)
        log.warning("MOLMO_DATA_DIR is not set and default path does not exist. Data loading might fail.")

# Set HuggingFace cache directory BEFORE importing any HuggingFace-related modules
# This must happen at module level to ensure it's set before any imports
def _setup_hf_cache_early():
    """Setup HF cache directory early, before any HuggingFace imports."""
    # Priority: use HF_HOME if set (e.g., from activate_env_anvil.sh)
    if "HF_HOME" in os.environ:
        cache_path = Path(os.environ["HF_HOME"])
    else:
        # Only set default if not already set
        cache_path = Path(os.path.expanduser("~/.cache/huggingface"))
        os.environ["HF_HOME"] = str(cache_path)
    
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Set all related environment variables
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
    os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "hub")
    
    # Ensure subdirectories exist
    (cache_path / "transformers").mkdir(parents=True, exist_ok=True)
    (cache_path / "hub").mkdir(parents=True, exist_ok=True)
    
    # Log for debugging
    import logging
    log = logging.getLogger(__name__)
    log.info(f"Early HF cache setup: HF_HOME={os.environ.get('HF_HOME')}")

_setup_hf_cache_early()

import numpy as np
import torch
from torch.utils.data import DataLoader

from olmo import Molmo
from olmo.config import DataConfig, ModelConfig
from olmo.data import build_torch_mm_eval_dataloader
from olmo.torch_util import move_to_device, get_world_size
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)

# Try to import get_evaluation from launch_scripts, fallback to manual config
try:
    from launch_scripts.utils import get_evaluation
    USE_GET_EVALUATION = True
except (ImportError, Exception) as e:
    USE_GET_EVALUATION = False
    log.warning(f"Could not import get_evaluation from launch_scripts.utils: {e}, using manual config")

log = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Base class for all motivational study experiments."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        output_dir: str = "./results",
        num_warmup: int = 3,
        hf_cache_dir: Optional[str] = None,
    ):
        self.model_path = model_path
        
        # Normalize device string to ensure consistency
        # In SLURM/CUDA_VISIBLE_DEVICES environments, we need to be explicit about device index
        if device == "cuda" and torch.cuda.is_available():
            # Use current device index to ensure consistency
            device_idx = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{device_idx}")
            log.info(f"Using GPU device: {self.device} (GPU {device_idx}: {torch.cuda.get_device_name(device_idx)})")
            # Check CUDA_VISIBLE_DEVICES if set
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible:
                log.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
        else:
            self.device = torch.device(device)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_warmup = num_warmup
        
        # Set HuggingFace cache directory FIRST, before any HuggingFace imports/usage
        # This must happen before prepare_cli_environment() and model loading
        self._setup_hf_cache(hf_cache_dir)
        
        prepare_cli_environment()
        self.model = self._load_model(self.model_path)
        
        # Ensure cache is still set before getting tokenizer
        # (in case prepare_cli_environment or model loading changed it)
        self._setup_hf_cache(hf_cache_dir)
        
        # Double-check environment variables before calling get_tokenizer
        # This is critical because HuggingFace may have cached the cache dir
        if hf_cache_dir:
            cache_path = Path(hf_cache_dir)
        elif "HF_HOME" in os.environ:
            cache_path = Path(os.environ["HF_HOME"])
        else:
            cache_path = Path(os.path.expanduser("~/.cache/huggingface"))
        
        # Force set all environment variables again right before tokenizer call
        os.environ["HF_HOME"] = str(cache_path)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
        os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "hub")
        
        # Ensure directories exist
        cache_path.mkdir(parents=True, exist_ok=True)
        (cache_path / "transformers").mkdir(parents=True, exist_ok=True)
        (cache_path / "hub").mkdir(parents=True, exist_ok=True)
        
        log.info(f"Final HF cache check before tokenizer:")
        log.info(f"  HF_HOME={os.environ.get('HF_HOME')}")
        log.info(f"  TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')}")
        log.info(f"  HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
        log.info(f"  HUGGINGFACE_HUB_CACHE={os.environ.get('HUGGINGFACE_HUB_CACHE')}")
        
        # Import huggingface_hub and force set cache dir
        try:
            import huggingface_hub
            # Try to set cache dir directly in huggingface_hub
            if hasattr(huggingface_hub, 'constants'):
                huggingface_hub.constants.HF_HUB_CACHE = str(cache_path / "hub")
            # Also try to set it in the file_download module
            if hasattr(huggingface_hub, 'file_download'):
                try:
                    huggingface_hub.file_download.default_hf_hub_cache = str(cache_path / "hub")
                except:
                    pass
            # Try to set in _hf_hub_download_to_cache_dir if available
            try:
                from huggingface_hub.file_download import _hf_hub_download_to_cache_dir
                # Monkey patch to use our cache dir
                original_func = _hf_hub_download_to_cache_dir
                def patched_download(*args, **kwargs):
                    kwargs['cache_dir'] = str(cache_path / "hub")
                    return original_func(*args, **kwargs)
                import huggingface_hub.file_download
                huggingface_hub.file_download._hf_hub_download_to_cache_dir = patched_download
            except Exception as e:
                log.debug(f"Could not patch huggingface_hub.file_download: {e}")
        except Exception as e:
            log.warning(f"Could not set huggingface_hub cache directly: {e}")
        
        # Final verification before calling get_tokenizer
        log.info(f"About to call get_tokenizer() with:")
        log.info(f"  HF_HOME={os.environ.get('HF_HOME')}")
        log.info(f"  HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
        log.info(f"  HUGGINGFACE_HUB_CACHE={os.environ.get('HUGGINGFACE_HUB_CACHE')}")
        self.tokenizer = self.model.config.get_tokenizer()
    
    def _setup_hf_cache(self, cache_dir: Optional[str] = None):
        """Setup HuggingFace cache directory to avoid permission errors.
        
        Priority:
        1. cache_dir parameter (if provided)
        2. HF_HOME environment variable (e.g., from activate_env_anvil.sh)
        3. Default to ~/.cache/huggingface
        """
        if cache_dir:
            # Use provided cache directory
            cache_path = Path(cache_dir)
        elif "HF_HOME" in os.environ:
            # Use HF_HOME if set (e.g., from activate_env_anvil.sh)
            cache_path = Path(os.environ["HF_HOME"])
        else:
            # Default to user's home
            cache_path = Path(os.path.expanduser("~/.cache/huggingface"))
        
        # Create cache directory if it doesn't exist
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for HuggingFace
        # These must be set before any HuggingFace library calls
        os.environ["HF_HOME"] = str(cache_path)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
        os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
        # Also set HUGGINGFACE_HUB_CACHE for compatibility
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "hub")
        
        # Create subdirectories
        (cache_path / "transformers").mkdir(parents=True, exist_ok=True)
        (cache_path / "hub").mkdir(parents=True, exist_ok=True)
        
        # Verify the environment variables are actually set
        actual_hf_home = os.environ.get("HF_HOME")
        if actual_hf_home != str(cache_path):
            log.warning(f"HF_HOME mismatch: expected {cache_path}, got {actual_hf_home}")
            # Force set it again
            os.environ["HF_HOME"] = str(cache_path)
        
        log.info(f"HuggingFace cache directory set to: {cache_path}")
        log.info(f"  HF_HOME={os.environ.get('HF_HOME')}")
        log.info(f"  TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')}")
        log.info(f"  HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
        log.info(f"  HUGGINGFACE_HUB_CACHE={os.environ.get('HUGGINGFACE_HUB_CACHE')}")
        

    def _load_model(self, checkpoint_dir: str) -> Molmo:
        """Load model from checkpoint."""
        with Timer(f"Loading model from {checkpoint_dir}"):
            log.info(f"{Color.BLUE}Loading model from {checkpoint_dir}...{Color.ENDC}")
            
            # Check if it's a HuggingFace model ID
            if not os.path.exists(checkpoint_dir) and "/" in checkpoint_dir and not checkpoint_dir.startswith("hf:"):
                log.info(f"{Color.BLUE}Detected HuggingFace model ID, using: hf:{checkpoint_dir}{Color.ENDC}")
                checkpoint_dir = f"hf:{checkpoint_dir}"
            
            # Load model
            try:
                with Timer("Molmo.from_checkpoint"):
                    model = Molmo.from_checkpoint(
                        checkpoint_dir,
                        device=str(self.device),
                    )
                
                # Verify device placement
                model_device = next(model.parameters()).device
                log.info(f"{Color.GREEN}Model verified on {model_device}{Color.ENDC}")
                
                if model_device.type != self.device.type:
                    log.warning(f"{Color.WARNING}Model loaded on {model_device}, expected {self.device}. Moving...{Color.ENDC}")
                    model = model.to(self.device)
                
                log.info(f"{Color.GREEN}Model loaded successfully on {self.device}{Color.ENDC}")
                return model
                
            except Exception as e:
                log.error(f"{Color.FAIL}Failed to load model: {e}{Color.ENDC}")
                raise
    
    def build_dataloader(
        self,
        dataset_name: str,
        split: str = "validation",
        batch_size: int = 1,
        max_steps: Optional[int] = None,
        shuffle: bool = False,
        seq_len: int = 1536,
    ) -> DataLoader:
        """Build a dataloader for the given dataset.
        
        Uses get_evaluation from launch_scripts if available for consistency
        with standard evaluation pipeline.
        """
        with Timer(f"Building dataloader for {dataset_name}/{split}"):
            if USE_GET_EVALUATION:
                # Use the same evaluation config as eval_downstream.py
                task_name = f"{dataset_name}:{split}" if split else dataset_name
                with Timer("get_evaluation config"):
                    eval_config = get_evaluation(
                        name=task_name,
                        seq_len=seq_len,
                        batch_size=batch_size * get_world_size(),
                        max_examples=max_steps if max_steps else -1,
                        num_workers=0,  # Use 0 for single-process experiments
                    )
                data_config = eval_config.data
            else:
                # Fallback to manual config
                data_config = DataConfig(
                    dataset=dataset_name,
                    split=split,
                    multi_modal="torch",
                    pad=True,
                    drop_last=False,
                    shuffle_messages=shuffle,
                    for_inference=True,
                    num_workers=0,
                    pin_memory=False,
                    sequence_length=seq_len,
                )
            
            with Timer("build_torch_mm_eval_dataloader"):
                dataloader = build_torch_mm_eval_dataloader(
                    batch_size=batch_size,
                    seed=42,
                    model_config=self.model.config,
                    data_config=data_config,
                    pad_batches=False,
                    max_steps=max_steps,
                )
            return dataloader
    
    def measure_inference_latency(
        self,
        batch: Dict[str, torch.Tensor],
        max_new_tokens: int = 128,
        measure_components: bool = True,
    ) -> Dict[str, float]:
        """
        Measure end-to-end inference latency and optionally component latencies.
        
        Returns:
            Dictionary with latency measurements in milliseconds:
            - T_total: Total end-to-end latency
            - T_vision: Vision encoder latency (if measure_components)
            - T_projector: Projector latency (if measure_components)
            - T_LLM_prefill: LLM prefill latency (if measure_components)
            - T_LLM_decode: LLM decode latency (if measure_components)
        """
        # Move batch to device with detailed error handling
        try:
            batch = move_to_device(batch, self.device)
            
            # Verify critical tensors are on correct device and accessible
            if self.device.type == 'cuda' and "input_ids" in batch:
                input_device = batch["input_ids"].device
                # Normalize device comparison - handle cuda vs cuda:0
                if input_device.type == 'cuda' and self.device.type == 'cuda':
                    # Both are CUDA, check if indices match
                    if input_device.index != self.device.index:
                        log.info(f"input_ids on {input_device}, moving to {self.device} for consistency")
                        batch["input_ids"] = batch["input_ids"].to(self.device)
                elif input_device != self.device:
                    log.warning(f"input_ids on wrong device: {input_device}, moving to {self.device}")
                    batch["input_ids"] = batch["input_ids"].to(self.device)
                
                # Test access to input_ids to catch memory errors early
                try:
                    _ = batch["input_ids"].shape
                    _ = batch["input_ids"].dtype
                    # Try a simple operation to verify memory is valid
                    test_mask = batch["input_ids"] != -1
                    _ = test_mask.shape
                except RuntimeError as e:
                    error_str = str(e)
                    if "illegal memory access" in error_str or "CUDA" in error_str:
                        log.error(f"input_ids has illegal memory access: {error_str}")
                        log.error("GPU memory appears corrupted. Please restart Python process.")
                        raise RuntimeError(f"GPU memory error in input_ids: {error_str}") from e
                    raise
        except RuntimeError as e:
            error_str = str(e)
            if "illegal memory access" in error_str or "CUDA" in error_str:
                log.error(f"CUDA error when moving batch to device: {error_str}")
                log.error("This suggests GPU memory corruption. Please restart Python process.")
                raise RuntimeError(f"GPU memory error during data transfer: {error_str}") from e
            else:
                raise
        
        if self.device.type == 'cuda':
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError as e:
                error_str = str(e)
                if "illegal memory access" in error_str:
                    log.error(f"CUDA error during synchronize: {error_str}")
                    log.error("GPU appears to be in bad state. Please restart Python process.")
                    raise RuntimeError(f"GPU error during synchronize: {error_str}") from e
                raise
        
        results = {}
        
        if measure_components:
            # Measure vision encoder and projector
            if "images" in batch and batch["images"] is not None:
                # Check if we can use the instrumented method
                if hasattr(self.model.vision_backbone, "forward_with_metrics"):
                    try:
                        _, metrics = self.model.vision_backbone.forward_with_metrics(
                            batch["images"], 
                            batch.get("image_masks")
                        )
                        results["T_vision"] = metrics.get("T_vision", 0.0)
                        results["T_projector"] = metrics.get("T_projector", 0.0)
                    except Exception as e:
                        log.warning(f"forward_with_metrics failed: {e}")
                        results["T_vision"] = 0.0
                        results["T_projector"] = 0.0
                else:
                    # Fallback to manual measurement (less accurate for projector)
                    start = time.perf_counter()
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    
                    with torch.inference_mode():
                        _ = self.model.vision_backbone.encode_image(
                            batch["images"]
                        )
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    results["T_vision"] = (time.perf_counter() - start) * 1000
                    results["T_projector"] = 0.0
            else:
                results["T_vision"] = 0.0
                results["T_projector"] = 0.0

            # Measure LLM Prefill explicitly
            # This runs a single forward pass with the inputs
            start = time.perf_counter()
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    try:
                        _ = self.model(
                            input_ids=batch["input_ids"],
                            images=batch.get("images"),
                            image_masks=batch.get("image_masks"),
                            image_input_idx=batch.get("image_input_idx"),
                        )
                    except Exception as e:
                        log.warning(f"Prefill measurement failed: {e}")
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            
            T_total_prefill = (time.perf_counter() - start) * 1000
            # T_total_prefill includes a re-run of vision/projector
            # So we subtract the measured vision/projector times to isolate LLM prefill
            results["T_LLM_prefill"] = max(0.0, T_total_prefill - results.get("T_vision", 0.0) - results.get("T_projector", 0.0))
        
        # Measure total end-to-end latency (Generation)
        start = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                # Clear cache before generate
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                try:
                    output = self.model.generate(
                        input_ids=batch["input_ids"],
                        images=batch.get("images"),
                        image_masks=batch.get("image_masks"),
                        image_input_idx=batch.get("image_input_idx"),
                        max_steps=max_new_tokens,
                        is_distributed=False,
                    )
                except RuntimeError as e:
                    # ... (error handling omitted for brevity, keeping existing logic if possible, but simplified here)
                    log.error(f"Generation failed: {e}")
                    raise
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        results["T_total"] = (time.perf_counter() - start) * 1000
        
        # Calculate Decode Latency
        # T_total includes Vision + Prefill + Decode
        # So T_decode = T_total - T_vision - T_LLM_prefill
        # Note: T_vision is only non-zero if we measured it above, but T_total includes it implicitly
        # If measure_components is False, we can't calculate breakdown
        if measure_components:
            results["T_LLM_decode"] = max(0.0, results["T_total"] - results["T_vision"] - results["T_LLM_prefill"])
        
        # Count tokens
        input_ids = batch["input_ids"]
        output_ids = output.token_ids[:, 0]  # beam_size=1
        
        results["num_input_text_tokens"] = int(input_ids.shape[1])
        
        # Debug logging for token shapes
        log.debug(f"Input shape: {input_ids.shape}, Output shape: {output_ids.shape}")
        
        # Output tokens logic
        if output_ids.shape[1] > input_ids.shape[1]:
            # Likely full sequence (input + new)
            results["num_output_tokens"] = int(output_ids.shape[1] - input_ids.shape[1])
        else:
            # Likely just new tokens
            results["num_output_tokens"] = int(output_ids.shape[1])
        
        # Count vision tokens (if available)
        if "images" in batch and batch["images"] is not None:
            images = batch["images"]
            if len(images.shape) == 4:  # (B, H, W, C) or similar
                results["num_vision_tokens"] = images.shape[1] * images.shape[2] if len(images.shape) == 4 else 0
            elif len(images.shape) == 3:  # (B, num_patches, patch_dim)
                results["num_vision_tokens"] = images.shape[1]
            else:
                results["num_vision_tokens"] = 0
        else:
            results["num_vision_tokens"] = 0
        
        return results
    
    def count_flops(
        self,
        batch: Dict[str, torch.Tensor],
        output_length: int,
    ) -> Dict[str, float]:
        """
        Estimate FLOPs for the inference.
        
        Returns:
            Dictionary with FLOP estimates:
            - flops_vision: Vision encoder FLOPs
            - flops_projector: Projector FLOPs
            - flops_llm_prefill: LLM prefill FLOPs
            - flops_llm_decode: LLM decode FLOPs
            - flops_total: Total FLOPs
        """
        results = {}
        
        # Vision encoder FLOPs (rough estimate)
        if "images" in batch and batch["images"] is not None:
            images = batch["images"]
            # Rough estimate: assume ViT-like architecture
            # This is a simplified calculation
            if len(images.shape) >= 3:
                num_patches = images.shape[1] if len(images.shape) == 3 else images.shape[1] * images.shape[2]
                # Rough estimate: ~6 FLOPs per parameter per patch
                vision_params = sum(p.numel() for p in self.model.vision_backbone.image_vit.parameters())
                results["flops_vision"] = vision_params * num_patches * 2  # 2 FLOPs per MAC
            else:
                results["flops_vision"] = 0.0
        else:
            results["flops_vision"] = 0.0
        
        # Projector FLOPs (rough estimate)
        if self.model.vision_backbone.image_projector is not None:
            projector_params = sum(
                p.numel() for p in self.model.vision_backbone.image_projector.parameters()
            )
            results["flops_projector"] = projector_params * 2
        else:
            results["flops_projector"] = 0.0
        
        # LLM prefill FLOPs
        input_length = batch["input_ids"].shape[1]
        llm_params = sum(p.numel() for p in self.model.transformer.parameters())
        # Prefill: process all input tokens
        results["flops_llm_prefill"] = llm_params * input_length * 2
        
        # LLM decode FLOPs
        # Decode: process one token at a time, but with growing KV cache
        # Simplified: assume linear scaling with output length
        results["flops_llm_decode"] = llm_params * output_length * 2
        
        results["flops_total"] = (
            results["flops_vision"]
            + results["flops_projector"]
            + results["flops_llm_prefill"]
            + results["flops_llm_decode"]
        )
        
        return results
    
    def compute_statistics(self, latencies: List[float]) -> Dict[str, float]:
        """Compute P50, P95, P99, mean, std from latency list."""
        latencies = np.array(latencies)
        return {
            "P50": float(np.percentile(latencies, 50)),
            "P95": float(np.percentile(latencies, 95)),
            "P99": float(np.percentile(latencies, 99)),
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        }
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {output_path}")
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the experiment. Must be implemented by subclasses."""
        pass

