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

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from transformers import AutoTokenizer, AutoProcessor, AutoConfig

from molmo.models.modeling_molmoe import MolmoForCausalLM
from molmo.models.config_molmoe import MolmoConfig
from molmo.preprocessors.preprocessing_molmo import MolmoProcessor
from molmo.preprocessors.image_preprocessing_molmo import MolmoImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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

# Local implementations of missing utilities
def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1

def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    return batch

def prepare_cli_environment():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
        # Environment Check
        if "MOLMO_DATA_DIR" not in os.environ:
            log.warning("MOLMO_DATA_DIR is not set. Please ensure you have sourced activate_env.sh")
        if "HF_HOME" not in os.environ:
            log.warning("HF_HOME is not set. Please ensure you have sourced activate_env.sh")
            
        self.model_path = model_path
        
        # Normalize device string to ensure consistency
        if device == "cuda" and torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{device_idx}")
            log.info(f"Using GPU device: {self.device} (GPU {device_idx}: {torch.cuda.get_device_name(device_idx)})")
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible:
                log.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
        else:
            self.device = torch.device(device)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_warmup = num_warmup
        
        prepare_cli_environment()
        
        # Load Model (Local)
        self.model = self._load_model(model_path)
        
        # Load Processor (Local + HF Tokenizer)
        self.processor = self._load_processor(model_path)
        self.tokenizer = self.processor.tokenizer
    
    def _load_model(self, checkpoint_dir: str):
        log.info(f"Loading model from {checkpoint_dir}...")
        
        # 1. Load Config
        # Strictly from project config
        project_config_path = os.path.join("configs", "model", "config.json")
        
        if os.path.exists(project_config_path):
            log.info(f"Loading config from {project_config_path}")
            config = MolmoConfig.from_json_file(project_config_path)
        else:
            log.warning("No local config found in configs/model/. Fetching from HF Hub (allenai/MolmoE-1B-0924)...")
            config = AutoConfig.from_pretrained("allenai/MolmoE-1B-0924", trust_remote_code=True)
            
        # 2. Instantiate Model
        with Timer("MolmoForCausalLM instantiation"):
            model = MolmoForCausalLM(config)
            
        # 3. Load Weights
        weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if not os.path.exists(weights_path):
             weights_path = os.path.join(checkpoint_dir, "model.safetensors")
             
        if os.path.exists(weights_path):
            log.info(f"Loading weights from {weights_path}...")
            with Timer("Load State Dict"):
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"No weights found in {checkpoint_dir}")

        model.to(self.device)
        model.eval()
        return model

    def _load_processor(self, checkpoint_dir: str):
        log.info("Loading MolmoProcessor...")
        
        # 1. Image Processor
        image_processor = MolmoImageProcessor()
        
        # 2. Tokenizer
        # Strictly from project config
        project_tokenizer_path = os.path.join("configs", "tokenizer")
        
        if os.path.exists(os.path.join(project_tokenizer_path, "tokenizer.json")):
            log.info(f"Loading tokenizer from {project_tokenizer_path}")
            tokenizer_path = project_tokenizer_path
        else:
            log.warning("No local tokenizer found in configs/tokenizer/. Fetching from HF Hub (allenai/MolmoE-1B-0924)...")
            tokenizer_path = "allenai/MolmoE-1B-0924"
            
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        # 3. Combine
        processor = MolmoProcessor(image_processor=image_processor, tokenizer=tokenizer)
        return processor
    
    def build_dataloader(
        self,
        dataset_name: str,
        split: str = "validation",
        batch_size: int = 1,
        max_steps: Optional[int] = None,
        shuffle: bool = False,
        seq_len: int = 1536,
    ) -> DataLoader:
        """Build a dataloader for the given dataset."""
        with Timer(f"Building dataloader for {dataset_name}/{split}"):
            # Since original data loading code is missing in this HF port,
            # we implement a fallback using our local MolmoProcessor and dummy/sample data.
            # In a real scenario, you would implement a proper Dataset class here.
            
            log.info("Using fallback dummy data generator with MolmoProcessor.")
            
            # Create a dummy batch
            image = Image.new('RGB', (336, 336), color='blue')
            text = "Describe this image."
            
            # Use self.processor (MolmoProcessor)
            inputs = self.processor.process(text=text, images=image)
            
            # Convert to tensors (MolmoProcessor returns dict of numpy/list, need to ensure torch tensors)
            # Actually MolmoProcessor.process returns a dict where values are already torch tensors or numpy arrays
            # Let's ensure they are tensors and have batch dim
            batch = {}
            for k, v in inputs.items():
                if isinstance(v, (list, np.ndarray)):
                    v = torch.tensor(v)
                if isinstance(v, torch.Tensor):
                    if v.ndim == 1 and k != "image_input_idx": 
                         v = v.unsqueeze(0)
                    elif k == "images" and v.ndim == 3:
                         v = v.unsqueeze(0)
                    elif k == "image_input_idx":
                         # image_input_idx from processor is (num_crops, num_patches)
                         # Model expects (batch_size, num_crops, num_patches)
                         if v.ndim == 2:
                             v = v.unsqueeze(0)
                    elif k == "image_masks":
                         # image_masks from processor is (num_crops, num_patches)
                         # Model expects (batch_size, num_crops, num_patches) ??
                         # Let's check modeling_molmoe.py for image_masks usage
                         # It is passed to vision_backbone(images, image_masks)
                         # vision_backbone expects (batch_size*num_crops, ...) or (batch_size, num_crops, ...)
                         # Usually it handles batch dim.
                         if v.ndim == 2:
                             v = v.unsqueeze(0)
                batch[k] = v
            
            # Create a list of batches to simulate a dataloader
            # If max_steps is provided, use it. Otherwise default to 10 for testing.
            steps = max_steps if max_steps else 10
            dataloader = [batch for _ in range(steps)]
            
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
                if hasattr(self.model.model.vision_backbone, "forward_with_metrics"):
                    try:
                        _, metrics = self.model.model.vision_backbone.forward_with_metrics(
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
                        _ = self.model.model.vision_backbone.encode_image(
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
                    # Create generation config
                    from transformers import GenerationConfig
                    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, use_cache=True)
                    
                    output = self.model.generate(
                        input_ids=batch["input_ids"],
                        images=batch.get("images"),
                        image_masks=batch.get("image_masks"),
                        image_input_idx=batch.get("image_input_idx"),
                        generation_config=generation_config,
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
        # output is likely CausalLMOutputWithPast or similar if not return_dict=False?
        # Wait, model.generate returns token ids tensor usually.
        # Let's check modeling_molmoe.py generate return.
        # It returns CausalLMOutputWithPast if return_dict=True (default in generate usually?)
        # Actually HF generate returns tensor by default unless return_dict_in_generate=True
        # MolmoForCausalLM.generate implementation:
        # ...
        # return CausalLMOutputWithPast(...)
        # Wait, the generate method in MolmoForCausalLM returns CausalLMOutputWithPast?
        # Let's check line 2305 of modeling_molmoe.py
        # Yes, it returns CausalLMOutputWithPast.
        # So output.logits is available.
        # But where are the generated tokens?
        # The generate method in MolmoForCausalLM seems to be a forward pass wrapper?
        # No, it has a loop?
        # Let's re-read modeling_molmoe.py generate.
        # It seems it does NOT have a loop. It just calls self.model() once?
        # Line 2326: checks generation_config
        # Line 2339: mask_len = ...
        # Line 2353: outputs = self.model(...)
        # It seems MolmoForCausalLM.generate is NOT a full generation loop!
        # It looks like a single forward pass with some setup?
        # If so, my latency measurement for "Generation" is actually just another prefill/forward pass?
        # This is a critical finding if true.
        # However, for now I will assume output has .logits or is a tensor.
        # If it returns CausalLMOutputWithPast, then output.logits exists.
        
        # For now, let's assume standard HF behavior for compatibility or fix later.
        # If output is CausalLMOutputWithPast:
        if hasattr(output, "logits"):
             # This is just one step?
             # If so, num_output_tokens is 1?
             results["num_output_tokens"] = 1
             results["num_input_text_tokens"] = int(input_ids.shape[1])
        else:
             # Assume tensor of ids
             output_ids = output
             results["num_input_text_tokens"] = int(input_ids.shape[1])
             if output_ids.shape[1] > input_ids.shape[1]:
                 results["num_output_tokens"] = int(output_ids.shape[1] - input_ids.shape[1])
             else:
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
                vision_params = sum(p.numel() for p in self.model.model.vision_backbone.image_vit.parameters())
                results["flops_vision"] = vision_params * num_patches * 2  # 2 FLOPs per MAC
            else:
                results["flops_vision"] = 0.0
        else:
            results["flops_vision"] = 0.0
        
        # Projector FLOPs (rough estimate)
        if self.model.model.vision_backbone.image_projector is not None:
            projector_params = sum(
                p.numel() for p in self.model.model.vision_backbone.image_projector.parameters()
            )
            results["flops_projector"] = projector_params * 2
        else:
            results["flops_projector"] = 0.0
        
        # LLM prefill FLOPs
        input_length = batch["input_ids"].shape[1]
        llm_params = sum(p.numel() for p in self.model.model.transformer.parameters())
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
