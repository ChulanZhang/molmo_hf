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

# New Data Loading Imports
from molmo.data import get_dataset_by_name
from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
from molmo.data.data_formatter import DataFormatter
from molmo.data.collator import MMCollator
from molmo.data.dataset import DeterministicDataset
from molmo.eval.vqa import vqa_score

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ANSI Colors for debug output
class Color:
    """ANSI escape codes for colored terminal output."""
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
    """Context manager for timing code blocks."""
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        print(f"{Color.CYAN}[DEBUG] Starting: {self.name}...{Color.ENDC}")
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        print(f"{Color.CYAN}[DEBUG] Finished: {self.name} in {self.end - self.start:.2f}s{Color.ENDC}")



def move_to_device(batch, device):
    """Recursively move batch items to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    return batch

def prepare_cli_environment():
    """Set random seeds for reproducibility."""
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

class BaseExperiment(ABC):
    """
    Base class for all motivational study experiments.
    Handles model initialization, data loading, and performance measurement.
    """
    
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
        
        # Device Setup
        if device == "cuda" and torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{device_idx}")
            log.info(f"Using GPU device: {self.device} (GPU {device_idx}: {torch.cuda.get_device_name(device_idx)})")
        else:
            self.device = torch.device(device)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_warmup = num_warmup
        
        prepare_cli_environment()
        
        # Load Model and Processor
        self.model = self._load_model(model_path)
        self.processor = self._load_processor(model_path)
        self.tokenizer = self.processor.tokenizer
    
    def _load_model(self, checkpoint_dir: str):
        """Load the Molmo model from a checkpoint directory."""
        log.info(f"Loading model from {checkpoint_dir}...")
        
        # 1. Load Config
        # Strictly from project config
        project_config_path = os.path.join("configs", "model", "config.json")
        
        if os.path.exists(project_config_path):
            log.info(f"Loading config from {project_config_path}")
            config = MolmoConfig.from_json_file(project_config_path)
        else:
            log.warning("No local config found. Fetching from HF Hub...")
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
        """Load the Molmo processor (tokenizer + image processor)."""
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
            log.warning("No local tokenizer found. Fetching from HF Hub...")
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
        """
        Build a dataloader for the given dataset name.
        Uses the project's data pipeline (MultiModalPreprocessor, DataFormatter, MMCollator).
        
        Returns:
            DataLoader and the original dataset (for getting dataset length)
        """
        with Timer(f"Building dataloader for {dataset_name}/{split}"):
            # 1. Get Dataset
            try:
                dataset = get_dataset_by_name(dataset_name, split=split)
            except Exception as e:
                log.error(f"Error loading dataset {dataset_name}: {e}")
                raise e

            # 2. Configure Preprocessor
            mm_preprocessor = MultiModalPreprocessor(
                tokenizer=self.tokenizer,
                crop_mode=self.model.config.crop_mode,  # "overlap-and-resize-c2" != default "resize"
                max_crops=self.model.config.max_crops,  # 12 != default 6
                overlap_margins=self.model.config.overlap_margins,  # Good practice to be explicit
                image_padding_mask=bool(self.model.config.image_padding_embed),  # True != default False
            )
            
            # Create DataFormatter
            formatter = DataFormatter(
                prompt_templates=self.model.config.prompt_type,
                message_format=self.model.config.message_formatting,
                system_prompt=self.model.config.system_prompt_kind,
                always_start_with_space=self.model.config.always_start_with_space,
            )
            
            # Create Preprocessor wrapper
            preprocessor = Preprocessor(
                formater=formatter,
                mm_preprocessor=mm_preprocessor,
                for_inference=True,  # Critical for evaluation performance
            )
            
            # 3. Wrap in DeterministicDataset for reproducibility
            # seed=42 is hardcoded for reproducibility in motivation experiments
            det_dataset = DeterministicDataset(dataset, preprocessor, seed=42)
            
            # Store dataset length for reference
            dataset_length = len(det_dataset)
            if max_steps is not None:
                dataset_length = min(dataset_length, max_steps)
            
            # 4. Create DataLoader with MMCollator
            # Use num_workers=0 for single-process evaluation (better performance for small batches)
            dataloader = DataLoader(
                det_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=MMCollator(
                    max_sequence_length=seq_len,
                    include_metadata=True,
                    pad=True, # Enable padding for batching
                    max_crops=self.model.config.max_crops
                ),
                num_workers=0,  # 0 workers for better performance in single-process evaluation
                pin_memory=False,  # Not needed with num_workers=0
            )
            
            # Store dataset length as attribute for access
            dataloader.dataset_length = dataset_length
            
            return dataloader
    
    def measure_inference_latency(
        self,
        batch: Dict[str, torch.Tensor],
        max_new_tokens: int = 128,
        measure_components: bool = True,
        num_runs: int = 1,
        use_hook_for_llm_prefill: bool = False,
    ) -> Dict[str, float]:
        """
        Measure end-to-end inference latency and component latencies.
        
        Args:
            batch: Input batch dictionary.
            max_new_tokens: Number of tokens to generate.
            measure_components: Whether to measure individual component latencies (Vision, Prefill).
            num_runs: Number of runs to average over (for stability).
            use_hook_for_llm_prefill: If True, use forward hooks to directly measure LLM prefill.
                                     If False (default), use subtraction method (T_prefill_step - T_vision_total).
            
        Returns:
            Dictionary with latency measurements (ms) and token counts.
        """
        # 1. Move batch to device
        try:
            batch = move_to_device(batch, self.device)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
        except RuntimeError as e:
            log.error(f"Error moving batch to device: {e}")
            raise
        
        results = {}
        
        if measure_components:
            # --- Measure Vision Components ---
            if "images" in batch and batch["images"] is not None:
                vision_backbone = self.model.model.vision_backbone
                
                # Warmup (only if averaging)
                if num_runs > 1:
                    with torch.inference_mode():
                        _ = vision_backbone.encode_image(batch["images"])
                
                # 1. Measure Vision Encoder (ViT only)
                latencies_vit = []
                for _ in range(num_runs):
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        _ = vision_backbone.encode_image(batch["images"])
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    latencies_vit.append((time.perf_counter() - start) * 1000)
                results["T_vision_encoder"] = np.mean(latencies_vit)
                
                # 2. Measure Total Vision (ViT + Projector)
                # We measure the full forward pass of the vision backbone which includes the projector
                latencies_vision_total = []
                for _ in range(num_runs):
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        _ = vision_backbone(batch["images"], batch.get("image_masks"))
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    latencies_vision_total.append((time.perf_counter() - start) * 1000)
                results["T_vision_total"] = np.mean(latencies_vision_total)
                
                # Calculate Projector Latency
                results["T_projector"] = max(0.0, results["T_vision_total"] - results["T_vision_encoder"])
                
                # For backward compatibility
                results["T_vision"] = results["T_vision_total"]
            else:
                results["T_vision_encoder"] = 0.0
                results["T_vision_total"] = 0.0
                results["T_projector"] = 0.0
                results["T_vision"] = 0.0

            # --- Measure LLM Prefill ---
            # Direct measurement: Use hooks to measure LLM prefill time directly.
            # Hook on first transformer block (start) and last transformer block (end).
            # This avoids the subtraction method which can have measurement errors.
            
            if "images" in batch and batch["images"] is not None:
                transformer = self.model.model.transformer
                
                # Use hooks to measure LLM prefill time directly
                llm_prefill_times = []
                llm_start_time = None
                
                def start_hook(module, input, output):
                    nonlocal llm_start_time
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    llm_start_time = time.perf_counter()
                
                def end_hook(module, input, output):
                    nonlocal llm_start_time
                    if llm_start_time is not None:
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize(self.device)
                        end_time = time.perf_counter()
                        llm_prefill_times.append((end_time - llm_start_time) * 1000)
                        llm_start_time = None
                
                # Register hooks on first and last transformer blocks
                start_hook_handle = None
                end_hook_handle = None
                if hasattr(transformer, 'blocks') and len(transformer.blocks) > 0:
                    start_hook_handle = transformer.blocks[0].register_forward_hook(start_hook)
                    end_hook_handle = transformer.blocks[-1].register_forward_hook(end_hook)
                
                # Warmup
                if num_runs > 1:
                    with torch.inference_mode():
                        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                            _ = self.model(
                                input_ids=batch["input_ids"],
                                images=batch.get("images"),
                                image_masks=batch.get("image_masks"),
                                image_input_idx=batch.get("image_input_idx"),
                                attention_mask=batch.get("attention_mask"),
                                attention_bias=batch.get("attention_bias"),
                                position_ids=batch.get("position_ids"),
                            )
                    llm_prefill_times.clear()  # Clear warmup measurement
                
                # Measure LLM prefill using hooks
                for _ in range(num_runs):
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    with torch.inference_mode():
                        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                            _ = self.model(
                                input_ids=batch["input_ids"],
                                images=batch.get("images"),
                                image_masks=batch.get("image_masks"),
                                image_input_idx=batch.get("image_input_idx"),
                                attention_mask=batch.get("attention_mask"),
                                attention_bias=batch.get("attention_bias"),
                                position_ids=batch.get("position_ids"),
                            )
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                
                # Remove hooks
                if start_hook_handle is not None:
                    start_hook_handle.remove()
                if end_hook_handle is not None:
                    end_hook_handle.remove()
                
                if llm_prefill_times:
                    results["T_LLM_prefill"] = np.mean(llm_prefill_times)
                else:
                    # Fallback to subtraction method if hooks failed
                    log.warning("Hooks did not capture LLM prefill time, falling back to subtraction method")
                    latencies_prefill_step = []
                    for _ in range(num_runs):
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize(self.device)
                        start = time.perf_counter()
                        with torch.inference_mode():
                            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                                _ = self.model(
                                    input_ids=batch["input_ids"],
                                    images=batch.get("images"),
                                    image_masks=batch.get("image_masks"),
                                    image_input_idx=batch.get("image_input_idx"),
                                )
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize(self.device)
                        latencies_prefill_step.append((time.perf_counter() - start) * 1000)
                    
                    T_prefill_step = np.mean(latencies_prefill_step)
                    results["T_LLM_prefill"] = max(0.0, T_prefill_step - results.get("T_vision_total", 0.0))
            else:
                # No images, measure LLM prefill directly
                transformer = self.model.model.transformer
                
                # Warmup
                if num_runs > 1:
                    with torch.inference_mode():
                        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                            _ = transformer(
                                input_ids=batch["input_ids"],
                                attention_mask=batch.get("attention_mask"),
                                attention_bias=batch.get("attention_bias"),
                                position_ids=batch.get("position_ids"),
                            )
                
                latencies_llm_prefill = []
                for _ in range(num_runs):
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                            _ = transformer(
                                input_ids=batch["input_ids"],
                                attention_mask=batch.get("attention_mask"),
                                attention_bias=batch.get("attention_bias"),
                                position_ids=batch.get("position_ids"),
                            )
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    latencies_llm_prefill.append((time.perf_counter() - start) * 1000)
                
                results["T_LLM_prefill"] = np.mean(latencies_llm_prefill)
        
        # --- Measure Total Generation Latency (Decode) ---
        # Only if max_new_tokens > 0
        if max_new_tokens > 0:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    from transformers import GenerationConfig
                    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, use_cache=True)
                    
                    output = self.model.generate(
                        input_ids=batch["input_ids"],
                        images=batch.get("images"),
                        image_masks=batch.get("image_masks"),
                        image_input_idx=batch.get("image_input_idx"),
                        generation_config=generation_config,
                    )
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            results["T_total"] = (time.perf_counter() - start) * 1000
            
            # Calculate Decode Latency
            # T_total includes Vision + Prefill + Decode
            if measure_components:
                results["T_LLM_decode"] = max(0.0, results["T_total"] - results.get("T_vision_total", 0.0) - results.get("T_LLM_prefill", 0.0))
        else:
            # If no generation, we still need to measure T_total (prefill step)
            # If measure_components=False, we need to measure it directly
            if not measure_components:
                # Measure total prefill latency directly (Vision + LLM Prefill in one pass)
                latencies_total = []
                for _ in range(num_runs):
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                            _ = self.model(
                                input_ids=batch["input_ids"],
                                images=batch.get("images"),
                                image_masks=batch.get("image_masks"),
                                image_input_idx=batch.get("image_input_idx"),
                            )
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    latencies_total.append((time.perf_counter() - start) * 1000)
                results["T_total"] = np.mean(latencies_total)
            else:
                # If measure_components=True, T_total is just the prefill step (Vision + LLM Prefill)
                results["T_total"] = results.get("T_vision_total", 0.0) + results.get("T_LLM_prefill", 0.0)
            results["T_LLM_decode"] = 0.0
            output = None # No generation output

        # --- Token Counting ---
        input_ids = batch["input_ids"]
        results["num_input_text_tokens"] = int(input_ids.shape[1])
        
        # Determine output tokens
        if output is not None:
            if hasattr(output, "logits"):
                 results["num_output_tokens"] = 1
            else:
                 if output.shape[1] > input_ids.shape[1]:
                     results["num_output_tokens"] = int(output.shape[1] - input_ids.shape[1])
                 else:
                     results["num_output_tokens"] = int(output.shape[1])
        else:
            results["num_output_tokens"] = 0
        
        # Count vision tokens (if available)
        # Use image_input_idx to count valid vision tokens
        if "image_input_idx" in batch and batch["image_input_idx"] is not None:
            # image_input_idx maps vision features to input_ids positions
            # Valid entries (>=0) represent actual vision tokens used
            results["num_vision_tokens"] = int((batch["image_input_idx"] >= 0).sum().item())
        elif "images" in batch and batch["images"] is not None:
            # Fallback: estimate from images shape
            images = batch["images"]
            if len(images.shape) == 4:  # (B, num_crops, H, W, C) or similar
                # Assume 576 tokens per crop (24x24 patches)
                num_crops = images.shape[1]
                results["num_vision_tokens"] = num_crops * 576
            elif len(images.shape) == 3:  # (B, num_patches, patch_dim)
                results["num_vision_tokens"] = int(images.shape[1])
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
        Note: This is a rough estimation based on parameter counts and sequence lengths.
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
        """Compute summary statistics (mean, std, percentiles) for a list of latencies."""
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
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters for each component (Vision Encoder, Projector, LLM).
        
        Returns:
            Dictionary with parameter counts for each component.
        """
        results = {}
        
        # Vision Encoder Parameters (only image_vit)
        vision_backbone = self.model.model.vision_backbone
        if hasattr(vision_backbone, 'image_vit') and vision_backbone.image_vit is not None:
            vision_params = sum(p.numel() for p in vision_backbone.image_vit.parameters())
            results["params_vision_encoder"] = vision_params
        else:
            results["params_vision_encoder"] = 0
        
        # Projector Parameters (includes image_projector + connector components: pooling, cls_projector, pad_embed)
        # pad_embed is used after encode_image but before pooling/projector, so it's part of the connector/projector phase
        projector_params = 0
        if hasattr(vision_backbone, 'image_projector') and vision_backbone.image_projector is not None:
            projector_params += sum(p.numel() for p in vision_backbone.image_projector.parameters())
        # Add connector components to projector
        if hasattr(vision_backbone, 'image_pooling_2d') and vision_backbone.image_pooling_2d is not None:
            projector_params += sum(p.numel() for p in vision_backbone.image_pooling_2d.parameters())
        if hasattr(vision_backbone, 'cls_projector') and vision_backbone.cls_projector is not None:
            projector_params += sum(p.numel() for p in vision_backbone.cls_projector.parameters())
        if hasattr(vision_backbone, 'pad_embed') and vision_backbone.pad_embed is not None:
            # pad_embed is a nn.Parameter (tensor), not a Module, so it doesn't have .parameters()
            # It's a tensor-like object, so we can directly call numel()
            projector_params += vision_backbone.pad_embed.numel()
        results["params_projector"] = projector_params
        # Keep connector for backward compatibility (set to 0)
        results["params_connector"] = 0
        
        # LLM Parameters
        # For MoE models, we need to handle expert parameters correctly
        if hasattr(self.model.model, 'transformer') and self.model.model.transformer is not None:
            transformer = self.model.model.transformer
            
            # Check if this is a MoE model
            moe_num_experts = getattr(self.model.config, 'moe_num_experts', 0)
            moe_top_k = getattr(self.model.config, 'moe_top_k', 1)
            is_moe = moe_num_experts > 0
            
            if is_moe:
                # For MoE models, calculate both total and active parameters
                # Total parameters: all experts (model size)
                llm_params_total = sum(p.numel() for p in transformer.parameters())
                
                # Active parameters: only top_k experts per MoE layer
                # Structure: transformer.blocks[i].mlp.experts[j] contains expert parameters
                #           transformer.blocks[i].mlp.gate contains gate parameters
                #           transformer.blocks[i].mlp.ff_norm contains normalization parameters
                
                moe_expert_params = 0  # Parameters in MoE experts
                moe_gate_params = 0    # Parameters in MoE gates (always active)
                moe_norm_params = 0    # Parameters in MoE normalization (always active)
                non_moe_params = 0     # Non-MoE parameters (always active)
                
                # Count parameters by type
                for name, param in transformer.named_parameters():
                    if '.mlp.experts.' in name:
                        # This is a MoE expert parameter
                        moe_expert_params += param.numel()
                    elif '.mlp.gate' in name:
                        # This is a MoE gate parameter (always active)
                        moe_gate_params += param.numel()
                    elif '.mlp.ff_norm' in name:
                        # This is a MoE normalization parameter (always active)
                        moe_norm_params += param.numel()
                    else:
                        # Non-MoE parameter (always active): embeddings, attention, layer norms, etc.
                        non_moe_params += param.numel()
                
                # Active MoE params = (top_k / num_experts) * expert params + gate params + norm params
                active_moe_params = int((moe_top_k / moe_num_experts) * moe_expert_params) + moe_gate_params + moe_norm_params
                llm_params_active = non_moe_params + active_moe_params
                
                # Use total parameters for consistency (model size)
                results["params_llm"] = llm_params_total
                results["params_llm_active"] = llm_params_active
                results["params_llm_total"] = llm_params_total  # Alias for clarity
                results["moe_num_experts"] = moe_num_experts
                results["moe_top_k"] = moe_top_k
                results["moe_expert_params"] = moe_expert_params
                results["moe_gate_params"] = moe_gate_params
            else:
                # Non-MoE model: all parameters are active
                llm_params = sum(p.numel() for p in transformer.parameters())
                results["params_llm"] = llm_params
                results["params_llm_active"] = llm_params
                results["params_llm_total"] = llm_params
                results["moe_num_experts"] = 0
                results["moe_top_k"] = 0
        else:
            results["params_llm"] = 0
            results["params_llm_active"] = 0
            results["params_llm_total"] = 0
            results["moe_num_experts"] = 0
            results["moe_top_k"] = 0
        
        # Total Parameters (all parameters in the model)
        results["params_total"] = (
            results["params_vision_encoder"] +
            results["params_projector"] +
            results["params_llm"]
        )
        
        # Active Parameters (for MoE: only top_k experts; for non-MoE: same as total)
        results["params_active"] = (
            results["params_vision_encoder"] +
            results["params_projector"] +
            results["params_llm_active"]
        )
        
        return results
    
    def compute_accuracy(
        self,
        batch: Dict[str, torch.Tensor],
        predictions: torch.Tensor,
        metric_name: str = "vqa_score",
    ) -> Dict[str, float]:
        """
        Compute accuracy for a batch of predictions.
        
        Args:
            batch: Input batch dictionary containing metadata with answers.
            predictions: Generated token IDs (shape: [batch_size, seq_len]).
                        Can be full sequence (including input) or just generated tokens.
            metric_name: Metric to use ("vqa_score" for VQA v2).
            
        Returns:
            Dictionary with accuracy scores per sample and average.
        """
        scores = []
        per_sample_scores = []
        
        # Get input_ids to extract only generated tokens
        input_ids = batch["input_ids"]
        input_len = input_ids.shape[1]
        
        # Extract only generated tokens (after input)
        if predictions.shape[1] > input_len:
            generated_tokens = predictions[:, input_len:]
        else:
            # If predictions are shorter, assume they're already just generated tokens
            generated_tokens = predictions
        
        # Decode predictions
        pred_texts = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        
        # Get metadata from batch
        metadatas = batch.get("metadata", [])
        if not metadatas:
            # If metadata is not in batch, try to extract from other fields
            log.warning("No metadata found in batch, cannot compute accuracy")
            return {"accuracy": 0.0, "per_sample_scores": [], "num_samples": 0}
        
        for i, pred_text in enumerate(pred_texts):
            # Extract answer from prediction (handle "Answer:" prefix and newlines)
            if "Answer:" in pred_text:
                pred_text = pred_text.split("Answer:")[1].strip()
            elif "\n" in pred_text:
                # Take the last line if multiple lines
                lines = [line.strip() for line in pred_text.split("\n") if line.strip()]
                pred_text = lines[-1] if lines else pred_text.strip()
            else:
                pred_text = " ".join(pred_text.strip().split())
            
            # Get ground truth answers
            metadata = metadatas[i] if i < len(metadatas) else {}
            if "answers" in metadata:
                answers = metadata["answers"]
                if isinstance(answers, str):
                    answers = [answers]
            elif "answer" in metadata:
                answer = metadata["answer"]
                answers = [answer] if isinstance(answer, str) else answer
            else:
                log.warning(f"Sample {i} has no answers in metadata")
                scores.append(0.0)
                per_sample_scores.append({
                    "sample_id": i,
                    "score": 0.0,
                    "pred": pred_text,
                    "answers": []
                })
                continue
            
            # Compute score
            if metric_name == "vqa_score":
                score = vqa_score(answers, pred_text)
            else:
                raise NotImplementedError(f"Metric {metric_name} not implemented")
            
            scores.append(score)
            per_sample_scores.append({
                "sample_id": i,
                "score": float(score),
                "pred": pred_text,
                "answers": answers if isinstance(answers, list) else [answers]
            })
        
        avg_accuracy = np.mean(scores) if scores else 0.0
        
        return {
            "accuracy": float(avg_accuracy),
            "per_sample_scores": per_sample_scores,
            "num_samples": len(scores)
        }
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results dictionary to a JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {output_path}")
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """Abstract method to run the experiment."""
        pass
