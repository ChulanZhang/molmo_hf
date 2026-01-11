"""
Simple model loader for controller training.
Avoids BaseExperiment abstract class issue.
"""

import logging
import os
from pathlib import Path

import torch
from transformers import AutoConfig

from molmo.models.modeling_molmoe import MolmoForCausalLM
from molmo.models.config_molmoe import MolmoConfig
from molmo.preprocessors.preprocessing_molmo import MolmoProcessor

log = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """
    Load model and tokenizer without using BaseExperiment.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to use
    
    Returns:
        model, tokenizer, processor
    """
    log.info(f"Loading model from {model_path}...")
    
    # 1. Load Config
    project_config_path = os.path.join("configs", "model", "config.json")
    
    if os.path.exists(project_config_path):
        log.info(f"Loading config from {project_config_path}")
        config = MolmoConfig.from_json_file(project_config_path)
    else:
        log.warning("No local config found. Fetching from HF Hub...")
        config = AutoConfig.from_pretrained("allenai/MolmoE-1B-0924", trust_remote_code=True)
    
    # 2. Instantiate Model
    model = MolmoForCausalLM(config)
    
    # 3. Load Weights
    # Try multiple possible locations for weights
    possible_paths = []
    
    # If model_path is a file, use it directly
    if os.path.isfile(model_path) and model_path.endswith((".bin", ".safetensors")):
        possible_paths.append(model_path)
    
    # If model_path is a directory, check inside it
    if os.path.isdir(model_path):
        possible_paths.extend([
            os.path.join(model_path, "pytorch_model.bin"),
            os.path.join(model_path, "model.safetensors"),
        ])
    
    # Also check parent directory (common case: model_path="checkpoints/molmo", weights in "checkpoints/")
    parent_dir = os.path.dirname(model_path) if os.path.isdir(model_path) else model_path
    if parent_dir and parent_dir != model_path:
        possible_paths.extend([
            os.path.join(parent_dir, "pytorch_model.bin"),
            os.path.join(parent_dir, "model.safetensors"),
        ])
    
    # Also check if model_path itself is a weight file
    if model_path.endswith((".bin", ".safetensors")):
        possible_paths.append(model_path)
    
    weights_path = None
    for path in possible_paths:
        if path and os.path.exists(path):
            weights_path = path
            break
    
    if weights_path:
        log.info(f"Loading weights from {weights_path}...")
        if weights_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
            except ImportError:
                log.warning("safetensors not available, trying torch.load...")
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        else:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(
            f"No weights found. Tried paths:\n" + 
            "\n".join([f"  - {p}" for p in possible_paths if p])
        )
    
    model.to(device)
    model.eval()
    
    # 4. Load Processor
    # Load processor from local configs (same as BaseExperiment)
    from molmo.preprocessors.multimodal_preprocessor import MolmoImageProcessor
    from transformers import AutoTokenizer
    
    log.info("Loading MolmoProcessor...")
    
    # 1. Image Processor (direct instantiation)
    image_processor = MolmoImageProcessor()
    
    # 2. Tokenizer (from local configs)
    project_tokenizer_path = os.path.join("configs", "tokenizer")
    
    if os.path.exists(os.path.join(project_tokenizer_path, "tokenizer.json")):
        log.info(f"Loading tokenizer from {project_tokenizer_path}")
        tokenizer_path = project_tokenizer_path
    else:
        log.warning("No local tokenizer found. Fetching from HF Hub...")
        tokenizer_path = "allenai/MolmoE-1B-0924"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 3. Combine into processor
    processor = MolmoProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    log.info("Model and tokenizer loaded successfully!")
    
    return model, tokenizer, processor

