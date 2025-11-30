"""
Molmo: Open Vision-Language Models by Allen Institute for AI

This is a HuggingFace-compatible version of Molmo, adapted from the official repository.
The main difference is that this version uses PyTorch instead of megablocks for MoE implementation,
making it easier to modify for dynamic MoE topK experiments.
"""

__version__ = "0.1.0"

# HF-style model exports
from molmo.models.modeling_molmoe import MolmoForCausalLM, MolmoModel
from molmo.models.config_molmoe import MolmoConfig
from molmo.preprocessors.preprocessing_molmo import MolmoProcessor
from molmo.preprocessors.image_preprocessing_molmo import MolmoImageProcessor

# Training and evaluation exports
from molmo.config import (
    ModelConfig,
    TrainConfig,
    EvalConfig,
    DataConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    VisionBackboneConfig,
    DatasetEvaluatorConfig,
    CheckpointType,
    BlockType,
    ActivationType,
    LayerNormType,
    # Configuration bridge functions
    model_config_to_molmo_config,
    molmo_config_to_model_config,
    load_model_config_from_hf_config,
)
from molmo.train import Trainer
from molmo.data import (
    build_train_dataloader,
    build_torch_mm_eval_dataloader,
    build_eval_dataloader,
)
from molmo.eval import build_loss_evaluators, build_inf_evaluators
from molmo.exceptions import OLMoCliError, OLMoConfigurationError

__all__ = [
    # HF-style model exports
    "MolmoForCausalLM",
    "MolmoModel",
    "MolmoConfig", 
    "MolmoProcessor",
    "MolmoImageProcessor",
    # Configuration classes
    "ModelConfig",
    "TrainConfig",
    "EvalConfig",
    "DataConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TokenizerConfig",
    "VisionBackboneConfig",
    "DatasetEvaluatorConfig",
    "CheckpointType",
    "BlockType",
    "ActivationType",
    "LayerNormType",
    # Configuration bridge functions
    "model_config_to_molmo_config",
    "molmo_config_to_model_config",
    "load_model_config_from_hf_config",
    # Training and evaluation
    "Trainer",
    "build_train_dataloader",
    "build_torch_mm_eval_dataloader",
    "build_eval_dataloader",
    "build_loss_evaluators",
    "build_inf_evaluators",
    # Exceptions
    "OLMoCliError",
    "OLMoConfigurationError",
]
