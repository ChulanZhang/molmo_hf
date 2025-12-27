"""
Integration tests: verify modules work together.
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestIntegration:
    """Integration tests."""
    
    def test_config_to_model_flow(self):
        """Config → model flow."""
        from molmo.config import ModelConfig
        from molmo.config import model_config_to_molmo_config
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        
        # Build a training config
        model_cfg = ModelConfig(
            d_model=128,
            n_heads=2,
            n_layers=2,
            max_sequence_length=512,
            vocab_size=1000,
        )
        
        # Convert to HF config
        molmo_cfg = model_config_to_molmo_config(model_cfg)
        assert isinstance(molmo_cfg, MolmoConfig)
        
        # Could be used to create a model (when CUDA is available)
        # model = MolmoModel(molmo_cfg)
    
    def test_model_training_methods_availability(self):
        """Check training helpers exist."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        # Required training methods
        required_methods = [
            'get_connector_parameters',
            'get_vit_parameters',
            'get_llm_parameters',
            'set_activation_checkpointing',
            'reset_with_pretrained_weights',
            'get_fsdp_wrap_policy',
            'num_params',
        ]
        
        for method_name in required_methods:
            assert hasattr(MolmoModel, method_name), f"Missing method: {method_name}"
    
    def test_data_to_train_flow(self):
        """Data → training flow (smoke)."""
        from molmo.data import get_dataset_by_name
        from molmo.train import Trainer
        
        # Ensure dataset getter and trainer exist
        assert get_dataset_by_name is not None
        assert Trainer is not None
    
    def test_eval_imports_complete(self):
        """Eval modules import cleanly."""
        from molmo.eval import (
            build_loss_evaluators,
            build_inf_evaluators,
        )
        from molmo.eval.evaluators import DatasetEvaluatorConfig
        from molmo.eval.inf_evaluator import InfDatasetEvaluator
        from molmo.eval.loss_evaluator import LossDatasetEvaluator
        
        assert callable(build_loss_evaluators)
        assert callable(build_inf_evaluators)
        assert DatasetEvaluatorConfig is not None
        assert InfDatasetEvaluator is not None
        assert LossDatasetEvaluator is not None

