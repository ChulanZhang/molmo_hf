"""
Tests for model-related functionality.
"""
import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestMolmoModel:
    """Validate MolmoModel."""
    
    def test_model_import(self):
        """Import model classes."""
        from molmo.models.modeling_molmoe import MolmoModel, MolmoForCausalLM
        from molmo.models.config_molmoe import MolmoConfig
        
        assert MolmoModel is not None
        assert MolmoForCausalLM is not None
        assert MolmoConfig is not None
    
    def test_model_config_creation(self):
        """Create a model config."""
        from molmo.models.config_molmoe import MolmoConfig
        
        config = MolmoConfig(
            d_model=128,
            n_heads=2,
            n_layers=2,
            vocab_size=1000,
            max_sequence_length=512,
        )
        
        assert config.d_model == 128
        assert config.n_heads == 2
        assert config.n_layers == 2
    
    def test_model_training_methods(self):
        """Check training methods exist."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        # Static methods
        assert hasattr(MolmoModel, 'get_connector_parameters')
        assert hasattr(MolmoModel, 'get_vit_parameters')
        assert hasattr(MolmoModel, 'get_llm_parameters')
        
        # Instance methods
        assert hasattr(MolmoModel, 'set_activation_checkpointing')
        assert hasattr(MolmoModel, 'reset_with_pretrained_weights')
        assert hasattr(MolmoModel, 'get_fsdp_wrap_policy')
        assert hasattr(MolmoModel, 'num_params')
    
    def test_model_static_methods(self):
        """Call static helpers."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        # Call static methods without creating a model
        connector_params = MolmoModel.get_connector_parameters()
        assert isinstance(connector_params, (list, tuple, set))
        
        vit_params = MolmoModel.get_vit_parameters()
        assert isinstance(vit_params, (list, tuple, set))
        
        llm_params = MolmoModel.get_llm_parameters()
        assert isinstance(llm_params, (list, tuple, set))
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_initialization(self):
        """Initialize model on CUDA."""
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        
        config = MolmoConfig(
            d_model=128,
            n_heads=2,
            n_layers=2,
            vocab_size=1000,
            max_sequence_length=512,
            init_device="cuda",
        )
        
        model = MolmoModel(config)
        assert model is not None
        assert hasattr(model, 'transformer')
    
    def test_model_num_params(self):
        """Count model parameters."""
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        
        config = MolmoConfig(
            d_model=64,
            n_heads=2,
            n_layers=1,
            vocab_size=100,
            max_sequence_length=128,
            init_device="cpu",
        )
        
        model = MolmoModel(config)
        num_params = model.num_params()
        assert isinstance(num_params, int)
        assert num_params > 0




