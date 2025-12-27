"""
Model functional tests: validate real model behaviors beyond imports.
"""
import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestModelFunctional:
    """Validate model functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample config."""
        from molmo.models.config_molmoe import MolmoConfig
        
        return MolmoConfig(
            d_model=128,
            n_heads=2,
            n_kv_heads=2,
            n_layers=1,
            vocab_size=1000,
            max_sequence_length=256,
            init_device="cpu",
            vision_backbone=None,
            layer_norm_eps=1e-5,
        )
    
    def test_model_forward_pass_cpu(self, sample_config):
        """Run a forward pass on CPU."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        model = MolmoModel(sample_config)
        model.eval()
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        assert outputs.last_hidden_states.shape == (batch_size, seq_len, sample_config.d_model)
        assert outputs.last_hidden_states.dtype == torch.float32
        print(f"✓ Forward pass successful: {outputs.last_hidden_states.shape}")
    
    def test_model_parameter_count(self, sample_config):
        """Count model parameters."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        model = MolmoModel(sample_config)
        
        # Validate parameter count
        num_params = model.num_params()
        assert num_params > 0
        
        # Manually compute parameter count for verification
        manual_count = sum(p.numel() for p in model.parameters())
        assert num_params == manual_count
        
        print(f"✓ Model has {num_params:,} parameters")
    
    def test_model_training_mode(self, sample_config):
        """Check training mode."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        model = MolmoModel(sample_config)
        model.train()
        
        assert model.training == True
        
        # Forward pass in training mode
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids=input_ids)
        assert outputs.last_hidden_states.shape == (batch_size, seq_len, sample_config.d_model)
        
        print("✓ Model training mode works")
    
    def test_model_eval_mode(self, sample_config):
        """Check eval mode."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        model = MolmoModel(sample_config)
        model.eval()
        
        assert model.training == False
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        assert outputs.last_hidden_states.shape == (batch_size, seq_len, sample_config.d_model)
        print("✓ Model eval mode works")
    
    def test_model_gradient_flow(self, sample_config):
        """Verify gradients flow."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        model = MolmoModel(sample_config)
        model.train()
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids=input_ids)
        
        # Create dummy loss
        loss = outputs.last_hidden_states.mean()
        loss.backward()
        
        # Ensure gradients exist
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad
        
        print("✓ Gradient flow works")
    
    def test_model_parameter_groups(self, sample_config):
        """Inspect parameter groups."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        model = MolmoModel(sample_config)
        
        # Static accessors
        connector_params = MolmoModel.get_connector_parameters()
        vit_params = MolmoModel.get_vit_parameters()
        llm_params = MolmoModel.get_llm_parameters()
        
        assert isinstance(connector_params, (list, tuple))
        assert isinstance(vit_params, (list, tuple))
        assert isinstance(llm_params, (list, tuple))
        
        print(f"✓ Parameter groups: connector={len(connector_params)}, vit={len(vit_params)}, llm={len(llm_params)}")
    
    def test_model_state_dict(self, sample_config):
        """Check state dict round-trip."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        model = MolmoModel(sample_config)
        
        state_dict = model.state_dict()
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        
        # Load state dict into a new model
        model2 = MolmoModel(sample_config)
        model2.load_state_dict(state_dict)
        
        print(f"✓ State dict has {len(state_dict)} keys")




