"""
End-to-end tests: verify the full workflow.
"""
import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEndToEnd:
    """Validate end-to-end functionality."""
    
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
    
    def test_config_to_model_to_forward(self, sample_config):
        """Full flow: config -> model -> forward."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        # 1. Build model from config
        model = MolmoModel(sample_config)
        model.eval()
        
        # 2. Create inputs
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
        
        # 3. Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        # 4. Validate outputs
        assert outputs.last_hidden_states.shape == (batch_size, seq_len, sample_config.d_model)
        print("✓ End-to-end: Config -> Model -> Forward works")
    
    def test_training_workflow(self, sample_config):
        """Test full training workflow."""
        from molmo.models.modeling_molmoe import MolmoModel
        from molmo.optim import build_optimizer, build_scheduler
        
        # 1. Build model
        model = MolmoModel(sample_config)
        model.train()
        
        # 2. Create optimizer and scheduler
        optimizer = build_optimizer(
            model,
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        
        scheduler = build_scheduler(
            optimizer,
            scheduler_type="cosine_with_warmup",
            num_training_steps=100,
            num_warmup_steps=10,
        )
        
        # 3. Simulate several training steps
        for step in range(3):
            batch_size = 2
            seq_len = 10
            input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids)
            loss = outputs.last_hidden_states.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        print("✓ End-to-end training workflow works")
    
    def test_model_state_management(self, sample_config):
        """Test model state management."""
        from molmo.models.modeling_molmoe import MolmoModel
        
        # 1. Build model
        model1 = MolmoModel(sample_config)
        model1.train()
        
        # 2. Get state dict
        state_dict = model1.state_dict()
        
        # 3. Create new model and load state
        model2 = MolmoModel(sample_config)
        model2.load_state_dict(state_dict)
        
        # 4. Verify outputs match
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model1(input_ids=input_ids)
            out2 = model2(input_ids=input_ids)
        
        # Check outputs match (allow small numerical error)
        assert torch.allclose(out1.last_hidden_states, out2.last_hidden_states, atol=1e-6)
        print("✓ Model state management works")
    
    def test_config_conversion_workflow(self):
        """Test the full config conversion workflow."""
        from molmo.config import ModelConfig, model_config_to_molmo_config
        from molmo.models.modeling_molmoe import MolmoModel
        
        # 1. Build a training config
        train_cfg = ModelConfig(
            d_model=128,
            n_heads=2,
            n_layers=1,
            max_sequence_length=256,
            vocab_size=1000,
        )
        
        # 2. Convert to HF config
        hf_cfg = model_config_to_molmo_config(train_cfg)
        
        # 3. Create model from HF config
        model = MolmoModel(hf_cfg)
        
        # 4. Verify the model runs
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, hf_cfg.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        assert outputs.last_hidden_states.shape == (batch_size, seq_len, hf_cfg.d_model)
        print("✓ Config conversion workflow works")




