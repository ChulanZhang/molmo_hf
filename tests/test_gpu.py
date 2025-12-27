"""
GPU tests: validate model behavior on GPU.
Uses the 4th GPU (CUDA device 3).
"""
import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Force use of GPU index 3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


@pytest.fixture(scope="module")
def device():
    """Return test device (the 4th GPU)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device_id = 0  # with CUDA_VISIBLE_DEVICES=3, device 0 maps to physical GPU 4
    assert torch.cuda.device_count() > 0, "No CUDA devices available"
    
    device = torch.device(f"cuda:{device_id}")
    print(f"\nUsing GPU: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(device_id)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
    
    return device


class TestGPUModels:
    """Validate model behavior on GPU."""
    
    def test_gpu_available(self, device):
        """Check GPU availability."""
        assert torch.cuda.is_available()
        assert device.type == 'cuda'
        print(f"✓ GPU {device} is available")
    
    def test_model_creation_on_gpu(self, device):
        """Ensure model can be created on GPU."""
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        
        config = MolmoConfig(
            d_model=256,
            n_heads=4,
            n_kv_heads=4,  # n_kv_heads must be set
            n_layers=2,
            vocab_size=1000,
            max_sequence_length=512,
            init_device="cuda",
            vision_backbone=None,  # explicitly set to None
        )
        
        model = MolmoModel(config)
        model = model.to(device)
        
        assert next(model.parameters()).device.type == 'cuda'
        print(f"✓ Model created on {device}")
    
    def test_model_forward_pass(self, device):
        """Run a forward pass on GPU."""
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        
        config = MolmoConfig(
            d_model=128,
            n_heads=2,
            n_kv_heads=2,  # n_kv_heads must be set
            n_layers=1,
            vocab_size=1000,
            max_sequence_length=256,
            init_device="cuda",
            vision_backbone=None,  # explicitly set to None
        )
        
        model = MolmoModel(config)
        model = model.to(device)
        model.eval()
        
        # Build test input
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        assert outputs.last_hidden_states.shape == (batch_size, seq_len, config.d_model)
        assert outputs.last_hidden_states.device.type == 'cuda'
        print(f"✓ Forward pass successful on {device}")
        print(f"  Output shape: {outputs.last_hidden_states.shape}")
    
    def test_model_training_methods_on_gpu(self, device):
        """Verify training helpers on GPU."""
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        
        config = MolmoConfig(
            d_model=128,
            n_heads=2,
            n_kv_heads=2,  # add n_kv_heads
            n_layers=1,
            vocab_size=1000,
            max_sequence_length=256,
            init_device="cuda",
            vision_backbone=None,  # explicitly set to None
        )
        
        model = MolmoModel(config)
        model = model.to(device)
        
        # Test parameter accessors
        connector_params = MolmoModel.get_connector_parameters()
        vit_params = MolmoModel.get_vit_parameters()
        llm_params = MolmoModel.get_llm_parameters()
        
        assert len(connector_params) >= 0
        assert len(vit_params) >= 0
        assert len(llm_params) >= 0
        
        # Test parameter counting
        num_params = model.num_params()
        assert num_params > 0
        print(f"✓ Model has {num_params:,} parameters")
        print(f"✓ Training methods work on {device}")
    
    def test_model_memory_usage(self, device):
        """Measure GPU memory usage."""
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        config = MolmoConfig(
            d_model=256,
            n_heads=4,
            n_kv_heads=4,  # add n_kv_heads
            n_layers=2,
            vocab_size=1000,
            max_sequence_length=512,
            init_device="cuda",
            vision_backbone=None,  # explicitly set to None
        )
        
        model = MolmoModel(config)
        model = model.to(device)
        
        peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"✓ Peak memory usage: {peak_memory:.2f} GB")
        
        assert peak_memory > 0

