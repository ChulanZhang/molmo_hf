"""
Tests for the config system.
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestConfigSystem:
    """Validate the config system."""
    
    def test_model_config_import(self):
        """Import ModelConfig."""
        from molmo.config import ModelConfig
        assert ModelConfig is not None
    
    def test_train_config_import(self):
        """Import TrainConfig."""
        from molmo.config import TrainConfig
        assert TrainConfig is not None
    
    def test_config_bridge_functions(self):
        """Import config bridge helpers."""
        from molmo.config import (
            model_config_to_molmo_config,
            molmo_config_to_model_config,
            load_model_config_from_hf_config,
        )
        
        assert callable(model_config_to_molmo_config)
        assert callable(molmo_config_to_model_config)
        assert callable(load_model_config_from_hf_config)
    
    def test_model_config_creation(self):
        """Create a ModelConfig."""
        from molmo.config import ModelConfig
        
        config = ModelConfig(
            d_model=128,
            n_heads=2,
            n_layers=2,
            max_sequence_length=1024,
            vocab_size=1000,
        )
        
        assert config.d_model == 128
        assert config.n_heads == 2
        assert config.n_layers == 2
    
    def test_config_conversion(self):
        """Convert configs."""
        from molmo.config import ModelConfig, model_config_to_molmo_config
        from molmo.models.config_molmoe import MolmoConfig
        
        # Create ModelConfig
        model_cfg = ModelConfig(
            d_model=128,
            n_heads=2,
            n_layers=2,
            max_sequence_length=1024,
            vocab_size=1000,
        )
        
        # Convert to MolmoConfig
        molmo_cfg = model_config_to_molmo_config(model_cfg)
        assert isinstance(molmo_cfg, MolmoConfig)
        assert molmo_cfg.d_model == 128
    
    def test_molmo_config_import(self):
        """Import MolmoConfig."""
        from molmo.models.config_molmoe import MolmoConfig
        assert MolmoConfig is not None
        
        config = MolmoConfig(
            d_model=128,
            n_heads=2,
            n_layers=2,
            vocab_size=1000,
        )
        assert config.d_model == 128




