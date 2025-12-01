"""
配置系统功能测试：测试配置的实际使用
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestConfigFunctional:
    """测试配置系统的实际功能"""
    
    def test_model_config_to_molmo_config(self):
        """测试 ModelConfig 到 MolmoConfig 的转换"""
        from molmo.config import ModelConfig, model_config_to_molmo_config
        
        model_cfg = ModelConfig(
            d_model=256,
            n_heads=4,
            n_layers=2,
            max_sequence_length=512,
            vocab_size=1000,
        )
        
        molmo_cfg = model_config_to_molmo_config(model_cfg)
        
        assert molmo_cfg.d_model == 256
        assert molmo_cfg.n_heads == 4
        assert molmo_cfg.n_layers == 2
        print("✓ ModelConfig to MolmoConfig conversion works")
    
    def test_molmo_config_to_model_config(self):
        """测试 MolmoConfig 到 ModelConfig 的转换"""
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.config import molmo_config_to_model_config
        
        molmo_cfg = MolmoConfig(
            d_model=256,
            n_heads=4,
            n_layers=2,
            vocab_size=1000,
            max_sequence_length=512,
        )
        
        model_cfg = molmo_config_to_model_config(molmo_cfg)
        
        assert model_cfg.d_model == 256
        assert model_cfg.n_heads == 4
        assert model_cfg.n_layers == 2
        print("✓ MolmoConfig to ModelConfig conversion works")
    
    def test_config_round_trip(self):
        """测试配置的双向转换"""
        from molmo.config import ModelConfig, model_config_to_molmo_config, molmo_config_to_model_config
        
        original = ModelConfig(
            d_model=128,
            n_heads=2,
            n_layers=1,
            max_sequence_length=256,
            vocab_size=500,
        )
        
        # ModelConfig -> MolmoConfig -> ModelConfig
        molmo_cfg = model_config_to_molmo_config(original)
        back_to_model = molmo_config_to_model_config(molmo_cfg)
        
        assert back_to_model.d_model == original.d_model
        assert back_to_model.n_heads == original.n_heads
        assert back_to_model.n_layers == original.n_layers
        print("✓ Config round-trip conversion works")
    
    def test_config_serialization(self):
        """测试配置的序列化"""
        from molmo.models.config_molmoe import MolmoConfig
        
        config = MolmoConfig(
            d_model=128,
            n_heads=2,
            n_layers=1,
            vocab_size=1000,
        )
        
        # 测试配置可以转换为字典
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "d_model" in config_dict
        
        # 测试从字典创建配置
        new_config = MolmoConfig.from_dict(config_dict)
        assert new_config.d_model == config.d_model
        
        print("✓ Config serialization works")



