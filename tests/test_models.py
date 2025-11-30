"""
测试模型相关的功能
"""
import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestMolmoModel:
    """测试 MolmoModel 类"""
    
    def test_model_import(self):
        """测试模型类的导入"""
        from molmo.models.modeling_molmoe import MolmoModel, MolmoForCausalLM
        from molmo.models.config_molmoe import MolmoConfig
        
        assert MolmoModel is not None
        assert MolmoForCausalLM is not None
        assert MolmoConfig is not None
    
    def test_model_config_creation(self):
        """测试模型配置的创建"""
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
        """测试模型训练方法的可用性"""
        from molmo.models.modeling_molmoe import MolmoModel
        
        # 检查静态方法
        assert hasattr(MolmoModel, 'get_connector_parameters')
        assert hasattr(MolmoModel, 'get_vit_parameters')
        assert hasattr(MolmoModel, 'get_llm_parameters')
        
        # 检查实例方法
        assert hasattr(MolmoModel, 'set_activation_checkpointing')
        assert hasattr(MolmoModel, 'reset_with_pretrained_weights')
        assert hasattr(MolmoModel, 'get_fsdp_wrap_policy')
        assert hasattr(MolmoModel, 'num_params')
    
    def test_model_static_methods(self):
        """测试模型的静态方法"""
        from molmo.models.modeling_molmoe import MolmoModel
        
        # 测试静态方法调用（不创建模型实例）
        connector_params = MolmoModel.get_connector_parameters()
        assert isinstance(connector_params, (list, tuple, set))
        
        vit_params = MolmoModel.get_vit_parameters()
        assert isinstance(vit_params, (list, tuple, set))
        
        llm_params = MolmoModel.get_llm_parameters()
        assert isinstance(llm_params, (list, tuple, set))
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_initialization(self):
        """测试模型初始化（需要 CUDA）"""
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
        """测试模型参数计数"""
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

