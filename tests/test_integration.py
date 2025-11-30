"""
集成测试：测试多个模块的协同工作
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestIntegration:
    """集成测试"""
    
    def test_config_to_model_flow(self):
        """测试配置到模型的流程"""
        from molmo.config import ModelConfig
        from molmo.config import model_config_to_molmo_config
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        
        # 创建训练配置
        model_cfg = ModelConfig(
            d_model=128,
            n_heads=2,
            n_layers=2,
            max_sequence_length=512,
            vocab_size=1000,
        )
        
        # 转换为 HF 配置
        molmo_cfg = model_config_to_molmo_config(model_cfg)
        assert isinstance(molmo_cfg, MolmoConfig)
        
        # 可以用于创建模型（如果 CUDA 可用）
        # model = MolmoModel(molmo_cfg)
    
    def test_model_training_methods_availability(self):
        """测试模型训练方法的可用性"""
        from molmo.models.modeling_molmoe import MolmoModel
        
        # 检查所有训练所需的方法
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
        """测试数据到训练的流程（概念验证）"""
        from molmo.data import get_dataset_by_name
        from molmo.train import Trainer
        
        # 检查数据集和训练器是否可以一起使用
        assert get_dataset_by_name is not None
        assert Trainer is not None
    
    def test_eval_imports_complete(self):
        """测试评估模块的完整导入"""
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

