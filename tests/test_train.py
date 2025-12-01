"""
测试训练模块
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTrainModules:
    """测试训练模块"""
    
    def test_trainer_import(self):
        """测试 Trainer 的导入"""
        from molmo.train import Trainer
        assert Trainer is not None
    
    def test_optimizer_import(self):
        """测试优化器的导入"""
        from molmo.optim import Optimizer, Scheduler
        from molmo.optim import build_optimizer, build_scheduler
        assert Optimizer is not None
        assert Scheduler is not None
        assert callable(build_optimizer)
        assert callable(build_scheduler)
    
    def test_checkpoint_import(self):
        """测试检查点模块的导入"""
        from molmo.checkpoint import Checkpointer, load_model_state
        assert Checkpointer is not None
        assert callable(load_model_state)
    
    def test_trainer_class_structure(self):
        """测试 Trainer 类的结构"""
        from molmo.train import Trainer
        
        # 检查 Trainer 类的主要属性
        assert hasattr(Trainer, '__init__')
        assert hasattr(Trainer, 'train')
        assert hasattr(Trainer, 'evaluate')
        assert hasattr(Trainer, 'save_checkpoint')
        assert hasattr(Trainer, 'load_checkpoint')
    
    def test_optimizer_types(self):
        """测试优化器类型"""
        from molmo.optim import OptimizerType, AdamW, LionW
        
        assert OptimizerType is not None
        assert AdamW is not None
        assert LionW is not None
    
    def test_scheduler_types(self):
        """测试调度器类型"""
        from molmo.optim import SchedulerType, CosWithWarmup, LinearWithWarmup
        
        assert SchedulerType is not None
        assert CosWithWarmup is not None
        assert LinearWithWarmup is not None



