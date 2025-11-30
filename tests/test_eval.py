"""
测试评估模块
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEvalModules:
    """测试评估模块"""
    
    def test_evaluator_base_classes(self):
        """测试评估器基类"""
        from molmo.eval.evaluators import DatasetEvaluatorConfig
        assert DatasetEvaluatorConfig is not None
    
    def test_inf_evaluator_import(self):
        """测试推理评估器的导入"""
        from molmo.eval.inf_evaluator import InfDatasetEvaluator
        assert InfDatasetEvaluator is not None
    
    def test_loss_evaluator_import(self):
        """测试损失评估器的导入"""
        from molmo.eval.loss_evaluator import LossDatasetEvaluator
        assert LossDatasetEvaluator is not None
    
    def test_vqa_eval_import(self):
        """测试 VQA 评估的导入"""
        # VQA 评估可能通过其他方式导入
        from molmo.eval import vqa
        assert vqa is not None
    
    def test_evaluator_builders(self):
        """测试评估器构建函数"""
        from molmo.eval import build_loss_evaluators, build_inf_evaluators
        
        assert callable(build_loss_evaluators)
        assert callable(build_inf_evaluators)
    
    def test_math_vista_utils(self):
        """测试 MathVista 工具"""
        from molmo.eval import math_vista_utils
        assert math_vista_utils is not None
    
    def test_mmmu_eval_utils(self):
        """测试 MMMU 评估工具"""
        from molmo.eval import mmmu_eval_utils
        assert mmmu_eval_utils is not None

