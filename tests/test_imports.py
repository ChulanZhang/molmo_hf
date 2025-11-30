"""
测试所有模块的导入功能
"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestImports:
    """测试所有核心模块的导入"""
    
    def test_import_molmo_package(self):
        """测试 molmo 包的导入"""
        import molmo
        assert hasattr(molmo, '__version__') or hasattr(molmo, '__name__')
    
    def test_import_models(self):
        """测试模型类的导入"""
        from molmo.models.modeling_molmoe import MolmoModel, MolmoForCausalLM
        from molmo.models.config_molmoe import MolmoConfig
        assert MolmoModel is not None
        assert MolmoForCausalLM is not None
        assert MolmoConfig is not None
    
    def test_import_config(self):
        """测试配置系统的导入"""
        from molmo.config import (
            ModelConfig,
            TrainConfig,
            EvalConfig,
            DataConfig,
            OptimizerConfig,
        )
        assert ModelConfig is not None
        assert TrainConfig is not None
    
    def test_import_data_modules(self):
        """测试数据集模块的导入"""
        from molmo.data.dataset import Dataset, DeterministicDataset, HfDataset
        from molmo.data.collator import MMCollator
        assert Dataset is not None
        assert MMCollator is not None
    
    def test_import_train_modules(self):
        """测试训练模块的导入"""
        from molmo.train import Trainer
        from molmo.optim import Optimizer, Scheduler
        from molmo.checkpoint import Checkpointer
        assert Trainer is not None
        assert Optimizer is not None
    
    def test_import_eval_modules(self):
        """测试评估模块的导入"""
        from molmo.eval.evaluators import DatasetEvaluatorConfig
        from molmo.eval.inf_evaluator import InfDatasetEvaluator
        from molmo.eval.loss_evaluator import LossDatasetEvaluator
        assert DatasetEvaluatorConfig is not None
        assert InfDatasetEvaluator is not None
    
    def test_import_utils(self):
        """测试工具模块的导入"""
        from molmo.torch_util import get_default_device, seed_all
        from molmo.tokenizer import build_tokenizer
        from molmo.util import prepare_cli_environment
        from molmo.exceptions import OLMoCliError
        assert get_default_device is not None
        assert build_tokenizer is not None
    
    def test_import_hf_datasets(self):
        """测试 hf_datasets 模块的导入"""
        from molmo.hf_datasets.a_okvqa import AOkVqaBuilder
        from molmo.hf_datasets.ai2d import Ai2dDatasetBuilder
        assert AOkVqaBuilder is not None
        assert Ai2dDatasetBuilder is not None
    
    def test_import_html_utils(self):
        """测试 html_utils 模块的导入"""
        from molmo.html_utils import build_html_table, postprocess_prompt
        assert build_html_table is not None
        assert postprocess_prompt is not None

