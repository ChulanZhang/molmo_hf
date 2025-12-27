"""
Import tests for all core modules.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestImports:
    """Import all core modules."""
    
    def test_import_molmo_package(self):
        """Import molmo package."""
        import molmo
        assert hasattr(molmo, '__version__') or hasattr(molmo, '__name__')
    
    def test_import_models(self):
        """Import model classes."""
        from molmo.models.modeling_molmoe import MolmoModel, MolmoForCausalLM
        from molmo.models.config_molmoe import MolmoConfig
        assert MolmoModel is not None
        assert MolmoForCausalLM is not None
        assert MolmoConfig is not None
    
    def test_import_config(self):
        """Import config system."""
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
        """Import dataset modules."""
        from molmo.data.dataset import Dataset, DeterministicDataset, HfDataset
        from molmo.data.collator import MMCollator
        assert Dataset is not None
        assert MMCollator is not None
    
    def test_import_train_modules(self):
        """Import training modules."""
        from molmo.train import Trainer
        from molmo.optim import Optimizer, Scheduler
        from molmo.checkpoint import Checkpointer
        assert Trainer is not None
        assert Optimizer is not None
    
    def test_import_eval_modules(self):
        """Import evaluation modules."""
        from molmo.eval.evaluators import DatasetEvaluatorConfig
        from molmo.eval.inf_evaluator import InfDatasetEvaluator
        from molmo.eval.loss_evaluator import LossDatasetEvaluator
        assert DatasetEvaluatorConfig is not None
        assert InfDatasetEvaluator is not None
    
    def test_import_utils(self):
        """Import utilities."""
        from molmo.torch_util import get_default_device, seed_all
        from molmo.tokenizer import build_tokenizer
        from molmo.util import prepare_cli_environment
        from molmo.exceptions import OLMoCliError
        assert get_default_device is not None
        assert build_tokenizer is not None
    
    def test_import_hf_datasets(self):
        """Import hf_datasets modules."""
        from molmo.hf_datasets.a_okvqa import AOkVqaBuilder
        from molmo.hf_datasets.ai2d import Ai2dDatasetBuilder
        assert AOkVqaBuilder is not None
        assert Ai2dDatasetBuilder is not None
    
    def test_import_html_utils(self):
        """Import html_utils."""
        from molmo.html_utils import build_html_table, postprocess_prompt
        assert build_html_table is not None
        assert postprocess_prompt is not None

