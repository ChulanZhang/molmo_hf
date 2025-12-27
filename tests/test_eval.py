"""
Tests for evaluation modules.
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEvalModules:
    """Validate evaluation modules."""
    
    def test_evaluator_base_classes(self):
        """Import evaluator base config."""
        from molmo.eval.evaluators import DatasetEvaluatorConfig
        assert DatasetEvaluatorConfig is not None
    
    def test_inf_evaluator_import(self):
        """Import inference evaluator."""
        from molmo.eval.inf_evaluator import InfDatasetEvaluator
        assert InfDatasetEvaluator is not None
    
    def test_loss_evaluator_import(self):
        """Import loss evaluator."""
        from molmo.eval.loss_evaluator import LossDatasetEvaluator
        assert LossDatasetEvaluator is not None
    
    def test_vqa_eval_import(self):
        """Import VQA evaluation helpers."""
        # VQA eval may be imported indirectly
        from molmo.eval import vqa
        assert vqa is not None
    
    def test_evaluator_builders(self):
        """Check evaluator builders."""
        from molmo.eval import build_loss_evaluators, build_inf_evaluators
        
        assert callable(build_loss_evaluators)
        assert callable(build_inf_evaluators)
    
    def test_math_vista_utils(self):
        """Import MathVista utils."""
        from molmo.eval import math_vista_utils
        assert math_vista_utils is not None
    
    def test_mmmu_eval_utils(self):
        """Import MMMU eval utils."""
        from molmo.eval import mmmu_eval_utils
        assert mmmu_eval_utils is not None

