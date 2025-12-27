"""
Tests for training modules.
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTrainModules:
    """Validate training modules."""
    
    def test_trainer_import(self):
        """Import Trainer."""
        from molmo.train import Trainer
        assert Trainer is not None
    
    def test_optimizer_import(self):
        """Import optimizers."""
        from molmo.optim import Optimizer, Scheduler
        from molmo.optim import build_optimizer, build_scheduler
        assert Optimizer is not None
        assert Scheduler is not None
        assert callable(build_optimizer)
        assert callable(build_scheduler)
    
    def test_checkpoint_import(self):
        """Import checkpoint helpers."""
        from molmo.checkpoint import Checkpointer, load_model_state
        assert Checkpointer is not None
        assert callable(load_model_state)
    
    def test_trainer_class_structure(self):
        """Inspect Trainer structure."""
        from molmo.train import Trainer
        
        # Check key Trainer attributes
        assert hasattr(Trainer, '__init__')
        assert hasattr(Trainer, 'train')
        assert hasattr(Trainer, 'evaluate')
        assert hasattr(Trainer, 'save_checkpoint')
        assert hasattr(Trainer, 'load_checkpoint')
    
    def test_optimizer_types(self):
        """Import optimizer types."""
        from molmo.optim import OptimizerType, AdamW, LionW
        
        assert OptimizerType is not None
        assert AdamW is not None
        assert LionW is not None
    
    def test_scheduler_types(self):
        """Import scheduler types."""
        from molmo.optim import SchedulerType, CosWithWarmup, LinearWithWarmup
        
        assert SchedulerType is not None
        assert CosWithWarmup is not None
        assert LinearWithWarmup is not None




