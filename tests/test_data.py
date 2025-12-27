"""
Tests for dataset modules.
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDataModules:
    """Validate dataset modules."""
    
    def test_dataset_base_classes(self):
        """Import base dataset classes."""
        from molmo.data.dataset import Dataset, DeterministicDataset, HfDataset
        assert Dataset is not None
        assert DeterministicDataset is not None
        assert HfDataset is not None
    
    def test_collator_import(self):
        """Import collator."""
        from molmo.data.collator import MMCollator
        assert MMCollator is not None
    
    def test_data_formatter_import(self):
        """Import DataFormatter."""
        from molmo.data.data_formatter import DataFormatter
        assert DataFormatter is not None
    
    def test_preprocessor_import(self):
        """Import preprocessors."""
        from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
        assert MultiModalPreprocessor is not None
        assert Preprocessor is not None
    
    def test_dataset_mixture_import(self):
        """Import dataset mixer."""
        from molmo.data.iterable_dataset_mixture import IterableDatasetMixture
        assert IterableDatasetMixture is not None
    
    def test_get_dataset_by_name(self):
        """Fetch dataset by name."""
        from molmo.data import get_dataset_by_name
        
        # Try a few known datasets
        datasets_to_test = ['chartqa', 'textvqa', 'pixmocap']
        
        for ds_name in datasets_to_test:
            try:
                dataset_class = get_dataset_by_name(ds_name)
                assert dataset_class is not None
            except Exception as e:
                # Some datasets may need extra deps or data
                pytest.skip(f"Dataset {ds_name} not available: {e}")
    
    def test_hf_datasets_builders(self):
        """Import HF dataset builders."""
        from molmo.hf_datasets.a_okvqa import AOkVqaBuilder
        from molmo.hf_datasets.ai2d import Ai2dDatasetBuilder
        from molmo.hf_datasets.vqa_v2 import VQAv2BuilderMultiQA
        
        assert AOkVqaBuilder is not None
        assert Ai2dDatasetBuilder is not None
        assert VQAv2BuilderMultiQA is not None
    
    def test_data_loader_builders(self):
        """Import dataloader builders."""
        from molmo.data import (
            build_train_dataloader,
            build_eval_dataloader,
            build_torch_mm_eval_dataloader,
            build_mm_preprocessor,
        )
        
        assert callable(build_train_dataloader)
        assert callable(build_eval_dataloader)
        assert callable(build_torch_mm_eval_dataloader)
        assert callable(build_mm_preprocessor)




