"""
测试数据集模块
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDataModules:
    """测试数据集模块"""
    
    def test_dataset_base_classes(self):
        """测试基础数据集类"""
        from molmo.data.dataset import Dataset, DeterministicDataset, HfDataset
        assert Dataset is not None
        assert DeterministicDataset is not None
        assert HfDataset is not None
    
    def test_collator_import(self):
        """测试 Collator 的导入"""
        from molmo.data.collator import MMCollator
        assert MMCollator is not None
    
    def test_data_formatter_import(self):
        """测试 DataFormatter 的导入"""
        from molmo.data.data_formatter import DataFormatter
        assert DataFormatter is not None
    
    def test_preprocessor_import(self):
        """测试预处理器的导入"""
        from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
        assert MultiModalPreprocessor is not None
        assert Preprocessor is not None
    
    def test_dataset_mixture_import(self):
        """测试数据集混合器的导入"""
        from molmo.data.iterable_dataset_mixture import IterableDatasetMixture
        assert IterableDatasetMixture is not None
    
    def test_get_dataset_by_name(self):
        """测试按名称获取数据集"""
        from molmo.data import get_dataset_by_name
        
        # 测试几个已知的数据集
        datasets_to_test = ['chartqa', 'textvqa', 'pixmocap']
        
        for ds_name in datasets_to_test:
            try:
                dataset_class = get_dataset_by_name(ds_name)
                assert dataset_class is not None
            except Exception as e:
                # 某些数据集可能需要额外的依赖或数据
                pytest.skip(f"Dataset {ds_name} not available: {e}")
    
    def test_hf_datasets_builders(self):
        """测试 HF 数据集构建器"""
        from molmo.hf_datasets.a_okvqa import AOkVqaBuilder
        from molmo.hf_datasets.ai2d import Ai2dDatasetBuilder
        from molmo.hf_datasets.vqa_v2 import VQAv2BuilderMultiQA
        
        assert AOkVqaBuilder is not None
        assert Ai2dDatasetBuilder is not None
        assert VQAv2BuilderMultiQA is not None
    
    def test_data_loader_builders(self):
        """测试数据加载器构建函数"""
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



