"""
测试工具模块
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestUtils:
    """测试工具模块"""
    
    def test_torch_util_import(self):
        """测试 PyTorch 工具函数的导入"""
        from molmo.torch_util import (
            get_default_device,
            seed_all,
            get_global_rank,
            get_world_size,
        )
        assert callable(get_default_device)
        assert callable(seed_all)
        assert callable(get_global_rank)
        assert callable(get_world_size)
    
    def test_tokenizer_import(self):
        """测试分词器的导入"""
        from molmo.tokenizer import build_tokenizer, HfTokenizerWrapper
        assert callable(build_tokenizer)
        assert HfTokenizerWrapper is not None
    
    def test_util_functions(self):
        """测试通用工具函数"""
        from molmo.util import prepare_cli_environment, resource_path
        assert callable(prepare_cli_environment)
        assert callable(resource_path)
    
    def test_exceptions(self):
        """测试异常类"""
        from molmo.exceptions import OLMoCliError, OLMoConfigurationError
        assert OLMoCliError is not None
        assert OLMoConfigurationError is not None
    
    def test_aliases(self):
        """测试类型别名"""
        from molmo.aliases import PathOrStr
        assert PathOrStr is not None
    
    def test_safetensors_util(self):
        """测试 Safetensors 工具"""
        from molmo.safetensors_util import (
            state_dict_to_safetensors_file,
            safetensors_file_to_state_dict,
        )
        assert callable(state_dict_to_safetensors_file)
        assert callable(safetensors_file_to_state_dict)
    
    def test_html_utils(self):
        """测试 HTML 工具"""
        from molmo.html_utils import (
            build_html_table,
            postprocess_prompt,
            BoxesToVisualize,
        )
        assert callable(build_html_table)
        assert callable(postprocess_prompt)
        assert BoxesToVisualize is not None




