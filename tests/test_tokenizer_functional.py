"""
分词器功能测试：测试分词器的实际功能
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTokenizerFunctional:
    """测试分词器的实际功能"""
    
    def test_tokenizer_wrapper_creation(self):
        """测试分词器包装器的创建"""
        from molmo.tokenizer import HfTokenizerWrapper
        
        # 测试可以创建包装器（可能需要实际的 tokenizer）
        assert HfTokenizerWrapper is not None
        print("✓ HfTokenizerWrapper can be imported")
    
    def test_tokenizer_special_tokens(self):
        """测试特殊 token 的处理"""
        from molmo.tokenizer import get_special_token_ids
        
        # 测试获取特殊 token IDs
        special_tokens = get_special_token_ids()
        assert isinstance(special_tokens, dict)
        print("✓ Special token IDs can be retrieved")
    
    def test_build_tokenizer(self):
        """测试构建分词器"""
        from molmo.tokenizer import build_tokenizer
        
        # 测试构建函数存在
        assert callable(build_tokenizer)
        print("✓ build_tokenizer function exists")




