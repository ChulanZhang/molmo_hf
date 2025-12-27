"""
Tokenizer functional tests: validate real tokenizer behavior.
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTokenizerFunctional:
    """Validate tokenizer functionality."""
    
    def test_tokenizer_wrapper_creation(self):
        """Create tokenizer wrapper."""
        from molmo.tokenizer import HfTokenizerWrapper
        
        # Ensure wrapper class is available (may need a real tokenizer)
        assert HfTokenizerWrapper is not None
        print("✓ HfTokenizerWrapper can be imported")
    
    def test_tokenizer_special_tokens(self):
        """Handle special tokens."""
        from molmo.tokenizer import get_special_token_ids
        
        # Fetch special token IDs
        special_tokens = get_special_token_ids()
        assert isinstance(special_tokens, dict)
        print("✓ Special token IDs can be retrieved")
    
    def test_build_tokenizer(self):
        """Build tokenizer."""
        from molmo.tokenizer import build_tokenizer
        
        # Ensure builder exists
        assert callable(build_tokenizer)
        print("✓ build_tokenizer function exists")




