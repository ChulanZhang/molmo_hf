"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return project root path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root_path):
    """Return test data directory."""
    test_data = project_root_path / "tests" / "data"
    test_data.mkdir(exist_ok=True)
    return test_data


@pytest.fixture(scope="session")
def sample_config():
    """Return a sample ModelConfig."""
    from molmo.config import ModelConfig
    
    return ModelConfig(
        d_model=128,
        n_heads=2,
        n_layers=2,
        max_sequence_length=512,
        vocab_size=1000,
    )


@pytest.fixture(scope="session")
def sample_molmo_config():
    """Return a sample MolmoConfig."""
    from molmo.models.config_molmoe import MolmoConfig
    
    return MolmoConfig(
        d_model=128,
        n_heads=2,
        n_layers=2,
        vocab_size=1000,
        max_sequence_length=512,
    )




