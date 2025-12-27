# Test Suite Guide

This directory contains the full test suite for `molmo_hf`.

## Structure

### 1. `test_imports.py` – import tests
Ensures core modules import cleanly, dependencies are present, and no cycles exist.

### 2. `test_models.py` – model tests
- Import/init model classes
- Create model configs
- Training method availability
- Parameter counting

### 3. `test_config.py` – config system tests
- Import/create configs
- Conversions (ModelConfig <-> MolmoConfig)
- Bridge helpers

### 4. `test_data.py` – dataset module tests
- Dataset class imports
- Dataloader builders
- Dataset fetching
- HF dataset builders

### 5. `test_train.py` – training module tests
- Trainer imports/structure
- Optimizers and schedulers
- Checkpoint management

### 6. `test_eval.py` – evaluation module tests
- Evaluator base
- Inference evaluators
- Loss evaluators
- Eval utilities

### 7. `test_utils.py` – utility tests
- PyTorch utilities
- Tokenizer utilities
- General utilities
- Exception classes
- Safetensors utilities

### 8. `test_integration.py` – integration tests
- Config → model flow
- Data → training flow
- Training method completeness

### 9. `conftest.py` – pytest config
Shared fixtures and configuration.

## Running tests

### All tests
```bash
pytest tests/
```

### Specific files
```bash
pytest tests/test_imports.py
pytest tests/test_models.py
```

### Specific class
```bash
pytest tests/test_models.py::TestMolmoModel
```

### Specific method
```bash
pytest tests/test_models.py::TestMolmoModel::test_model_import
```

### Verbose output
```bash
pytest tests/ -v
```

### Show print statements
```bash
pytest tests/ -s
```

### Failed tests only
```bash
pytest tests/ --lf
```

### With coverage
```bash
pytest tests/ --cov=molmo --cov-report=html
```

## Requirements
- Python 3.10+
- pytest >= 7.0.0
- All project deps (see `setup.py`)

## Install test deps
```bash
pip install -e ".[dev]"
```

## Notes
1. **CUDA tests**: skipped if CUDA is unavailable.
2. **Data dependencies**: some dataset tests may skip if data is missing.
3. **Optional deps**: some features require optional packages (e.g., boto3); related tests skip if absent.

## CI
The suite is CI-friendly to prevent regressions.




