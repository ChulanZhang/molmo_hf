from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="molmo",
    version="0.1.0",
    author="Allen Institute for AI",
    description="Molmo: Open Vision-Language Models with MoE architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://molmo.allenai.org/",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "transformers>=4.50.0",
        "einops>=0.8.0",
        "einops-exts>=0.0.4",
        # Configuration and utilities
        "omegaconf>=2.3.0",
        "packaging>=21.0",
        "cached-path>=1.6.4",
        # Data and datasets
        "datasets>=2.14.0",
        "huggingface-hub>=0.16.0",
        "Pillow>=11.0.0",
        "requests>=2.32.3",
        # Tokenization
        "tokenizers>=0.20.0",
        "sentencepiece>=0.1.99",
        # Cloud storage (optional but commonly used)
        "boto3>=1.26.0",
        "google-cloud-storage>=2.10.0",
        "gcsfs==2023.9.2",
        # Utilities
        "tqdm>=4.64.0",
        "rich>=12.0.0",
        "numpy>=1.22.0",
        "scipy>=1.16.3",
        "absl-py>=1.0.0",
        "cached-property>=1.5.2",
        # Accelerate for distributed training
        "accelerate>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "ruff>=0.0.200",
            "mypy>=1.0,<1.4",
            "isort>=5.12.0",
        ],
        "train": [
            "wandb>=0.15.0",
            "torchmetrics>=0.11.0",
            "safetensors>=0.3.0",
            "scikit-learn>=1.0.0",
            "msgspec>=0.14.0",
            "openai>=1.0.0",
            "python-Levenshtein>=0.20.0",
            "editdistance>=0.6.0",
        ],
        "experiments": [
            "ipdb>=0.13.0",
            "matplotlib>=3.5.0",
            "pandas>=1.4.0",
            "nvitop>=1.6.0",
        ],
        "all": [
            "molmo[dev,train,experiments]",
        ],
    },
)