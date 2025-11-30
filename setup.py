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
        "torch==2.5.1",
        "transformers==4.50.0",
        "einops==0.8.0",
        "torchvision==0.20.1",
        "Pillow==11.0.0",
        "requests==2.32.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "experiments": [
            "ipdb", 
            "matplotlib>=3.5.0",
            "numpy>=1.22.0",
            "pandas>=1.4.0",
        ],
    },
)

# pip install -e ".[experiments]"