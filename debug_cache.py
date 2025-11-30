import datasets
from molmo.hf_datasets.vqa_v2 import VQAv2BuilderMultiQA
import os

print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print(f"Default cache: {datasets.config.HF_DATASETS_CACHE}")

builder = VQAv2BuilderMultiQA()
print(f"Builder cache dir: {builder.cache_dir}")
print(f"Is cached? {builder.is_cached()}")
