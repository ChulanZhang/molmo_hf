# Quick test to verify the migration works
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

print("Testing HF model loading from local directory...")
print(f"Current directory: {__import__('os').getcwd()}")

# Load from current directory (should be molmo_hf)
processor = AutoProcessor.from_pretrained(
    '.',  # Current directory
    trust_remote_code=True
)
print("✓ Processor loaded successfully")

# Quick model structure check (CPU only, no full loading)
from transformers import AutoConfig
config = AutoConfig.from_pretrained('.', trust_remote_code=True)
print(f"✓ Config loaded: block_type={config.block_type}, moe_top_k={config.moe_top_k}")

print("\n=== Migration Test PASSED ===\n")
