
import json
import os
from transformers import AutoConfig

def check_config():
    model_path = "allenai/MolmoE-1B-0924"
    print(f"Loading config for {model_path}...")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print("Config loaded successfully.")
        print(f"Model Type: {config.model_type}")
        if hasattr(config, 'block_type'):
            print(f"Block Type: {config.block_type}")
        else:
            print("No 'block_type' in config.")
            
        if hasattr(config, 'moe_num_experts'):
             print(f"Num Experts: {config.moe_num_experts}")
        if hasattr(config, 'moe_top_k'):
             print(f"Top K: {config.moe_top_k}")
             
    except Exception as e:
        print(f"Error loading config: {e}")

if __name__ == "__main__":
    check_config()
