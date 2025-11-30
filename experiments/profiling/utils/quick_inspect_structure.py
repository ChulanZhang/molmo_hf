"""
Quick script to inspect the actual model structure from Molmo.from_checkpoint()
"""
import sys
import os
sys.path.append(os.getcwd())

from olmo import Molmo

print("Loading model...")
model = Molmo.from_checkpoint("hf:allenai/MolmoE-1B-0924", device="cpu")

print("\n=== Model Top Level ===")
print(f"Type: {type(model)}")
print(f"Attributes: {[a for a in dir(model) if not a.startswith('_')][:20]}")

print("\n=== model.transformer ===")
print(f"Type: {type(model.transformer)}")
if hasattr(model.transformer, 'blocks'):
    print(f"Has blocks: True")
    print(f"Number of blocks: {len(model.transformer.blocks)}")
    
    print("\n=== First Block ===")
    block = model.transformer.blocks[0]
    print(f"Type: {type(block)}")
    print(f"Attributes: {[a for a in dir(block) if not a.startswith('_')]}")
    
    # Check for different attribute names
    print("\n=== Checking for MoE attributes ===")
    if hasattr(block, 'mlp'):
        print(f"block.mlp exists: {type(block.mlp)}")
        print(f"block.mlp attrs: {[a for a in dir(block.mlp) if not a.startswith('_')][:20]}")
        if hasattr(block.mlp, 'top_k'):
            print(f"block.mlp.top_k = {block.mlp.top_k}")
    else:
        print("block.mlp does NOT exist")
    
    if hasattr(block, 'ffn'):
        print(f"block.ffn exists: {type(block.ffn)}")
        print(f"block.ffn attrs: {[a for a in dir(block.ffn) if not a.startswith('_')][:20]}")
        if hasattr(block.ffn, 'top_k'):
            print(f"block.ffn.top_k = {block.ffn.top_k}")
        if hasattr(block.ffn, 'args'):
            print(f"block.ffn.args exists: {type(block.ffn.args)}")
            if hasattr(block.ffn.args, 'top_k'):
                print(f"block.ffn.args.top_k = {block.ffn.args.top_k}")
    else:
        print("block.ffn does NOT exist")
else:
    print("model.transformer does NOT have blocks")

print("\n=== Config ===")
print(f"Config type: {type(model.config)}")
print(f"block_type: {getattr(model.config, 'block_type', 'N/A')}")
print(f"moe_num_experts: {getattr(model.config, 'moe_num_experts', 'N/A')}")
print(f"moe_top_k: {getattr(model.config, 'moe_top_k', 'N/A')}")
