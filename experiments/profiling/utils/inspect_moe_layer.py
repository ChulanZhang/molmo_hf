
import torch
from olmo import Molmo
from transformers import AutoProcessor
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def inspect_moe():
    model_path = "hf:allenai/MolmoE-1B-0924"
    log.info(f"Loading model from {model_path}...")
    model = Molmo.from_checkpoint(model_path, device="cpu") # CPU is enough for inspection
    
    log.info(f"Model Config Block Type: {model.config.block_type}")
    
    for i, block in enumerate(model.transformer.blocks):
        log.info(f"Block {i} Type: {type(block)}")
        if i >= 2: break # Just check first few
        
    # Try to find an MoE block
    found_moe = False
    for i, block in enumerate(model.transformer.blocks):
        if hasattr(block, 'ffn'):
            log.info(f"Found MoE block at index {i}")
            moe_layer = block.ffn
            log.info(f"MoE Layer Type: {type(moe_layer)}")
            if hasattr(moe_layer, 'args'):
                log.info(f"MoE Args: {moe_layer.args}")
                if hasattr(moe_layer.args, 'top_k'):
                    log.info(f"Current Top-K: {moe_layer.args.top_k}")
            found_moe = True
            break
    
    if not found_moe:
        log.warning("No MoE block found (no 'ffn' attribute).")

if __name__ == "__main__":
    inspect_moe()
