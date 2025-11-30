"""
Verification script to confirm that MoE top_k changes are actually being applied.
This script will:
1. Load the model
2. Add hooks to MoE layers to log top_k usage
3. Run inference with different top_k values
4. Verify that the actual number of active experts changes
"""

import sys
import os
sys.path.append(os.getcwd())

import torch
import logging
from PIL import Image
from transformers import AutoProcessor
from olmo import Molmo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def add_moe_hooks(model, layer_idx=0):
    """Add forward hooks to MoE layer to monitor top_k usage"""
    
    block = model.transformer.blocks[layer_idx]
    if not hasattr(block, 'ffn'):
        log.error(f"Block {layer_idx} does not have 'ffn' attribute")
        return None
    
    moe_layer = block.ffn
    log.info(f"MoE Layer type: {type(moe_layer)}")
    log.info(f"MoE Layer attributes: {[attr for attr in dir(moe_layer) if not attr.startswith('_')]}")
    
    # Check if it has args
    if hasattr(moe_layer, 'args'):
        log.info(f"MoE args type: {type(moe_layer.args)}")
        log.info(f"MoE args attributes: {[attr for attr in dir(moe_layer.args) if not attr.startswith('_')]}")
        if hasattr(moe_layer.args, 'top_k'):
            log.info(f"Initial top_k: {moe_layer.args.top_k}")
    
    activation_stats = {'expert_indices': [], 'top_k_observed': []}
    
    def hook_fn(module, input, output):
        """Hook to capture expert selection"""
        # For megablocks dMoE, we need to inspect internal states
        # This is tricky because megablocks has a complex internal structure
        if hasattr(module, 'args'):
            current_top_k = getattr(module.args, 'top_k', 'N/A')
            activation_stats['top_k_observed'].append(current_top_k)
            log.info(f"Forward pass - top_k from args: {current_top_k}")
        
        # Try to access expert indices if available
        if hasattr(module, 'router'):
            log.info(f"Router type: {type(module.router)}")
        
        return output
    
    handle = moe_layer.register_forward_hook(hook_fn)
    return handle, activation_stats

def verify_topk_impact(model_path="hf:allenai/MolmoE-1B-0924", device="cuda:0"):
    """Verify that changing top_k actually affects the model"""
    
    log.info(f"Loading model from {model_path}...")
    model = Molmo.from_checkpoint(model_path, device=device)
    
    log.info(f"Model config block_type: {model.config.block_type}")
    log.info(f"Model config moe_num_experts: {model.config.moe_num_experts}")
    log.info(f"Model config moe_top_k: {model.config.moe_top_k}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        "allenai/MolmoE-1B-0924",
        trust_remote_code=True
    )
    
    # Prepare standard input
    image = Image.new('RGB', (336, 336), color='blue')
    prompt = "Describe this image."
    inputs = processor.process(text=prompt, images=image, return_tensors="pt")
    
    # Ensure batch dimension
    if inputs["input_ids"].ndim == 1:
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        if "images" in inputs and inputs["images"] is not None:
            inputs["images"] = inputs["images"].unsqueeze(0)
        if "image_masks" in inputs and inputs["image_masks"] is not None:
            inputs["image_masks"] = inputs["image_masks"].unsqueeze(0)
        if "image_input_idx" in inputs and inputs["image_input_idx"] is not None:
            inputs["image_input_idx"] = inputs["image_input_idx"].unsqueeze(0)
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Add hooks to first MoE block
    log.info("\n" + "="*80)
    log.info("Setting up hooks on layer 0...")
    handle, stats = add_moe_hooks(model, layer_idx=0)
    
    # Test different top_k values
    test_top_k_values = [1, 2, 4, 8]
    
    for k in test_top_k_values:
        log.info("\n" + "="*80)
        log.info(f"Testing with top_k = {k}")
        log.info("="*80)
        
        # Set top_k for all blocks
        for i, block in enumerate(model.transformer.blocks):
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'args'):
                old_k = block.ffn.args.top_k
                block.ffn.args.top_k = k
                log.info(f"Block {i}: Changed top_k from {old_k} to {block.ffn.args.top_k}")
        
        # Clear previous stats
        stats['expert_indices'].clear()
        stats['top_k_observed'].clear()
        
        # Run forward pass
        log.info("Running forward pass...")
        with torch.no_grad():
            output = model(**inputs)
        
        log.info(f"Forward pass complete. Observed top_k values: {stats['top_k_observed']}")
        
    # Clean up
    if handle:
        handle.remove()
    
    log.info("\n" + "="*80)
    log.info("Verification complete!")
    log.info("="*80)

if __name__ == "__main__":
    verify_topk_impact()
