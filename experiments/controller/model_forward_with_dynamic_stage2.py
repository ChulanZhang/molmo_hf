"""
Model forward pass with dynamic Stage2 controller insertion.

This module implements a custom forward pass that:
1. Runs vision encoder + projector
2. Runs transformer blocks up to insertion position
3. Extracts latency token from insertion position block output
4. Stage2 controller predicts knob2 & knob3 based on insertion position
5. Applies knob2 & knob3 to subsequent blocks
6. Continues forward pass
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import logging

log = logging.getLogger(__name__)


def forward_with_dynamic_stage2_controller(
    model,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    image_masks: Optional[torch.Tensor] = None,
    image_input_idx: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    knob1_tier: str = "medium",
    insertion_position: int = 1,  # Insert after this block (1-5)
    knob2_top_k: int = 8,
    knob3_num_blocks: int = 16,
    stage2_controller: Optional[nn.Module] = None,
    budget_feat: Optional[torch.Tensor] = None,
    return_latency_token: bool = False,
) -> Dict[str, Any]:
    """
    Forward pass with Stage2 controller inserted at dynamic position.
    
    Args:
        model: MolmoModel instance
        input_ids: Input token IDs
        images: Optional images
        image_masks: Optional image masks
        image_input_idx: Optional image input indices
        attention_mask: Optional attention mask
        attention_bias: Optional attention bias
        position_ids: Optional position IDs
        knob1_tier: Tier from Stage1 (for image processing, already applied)
        insertion_position: Position after which to insert Stage2 (1-5, meaning after block 1-5)
        knob2_top_k: Top-K value (if not using controller, or as fallback)
        knob3_num_blocks: Number of active blocks (if not using controller, or as fallback)
        stage2_controller: Optional Stage2 controller (if None, uses knob2_top_k and knob3_num_blocks)
        budget_feat: Optional pre-extracted budget features (for Stage2)
        return_latency_token: If True, return latency token extracted for Stage2
    
    Returns:
        Dict with 'logits', 'hidden_states', 'latency_token' (if return_latency_token)
    """
    batch_size, seq_len = input_ids.shape
    
    # Step 1: Get embeddings
    x = model.model.transformer.wte(input_ids)  # (B, seq_len, d_model)
    
    # Step 2: Process images (if provided)
    num_image: Optional[int] = None
    if images is not None:
        # Vision encoder + projector
        image_features, cls_embed = model.model.vision_backbone(images, image_masks)
        num_image, num_patch = image_features.shape[1:3]
        
        # Insert image features into embeddings
        image_features = image_features.view(batch_size, num_image * num_patch, -1)
        image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)
        
        x_flat = x.view(batch_size * seq_len, -1)
        # Ensure image_flat has the same dtype and device as x_flat for index_add_
        image_flat = image_features.to(device=x.device, dtype=x.dtype)
        # Flatten image_flat to 2D: (batch_size * num_image * num_patch, d_model)
        image_flat = image_flat.view(-1, image_flat.shape[-1])
        batch_offsets = (
            torch.arange(batch_size, device=x.device)
            .unsqueeze(1)
            .expand_as(image_input_idx)
            * seq_len
        )
        linear_idx = (batch_offsets + image_input_idx).view(-1)
        
        # Filter out invalid indices (negative or out-of-bounds)
        max_valid_idx = batch_size * seq_len - 1
        valid_mask = (linear_idx >= 0) & (linear_idx <= max_valid_idx)
        if (~valid_mask).sum().item() > 0:
            linear_idx = linear_idx[valid_mask]
            image_flat = image_flat[valid_mask]
        
        # Only call index_add_ if there are valid indices
        if linear_idx.numel() > 0:
            x_flat.index_add_(0, linear_idx, image_flat)
        x = x_flat.view(batch_size, seq_len, -1)
        
        if model.model.config.use_cls_feature:
            x = torch.cat([x[:, :1], cls_embed, x[:, 1:-num_image]], dim=1)
    
    # Step 3: Run transformer blocks up to insertion position
    # Save original top_k for blocks before insertion
    blocks_before_insertion = min(insertion_position, len(model.model.transformer.blocks))
    original_top_ks = {}
    
    for i in range(blocks_before_insertion):
        block = model.model.transformer.blocks[i]
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
            original_top_ks[i] = block.mlp.top_k
            # First block always fixed at 8, others can be set
            if i == 0:
                block.mlp.top_k = 8  # Fixed for first block
            # For blocks 1 to insertion_position-1, we can set a default or keep original
    
    # Forward through blocks up to insertion position
    layer_past = None
    for i in range(blocks_before_insertion):
        block = model.model.transformer.blocks[i]
        x, cache = block(
            x,
            attention_bias=attention_bias,
            position_ids=position_ids,
            drop_mask=None,
            layer_past=layer_past,
            use_cache=False,
        )
    
    # Restore original top_k for blocks before insertion
    for i, original_top_k in original_top_ks.items():
        if original_top_k is not None:
            model.model.transformer.blocks[i].mlp.top_k = original_top_k
    
    # Step 4: Extract latency token from insertion position block output
    # Use the last token (latency token) which has attended to vision and language tokens
    latency_token = x[:, -1, :]  # (B, d_model) - last token (latency token)
    
    # Step 5: Stage2 controller predicts knob2 & knob3 (if controller provided)
    if stage2_controller is not None and budget_feat is not None:
        # Use latency token + budget feat
        insertion_position_tensor = torch.tensor([insertion_position], device=x.device).expand(batch_size)
        knob2_knob3_output = stage2_controller(
            latency_token, budget_feat, insertion_position_tensor
        )
        
        knob2_logits = knob2_knob3_output['knob2_logits']
        knob3_logits = knob2_knob3_output['knob3_logits']
        
        # Sample actions (deterministic for inference, non-deterministic for training)
        knob2_idx = knob2_logits.argmax(dim=-1)  # (B,)
        knob3_idx = knob3_logits.argmax(dim=-1)  # (B,)
        
        # Map to values
        knob2_values = [4, 5, 6, 7, 8]
        knob2_top_k = torch.tensor([knob2_values[idx.item()] for idx in knob2_idx], device=x.device)
        
        # Get dynamic knob3 options based on insertion position
        knob3_options = stage2_controller.get_knob3_options(insertion_position)
        # For now, use first sample's knob3_idx to select from options
        # In practice, we might need to handle per-sample selection
        if len(knob3_options) > 0:
            knob3_idx_clamped = min(knob3_idx[0].item(), len(knob3_options) - 1)
            knob3_num_blocks = torch.tensor([knob3_options[knob3_idx_clamped]], device=x.device)
        else:
            knob3_num_blocks = torch.tensor([knob3_num_blocks], device=x.device)
    else:
        # Use provided values or defaults
        knob2_top_k = torch.tensor([knob2_top_k] * batch_size, device=x.device)
        knob3_num_blocks = torch.tensor([knob3_num_blocks] * batch_size, device=x.device)
    
    # Step 6: Apply knob2 & knob3 to subsequent blocks
    # Set top_k for blocks after insertion position
    total_blocks = len(model.model.transformer.blocks)
    for i in range(insertion_position, total_blocks):
        block = model.model.transformer.blocks[i]
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
            # Use the top_k value (can be per-sample, but for simplicity use first sample's value)
            block.mlp.top_k = knob2_top_k[0].item()
    
    # Apply block mask (knob3)
    # Total blocks = insertion_position + remaining blocks
    num_active_blocks = knob3_num_blocks[0].item()
    
    # Step 7: Continue forward pass through remaining blocks
    # We need to run blocks from insertion_position to num_active_blocks
    # But num_active_blocks is total, so we run from insertion_position to min(num_active_blocks, total_blocks)
    for i in range(insertion_position, min(num_active_blocks, total_blocks)):
        block = model.model.transformer.blocks[i]
        layer_past = None
        x, cache = block(
            x,
            attention_bias=attention_bias,
            position_ids=position_ids,
            drop_mask=None,
            layer_past=layer_past,
            use_cache=False,
        )
    
    # Step 8: Final layer norm
    x = model.model.transformer.ln_f(x)  # (B, seq_len, d_model)
    
    # Step 9: Get logits
    if model.model.config.weight_tying:
        logits = torch.nn.functional.linear(x, model.model.transformer.wte.weight, None)
    else:
        logits = model.model.lm_head(x)
    
    result = {
        'logits': logits,
        'hidden_states': x,
    }
    
    if return_latency_token:
        result['latency_token'] = latency_token
    
    return result

