"""
Model forward pass with Stage2 controller insertion after first transformer block.

This module implements a custom forward pass that:
1. Runs vision encoder + projector
2. Runs first transformer block (fixed top_k=8)
3. Extracts features from first block output
4. Stage2 controller predicts knob2 & knob3
5. Applies knob2 & knob3 to subsequent blocks
6. Continues forward pass
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import logging

log = logging.getLogger(__name__)


def forward_with_stage2_controller(
    model,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    image_masks: Optional[torch.Tensor] = None,
    image_input_idx: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    knob1_tier: str = "medium",
    knob2_top_k: int = 8,
    knob3_num_blocks: int = 16,
    stage2_controller: Optional[nn.Module] = None,
    vision_feat: Optional[torch.Tensor] = None,
    lang_feat: Optional[torch.Tensor] = None,
    budget_feat: Optional[torch.Tensor] = None,
    return_stage2_features: bool = False,
) -> Dict[str, Any]:
    """
    Forward pass with Stage2 controller inserted after first transformer block.
    
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
        knob2_top_k: Top-K value (if not using controller, or as fallback)
        knob3_num_blocks: Number of active blocks (if not using controller, or as fallback)
        stage2_controller: Optional Stage2 controller (if None, uses knob2_top_k and knob3_num_blocks)
        vision_feat: Optional pre-extracted vision features (for Stage2)
        lang_feat: Optional pre-extracted language features (for Stage2)
        budget_feat: Optional pre-extracted budget features (for Stage2)
        return_stage2_features: If True, return features extracted for Stage2
    
    Returns:
        Dict with 'logits', 'hidden_states', 'stage2_features' (if return_stage2_features)
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
    
    # Step 3: Run first transformer block (fixed top_k=8)
    # Save original top_k for first block
    first_block = model.model.transformer.blocks[0]
    original_first_top_k = None
    if hasattr(first_block, 'mlp') and hasattr(first_block.mlp, 'top_k'):
        original_first_top_k = first_block.mlp.top_k
        first_block.mlp.top_k = 8  # Fixed
    
    # Forward through first block
    layer_past = None
    x, cache = first_block(
        x,
        attention_bias=attention_bias,
        position_ids=position_ids,
        drop_mask=None,
        layer_past=layer_past,
        use_cache=False,
    )
    
    # Restore original top_k
    if original_first_top_k is not None:
        first_block.mlp.top_k = original_first_top_k
    
    # Step 4: Extract features from first block output for Stage2
    # Use the last token (latency token) which has attended to vision and language tokens
    # The last token contains rich information after attention with vision and language tokens
    stage2_input_feat = x[:, -1, :]  # (B, d_model) - last token (latency token)
    
    # Step 5: Stage2 controller predicts knob2 & knob3 (if controller provided)
    if stage2_controller is not None and vision_feat is not None and lang_feat is not None and budget_feat is not None:
        # Use pre-extracted features
        knob2_logits, knob3_logits = stage2_controller(
            vision_feat, lang_feat, budget_feat
        )
        
        # Sample actions (deterministic for inference, non-deterministic for training)
        knob2_idx = knob2_logits.argmax(dim=-1)  # (B,)
        knob3_idx = knob3_logits.argmax(dim=-1)  # (B,)
        
        # Map to values
        knob2_values = [4, 5, 6, 7, 8]
        knob3_values = [12, 13, 14, 15, 16]
        knob2_top_k = torch.tensor([knob2_values[idx.item()] for idx in knob2_idx], device=x.device)
        knob3_num_blocks = torch.tensor([knob3_values[idx.item()] for idx in knob3_idx], device=x.device)
    else:
        # Use provided values or defaults
        knob2_top_k = torch.tensor([knob2_top_k] * batch_size, device=x.device)
        knob3_num_blocks = torch.tensor([knob3_num_blocks] * batch_size, device=x.device)
    
    # Step 6: Apply knob2 & knob3 to subsequent blocks
    # Set top_k for blocks 1 onwards
    for i in range(1, len(model.model.transformer.blocks)):
        block = model.model.transformer.blocks[i]
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
            # Use the top_k value (can be per-sample, but for simplicity use first sample's value)
            block.mlp.top_k = knob2_top_k[0].item()
    
    # Apply block mask (knob3)
    # For simplicity, we'll use prefix blocks (first N blocks)
    # In practice, you might want to use importance-based selection
    num_active_blocks = knob3_num_blocks[0].item()
    total_blocks = len(model.model.transformer.blocks)
    
    # Step 7: Continue forward pass through remaining blocks
    for i in range(1, min(num_active_blocks, total_blocks)):
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
    
    if return_stage2_features:
        result['stage2_features'] = stage2_input_feat
    
    return result

