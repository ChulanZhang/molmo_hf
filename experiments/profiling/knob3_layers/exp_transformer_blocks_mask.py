
import argparse
import logging
import sys
import os
from typing import Optional, List, Tuple, Sequence
from functools import wraps

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())
from experiments.base_experiment import BaseExperiment
from molmo.models.modeling_molmoe import (
    MolmoModel, 
    MolmoOutput,
    get_causal_attention_bias,
    ensure_finite_,
)

log = logging.getLogger(__name__)


class BlockMaskWrapper:
    """
    Wrapper to apply block masks during forward pass.
    This allows skipping certain transformer blocks without removing them from the model.
    """
    
    def __init__(self, model: MolmoModel, block_mask: torch.Tensor):
        """
        Args:
            model: The MolmoModel instance
            block_mask: Boolean tensor of shape (n_layers,) where True means the block is active
        """
        self.model = model
        self.block_mask = block_mask
        self.original_forward = None
        self.n_layers = len(model.transformer.blocks)
        
        assert len(block_mask) == self.n_layers, \
            f"Block mask length {len(block_mask)} must match number of layers {self.n_layers}"
        
        # Store the number of active blocks
        self.num_active_blocks = int(block_mask.sum().item())
        
    def _masked_forward(self, *args, **kwargs):
        """
        Modified forward pass that skips masked blocks.
        """
        # Get the original forward method's signature
        # We need to handle all the arguments that MolmoModel.forward accepts
        input_ids = kwargs.get('input_ids', args[0] if args else None)
        input_embeddings = kwargs.get('input_embeddings', None)
        attention_mask = kwargs.get('attention_mask', None)
        attention_bias = kwargs.get('attention_bias', None)
        response_mask = kwargs.get('response_mask', None)
        images = kwargs.get('images', None)
        image_masks = kwargs.get('image_masks', None)
        image_input_idx = kwargs.get('image_input_idx', None)
        subsegment_ids = kwargs.get('subsegment_ids', None)
        position_ids = kwargs.get('position_ids', None)
        past_key_values = kwargs.get('past_key_values', None)
        use_cache = kwargs.get('use_cache', False)
        last_logits_only = kwargs.get('last_logits_only', False)
        output_hidden_states = kwargs.get('output_hidden_states', None)
        append_last_valid_logits = kwargs.get('append_last_valid_logits', None)
        
        # Call the original forward up to the blocks loop
        # We need to replicate the logic from MolmoModel.forward up to the blocks loop
        # and then modify the blocks loop to respect the mask
        
        # This is a simplified approach: we'll call the original forward but intercept
        # at the blocks level. However, since we can't easily intercept, we'll need
        # to patch the forward method more carefully.
        
        # Actually, a better approach is to temporarily replace the blocks ModuleList
        # with a custom one that respects the mask. But that's complex.
        
        # Best approach: Monkey patch the forward method to check mask before each block
        # We'll do this by wrapping the blocks iteration
        
        # For now, let's use a simpler approach: create a custom forward that
        # replicates the original logic but with mask checking
        
        # Get the model's internal state
        model = self.model
        config = model.config
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        
        if past_key_values:
            assert len(past_key_values) == config.n_layers
        
        has_image = images is not None
        assert not (has_image and input_embeddings is not None), "Cannot provide both images and input embeddings."
        assert not (has_image and past_key_values is not None), "Cached key and values should not be used with images."
        
        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            # Find first active layer to get past_length
            # past_key_values length equals total layers, but some may be masked
            first_active_idx = None
            for idx in range(len(self.block_mask)):
                if self.block_mask[idx] and idx < len(past_key_values):
                    first_active_idx = idx
                    break
            if first_active_idx is not None and past_key_values[first_active_idx] is not None:
                past_length = past_key_values[first_active_idx][0].size(-2)
            else:
                # Fallback: use first layer if available
                if len(past_key_values) > 0 and past_key_values[0] is not None:
                    past_length = past_key_values[0][0].size(-2)
                else:
                    past_length = 0
        
        if config.unconditioned and input_embeddings is None:
            images = None
            image_input_idx = None
        
        if config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
        
        if subsegment_ids is not None:
            assert not use_cache, "Subsegment_ids cannot be used with cache."
            subsegment_mask = subsegment_ids.unsqueeze(2) <= subsegment_ids.unsqueeze(1)
            attention_mask = (
                subsegment_mask.to(attention_mask.dtype) *
                attention_mask.unsqueeze(2) *
                attention_mask.unsqueeze(1))
            if position_ids is None:
                raise ValueError(f"Positioned ids must be given if using subsegment_ids")
        else:
            if config.use_position_ids and position_ids is None:
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                    min=0,
                ).broadcast_to((batch_size, attention_mask.shape[-1]))
        
        # Get embeddings
        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        x = model.transformer.wte(input_ids) if input_embeddings is None else input_embeddings
        
        num_image: Optional[int] = None
        if images is not None:
            image_features, cls_embed = model.vision_backbone(images, image_masks)
            num_image, num_patch = image_features.shape[1:3]
            assert image_input_idx.shape == (batch_size, num_image, num_patch)
            
            image_features = image_features.view(batch_size, num_image * num_patch, -1)
            image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)

            # Use the original simpler approach with direct indexing
            valid = image_input_idx >= 0
            batch_idx = torch.arange(batch_size, device=x.device)
            batch_idx = torch.tile(batch_idx[:, None], [1, image_features.shape[1]])
            
            image_features = image_features.to(device=x.device, dtype=x.dtype)
            x[batch_idx[valid], image_input_idx[valid]] += image_features[valid]
            
            if config.use_cls_feature:
                x = torch.cat([x[:, :1], cls_embed, x[:, 1:-num_image]], dim=1)
                # Update seq_len after inserting cls_embed
                seq_len = x.shape[1]
                valid_images = torch.any(
                    (image_input_idx >= 0).view(batch_size, num_image, num_patch), dim=-1
                )
                valid_images = valid_images.to(attention_mask.dtype)
                attention_mask = torch.cat(
                    [attention_mask[:, :1], valid_images, attention_mask[:, 1:-num_image]],
                    dim=1,
                )
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask, dim=-1) - 1,
                    min=0,
                ).broadcast_to((batch_size, attention_mask.shape[-1]))
        
        # Add input + positional embeddings and apply dropout
        x = model.transformer.emb_drop(x)
        
        if config.normalize_input_embeds:
            x = x * (config.d_model ** 0.5)
        
        # Transform attention mask
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, :past_length + seq_len]
                attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            else:
                attention_mask = attention_mask.unsqueeze(1).to(dtype=torch.float)
        
        # Handle attention bias
        if attention_bias is None:
            attention_bias = get_causal_attention_bias(model._MolmoModel__cache, past_length + seq_len, x.device)
        elif attention_bias.dtype in (torch.int8, torch.bool):
            attention_bias = attention_bias.to(dtype=torch.float)
            attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)
        
        mask_len = seq_len
        if attention_mask is not None:
            mask_len = attention_mask.shape[-1]
        elif past_key_values is not None:
            # Find first active layer to get past_length
            # past_key_values length equals total layers, but some may be masked
            # We need to find the first active layer's cache
            first_active_idx = None
            for idx in range(len(self.block_mask)):
                if self.block_mask[idx] and idx < len(past_key_values):
                    first_active_idx = idx
                    break
            if first_active_idx is not None and past_key_values[first_active_idx] is not None:
                mask_len = past_key_values[first_active_idx][0].shape[-2] + seq_len
            else:
                # Fallback: use first layer if available
                if len(past_key_values) > 0 and past_key_values[0] is not None:
                    mask_len = past_key_values[0][0].shape[-2] + seq_len
        attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
        
        if attention_mask is not None:
            attention_bias = attention_bias + attention_mask
            ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)
        
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        
        # MODIFIED: Apply blocks with mask
        # IMPORTANT: We still call the layer even if masked, so that hooks registered on blocks
        # (e.g., by BaseExperiment.measure_inference_latency) can work correctly.
        # For masked blocks, we'll use a workaround: call the layer but with a flag to skip computation.
        # However, since we can't easily modify the layer's internal logic, we'll use a different approach:
        # For masked blocks, we'll still call the layer, but immediately return the input.
        # This ensures hooks are triggered, but we need to handle the cache properly.
        
        all_hidden_states = []
        for block_idx, layer in enumerate(model.transformer.blocks):
            if output_hidden_states:
                all_hidden_states.append(x)
            
            # Check mask: if block is masked (False), we still need to call it for hooks to work
            # but we'll make it a no-op by using a special approach
            if not self.block_mask[block_idx]:
                # For masked blocks, we still call the layer to ensure hooks work,
                # but we'll immediately return the input without computation.
                # However, since we can't easily modify layer's forward, we'll use a wrapper.
                # Actually, the simplest approach: call the layer normally, but it will be slow.
                # Better: use a context manager or hook to skip computation.
                # 
                # Actually, the best approach for hooks: we need to ensure the layer is called
                # so hooks can fire. But we want to skip the computation.
                # 
                # Solution: We'll create a temporary identity wrapper that preserves hooks.
                # But this is complex. Let's use a simpler approach:
                # For masked blocks, we'll still iterate and call the layer, but we'll
                # use a no-op wrapper. However, this requires modifying the layer.
                #
                # Actually, the simplest solution: For masked blocks, we still call the layer
                # but we'll make it return immediately. We can do this by patching the layer's
                # forward method temporarily, or by using a wrapper.
                #
                # For now, let's use a pragmatic approach: For masked blocks, we'll still call
                # the layer (so hooks work), but we'll wrap it to make it a no-op.
                # However, this is complex and may have performance impact.
                #
                # Better solution: Find first and last active blocks, and ensure hooks work on those.
                # But BaseExperiment registers hooks on blocks[0] and blocks[-1], which may be masked.
                #
                # Best solution: We'll ensure that even masked blocks are "called" in a way that
                # triggers hooks. We can do this by creating a dummy forward that just returns
                # the input, but still triggers the hook.
                
                # For masked blocks: We need to trigger hooks but skip computation.
                # The hook is registered on the layer itself, so we need to call the layer.
                # But we want to skip the computation. We can do this by temporarily
                # replacing the layer's forward with an identity function.
                
                # Store original forward
                original_layer_forward = layer.forward
                
                # Create identity forward that preserves hook behavior
                # We need to capture block_idx, use_cache, and past_key_values in closure
                def make_identity_forward(block_idx_inner, use_cache_inner, past_key_values_inner, seq_len_inner):
                    def identity_forward(layer_self, x_inner, attention_bias=None, position_ids=None, 
                                       drop_mask=None, layer_past=None, use_cache=False):
                        # Return input unchanged, but with proper cache handling
                        if use_cache_inner:
                            if past_key_values_inner is not None and block_idx_inner < len(past_key_values_inner):
                                # Use existing cache if available
                                existing_cache = past_key_values_inner[block_idx_inner]
                                if existing_cache is not None:
                                    # Check if cache size matches current sequence length
                                    # If layer_past is provided, it should match the expected size
                                    if layer_past is not None:
                                        cache = layer_past
                                    else:
                                        # Use existing cache, but ensure it matches current seq_len
                                        # For skipped layers, we need to maintain cache structure
                                        cache = existing_cache
                                else:
                                    # Create dummy cache with same shape as input
                                    batch_size, seq_len, d_model = x_inner.shape
                                    cache = (torch.zeros_like(x_inner), torch.zeros_like(x_inner))
                            else:
                                # Create dummy cache with same shape as input
                                batch_size, seq_len, d_model = x_inner.shape
                                cache = (torch.zeros_like(x_inner), torch.zeros_like(x_inner))
                        else:
                            cache = None
                        return x_inner, cache
                    return identity_forward
                
                # Temporarily replace forward with identity
                layer.forward = make_identity_forward(block_idx, use_cache, past_key_values, seq_len).__get__(layer, type(layer))
                
                # Call layer (this will trigger hooks but do no computation)
                layer_past = None if past_key_values is None else past_key_values[block_idx]
                x, cache = layer(x, attention_bias=attention_bias, position_ids=position_ids, 
                                drop_mask=response_mask, layer_past=layer_past, use_cache=use_cache)
                
                # Restore original forward
                layer.forward = original_layer_forward
            else:
                # Normal block execution
                layer_past = None if past_key_values is None else past_key_values[block_idx]
                x, cache = layer(x, attention_bias=attention_bias, position_ids=position_ids, 
                                drop_mask=response_mask, layer_past=layer_past, use_cache=use_cache)
            
            if attn_key_values is not None:
                assert cache is not None, f"Cache must not be None when use_cache=True (block {block_idx})"
                attn_key_values.append(cache)
        
        # Continue with rest of forward pass
        if images is not None and config.use_cls_feature:
            assert num_image is not None
            x = torch.cat(
                [x[:, :1], x[:, num_image+1:], torch.zeros_like(x[:, :num_image])],
                dim=1,
            )
        
        if last_logits_only:
            if append_last_valid_logits is not None:
                last_valid_output = x[
                    torch.arange(x.shape[0], device=x.device), append_last_valid_logits
                ]
                x = last_valid_output.unsqueeze(1)
            else:
                x = x[:, -1:, :]
        
        # Apply final layer norm
        x = model.transformer.ln_f(x)
        
        if output_hidden_states:
            all_hidden_states.append(x)
        
        return MolmoOutput(
            attn_key_values=tuple(attn_key_values) if attn_key_values is not None else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            last_hidden_states=x,
        )
    
    def apply(self):
        """Apply the mask by replacing the forward method."""
        self.original_forward = self.model.forward
        self.model.forward = self._masked_forward
    
    def remove(self):
        """Remove the mask and restore original forward method."""
        if self.original_forward is not None:
            self.model.forward = self.original_forward
            self.original_forward = None


class TransformerBlocksMaskExperiment(BaseExperiment):
    """
    Experiment to profile the impact of number of active transformer blocks
    using a mask mechanism (not early exit, but skipping blocks via mask).
    """
    
    def run(
        self,
        dataset_name: str = None,
        split: str = "validation",
        num_samples: int = 5000,
        batch_size: int = 1,
        max_new_tokens: int = 128,
        num_active_blocks_list: List[int] = None, 
        active_block_indices_list: List[List[int]] = None,
    ):
        """
        Run Transformer Blocks Mask experiment.
        
        Args:
            dataset_name: Dataset name (e.g., "coco_2014_vqa"). If None, uses dummy images.
            split: Dataset split (default: "validation")
            num_samples: Number of samples to measure (default: 5000 for dataset mode, 100 for dummy mode)
            batch_size: Batch size (default: 1 for latency measurement)
            max_new_tokens: Maximum tokens to generate
            num_active_blocks_list: List of numbers of active blocks to test.
                                   If None, defaults to testing various fractions of total blocks.
                                   Used when active_block_indices_list is None.
            active_block_indices_list: List of lists, each containing specific block indices to activate.
                                      If provided, this takes precedence over num_active_blocks_list.
                                      Example: [[0, 1, 2], [0, 5, 10, 15]] to test different block combinations.
        """
        # Determine mode: dataset mode or dummy image mode
        use_dataset = dataset_name is not None
        
        if not use_dataset:
            # Dummy image mode: original behavior
            processor = self.processor
            
            # Standard input
            image = Image.new('RGB', (336, 336), color='blue')
            prompt = "Describe this image."
            inputs = processor.process(text=prompt, images=image)
            
            # Ensure batch dimension
            if inputs["input_ids"].ndim == 1:
                inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
                if "images" in inputs and inputs["images"] is not None:
                    inputs["images"] = inputs["images"].unsqueeze(0)
                if "image_masks" in inputs and inputs["image_masks"] is not None:
                    inputs["image_masks"] = inputs["image_masks"].unsqueeze(0)
                if "image_input_idx" in inputs and inputs["image_input_idx"] is not None:
                    inputs["image_input_idx"] = inputs["image_input_idx"].unsqueeze(0)
        
        # Get total number of blocks
        total_blocks = len(self.model.model.transformer.blocks)
        log.info(f"Total transformer blocks: {total_blocks}")
        
        # Determine which mode to use: specific indices or count-based
        if active_block_indices_list is not None:
            # Mode 1: Use specific block indices
            log.info(f"Using specific block indices mode")
            log.info(f"Testing {len(active_block_indices_list)} block configurations")
            
            # Validate all indices
            for idx_list in active_block_indices_list:
                if not all(0 <= idx < total_blocks for idx in idx_list):
                    raise ValueError(f"Invalid block indices: {idx_list}. Must be in range [0, {total_blocks-1}]")
            
            configs_to_test = active_block_indices_list
            use_specific_indices = True
        else:
            # Mode 2: Use count-based (default)
            # Default: test all block counts from 1 to total_blocks
            if num_active_blocks_list is None:
                num_active_blocks_list = list(range(1, total_blocks + 1))
                log.info(f"Using default: testing all block counts from 1 to {total_blocks}")
            else:
                log.info(f"Using custom block counts: {num_active_blocks_list}")
            
            # Validate
            num_active_blocks_list = [n for n in num_active_blocks_list if 1 <= n <= total_blocks]
            if not num_active_blocks_list:
                raise ValueError("No valid number of active blocks to test")
            
            log.info(f"Using count-based mode")
            log.info(f"Testing active block counts: {num_active_blocks_list}")
            
            # Convert counts to indices (first N blocks)
            configs_to_test = [list(range(n)) for n in num_active_blocks_list]
            use_specific_indices = False
        
        results_data = []
        mask_wrapper = None
        
        try:
            for block_indices in tqdm(configs_to_test, desc="Block Mask Scaling"):
                num_active = len(block_indices)
                log.info(f"Setting {num_active} active blocks: {block_indices}")
                
                # Create mask: activate specified blocks
                block_mask = torch.zeros(total_blocks, dtype=torch.bool)
                for idx in block_indices:
                    block_mask[idx] = True
                
                log.info(f"Block mask: {block_mask.tolist()}")
                log.info(f"Active blocks: {block_indices}")
                
                # Apply mask wrapper
                if mask_wrapper is not None:
                    mask_wrapper.remove()
                
                mask_wrapper = BlockMaskWrapper(self.model.model, block_mask)
                mask_wrapper.apply()
                
                log.info(f"Applied mask: {num_active} active blocks")
                
                # Run measurements
                if use_dataset:
                    # Dataset mode: use VQA v2 validation set
                    dataloader = self.build_dataloader(
                        dataset_name=dataset_name,
                        split=split,
                        batch_size=batch_size,
                        max_steps=num_samples,
                        shuffle=False,
                    )
                    
                    all_latencies = []
                    per_sample_results = []
                    
                    # Warmup
                    log.info("Warming up...")
                    warmup_batch = next(iter(dataloader))
                    for _ in range(self.num_warmup):
                        self.measure_inference_latency(
                            warmup_batch,
                            max_new_tokens=max_new_tokens,
                            measure_components=True,
                            num_runs=1,
                        )
                    
                    # Measure latency
                    log.info(f"Measuring latency for {num_active}/{total_blocks} active blocks...")
                    for sample_idx, batch in enumerate(tqdm(dataloader, total=min(num_samples, len(dataloader)))):
                        if sample_idx >= num_samples:
                            break
                        
                        metrics = self.measure_inference_latency(
                            batch,
                            max_new_tokens=max_new_tokens,
                            measure_components=True,
                            num_runs=1,
                        )
                        
                        total_latency = metrics.get("T_total", 0.0)
                        all_latencies.append(total_latency)
                        
                        per_sample_results.append({
                            "sample_id": sample_idx,
                            "num_active_blocks": num_active,
                            "num_total_blocks": total_blocks,
                            "active_block_indices": block_indices,
                            **metrics
                        })
                    
                    # Compute statistics
                    stats = self.compute_statistics(all_latencies)
                    stats["num_active_blocks"] = num_active
                    stats["num_total_blocks"] = total_blocks
                    stats["active_block_indices"] = block_indices
                    stats["per_sample_results"] = per_sample_results
                    
                    results_data.append(stats)
                    
                    log.info(f"{num_active}/{total_blocks} active blocks: Mean Latency={stats['mean']:.2f}ms, "
                            f"P50={stats['P50']:.2f}ms")
                else:
                    # Dummy image mode: original behavior
                    latencies_prefill = []
                    latencies_decode = []
                    per_sample_results = []
                    
                    # Warmup
                    for _ in range(self.num_warmup):
                        self.measure_inference_latency(inputs, max_new_tokens=10, measure_components=True)
                    
                    for sample_idx in range(num_samples):
                        metrics = self.measure_inference_latency(inputs, max_new_tokens=10, measure_components=True)
                        latencies_prefill.append(metrics["T_LLM_prefill"])
                        latencies_decode.append(metrics["T_LLM_decode"])
                        
                        # Save per-sample result
                        per_sample_results.append({
                            "sample_id": sample_idx,
                            "num_active_blocks": num_active,
                            "num_total_blocks": total_blocks,
                            "active_block_indices": block_indices,
                            **metrics  # Include all metrics from measure_inference_latency
                        })
                    
                    stats_prefill = self.compute_statistics(latencies_prefill)
                    stats_decode = self.compute_statistics(latencies_decode)
                    
                    combined_stats = {
                        "num_active_blocks": num_active,
                        "num_total_blocks": total_blocks,
                        "active_block_indices": block_indices,
                        "prefill": stats_prefill,
                        "decode": stats_decode,
                        "per_sample_results": per_sample_results
                    }
                    results_data.append(combined_stats)
                    
                    log.info(f"Active Blocks {num_active}/{total_blocks} (indices: {block_indices}): "
                            f"P50 Prefill={stats_prefill['P50']:.2f}ms, "
                            f"P50 Decode={stats_decode['P50']:.2f}ms")
        
        finally:
            # Always restore original forward method
            if mask_wrapper is not None:
                mask_wrapper.remove()
                log.info("Restored original forward method")
        
        # Save results with per-sample data
        # Format: {"summary": [...], "all_samples": [...]}
        all_samples = []
        summary = []
        
        for config_result in results_data:
            # Extract summary stats
            summary_entry = {
                "num_active_blocks": config_result["num_active_blocks"],
                "num_total_blocks": config_result["num_total_blocks"],
                "active_block_indices": config_result["active_block_indices"],
                "prefill": config_result["prefill"],
                "decode": config_result["decode"]
            }
            summary.append(summary_entry)
            
            # Collect all per-sample results
            all_samples.extend(config_result["per_sample_results"])
        
        final_results = {
            "summary": summary,
            "all_samples": all_samples
        }
        
        self.save_results(final_results, "exp3_transformer_blocks_mask_results.json")
        log.info(f"Results saved to {self.output_dir}/exp3_transformer_blocks_mask_results.json")
        log.info(f"Total samples: {len(all_samples)}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile transformer blocks using mask mechanism (not early exit)"
    )
    parser.add_argument("--model_path", type=str, default="checkpoints",
                        help="Path to model checkpoint directory (local path, not HF hub path)")
    parser.add_argument("--output_dir", type=str, default="./results/transformer_blocks_mask",
                        help="Output directory for results")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset name (e.g., 'coco_2014_vqa'). If None, uses dummy images.")
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split (default: validation)")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of samples to measure (default: 5000 for dataset mode, 50 for dummy mode)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1 for latency measurement)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--num_active_blocks", type=int, nargs="+", default=None,
                        help="List of numbers of active blocks to test. "
                             "If not provided, defaults to testing various fractions of total blocks. "
                             "This activates the first N blocks (from early to late). "
                             "Example: --num_active_blocks 4 8 12 16")
    parser.add_argument("--active_block_indices", type=int, nargs="+", action="append", default=None,
                        help="List of specific block indices to activate. Can be specified multiple times "
                             "to test different block combinations. "
                             "This takes precedence over --num_active_blocks. "
                             "Example: --active_block_indices 0 1 2 3 --active_block_indices 0 5 10 15")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    experiment = TransformerBlocksMaskExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    experiment.run(
        dataset_name=args.dataset_name,
        split=args.split,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_active_blocks_list=args.num_active_blocks,
        active_block_indices_list=args.active_block_indices
    )


if __name__ == "__main__":
    main()

