"""
Joint GRPO Trainer for Stage1 and Stage2.
Trains both stages end-to-end with shared reward signal.

Key design:
- Stage1 (Knob1) and Stage2 (Knob2 & Knob3) are trained jointly
- Both stages contribute to the same reward (accuracy + latency constraint)
- Uses GRPO for both stages to optimize end-to-end performance
- Directly measures latency using hooks (since batch_size=1 per sample)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json
import time
import random

log = logging.getLogger(__name__)


class JointGRPOTrainer:
    """
    Joint GRPO trainer for One-Stage Controller.
    
    One-stage controller predicts all knobs upfront:
    - Tier (Knob1): Vision tokens tier (low/medium/high)
    - Block Mask (Knob3): Binary mask for blocks 1-15 (block0 always on)
    - Per-block Top-K (Knob2): Top-K value for each block (block0 fixed at 8)
    
    Key design:
    - Direct Latency Measurement: Uses PyTorch hooks (batch_size=1 per sample)
    - Budget Token: Encoded and concatenated in prefill phase only
    - Decode Phase: Uses prefill configuration, no controller re-run
    - Block Activation Quota: At least 12 blocks, at most 16 blocks (including block0)
    - Block0 Top-K: Fixed at 8
    """
    
    def __init__(
        self,
        one_stage_controller: nn.Module,  # OneStageControllerPredictor
        model,
        reward_fn,
        budget_encoder: Optional[nn.Module] = None,  # LatencyBudgetEncoder instance
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
        group_size: int = 5,
        importance_scores_file: Optional[str] = None,  # Path to importance scores JSON file (for fallback)
        min_active_blocks: int = 12,  # Minimum active blocks (including block0)
        max_active_blocks: int = 16,  # Maximum active blocks (including block0)
    ):
        """
        Args:
            one_stage_controller: One-stage controller predictor
            model: MolmoModel instance
            reward_fn: Reward function
            budget_encoder: LatencyBudgetEncoder instance
            device: Device to use
            lr: Learning rate
            weight_decay: Weight decay
            max_grad_norm: Maximum gradient norm
            group_size: Group size for GRPO
            importance_scores_file: Path to importance scores JSON file (for fallback block selection)
            min_active_blocks: Minimum active blocks (including block0)
            max_active_blocks: Maximum active blocks (including block0)
        """
        self.one_stage_controller = one_stage_controller.to(device)
        self.model = model.to(device)
        self.reward_fn = reward_fn
        self.budget_encoder = budget_encoder  # Store budget encoder for model forward
        # Convert device string to torch.device object
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_grad_norm = max_grad_norm
        self.group_size = group_size
        self.min_active_blocks = min_active_blocks
        self.max_active_blocks = max_active_blocks
        
        # Optimizer: train controller and budget encoder MLP
        optimizer_params = [
            {'params': self.one_stage_controller.parameters(), 'lr': lr},
        ]
        
        # Add budget encoder MLP parameters if budget_encoder is provided
        if budget_encoder is not None:
            if hasattr(budget_encoder, 'mlp'):
                optimizer_params.append({
                    'params': budget_encoder.mlp.parameters(),
                    'lr': lr,
                })
            elif hasattr(budget_encoder, 'encoder'):
                optimizer_params.append({
                    'params': budget_encoder.encoder.parameters(),
                    'lr': lr,
                })
        
        self.optimizer = optim.Adam(
            optimizer_params,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # Training history
        self.train_losses = []
        self.train_rewards = []
        self.val_metrics_history = []
        
        # Load importance scores for fallback block selection
        self.importance_scores = None
        if importance_scores_file is not None:
            try:
                from experiments.controller.importance_based_block_selection import load_importance_scores
                self.importance_scores = load_importance_scores(importance_scores_file)
                log.info(f"Loaded importance scores from {importance_scores_file}")
            except Exception as e:
                log.warning(f"Failed to load importance scores from {importance_scores_file}: {e}")
                log.warning("Will use prefix blocks (first N blocks) as fallback")
    
    def _get_model(self):
        """Get the underlying model, handling DataParallel/DistributedDataParallel wrapper."""
        return self.model.module if isinstance(self.model, (DataParallel, DistributedDataParallel)) else self.model
    
    def _set_per_block_top_k(self, per_block_top_k: Dict[int, int]):
        """
        Set per-block top_k for MoE blocks.
        
        Args:
            per_block_top_k: Dictionary mapping block index to top_k value
                            {block_idx: top_k_value}
                            Block0 is always fixed at 8 (will be overridden if in dict)
        
        Note: When using DataParallel, we need to set top_k on all replicas.
        """
        model = self._get_model()
        transformer = model.model.transformer
        if hasattr(transformer, 'blocks'):
            blocks = transformer.blocks
        else:
            blocks = []
        
        # Set top_k for each block
        for block_idx, top_k in per_block_top_k.items():
            if 0 <= block_idx < len(blocks):
                block = blocks[block_idx]
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
                    block.mlp.top_k = top_k
        
        # Block0 is always fixed at 8 (override if needed)
        if len(blocks) > 0:
            block0 = blocks[0]
            if hasattr(block0, 'mlp') and hasattr(block0.mlp, 'top_k'):
                block0.mlp.top_k = 8
    
    def _select_blocks_by_importance(
        self, 
        num_active_blocks: int, 
        total_blocks: int = 16,
        start_block: int = 0,
    ) -> List[int]:
        """
        Select blocks based on importance scores.
        
        Args:
            num_active_blocks: Number of blocks to select from remaining blocks after start_block
            total_blocks: Total number of blocks in model (default: 16)
            start_block: Starting block index (blocks before this are always included)
        
        Returns:
            selected_blocks: List of selected block indices
            - Always includes blocks [0, 1, ..., start_block-1]
            - Selects num_active_blocks from blocks [start_block, ..., total_blocks-1]
        """
        # Always include blocks before start_block
        blocks_before = list(range(start_block))
        
        # Remaining blocks to select from
        remaining_blocks_count = total_blocks - start_block
        
        if num_active_blocks >= remaining_blocks_count:
            # Select all remaining blocks
            return blocks_before + list(range(start_block, total_blocks))
        
        if num_active_blocks <= 0:
            return blocks_before
        
        # Select from remaining blocks based on importance scores
        if self.importance_scores is not None:
            # Get importance scores for remaining blocks
            remaining_scores = {
                k: v for k, v in self.importance_scores.items() 
                if k >= start_block
            }
            sorted_blocks = sorted(
                remaining_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            # Select top-N from remaining blocks
            selected_remaining = [block_idx for block_idx, _ in sorted_blocks[:num_active_blocks]]
            selected_blocks = blocks_before + sorted(selected_remaining)
        else:
            # Fallback: use prefix blocks (first N blocks after start_block)
            selected_blocks = blocks_before + list(range(start_block, start_block + num_active_blocks))
        
        return selected_blocks
    
    def _apply_block_mask(self, block_mask: torch.Tensor, total_blocks: int = 16):
        """
        Apply block mask to model (store for use during forward pass).
        
        Note: The actual block skipping is handled in model.forward() or via hooks.
        For now, we store the mask and the model should check it during forward.
        
        Args:
            block_mask: (total_blocks,) boolean mask indicating which blocks are active
            total_blocks: Total number of blocks in model (default: 16)
        """
        # Store block mask for use during forward pass
        # The model's forward method should check this mask
        # For now, we'll use a simple approach: store in model attribute
        model = self._get_model()
        if not hasattr(model, '_active_block_mask'):
            model._active_block_mask = None
        model._active_block_mask = block_mask.cpu().numpy() if isinstance(block_mask, torch.Tensor) else block_mask
    
    def _sample_block_mask_from_logits(
        self,
        block_logits: torch.Tensor,  # (B, 15) for blocks 1-15
        min_active: int = 12,  # Minimum active blocks (including block0)
        max_active: int = 16,  # Maximum active blocks (including block0)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample block mask from logits, ensuring block0 is always active and total active blocks
        is between min_active and max_active.
        
        Args:
            block_logits: (B, 15) logits for blocks 1-15 (block0 always on)
            min_active: Minimum active blocks (including block0)
            max_active: Maximum active blocks (including block0)
        
        Returns:
            block_mask: (B, 16) boolean mask (block0 always True)
            log_probs: (B,) log probability of sampled mask
        """
        batch_size = block_logits.shape[0]
        total_blocks = 16
        
        # We need to select between (min_active - 1) and (max_active - 1) blocks from blocks 1-15
        # (since block0 is always on)
        min_selected = max(0, min_active - 1)  # At least this many from blocks 1-15
        max_selected = min(15, max_active - 1)  # At most this many from blocks 1-15
        
        block_masks = []
        log_probs_list = []
        
        for b in range(batch_size):
            # Get logits for this sample
            sample_logits = block_logits[b]  # (15,)
            
            # Sample number of blocks to activate (from blocks 1-15)
            num_to_select = torch.randint(min_selected, max_selected + 1, (1,), device=block_logits.device).item()
            
            # Sample top-k blocks based on logits
            block_probs = F.softmax(sample_logits, dim=-1)  # (15,)
            selected_indices = torch.multinomial(block_probs, num_to_select, replacement=False)  # (num_to_select,)
            
            # Create mask: block0 always True, selected blocks from 1-15 also True
            mask = torch.zeros(total_blocks, dtype=torch.bool, device=block_logits.device)
            mask[0] = True  # Block0 always on
            mask[selected_indices + 1] = True  # Blocks 1-15 (indices 0-14 in logits -> block indices 1-15)
            
            # Compute log probability
            selected_log_probs = F.log_softmax(sample_logits, dim=-1)[selected_indices]  # (num_to_select,)
            log_prob = selected_log_probs.sum()  # Sum of log probs for selected blocks
            
            block_masks.append(mask)
            log_probs_list.append(log_prob)
        
        block_mask = torch.stack(block_masks)  # (B, 16)
        log_probs = torch.stack(log_probs_list)  # (B,)
        
        return block_mask, log_probs
    
    def _measure_latency_with_hooks(
        self,
        batch: Dict[str, torch.Tensor],
        max_new_tokens: int,
        generation_config,
        latency_budget: Optional[torch.Tensor] = None,  # (B,) latency budget in ms
    ) -> Dict[str, Any]:
        """
        Measure prefill and decode latency using hooks (similar to BaseExperiment._measure_with_hooks).
        
        Since we're executing samples one by one (batch_size=1), we can directly measure
        actual latency instead of using an estimator. Use prefill latency as primary metric.
        
        Args:
            batch: Single sample batch (batch_size=1)
            max_new_tokens: Maximum tokens to generate
            generation_config: GenerationConfig object
        
        Returns:
            Dictionary with latency measurements and output_ids
        """
        # Initialize measurement containers
        vision_times = []
        prefill_times = []
        decode_times = []
        
        # Hook state
        vision_start_time = None
        prefill_start_time = None
        forward_count = 0  # Track prefill (0) vs decode (>0)
        decode_start_time = None
        
        # Vision hook (only for prefill step)
        vision_hook_handle = None
        if "images" in batch and batch["images"] is not None:
            model = self._get_model()
            vision_backbone = model.model.vision_backbone
            
            def vision_hook(module, input, output):
                nonlocal vision_start_time
                if vision_start_time is not None:
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    vision_end_time = time.perf_counter()
                    vision_times.append((vision_end_time - vision_start_time) * 1000)
                    vision_start_time = None
            
            vision_hook_handle = vision_backbone.register_forward_hook(vision_hook)
        
        # Transformer hooks (for prefill)
        model = self._get_model()
        transformer = model.model.transformer
        prefill_start_hook_handle = None
        prefill_end_hook_handle = None
        
        if hasattr(transformer, 'blocks') and len(transformer.blocks) > 0:
            def prefill_start_hook(module, input, output):
                nonlocal prefill_start_time, forward_count
                if forward_count == 0:  # Only in prefill step
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    prefill_start_time = time.perf_counter()
            
            def prefill_end_hook(module, input, output):
                nonlocal prefill_start_time, forward_count
                if forward_count == 0 and prefill_start_time is not None:
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    end_time = time.perf_counter()
                    prefill_times.append((end_time - prefill_start_time) * 1000)
                    prefill_start_time = None
            
            prefill_start_hook_handle = transformer.blocks[0].register_forward_hook(prefill_start_hook)
            prefill_end_hook_handle = transformer.blocks[-1].register_forward_hook(prefill_end_hook)
        
        # Custom forward wrapper to track forward count and measure decode
        model = self._get_model()
        original_forward = model.model.forward
        
        def tracked_forward(*args, **kwargs):
            nonlocal forward_count, vision_start_time, decode_start_time
            is_prefill = forward_count == 0
            
            # Record vision start time (before forward, only in prefill step)
            if is_prefill and vision_start_time is None and "images" in batch and batch["images"] is not None:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                vision_start_time = time.perf_counter()
            
            # Record decode start time (only on first decode step)
            if not is_prefill and decode_start_time is None:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                decode_start_time = time.perf_counter()
            
            # Call original forward
            output = original_forward(*args, **kwargs)
            
            # Increment forward count after prefill
            if is_prefill:
                forward_count += 1
            
            return output
        
        # Replace forward with tracked version
        model.model.forward = tracked_forward
        
        try:
            # Measure total time
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            total_start = time.perf_counter()
            
            # Generate
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    # Pass latency_budget and budget_encoder to model.generate
                    # Note: model.generate will pass these to model.forward
                    output_ids = self.model.generate(
                        input_ids=batch["input_ids"],
                        images=batch.get("images"),
                        image_masks=batch.get("image_masks"),
                        image_input_idx=batch.get("image_input_idx"),
                        generation_config=generation_config,
                        latency_budget=latency_budget if latency_budget is not None else batch.get("latency_budget"),  # (B,) latency budget in ms
                        budget_encoder=self.budget_encoder,  # LatencyBudgetEncoder instance
                    )
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            total_end = time.perf_counter()
            total_latency = (total_end - total_start) * 1000
            
            # Record decode end time
            if decode_start_time is not None:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                decode_end_time = time.perf_counter()
                decode_times.append((decode_end_time - decode_start_time) * 1000)
        finally:
            # Restore original forward
            model.model.forward = original_forward
            
            # Remove hooks
            if vision_hook_handle is not None:
                vision_hook_handle.remove()
            if prefill_start_hook_handle is not None:
                prefill_start_hook_handle.remove()
            if prefill_end_hook_handle is not None:
                prefill_end_hook_handle.remove()
        
        # Compute results
        results = {
            'output_ids': output_ids,
            'T_vision_total': vision_times[0] if vision_times else 0.0,
            'T_LLM_prefill': prefill_times[0] if prefill_times else 0.0,
            'T_LLM_decode': decode_times[0] if decode_times else 0.0,
            'T_total': total_latency,
        }
        
        return results

    def _get_max_new_tokens(self, dataset_name: Optional[str]) -> int:
        """
        Align max_new_tokens with core_exp configs.
        Defaults to short-answer setting if dataset unknown.
        """
        if not dataset_name:
            return 16
        name = dataset_name.lower()
        if name in ["text_vqa"]:
            return 64
        if name in ["coco_caption", "coco_captioning"]:
            return 64
        if name in ["st_qa", "doc_qa"]:
            return 32
        # common short-answer QA
        return 16
    
    def _execute_model(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        tier: str = "medium",
        per_block_top_k: Optional[Dict[int, int]] = None,  # {block_idx: top_k_value}
        block_mask: Optional[torch.Tensor] = None,  # (16,) boolean mask
        latency_budget: Optional[torch.Tensor] = None,  # (B,) latency budget in ms
        max_new_tokens: int = 64,
        metadata: Optional[Dict] = None,
        tokenizer = None,
    ) -> Dict[str, Any]:
        """
        Execute model with one-stage controller configuration.
        
        Since batch_size=1 per sample, we directly measure latency using hooks.
        Prefill latency is used as the primary metric.
        
        Args:
            input_ids: Input token IDs
            images: Optional images
            image_masks: Optional image masks
            image_input_idx: Optional image input indices
            tier: Vision tokens tier (low/medium/high)
            per_block_top_k: Dictionary mapping block index to top_k value
                            {block_idx: top_k_value}. Block0 is always fixed at 8.
            block_mask: (16,) boolean mask indicating which blocks are active
                        Block0 is always True. Blocks 1-15 can be True/False.
            latency_budget: (B,) latency budget in ms
            max_new_tokens: Maximum number of tokens to generate
            metadata: Optional metadata for accuracy computation
            tokenizer: Optional tokenizer for decoding
        
        Returns:
            results: {
                'output_ids': Generated token IDs,
                'accuracy': Accuracy score (if metadata provided),
                'prefill_latency': Prefill latency in ms (primary metric),
                'decode_latency': Decode latency in ms,
                'total_latency': Total latency in ms,
            }
        """
        # Apply knobs
        # Set per-block top_k
        if per_block_top_k is not None:
            self._set_per_block_top_k(per_block_top_k)
        else:
            # Fallback: set all blocks to top_k=8 (block0 is always 8)
            model = self._get_model()
            transformer = model.model.transformer
            if hasattr(transformer, 'blocks'):
                for block in transformer.blocks:
                    if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
                        block.mlp.top_k = 8
        
        # Apply block mask (store for model forward to check)
        if block_mask is not None:
            self._apply_block_mask(block_mask)
        else:
            # Fallback: all blocks active
            all_active = torch.ones(16, dtype=torch.bool, device=self.device)
            self._apply_block_mask(all_active)
        
        # Create GenerationConfig
        from transformers import GenerationConfig
        
        eos_token_id = None
        pad_token_id = None
        if tokenizer is not None:
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is None:
                eos_token_id = getattr(self.model.config, 'eos_token_id', None)
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = getattr(self.model.config, 'pad_token_id', None)
        else:
            # Fallback to model config if tokenizer not provided
            eos_token_id = getattr(self.model.config, 'eos_token_id', None)
            pad_token_id = getattr(self.model.config, 'pad_token_id', None)
        
        # Derive dataset name (if available) to align max_new_tokens with core_exp configs
        dataset_name = None
        if isinstance(metadata, dict):
            dataset_name = metadata.get('dataset_name', metadata.get('dataset', None))
        max_new_tokens_local = self._get_max_new_tokens(dataset_name) if dataset_name else max_new_tokens

        generation_config_kwargs = {
            "max_new_tokens": max_new_tokens_local,  # Maximum tokens, but will stop at EOS
            "use_cache": True,
            "do_sample": False,
        }
        if eos_token_id is not None:
            generation_config_kwargs["eos_token_id"] = eos_token_id
        if pad_token_id is not None:
            generation_config_kwargs["pad_token_id"] = pad_token_id
        # Note: GenerationConfig will automatically stop at EOS token
        # max_new_tokens is just the upper limit
            generation_config_kwargs["pad_token_id"] = pad_token_id
        
        generation_config = GenerationConfig(**generation_config_kwargs)
        
        # Since we're executing samples one by one (batch_size=1), we can directly
        # measure actual latency using hooks instead of using an estimator
        # Use prefill latency as primary metric for latency budget checking
        latency_results = self._measure_latency_with_hooks(
            batch={
                'input_ids': input_ids,
                'images': images,
                'image_masks': image_masks,
                'image_input_idx': image_input_idx,
            },
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
            latency_budget=latency_budget,  # (B,) latency budget in ms
        )
        
        output_ids = latency_results['output_ids']
        prefill_latency = latency_results.get('T_LLM_prefill', 0.0)
        decode_latency = latency_results.get('T_LLM_decode', 0.0)
        total_latency = latency_results.get('T_total', 0.0)
        
        # Compute accuracy using EXACT same logic as BaseExperiment.compute_accuracy
        # This ensures consistency with core_exp_h100 experiments
        accuracy = torch.tensor(0.0, device=self.device)
        debug_pred_text = None
        debug_answers = []
        if metadata is not None and tokenizer is not None:
            try:
                # Extract generated tokens - same as BaseExperiment
                input_len = input_ids.shape[1]
                if output_ids.shape[1] > input_len:
                    generated_tokens = output_ids[:, input_len:]
                else:
                    generated_tokens = output_ids
                
                # Get metric name FIRST (before processing pred_text)
                dataset_name = None
                if isinstance(metadata, dict):
                    dataset_name = metadata.get('dataset_name', metadata.get('dataset', None))
                
                from experiments.base_experiment import get_metric_for_dataset
                if dataset_name:
                    metric_name = get_metric_for_dataset(dataset_name)
                else:
                    metric_name = "vqa_score"  # Default
                
                # Decode to text - EXACT same as BaseExperiment
                pred_text = tokenizer.decode(generated_tokens[0].cpu().tolist(), skip_special_tokens=True)
                
                # Extract answer from prediction - EXACT same as BaseExperiment (no short-answer normalization)
                if "Answer:" in pred_text:
                    pred_text = pred_text.split("Answer:")[1].strip()
                elif "\n" in pred_text:
                    # Take the last line if multiple lines - EXACT same as BaseExperiment
                    lines = [line.strip() for line in pred_text.split("\n") if line.strip()]
                    pred_text = lines[-1] if lines else pred_text.strip()
                else:
                    pred_text = " ".join(pred_text.strip().split())
                debug_pred_text = pred_text
                
                # Get ground truth answers - EXACT same logic as BaseExperiment.compute_accuracy
                if isinstance(metadata, dict):
                    if "answers" in metadata:
                        answers = metadata["answers"]
                        if isinstance(answers, str):
                            answers = [answers]
                    elif "answer" in metadata:
                        answer = metadata["answer"]
                        answers = [answer] if isinstance(answer, str) else answer
                    else:
                        answers = []
                else:
                    answers = []
                
                # Compute score - EXACT same logic as BaseExperiment (no modifications!)
                if answers and len(answers) > 0:
                    debug_answers = answers if isinstance(answers, list) else [answers]
                    if metric_name == "vqa_score":
                        from molmo.eval.vqa import vqa_score
                        score = vqa_score(answers, pred_text)
                    elif metric_name == "mc":
                        # Multiple choice - EXACT same logic as BaseExperiment
                        from molmo.eval.vqa import select_mc_option
                        if "answer_idx" not in metadata or "options" not in metadata:
                            log.warning(f"Missing answer_idx or options for MC evaluation")
                            score = 0.0
                        else:
                            options = metadata["options"]
                            correct_idx = metadata["answer_idx"]
                            
                            # Extract letter - EXACT same logic as BaseExperiment
                            import re
                            pred_clean = pred_text.strip().upper()
                            letter_match = re.search(r'\b([A-Z])\b', pred_clean)
                            if letter_match:
                                letter = letter_match.group(1)
                                letter_idx = ord(letter) - ord('A')
                                if 0 <= letter_idx < len(options):
                                    predicted_idx = letter_idx
                                else:
                                    predicted_idx = select_mc_option(pred_text, options)
                            else:
                                predicted_idx = select_mc_option(pred_text, options)
                            
                            score = 1.0 if predicted_idx == correct_idx else 0.0
                    elif metric_name == "em":
                        # Exact match - EXACT same logic as BaseExperiment
                        if isinstance(answers, str):
                            answers = [answers]
                        score = 1.0 if pred_text.lower().strip() in [ans.lower().strip() for ans in answers] else 0.0
                    elif metric_name == "ansl":
                        # ANLS - EXACT same logic as BaseExperiment
                        from molmo.eval.vqa import anls_metric
                        if isinstance(answers, str):
                            answers = [answers]
                        score = max(anls_metric(ref, pred_text) for ref in answers) if answers else 0.0
                    elif metric_name == "ansl_em":
                        # ANLS + EM - EXACT same logic as BaseExperiment
                        from molmo.eval.vqa import anls_metric
                        if isinstance(answers, str):
                            answers = [answers]
                        score = max(anls_metric(ref, pred_text) for ref in answers) if answers else 0.0
                    else:
                        # Fallback - same as BaseExperiment
                        from molmo.eval.vqa import vqa_score
                        score = vqa_score(answers, pred_text)
                    
                    accuracy = torch.tensor(score, device=self.device)
                    
                    # DEBUG: Log successful accuracy computation (only occasionally)
                    if random.random() < 0.05:  # Log 5% of successful cases
                        log.info(f"[_execute_model] Accuracy computed: score={score:.4f}, "
                                f"answers={answers[:2] if len(answers) > 2 else answers}, "
                                f"pred={pred_text[:50]}")
                else:
                    # No answers found - log for debugging
                    if random.random() < 0.1:  # Log 10% of cases
                        log.warning(f"[_execute_model] No answers found! metadata_keys={list(metadata.keys()) if isinstance(metadata, dict) else 'N/A'}, "
                                f"pred_text={pred_text[:50] if pred_text else 'empty'}")
                    accuracy = torch.tensor(0.0, device=self.device)
            except Exception as e:
                log.warning(f"[_execute_model] Error computing accuracy: {e}")
                import traceback
                log.debug(traceback.format_exc())
                accuracy = torch.tensor(0.0, device=self.device)
        
        result = {
            'output_ids': output_ids,
            'accuracy': accuracy,
            'prefill_latency': prefill_latency,  # Primary metric
            'decode_latency': decode_latency,
            'total_latency': total_latency,
            'pred_text': debug_pred_text,
            'answers': debug_answers,
        }
        
        return result
    
    def form_groups(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Form groups from batch data.
        Groups are formed by (sample_id, latency_budget) pairs.
        
        Args:
            batch: Batch of data
        
        Returns:
            List of groups
        """
        batch_size = batch['input_ids'].shape[0]
        
        # Group by (sample_id, latency_budget)
        groups_dict = {}
        sample_ids = batch.get('sample_id', [i for i in range(batch_size)])
        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.cpu().tolist()
        budgets = batch['latency_budget'].cpu().numpy()
        
        for i in range(batch_size):
            key = (sample_ids[i] if isinstance(sample_ids, list) else i, float(budgets[i]))
            if key not in groups_dict:
                groups_dict[key] = []
            groups_dict[key].append(i)
        
        # Form groups
        groups = []
        for key, indices in groups_dict.items():
            if len(indices) >= 2:  # Need at least 2 samples for ranking
                group = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        group[k] = v[indices]
                    else:
                        group[k] = [v[i] for i in indices] if isinstance(v, list) else v
                groups.append(group)
        
        return groups
    
    def compute_joint_grpo_loss(
        self,
        knob1_logits: torch.Tensor,
        knob2_logits: torch.Tensor,
        knob3_logits: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        groups: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute joint GRPO loss for both Stage1 and Stage2.
        
        Args:
            knob1_logits: (B, 3) logits for tier
            knob2_logits: (B, 5) logits for top_k
            knob3_logits: (B, 5) logits for num_blocks
            actions: {
                'knob1_idx': (B,),
                'knob2_idx': (B,),
                'knob3_idx': (B,),
            }
            rewards: (B,) reward values
            groups: List of groups
        
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics
        """
        # Compute log probabilities for each knob
        knob1_log_probs = F.log_softmax(knob1_logits, dim=-1)
        knob2_log_probs = F.log_softmax(knob2_logits, dim=-1)
        knob3_log_probs = F.log_softmax(knob3_logits, dim=-1)
        
        knob1_log_prob = knob1_log_probs.gather(1, actions['knob1_idx'].unsqueeze(-1)).squeeze(-1)
        knob2_log_prob = knob2_log_probs.gather(1, actions['knob2_idx'].unsqueeze(-1)).squeeze(-1)
        knob3_log_prob = knob3_log_probs.gather(1, actions['knob3_idx'].unsqueeze(-1)).squeeze(-1)
        
        # Total log probability (sum of all three knobs)
        total_log_probs = knob1_log_prob + knob2_log_prob + knob3_log_prob
        
        # Group relative ranking loss
        total_loss = 0.0
        num_pairs = 0
        
        start_idx = 0
        group_losses = []
        
        for group in groups:
            group_size = group['input_ids'].shape[0]
            group_log_probs = total_log_probs[start_idx:start_idx + group_size]
            group_rewards = rewards[start_idx:start_idx + group_size]
            
            # Sort by reward (descending)
            sorted_indices = torch.argsort(group_rewards, descending=True)
            sorted_log_probs = group_log_probs[sorted_indices]
            sorted_rewards = group_rewards[sorted_indices]
            
            # For each pair (i, j) where i > j in ranking
            group_loss = 0.0
            group_pairs = 0
            
            for i in range(group_size):
                for j in range(i + 1, group_size):
                    log_prob_diff = sorted_log_probs[i] - sorted_log_probs[j]
                    reward_diff = sorted_rewards[i] - sorted_rewards[j]
                    
                    pair_loss = -F.logsigmoid(log_prob_diff * torch.sign(reward_diff))
                    group_loss += pair_loss
                    group_pairs += 1
            
            if group_pairs > 0:
                group_loss = group_loss / group_pairs
                group_losses.append(group_loss.item())
                total_loss += group_loss
                num_pairs += group_pairs
            
            start_idx += group_size
        
        if num_pairs > 0:
            loss = total_loss / len(groups)  # Average over groups
        else:
            # Fallback: standard policy gradient
            advantages = rewards - rewards.mean()
            loss = -(total_log_probs * advantages).mean()
        
        metrics = {
            'loss': loss.item(),
            'num_groups': len(groups),
            'num_pairs': num_pairs,
            'group_loss_mean': np.mean(group_losses) if group_losses else 0.0,
        }
        
        return loss, metrics
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        lang_extractor,
        budget_encoder,
    ) -> Dict[str, float]:
        """
        Run one training step with joint training.
        
        Args:
            batch: Batch of data with input_ids, images, prompts, etc.
            lang_extractor: Language feature extractor
            budget_encoder: Budget encoder
        
        Returns:
            Metrics dictionary
        """
        self.one_stage_controller.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        # Note: When using DataParallel, all tensors should be on the main device (cuda:0)
        # If there's a CUDA error, it may be from a previous operation that left GPU in bad state
        try:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e):
                log.error(f"CUDA error when moving batch to device: {e}")
                log.error("This may indicate a previous CUDA error that left the GPU in a bad state.")
                log.error("The error is likely from a previous operation (e.g., model forward, index_add_).")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA device-side assert error. This is likely due to index out of bounds in a previous operation. Original error: {e}")
            else:
                raise
        
        batch_size = batch['input_ids'].shape[0]
        expanded_batch_size = batch_size * self.group_size  # Total number of configs (B * group_size)
        
        # ========================================================================
        # GRPO Implementation: For each sample, sample group_size configs
        # ========================================================================
        # Instead of sampling once per sample, we sample group_size times per sample
        # This creates batch_size * group_size configs total
        # Then we group by sample_id and perform relative ranking within each group
        
        # Extract features for one-stage controller
        # Need vision_feat, lang_feat, budget_feat
        
        # Extract vision features (global crop from vision_backbone)
        images = batch.get('images', None)
        vision_feats = None
        if images is not None:
            with torch.no_grad():
                model = self._get_model()
                image_features, cls_embed = model.model.vision_backbone(images, batch.get('image_masks'))
                batch_size_vis, num_crops, num_patches, d_model = image_features.shape
                # Use global crop (first crop) or mean pooling
                vision_feats = image_features[:, 0, :, :].mean(dim=1)  # (B, d_model) - use first crop and mean pool patches
        else:
            vision_feats = torch.zeros(batch_size, 768, device=self.device)  # Default vision_dim=768
        
        # Extract language features
        prompts = batch.get('prompts', [''] * batch_size)
        lang_feats = []
        for prompt in prompts:
            lang_feat = lang_extractor.extract(prompt).squeeze(0)  # Automatically uses wte_layer's device
            lang_feats.append(lang_feat)
        lang_feats = torch.stack(lang_feats)  # (B, d_model)
        
        # Extract budget features
        budget_feats = budget_encoder(batch['latency_budget'])  # (B, d_model)
        
        # Repeat features for group_size samples per original sample
        vision_feats_expanded = vision_feats.repeat_interleave(self.group_size, dim=0)  # (B * group_size, 768)
        lang_feats_expanded = lang_feats.repeat_interleave(self.group_size, dim=0)  # (B * group_size, d_model)
        budget_feats_expanded = budget_feats.repeat_interleave(self.group_size, dim=0)  # (B * group_size, d_model)
        
        # One-stage controller: Predict tier, block_logits, block_topk_logits
        controller_output = self.one_stage_controller(
            vision_feat=vision_feats_expanded,  # (B * group_size, 768)
            lang_feat=lang_feats_expanded,      # (B * group_size, d_model)
            budget_feat=budget_feats_expanded,   # (B * group_size, d_model)
        )
        tier_logits = controller_output['tier_logits']  # (B * group_size, 3)
        block_logits = controller_output['block_logits']  # (B * group_size, 15) for blocks 1-15
        block_topk_logits = controller_output['block_topk_logits']  # (B * group_size, 16, 5) for all blocks
        
        # Sample actions (non-deterministic for training) - now we have B * group_size samples
        # 1. Sample tier
        tier_probs = F.softmax(tier_logits, dim=-1)  # (B * group_size, 3)
        tier_actions = torch.multinomial(tier_probs, 1).squeeze(-1)  # (B * group_size,)
        tier_options = ["low", "medium", "high"]
        tiers = [tier_options[idx.item()] for idx in tier_actions]
        
        # 2. Sample block mask from block_logits (blocks 1-15, block0 always on)
        block_masks, block_mask_log_probs = self._sample_block_mask_from_logits(
            block_logits,  # (B * group_size, 15)
            min_active=self.min_active_blocks,
            max_active=self.max_active_blocks,
        )  # block_masks: (B * group_size, 16), log_probs: (B * group_size,)
        
        # 3. Sample per-block top-k from block_topk_logits
        topk_choices = [4, 5, 6, 7, 8]
        per_block_topk_list = []  # List of dicts: [{block_idx: top_k_value}]
        per_block_topk_log_probs_list = []  # List of log probs
        
        for i in range(expanded_batch_size):
            # For each sample, sample top-k for each block
            block_topk_dict = {}
            block_topk_log_prob_sum = 0.0
            
            for block_idx in range(16):
                if block_idx == 0:
                    # Block0: fixed at 8
                    block_topk_dict[0] = 8
                    # Log prob is 1.0 (deterministic), so log_prob = 0.0
                else:
                    # Blocks 1-15: sample from logits
                    block_topk_logits_i = block_topk_logits[i, block_idx, :]  # (5,)
                    block_topk_probs_i = F.softmax(block_topk_logits_i, dim=-1)
                    topk_action = torch.multinomial(block_topk_probs_i, 1).item()
                    block_topk_dict[block_idx] = topk_choices[topk_action]
                    
                    # Compute log prob
                    block_topk_log_probs_i = F.log_softmax(block_topk_logits_i, dim=-1)
                    block_topk_log_prob_sum += block_topk_log_probs_i[topk_action].item()
            
            per_block_topk_list.append(block_topk_dict)
            per_block_topk_log_probs_list.append(torch.tensor(block_topk_log_prob_sum, device=self.device))
        
        per_block_topk_log_probs = torch.stack(per_block_topk_log_probs_list)  # (B * group_size,)
        
        # Process images with predicted tiers
        images = batch.get('images', None)
        image_masks = batch.get('image_masks', None)
        image_input_idx = batch.get('image_input_idx', None)
        
        # Expand batch data for group_size configs per sample
        input_ids_expanded = batch['input_ids'].repeat_interleave(self.group_size, dim=0)  # (B * group_size, seq_len)
        if images is not None:
            images_expanded = images.repeat_interleave(self.group_size, dim=0)  # (B * group_size, ...)
        else:
            images_expanded = None
        if image_masks is not None:
            image_masks_expanded = image_masks.repeat_interleave(self.group_size, dim=0)
        else:
            image_masks_expanded = None
        if image_input_idx is not None:
            image_input_idx_expanded = image_input_idx.repeat_interleave(self.group_size, dim=0)
        else:
            image_input_idx_expanded = None
        
        # Expand latency_budget for group_size configs per sample
        latency_budget_expanded = batch['latency_budget'].repeat_interleave(self.group_size, dim=0)  # (B * group_size,)
        
        # Expand metadata
        metadatas = batch.get('metadata', [])
        if not isinstance(metadatas, list):
            metadatas = [metadatas] * batch_size if metadatas else [None] * batch_size
        metadatas_expanded = []
        for meta in metadatas:
            for _ in range(self.group_size):
                metadatas_expanded.append(meta)
        
        # Get tokenizer for accuracy computation
        tokenizer = lang_extractor.tokenizer if hasattr(lang_extractor, 'tokenizer') else None
        
        # Execute model and compute rewards for all expanded configs
        rewards = []
        accuracies = []
        latencies = []
        
        log.info(f"[train_step] Executing model.generate() for {expanded_batch_size} configs...")
        
        for i in range(expanded_batch_size):
            if expanded_batch_size > 10 and i % max(1, expanded_batch_size // 10) == 0:
                log.info(f"[train_step] Executing config {i+1}/{expanded_batch_size} (sample {i // self.group_size + 1}, config {i % self.group_size + 1})...")
            
            # Get original sample index
            orig_sample_idx = i // self.group_size
            metadata = metadatas_expanded[i] if i < len(metadatas_expanded) else None
            
            # DEBUG: Log metadata for first few configs to diagnose accuracy=0
            if i < 3:
                log.info(f"[train_step] Config {i}: metadata={type(metadata)}, "
                        f"metadata_keys={list(metadata.keys()) if isinstance(metadata, dict) else 'N/A'}, "
                        f"has_answers={'answers' in metadata if isinstance(metadata, dict) else 'N/A'}")
            
            result = self._execute_model(
                input_ids=input_ids_expanded[i:i+1],
                images=images_expanded[i:i+1] if images_expanded is not None else None,
                image_masks=image_masks_expanded[i:i+1] if image_masks_expanded is not None else None,
                image_input_idx=image_input_idx_expanded[i:i+1] if image_input_idx_expanded is not None else None,
                tier=tiers[i],
                per_block_top_k=per_block_topk_list[i],  # Dict: {block_idx: top_k_value}
                block_mask=block_masks[i],  # (16,) boolean mask
                latency_budget=latency_budget_expanded[i:i+1],
                max_new_tokens=64,
                metadata=metadata,
                tokenizer=tokenizer,
            )
            
            # Extract accuracy - handle both tensor and float
            accuracy_val = result['accuracy']
            if isinstance(accuracy_val, torch.Tensor):
                accuracy = accuracy_val.item()
            else:
                accuracy = float(accuracy_val)
            accuracies.append(accuracy)

            # Debug: log per-group predictions and ground truth for first few samples
            if orig_sample_idx < 2:  # only log first two samples to avoid spam
                pred_text_dbg = result.get('pred_text', '')
                answers_dbg = result.get('answers', [])
                log.info(f"[train_step][group {orig_sample_idx}] cfg {i % self.group_size + 1}/{self.group_size}: "
                         f"pred='{str(pred_text_dbg)[:120]}', answers={answers_dbg[:3] if isinstance(answers_dbg, list) else answers_dbg}")
            
            # Use prefill latency as primary metric (as user requested)
            prefill_latency = result.get('prefill_latency', 0.0)
            latency = prefill_latency  # Primary metric
            latencies.append(latency)
            
            # Compute reward
            tier_to_max_crops = {'low': 3, 'medium': 6, 'high': 12}
            num_active_blocks = block_masks[i].sum().item()  # Count active blocks
            # Average top-k for active blocks (excluding block0 which is fixed at 8)
            active_topks = [per_block_topk_list[i][idx] for idx in range(1, 16) if block_masks[i][idx].item()]
            avg_topk = sum(active_topks) / len(active_topks) if active_topks else 8.0
            
            config = {
                'max_crops': tier_to_max_crops[tiers[i]],
                'top_k': int(avg_topk),
                'num_active_blocks': num_active_blocks,
            }
            
            reward = self.reward_fn(
                accuracy=torch.tensor(accuracy, device=self.device),
                latency=torch.tensor(latency, device=self.device),
                latency_budget=latency_budget_expanded[i:i+1],  # Use expanded budget
                config={
                    'max_crops': torch.tensor([config['max_crops']], device=self.device),
                    'top_k': torch.tensor([config['top_k']], device=self.device),
                    'num_active_blocks': torch.tensor([config['num_active_blocks']], device=self.device),
                },
            )
            # Extract reward - handle both tensor and float
            if isinstance(reward, torch.Tensor):
                rewards.append(reward.item())
            else:
                rewards.append(float(reward))
        
        # Convert to tensors
        rewards = torch.tensor(rewards, device=self.device)  # (B * group_size,)
        
        # ========================================================================
        # Compute log probabilities for GRPO loss (one-stage)
        # ========================================================================
        # One-stage controller outputs:
        # - tier_logits: (B * group_size, 3)
        # - block_logits: (B * group_size, 15) - already sampled to get block_mask_log_probs
        # - block_topk_logits: (B * group_size, 16, 5) - already sampled to get per_block_topk_log_probs
        
        # 1. Tier log prob
        tier_log_probs = F.log_softmax(tier_logits, dim=-1)  # (B * group_size, 3)
        tier_log_prob = tier_log_probs.gather(1, tier_actions.unsqueeze(-1)).squeeze(-1)  # (B * group_size,)
        
        # 2. Block mask log prob (already computed in _sample_block_mask_from_logits)
        block_mask_log_prob = block_mask_log_probs  # (B * group_size,)
        
        # 3. Per-block top-k log prob (already computed above)
        per_block_topk_log_prob = per_block_topk_log_probs  # (B * group_size,)
        
        # Total log probability (for all B * group_size configs)
        total_log_probs = tier_log_prob + block_mask_log_prob + per_block_topk_log_prob  # (B * group_size,)
        
        # ========================================================================
        # GRPO Grouping: Group by (sample_id, latency_budget)
        # ========================================================================
        # According to GRPO theory: groups should be formed by (sample_id, budget)
        # because for the same sample, different budgets may require different optimal strategies
        # Each (sample_id, budget) combination has group_size configs
        groups = []
        groups_dict = {}
        
        # Get sample_ids and budgets for expanded configs
        sample_ids = batch.get('sample_id', list(range(batch_size)))
        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.cpu().tolist()
        elif not isinstance(sample_ids, list):
            sample_ids = list(range(batch_size))
        
        # Expand sample_ids and get budgets
        expanded_sample_ids = []
        expanded_budgets = []
        for orig_idx in range(batch_size):
            sample_id = sample_ids[orig_idx] if orig_idx < len(sample_ids) else orig_idx
            budget = float(latency_budget_expanded[orig_idx * self.group_size].item())
            for _ in range(self.group_size):
                expanded_sample_ids.append(sample_id)
                expanded_budgets.append(budget)
        
        # Group by (sample_id, budget)
        for i in range(expanded_batch_size):
            key = (expanded_sample_ids[i], expanded_budgets[i])
            if key not in groups_dict:
                groups_dict[key] = []
            groups_dict[key].append(i)
        
        # Create groups (only if group_size >= 2)
        for key, indices in groups_dict.items():
            if len(indices) >= 2:  # Need at least 2 configs for GRPO
                group = {
                    'sample_id': key[0],
                    'budget': key[1],
                    'indices': indices,
                    'log_probs': total_log_probs[indices],  # (len(indices),)
                    'rewards': rewards[indices],  # (len(indices),)
                }
                groups.append(group)
        
        if len(groups) == 0:
            log.warning(f"No groups formed (need at least 2 configs per (sample_id, budget) pair). "
                       f"Total configs: {expanded_batch_size}, unique (sample_id, budget) pairs: {len(groups_dict)}")  # (B * group_size,)
        
        # ========================================================================
        # GRPO Loss: Standard GRPO with group-normalized advantages
        # ========================================================================
        # According to DeepSeekMath GRPO paper:
        # 1. Compute group-normalized advantages: adv = (r - mean(r)) / (std(r) + eps)
        # 2. Use PPO clipped surrogate objective (simplified: no old policy, just policy gradient)
        # 3. For simplicity, we use vanilla policy gradient: -log_prob * adv
        #    (PPO clip requires old_log_probs which we don't store, but this is acceptable for initial implementation)
        
        # Initialize total_loss as a tensor that will participate in gradient computation
        total_loss = torch.zeros_like(total_log_probs[0]) if len(total_log_probs) > 0 else torch.tensor(0.0, device=total_log_probs.device, dtype=total_log_probs.dtype, requires_grad=True)
        group_losses = []
        
        for group in groups:
            group_indices = group['indices']
            group_log_probs = total_log_probs[group_indices]  # (group_size_actual,)
            group_rewards = group['rewards']  # (group_size_actual,)
            group_size_actual = len(group_indices)
            
            if group_size_actual < 2:
                continue  # Skip groups with < 2 configs
            
            # GRPO: Compute group-normalized advantages
            # adv = (r - mean(r)) / (std(r) + eps)
            # This uses the group mean as baseline, eliminating the need for a value network
            group_rewards_mean = group_rewards.mean()
            group_rewards_std = group_rewards.std()
            eps = 1e-8
            
            if group_rewards_std < eps:
                # If all rewards are the same, advantages are 0, skip this group
                log.debug(f"Group rewards have zero std, skipping (sample_id={group['sample_id']}, budget={group['budget']:.2f})")
                continue
            
            advantages = (group_rewards - group_rewards_mean) / (group_rewards_std + eps)  # (group_size_actual,)
            
            # GRPO Policy Gradient Loss: -log_prob * advantage
            # This encourages configs with positive advantage (above group mean) to have higher probability
            # and discourages configs with negative advantage (below group mean)
            group_loss = -(group_log_probs * advantages).mean()  # Scalar
            
            total_loss = total_loss + group_loss
            group_losses.append(group_loss.item())
        
        # Compute loss: ensure it's always a tensor with gradients
        if len(groups) > 0 and group_losses:
            # total_loss should have gradients from the accumulation above
            # Average over all groups
            loss = total_loss / len(groups)
            # Ensure loss requires grad (it should already, but double-check)
            if not loss.requires_grad:
                # If somehow loss doesn't have grad, use fallback
                log.warning("Loss doesn't require grad, using fallback policy gradient")
                # Fallback: global advantage (not group-normalized, but still works)
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                loss = -(total_log_probs * advantages).mean()
        else:
            # Fallback: standard policy gradient with global normalization
            # This should not happen with correct GRPO implementation, but keep as safety
            log.warning(f"No groups formed or no valid groups. Using global normalized policy gradient. "
                       f"groups={len(groups)}, group_losses={len(group_losses)}")
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            loss = -(total_log_probs * advantages).mean()
        
        # Final check: ensure loss has gradients
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=total_log_probs.device, dtype=total_log_probs.dtype, requires_grad=True)
        elif not loss.requires_grad:
            # If loss is a tensor but doesn't require grad, we need to recompute it
            # This shouldn't happen, but as a safety measure
            advantages = rewards - rewards.mean()
            loss = -(total_log_probs * advantages).mean()
        
        loss_metrics = {
            'loss': loss.item() if hasattr(loss, 'item') else float(loss),
            'num_groups': len(groups),
            'group_loss_mean': np.mean(group_losses) if group_losses else 0.0,
            'num_configs': len(rewards),  # Total configs (B * group_size)
            'num_valid_groups': len([g for g in groups if len(g['indices']) >= 2]),
        }
        
        # Debug logging for loss
        if loss.item() == 0.0 or abs(loss.item()) < 1e-6:
            log.debug(f"Loss is very small ({loss.item():.6f}): num_groups={len(groups)}, "
                     f"rewards_range=[{rewards.min().item():.4f}, {rewards.max().item():.4f}], "
                     f"rewards_mean={rewards.mean().item():.4f}, rewards_std={rewards.std().item():.4f}")
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.one_stage_controller.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()
        
        # Metrics
        accuracy_mean = np.mean(accuracies) if accuracies else 0.0
        
        # Debug logging for accuracy
        if accuracy_mean == 0.0 and accuracies:
            log.debug(f"All accuracies are 0.0: {len(accuracies)} samples, "
                     f"accuracy_values={accuracies[:5]}... (showing first 5)")
        elif not accuracies:
            log.debug(f"No accuracies computed (accuracies list is empty)")
        
        # Calculate budget violation rate
        # Compare latencies with corresponding latency budgets
        if latencies and len(latencies) == len(latency_budget_expanded):
            budget_violations = []
            for i, latency in enumerate(latencies):
                budget = latency_budget_expanded[i].item() if isinstance(latency_budget_expanded[i], torch.Tensor) else float(latency_budget_expanded[i])
                budget_violations.append(1.0 if latency > budget else 0.0)
            budget_violation_rate = np.mean(budget_violations) if budget_violations else 0.0
        else:
            budget_violation_rate = 0.0
        
        metrics = {
            **loss_metrics,
            'loss': loss_metrics.get('loss', 0.0),
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'reward_min': rewards.min().item(),
            'reward_max': rewards.max().item(),
            'accuracy_mean': accuracy_mean,
            'accuracy_std': np.std(accuracies) if accuracies else 0.0,
            'latency_mean': np.mean(latencies) if latencies else 0.0,
            'latency_std': np.std(latencies) if latencies else 0.0,
            'budget_violation_rate': budget_violation_rate,
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        lang_extractor,
        budget_encoder,
    ) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation dataloader
            lang_extractor: Language feature extractor
            budget_encoder: Budget encoder
        
        Returns:
            Metrics dictionary
        """
        self.one_stage_controller.eval()
        
        all_rewards = []
        all_accuracies = []
        all_latencies = []
        all_budget_violations = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                batch_size = batch['input_ids'].shape[0]
                
                # Extract features for one-stage controller
                # Vision features
                images = batch.get('images', None)
                if images is not None:
                    with torch.no_grad():
                        model = self._get_model()
                        image_features, cls_embed = model.model.vision_backbone(images, batch.get('image_masks'))
                        batch_size_vis, num_crops, num_patches, d_model = image_features.shape
                        # Use global crop (first crop) and mean pool patches
                        vision_feats = image_features[:, 0, :, :].mean(dim=1)  # (B, d_model)
                else:
                    vision_feats = torch.zeros(batch_size, 768, device=self.device)  # Default vision_dim=768
                
                # Language features
                prompts = batch.get('prompts', [''] * batch_size)
                lang_feats = []
                for prompt in prompts:
                    lang_feat = lang_extractor.extract(prompt).squeeze(0)
                    lang_feats.append(lang_feat)
                lang_feats = torch.stack(lang_feats)  # (B, d_model)
                
                # Budget features
                budget_feats = budget_encoder(batch['latency_budget'])  # (B, d_model)
                
                # One-stage controller: Predict tier, block_logits, block_topk_logits (deterministic)
                controller_output = self.one_stage_controller(
                    vision_feat=vision_feats,  # (B, 768)
                    lang_feat=lang_feats,      # (B, d_model)
                    budget_feat=budget_feats,  # (B, d_model)
                )
                tier_logits = controller_output['tier_logits']  # (B, 3)
                block_logits = controller_output['block_logits']  # (B, 15) for blocks 1-15
                block_topk_logits = controller_output['block_topk_logits']  # (B, 16, 5) for all blocks
                
                # Deterministic: argmax for tier
                tier_actions = tier_logits.argmax(dim=-1)  # (B,)
                tier_options = ["low", "medium", "high"]
                tiers = [tier_options[idx.item()] for idx in tier_actions]
                
                # Deterministic: sample block mask (select top blocks based on logits)
                # For validation, we'll use a deterministic selection: select top-N blocks
                block_masks = []
                for i in range(batch_size):
                    # Select number of blocks to activate (between min_active and max_active)
                    # For simplicity, use a fixed number (e.g., 14) or sample from logits
                    num_to_select = torch.randint(
                        self.min_active_blocks - 1, 
                        self.max_active_blocks - 1 + 1, 
                        (1,)
                    ).item()  # Number from blocks 1-15 (excluding block0)
                    
                    # Select top-N blocks based on logits
                    sample_logits = block_logits[i]  # (15,)
                    _, top_indices = torch.topk(sample_logits, num_to_select, dim=-1)  # (num_to_select,)
                    
                    # Create mask: block0 always True, selected blocks from 1-15 also True
                    mask = torch.zeros(16, dtype=torch.bool, device=self.device)
                    mask[0] = True  # Block0 always on
                    mask[top_indices + 1] = True  # Blocks 1-15 (indices 0-14 in logits -> block indices 1-15)
                    block_masks.append(mask)
                
                block_masks = torch.stack(block_masks)  # (B, 16)
                
                # Deterministic: argmax for per-block top-k
                topk_choices = [4, 5, 6, 7, 8]
                per_block_topk_list = []
                for i in range(batch_size):
                    block_topk_dict = {}
                    for block_idx in range(16):
                        if block_idx == 0:
                            # Block0: fixed at 8
                            block_topk_dict[0] = 8
                        else:
                            # Blocks 1-15: argmax
                            block_topk_logits_i = block_topk_logits[i, block_idx, :]  # (5,)
                            topk_action = block_topk_logits_i.argmax().item()
                            block_topk_dict[block_idx] = topk_choices[topk_action]
                    per_block_topk_list.append(block_topk_dict)
                
                # Get tokenizer
                tokenizer = lang_extractor.tokenizer if hasattr(lang_extractor, 'tokenizer') else None
                
                # Get metadata
                metadatas = batch.get('metadata', [])
                if not isinstance(metadatas, list):
                    metadatas = [metadatas] * batch_size if metadatas else [None] * batch_size
                
                # Execute model and compute metrics
                for i in range(batch_size):
                    metadata = metadatas[i] if i < len(metadatas) else None
                    
                    result = self._execute_model(
                        input_ids=batch['input_ids'][i:i+1],
                        images=images[i:i+1] if images is not None else None,
                        image_masks=batch.get('image_masks', [None] * batch_size)[i:i+1] if batch.get('image_masks') is not None else None,
                        image_input_idx=batch.get('image_input_idx', [None] * batch_size)[i:i+1] if batch.get('image_input_idx') is not None else None,
                        tier=tiers[i],
                        per_block_top_k=per_block_topk_list[i],  # Dict: {block_idx: top_k_value}
                        block_mask=block_masks[i],  # (16,) boolean mask
                        latency_budget=batch['latency_budget'][i:i+1],
                        max_new_tokens=64,
                        metadata=metadata,
                        tokenizer=tokenizer,
                    )
                    
                    accuracy = result['accuracy'].item()
                    all_accuracies.append(accuracy)
                    
                    # Use prefill latency as primary metric
                    prefill_latency = result.get('prefill_latency', 0.0)
                    latency = prefill_latency  # Primary metric
                    all_latencies.append(latency)
                    
                    # Compute reward
                    tier_to_max_crops = {'low': 3, 'medium': 6, 'high': 12}
                    num_active_blocks = block_masks[i].sum().item()
                    # Average top-k for active blocks (excluding block0 which is fixed at 8)
                    active_topks = [per_block_topk_list[i][idx] for idx in range(1, 16) if block_masks[i][idx].item()]
                    avg_topk = sum(active_topks) / len(active_topks) if active_topks else 8.0
                    
                    config = {
                        'max_crops': tier_to_max_crops[tiers[i]],
                        'top_k': int(avg_topk),
                        'num_active_blocks': num_active_blocks,
                    }
                    
                    reward = self.reward_fn(
                        accuracy=torch.tensor(accuracy, device=self.device),
                        latency=torch.tensor(latency, device=self.device),
                        latency_budget=batch['latency_budget'][i:i+1],
                        config={
                            'max_crops': torch.tensor([config['max_crops']], device=self.device),
                            'top_k': torch.tensor([config['top_k']], device=self.device),
                            'num_active_blocks': torch.tensor([config['num_active_blocks']], device=self.device),
                        },
                    )
                    all_rewards.append(reward.item())
                    all_budget_violations.append(1.0 if latency > batch['latency_budget'][i].item() else 0.0)
        
        metrics = {
            'reward_mean': np.mean(all_rewards) if all_rewards else 0.0,
            'accuracy_mean': np.mean(all_accuracies) if all_accuracies else 0.0,
            'latency_mean': np.mean(all_latencies) if all_latencies else 0.0,
            'budget_violation_rate': np.mean(all_budget_violations) if all_budget_violations else 0.0,
        }
        
        return metrics
    
    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        is_best: bool = False,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Save checkpoint."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'one_stage_controller_state_dict': self.one_stage_controller.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics or {},
        }
        
        if is_best:
            torch.save(checkpoint, save_path / 'best_model.pt')
        else:
            torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch}.pt')
        
        log.info(f"Saved checkpoint to {save_path}")
