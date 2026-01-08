"""
Latency Estimator: Predict latency from knob configuration.
Lightweight model trained on core experiment data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

log = logging.getLogger(__name__)


class LatencyEstimator(nn.Module):
    """
    Lightweight latency estimator.
    
    Predicts stage-wise latencies from knob configuration:
    - T_vision_total: Vision backbone latency (ViT + Projector)
    - T_LLM_prefill: LLM prefill latency
    - T_LLM_decode_per_token: LLM decode latency per token
    - T_total: Total latency
    
    Input features:
    - vision_tokens: Number of vision tokens
    - text_tokens: Number of text tokens
    - output_tokens: Expected number of output tokens (for decode latency)
    - tier_idx: Tier index (0=low, 1=medium, 2=high)
    - top_k: MoE top-K value
    - num_active_blocks: Number of active transformer blocks
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        use_output_tokens: bool = True,
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of MLP layers
            use_output_tokens: If True, include output_tokens in features
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_output_tokens = use_output_tokens
        
        input_dim = 6 if use_output_tokens else 5
        
        # Feature encoder
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        
        # Stage-wise prediction heads
        self.vision_head = nn.Linear(hidden_dim, 1)  # Predicts T_vision_total
        self.prefill_head = nn.Linear(hidden_dim, 1)
        self.decode_head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        tier_idx: torch.Tensor,
        top_k: torch.Tensor,
        num_active_blocks: torch.Tensor,
        output_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict latencies.
        
        Args:
            vision_tokens: (B,) number of vision tokens
            text_tokens: (B,) number of text tokens
            tier_idx: (B,) tier index (0=low, 1=medium, 2=high)
            top_k: (B,) MoE top-K value
            num_active_blocks: (B,) number of active blocks
            output_tokens: (B,) expected output tokens (optional)
        
        Returns:
            latencies: {
                'T_vision_total': (B,),
                'T_LLM_prefill': (B,),
                'T_LLM_decode_per_token': (B,),
                'T_LLM_decode': (B,),  # Total decode latency
                'T_total': (B,),
            }
        """
        # Build feature vector
        if self.use_output_tokens and output_tokens is not None:
            features = torch.stack([
                vision_tokens.float(),
                text_tokens.float(),
                output_tokens.float(),
                tier_idx.float(),
                top_k.float(),
                num_active_blocks.float(),
            ], dim=-1)  # (B, 6)
        else:
            features = torch.stack([
                vision_tokens.float(),
                text_tokens.float(),
                tier_idx.float(),
                top_k.float(),
                num_active_blocks.float(),
            ], dim=-1)  # (B, 5)
            output_tokens = torch.zeros_like(vision_tokens)  # Dummy for decode calculation
        
        # Encode features
        encoded = self.encoder(features)  # (B, hidden_dim)
        
        # Predict stage-wise latencies
        T_vision_total = F.relu(self.vision_head(encoded)).squeeze(-1)  # (B,)
        T_prefill = F.relu(self.prefill_head(encoded)).squeeze(-1)  # (B,)
        T_decode_per_token = F.relu(self.decode_head(encoded)).squeeze(-1)  # (B,)
        
        # Total decode latency
        T_decode = T_decode_per_token * output_tokens.float()  # (B,)
        
        # Total latency
        T_total = T_vision_total + T_prefill + T_decode
        
        return {
            'T_vision_total': T_vision_total,
            'T_LLM_prefill': T_prefill,
            'T_LLM_decode_per_token': T_decode_per_token,
            'T_LLM_decode': T_decode,
            'T_total': T_total,
        }
    
    def predict_from_config(
        self,
        config: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method to predict from config dictionary.
        
        Args:
            config: {
                'vision_tokens': (B,),
                'text_tokens': (B,),
                'tier': (B,) list of strings or (B,) tier indices
                'top_k': (B,),
                'num_active_blocks': (B,),
                'output_tokens': (B,) optional
            }
        
        Returns:
            latencies: Same as forward()
        """
        # Convert tier to index if needed
        tier = config.get('tier')
        if tier is not None:
            if isinstance(tier, list) or (isinstance(tier, torch.Tensor) and tier.dtype == torch.long):
                # Already indices or strings
                if isinstance(tier, list) and isinstance(tier[0], str):
                    tier_map = {'low': 0, 'medium': 1, 'high': 2}
                    tier_idx = torch.tensor([tier_map[t] for t in tier], device=next(self.parameters()).device)
                else:
                    tier_idx = tier if isinstance(tier, torch.Tensor) else torch.tensor(tier)
            else:
                tier_idx = tier
        else:
            tier_idx = torch.zeros(config['vision_tokens'].shape[0], device=next(self.parameters()).device)
        
        return self.forward(
            vision_tokens=config['vision_tokens'],
            text_tokens=config['text_tokens'],
            tier_idx=tier_idx,
            top_k=config['top_k'],
            num_active_blocks=config['num_active_blocks'],
            output_tokens=config.get('output_tokens'),
        )


class LatencyEstimatorTrainer:
    """
    Trainer for latency estimator.
    """
    
    def __init__(
        self,
        model: LatencyEstimator,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.criterion = nn.MSELoss()
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Training step.
        
        Args:
            batch: {
                'vision_tokens': (B,),
                'text_tokens': (B,),
                'tier_idx': (B,),
                'top_k': (B,),
                'num_active_blocks': (B,),
                'output_tokens': (B,),
                'T_vision_total': (B,),
                'T_LLM_prefill': (B,),
                'T_LLM_decode': (B,),
                'T_total': (B,),
            }
        
        Returns:
            metrics: Loss and error metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        batch_device = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Forward
        pred = self.model(
            vision_tokens=batch_device['vision_tokens'],
            text_tokens=batch_device['text_tokens'],
            tier_idx=batch_device['tier_idx'],
            top_k=batch_device['top_k'],
            num_active_blocks=batch_device['num_active_blocks'],
            output_tokens=batch_device.get('output_tokens'),
        )
        
        # Compute losses for each stage
        T_vision_total_target = batch_device['T_vision_total']
        loss_vision = self.criterion(pred['T_vision_total'], T_vision_total_target)
        loss_prefill = self.criterion(pred['T_LLM_prefill'], batch_device['T_LLM_prefill'])
        loss_decode = self.criterion(pred['T_LLM_decode'], batch_device['T_LLM_decode'])
        loss_total = self.criterion(pred['T_total'], batch_device['T_total'])
        
        # Combined loss
        loss = loss_vision + loss_prefill + loss_decode + loss_total
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        # Compute errors
        mae_vision = F.l1_loss(pred['T_vision_total'], T_vision_total_target).item()
        mae_prefill = F.l1_loss(pred['T_LLM_prefill'], batch_device['T_LLM_prefill']).item()
        mae_total = F.l1_loss(pred['T_total'], batch_device['T_total']).item()
        
        # Relative errors
        rel_error_total = (mae_total / (batch_device['T_total'].mean() + 1e-6)).item()
        
        return {
            'loss': loss.item(),
            'loss_vision': loss_vision.item(),
            'loss_prefill': loss_prefill.item(),
            'loss_total': loss_total.item(),
            'mae_vision': mae_vision,
            'mae_prefill': mae_prefill,
            'mae_total': mae_total,
            'rel_error_total': rel_error_total,
        }
    
    def validate(
        self,
        val_loader,
    ) -> Dict[str, float]:
        """
        Validation.
        """
        self.model.eval()
        
        all_metrics = {
            'loss': [],
            'mae_total': [],
            'rel_error_total': [],
        }
        
        with torch.no_grad():
            for batch in val_loader:
                metrics = self.train_step(batch)  # Reuse train_step for validation
                for key in all_metrics:
                    if key in metrics:
                        all_metrics[key].append(metrics[key])
        
        return {k: sum(v) / len(v) for k, v in all_metrics.items() if v}

