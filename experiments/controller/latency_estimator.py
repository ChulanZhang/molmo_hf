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
    Lightweight latency estimator with positioned decode per-token latency.
    
    Predicts two key latencies from knob configuration:
    - T_prefill_total: Total prefill latency (vision encoder + projector + LLM prefill) [PRIMARY]
    - T_decode_per_token: Decode latency per output token at a given position [SECONDARY]
    
    Design Philosophy:
    - **Prefill latency is PRIMARY**: It's deterministic once configuration is fixed, and directly
      used for latency budget checking. This is the main metric for configuration selection.
    - **Positioned decode per-token latency**: Decode latency per token increases with position
      due to KV cache growth. First tokens are faster (~25ms/token), later tokens are slower (~45ms/token).
      This models the progressive slowdown as KV cache size increases.
    
    Usage:
    1. **Primary use**: Check if configuration satisfies budget using prefill latency
       - T_prefill_total <= latency_budget * safety_factor (e.g., 0.8, leaving 20% for decode)
    2. **Secondary use**: Estimate decode latency for a given token position
       - T_decode_per_token(pos) = model(config, pos)
       - For total decode latency estimation, integrate over expected output positions
       - Or use average: T_decode_avg = model(config, avg_position)
    
    Input features (6 dimensions):
    - vision_tokens: Number of vision tokens
    - text_tokens: Number of text tokens
    - tier_idx: Tier index (0=low, 1=medium, 2=high)
    - top_k: MoE top-K value
    - num_active_blocks: Number of active transformer blocks
    - token_position: Position of the token in the output sequence (1-indexed)
    
    Note: token_position is used during training (from output_tokens) and can be estimated
    during inference (e.g., using expected average position or sampling multiple positions).
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of MLP layers
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input: 5 config features + 1 position feature
        config_dim = 5  # vision_tokens, text_tokens, tier_idx, top_k, num_active_blocks
        position_dim = 1  # token_position
        
        # Config feature encoder (shared for both prefill and decode)
        config_layers = []
        config_layers.append(nn.Linear(config_dim, hidden_dim))
        config_layers.append(nn.LayerNorm(hidden_dim))
        config_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            config_layers.append(nn.Linear(hidden_dim, hidden_dim))
            config_layers.append(nn.LayerNorm(hidden_dim))
            config_layers.append(nn.ReLU())
        
        self.config_encoder = nn.Sequential(*config_layers)
        
        # Position encoder for decode latency (simple MLP)
        # Position affects decode latency due to KV cache growth
        position_layers = []
        position_layers.append(nn.Linear(position_dim, hidden_dim // 4))
        position_layers.append(nn.ReLU())
        position_layers.append(nn.Linear(hidden_dim // 4, hidden_dim // 4))
        position_layers.append(nn.ReLU())
        self.position_encoder = nn.Sequential(*position_layers)
        
        # Prefill head (only uses config features)
        self.prefill_head = nn.Linear(hidden_dim, 1)  # Predicts T_prefill_total
        
        # Decode head (uses config + position features)
        # Concatenate config encoding and position encoding
        decode_input_dim = hidden_dim + hidden_dim // 4
        self.decode_head = nn.Sequential(
            nn.Linear(decode_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )  # Predicts T_decode_per_token at given position
    
    def forward(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        tier_idx: torch.Tensor,
        top_k: torch.Tensor,
        num_active_blocks: torch.Tensor,
        token_position: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict latencies with positioned decode per-token latency.
        
        Args:
            vision_tokens: (B,) number of vision tokens
            text_tokens: (B,) number of text tokens
            tier_idx: (B,) tier index (0=low, 1=medium, 2=high)
            top_k: (B,) MoE top-K value
            num_active_blocks: (B,) number of active blocks
            token_position: (B,) position of token in output sequence (1-indexed)
        
        Returns:
            latencies: {
                'T_prefill_total': (B,),  # Total prefill latency (vision + projector + LLM prefill)
                'T_decode_per_token': (B,),  # Decode latency per token at given position
            }
        """
        # Build config feature vector
        config_features = torch.stack([
            vision_tokens.float(),
            text_tokens.float(),
            tier_idx.float(),
            top_k.float(),
            num_active_blocks.float(),
        ], dim=-1) # (B, 5)
        
        # Encode config features
        config_encoded = self.config_encoder(config_features) # (B, hidden_dim)
        
        # Prefill latency (only depends on config)
        T_prefill_total = F.relu(self.prefill_head(config_encoded)).squeeze(-1) # (B,)
        
        # Decode latency (depends on config + position)
        # Normalize position to reasonable range (log scale helps with growth)
        # Use log(position + 1) to handle position=1 and allow smooth growth
        position_normalized = torch.log(token_position.float() + 1.0) # (B,)
        position_features = position_normalized.unsqueeze(-1) # (B, 1)
        position_encoded = self.position_encoder(position_features) # (B, hidden_dim // 4)
        
        # Concatenate config and position encodings
        decode_features = torch.cat([config_encoded, position_encoded], dim=-1) # (B, hidden_dim + hidden_dim // 4)
        T_decode_per_token = F.relu(self.decode_head(decode_features)).squeeze(-1) # (B,)
        
        return {
            'T_prefill_total': T_prefill_total,
            'T_decode_per_token': T_decode_per_token,
        }
    
    def predict_decode_at_positions(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        tier_idx: torch.Tensor,
        top_k: torch.Tensor,
        num_active_blocks: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict decode per-token latency at multiple positions.
        
        Useful for estimating total decode latency by integrating over positions.
        
        Args:
            vision_tokens: (B,) number of vision tokens
            text_tokens: (B,) number of text tokens
            tier_idx: (B,) tier index
            top_k: (B,) MoE top-K value
            num_active_blocks: (B,) number of active blocks
            positions: (P,) positions to predict (e.g., [1, 2, 3, ..., N])
        
        Returns:
            decode_latencies: (B, P) decode per-token latency at each position
        """
        B = vision_tokens.shape[0]
        P = positions.shape[0]
        
        # Expand config features for all positions
        config_features = torch.stack([
            vision_tokens.float(),
            text_tokens.float(),
            tier_idx.float(),
            top_k.float(),
            num_active_blocks.float(),
        ], dim=-1) # (B, 5)
        
        config_encoded = self.config_encoder(config_features) # (B, hidden_dim)
        
        # Expand positions for all batches
        positions_expanded = positions.unsqueeze(0).expand(B, -1) # (B, P)
        position_normalized = torch.log(positions_expanded.float() + 1.0) # (B, P)
        position_features = position_normalized.unsqueeze(-1) # (B, P, 1)
        
        # Encode positions
        position_encoded = self.position_encoder(position_features) # (B, P, hidden_dim // 4)
        
        # Expand config encoding
        config_encoded_expanded = config_encoded.unsqueeze(1).expand(-1, P, -1) # (B, P, hidden_dim)
        
        # Concatenate
        decode_features = torch.cat([config_encoded_expanded, position_encoded], dim=-1) # (B, P, hidden_dim + hidden_dim // 4)
        
        # Predict
        T_decode_per_token = F.relu(self.decode_head(decode_features)).squeeze(-1) # (B, P)
        
        return T_decode_per_token
    
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
            }
        
        Returns:
            latencies: {
                'T_prefill_total': (B,),
                'T_decode_per_token': (B,),
            }
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
        
        # Get token_position from config, default to 1 if not provided
        token_position = config.get('token_position', torch.ones_like(config['vision_tokens']))
        
        return self.forward(
            vision_tokens=config['vision_tokens'],
            text_tokens=config['text_tokens'],
            tier_idx=tier_idx,
            top_k=config['top_k'],
            num_active_blocks=config['num_active_blocks'],
            token_position=token_position,
        )
    
    def check_budget_feasibility(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        tier_idx: torch.Tensor,
        top_k: torch.Tensor,
        num_active_blocks: torch.Tensor,
        latency_budget: torch.Tensor,
        safety_factor: float = 0.8,
        expected_output_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Check if configurations satisfy latency budget.
        
        Design: Use prefill latency as PRIMARY metric, decode per-token as SECONDARY reference.
        
        Args:
            vision_tokens: (B,) number of vision tokens
            text_tokens: (B,) number of text tokens
            tier_idx: (B,) tier index
            top_k: (B,) MoE top-K value
            num_active_blocks: (B,) number of active blocks
            latency_budget: (B,) latency budget in ms
            safety_factor: (float) Safety factor for prefill budget (default 0.8, leaving 20% for decode)
            expected_output_tokens: (B,) optional expected number of output tokens (for conservative check)
        
        Returns:
            feasible: (B,) boolean tensor indicating if config satisfies budget
        """
        latencies = self.forward(
            vision_tokens=vision_tokens,
            text_tokens=text_tokens,
            tier_idx=tier_idx,
            top_k=top_k,
            num_active_blocks=num_active_blocks,
        )
        
        # PRIMARY: Check prefill latency (deterministic)
        prefill_budget = latency_budget * safety_factor
        prefill_feasible = latencies['T_prefill_total'] <= prefill_budget
        
        # SECONDARY: If expected_output_tokens provided, do conservative total latency check
        if expected_output_tokens is not None:
            # Use positioned decode latency: integrate over positions
            # For simplicity, use average decode latency at mid-position
            # More accurate: integrate decode latency over all positions
            positions = torch.arange(1, expected_output_tokens.max().int() + 1, device=expected_output_tokens.device).float()
            positions_expanded = positions.unsqueeze(0).expand(expected_output_tokens.shape[0], -1) # (B, max_tokens)
            
            # Predict decode latency at each position
            decode_latencies = self.predict_decode_at_positions(
                vision_tokens=vision_tokens,
                text_tokens=text_tokens,
                tier_idx=tier_idx,
                top_k=top_k,
                num_active_blocks=num_active_blocks,
                positions=positions,
            ) # (B, max_tokens)
            
            # For each sample, sum decode latency up to its expected_output_tokens
            total_decode_latency = torch.zeros_like(expected_output_tokens, dtype=torch.float32)
            for i in range(expected_output_tokens.shape[0]):
                num_tokens = int(expected_output_tokens[i].item())
                if num_tokens > 0:
                    total_decode_latency[i] = decode_latencies[i, :num_tokens].sum()
            
            total_latency = latencies['T_prefill_total'] + total_decode_latency
            total_feasible = total_latency <= latency_budget
            # Both conditions must be satisfied
            return prefill_feasible & total_feasible
        else:
            # Only check prefill (primary metric)
            return prefill_feasible
    
    def estimate_max_output_tokens(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        tier_idx: torch.Tensor,
        top_k: torch.Tensor,
        num_active_blocks: torch.Tensor,
        latency_budget: torch.Tensor,
        safety_factor: float = 0.8,
    ) -> torch.Tensor:
        """
        Estimate maximum possible output tokens given latency budget.
        
        This is a SECONDARY use case - provides an estimate, not a guarantee.
        
        Args:
            vision_tokens: (B,) number of vision tokens
            text_tokens: (B,) number of text tokens
            tier_idx: (B,) tier index
            top_k: (B,) MoE top-K value
            num_active_blocks: (B,) number of active blocks
            latency_budget: (B,) latency budget in ms
            safety_factor: (float) Safety factor for prefill budget
        
        Returns:
            max_output_tokens: (B,) estimated maximum output tokens
        """
        # Get prefill latency (position doesn't matter for prefill)
        prefill_latency = self.forward(
            vision_tokens=vision_tokens,
            text_tokens=text_tokens,
            tier_idx=tier_idx,
            top_k=top_k,
            num_active_blocks=num_active_blocks,
            token_position=torch.ones_like(vision_tokens),  # Position doesn't matter for prefill
        )['T_prefill_total']
        
        remaining_budget = latency_budget - prefill_latency
        
        # Estimate max tokens by iteratively checking positions
        # Start with a reasonable max and check cumulative latency
        max_possible = 100  # Reasonable upper bound
        positions = torch.arange(1, max_possible + 1, device=vision_tokens.device).float()
        
        decode_latencies = self.predict_decode_at_positions(
            vision_tokens=vision_tokens,
            text_tokens=text_tokens,
            tier_idx=tier_idx,
            top_k=top_k,
            num_active_blocks=num_active_blocks,
            positions=positions,
        ) # (B, max_possible)
        
        # Cumulative decode latency
        cum_decode_latency = torch.cumsum(decode_latencies, dim=1) # (B, max_possible)
        
        # Find max tokens where cum_decode_latency <= remaining_budget
        remaining_budget_expanded = remaining_budget.unsqueeze(1).expand(-1, max_possible) # (B, max_possible)
        feasible = cum_decode_latency <= remaining_budget_expanded # (B, max_possible)
        max_tokens = feasible.sum(dim=1).float() # (B,)
        
        return max_tokens.clamp(min=0.0)


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
        Training step with positioned decode latency.
        
        Design: 
        - For prefill: Direct prediction (deterministic)
        - For decode: Predict positioned latencies and sum to match total decode latency
        - This allows learning the progressive slowdown even without positioned data
        
        Args:
            batch: {
                'vision_tokens': (B,),
                'text_tokens': (B,),
                'tier_idx': (B,),
                'top_k': (B,),
                'num_active_blocks': (B,),
                'output_tokens': (B,),  # Number of output tokens (for training only)
                'T_prefill_total': (B,),  # T_vision_total + T_LLM_prefill
                'T_LLM_decode': (B,),  # Total decode latency (target)
                'T_decode_per_token_avg': (B,),  # Average per-token latency (for reference)
                'positioned_decode_latencies': List[List[float]],  # Optional: actual positioned latencies
                'has_positioned_data': List[bool],  # Optional: flag for each sample
            }
        
        Returns:
            metrics: Loss and error metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device (handle both tensors and lists)
        batch_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_device[k] = v.to(self.device)
            else:
                batch_device[k] = v  # Keep lists/other types on CPU
        
        # Predict prefill latency (doesn't depend on position)
        pred_prefill = self.model(
            vision_tokens=batch_device['vision_tokens'],
            text_tokens=batch_device['text_tokens'],
            tier_idx=batch_device['tier_idx'],
            top_k=batch_device['top_k'],
            num_active_blocks=batch_device['num_active_blocks'],
            token_position=torch.ones_like(batch_device['vision_tokens']),  # Position doesn't matter
        )['T_prefill_total']
        
        # Predict decode latency at all positions for each sample
        B = batch_device['vision_tokens'].shape[0]
        output_tokens = batch_device['output_tokens']
        max_tokens = int(output_tokens.max().item())
        
        # Create positions tensor [1, 2, ..., max_tokens]
        positions = torch.arange(1, max_tokens + 1, device=self.device).float()
        
        # Predict decode latency at all positions
        decode_latencies_all = self.model.predict_decode_at_positions(
            vision_tokens=batch_device['vision_tokens'],
            text_tokens=batch_device['text_tokens'],
            tier_idx=batch_device['tier_idx'],
            top_k=batch_device['top_k'],
            num_active_blocks=batch_device['num_active_blocks'],
            positions=positions,
        ) # (B, max_tokens)
        
        # For each sample, sum decode latency up to its output_tokens
        pred_decode_total = torch.zeros(B, device=self.device, dtype=torch.float32)
        for i in range(B):
            num_tokens = int(output_tokens[i].item())
            if num_tokens > 0 and num_tokens <= max_tokens:
                pred_decode_total[i] = decode_latencies_all[i, :num_tokens].sum()
        
        # Targets
        T_prefill_total_target = batch_device['T_prefill_total']
        T_LLM_decode_target = batch_device['T_LLM_decode']
        
        # Losses
        loss_prefill = self.criterion(pred_prefill, T_prefill_total_target)
        loss_decode_total = self.criterion(pred_decode_total, T_LLM_decode_target)
        
        # Optional: If we have positioned data, also compute per-position loss
        # Note: Currently, most data doesn't have positioned decode latency (T_decode_per_step),
        # so we skip this for now. When positioned data is available, we can add this back.
        loss_decode_positioned = torch.tensor(0.0, device=self.device)
        
        # Check if we have positioned data (from DataLoader, it's a list)
        if 'positioned_decode_latencies' in batch_device:
            has_positioned = batch_device.get('has_positioned_data', [])
            positioned_latencies = batch_device.get('positioned_decode_latencies', [])
            
            # has_positioned is a list from DataLoader, check if any sample has positioned data
            has_any_positioned = False
            if isinstance(has_positioned, list):
                has_any_positioned = any(has_positioned) if len(has_positioned) > 0 else False
            elif isinstance(has_positioned, torch.Tensor):
                has_any_positioned = has_positioned.any().item() if has_positioned.numel() > 0 else False
            
            if has_any_positioned and isinstance(positioned_latencies, list) and len(positioned_latencies) > 0:
                # Compute per-position loss for samples with positioned data
                positioned_losses = []
                for i in range(B):
                    if (i < len(has_positioned) and 
                        has_positioned[i] and 
                        i < len(positioned_latencies)):
                        pos_lats = positioned_latencies[i]
                        if isinstance(pos_lats, list) and len(pos_lats) > 0:
                            # Predict at these specific positions
                            pos_tensor = torch.tensor([j+1 for j in range(len(pos_lats))], device=self.device, dtype=torch.float32)
                            pred_pos = self.model.predict_decode_at_positions(
                                vision_tokens=batch_device['vision_tokens'][i:i+1],
                                text_tokens=batch_device['text_tokens'][i:i+1],
                                tier_idx=batch_device['tier_idx'][i:i+1],
                                top_k=batch_device['top_k'][i:i+1],
                                num_active_blocks=batch_device['num_active_blocks'][i:i+1],
                                positions=pos_tensor,
                            )[0, :len(pos_lats)]  # (len(pos_lats),)
                            
                            target_pos = torch.tensor(pos_lats, device=self.device, dtype=torch.float32)
                            positioned_losses.append(self.criterion(pred_pos, target_pos))
                
                if len(positioned_losses) > 0:
                    loss_decode_positioned = torch.stack(positioned_losses).mean()
        
        # Combined loss
        # Primary: prefill (weight 2.0)
        # Secondary: total decode latency (weight 1.0)
        # Optional: positioned decode latency (weight 0.5, if available)
        loss = 2.0 * loss_prefill + 1.0 * loss_decode_total
        if loss_decode_positioned.item() > 0:
            loss = loss + 0.5 * loss_decode_positioned
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        # Compute errors
        mae_prefill = F.l1_loss(pred_prefill, T_prefill_total_target).item()
        mae_decode_total = F.l1_loss(pred_decode_total, T_LLM_decode_target).item()
        
        # Relative errors
        rel_error_prefill = (mae_prefill / (T_prefill_total_target.mean() + 1e-6)).item()
        rel_error_decode_total = (mae_decode_total / (T_LLM_decode_target.mean() + 1e-6)).item()
        
        return {
            'loss': loss.item(),
            'loss_prefill': loss_prefill.item(),
            'loss_decode_total': loss_decode_total.item(),
            'loss_decode_positioned': loss_decode_positioned.item() if loss_decode_positioned.item() > 0 else 0.0,
            'mae_prefill': mae_prefill,
            'mae_decode_total': mae_decode_total,
            'rel_error_prefill': rel_error_prefill,
            'rel_error_decode_total': rel_error_decode_total,
        }
    
    def validate(
        self,
        val_loader,
    ) -> Dict[str, float]:
        """
        Validation with positioned decode latency.
        """
        self.model.eval()
        
        all_metrics = {
            'loss': [],
            'mae_prefill': [],
            'mae_decode_total': [],
            'rel_error_prefill': [],
            'rel_error_decode_total': [],
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device (handle both tensors and lists)
                batch_device = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_device[k] = v.to(self.device)
                    else:
                        batch_device[k] = v
                
                # Predict prefill (position doesn't matter)
                pred_prefill = self.model(
                    vision_tokens=batch_device['vision_tokens'],
                    text_tokens=batch_device['text_tokens'],
                    tier_idx=batch_device['tier_idx'],
                    top_k=batch_device['top_k'],
                    num_active_blocks=batch_device['num_active_blocks'],
                    token_position=torch.ones_like(batch_device['vision_tokens']),  # Position doesn't matter for prefill
                )['T_prefill_total']
                
                # Predict decode total
                B = batch_device['vision_tokens'].shape[0]
                output_tokens = batch_device['output_tokens']
                max_tokens = int(output_tokens.max().item())
                
                positions = torch.arange(1, max_tokens + 1, device=self.device).float()
                decode_latencies_all = self.model.predict_decode_at_positions(
                    vision_tokens=batch_device['vision_tokens'],
                    text_tokens=batch_device['text_tokens'],
                    tier_idx=batch_device['tier_idx'],
                    top_k=batch_device['top_k'],
                    num_active_blocks=batch_device['num_active_blocks'],
                    positions=positions,
                )
                
                pred_decode_total = torch.zeros(B, device=self.device, dtype=torch.float32)
                for i in range(B):
                    num_tokens = int(output_tokens[i].item())
                    if num_tokens > 0 and num_tokens <= max_tokens:
                        pred_decode_total[i] = decode_latencies_all[i, :num_tokens].sum()
                
                # Targets
                T_prefill_total_target = batch_device['T_prefill_total']
                T_LLM_decode_target = batch_device['T_LLM_decode']
                
                # Compute losses
                loss_prefill = self.criterion(pred_prefill, T_prefill_total_target)
                loss_decode_total = self.criterion(pred_decode_total, T_LLM_decode_target)
                loss = 2.0 * loss_prefill + 1.0 * loss_decode_total
                
                # Compute errors
                mae_prefill = F.l1_loss(pred_prefill, T_prefill_total_target).item()
                mae_decode_total = F.l1_loss(pred_decode_total, T_LLM_decode_target).item()
                
                # Relative errors
                rel_error_prefill = (mae_prefill / (T_prefill_total_target.mean() + 1e-6)).item()
                rel_error_decode_total = (mae_decode_total / (T_LLM_decode_target.mean() + 1e-6)).item()
                
                metrics = {
                    'loss': loss.item(),
                    'loss_prefill': loss_prefill.item(),
                    'loss_decode_total': loss_decode_total.item(),
                    'mae_prefill': mae_prefill,
                    'mae_decode_total': mae_decode_total,
                    'rel_error_prefill': rel_error_prefill,
                    'rel_error_decode_total': rel_error_decode_total,
                }
                
                for key in all_metrics:
                    if key in metrics:
                        all_metrics[key].append(metrics[key])
        
        return {k: sum(v) / len(v) for k, v in all_metrics.items() if v}

