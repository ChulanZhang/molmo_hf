"""
LLM-based Controller using first few layers of the base LLM.
This is the recommended architecture when transformer blocks < 12 show significant degradation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

log = logging.getLogger(__name__)


class LLMBasedController(nn.Module):
    """
    Controller using LLM's first few layers (like AdaLLaVA).
    
    Architecture:
    1. Vision/Language/Budget tokens → LLM first N layers (with default top_k=8)
    2. Extract latency token's hidden state
    3. Scheduler (lightweight MLP)
    4. Three heads for three knobs
    """
    
    def __init__(
        self,
        base_llm,
        num_controller_layers: int = 4,
        hidden_dim: int = 256,
        freeze_llm: bool = True,
        vision_feat_dim: int = 768,
        d_model: Optional[int] = None,
    ):
        """
        Args:
            base_llm: Base LLM model (MolmoModel)
            num_controller_layers: Number of LLM layers to use for controller (default: 4)
            hidden_dim: Hidden dimension for scheduler and heads
            freeze_llm: If True, freeze LLM parameters
            vision_feat_dim: Dimension of vision features (from CLIP encoder)
            d_model: Model dimension (auto-detected from base_llm if None)
        """
        super().__init__()
        self.base_llm = base_llm
        self.num_controller_layers = num_controller_layers
        self.hidden_dim = hidden_dim
        
        # Get d_model from config
        if d_model is None:
            d_model = base_llm.config.d_model
        self.d_model = d_model
        
        # Freeze LLM layers
        if freeze_llm:
            for param in self.base_llm.parameters():
                param.requires_grad = False
        
        # Feature projectors (align dimensions to d_model)
        self.vision_proj = nn.Linear(vision_feat_dim, d_model)
        self.lang_proj = nn.Linear(d_model, d_model)  # Identity (already d_model)
        self.budget_proj = nn.Linear(hidden_dim, d_model)  # Budget encoder outputs hidden_dim
        
        # Scheduler (lightweight, like AdaLLaVA)
        self.scheduler = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Three separate heads for three knobs
        # Knob1: Vision tokens tier (low, medium, high)
        self.knob1_head = nn.Linear(hidden_dim, 3)
        
        # Knob2: MoE top-K (4, 6, 8, 10, 12)
        self.knob2_head = nn.Linear(hidden_dim, 5)
        
        # Knob3: Transformer blocks count (8, 10, 12, 14, 16)
        self.knob3_head = nn.Linear(hidden_dim, 5)
        
        # Knob value mappings
        self.knob1_values = ["low", "medium", "high"]
        self.knob2_values = [4, 6, 8, 10, 12]
        self.knob3_values = [8, 10, 12, 14, 16]
    
    def forward(
        self,
        vision_feat: torch.Tensor,  # (B, vision_feat_dim)
        lang_feat: torch.Tensor,     # (B, d_model)
        budget_feat: torch.Tensor,    # (B, hidden_dim) - from budget encoder
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for controller decision.
        
        Note: This only makes decisions. The actual inference will use
        these decisions for layers num_controller_layers to end.
        
        Args:
            vision_feat: (B, vision_feat_dim) vision features
            lang_feat: (B, d_model) language features
            budget_feat: (B, hidden_dim) budget features
        
        Returns:
            knob_logits: dict with knob1, knob2, knob3 logits
        """
        batch_size = vision_feat.shape[0]
        
        # Project features to d_model
        vision_proj = self.vision_proj(vision_feat)  # (B, d_model)
        lang_proj = self.lang_proj(lang_feat)         # (B, d_model)
        budget_proj = self.budget_proj(budget_feat)  # (B, d_model)
        
        # Create token sequence: [vision_token, lang_token, budget_token]
        vision_token = vision_proj.unsqueeze(1)  # (B, 1, d_model)
        lang_token = lang_proj.unsqueeze(1)      # (B, 1, d_model)
        budget_token = budget_proj.unsqueeze(1)  # (B, 1, d_model)
        
        # Concatenate tokens
        input_tokens = torch.cat([vision_token, lang_token, budget_token], dim=1)  # (B, 3, d_model)
        
        # Process through first N layers of LLM (using default top_k=8)
        hidden_states = input_tokens
        for i in range(self.num_controller_layers):
            block = self.base_llm.transformer.blocks[i]
            # Forward through block
            hidden_states = block(hidden_states, use_cache=False)[0]  # (B, 3, d_model)
        
        # Extract budget token's hidden state (last token)
        budget_hidden = hidden_states[:, -1, :]  # (B, d_model)
        
        # Scheduler
        scheduler_output = self.scheduler(budget_hidden)  # (B, hidden_dim)
        
        # Three heads
        knob1_logits = self.knob1_head(scheduler_output)  # (B, 3)
        knob2_logits = self.knob2_head(scheduler_output)  # (B, 5)
        knob3_logits = self.knob3_head(scheduler_output)  # (B, 5)
        
        return {
            'knob1_logits': knob1_logits,
            'knob2_logits': knob2_logits,
            'knob3_logits': knob3_logits,
        }
    
    def sample_actions(
        self,
        logits: Dict[str, torch.Tensor],
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample actions from logits.
        
        Args:
            logits: Forward outputs
            temperature: Sampling temperature
            deterministic: If True, take argmax actions
        
        Returns:
            actions: dict with knob values and indices
        """
        if deterministic:
            knob1_idx = logits['knob1_logits'].argmax(dim=-1)
            knob2_idx = logits['knob2_logits'].argmax(dim=-1)
            knob3_idx = logits['knob3_logits'].argmax(dim=-1)
        else:
            knob1_dist = F.softmax(logits['knob1_logits'] / temperature, dim=-1)
            knob2_dist = F.softmax(logits['knob2_logits'] / temperature, dim=-1)
            knob3_dist = F.softmax(logits['knob3_logits'] / temperature, dim=-1)
            
            knob1_idx = torch.multinomial(knob1_dist, 1).squeeze(-1)
            knob2_idx = torch.multinomial(knob2_dist, 1).squeeze(-1)
            knob3_idx = torch.multinomial(knob3_dist, 1).squeeze(-1)
        
        # Map indices to values
        knob1_value = [self.knob1_values[i] for i in knob1_idx.cpu().tolist()]
        knob2_value = torch.tensor(
            [self.knob2_values[i] for i in knob2_idx.cpu().tolist()],
            device=knob2_idx.device,
            dtype=torch.long
        )
        knob3_value = torch.tensor(
            [self.knob3_values[i] for i in knob3_idx.cpu().tolist()],
            device=knob3_idx.device,
            dtype=torch.long
        )
        
        return {
            'knob1': knob1_value,
            'knob2': knob2_value,
            'knob3': knob3_value,
            'knob1_idx': knob1_idx,
            'knob2_idx': knob2_idx,
            'knob3_idx': knob3_idx,
        }
    
    def compute_log_probs(
        self,
        logits: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute log-probability of actions.
        
        Args:
            logits: Forward outputs
            actions: dict with indices ('knob1_idx', 'knob2_idx', 'knob3_idx')
        
        Returns:
            log_probs: (B,) per-sample log prob
        """
        knob1_log_probs = F.log_softmax(logits['knob1_logits'], dim=-1)
        knob2_log_probs = F.log_softmax(logits['knob2_logits'], dim=-1)
        knob3_log_probs = F.log_softmax(logits['knob3_logits'], dim=-1)
        
        # Select log-probs for chosen actions
        knob1_log_prob = knob1_log_probs.gather(
            1, actions['knob1_idx'].unsqueeze(-1)
        ).squeeze(-1)
        knob2_log_prob = knob2_log_probs.gather(
            1, actions['knob2_idx'].unsqueeze(-1)
        ).squeeze(-1)
        knob3_log_prob = knob3_log_probs.gather(
            1, actions['knob3_idx'].unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum log-probs (assume independence)
        total_log_prob = knob1_log_prob + knob2_log_prob + knob3_log_prob
        
        return total_log_prob

