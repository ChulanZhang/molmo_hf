"""
Two-Stage Controller for adaptive inference (Joint Training).

Implements the two-stage controller architecture with joint training:
- Stage 1: Predict Knob1 (vision tokens tier + insertion position) before vision encoding
- Stage 2: Predict Knob2 (MoE top-K) and Knob3 (transformer blocks) after insertion position

Key design:
- Joint Training: Both stages trained together end-to-end with shared reward
- Dynamic Insertion: Stage1 decides where to insert Stage2 (after block 1-5)
- Budget Token: Encoded as d_model-dim token, concatenated to input sequence
- Latency Token: Extracted from LLM after insertion position, contains all necessary info
- Decode Phase: Uses prefill configuration, no controller re-run

Current implementation:
- Knob1PredictorBudgetLanguage: Stage1 predictor (tier + insertion position)
- Knob2Knob3Predictor: Stage2 predictor (top_k + num_blocks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import logging

from experiments.controller.importance_based_block_selection import (
    ImportanceBasedBlockSelector,
    load_importance_scores,
    select_blocks_by_importance,
)

log = logging.getLogger(__name__)


class Knob1PredictorBudgetOnly(nn.Module):
    """
    Variant A: Budget-Only predictor (minimal overhead, ~0.01ms).
    
    Input: Latency budget only
    Output: Tier prediction (low/medium/high)
    
    Design: Tiny MLP (~10K params)
    """
    
    def __init__(
        self,
        budget_feat_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Direct scalar input encoding
        self.budget_encoder = nn.Sequential(
            nn.Linear(1, budget_feat_dim),
            nn.ReLU(),
            nn.LayerNorm(budget_feat_dim),
        )
        
        # Tiny head
        self.head = nn.Sequential(
            nn.Linear(budget_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # low, medium, high
        )
        
        self.tier_values = ["low", "medium", "high"]
    
    def forward(self, latency_budget: torch.Tensor) -> torch.Tensor:
        """
        Predict tier from budget only.
        
        Args:
            latency_budget: (B,) scalar latency budget in ms
        
        Returns:
            logits: (B, 3) logits for [low, medium, high]
        """
        budget_feat = self.budget_encoder(latency_budget.unsqueeze(-1))
        return self.head(budget_feat)
    
    def predict(
        self,
        latency_budget: float,
        deterministic: bool = True,
    ) -> str:
        """Predict tier value."""
        budget_tensor = torch.tensor([latency_budget], device=next(self.parameters()).device)
        logits = self.forward(budget_tensor)
        
        if deterministic:
            tier_idx = logits.argmax().item()
        else:
            probs = F.softmax(logits, dim=-1)
            tier_idx = torch.multinomial(probs, 1).item()
        
        return self.tier_values[tier_idx]


class Knob1PredictorBudgetLanguage(nn.Module):
    """
    Stage1 Controller: Predicts tier and insertion position.
    
    Input: 
        - Language Feature (d_model-dim): From prompt via WTE layer
        - Budget Feature (d_model-dim): From budget token (encoded)
    
    Output:
        - Tier (low/medium/high): Vision tokens tier
        - Insertion Position (1-5): Where to insert Stage2 controller
    
    Design: Lightweight MLP (~50K-100K params)
    - Two heads: tier prediction + insertion position prediction
    - Budget feature is d_model dimension (from budget token)
    """
    
    def __init__(
        self,
        lang_feat_dim: int = 2048,
        budget_feat_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        max_insertion_position: int = 5,  # Maximum insertion position (1-5 means after block 1-5)
    ):
        super().__init__()
        
        self.lang_proj = nn.Sequential(
            nn.Linear(lang_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.budget_proj = nn.Sequential(
            nn.Linear(budget_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Tier prediction head
        self.tier_head = nn.Linear(hidden_dim, 3)  # low, medium, high
        
        # Insertion position prediction head
        self.insertion_head = nn.Linear(hidden_dim, max_insertion_position)  # 1-5 (after block 1-5)
        
        self.tier_values = ["low", "medium", "high"]
        self.max_insertion_position = max_insertion_position
    
    def forward(
        self,
        lang_feat: torch.Tensor,  # (B, lang_feat_dim)
        budget_feat: torch.Tensor,  # (B, budget_feat_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Predict tier and insertion position from language + budget.
        
        Returns:
            {
                'tier_logits': (B, 3) logits for [low, medium, high],
                'insertion_logits': (B, max_insertion_position) logits for insertion position
            }
        """
        lang_proj = self.lang_proj(lang_feat)
        budget_proj = self.budget_proj(budget_feat)
        
        fused = torch.cat([lang_proj, budget_proj], dim=-1)
        hidden = self.fusion(fused)
        
        return {
            'tier_logits': self.tier_head(hidden),
            'insertion_logits': self.insertion_head(hidden),
        }


class Knob1PredictorBudgetLanguageVision(nn.Module):
    """
    Variant C: Budget + Language + Vision predictor (highest accuracy, ~30-50ms overhead).
    
    Input: Vision feature (global crop) + Language feature + Budget feature
    Output: Tier prediction (low/medium/high)
    
    Design: MLP or Transformer
    Note: Requires additional vision encoder pass (global crop), needs optimization.
    """
    
    def __init__(
        self,
        vision_feat_dim: int = 768,  # CLIP vision encoder output
        lang_feat_dim: int = 2048,
        budget_feat_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.lang_proj = nn.Sequential(
            nn.Linear(lang_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.budget_proj = nn.Sequential(
            nn.Linear(budget_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.head = nn.Linear(hidden_dim, 3)  # low, medium, high
        
        self.tier_values = ["low", "medium", "high"]
    
    def forward(
        self,
        vision_feat: torch.Tensor,  # (B, vision_feat_dim) - from global crop
        lang_feat: torch.Tensor,  # (B, lang_feat_dim)
        budget_feat: torch.Tensor,  # (B, budget_feat_dim)
    ) -> torch.Tensor:
        """
        Predict tier from vision + language + budget.
        
        Returns:
            logits: (B, 3) logits for [low, medium, high]
        """
        vision_proj = self.vision_proj(vision_feat)
        lang_proj = self.lang_proj(lang_feat)
        budget_proj = self.budget_proj(budget_feat)
        
        fused = torch.cat([vision_proj, lang_proj, budget_proj], dim=-1)
        hidden = self.fusion(fused)
        
        return self.head(hidden)


class Knob2Knob3Predictor(nn.Module):
    """
    Stage2 Controller: Predicts top-K and total blocks after insertion position.
    
    Input (two modes):
        Mode A (latency_token): Latency Token (d_model-dim) extracted from LLM after insertion position
        Mode B (optimized): lang_feat + budget_feat + insertion_position embedding (no partial forward needed)
    
    Output:
        - Top-K (4/5/6/7/8): MoE top-K for blocks after insertion position
        - Total Blocks (12/13/14/15/16): Total active blocks (including first block)
    
    Design: Lightweight MLP (~10K-30K params)
    
    Key design:
    - Supports two input modes for flexibility
    - Mode B (optimized) allows joint sampling before forward pass (5 forward passes instead of 10)
    - Dynamic knob3 options based on insertion position
    - First block fixed: top_k=8, always included
    """
    
    def __init__(
        self,
        latency_token_dim: int = 2048,  # d_model from transformer
        lang_feat_dim: int = 2048,  # d_model for language features
        budget_feat_dim: int = 2048,  # d_model for budget features
        hidden_dim: int = 128,
        dropout: float = 0.1,
        max_insertion_position: int = 5,  # Maximum insertion position
        total_blocks: int = 16,  # Total transformer blocks
        use_optimized_mode: bool = True,  # If True, use lang_feat+budget_feat instead of latency_token
    ):
        super().__init__()
        
        self.use_optimized_mode = use_optimized_mode
        
        if use_optimized_mode:
            # Mode B: lang_feat + budget_feat + insertion_position embedding
            self.lang_proj = nn.Sequential(
                nn.Linear(lang_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            self.budget_proj = nn.Sequential(
                nn.Linear(budget_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            # Insertion position embedding (1-5 -> embedding)
            self.insertion_embedding = nn.Embedding(max_insertion_position, hidden_dim)
            
            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            # Mode A: latency_token only
            self.latency_token_proj = nn.Sequential(
                nn.Linear(latency_token_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # Dynamic heads: will be created based on insertion position
        # For now, create heads for max possible options
        # knob2: top_k options (4,5,6,7,8) - same for all positions
        # knob3: num_blocks options depend on insertion position
        self.knob2_head = nn.Linear(hidden_dim, 5)  # top_k: 4,5,6,7,8
        
        # knob3 head will output max possible blocks
        # For insertion at position 1: max 15 remaining blocks -> total 16
        # For insertion at position 5: max 11 remaining blocks -> total 12
        # We use a fixed-size head for max possible options (5 options: 11-15 remaining blocks)
        # This covers all insertion positions (1-5)
        max_options = 5  # Maximum number of knob3 options across all insertion positions
        self.knob3_head = nn.Linear(hidden_dim, max_options)
        
        self.knob2_values = [4, 5, 6, 7, 8]  # Top-K for blocks after insertion position
        self.total_blocks = total_blocks
        self.max_insertion_position = max_insertion_position
    
    def get_knob3_options(self, insertion_position: int) -> List[int]:
        """
        Get knob3 options based on insertion position.
        
        Args:
            insertion_position: Position after which to insert (1-5, meaning after block 1-5)
        
        Returns:
            List of total block counts (including blocks before insertion)
        """
        # Remaining blocks after insertion position
        remaining_blocks = self.total_blocks - insertion_position
        
        # Options: keep 11-15 of remaining blocks (or fewer if remaining < 15)
        # Total = insertion_position + selected_remaining
        min_selected = max(11, remaining_blocks - 4)  # At least 11, or remaining-4 if remaining < 15
        max_selected = min(15, remaining_blocks)  # At most 15, or remaining if remaining < 15
        
        # Generate options: [min_selected, min_selected+1, ..., max_selected]
        options = list(range(min_selected, max_selected + 1))
        
        # Total blocks = insertion_position + selected_remaining
        total_options = [insertion_position + selected for selected in options]
        
        return total_options
    
    def forward(
        self,
        latency_token: Optional[torch.Tensor] = None,  # (B, latency_token_dim) - Mode A
        insertion_position: torch.Tensor = None,  # (B,) insertion position (1-5)
        lang_feat: Optional[torch.Tensor] = None,  # (B, lang_feat_dim) - Mode B
        budget_feat: Optional[torch.Tensor] = None,  # (B, budget_feat_dim) - Mode B
    ) -> Dict[str, torch.Tensor]:
        """
        Predict Knob2 and Knob3 based on insertion position.
        
        Supports two modes:
        - Mode A (latency_token): Uses latency token extracted from LLM after insertion position
        - Mode B (optimized): Uses lang_feat + budget_feat + insertion_position embedding
                              (allows joint sampling before forward pass, reducing from 10 to 5 forward passes)
        
        Returns:
            {
                'knob2_logits': (B, 5),
                'knob3_logits': (B, num_options),  # num_options depends on insertion_position
            }
        """
        if self.use_optimized_mode and lang_feat is not None and budget_feat is not None:
            # Mode B: lang_feat + budget_feat + insertion_position embedding
            lang_proj = self.lang_proj(lang_feat)  # (B, hidden_dim)
            budget_proj = self.budget_proj(budget_feat)  # (B, hidden_dim)
            
            # Insertion position embedding (convert 1-5 to 0-4 for embedding)
            insertion_idx = insertion_position - 1  # (B,) convert 1-5 to 0-4
            insertion_emb = self.insertion_embedding(insertion_idx)  # (B, hidden_dim)
            
            # Fuse features
            fused = torch.cat([lang_proj, budget_proj, insertion_emb], dim=-1)  # (B, hidden_dim * 3)
            fused = self.fusion(fused)  # (B, hidden_dim)
        else:
            # Mode A: latency_token only
            if latency_token is None:
                raise ValueError("latency_token must be provided in Mode A")
            fused = self.latency_token_proj(latency_token)  # (B, hidden_dim)
        
        # Predict knob2 (same for all positions)
        knob2_logits = self.knob2_head(fused)  # (B, 5)
        
        # Predict knob3 (dynamic based on insertion position)
        # For simplicity, we'll use a fixed-size head and mask invalid options
        # Or we can use a separate head per insertion position
        # For now, use a single head and handle masking in loss computation
        knob3_logits_base = self.knob3_head(fused)  # (B, max_options)
        
        # For each sample, get valid options based on insertion_position
        # This is a bit complex, so we'll handle it in the calling code
        # For now, return base logits and insertion_position
        return {
            'knob2_logits': knob2_logits,
            'knob3_logits': knob3_logits_base,  # Will be masked/selected based on insertion_position
            'insertion_position': insertion_position,
        }
    
    def predict(
        self,
        vision_feat: torch.Tensor,
        lang_feat: torch.Tensor,
        budget_feat: torch.Tensor,
        deterministic: bool = True,
    ) -> Dict:
        """Predict knob values."""
        logits = self.forward(vision_feat, lang_feat, budget_feat)
        
        if deterministic:
            knob2_idx = logits['knob2_logits'].argmax(dim=-1)
            knob3_idx = logits['knob3_logits'].argmax(dim=-1)
        else:
            knob2_probs = F.softmax(logits['knob2_logits'], dim=-1)
            knob3_probs = F.softmax(logits['knob3_logits'], dim=-1)
            knob2_idx = torch.multinomial(knob2_probs, 1).squeeze(-1)
            knob3_idx = torch.multinomial(knob3_probs, 1).squeeze(-1)
        
        knob2 = torch.tensor(
            [self.knob2_values[i] for i in knob2_idx.cpu().tolist()],
            device=knob2_idx.device,
            dtype=torch.long
        )
        knob3 = torch.tensor(
            [self.knob3_values[i] for i in knob3_idx.cpu().tolist()],
            device=knob3_idx.device,
            dtype=torch.long
        )
        
        return {
            'knob2': knob2,
            'knob3': knob3,
            'knob2_idx': knob2_idx,
            'knob3_idx': knob3_idx,
        }


class TwoStageController(nn.Module):
    """
    Complete two-stage controller for adaptive inference.
    
    Architecture:
    - Stage 1: Predict Knob1 (before vision encoding)
    - Stage 2: Predict Knob2 & Knob3 (after vision encoding)
    
    Knob3 uses importance-based block selection (deterministic, based on pre-computed scores).
    Importance scores are data-agnostic but task-dependent.
    """
    
    def __init__(
        self,
        knob1_variant: str = "budget_language",  # "budget_only", "budget_language", "budget_language_vision"
        lang_feat_dim: int = 2048,
        vision_feat_dim: int = 2048,  # For Stage 2 (after projector)
        budget_feat_dim: int = 256,
        hidden_dim: int = 128,
        importance_scores: Optional[Dict[int, float]] = None,
        importance_file: Optional[str] = None,
        task_type: Optional[str] = None,  # For task-dependent importance scores
    ):
        """
        Args:
            knob1_variant: Knob1 predictor variant ("budget_only", "budget_language", "budget_language_vision")
            lang_feat_dim: Language feature dimension
            vision_feat_dim: Vision feature dimension (after projector)
            budget_feat_dim: Budget feature dimension
            hidden_dim: Hidden dimension for MLPs
            importance_scores: Pre-computed importance scores (optional)
            importance_file: Path to importance scores file (optional)
            task_type: Task type for task-dependent importance scores (e.g., "vqa", "science_qa")
        """
        super().__init__()
        
        # Stage 1: Knob1 predictor (select variant)
        if knob1_variant == "budget_only":
            self.knob1_predictor = Knob1PredictorBudgetOnly(
                budget_feat_dim=budget_feat_dim,
                hidden_dim=64,
            )
        elif knob1_variant == "budget_language":
            self.knob1_predictor = Knob1PredictorBudgetLanguage(
                lang_feat_dim=lang_feat_dim,
                budget_feat_dim=budget_feat_dim,
                hidden_dim=256,
            )
        elif knob1_variant == "budget_language_vision":
            self.knob1_predictor = Knob1PredictorBudgetLanguageVision(
                vision_feat_dim=768,  # CLIP encoder output
                lang_feat_dim=lang_feat_dim,
                budget_feat_dim=budget_feat_dim,
                hidden_dim=256,
            )
        else:
            raise ValueError(f"Unknown knob1_variant: {knob1_variant}")
        
        self.knob1_variant = knob1_variant
        
        # Stage 2: Knob2 & Knob3 predictor
        self.knob2_knob3_predictor = Knob2Knob3Predictor(
            vision_feat_dim=vision_feat_dim,
            lang_feat_dim=lang_feat_dim,
            budget_feat_dim=budget_feat_dim,
            hidden_dim=hidden_dim,
        )
        
        # Importance-based block selector
        # Note: Importance scores are data-agnostic but task-dependent
        # If task_type is provided, we can select task-specific importance scores
        self.task_type = task_type
        self.block_selector = None
        
        if importance_scores is not None:
            self.block_selector = ImportanceBasedBlockSelector(
                importance_scores=importance_scores,
                task_type=task_type,
            )
        elif importance_file is not None:
            # Load importance scores (can be task-specific)
            self.block_selector = ImportanceBasedBlockSelector(
                importance_file=importance_file,
                task_type=task_type,
            )
        else:
            log.warning("No importance scores provided. Knob3 will use default selection (first N blocks).")
    
    def forward_stage1(
        self,
        latency_budget: torch.Tensor,  # (B,) scalar
        lang_feat: Optional[torch.Tensor] = None,  # (B, lang_feat_dim)
        vision_feat: Optional[torch.Tensor] = None,  # (B, vision_feat_dim) - for variant C
        budget_feat: Optional[torch.Tensor] = None,  # (B, budget_feat_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 1: Predict Knob1 (before vision encoding).
        
        Args:
            latency_budget: (B,) scalar latency budget in ms
            lang_feat: (B, lang_feat_dim) - required for variant B and C
            vision_feat: (B, vision_feat_dim) - required for variant C (global crop)
            budget_feat: (B, budget_feat_dim) - optional, will encode from latency_budget if not provided
        
        Returns:
            {'knob1_logits': (B, 3)}
        """
        # Predict based on variant
        if self.knob1_variant == "budget_only":
            # Variant A: Budget only
            knob1_logits = self.knob1_predictor(latency_budget)
        elif self.knob1_variant == "budget_language":
            # Variant B: Budget + Language
            if lang_feat is None:
                raise ValueError("lang_feat is required for budget_language variant")
            # Encode budget if not provided
            if budget_feat is None:
                # Use a simple encoding (can be improved with proper budget encoder)
                budget_feat = self._encode_budget_simple(latency_budget)
            knob1_logits = self.knob1_predictor(lang_feat, budget_feat)
        elif self.knob1_variant == "budget_language_vision":
            # Variant C: Budget + Language + Vision
            if lang_feat is None or vision_feat is None:
                raise ValueError("lang_feat and vision_feat are required for budget_language_vision variant")
            # Encode budget if not provided
            if budget_feat is None:
                budget_feat = self._encode_budget_simple(latency_budget)
            knob1_logits = self.knob1_predictor(vision_feat, lang_feat, budget_feat)
        else:
            raise ValueError(f"Unknown knob1_variant: {self.knob1_variant}")
        
        return {'knob1_logits': knob1_logits}
    
    def _encode_budget_simple(self, latency_budget: torch.Tensor) -> torch.Tensor:
        """
        Simple budget encoding: scalar -> feature.
        
        Note: This is a placeholder. In practice, should use LatencyBudgetEncoder
        from feature_extractors.py for proper encoding.
        
        Args:
            latency_budget: (B,) scalar latency budget
        
        Returns:
            budget_feat: (B, budget_feat_dim)
        """
        # Simple encoding using the budget_proj from knob2_knob3_predictor
        # This is a workaround - ideally should have a shared budget encoder
        budget_feat_dim = 256
        budget_feat = F.relu(
            torch.matmul(
                latency_budget.unsqueeze(-1),  # (B, 1)
                self.knob2_knob3_predictor.budget_proj.weight.t()  # (1, budget_feat_dim)
            )
        )
        return budget_feat
    
    def forward_stage2(
        self,
        vision_feat: torch.Tensor,  # (B, vision_feat_dim) - after encoder+projector, pooled
        lang_feat: torch.Tensor,     # (B, lang_feat_dim)
        budget_feat: torch.Tensor,    # (B, budget_feat_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 2: Predict Knob2 & Knob3 (after vision encoding).
        
        Returns:
            {
                'knob2_logits': (B, 5),
                'knob3_logits': (B, 5),
            }
        """
        return self.knob2_knob3_predictor(vision_feat, lang_feat, budget_feat)
    
    def compute_log_probs(
        self,
        knob1_logits: torch.Tensor,
        knob2_logits: torch.Tensor,
        knob3_logits: torch.Tensor,
        actions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute log-probability of actions.
        
        Args:
            knob1_logits: (B, 3)
            knob2_logits: (B, 5)
            knob3_logits: (B, 5)
            actions: {
                'knob1_idx': (B,),
                'knob2_idx': (B,),
                'knob3_idx': (B,),
            }
        
        Returns:
            log_probs: (B,) total log-prob
        """
        knob1_log_probs = F.log_softmax(knob1_logits, dim=-1)
        knob2_log_probs = F.log_softmax(knob2_logits, dim=-1)
        knob3_log_probs = F.log_softmax(knob3_logits, dim=-1)
        
        knob1_log_prob = knob1_log_probs.gather(1, actions['knob1_idx'].unsqueeze(-1)).squeeze(-1)
        knob2_log_prob = knob2_log_probs.gather(1, actions['knob2_idx'].unsqueeze(-1)).squeeze(-1)
        knob3_log_prob = knob3_log_probs.gather(1, actions['knob3_idx'].unsqueeze(-1)).squeeze(-1)
        
        return knob1_log_prob + knob2_log_prob + knob3_log_prob
    
    def get_selected_blocks_for_knob3(
        self,
        knob3_value: int,  # num_blocks: 8, 10, 12, 14, 16
        task_type: Optional[str] = None,
    ) -> List[int]:
        """
        Get selected blocks for given Knob3 value using importance scores.
        
        Importance scores are data-agnostic but task-dependent.
        If task_type is provided, use task-specific importance scores if available.
        
        Args:
            knob3_value: Number of blocks to select
            task_type: Task type for task-dependent selection (e.g., "vqa", "science_qa")
        
        Returns:
            selected_blocks: List of selected block indices
        """
        if self.block_selector is None:
            # Fallback: use first N blocks
            return list(range(knob3_value))
        
        # Use task-specific importance scores if available
        # For now, use the provided importance scores (can be extended to task-specific)
        return self.block_selector.select_blocks(knob3_value)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters for overhead analysis."""
        return {
            'knob1': sum(p.numel() for p in self.knob1_predictor.parameters()),
            'knob2_knob3': sum(p.numel() for p in self.knob2_knob3_predictor.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
        }

# ---------------- One-Stage Controller ----------------

class OneStageControllerPredictor(nn.Module):
    """
    One-stage controller (no insertion position):
      - Predicts tier (Knob1)
      - Predicts block activation scores (for blocks 1-15; block0 always on)
      - Predicts per-block top-k logits (for blocks 0-15)
    """
    def __init__(
        self,
        vision_dim: int = 768,
        lang_dim: int = 2048,
        budget_dim: int = 2048,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        total_blocks: int = 16,
    ):
        super().__init__()
        self.total_blocks = total_blocks
        self.tier_values = ["low", "medium", "high"]
        self.topk_choices = [4, 5, 6, 7, 8]
        
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lang_proj = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.budget_proj = nn.Sequential(
            nn.Linear(budget_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.tier_head = nn.Linear(hidden_dim, 3)  # low/medium/high
        # block logits for blocks 1-15 (block0 always on)
        self.block_head = nn.Linear(hidden_dim, total_blocks - 1)
        # per-block topk logits (16 blocks, each 5 options)
        self.block_topk_head = nn.Linear(hidden_dim, total_blocks * len(self.topk_choices))
    
    def forward(
        self,
        vision_feat: torch.Tensor,  # (B, vision_dim)
        lang_feat: torch.Tensor,    # (B, lang_dim)
        budget_feat: torch.Tensor,  # (B, budget_dim)
    ) -> Dict[str, torch.Tensor]:
        v = self.vision_proj(vision_feat)
        l = self.lang_proj(lang_feat)
        b = self.budget_proj(budget_feat)
        fused = self.fusion(torch.cat([v, l, b], dim=-1))
        
        tier_logits = self.tier_head(fused)
        block_logits = self.block_head(fused)  # (B, 15) for blocks 1-15
        block_topk_logits = self.block_topk_head(fused).view(
            fused.shape[0], self.total_blocks, len(self.topk_choices)
        )  # (B,16,5)
        
        return {
            "tier_logits": tier_logits,
            "block_logits": block_logits,
            "block_topk_logits": block_topk_logits,
        }

