"""
Feature extractors for controller training.
Extracts vision, language, and budget features from inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
import numpy as np

log = logging.getLogger(__name__)


class VisionFeatureExtractor:
    """
    Extract vision features using CLIP vision encoder.
    Uses global crop + CLIP encoder + pooling (no projector).
    """
    
    def __init__(self, vision_encoder, use_cls_token: bool = True):
        """
        Args:
            vision_encoder: CLIP vision encoder (from model)
            use_cls_token: If True, use CLS token; else use mean pooling
        """
        self.vision_encoder = vision_encoder
        self.use_cls_token = use_cls_token
        
        # Freeze vision encoder
        self.vision_encoder.eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract vision feature from image.
        
        Args:
            image: (B, 3, 336, 336) - global crop image
        
        Returns:
            vision_feat: (B, d_vision) - pooled vision feature
        """
        with torch.no_grad():
            # Encode image
            # Assuming vision_encoder returns patch features or tuple
            output = self.vision_encoder(image)
            
            if isinstance(output, tuple):
                patch_features = output[0]  # (B, num_patches, d_vision)
            elif isinstance(output, list):
                patch_features = output[-1]  # Last layer
            else:
                patch_features = output
            
            # Handle different output formats
            if len(patch_features.shape) == 3:
                # (B, num_patches, d_vision)
                if self.use_cls_token and patch_features.shape[1] > 0:
                    # Use CLS token (first token)
                    vision_feat = patch_features[:, 0]  # (B, d_vision)
                else:
                    # Mean pooling
                    vision_feat = patch_features.mean(dim=1)  # (B, d_vision)
            elif len(patch_features.shape) == 2:
                # Already pooled: (B, d_vision)
                vision_feat = patch_features
            else:
                raise ValueError(f"Unexpected vision encoder output shape: {patch_features.shape}")
        
        return vision_feat


class LanguageFeatureExtractor:
    """
    Extract language features using tokenizer + word embedding + pooling.
    """
    
    def __init__(self, tokenizer, wte_layer, max_length: int = 512):
        """
        Args:
            tokenizer: Tokenizer instance
            wte_layer: Word embedding layer (transformer.wte)
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.wte_layer = wte_layer
        self.max_length = max_length
        
        # Freeze WTE layer
        self.wte_layer.eval()
        for param in self.wte_layer.parameters():
            param.requires_grad = False
    
    def extract(self, prompt: str) -> torch.Tensor:
        """
        Extract language feature from prompt.
        
        Args:
            prompt: str - language prompt
        
        Returns:
            lang_feat: (1, d_model) - pooled language feature
        """
        # Tokenize
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        input_ids = tokens['input_ids']  # (1, seq_len)
        
        with torch.no_grad():
            # Get embeddings
            token_embeds = self.wte_layer(input_ids)  # (1, seq_len, d_model)
            
            # Mean pooling
            lang_feat = token_embeds.mean(dim=1)  # (1, d_model)
        
        return lang_feat


class LatencyBudgetEncoder(nn.Module):
    """
    Encode latency budget to feature vector.
    Uses simple MLP (or sinusoidal + MLP like AdaLLaVA).
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        use_sinusoidal: bool = False,
        normalize_budget: bool = True,
        budget_min: float = 50.0,
        budget_max: float = 500.0,
    ):
        """
        Args:
            hidden_dim: Hidden dimension for budget encoding
            use_sinusoidal: If True, use sinusoidal encoding (like AdaLLaVA)
            normalize_budget: If True, normalize budget to [0, 1]
            budget_min: Minimum budget value for normalization
            budget_max: Maximum budget value for normalization
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_sinusoidal = use_sinusoidal
        self.normalize_budget = normalize_budget
        self.budget_min = budget_min
        self.budget_max = budget_max
        
        if use_sinusoidal:
            # AdaLLaVA style: sinusoidal + MLP
            self.pos_encoding_dim = 256
            self.mlp = nn.Sequential(
                nn.Linear(self.pos_encoding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            # Simple MLP
            self.encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
    
    def _sinusoidal_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal positional encoding.
        
        Args:
            x: (B, 1) scalar values
        
        Returns:
            encoded: (B, d_model)
        """
        position = x.squeeze(-1)  # (B,)
        d_model = self.pos_encoding_dim
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device, dtype=torch.float32)
            * -(np.log(10000.0) / d_model)
        )
        
        encoded = torch.zeros(x.shape[0], d_model, device=x.device)
        encoded[:, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        encoded[:, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
        
        return encoded
    
    def forward(self, budget: torch.Tensor) -> torch.Tensor:
        """
        Encode latency budget.
        
        Args:
            budget: (B,) latency budget in ms
        
        Returns:
            budget_feat: (B, hidden_dim)
        """
        # Normalize budget if needed
        if self.normalize_budget:
            budget = (budget - self.budget_min) / (self.budget_max - self.budget_min + 1e-6)
            budget = torch.clamp(budget, 0.0, 1.0)
        
        budget = budget.unsqueeze(-1)  # (B, 1)
        
        if self.use_sinusoidal:
            # Sinusoidal encoding
            pos_encoded = self._sinusoidal_encoding(budget)  # (B, 256)
            return self.mlp(pos_encoded)  # (B, hidden_dim)
        else:
            # Direct MLP
            return self.encoder(budget)  # (B, hidden_dim)


class FeatureExtractionPipeline:
    """
    Complete pipeline for extracting all features needed for controller.
    """
    
    def __init__(
        self,
        vision_encoder,
        tokenizer,
        wte_layer,
        device: str = "cuda",
        use_cls_token: bool = True,
        budget_encoder_config: Optional[Dict] = None,
    ):
        """
        Args:
            vision_encoder: CLIP vision encoder
            tokenizer: Tokenizer
            wte_layer: Word embedding layer
            device: Device to use
            use_cls_token: Whether to use CLS token for vision
            budget_encoder_config: Config for budget encoder
        """
        self.device = device
        
        # Initialize extractors
        self.vision_extractor = VisionFeatureExtractor(
            vision_encoder,
            use_cls_token=use_cls_token
        )
        
        self.lang_extractor = LanguageFeatureExtractor(
            tokenizer,
            wte_layer,
            max_length=512
        )
        
        budget_config = budget_encoder_config or {}
        self.budget_encoder = LatencyBudgetEncoder(**budget_config)
        self.budget_encoder.to(device)
    
    def extract_features(
        self,
        images: torch.Tensor,
        prompts: list,
        budgets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all features for a batch.
        
        Args:
            images: (B, 3, 336, 336) - global crop images
            prompts: List of B strings - language prompts
            budgets: (B,) - latency budgets in ms
        
        Returns:
            features: {
                'vision_feat': (B, d_vision),
                'lang_feat': (B, d_model),
                'budget_feat': (B, hidden_dim),
            }
        """
        # Extract vision features
        vision_feat = self.vision_extractor.extract(images.to(self.device))
        
        # Extract language features
        lang_feats = []
        for prompt in prompts:
            lang_feat = self.lang_extractor.extract(prompt)
            lang_feats.append(lang_feat)
        lang_feat = torch.cat(lang_feats, dim=0).to(self.device)  # (B, d_model)
        
        # Encode budgets
        budget_feat = self.budget_encoder(budgets.to(self.device))  # (B, hidden_dim)
        
        return {
            'vision_feat': vision_feat,
            'lang_feat': lang_feat,
            'budget_feat': budget_feat,
        }

