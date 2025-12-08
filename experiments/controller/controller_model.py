"""
GRPO Controller模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

log = logging.getLogger(__name__)


class GRPOController(nn.Module):
    """
    GRPO Controller网络
    
    输入：图像特征、语言特征、延迟预算
    输出：动作分布（max_crops, top_k, num_active_blocks）
    """
    
    def __init__(
        self,
        image_feat_dim: int = 768,      # 图像特征维度
        lang_feat_dim: int = 2048,       # 语言特征维度（embedding维度）
        budget_dim: int = 32,            # 预算编码维度
        hidden_dim: int = 256,           # 隐藏层维度
        num_max_crops: int = 6,          # max_crops选项数量
        num_top_k: int = 8,              # top_k选项数量
        num_blocks: int = 9,             # num_active_blocks选项数量
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.image_feat_dim = image_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.budget_dim = budget_dim
        self.hidden_dim = hidden_dim
        
        # 特征投影层（用于维度对齐和降维）
        self.image_proj = nn.Sequential(
            nn.Linear(image_feat_dim, hidden_dim),
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
        
        # 预算编码
        self.budget_encoder = nn.Sequential(
            nn.Linear(1, budget_dim),
            nn.ReLU(),
            nn.Linear(budget_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # 策略头（输出动作分布）
        self.max_crops_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_max_crops),
        )
        
        self.top_k_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_top_k),
        )
        
        self.blocks_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_blocks),
        )
        
        # 动作空间定义（需要与训练数据一致）
        self.max_crops_options = [2, 4, 6, 8, 10, 12]
        self.top_k_options = [4, 8, 12, 16, 20, 24, 28, 32]
        self.blocks_options = [8, 9, 10, 11, 12, 13, 14, 15, 16]
        
        assert len(self.max_crops_options) == num_max_crops
        assert len(self.top_k_options) == num_top_k
        assert len(self.blocks_options) == num_blocks
    
    def forward(
        self,
        image_feat: torch.Tensor,
        lang_feat: torch.Tensor,
        budget: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            image_feat: (B, image_feat_dim) 图像特征
            lang_feat: (B, lang_feat_dim) 语言特征
            budget: (B,) 延迟预算（标量）
        
        Returns:
            {
                'max_crops_logits': (B, num_max_crops),
                'top_k_logits': (B, num_top_k),
                'blocks_logits': (B, num_blocks),
            }
        """
        batch_size = image_feat.shape[0]
        
        # 投影特征
        img_proj = self.image_proj(image_feat)  # (B, hidden_dim)
        lang_proj = self.lang_proj(lang_feat)   # (B, hidden_dim)
        
        # 编码预算
        budget = budget.unsqueeze(-1)  # (B, 1)
        budget_proj = self.budget_encoder(budget)  # (B, hidden_dim)
        
        # 融合
        fused = torch.cat([img_proj, lang_proj, budget_proj], dim=-1)  # (B, hidden_dim * 3)
        hidden = self.fusion(fused)  # (B, hidden_dim)
        
        # 输出动作分布
        max_crops_logits = self.max_crops_head(hidden)  # (B, num_max_crops)
        top_k_logits = self.top_k_head(hidden)           # (B, num_top_k)
        blocks_logits = self.blocks_head(hidden)          # (B, num_blocks)
        
        return {
            'max_crops_logits': max_crops_logits,
            'top_k_logits': top_k_logits,
            'blocks_logits': blocks_logits,
        }
    
    def sample_actions(
        self,
        logits: Dict[str, torch.Tensor],
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        从动作分布中采样
        
        Args:
            logits: forward的输出
            temperature: 采样温度
            deterministic: 如果True，选择最大概率的动作
        
        Returns:
            {
                'max_crops': (B,),
                'top_k': (B,),
                'num_active_blocks': (B,),
            }
        """
        if deterministic:
            max_crops_idx = logits['max_crops_logits'].argmax(dim=-1)
            top_k_idx = logits['top_k_logits'].argmax(dim=-1)
            blocks_idx = logits['blocks_logits'].argmax(dim=-1)
        else:
            max_crops_dist = F.softmax(logits['max_crops_logits'] / temperature, dim=-1)
            top_k_dist = F.softmax(logits['top_k_logits'] / temperature, dim=-1)
            blocks_dist = F.softmax(logits['blocks_logits'] / temperature, dim=-1)
            
            max_crops_idx = torch.multinomial(max_crops_dist, 1).squeeze(-1)
            top_k_idx = torch.multinomial(top_k_dist, 1).squeeze(-1)
            blocks_idx = torch.multinomial(blocks_dist, 1).squeeze(-1)
        
        # 映射到实际值
        max_crops = torch.tensor([self.max_crops_options[i] for i in max_crops_idx.cpu()], 
                                device=max_crops_idx.device)
        top_k = torch.tensor([self.top_k_options[i] for i in top_k_idx.cpu()], 
                            device=top_k_idx.device)
        blocks = torch.tensor([self.blocks_options[i] for i in blocks_idx.cpu()], 
                             device=blocks_idx.device)
        
        return {
            'max_crops': max_crops,
            'top_k': top_k,
            'num_active_blocks': blocks,
            'max_crops_idx': max_crops_idx,
            'top_k_idx': top_k_idx,
            'blocks_idx': blocks_idx,
        }
    
    def compute_log_probs(
        self,
        logits: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算动作的对数概率
        
        Args:
            logits: forward的输出
            actions: 包含'max_crops_idx', 'top_k_idx', 'blocks_idx'的字典
        
        Returns:
            log_probs: (B,) 每个样本的对数概率
        """
        max_crops_log_probs = F.log_softmax(logits['max_crops_logits'], dim=-1)
        top_k_log_probs = F.log_softmax(logits['top_k_logits'], dim=-1)
        blocks_log_probs = F.log_softmax(logits['blocks_logits'], dim=-1)
        
        # 选择对应动作的对数概率
        max_crops_log_prob = max_crops_log_probs.gather(
            1, actions['max_crops_idx'].unsqueeze(-1)
        ).squeeze(-1)
        top_k_log_prob = top_k_log_probs.gather(
            1, actions['top_k_idx'].unsqueeze(-1)
        ).squeeze(-1)
        blocks_log_prob = blocks_log_probs.gather(
            1, actions['blocks_idx'].unsqueeze(-1)
        ).squeeze(-1)
        
        # 总对数概率（假设三个动作独立）
        total_log_prob = max_crops_log_prob + top_k_log_prob + blocks_log_prob
        
        return total_log_prob


class RewardFunction:
    """Reward函数"""
    
    def __init__(
        self,
        alpha: float = 1.0,      # 准确率权重
        beta: float = 0.5,       # 延迟惩罚权重
        gamma: float = 10.0,     # 预算违反惩罚权重
        delta: float = 0.1,      # 效率奖励权重
        epsilon: float = 0.05,   # 复杂度惩罚权重
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
    
    def __call__(
        self,
        accuracy: torch.Tensor,
        latency: torch.Tensor,
        latency_budget: torch.Tensor,
        config: Dict[str, torch.Tensor],
        baseline_accuracy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算reward
        
        Args:
            accuracy: (B,) 准确率
            latency: (B,) 实际延迟
            latency_budget: (B,) 延迟预算
            config: 包含'max_crops', 'top_k', 'num_active_blocks'的字典
            baseline_accuracy: (B,) 可选，baseline准确率
        
        Returns:
            reward: (B,) 奖励值
        """
        # 相对准确率
        if baseline_accuracy is not None:
            relative_accuracy = (accuracy - baseline_accuracy) / (1.0 - baseline_accuracy + 1e-6)
        else:
            relative_accuracy = accuracy
        
        accuracy_reward = self.alpha * relative_accuracy
        
        # 平滑的延迟惩罚
        latency_ratio = latency / (latency_budget + 1e-6)
        latency_penalty = self.beta * torch.sigmoid(10.0 * (latency_ratio - 1.0))
        
        # 预算违反惩罚（硬约束）
        budget_violation = torch.clamp(latency - latency_budget, min=0.0)
        budget_violation_penalty = self.gamma * (budget_violation / (latency_budget + 1e-6)) ** 2
        
        # 配置复杂度惩罚
        max_crops_norm = config['max_crops'] / 12.0
        top_k_norm = config['top_k'] / 32.0
        blocks_norm = config['num_active_blocks'] / 16.0
        complexity = (max_crops_norm + top_k_norm + blocks_norm) / 3.0
        complexity_penalty = self.epsilon * complexity
        
        # 效率奖励（在预算内时）
        efficiency_bonus = torch.where(
            latency <= latency_budget,
            self.delta * (1.0 - latency / (latency_budget + 1e-6)),
            torch.zeros_like(latency)
        )
        
        reward = (accuracy_reward - 
                 latency_penalty - 
                 budget_violation_penalty - 
                 complexity_penalty + 
                 efficiency_bonus)
        
        return reward

