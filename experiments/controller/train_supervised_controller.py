"""
Train supervised learning controller from profiling results.
Simple baseline: learn latency_budget -> (tier, top_k, num_blocks) mapping.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.controller.feature_extractors import LatencyBudgetEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


class SupervisedController(nn.Module):
    """
    Supervised learning controller.
    Trained on profiling data: latency_budget -> (tier, top_k, num_blocks)
    """
    
    def __init__(
        self,
        budget_feat_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Budget encoder
        self.budget_encoder = LatencyBudgetEncoder(
            hidden_dim=budget_feat_dim,
            use_sinusoidal=False,
        )
        
        # Stage 1: Knob1 (tier)
        self.knob1_head = nn.Sequential(
            nn.Linear(budget_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # low, medium, high
        )
        
        # Stage 2: Knob2 & Knob3
        self.knob2_head = nn.Sequential(
            nn.Linear(budget_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),  # top_k: 4,6,8,10,12
        )
        
        self.knob3_head = nn.Sequential(
            nn.Linear(budget_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),  # num_blocks: 8,10,12,14,16
        )
        
        self.knob1_values = ["low", "medium", "high"]
        self.knob2_values = [4, 6, 8, 10, 12]
        self.knob3_values = [8, 10, 12, 14, 16]
    
    def forward(self, budget_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict knobs from budget.
        
        Args:
            budget_feat: (B, budget_feat_dim) budget features
        
        Returns:
            {
                'knob1_logits': (B, 3),
                'knob2_logits': (B, 5),
                'knob3_logits': (B, 5),
            }
        """
        knob1_logits = self.knob1_head(budget_feat)
        knob2_logits = self.knob2_head(budget_feat)
        knob3_logits = self.knob3_head(budget_feat)
        
        return {
            'knob1_logits': knob1_logits,
            'knob2_logits': knob2_logits,
            'knob3_logits': knob3_logits,
        }


class ProfilingDataset(Dataset):
    """Dataset from profiling results."""
    
    def __init__(
        self,
        profiling_results: List[Dict],
        budget_encoder: LatencyBudgetEncoder,
        device: str = "cuda",
    ):
        self.data = []
        self.budget_encoder = budget_encoder
        self.device = device
        
        # Map values to indices
        tier_map = {'low': 0, 'medium': 1, 'high': 2}
        top_k_map = {4: 0, 6: 1, 8: 2, 10: 3, 12: 4}
        num_blocks_map = {8: 0, 10: 1, 12: 2, 14: 3, 16: 4}
        
        for result in profiling_results:
            tier = result.get('tier', 'medium')
            top_k = result.get('top_k', 8)
            num_blocks = result.get('num_blocks', 16)
            latency_budget = result.get('T_total', 0.0)  # Use actual latency as budget
            
            if latency_budget <= 0:
                continue
            
            # Encode budget
            budget_tensor = torch.tensor([latency_budget], device=device)
            budget_feat = budget_encoder(budget_tensor).squeeze(0).cpu()
            
            self.data.append({
                'budget_feat': budget_feat,
                'tier_idx': tier_map.get(tier, 1),
                'top_k_idx': top_k_map.get(top_k, 2),
                'num_blocks_idx': num_blocks_map.get(num_blocks, 4),
                'accuracy': result.get('accuracy', 0.0),
                'latency': latency_budget,
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'budget_feat': item['budget_feat'],
            'tier_idx': torch.tensor(item['tier_idx'], dtype=torch.long),
            'top_k_idx': torch.tensor(item['top_k_idx'], dtype=torch.long),
            'num_blocks_idx': torch.tensor(item['num_blocks_idx'], dtype=torch.long),
            'accuracy': torch.tensor(item['accuracy'], dtype=torch.float32),
            'latency': torch.tensor(item['latency'], dtype=torch.float32),
        }


def train_supervised_controller(
    profiling_results_file: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-4,
    train_split: float = 0.8,
    seed: int = 42,
):
    """
    Train supervised learning controller.
    
    Args:
        profiling_results_file: Path to profiling results JSON file
        output_dir: Output directory
        device: Device to use
        batch_size: Batch size
        num_epochs: Number of epochs
        lr: Learning rate
        train_split: Train/val split ratio
        seed: Random seed
    """
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    log.info("Loading profiling results...")
    with open(profiling_results_file, 'r') as f:
        profiling_results = json.load(f)
    
    log.info(f"Loaded {len(profiling_results)} profiling results")
    
    # Initialize budget encoder
    budget_encoder = LatencyBudgetEncoder(hidden_dim=256, use_sinusoidal=False)
    budget_encoder.to(device)
    
    # Create dataset
    dataset = ProfilingDataset(profiling_results, budget_encoder, device)
    
    # Split train/val
    random.shuffle(dataset.data)
    split_idx = int(len(dataset) * train_split)
    train_dataset = ProfilingDataset(
        [dataset.data[i] for i in range(split_idx)],
        budget_encoder,
        device,
    )
    val_dataset = ProfilingDataset(
        [dataset.data[i] for i in range(split_idx, len(dataset))],
        budget_encoder,
        device,
    )
    
    log.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Create model
    model = SupervisedController(
        budget_feat_dim=256,
        hidden_dim=256,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_losses = []
        train_correct = {'knob1': 0, 'knob2': 0, 'knob3': 0}
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            budget_feat = batch['budget_feat'].to(device)
            tier_idx = batch['tier_idx'].to(device)
            top_k_idx = batch['top_k_idx'].to(device)
            num_blocks_idx = batch['num_blocks_idx'].to(device)
            
            # Forward
            logits = model(budget_feat)
            
            # Compute losses
            loss1 = criterion(logits['knob1_logits'], tier_idx)
            loss2 = criterion(logits['knob2_logits'], top_k_idx)
            loss3 = criterion(logits['knob3_logits'], num_blocks_idx)
            loss = loss1 + loss2 + loss3
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            train_losses.append(loss.item())
            pred1 = logits['knob1_logits'].argmax(dim=-1)
            pred2 = logits['knob2_logits'].argmax(dim=-1)
            pred3 = logits['knob3_logits'].argmax(dim=-1)
            
            train_correct['knob1'] += (pred1 == tier_idx).sum().item()
            train_correct['knob2'] += (pred2 == top_k_idx).sum().item()
            train_correct['knob3'] += (pred3 == num_blocks_idx).sum().item()
            train_total += tier_idx.shape[0]
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc1': f"{train_correct['knob1']/train_total:.3f}",
                'acc2': f"{train_correct['knob2']/train_total:.3f}",
                'acc3': f"{train_correct['knob3']/train_total:.3f}",
            })
        
        avg_train_loss = np.mean(train_losses)
        train_acc = {
            'knob1': train_correct['knob1'] / train_total,
            'knob2': train_correct['knob2'] / train_total,
            'knob3': train_correct['knob3'] / train_total,
        }
        log.info(f"Train Epoch {epoch+1}: loss={avg_train_loss:.4f}, acc={train_acc}")
        
        # Validate
        model.eval()
        val_losses = []
        val_correct = {'knob1': 0, 'knob2': 0, 'knob3': 0}
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                budget_feat = batch['budget_feat'].to(device)
                tier_idx = batch['tier_idx'].to(device)
                top_k_idx = batch['top_k_idx'].to(device)
                num_blocks_idx = batch['num_blocks_idx'].to(device)
                
                logits = model(budget_feat)
                
                loss1 = criterion(logits['knob1_logits'], tier_idx)
                loss2 = criterion(logits['knob2_logits'], top_k_idx)
                loss3 = criterion(logits['knob3_logits'], num_blocks_idx)
                loss = loss1 + loss2 + loss3
                
                val_losses.append(loss.item())
                pred1 = logits['knob1_logits'].argmax(dim=-1)
                pred2 = logits['knob2_logits'].argmax(dim=-1)
                pred3 = logits['knob3_logits'].argmax(dim=-1)
                
                val_correct['knob1'] += (pred1 == tier_idx).sum().item()
                val_correct['knob2'] += (pred2 == top_k_idx).sum().item()
                val_correct['knob3'] += (pred3 == num_blocks_idx).sum().item()
                val_total += num_blocks_idx.shape[0]
        
        avg_val_loss = np.mean(val_losses)
        val_acc = {
            'knob1': val_correct['knob1'] / val_total,
            'knob2': val_correct['knob2'] / val_total,
            'knob3': val_correct['knob3'] / val_total,
        }
        avg_val_acc = np.mean(list(val_acc.values()))
        
        log.info(f"Val Epoch {epoch+1}: loss={avg_val_loss:.4f}, acc={val_acc}, avg={avg_val_acc:.4f}")
        
        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'avg_val_acc': avg_val_acc,
            }
            torch.save(checkpoint, output_path / 'best_supervised_controller.pt')
            log.info(f"Saved best model (avg_val_acc={best_val_acc:.4f})")
    
    log.info(f"Training completed! Best model saved to {output_path / 'best_supervised_controller.pt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train supervised learning controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--profiling_results",
        type=str,
        required=True,
        help="Path to profiling results JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/controller/supervised",
        help="Output directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Train/val split ratio"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    train_supervised_controller(
        profiling_results_file=args.profiling_results,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        train_split=args.train_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()







