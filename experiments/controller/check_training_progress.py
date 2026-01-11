#!/usr/bin/env python3
"""
Quick script to check training progress from CSV files.
"""

import pandas as pd
import sys
from pathlib import Path
import argparse

def check_training_progress(output_dir: str = "checkpoints/joint_controller", num_epochs: int = 10):
    """Check training progress from CSV files."""
    output_path = Path(output_dir)
    
    # Load training history
    train_file = output_path / 'training_history.csv'
    val_file = output_path / 'validation_history.csv'
    
    if not train_file.exists():
        print(f"❌ Training history not found: {train_file}")
        print("   Training may not have started yet.")
        return
    
    # Load data
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file) if val_file.exists() else None
    
    print("=" * 80)
    print("Training Progress Summary")
    print("=" * 80)
    
    # Show recent epochs
    print(f"\n📊 Recent {min(num_epochs, len(df_train))} Epochs:")
    print("-" * 80)
    recent = df_train.tail(num_epochs)
    print(recent[['epoch', 'loss', 'reward_mean', 'accuracy_mean', 'budget_violation_rate']].to_string(index=False))
    
    # Analyze trends
    print(f"\n📈 Trend Analysis (Last {min(num_epochs, len(df_train))} Epochs):")
    print("-" * 80)
    
    if len(recent) >= 3:
        # Reward trend
        reward_start = recent['reward_mean'].iloc[0]
        reward_end = recent['reward_mean'].iloc[-1]
        reward_change = reward_end - reward_start
        reward_trend = "📈 Improving" if reward_change > 0 else "📉 Declining" if reward_change < -0.1 else "➡️ Stable"
        print(f"Reward: {reward_start:.4f} → {reward_end:.4f} ({reward_change:+.4f}) {reward_trend}")
        
        # Accuracy trend
        acc_start = recent['accuracy_mean'].iloc[0]
        acc_end = recent['accuracy_mean'].iloc[-1]
        acc_change = acc_end - acc_start
        acc_trend = "📈 Improving" if acc_change > 0.01 else "📉 Declining" if acc_change < -0.01 else "➡️ Stable"
        print(f"Accuracy: {acc_start:.4f} → {acc_end:.4f} ({acc_change:+.4f}) {acc_trend}")
        
        # Budget violation trend
        viol_start = recent['budget_violation_rate'].iloc[0]
        viol_end = recent['budget_violation_rate'].iloc[-1]
        viol_change = viol_end - viol_start
        viol_trend = "📉 Improving" if viol_change < -0.05 else "📈 Worsening" if viol_change > 0.05 else "➡️ Stable"
        print(f"Budget Violation: {viol_start:.4f} → {viol_end:.4f} ({viol_change:+.4f}) {viol_trend}")
        
        # Loss trend
        loss_start = recent['loss'].iloc[0]
        loss_end = recent['loss'].iloc[-1]
        loss_change = loss_end - loss_start
        loss_trend = "📉 Improving" if abs(loss_change) < 0.1 or (loss_change < 0 and loss_start > 0) else "📈 Worsening" if loss_change > 0.5 else "➡️ Stable"
        print(f"Loss: {loss_start:.4f} → {loss_end:.4f} ({loss_change:+.4f}) {loss_trend}")
    else:
        print("⚠️  Not enough epochs to analyze trends (need at least 3)")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    print("-" * 80)
    
    if len(df_train) >= 5:
        # Check if training is improving
        recent_5 = df_train.tail(5)
        reward_improving = recent_5['reward_mean'].iloc[-1] > recent_5['reward_mean'].iloc[0]
        acc_improving = recent_5['accuracy_mean'].iloc[-1] > recent_5['accuracy_mean'].iloc[0]
        viol_decreasing = recent_5['budget_violation_rate'].iloc[-1] < recent_5['budget_violation_rate'].iloc[0]
        
        if reward_improving and acc_improving and viol_decreasing:
            print("✅ Training is progressing well!")
            print("   - Reward is improving")
            print("   - Accuracy is improving")
            print("   - Budget violation rate is decreasing")
        elif reward_improving or acc_improving:
            print("⚠️  Training is progressing, but some metrics need attention")
            if not reward_improving:
                print("   - Reward is not improving (check reward function)")
            if not acc_improving:
                print("   - Accuracy is not improving (check metadata/answers)")
            if not viol_decreasing:
                print("   - Budget violation rate is not decreasing (check latency measurement)")
        else:
            print("❌ Training may need attention")
            print("   - Consider checking hyperparameters or training setup")
    else:
        print("⚠️  Too early to assess (need at least 5 epochs)")
    
    # Validation metrics (if available)
    if df_val is not None and len(df_val) > 0:
        print(f"\n📊 Validation Metrics (Last {min(num_epochs, len(df_val))} Epochs):")
        print("-" * 80)
        recent_val = df_val.tail(num_epochs)
        print(recent_val[['epoch', 'reward_mean', 'accuracy_mean', 'budget_violation_rate']].to_string(index=False))
    
    # File locations
    print(f"\n📁 Log Files:")
    print("-" * 80)
    print(f"Training CSV: {train_file}")
    if df_val is not None:
        print(f"Validation CSV: {val_file}")
    print(f"TensorBoard: {output_path / 'tensorboard'}")
    print(f"  View with: tensorboard --logdir={output_path / 'tensorboard'}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check training progress from CSV files")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/joint_controller",
        help="Directory containing training history CSV files"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of recent epochs to show"
    )
    
    args = parser.parse_args()
    check_training_progress(args.output_dir, args.num_epochs)

