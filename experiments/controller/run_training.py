#!/usr/bin/env python3
"""
Python wrapper script for training joint controller.
Replaces run_training.sh with Python implementation and enables wandb by default.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Train Joint Controller (Stage1 + Stage2) with GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints",
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/joint_controller",
        help="Directory to save checkpoints and logs"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/core_exp_h100/5run_2000samples_w_new_importance_score",
        help="Directory containing core experiment results"
    )
    
    # Dataset
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=["text_vqa", "coco_2014_vqa", "okvqa"],
        help="Dataset names to load"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (will be auto-adjusted: 1 for single GPU, num_gpus for multi-GPU)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--stage1_lr_ratio",
        type=float,
        default=1.0,
        help="Stage1 learning rate ratio (relative to Stage2)"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=5,
        help="Group size for GRPO"
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
    
    # Optional
    parser.add_argument(
        "--importance_scores_file",
        type=str,
        default="results/layer_importance_scores_exp3_recommended.json",
        help="Path to JSON file containing importance scores for transformer blocks"
    )
    parser.add_argument(
        "--load_stage1_checkpoint",
        type=str,
        default=None,
        help="Optional: Path to pre-trained Stage1 checkpoint (for fine-tuning)"
    )
    
    # Multi-GPU
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="Use DataParallel for multi-GPU training (if multiple GPUs available)"
    )
    
    # Weights & Biases (default enabled)
    # Use --no_wandb to disable (default is enabled)
    parser.add_argument(
        "--no_wandb",
        action="store_false",
        dest="use_wandb",
        default=True,  # Default to True (wandb enabled by default)
        help="Disable Weights & Biases logging (default: enabled)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="molmo-controller",
        help="W&B project name (default: 'molmo-controller')"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name (optional)"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated from config)"
    )
    
    args = parser.parse_args()
    
    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        "experiments/controller/train_joint_controller.py",
        "--results_dir", args.results_dir,
        "--dataset_names"] + args.dataset_names + [
        "--model_path", args.model_path,
        "--output_dir", args.output_dir,
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs),
        "--lr", str(args.lr),
        "--stage1_lr_ratio", str(args.stage1_lr_ratio),
        "--group_size", str(args.group_size),
        "--device", args.device,
        "--seed", str(args.seed),
        "--importance_scores_file", args.importance_scores_file,
    ]
    
    # Add optional arguments
    if args.load_stage1_checkpoint:
        cmd.extend(["--load_stage1_checkpoint", args.load_stage1_checkpoint])
    
    if args.use_multi_gpu:
        cmd.append("--use_multi_gpu")
        print("✅ Multi-GPU training enabled")
    else:
        print("ℹ️  Single-GPU training (use --use_multi_gpu to enable multi-GPU)")
    
    # Add wandb arguments (default enabled)
    if args.use_wandb:
        cmd.append("--use_wandb")
        if args.wandb_project:
            cmd.extend(["--wandb_project", args.wandb_project])
        if args.wandb_entity:
            cmd.extend(["--wandb_entity", args.wandb_entity])
        if args.wandb_name:
            cmd.extend(["--wandb_name", args.wandb_name])
        print("✅ Weights & Biases logging enabled (default)")
        print(f"   Project: {args.wandb_project}")
        if args.wandb_entity:
            print(f"   Entity: {args.wandb_entity}")
        if args.wandb_name:
            print(f"   Run name: {args.wandb_name}")
        else:
            print("   Run name: auto-generated")
    else:
        print("⚠️  Weights & Biases logging disabled (use --use_wandb to enable)")
    
    # Print command
    print("\n" + "=" * 80)
    print("Training Command:")
    print("=" * 80)
    print(" ".join(cmd))
    print("=" * 80 + "\n")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()

