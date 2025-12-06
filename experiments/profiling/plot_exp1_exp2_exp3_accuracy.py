#!/usr/bin/env python3
"""
Plot script for Exp1, Exp2, Exp3 Accuracy Results
Visualizes accuracy vs max_crops, top_k, and num_active_blocks
Also includes exp3 sensitivity results (importance scores and pruning accuracy)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_exp1_results(results_dir: Path) -> List[Dict]:
    """Load exp1 accuracy results (max_crops vs accuracy)."""
    results = []
    
    # Look for individual result files
    pattern = str(results_dir / "exp1_accuracy_results_max_crops_*.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        # Try main results file
        main_file = results_dir / "exp1_accuracy_results.json"
        if main_file.exists():
            with open(main_file, 'r') as f:
                data = json.load(f)
                if "summary" in data:
                    return data["summary"]
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if "summary" in data and len(data["summary"]) > 0:
                    result = data["summary"][0]
                    results.append({
                        "max_crops": result.get("max_crops"),
                        "accuracy": result.get("accuracy", 0.0),
                        "num_samples": result.get("num_samples", 0),
                    })
        except Exception as e:
            log.warning(f"Failed to load {file_path}: {e}")
            continue
    
    return sorted(results, key=lambda x: x["max_crops"])


def load_exp2_results(results_dir: Path) -> List[Dict]:
    """Load exp2 accuracy results (top_k vs accuracy)."""
    results = []
    
    # Look for individual result files
    pattern = str(results_dir / "exp2_accuracy_results_top_k_*.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        # Try main results file
        main_file = results_dir / "exp2_accuracy_results.json"
        if main_file.exists():
            with open(main_file, 'r') as f:
                data = json.load(f)
                if "summary" in data:
                    return data["summary"]
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if "summary" in data and len(data["summary"]) > 0:
                    result = data["summary"][0]
                    results.append({
                        "top_k": result.get("top_k"),
                        "accuracy": result.get("accuracy", 0.0),
                        "num_samples": result.get("num_samples", 0),
                    })
        except Exception as e:
            log.warning(f"Failed to load {file_path}: {e}")
            continue
    
    return sorted(results, key=lambda x: x["top_k"])


def load_exp3_results(results_dir: Path) -> List[Dict]:
    """Load exp3 accuracy results (num_active_blocks vs accuracy)."""
    results = []
    
    # Look for individual result files
    pattern = str(results_dir / "exp3_accuracy_results_blocks_*.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        # Try main results file
        main_file = results_dir / "exp3_accuracy_results.json"
        if main_file.exists():
            with open(main_file, 'r') as f:
                data = json.load(f)
                if "summary" in data:
                    return data["summary"]
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if "summary" in data and len(data["summary"]) > 0:
                    result = data["summary"][0]
                    results.append({
                        "num_active_blocks": result.get("num_active_blocks"),
                        "accuracy": result.get("accuracy", 0.0),
                        "num_samples": result.get("num_samples", 0),
                        "active_block_indices": result.get("active_block_indices", []),
                    })
        except Exception as e:
            log.warning(f"Failed to load {file_path}: {e}")
            continue
    
    return sorted(results, key=lambda x: x["num_active_blocks"])


def load_exp3_sensitivity_results(results_dir: Path) -> Tuple[List[Dict], Optional[Dict]]:
    """Load exp3 sensitivity results (importance scores and pruning accuracy)."""
    results = []
    importance_scores = None
    
    # Load main results file
    main_file = results_dir / "exp3_accuracy_sensitivity_results.json"
    if main_file.exists():
        with open(main_file, 'r') as f:
            data = json.load(f)
            if "summary" in data:
                results = data["summary"]
            if "importance_scores" in data:
                importance_scores = {int(k): float(v) for k, v in data["importance_scores"].items()}
    
    # Also try loading from separate importance scores file
    importance_file = results_dir / "layer_importance_scores.json"
    if importance_file.exists() and importance_scores is None:
        with open(importance_file, 'r') as f:
            importance_scores = {int(k): float(v) for k, v in json.load(f).items()}
    
    # Sort results by num_active_blocks
    results = sorted(results, key=lambda x: x.get("num_active_blocks", 0))
    
    return results, importance_scores


def plot_exp1_accuracy(exp1_results: List[Dict], output_dir: Path):
    """Plot Exp1: max_crops vs accuracy."""
    if not exp1_results:
        log.warning("No exp1 results to plot")
        return
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    max_crops = [r["max_crops"] for r in exp1_results]
    accuracies = [r["accuracy"] for r in exp1_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(max_crops, accuracies, 'o-', linewidth=2.5, markersize=10, color='#1F77B4')
    plt.xlabel("Max Crops", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title("Exp1: Accuracy vs Max Crops", fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)
    
    # Add value labels on points
    for x, y in zip(max_crops, accuracies):
        plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "exp1_accuracy_vs_max_crops.png", dpi=300, bbox_inches="tight")
    log.info(f"Saved Exp1 plot to {fig_dir / 'exp1_accuracy_vs_max_crops.png'}")
    plt.close()
    
    log.info(f"Exp1 Results: {len(exp1_results)} configurations")
    for r in exp1_results:
        log.info(f"  max_crops={r['max_crops']}: accuracy={r['accuracy']:.4f}")


def plot_exp2_accuracy(exp2_results: List[Dict], output_dir: Path):
    """Plot Exp2: top_k vs accuracy."""
    if not exp2_results:
        log.warning("No exp2 results to plot")
        return
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    top_k = [r["top_k"] for r in exp2_results]
    accuracies = [r["accuracy"] for r in exp2_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(top_k, accuracies, 'o-', linewidth=2.5, markersize=10, color='#FF7F0E')
    plt.xlabel("Top-K", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title("Exp2: Accuracy vs Top-K", fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)
    
    # Add value labels on points
    for x, y in zip(top_k, accuracies):
        plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "exp2_accuracy_vs_top_k.png", dpi=300, bbox_inches="tight")
    log.info(f"Saved Exp2 plot to {fig_dir / 'exp2_accuracy_vs_top_k.png'}")
    plt.close()
    
    log.info(f"Exp2 Results: {len(exp2_results)} configurations")
    for r in exp2_results:
        log.info(f"  top_k={r['top_k']}: accuracy={r['accuracy']:.4f}")


def plot_exp3_accuracy(exp3_results: List[Dict], output_dir: Path):
    """Plot Exp3: num_active_blocks vs accuracy."""
    if not exp3_results:
        log.warning("No exp3 results to plot")
        return
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    num_blocks = [r["num_active_blocks"] for r in exp3_results]
    accuracies = [r["accuracy"] for r in exp3_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_blocks, accuracies, 'o-', linewidth=2.5, markersize=10, color='#2CA02C')
    plt.xlabel("Number of Active Blocks", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title("Exp3: Accuracy vs Number of Active Blocks", fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)
    
    # Add value labels on points
    for x, y in zip(num_blocks, accuracies):
        plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "exp3_accuracy_vs_num_blocks.png", dpi=300, bbox_inches="tight")
    log.info(f"Saved Exp3 plot to {fig_dir / 'exp3_accuracy_vs_num_blocks.png'}")
    plt.close()
    
    log.info(f"Exp3 Results: {len(exp3_results)} configurations")
    for r in exp3_results:
        log.info(f"  num_active_blocks={r['num_active_blocks']}: accuracy={r['accuracy']:.4f}")


def plot_exp3_sensitivity(exp3_sensitivity_results: List[Dict], 
                         importance_scores: Optional[Dict], 
                         output_dir: Path):
    """Plot Exp3 Sensitivity: importance scores and pruning accuracy."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Layer Importance Scores
    if importance_scores:
        layers = sorted(importance_scores.keys())
        scores = [importance_scores[l] for l in layers]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(layers, scores, color='#9467BD', edgecolor='black', linewidth=1.0)
        plt.xlabel("Layer Index", fontsize=16)
        plt.ylabel("Importance Score (ΔAccuracy)", fontsize=16)
        plt.title("Exp3 Sensitivity: Layer Importance Scores", fontsize=18)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tick_params(labelsize=14)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(fig_dir / "exp3_sensitivity_layer_importance.png", dpi=300, bbox_inches="tight")
        log.info(f"Saved importance scores plot to {fig_dir / 'exp3_sensitivity_layer_importance.png'}")
        plt.close()
        
        log.info("Layer Importance Scores (sorted by importance):")
        sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        for layer, score in sorted_layers:
            log.info(f"  Layer {layer}: {score:.4f}")
    
    # Plot 2: Pruning Accuracy (num_active_blocks vs accuracy)
    if exp3_sensitivity_results:
        num_blocks = [r["num_active_blocks"] for r in exp3_sensitivity_results]
        accuracies = [r["accuracy"] for r in exp3_sensitivity_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(num_blocks, accuracies, 'o-', linewidth=2.5, markersize=10, color='#D62728')
        plt.xlabel("Number of Active Blocks", fontsize=16)
        plt.ylabel("Accuracy", fontsize=16)
        plt.title("Exp3 Sensitivity: Accuracy vs Number of Active Blocks (Importance-based Pruning)", fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=14)
        
        # Add value labels on points
        for x, y in zip(num_blocks, accuracies):
            plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(fig_dir / "exp3_sensitivity_pruning_accuracy.png", dpi=300, bbox_inches="tight")
        log.info(f"Saved pruning accuracy plot to {fig_dir / 'exp3_sensitivity_pruning_accuracy.png'}")
        plt.close()
        
        log.info(f"Exp3 Sensitivity Pruning Results: {len(exp3_sensitivity_results)} configurations")
        for r in exp3_sensitivity_results:
            log.info(f"  num_active_blocks={r['num_active_blocks']}: accuracy={r['accuracy']:.4f}")
    
    # Plot 3: Combined comparison (Exp3 vs Exp3 Sensitivity)
    # This will be added if both datasets are available


def plot_combined_comparison(exp3_results: List[Dict], 
                            exp3_sensitivity_results: List[Dict],
                            output_dir: Path):
    """Plot combined comparison of Exp3 and Exp3 Sensitivity."""
    if not exp3_results or not exp3_sensitivity_results:
        return
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Exp3 (sequential activation)
    exp3_blocks = [r["num_active_blocks"] for r in exp3_results]
    exp3_acc = [r["accuracy"] for r in exp3_results]
    
    # Exp3 Sensitivity (importance-based pruning)
    sens_blocks = [r["num_active_blocks"] for r in exp3_sensitivity_results]
    sens_acc = [r["accuracy"] for r in exp3_sensitivity_results]
    
    plt.figure(figsize=(12, 7))
    plt.plot(exp3_blocks, exp3_acc, 'o-', linewidth=2.5, markersize=10, 
            color='#2CA02C', label='Exp3 (Sequential Activation)', alpha=0.8)
    plt.plot(sens_blocks, sens_acc, 's-', linewidth=2.5, markersize=10, 
            color='#D62728', label='Exp3 Sensitivity (Importance-based Pruning)', alpha=0.8)
    
    plt.xlabel("Number of Active Blocks", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title("Exp3: Sequential vs Importance-based Layer Selection", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "exp3_comparison_sequential_vs_importance.png", dpi=300, bbox_inches="tight")
    log.info(f"Saved comparison plot to {fig_dir / 'exp3_comparison_sequential_vs_importance.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Exp1, Exp2, Exp3 Accuracy Results")
    parser.add_argument("--results_dir", type=str, default="./results/profiling",
                       help="Base directory containing exp1_accuracy, exp2_accuracy, exp3_accuracy, exp3_accuracy_sensitivity")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots (default: same as results_dir)")
    
    args = parser.parse_args()
    
    results_base = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_base
    
    log.info("=" * 80)
    log.info("Plotting Exp1, Exp2, Exp3 Accuracy Results")
    log.info("=" * 80)
    
    # Load results
    exp1_dir = results_base / "exp1_accuracy"
    exp2_dir = results_base / "exp2_accuracy"
    exp3_dir = results_base / "exp3_accuracy"
    exp3_sens_dir = results_base / "exp3_accuracy_sensitivity"
    
    exp1_results = []
    exp2_results = []
    exp3_results = []
    exp3_sensitivity_results = []
    importance_scores = None
    
    if exp1_dir.exists():
        log.info(f"Loading Exp1 results from {exp1_dir}")
        exp1_results = load_exp1_results(exp1_dir)
        log.info(f"Loaded {len(exp1_results)} Exp1 configurations")
    else:
        log.warning(f"Exp1 results directory not found: {exp1_dir}")
    
    if exp2_dir.exists():
        log.info(f"Loading Exp2 results from {exp2_dir}")
        exp2_results = load_exp2_results(exp2_dir)
        log.info(f"Loaded {len(exp2_results)} Exp2 configurations")
    else:
        log.warning(f"Exp2 results directory not found: {exp2_dir}")
    
    if exp3_dir.exists():
        log.info(f"Loading Exp3 results from {exp3_dir}")
        exp3_results = load_exp3_results(exp3_dir)
        log.info(f"Loaded {len(exp3_results)} Exp3 configurations")
    else:
        log.warning(f"Exp3 results directory not found: {exp3_dir}")
    
    if exp3_sens_dir.exists():
        log.info(f"Loading Exp3 Sensitivity results from {exp3_sens_dir}")
        exp3_sensitivity_results, importance_scores = load_exp3_sensitivity_results(exp3_sens_dir)
        log.info(f"Loaded {len(exp3_sensitivity_results)} Exp3 Sensitivity configurations")
        if importance_scores:
            log.info(f"Loaded importance scores for {len(importance_scores)} layers")
    else:
        log.warning(f"Exp3 Sensitivity results directory not found: {exp3_sens_dir}")
    
    # Plot results
    log.info("=" * 80)
    log.info("Generating plots...")
    log.info("=" * 80)
    
    plot_exp1_accuracy(exp1_results, output_dir)
    plot_exp2_accuracy(exp2_results, output_dir)
    plot_exp3_accuracy(exp3_results, output_dir)
    plot_exp3_sensitivity(exp3_sensitivity_results, importance_scores, output_dir)
    plot_combined_comparison(exp3_results, exp3_sensitivity_results, output_dir)
    
    log.info("=" * 80)
    log.info("Plotting complete! All figures saved to:")
    log.info(f"  {output_dir / 'figures'}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()


