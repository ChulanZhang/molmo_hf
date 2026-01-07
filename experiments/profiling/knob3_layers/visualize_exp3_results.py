#!/usr/bin/env python3
"""
Generate visualizations and LaTeX tables for EXP3 results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_all_results(base_dir: Path) -> List[Dict]:
    """Load all importance comparison results."""
    datasets = [
        'coco_2014_vqa', 'text_vqa', 'science_qa_img', 'okvqa', 
        'st_qa', 'doc_qa', 'tally_qa', 'mmmu', 'coco_caption'
    ]
    
    results = []
    for dataset in datasets:
        json_file = base_dir / dataset / f'importance_comparison_{dataset}.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append({
                    'dataset': dataset,
                    'correlation': data.get('spearman_correlation', 0),
                    'p_value': data.get('p_value', 1),
                    'is_consistent': data.get('is_consistent', False),
                    'train_scores': data.get('train_scores', {}),
                    'validation_scores': data.get('validation_scores', {}),
                })
    
    return sorted(results, key=lambda x: x['correlation'], reverse=True)

def plot_correlation_bar_chart(results: List[Dict], output_path: Path):
    """Plot bar chart of Spearman correlations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    datasets = [r['dataset'] for r in results]
    correlations = [r['correlation'] for r in results]
    colors = ['#2ecc71' if r['is_consistent'] else '#e74c3c' for r in results]
    
    bars = ax.barh(datasets, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add threshold line
    ax.axvline(x=0.9, color='orange', linestyle='--', linewidth=2, label='Consistency Threshold (0.9)')
    
    # Add value labels
    for i, (dataset, corr) in enumerate(zip(datasets, correlations)):
        ax.text(corr + 0.01, i, f'{corr:.4f}', va='center', fontweight='bold')
    
    ax.set_xlabel('Spearman Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('Train vs Validation Importance Score Consistency', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_scatter_comparison(results: List[Dict], output_path: Path, top_n: int = 6):
    """Plot scatter plots comparing train vs validation scores for top datasets."""
    top_results = results[:top_n]
    
    n_cols = 3
    n_rows = (len(top_results) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, result in enumerate(top_results):
        ax = axes[idx]
        
        train_scores = result['train_scores']
        val_scores = result['validation_scores']
        
        # Extract block indices and scores
        blocks = sorted([int(k) for k in train_scores.keys()])
        train_vals = [train_scores[str(b)] for b in blocks]
        val_vals = [val_scores[str(b)] for b in blocks]
        
        # Plot scatter
        ax.scatter(train_vals, val_vals, s=100, alpha=0.6, edgecolors='black', linewidth=1)
        
        # Add diagonal line
        min_val = min(min(train_vals), min(val_vals))
        max_val = max(max(train_vals), max(val_vals))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2)
        
        # Add block labels
        for b, tx, ty in zip(blocks, train_vals, val_vals):
            ax.annotate(f'B{b}', (tx, ty), fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Train Importance Score', fontsize=10)
        ax.set_ylabel('Validation Importance Score', fontsize=10)
        ax.set_title(f"{result['dataset']}\nρ={result['correlation']:.4f}", fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(top_results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_importance_heatmap(results: List[Dict], output_dir: Path, top_n: int = 6):
    """Plot heatmap of importance scores across datasets (Train and Validation as separate figures)."""
    top_results = results[:top_n]
    
    # Prepare data
    datasets = [r['dataset'] for r in top_results]
    blocks = list(range(16))
    
    # Prepare train and validation data
    train_data = []
    val_data = []
    for result in top_results:
        train_scores = result['train_scores']
        val_scores = result['validation_scores']
        train_row = [train_scores.get(str(b), 0) for b in blocks]
        val_row = [val_scores.get(str(b), 0) for b in blocks]
        train_data.append(train_row)
        val_data.append(val_row)
    
    train_data = np.array(train_data)
    val_data = np.array(val_data)
    
    # Find common scale for both heatmaps
    vmin = min(train_data.min(), val_data.min())
    vmax = max(train_data.max(), val_data.max())
    
    # Plot train set heatmap
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    im1 = ax1.imshow(train_data, cmap='YlOrRd', aspect='auto', interpolation='nearest', 
                     vmin=vmin, vmax=vmax)
    
    # Set ticks for train set
    ax1.set_xticks(range(16))
    ax1.set_xticklabels([f'Block {i}' for i in range(16)])
    ax1.set_yticks(range(len(datasets)))
    ax1.set_yticklabels(datasets)
    
    # Add text annotations for train set
    for i in range(len(datasets)):
        for j in range(16):
            text = ax1.text(j, i, f'{train_data[i, j]:.3f}', ha="center", va="center", 
                          color="black" if train_data[i, j] < vmax * 0.6 else "white",
                          fontsize=8)
    
    ax1.set_title('Block Importance Scores Across Datasets (Train Set)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Transformer Block Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Importance Score', rotation=270, labelpad=20)
    
    train_output_path = output_dir / 'importance_heatmap_train.png'
    plt.tight_layout()
    plt.savefig(train_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {train_output_path}")
    
    # Plot validation set heatmap
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    im2 = ax2.imshow(val_data, cmap='YlOrRd', aspect='auto', interpolation='nearest',
                     vmin=vmin, vmax=vmax)
    
    # Set ticks for validation set
    ax2.set_xticks(range(16))
    ax2.set_xticklabels([f'Block {i}' for i in range(16)])
    ax2.set_yticks(range(len(datasets)))
    ax2.set_yticklabels(datasets)
    
    # Add text annotations for validation set
    for i in range(len(datasets)):
        for j in range(16):
            text = ax2.text(j, i, f'{val_data[i, j]:.3f}', ha="center", va="center", 
                          color="black" if val_data[i, j] < vmax * 0.6 else "white",
                          fontsize=8)
    
    ax2.set_title('Block Importance Scores Across Datasets (Validation Set)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Transformer Block Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Importance Score', rotation=270, labelpad=20)
    
    val_output_path = output_dir / 'importance_heatmap_validation.png'
    plt.tight_layout()
    plt.savefig(val_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {val_output_path}")

def generate_latex_table(results: List[Dict], output_path: Path):
    """Generate LaTeX table of results."""
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Train vs Validation Importance Score Consistency Analysis}
\\label{tab:exp3_consistency}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Dataset} & \\textbf{Spearman $\\rho$} & \\textbf{P-value} & \\textbf{Consistent} \\\\
\\midrule
"""
    
    for r in results:
        dataset = r['dataset'].replace('_', '\\_')
        correlation = f"{r['correlation']:.4f}"
        p_value = f"{r['p_value']:.2e}"
        consistent = "\\checkmark" if r['is_consistent'] else "$\\times$"
        
        latex += f"{dataset} & {correlation} & {p_value} & {consistent} \\\\\n"
    
    # Add summary row
    consistent_count = sum(1 for r in results if r['is_consistent'])
    avg_corr = sum(r['correlation'] for r in results) / len(results)
    latex += f"""\\midrule
\\textbf{{Summary}} & \\textbf{{{avg_corr:.4f}}} & -- & \\textbf{{{consistent_count}/{len(results)}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {output_path}")

def generate_detailed_latex_table(results: List[Dict], output_path: Path):
    """Generate detailed LaTeX table with block importance rankings."""
    latex = """\\begin{table*}[htbp]
\\centering
\\caption{Block Importance Rankings: Top 5 Least Important Blocks}
\\label{tab:exp3_block_importance}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{l|ccccc}
\\toprule
\\textbf{Dataset} & \\textbf{Rank 1} & \\textbf{Rank 2} & \\textbf{Rank 3} & \\textbf{Rank 4} & \\textbf{Rank 5} \\\\
\\midrule
"""
    
    for r in results:
        dataset = r['dataset'].replace('_', '\\_')
        train_scores = r['train_scores']
        
        # Sort by importance (ascending - least important first)
        sorted_blocks = sorted(train_scores.items(), key=lambda x: x[1])
        top5 = sorted_blocks[:5]
        
        blocks_str = " & ".join([f"Block {int(b)} ({float(s):.4f})" for b, s in top5])
        latex += f"{dataset} & {blocks_str} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}%
}
\\end{table*}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {output_path}")

def main():
    base_dir = Path("results/profiling/exp3_importance_comparison")
    output_dir = Path("results/profiling/exp3_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_all_results(base_dir)
    print(f"Loaded {len(results)} datasets")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_correlation_bar_chart(results, output_dir / "correlation_bar_chart.png")
    plot_scatter_comparison(results, output_dir / "scatter_comparison.png", top_n=6)
    plot_importance_heatmap(results, output_dir, top_n=6)
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex_table(results, output_dir / "consistency_table.tex")
    generate_detailed_latex_table(results, output_dir / "block_importance_table.tex")
    
    print(f"\n✅ All visualizations and tables saved to: {output_dir}")

if __name__ == "__main__":
    main()

