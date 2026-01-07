#!/usr/bin/env python3
"""
Analyze why doc_qa, coco_2014_vqa, and mmmu have low correlation between train and validation sets.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def load_dataset_result(dataset_name: str) -> Dict:
    """Load importance comparison result for a dataset."""
    json_file = Path(f"results/profiling/exp3_importance_comparison/{dataset_name}/importance_comparison_{dataset_name}.json")
    with open(json_file, 'r') as f:
        return json.load(f)

def calculate_statistics(scores: Dict[str, float]) -> Dict:
    """Calculate statistics for importance scores."""
    values = list(scores.values())
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'range': np.max(values) - np.min(values),
        'block0_importance': scores.get('0', 0),
        'non_zero_count': sum(1 for v in values if v != 0),
        'zero_count': sum(1 for v in values if v == 0),
    }

def analyze_differences(train_scores: Dict[str, float], val_scores: Dict[str, float]) -> Dict:
    """Analyze differences between train and validation scores."""
    differences = {}
    for block_idx in range(16):
        block_str = str(block_idx)
        train_val = train_scores.get(block_str, 0)
        val_val = val_scores.get(block_str, 0)
        diff = train_val - val_val
        abs_diff = abs(diff)
        differences[block_idx] = {
            'train': train_val,
            'val': val_val,
            'diff': diff,
            'abs_diff': abs_diff,
            'relative_diff': abs_diff / max(abs(train_val), abs(val_val), 1e-10)
        }
    
    return differences

def main():
    datasets_to_analyze = ['doc_qa', 'coco_2014_vqa', 'mmmu']
    
    # Dataset characteristics (from documentation)
    dataset_info = {
        'doc_qa': {
            'sample_count': 5349,
            'task_type': 'Document VQA',
            'answer_length_mean': 4.6,
            'answer_length_max': 28,
            'data_type': 'Document images (scanned documents, forms, etc.)',
        },
        'coco_2014_vqa': {
            'sample_count': 214354,
            'task_type': 'Visual Question Answering',
            'answer_length_mean': 1.4,
            'answer_length_max': 12,
            'data_type': 'Natural images (COCO 2014 validation images)',
        },
        'mmmu': {
            'sample_count': 900,
            'task_type': 'Multi-modal Multi-discipline Understanding',
            'answer_length_mean': 1.1,
            'answer_length_max': 15,
            'data_type': 'Mixed (diagrams, charts, scientific images)',
            'note': 'Uses "dev" split for train, "validation" split for validation',
        }
    }
    
    print("=" * 80)
    print("Analysis: Low Correlation Datasets")
    print("=" * 80)
    print()
    
    for dataset_name in datasets_to_analyze:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        # Load results
        result = load_dataset_result(dataset_name)
        correlation = result['spearman_correlation']
        p_value = result['p_value']
        train_scores = result['train_scores']
        val_scores = result['validation_scores']
        
        # Dataset characteristics
        info = dataset_info[dataset_name]
        print("Dataset Characteristics:")
        print(f"  - Sample count: {info['sample_count']:,}")
        print(f"  - Task type: {info['task_type']}")
        print(f"  - Data type: {info['data_type']}")
        print(f"  - Answer length: mean={info['answer_length_mean']:.1f}, max={info['answer_length_max']}")
        if 'note' in info:
            print(f"  - Note: {info['note']}")
        print()
        
        # Correlation info
        print(f"Correlation Analysis:")
        print(f"  - Spearman correlation: {correlation:.4f}")
        print(f"  - P-value: {p_value:.2e}")
        print(f"  - Consistent: {'No' if correlation < 0.9 else 'Yes'}")
        print()
        
        # Statistics
        train_stats = calculate_statistics(train_scores)
        val_stats = calculate_statistics(val_scores)
        
        print("Importance Score Statistics:")
        print(f"  Train set:")
        print(f"    - Mean: {train_stats['mean']:.4f}")
        print(f"    - Std: {train_stats['std']:.4f}")
        print(f"    - Range: [{train_stats['min']:.4f}, {train_stats['max']:.4f}]")
        print(f"    - Block 0 importance: {train_stats['block0_importance']:.4f}")
        print(f"    - Zero scores: {train_stats['zero_count']}")
        print(f"  Validation set:")
        print(f"    - Mean: {val_stats['mean']:.4f}")
        print(f"    - Std: {val_stats['std']:.4f}")
        print(f"    - Range: [{val_stats['min']:.4f}, {val_stats['max']:.4f}]")
        print(f"    - Block 0 importance: {val_stats['block0_importance']:.4f}")
        print(f"    - Zero scores: {val_stats['zero_count']}")
        print()
        
        # Differences
        differences = analyze_differences(train_scores, val_scores)
        
        # Find blocks with largest differences
        sorted_diffs = sorted(differences.items(), key=lambda x: x[1]['abs_diff'], reverse=True)
        top5_diffs = sorted_diffs[:5]
        
        print("Top 5 Blocks with Largest Train-Val Differences:")
        for block_idx, diff_info in top5_diffs:
            print(f"  Block {block_idx}:")
            print(f"    Train: {diff_info['train']:.4f}, Val: {diff_info['val']:.4f}")
            print(f"    Absolute diff: {diff_info['abs_diff']:.4f}")
            print(f"    Relative diff: {diff_info['relative_diff']:.2%}")
        print()
        
        # Block 0 analysis
        block0_diff = differences[0]
        print("Block 0 Analysis (Most Important Block):")
        print(f"  Train importance: {block0_diff['train']:.4f}")
        print(f"  Val importance: {block0_diff['val']:.4f}")
        print(f"  Difference: {block0_diff['diff']:.4f} ({block0_diff['relative_diff']:.2%})")
        print()
        
        # Check for negative values
        train_neg = [b for b, s in train_scores.items() if float(s) < 0]
        val_neg = [b for b, s in val_scores.items() if float(s) < 0]
        if train_neg or val_neg:
            print("⚠️  Negative Importance Scores Found:")
            if train_neg:
                print(f"  Train set blocks with negative scores: {train_neg}")
            if val_neg:
                print(f"  Validation set blocks with negative scores: {val_neg}")
            print("  (Negative scores indicate removing the block actually improves accuracy)")
            print()
        
        # Ranking comparison
        train_ranking = sorted([(int(k), float(v)) for k, v in train_scores.items()], 
                              key=lambda x: x[1])
        val_ranking = sorted([(int(k), float(v)) for k, v in val_scores.items()], 
                            key=lambda x: x[1])
        
        train_top5 = set([b for b, _ in train_ranking[:5]])
        val_top5 = set([b for b, _ in val_ranking[:5]])
        overlap = train_top5 & val_top5
        
        print("Ranking Comparison (Top 5 Least Important Blocks):")
        print(f"  Train top 5: {sorted(train_top5)}")
        print(f"  Val top 5: {sorted(val_top5)}")
        print(f"  Overlap: {len(overlap)}/5 blocks ({sorted(overlap)})")
        print()
        
        # Potential causes
        print("Potential Causes of Low Correlation:")
        causes = []
        
        # Sample size
        if info['sample_count'] < 2000:
            causes.append(f"❌ Small sample size ({info['sample_count']:,}) - may lead to statistical instability")
        elif info['sample_count'] < 10000:
            causes.append(f"⚠️  Moderate sample size ({info['sample_count']:,}) - may have some variance")
        else:
            causes.append(f"✅ Large sample size ({info['sample_count']:,}) - should be statistically stable")
        
        # Block 0 difference
        if abs(block0_diff['diff']) > 0.1:
            causes.append(f"❌ Large Block 0 difference ({block0_diff['diff']:.4f}) - suggests fundamental train/val split differences")
        elif abs(block0_diff['diff']) > 0.05:
            causes.append(f"⚠️  Moderate Block 0 difference ({block0_diff['diff']:.4f})")
        else:
            causes.append(f"✅ Small Block 0 difference ({block0_diff['diff']:.4f})")
        
        # Zero scores
        if train_stats['zero_count'] > 0 or val_stats['zero_count'] > 0:
            causes.append(f"⚠️  Zero importance scores found (train: {train_stats['zero_count']}, val: {val_stats['zero_count']}) - may indicate insufficient samples or blocks with truly zero impact")
        
        # Negative scores
        if train_neg or val_neg:
            causes.append(f"⚠️  Negative importance scores found - suggests removing blocks can improve accuracy (possible overfitting or dataset-specific effects)")
        
        # Variance
        if abs(train_stats['std'] - val_stats['std']) > 0.05:
            causes.append(f"⚠️  Different variance (train std: {train_stats['std']:.4f}, val std: {val_stats['std']:.4f}) - suggests different score distributions")
        
        # Ranking overlap
        if len(overlap) < 3:
            causes.append(f"❌ Low ranking overlap ({len(overlap)}/5) - train and val disagree on which blocks are least important")
        elif len(overlap) < 4:
            causes.append(f"⚠️  Moderate ranking overlap ({len(overlap)}/5)")
        else:
            causes.append(f"✅ High ranking overlap ({len(overlap)}/5)")
        
        for cause in causes:
            print(f"  {cause}")
        print()
    
    # Summary and recommendations
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    print("Key Findings:")
    print()
    print("1. MMMU (Correlation: 0.2558):")
    print("   - Smallest dataset (900 samples) - major factor")
    print("   - Uses different splits (dev vs validation) - may have distribution shift")
    print("   - Large Block 0 difference (0.16) - fundamental split differences")
    print("   - Many zero scores in train set - insufficient samples for reliable estimation")
    print("   - Negative scores in validation - suggests dataset-specific effects")
    print("   → Recommendation: Use more samples or combine dev+validation for analysis")
    print()
    
    print("2. COCO 2014 VQA (Correlation: 0.8374):")
    print("   - Largest dataset (214K samples) - sample size is NOT the issue")
    print("   - Very short answers (mean 1.4 tokens) - task may be too simple")
    print("   - Moderate Block 0 difference (0.03) - acceptable")
    print("   - Good ranking overlap - trends are similar")
    print("   → Recommendation: Correlation is acceptable; may reflect task-specific patterns")
    print()
    
    print("3. DOC_QA (Correlation: 0.8853):")
    print("   - Moderate sample size (5,349) - should be sufficient")
    print("   - Document images - different from natural images")
    print("   - Longer answers (mean 4.6 tokens) - more complex task")
    print("   - Large Block 0 difference (0.12) - main issue")
    print("   - Different block importance patterns (blocks 6,7 more important in val)")
    print("   → Recommendation: May reflect train/val distribution differences in document types")
    print()
    
    print("General Recommendations:")
    print("  1. For small datasets (< 2K samples): Use all samples, not a subset")
    print("  2. For datasets with different splits: Verify split distributions are similar")
    print("  3. For document-specific tasks: Consider if train/val have different document types")
    print("  4. Block importance is a structural property - high correlation (>0.9) is expected")
    print("     Lower correlation may indicate:")
    print("     - Insufficient samples (statistical noise)")
    print("     - Distribution shift between splits")
    print("     - Task-specific effects (e.g., overfitting to training data)")

if __name__ == "__main__":
    main()


