"""
Analyze VQA v2 validation set answer lengths (in tokens).
This helps determine the optimal max_new_tokens setting.
"""

import argparse
import logging
import sys
import os
import random
from collections import Counter
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
from molmo.data import get_dataset_by_name
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def analyze_answer_lengths(
    tokenizer_path: str = None,
    dataset_name: str = "coco_2014_vqa",
    split: str = "validation",
    sample_size: int = None,
):
    """
    Analyze answer lengths in VQA v2 validation set.
    
    Args:
        tokenizer_path: Path to tokenizer (default: None, will use configs/tokenizer or HF Hub)
        dataset_name: Dataset name (default: "coco_2014_vqa")
        split: Dataset split (default: "validation")
        sample_size: If provided, only analyze first N samples (for quick testing)
    """
    log.info("Loading tokenizer...")
    # Load tokenizer directly (no need for model checkpoint)
    project_tokenizer_path = os.path.join("configs", "tokenizer")
    
    if tokenizer_path is None:
        if os.path.exists(os.path.join(project_tokenizer_path, "tokenizer.json")):
            log.info(f"Loading tokenizer from {project_tokenizer_path}")
            tokenizer_path = project_tokenizer_path
        else:
            log.warning("No local tokenizer found. Fetching from HF Hub...")
            tokenizer_path = "allenai/MolmoE-1B-0924"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    log.info("Loading dataset...")
    # Load dataset directly (no preprocessing needed - we just need the answers)
    dataset = get_dataset_by_name(dataset_name, split=split)
    
    # Limit sample size if specified
    total_samples = len(dataset)
    if sample_size is not None:
        total_samples = min(sample_size, total_samples)
        log.info(f"Analyzing first {total_samples} samples (out of {len(dataset)} total)")
    else:
        log.info(f"Analyzing all {total_samples} samples")
    
    # Collect answer lengths
    answer_lengths = []
    answer_texts = []
    
    log.info("Processing samples...")
    rng = random.Random(42)  # For deterministic dataset access
    
    for i in tqdm(range(total_samples), desc="Analyzing answers"):
        try:
            # Get sample directly from dataset (no preprocessing needed)
            sample = dataset.get(i, rng=rng)
        except Exception as e:
            log.warning(f"Error getting sample {i}: {e}")
            continue
        
        # Get answers from sample
        # VQA2 dataset returns answers directly in the sample dict
        answers = None
        if "answers" in sample:
            answers = sample["answers"]
        elif "metadata" in sample and "answers" in sample["metadata"]:
            answers = sample["metadata"]["answers"]
        
        if answers is None:
            continue
        
        # Normalize to list
        if isinstance(answers, str):
            answers = [answers]
        elif not isinstance(answers, list):
            continue
        
        # For each answer, tokenize and get length
        for answer in answers:
            if answer is None or not isinstance(answer, str):
                continue
            
            # Tokenize the answer
            # Note: We tokenize the answer text directly, similar to how the model would generate it
            try:
                tokens = tokenizer.encode(answer, add_special_tokens=False)
                length = len(tokens)
                
                answer_lengths.append(length)
                answer_texts.append(answer)
            except Exception as e:
                log.warning(f"Error tokenizing answer '{answer[:50]}...': {e}")
                continue
    
    # Statistics
    answer_lengths = np.array(answer_lengths)
    
    log.info("\n" + "="*80)
    log.info("VQA v2 Validation Set Answer Length Statistics (in tokens)")
    log.info("="*80)
    log.info(f"Total answers analyzed: {len(answer_lengths)}")
    log.info(f"Min length: {answer_lengths.min()}")
    log.info(f"Max length: {answer_lengths.max()}")
    log.info(f"Mean length: {answer_lengths.mean():.2f}")
    log.info(f"Median length: {np.median(answer_lengths):.2f}")
    log.info(f"Std deviation: {answer_lengths.std():.2f}")
    log.info("\nPercentiles:")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        percentile = np.percentile(answer_lengths, p)
        log.info(f"  P{p}: {percentile:.2f} tokens")
    
    # Distribution
    log.info("\nLength distribution:")
    length_counts = Counter(answer_lengths)
    sorted_lengths = sorted(length_counts.items())
    
    # Group by ranges
    ranges = [
        (0, 1, "1"),
        (2, 5, "2-5"),
        (6, 10, "6-10"),
        (11, 15, "11-15"),
        (16, 20, "16-20"),
        (21, 30, "21-30"),
        (31, 50, "31-50"),
        (51, 100, "51-100"),
        (101, float('inf'), "100+"),
    ]
    
    range_counts = {}
    for min_len, max_len, label in ranges:
        if max_len == float('inf'):
            # For infinite range, count all lengths >= min_len
            count = sum(length_counts.get(l, 0) for l in length_counts if l >= min_len)
        else:
            # For finite range, count lengths in [min_len, max_len]
            count = sum(length_counts.get(l, 0) for l in range(min_len, int(max_len) + 1) if l in length_counts)
        if count > 0:
            range_counts[label] = count
            percentage = count / len(answer_lengths) * 100
            log.info(f"  {label:10s} tokens: {count:6d} answers ({percentage:5.2f}%)")
    
    # Coverage analysis for different max_new_tokens values
    log.info("\n" + "="*80)
    log.info("Coverage Analysis for Different max_new_tokens Settings")
    log.info("="*80)
    
    for max_tokens in [8, 16, 32, 64, 128]:
        covered = np.sum(answer_lengths <= max_tokens)
        coverage = covered / len(answer_lengths) * 100
        truncated = len(answer_lengths) - covered
        truncation_rate = truncated / len(answer_lengths) * 100
        
        log.info(f"max_new_tokens={max_tokens:3d}: "
                f"Coverage={coverage:6.2f}% ({covered:6d}/{len(answer_lengths)}), "
                f"Truncation={truncation_rate:5.2f}% ({truncated:6d} answers)")
    
    # Sample some long answers
    log.info("\n" + "="*80)
    log.info("Sample Long Answers (>16 tokens)")
    log.info("="*80)
    
    long_indices = np.where(answer_lengths > 16)[0]
    if len(long_indices) > 0:
        log.info(f"Found {len(long_indices)} answers longer than 16 tokens")
        log.info("Sample answers:")
        for idx in long_indices[:10]:  # Show first 10
            answer = answer_texts[idx]
            length = answer_lengths[idx]
            # Truncate if too long
            if len(answer) > 100:
                answer = answer[:100] + "..."
            log.info(f"  [{length:2d} tokens] {answer}")
    else:
        log.info("No answers longer than 16 tokens found!")
    
    # Sample some very long answers (>32 tokens)
    very_long_indices = np.where(answer_lengths > 32)[0]
    if len(very_long_indices) > 0:
        log.info(f"\nFound {len(very_long_indices)} answers longer than 32 tokens")
        log.info("Sample very long answers:")
        for idx in very_long_indices[:5]:  # Show first 5
            answer = answer_texts[idx]
            length = answer_lengths[idx]
            if len(answer) > 100:
                answer = answer[:100] + "..."
            log.info(f"  [{length:2d} tokens] {answer}")
    
    # Recommendations
    log.info("\n" + "="*80)
    log.info("Recommendations")
    log.info("="*80)
    
    coverage_8 = np.sum(answer_lengths <= 8) / len(answer_lengths) * 100
    coverage_16 = np.sum(answer_lengths <= 16) / len(answer_lengths) * 100
    
    log.info(f"max_new_tokens=8:  {coverage_8:.2f}% coverage")
    log.info(f"max_new_tokens=16: {coverage_16:.2f}% coverage")
    
    if coverage_16 >= 99.0:
        log.info("\n✅ RECOMMENDATION: max_new_tokens=16 is safe (>99% coverage)")
    elif coverage_16 >= 95.0:
        log.info("\n⚠️  RECOMMENDATION: max_new_tokens=16 covers >95% but may truncate some answers")
        log.info("   Consider max_new_tokens=32 for better coverage")
    else:
        log.info("\n❌ RECOMMENDATION: max_new_tokens=16 may truncate too many answers")
        log.info("   Consider max_new_tokens=32 or higher")
    
    if coverage_8 >= 95.0:
        log.info("\n✅ max_new_tokens=8 might work but is risky (may truncate 5%+ answers)")
    else:
        log.info("\n❌ max_new_tokens=8 is too small (will truncate many answers)")
    
    return {
        "total_answers": len(answer_lengths),
        "min": int(answer_lengths.min()),
        "max": int(answer_lengths.max()),
        "mean": float(answer_lengths.mean()),
        "median": float(np.median(answer_lengths)),
        "std": float(answer_lengths.std()),
        "percentiles": {p: float(np.percentile(answer_lengths, p)) for p in [50, 75, 90, 95, 99, 99.5, 99.9]},
        "coverage": {
            8: float(coverage_8),
            16: float(coverage_16),
            32: float(np.sum(answer_lengths <= 32) / len(answer_lengths) * 100),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze VQA v2 answer lengths")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to tokenizer (default: configs/tokenizer or HF Hub)")
    parser.add_argument("--dataset_name", type=str, default="coco_2014_vqa",
                       help="Dataset name")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Only analyze first N samples (for quick testing)")
    
    args = parser.parse_args()
    
    stats = analyze_answer_lengths(
        tokenizer_path=args.tokenizer_path,
        dataset_name=args.dataset_name,
        split=args.split,
        sample_size=args.sample_size,
    )
    
    log.info("\n" + "="*80)
    log.info("Analysis complete!")
    log.info("="*80)


if __name__ == "__main__":
    main()

