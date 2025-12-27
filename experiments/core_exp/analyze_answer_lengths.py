"""
Script to analyze groundtruth answer lengths for different datasets.
Helps determine appropriate max_new_tokens values.
"""

import sys
import os
sys.path.append(os.getcwd())

import logging
import random
from molmo.data import get_dataset_by_name
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Datasets to analyze (from run_multi_datasets.sh)
DATASETS = [
    ("coco_2014_vqa", "validation"),
    ("text_vqa", "validation"),
    ("okvqa", "validation"),
    ("science_qa_img", "validation"),
    ("st_qa", "validation"),
    ("doc_qa", "validation"),
    ("tally_qa", "test"),
    ("mmmu", "validation"),
    ("coco_caption", "validation"),
]

def analyze_dataset(dataset_name: str, split: str, tokenizer, max_samples: int = None):
    """Analyze answer lengths for a dataset."""
    try:
        log.info(f"\n{'='*60}")
        log.info(f"Analyzing {dataset_name} ({split})")
        log.info(f"{'='*60}")
        
        dataset = get_dataset_by_name(dataset_name, split=split)
        total_size = len(dataset)
        log.info(f"Dataset size: {total_size}")
        
        # Limit samples if specified (for faster analysis)
        if max_samples is not None and max_samples < total_size:
            log.info(f"Analyzing first {max_samples} samples (for faster analysis)")
            sample_indices = range(max_samples)
        else:
            sample_indices = range(total_size)
        
        answer_lengths = []
        max_length = 0
        max_answer = None
        max_sample_idx = None
        
        rng = random.Random(66)  # Fixed seed for reproducibility
        
        for idx in tqdm(sample_indices, desc=f"Processing {dataset_name}"):
            try:
                # Get sample from dataset
                sample = dataset.get(idx, rng=rng)
            except Exception as e:
                log.warning(f"Error getting sample {idx}: {e}")
                continue
            
            # Extract answers from sample
            answers = None
            if isinstance(sample, dict):
                if "answers" in sample:
                    answers = sample["answers"]
                elif "captions" in sample:
                    # COCO Caption: captions is a list of reference captions
                    answers = sample["captions"]
                elif "metadata" in sample and isinstance(sample["metadata"], dict):
                    if "answers" in sample["metadata"]:
                        answers = sample["metadata"]["answers"]
                    elif "answer" in sample["metadata"]:
                        answers = [sample["metadata"]["answer"]]
                    elif "answer_idx" in sample["metadata"]:
                        # Multiple choice: get the correct option text
                        if "options" in sample["metadata"]:
                            options = sample["metadata"]["options"]
                            answer_idx = sample["metadata"].get("answer_idx", -1)
                            if 0 <= answer_idx < len(options):
                                answers = [options[answer_idx]]
            elif hasattr(sample, "answers"):
                answers = sample.answers
            elif hasattr(sample, "captions"):
                # COCO Caption: captions is a list of reference captions
                answers = sample.captions
            elif hasattr(sample, "metadata") and hasattr(sample.metadata, "answers"):
                answers = sample.metadata.answers
            
            # Handle different answer formats
            if answers is None:
                continue
            
            # Normalize to list
            if isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, list):
                continue
            
            # Process each answer
            for answer in answers:
                if answer is None:
                    continue
                
                try:
                    # Tokenize answer
                    tokens = tokenizer.encode(str(answer), add_special_tokens=False)
                    length = len(tokens)
                    answer_lengths.append(length)
                    
                    if length > max_length:
                        max_length = length
                        max_answer = str(answer)
                        max_sample_idx = idx
                except Exception as e:
                    log.warning(f"Error tokenizing answer '{str(answer)[:50]}...': {e}")
                    continue
        
        if not answer_lengths:
            log.warning(f"No answers found in {dataset_name}")
            return None
        
        # Statistics
        answer_lengths = np.array(answer_lengths)
        mean_length = np.mean(answer_lengths)
        median_length = np.median(answer_lengths)
        p95_length = np.percentile(answer_lengths, 95)
        p99_length = np.percentile(answer_lengths, 99)
        p999_length = np.percentile(answer_lengths, 99.9)
        max_length = int(answer_lengths.max())
        
        log.info(f"Answer length statistics (tokens):")
        log.info(f"  Mean: {mean_length:.1f}")
        log.info(f"  Median: {median_length:.1f}")
        log.info(f"  P95: {p95_length:.1f}")
        log.info(f"  P99: {p99_length:.1f}")
        log.info(f"  P99.9: {p999_length:.1f}")
        log.info(f"  Max: {max_length}")
        if max_answer:
            preview = max_answer[:200] + "..." if len(max_answer) > 200 else max_answer
            log.info(f"  Max answer (sample {max_sample_idx}): {preview}")
        
        # Coverage analysis
        log.info(f"\nCoverage for different max_new_tokens:")
        for max_tokens in [16, 32, 64, 128, 256, 512, 1024]:
            covered = np.sum(answer_lengths <= max_tokens)
            coverage = covered / len(answer_lengths) * 100
            log.info(f"  max_new_tokens={max_tokens:4d}: {coverage:6.2f}% coverage ({covered}/{len(answer_lengths)})")
        
        return {
            "dataset": dataset_name,
            "split": split,
            "size": total_size,
            "samples_analyzed": len(answer_lengths),
            "mean": float(mean_length),
            "median": float(median_length),
            "p95": float(p95_length),
            "p99": float(p99_length),
            "p999": float(p999_length),
            "max": max_length,
            "max_answer": max_answer,
            "max_sample_idx": max_sample_idx,
        }
        
    except Exception as e:
        log.error(f"Error analyzing {dataset_name}: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze groundtruth answer lengths for datasets")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to analyze per dataset (for faster analysis)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to tokenizer (default: configs/tokenizer or HF Hub)")
    args = parser.parse_args()
    
    # Load tokenizer
    log.info("Loading tokenizer...")
    project_tokenizer_path = os.path.join("configs", "tokenizer")
    
    if args.tokenizer_path is None:
        if os.path.exists(os.path.join(project_tokenizer_path, "tokenizer.json")):
            log.info(f"Loading tokenizer from {project_tokenizer_path}")
            tokenizer_path = project_tokenizer_path
        else:
            log.info("No local tokenizer found. Fetching from HF Hub...")
            tokenizer_path = "allenai/MolmoE-1B-0924"
    else:
        tokenizer_path = args.tokenizer_path
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        log.warning(f"Could not load tokenizer from {tokenizer_path}: {e}")
        log.warning("Falling back to GPT2 tokenizer (may not match model tokenizer)")
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    log.info("="*60)
    log.info("Analyzing Groundtruth Answer Lengths")
    log.info("="*60)
    log.info(f"Using tokenizer: {type(tokenizer).__name__}")
    if args.max_samples:
        log.info(f"Analyzing up to {args.max_samples} samples per dataset")
    log.info("")
    
    results = []
    for dataset_name, split in DATASETS:
        result = analyze_dataset(dataset_name, split, tokenizer, max_samples=args.max_samples)
        if result:
            results.append(result)
    
    # Summary
    log.info("\n" + "="*80)
    log.info("SUMMARY - Recommended max_new_tokens")
    log.info("="*80)
    log.info(f"{'Dataset':<20} {'Split':<12} {'Mean':<8} {'P95':<8} {'P99':<8} {'P99.9':<8} {'Max':<8} {'Recommended':<12}")
    log.info("-"*80)
    
    for result in results:
        # Recommended: P99.9 + some margin (e.g., 20% or at least 20 tokens)
        # Round up to nearest power of 2 for cleaner values
        recommended_raw = max(result["p999"] * 1.2, result["p999"] + 20)
        # Round to nearest power of 2 (16, 32, 64, 128, 256, 512, 1024)
        recommended = 16
        for val in [16, 32, 64, 128, 256, 512, 1024, 2048]:
            if recommended_raw <= val:
                recommended = val
                break
        else:
            recommended = int(recommended_raw)
        
        log.info(f"{result['dataset']:<20} {result['split']:<12} {result['mean']:<8.1f} "
                f"{result['p95']:<8.1f} {result['p99']:<8.1f} {result['p999']:<8.1f} "
                f"{result['max']:<8.0f} {recommended:<12}")
    
    log.info("\n" + "="*80)
    log.info("Current settings in run_multi_datasets.sh:")
    log.info("="*80)
    current_settings = {
        "st_qa": 512,
        "doc_qa": 512,
        "mmmu": 1024,
    }
    for dataset_name, split in DATASETS:
        current = current_settings.get(dataset_name, 64)
        result = next((r for r in results if r["dataset"] == dataset_name), None)
        if result:
            recommended = max(result["p999"] * 1.2, result["p999"] + 20)
            recommended = int(recommended)
            # Round to nearest power of 2
            for val in [16, 32, 64, 128, 256, 512, 1024, 2048]:
                if recommended <= val:
                    recommended = val
                    break
            status = "✓" if current >= recommended else "⚠"
            log.info(f"  {status} {dataset_name}: {current} (recommended: {recommended}, max: {result['max']})")
        else:
            log.info(f"  ? {dataset_name}: {current} (analysis failed)")
    
    log.info("\n" + "="*80)

if __name__ == "__main__":
    main()

