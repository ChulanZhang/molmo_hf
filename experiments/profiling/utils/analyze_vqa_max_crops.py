"""
Analyze max_crops usage in VQA v2 validation set.
This script helps determine the optimal max_crops value for profiling experiments.
"""

import argparse
import logging
import sys
import os
from collections import Counter
from typing import Dict, List
from multiprocessing import Pool, cpu_count
from functools import partial

# Set TOKENIZERS_PARALLELISM to avoid warnings when using multiprocessing
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.getcwd())

from molmo.data import get_dataset_by_name
from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
from molmo.data.data_formatter import DataFormatter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)


def analyze_max_crops_usage(
    model_path: str = None,
    dataset_name: str = "coco_2014_vqa",
    split: str = "validation",
    sample_size: int = None,
    max_crops: int = 100,  # Set to a large value to avoid truncation during analysis
):
    """
    Analyze the distribution of actual crop counts in VQA v2 dataset.
    
    Args:
        model_path: Path to model directory (for loading tokenizer/config)
        dataset_name: Dataset name
        split: Dataset split
        sample_size: If provided, only analyze this many samples
        max_crops: Maximum crops to use (default: 100, set large to avoid truncation during analysis)
    """
    log.info(f"Analyzing max_crops usage for {dataset_name}/{split}")
    log.info(f"Using max_crops={max_crops}")
    
    # Load dataset
    dataset = get_dataset_by_name(dataset_name, split=split)
    total_samples = len(dataset)
    log.info(f"Total samples in dataset: {total_samples}")
    
    if sample_size is not None:
        log.info(f"Analyzing first {sample_size} samples")
        dataset = dataset.select(range(min(sample_size, total_samples)))
    
    # Load tokenizer (minimal setup for preprocessing)
    # Try local tokenizer first, then fallback to HF Hub
    log.info("Loading tokenizer...")
    project_tokenizer_path = os.path.join("configs", "tokenizer")
    
    if os.path.exists(os.path.join(project_tokenizer_path, "tokenizer.json")):
        log.info(f"Loading tokenizer from {project_tokenizer_path}")
        tokenizer_path = project_tokenizer_path
    else:
        log.warning("No local tokenizer found. Fetching from HF Hub...")
        tokenizer_path = "allenai/MolmoE-1B-0924"
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Create preprocessor
    from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
    from molmo.data.data_formatter import DataFormatter
    
    mm_preprocessor = MultiModalPreprocessor(
        tokenizer=tokenizer,
        crop_mode="overlap-and-resize-c2",  # Default crop mode
        max_crops=max_crops,
        overlap_margins=(4, 4),
        image_padding_mask=False,
    )
    
    formatter = DataFormatter(
        prompt_templates="uber_model",
        message_format="role",
        system_prompt="demo_or_style",
        always_start_with_space=True,
    )
    
    preprocessor = Preprocessor(
        formater=formatter,
        mm_preprocessor=mm_preprocessor,
        for_inference=True,
    )
    
    # Analyze crop counts
    crop_counts = []
    crop_count_distribution = Counter()
    
    # Check if multiprocessing should be used
    # For I/O-intensive tasks (image loading), multiprocessing overhead may outweigh benefits
    # Default to single-threaded for better performance
    use_multiprocessing = os.environ.get("USE_MULTIPROCESSING", "false").lower() == "true"
    default_workers = min(4, cpu_count())  # Cap at 4 if enabled
    num_workers = int(os.environ.get("NUM_WORKERS", default_workers))
    
    log.info("Processing samples to count crops...")
    if use_multiprocessing and num_workers > 1:
        log.info(f"Using multiprocessing with {num_workers} workers (CPU only, no GPU needed)")
        log.info(f"Note: Each worker initializes preprocessor once, then processes samples")
        log.warning("Multiprocessing may be slower for I/O-intensive tasks. Consider using single-threaded mode.")
        
        # Prepare items for multiprocessing
        items = list(dataset)
        
        # Process in parallel with worker initialization
        with Pool(processes=num_workers, initializer=_init_worker, initargs=(max_crops,)) as pool:
            results = list(tqdm(
                pool.imap(_process_single_sample, items, chunksize=500),  # Larger chunksize for better performance
                total=len(items),
                desc="Processing samples"
            ))
        
        # Collect results
        for crop_count, error in results:
            if error is not None:
                log.warning(f"Error processing sample: {error}")
                continue
            if crop_count is not None:
                crop_counts.append(crop_count)
                crop_count_distribution[crop_count] += 1
    else:
        log.info("Using single-threaded processing (recommended for I/O-intensive tasks)")
        import numpy as np
        rng = np.random.RandomState(42)  # For deterministic processing
        
        for item in tqdm(dataset, total=len(dataset)):
            try:
                # Process the item using __call__ method
                processed = preprocessor(item, rng=rng)
                
                # Count crops from images
                if "images" in processed and processed["images"] is not None:
                    images = processed["images"]
                    if hasattr(images, 'shape'):
                        # images shape: (num_crops, n_patches, patch_dim) for preprocessed images
                        # or (num_crops, H, W, C) for raw images
                        if len(images.shape) >= 2:
                            # For both 3D (n_crops, n_patches, patch_dim) and 4D (n_crops, H, W, C)
                            # the first dimension is the number of crops
                            num_crops = images.shape[0]
                        else:
                            num_crops = 1
                    elif isinstance(images, list):
                        num_crops = len(images)
                    else:
                        num_crops = 1
                else:
                    num_crops = 0
                
                # Debug: print num_crops for first few samples (comment out if not needed)
                if len(crop_counts) < 10:
                    log.info(f"Sample {len(crop_counts)}: num_crops={num_crops}, images.shape={images.shape if 'images' in processed and processed['images'] is not None else 'None'}")
                
                crop_counts.append(num_crops)
                crop_count_distribution[num_crops] += 1
                
            except Exception as e:
                log.warning(f"Error processing sample: {e}")
                continue
    
    # Statistics
    total_analyzed = len(crop_counts)
    if total_analyzed == 0:
        log.error("No samples were successfully processed!")
        return
    
    log.info(f"\n{'='*80}")
    log.info(f"Analysis Results (analyzed {total_analyzed} samples)")
    log.info(f"{'='*80}")
    
    log.info(f"\nCrop Count Distribution:")
    log.info(f"{'Crop Count':<15} {'Count':<15} {'Percentage':<15}")
    log.info(f"{'-'*45}")
    
    sorted_counts = sorted(crop_count_distribution.items())
    for crop_count, count in sorted_counts:
        percentage = (count / total_analyzed) * 100
        log.info(f"{crop_count:<15} {count:<15} {percentage:>6.2f}%")
    
    # Summary statistics
    min_crops = min(crop_counts)
    max_crops_actual = max(crop_counts)
    mean_crops = sum(crop_counts) / len(crop_counts)
    
    # Calculate percentiles
    sorted_crops = sorted(crop_counts)
    p50 = sorted_crops[len(sorted_crops) // 2]
    p75 = sorted_crops[int(len(sorted_crops) * 0.75)]
    p90 = sorted_crops[int(len(sorted_crops) * 0.90)]
    p95 = sorted_crops[int(len(sorted_crops) * 0.95)]
    p99 = sorted_crops[int(len(sorted_crops) * 0.99)]
    
    log.info(f"\n{'='*80}")
    log.info(f"Summary Statistics:")
    log.info(f"{'='*80}")
    log.info(f"Min crops: {min_crops}")
    log.info(f"Max crops: {max_crops_actual}")
    log.info(f"Mean crops: {mean_crops:.2f}")
    log.info(f"Median (P50): {p50}")
    log.info(f"P75: {p75}")
    log.info(f"P90: {p90}")
    log.info(f"P95: {p95}")
    log.info(f"P99: {p99}")
    
    # Coverage analysis for different max_crops values
    log.info(f"\n{'='*80}")
    log.info(f"Coverage Analysis for Different max_crops Settings:")
    log.info(f"{'='*80}")
    
    test_max_crops = [6, 8, 10, 12, 14, 16, 18, 20]
    for test_max in test_max_crops:
        covered = sum(1 for c in crop_counts if c <= test_max)
        coverage = (covered / total_analyzed) * 100
        truncated = total_analyzed - covered
        log.info(f"max_crops={test_max:2d}: Coverage={coverage:6.2f}% ({covered}/{total_analyzed}), "
                f"Truncation={truncated:6d} samples ({100-coverage:6.2f}%)")
    
    # Recommendation
    log.info(f"\n{'='*80}")
    log.info(f"Recommendations:")
    log.info(f"{'='*80}")
    
    # Find max_crops that covers 99% of samples
    for test_max in sorted(test_max_crops):
        covered = sum(1 for c in crop_counts if c <= test_max)
        coverage = (covered / total_analyzed) * 100
        if coverage >= 99.0:
            log.info(f"✅ max_crops={test_max} covers {coverage:.2f}% of samples (recommended)")
            break
    
    # Find max_crops that covers 95% of samples
    for test_max in sorted(test_max_crops):
        covered = sum(1 for c in crop_counts if c <= test_max)
        coverage = (covered / total_analyzed) * 100
        if coverage >= 95.0:
            log.info(f"⚠️  max_crops={test_max} covers {coverage:.2f}% of samples (acceptable, may truncate some)")
            break
    
    return {
        "total_samples": total_analyzed,
        "min_crops": min_crops,
        "max_crops": max_crops_actual,
        "mean_crops": mean_crops,
        "percentiles": {
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "p95": p95,
            "p99": p99,
        },
        "distribution": dict(crop_count_distribution),
    }


# Global preprocessor for worker processes (initialized once per worker)
_worker_preprocessor = None
_worker_max_crops = None

def _init_worker(max_crops):
    """Initialize preprocessor for a worker process (called once per worker)."""
    global _worker_preprocessor, _worker_max_crops
    
    # Set TOKENIZERS_PARALLELISM for this worker
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    from transformers import AutoTokenizer
    from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
    from molmo.data.data_formatter import DataFormatter
    
    # Load tokenizer
    project_tokenizer_path = os.path.join("configs", "tokenizer")
    if os.path.exists(os.path.join(project_tokenizer_path, "tokenizer.json")):
        tokenizer_path = project_tokenizer_path
    else:
        tokenizer_path = "allenai/MolmoE-1B-0924"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Create preprocessor
    mm_preprocessor = MultiModalPreprocessor(
        tokenizer=tokenizer,
        crop_mode="overlap-and-resize-c2",
        max_crops=max_crops,
        overlap_margins=(4, 4),
        image_padding_mask=False,
    )
    
    formatter = DataFormatter(
        prompt_templates="uber_model",
        message_format="role",
        system_prompt="demo_or_style",
        always_start_with_space=True,
    )
    
    _worker_preprocessor = Preprocessor(
        formater=formatter,
        mm_preprocessor=mm_preprocessor,
        for_inference=True,
    )
    _worker_max_crops = max_crops

def _process_single_sample(item):
    """Process a single sample to count crops. Used for multiprocessing."""
    global _worker_preprocessor
    
    try:
        import numpy as np
        
        # Process the item using pre-initialized preprocessor
        rng = np.random.RandomState(42)  # Deterministic
        processed = _worker_preprocessor(item, rng=rng)
        
        # Count crops
        if "images" in processed and processed["images"] is not None:
            images = processed["images"]
            if hasattr(images, 'shape'):
                # images shape: (num_crops, n_patches, patch_dim) for preprocessed images
                # or (num_crops, H, W, C) for raw images
                if len(images.shape) >= 2:
                    # For both 3D (n_crops, n_patches, patch_dim) and 4D (n_crops, H, W, C)
                    # the first dimension is the number of crops
                    num_crops = images.shape[0]
                else:
                    num_crops = 1
            elif isinstance(images, list):
                num_crops = len(images)
            else:
                num_crops = 1
        else:
            num_crops = 0
        
        # Debug: print num_crops for first few samples (comment out if not needed)
        # Note: This is in multiprocessing worker, so logs may appear out of order
        # if len(crop_counts) < 10:  # Can't access crop_counts here in worker process
        #     log.info(f"Sample: num_crops={num_crops}, images.shape={images.shape if 'images' in processed and processed['images'] is not None else 'None'}")
        
        return num_crops, None  # (crop_count, error)
        
    except Exception as e:
        return None, str(e)  # (None, error_message)


def main():
    parser = argparse.ArgumentParser(description="Analyze max_crops usage in VQA v2 dataset")
    parser.add_argument("--model_path", type=str, default="checkpoints",
                       help="Path to model directory")
    parser.add_argument("--dataset_name", type=str, default="coco_2014_vqa",
                       help="Dataset name")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of samples to analyze (default: all)")
    parser.add_argument("--max_crops", type=int, default=100,
                       help="Maximum crops to use (default: 100, set large to avoid truncation during analysis)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of worker processes for multiprocessing (default: 4, only used if --use_multiprocessing)")
    parser.add_argument("--use_multiprocessing", action="store_true",
                       help="Enable multiprocessing (default: disabled, single-threaded is usually faster for I/O)")
    
    args = parser.parse_args()
    
    # Set environment variables for multiprocessing
    if args.num_workers is not None:
        os.environ["NUM_WORKERS"] = str(args.num_workers)
    if args.use_multiprocessing:
        os.environ["USE_MULTIPROCESSING"] = "true"
    
    analyze_max_crops_usage(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        split=args.split,
        sample_size=args.sample_size,
        max_crops=args.max_crops,
    )


if __name__ == "__main__":
    main()

