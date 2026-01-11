"""
Batch evaluation script for Lookup Table Baseline Controller.

Evaluates on multiple datasets and/or multiple latency budgets in batch.

Usage:
    python experiments/controller/evaluate_lookup_table_baseline_batch.py \
        --model_path checkpoints/molmo \
        --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
        --datasets text_vqa okvqa coco_2014_vqa \
        --latency_budgets 170 200 230 260 290 320 350 380 \
        --num_samples 1000
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.controller.evaluate_lookup_table_baseline import evaluate_lookup_table_baseline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Batch Evaluation for Lookup Table Baseline Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--lookup_table_path",
        type=str,
        required=True,
        help="Path to lookup table JSON file"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["text_vqa"],
        help="List of dataset names to evaluate"
    )
    parser.add_argument(
        "--latency_budgets",
        type=float,
        nargs="+",
        default=[170.0, 200.0, 230.0, 260.0, 290.0, 320.0, 350.0, 380.0],
        help="List of latency budgets to evaluate (in ms)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate per dataset"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/logs_eval/lookup_table_baseline/",
        help="Base output directory for results"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save individual predictions to file"
    )
    
    args = parser.parse_args()
    
    log.info("=" * 80)
    log.info("Batch Evaluation for Lookup Table Baseline Controller")
    log.info("=" * 80)
    log.info(f"Model path: {args.model_path}")
    log.info(f"Lookup table path: {args.lookup_table_path}")
    log.info(f"Datasets: {args.datasets}")
    log.info(f"Latency budgets: {args.latency_budgets}")
    log.info(f"Num samples per dataset: {args.num_samples}")
    log.info(f"Output path: {args.output_path}")
    log.info("=" * 80)
    
    total_experiments = len(args.datasets) * len(args.latency_budgets)
    log.info(f"Total experiments: {total_experiments}")
    
    completed = 0
    failed = []
    
    for dataset in args.datasets:
        for budget in args.latency_budgets:
            completed += 1
            log.info(f"\n[{completed}/{total_experiments}] Evaluating {dataset} with budget {budget}ms")
            
            try:
                output_dir = f"{args.output_path}/{dataset}/budget_{budget:.0f}/"
                
                evaluate_lookup_table_baseline(
                    model_path=args.model_path,
                    lookup_table_path=args.lookup_table_path,
                    dataset=dataset,
                    split="validation",
                    num_samples=args.num_samples,
                    latency_budget=budget,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=1,
                    device=args.device,
                    output_path=output_dir,
                    save_predictions=args.save_predictions,
                )
                
                log.info(f"✓ Completed {dataset} @ {budget}ms")
                
            except Exception as e:
                log.error(f"✗ Failed {dataset} @ {budget}ms: {e}")
                failed.append((dataset, budget, str(e)))
                import traceback
                traceback.print_exc()
                continue
    
    log.info("\n" + "=" * 80)
    log.info("Batch Evaluation Summary")
    log.info("=" * 80)
    log.info(f"Completed: {completed - len(failed)}/{total_experiments}")
    log.info(f"Failed: {len(failed)}/{total_experiments}")
    
    if failed:
        log.warning("\nFailed experiments:")
        for dataset, budget, error in failed:
            log.warning(f"  {dataset} @ {budget}ms: {error}")
    
    log.info(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()

