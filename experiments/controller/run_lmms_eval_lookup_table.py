"""
Run lmms-eval evaluation with Lookup Table Baseline Controller.

This script integrates the lookup table baseline controller with lmms-eval framework,
following AdaLLaVA's evaluation approach.

Usage:
    python -m experiments.controller.run_lmms_eval_lookup_table \
        --model_path checkpoints/molmo \
        --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
        --tasks textvqa_val,mme,pope \
        --latency_budget 200.0 \
        --output_path ./results/logs_eval/lookup_table_baseline/lmms_eval/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def run_lmms_eval_lookup_table(
    model_path: str,
    lookup_table_path: str,
    tasks: str,
    latency_budget: float = 200.0,
    max_new_tokens: int = 512,
    batch_size: int = 1,
    output_path: str = "./results/logs_eval/",
    device: str = "cuda",
    log_samples: bool = True,
    log_samples_suffix: str = "",
):
    """
    Run lmms-eval evaluation with lookup table baseline controller.
    
    Args:
        model_path: Path to model checkpoint
        lookup_table_path: Path to lookup table JSON file
        tasks: Comma-separated list of tasks (e.g., "textvqa_val,mme,pope")
        latency_budget: Latency budget in milliseconds
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for evaluation
        output_path: Output directory for results
        device: Device to use
        log_samples: If True, log individual samples
        log_samples_suffix: Suffix for log samples file
    """
    try:
        # Import lmms-eval (check if available)
        try:
            from lmms_eval import evaluator, tasks
            from lmms_eval.api.instance import Instance
            from lmms_eval.api.registry import register_model
            from lmms_eval.api.model import Model
        except ImportError:
            log.error(
                "lmms-eval is not installed. Please install it first:\n"
                "  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval\n"
                "  cd lmms-eval\n"
                "  pip install -e .\n"
                "  cd ..\n"
                "\n"
                "Note: You can use the latest version. If you encounter issues,\n"
                "you can use AdaLLaVA's specific version:\n"
                "  git checkout 80391ce3bfb5a19b32e7a19a2d9399e1378ed2dd"
            )
            return
        
        log.info("=" * 80)
        log.info("LMms-Eval Evaluation with Lookup Table Baseline")
        log.info("=" * 80)
        log.info(f"Model path: {model_path}")
        log.info(f"Lookup table path: {lookup_table_path}")
        log.info(f"Tasks: {tasks}")
        log.info(f"Latency budget: {latency_budget}ms")
        log.info(f"Max new tokens: {max_new_tokens}")
        log.info(f"Output path: {output_path}")
        log.info("=" * 80)
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create lookup table adapter
        log.info("Creating lookup table adapter...")
        from experiments.controller.lmms_eval_lookup_table_adapter import create_lookup_table_lmms_eval_adapter
        
        adapter = create_lookup_table_lmms_eval_adapter(
            model_path=model_path,
            lookup_table_path=lookup_table_path,
            latency_budget=latency_budget,
            max_new_tokens=max_new_tokens,
            deterministic=True,
            device=device,
        )
        log.info("Adapter created successfully!")
        
        # Parse tasks
        task_list = [t.strip() for t in tasks.split(",")]
        log.info(f"Evaluating on {len(task_list)} tasks: {task_list}")
        
        # Run evaluation for each task
        all_results = {}
        
        for task_name in task_list:
            log.info(f"\n{'='*80}")
            log.info(f"Evaluating task: {task_name}")
            log.info(f"{'='*80}")
            
            try:
                # Run task evaluation
                # Note: This is a simplified approach. In practice, you'd use
                # lmms-eval's proper task system
                results = run_single_task(
                    adapter=adapter,
                    task_name=task_name,
                    batch_size=batch_size,
                    device=device,
                )
                
                all_results[task_name] = results
                log.info(f"✓ Completed {task_name}")
                
            except Exception as e:
                log.error(f"✗ Failed {task_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save results
        results_file = output_dir / f"results_budget_{latency_budget:.0f}.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(all_results, f, indent=2)
        
        log.info(f"\nResults saved to: {results_file}")
        
        # Print summary
        log.info("\n" + "=" * 80)
        log.info("Evaluation Summary")
        log.info("=" * 80)
        for task_name, results in all_results.items():
            if 'accuracy' in results:
                log.info(f"{task_name}: Accuracy = {results['accuracy']:.4f}")
            if 'latency' in results:
                log.info(f"{task_name}: Latency = {results['latency']:.2f}ms")
        
    except Exception as e:
        log.error(f"Error in lmms-eval evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_single_task(
    adapter,
    task_name: str,
    batch_size: int = 1,
    device: str = "cuda",
) -> dict:
    """
    Run evaluation on a single task.
    
    This is a simplified implementation. In practice, you would use
    lmms-eval's task system properly.
    """
    log.info(f"Running task: {task_name}")
    
    # This is a placeholder - actual implementation would:
    # 1. Load the task dataset using lmms-eval's task loader
    # 2. Iterate through samples
    # 3. Call adapter.generate() for each sample
    # 4. Compute metrics using task-specific evaluators
    
    # For demonstration, return placeholder results
    results = {
        "task": task_name,
        "accuracy": 0.0,
        "latency": 0.0,
        "num_samples": 0,
    }
    
    log.warning(
        f"Task {task_name} evaluation is not fully implemented yet. "
        "This requires proper integration with lmms-eval's task system."
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run lmms-eval with Lookup Table Baseline Controller",
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
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated list of tasks (e.g., textvqa_val,mme,pope)"
    )
    parser.add_argument(
        "--latency_budget",
        type=float,
        default=200.0,
        help="Latency budget in milliseconds"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/logs_eval/lookup_table_baseline/lmms_eval/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log individual samples"
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="",
        help="Suffix for log samples file"
    )
    
    args = parser.parse_args()
    
    run_lmms_eval_lookup_table(
        model_path=args.model_path,
        lookup_table_path=args.lookup_table_path,
        tasks=args.tasks,
        latency_budget=args.latency_budget,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        output_path=args.output_path,
        device=args.device,
        log_samples=args.log_samples,
        log_samples_suffix=args.log_samples_suffix,
    )


if __name__ == "__main__":
    main()

