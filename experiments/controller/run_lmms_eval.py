"""
Run evaluation using lmms-eval framework with adaptive inference engine.
Based on AdaLLaVA's evaluation approach.

Usage:
    python -m experiments.controller.run_lmms_eval \
        --model_path checkpoints/molmo \
        --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
        --tasks textvqa_val,mme,pope \
        --latency_budget 200.0 \
        --output_path ./results/logs_eval/
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.controller.lmms_eval_adapter import create_lmms_eval_adapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def run_lmms_eval(
    model_path: str,
    controller_path: str,
    tasks: str,
    latency_budget: float = 200.0,
    max_new_tokens: int = 512,
    batch_size: int = 1,
    output_path: str = "./results/logs_eval/",
    device: str = "cuda",
    deterministic: bool = True,
    log_samples: bool = True,
    log_samples_suffix: str = "",
):
    """
    Run lmms-eval evaluation with adaptive inference engine.
    
    Args:
        model_path: Path to model checkpoint
        controller_path: Path to controller checkpoint
        tasks: Comma-separated list of tasks (e.g., "textvqa_val,mme,pope")
        latency_budget: Latency budget in milliseconds
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for evaluation
        output_path: Output directory for results
        device: Device to use
        deterministic: If True, use deterministic actions
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
        log.info("Starting LMms-Eval Evaluation with Adaptive Inference")
        log.info("=" * 80)
        log.info(f"Model path: {model_path}")
        log.info(f"Controller path: {controller_path}")
        log.info(f"Tasks: {tasks}")
        log.info(f"Latency budget: {latency_budget}ms")
        log.info(f"Max new tokens: {max_new_tokens}")
        log.info(f"Output path: {output_path}")
        log.info("=" * 80)
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create adapter
        log.info("Creating adaptive inference engine and adapter...")
        adapter = create_lmms_eval_adapter(
            model_path=model_path,
            controller_path=controller_path,
            latency_budget=latency_budget,
            max_new_tokens=max_new_tokens,
            deterministic=deterministic,
            device=device,
        )
        log.info("Adapter created successfully!")
        
        # Register model with lmms-eval
        # Note: This is a simplified approach. In practice, you might need to
        # create a proper Model class that inherits from lmms_eval.api.model.Model
        
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
                # Run evaluation using lmms-eval's evaluator
                # Note: This is a simplified interface. The actual implementation
                # would need to properly integrate with lmms-eval's task system
                
                results = run_single_task(
                    adapter=adapter,
                    task_name=task_name,
                    batch_size=batch_size,
                    device=device,
                )
                
                all_results[task_name] = results
                
                # Save results
                results_file = output_dir / f"{task_name}_results.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                log.info(f"Results saved to {results_file}")
                
            except Exception as e:
                log.error(f"Error evaluating task {task_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[task_name] = {"error": str(e)}
        
        # Save combined results
        combined_results_file = output_dir / "combined_results.json"
        with open(combined_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        log.info(f"\nCombined results saved to {combined_results_file}")
        
        # Print summary
        log.info("\n" + "=" * 80)
        log.info("Evaluation Summary")
        log.info("=" * 80)
        for task_name, results in all_results.items():
            if "error" in results:
                log.info(f"{task_name}: ERROR - {results['error']}")
            else:
                # Print key metrics
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        log.info(f"{task_name}/{key}: {value:.4f}")
        
        # Print adapter statistics
        stats = adapter.get_stats()
        log.info("\n" + "=" * 80)
        log.info("Adapter Statistics")
        log.info("=" * 80)
        log.info(f"Total requests: {stats['total_requests']}")
        log.info(f"Average latency: {stats.get('avg_latency', 0.0):.2f}ms")
        log.info(f"Knob distribution:")
        log.info(f"  Tier: {stats['knob_distribution']['tier']}")
        log.info(f"  Top-K: {dict(list(stats['knob_distribution']['top_k'].items())[:10])}")  # Show first 10
        log.info(f"  Active blocks: {dict(list(stats['knob_distribution']['num_active_blocks'].items())[:10])}")
        
        log.info("\n" + "=" * 80)
        log.info("Evaluation completed!")
        log.info("=" * 80)
        
    except Exception as e:
        log.error(f"Error in evaluation: {e}")
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
    # For now, we'll use a simplified approach that works with common tasks
    # In production, you'd use lmms-eval's task registry
    
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
        "num_samples": 0,
    }
    
    log.warning(
        f"Task {task_name} evaluation is not fully implemented yet. "
        "This requires proper integration with lmms-eval's task system."
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run LMms-Eval evaluation with adaptive inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--controller_path",
        type=str,
        required=True,
        help="Path to controller checkpoint"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="textvqa_val",
        help="Comma-separated list of tasks (e.g., 'textvqa_val,mme,pope')"
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
        default="./results/logs_eval/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic (argmax) actions"
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=True,
        help="Log individual samples"
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="",
        help="Suffix for log samples file"
    )
    
    args = parser.parse_args()
    
    run_lmms_eval(
        model_path=args.model_path,
        controller_path=args.controller_path,
        tasks=args.tasks,
        latency_budget=args.latency_budget,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        output_path=args.output_path,
        device=args.device,
        deterministic=args.deterministic,
        log_samples=args.log_samples,
        log_samples_suffix=args.log_samples_suffix,
    )


if __name__ == "__main__":
    main()

