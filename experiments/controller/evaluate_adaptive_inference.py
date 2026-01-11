"""
Complete evaluation script for adaptive inference using datasets.
This script evaluates the adaptive inference engine on actual datasets
and computes accuracy, latency, and other metrics.

Usage:
    python experiments/controller/evaluate_adaptive_inference.py \
        --model_path checkpoints/molmo \
        --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
        --dataset text_vqa --num_samples 100 --latency_budget 200.0 --device cuda
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.base_experiment import BaseExperiment
from experiments.controller.adaptive_inference import create_adaptive_inference_engine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def evaluate_adaptive_inference(
    model_path: str,
    controller_path: str,
    dataset: str = "text_vqa",
    split: str = "validation",
    num_samples: Optional[int] = None,
    latency_budget: float = 200.0,
    max_new_tokens: int = 128,
    batch_size: int = 1,
    device: str = "cuda",
    output_path: str = "./results/logs_eval/",
    save_predictions: bool = True,
):
    """
    Evaluate adaptive inference engine on a dataset.
    
    Args:
        model_path: Path to model checkpoint
        controller_path: Path to controller checkpoint
        dataset: Dataset name (text_vqa, okvqa, etc.)
        split: Dataset split (validation, test, etc.)
        num_samples: Number of samples to evaluate (None = all)
        latency_budget: Latency budget in milliseconds
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for evaluation (default: 1)
                    Note: Latency is measured with batch_size=1 for accurate per-sample latency
        device: Device to use
        output_path: Output directory for results
        save_predictions: If True, save predictions to file
    """
    log.info("=" * 80)
    log.info("Adaptive Inference Evaluation")
    log.info("=" * 80)
    log.info(f"Model path: {model_path}")
    log.info(f"Controller path: {controller_path}")
    log.info(f"Dataset: {dataset} ({split})")
    log.info(f"Num samples: {num_samples if num_samples else 'all'}")
    log.info(f"Latency budget: {latency_budget}ms")
    log.info(f"Max new tokens: {max_new_tokens}")
    log.info(f"Batch size: {batch_size} (latency measured with batch_size=1)")
    log.info("=" * 80)
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create adaptive inference engine
    log.info("Creating adaptive inference engine...")
    engine = create_adaptive_inference_engine(
        model_path=model_path,
        controller_path=controller_path,
        device=device,
    )
    log.info("Engine created successfully!")
    
    # Load dataset
    log.info(f"Loading dataset: {dataset} ({split})...")
    experiment = BaseExperiment(model_path=model_path, device=device)
    dataloader = experiment.build_dataloader(
        dataset_name=dataset,
        split=split,
        batch_size=batch_size,
        max_steps=num_samples,
        shuffle=False,
    )
    
    # Get metric for dataset
    from experiments.base_experiment import get_metric_for_dataset
    metric_name = get_metric_for_dataset(dataset)
    log.info(f"Using metric: {metric_name}")
    
    # Evaluation loop
    all_predictions = []
    all_metadata = []
    all_knobs = []
    all_latencies = []
    all_accuracies = []
    
    log.info("Starting evaluation...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Extract prompts and images from batch
            # Note: This depends on how the batch is structured
            # For now, we'll use the model's processor to handle this
            
            # Get metadata
            metadatas = batch.get("metadata", [])
            if not metadatas:
                log.warning(f"Batch {batch_idx} has no metadata, skipping")
                continue
            
            # Process each sample in the batch
            for sample_idx in range(len(metadatas)):
                metadata = metadatas[sample_idx]
                
                # Extract prompt/question
                question = metadata.get("question", "")
                if not question:
                    log.warning(f"Sample {sample_idx} in batch {batch_idx} has no question")
                    continue
                
                # Get image (if available)
                # Note: Images are in the batch as tensors, we need to extract the right one
                images = batch.get("images")
                image = None
                if images is not None:
                    # Extract single image from batch
                    if len(images.shape) == 5:  # (B, num_crops, H, W, C)
                        image = images[sample_idx]  # (num_crops, H, W, C)
                    elif len(images.shape) == 4:  # (B, H, W, C)
                        image = images[sample_idx]  # (H, W, C)
                
                # Run adaptive inference
                try:
                    result = engine.infer(
                        prompt=question,
                        images=image,
                        latency_budget=latency_budget,
                        max_new_tokens=max_new_tokens,
                        deterministic=True,
                        return_knobs=True,
                    )
                    
                    generated_text = result.get("text", "")
                    knobs = result.get("knobs", {})
                    latency = result.get("latency", 0.0)
                    
                    # Compute accuracy
                    # We need to format the prediction properly for the metric
                    # For now, we'll use a simplified approach
                    # In practice, you'd decode the generated tokens properly
                    
                    # Create a prediction batch for accuracy computation
                    # This is a simplified version - in practice, you'd need to
                    # properly format the prediction as token IDs
                    pred_tokens = experiment.tokenizer.encode(
                        generated_text, return_tensors="pt"
                    ).to(device)
                    
                    # Create a mini-batch for accuracy computation
                    pred_batch = {
                        "input_ids": batch["input_ids"][sample_idx:sample_idx+1],
                        "metadata": [metadata],
                    }
                    
                    # Compute accuracy
                    accuracy_result = experiment.compute_accuracy(
                        batch=pred_batch,
                        predictions=pred_tokens,
                        metric_name=metric_name,
                    )
                    
                    accuracy = accuracy_result.get("accuracy", 0.0)
                    
                    # Store results
                    all_predictions.append(generated_text)
                    all_metadata.append(metadata)
                    all_knobs.append(knobs)
                    all_latencies.append(latency)
                    all_accuracies.append(accuracy)
                    
                except Exception as e:
                    log.error(f"Error processing sample {sample_idx} in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Compute statistics
    log.info("\n" + "=" * 80)
    log.info("Evaluation Results")
    log.info("=" * 80)
    
    if len(all_accuracies) > 0:
        avg_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        log.info(f"Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        log.info(f"Accuracy range: [{np.min(all_accuracies):.4f}, {np.max(all_accuracies):.4f}]")
    else:
        avg_accuracy = 0.0
        log.warning("No accuracy scores computed")
    
    if len(all_latencies) > 0:
        avg_latency = np.mean(all_latencies)
        std_latency = np.std(all_latencies)
        log.info(f"Average latency: {avg_latency:.2f} ± {std_latency:.2f}ms")
        log.info(f"Latency range: [{np.min(all_latencies):.2f}, {np.max(all_latencies):.2f}]ms")
    else:
        avg_latency = 0.0
    
    log.info(f"Total samples: {len(all_predictions)}")
    log.info(f"Total time: {total_time:.2f}s")
    log.info(f"Throughput: {len(all_predictions) / total_time:.2f} samples/s")
    
    # Knob statistics
    if all_knobs:
        tier_counts = {"low": 0, "medium": 0, "high": 0}
        top_k_counts = {}
        num_blocks_counts = {}
        
        for knobs in all_knobs:
            tier = knobs.get("tier", "unknown")
            if tier in tier_counts:
                tier_counts[tier] += 1
            
            top_k = knobs.get("top_k", 0)
            top_k_counts[top_k] = top_k_counts.get(top_k, 0) + 1
            
            num_blocks = knobs.get("num_active_blocks", 0)
            num_blocks_counts[num_blocks] = num_blocks_counts.get(num_blocks, 0) + 1
        
        log.info("\nKnob Distribution:")
        log.info(f"  Tier: {tier_counts}")
        log.info(f"  Top-K: {dict(sorted(top_k_counts.items()))}")
        log.info(f"  Active blocks: {dict(sorted(num_blocks_counts.items()))}")
    
    # Save results
    results = {
        "dataset": dataset,
        "split": split,
        "num_samples": len(all_predictions),
        "latency_budget": latency_budget,
        "max_new_tokens": max_new_tokens,
        "metrics": {
            "accuracy": float(avg_accuracy),
            "accuracy_std": float(std_accuracy) if len(all_accuracies) > 0 else 0.0,
            "avg_latency_ms": float(avg_latency),
            "latency_std_ms": float(std_latency) if len(all_latencies) > 0 else 0.0,
            "throughput_samples_per_sec": float(len(all_predictions) / total_time) if total_time > 0 else 0.0,
        },
        "knob_distribution": {
            "tier": tier_counts if all_knobs else {},
            "top_k": top_k_counts if all_knobs else {},
            "num_active_blocks": num_blocks_counts if all_knobs else {},
        },
    }
    
    # Save predictions if requested
    if save_predictions:
        predictions_data = []
        for i, (pred, metadata, knobs, latency, accuracy) in enumerate(
            zip(all_predictions, all_metadata, all_knobs, all_latencies, all_accuracies)
        ):
            predictions_data.append({
                "sample_id": i,
                "prediction": pred,
                "metadata": {
                    k: v for k, v in metadata.items() 
                    if k in ["question", "answers", "answer", "image_id", "question_id"]
                },
                "knobs": knobs,
                "latency_ms": float(latency),
                "accuracy": float(accuracy),
            })
        
        results["predictions"] = predictions_data
    
    # Save results
    results_file = output_dir / f"{dataset}_{split}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {results_file}")
    
    log.info("=" * 80)
    log.info("Evaluation completed!")
    log.info("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Adaptive Inference on Datasets",
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
        "--dataset",
        type=str,
        default="text_vqa",
        help="Dataset name (text_vqa, okvqa, coco_2014_vqa, etc.)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split (validation, test, train)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)"
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
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
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
        default="./results/logs_eval/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=True,
        help="Save individual predictions to results file"
    )
    
    args = parser.parse_args()
    
    evaluate_adaptive_inference(
        model_path=args.model_path,
        controller_path=args.controller_path,
        dataset=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        latency_budget=args.latency_budget,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output_path,
        save_predictions=args.save_predictions,
    )


if __name__ == "__main__":
    main()

