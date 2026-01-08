"""
Measure vision encoder overhead for single global crop.
This helps evaluate the overhead of using vision feature in Stage 1.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.base_experiment import BaseExperiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def measure_vision_encoder_overhead(
    vision_encoder,
    device: str = "cuda",
    num_warmup: int = 10,
    num_runs: int = 100,
    image_size: tuple = (336, 336),
) -> dict:
    """
    Measure vision encoder latency for a single global crop.
    
    This helps evaluate the overhead of Variant 3 (budget + vision).
    
    Args:
        vision_encoder: Vision encoder model
        device: Device to use
        num_warmup: Number of warmup runs
        num_runs: Number of measurement runs
        image_size: Image size (H, W)
    
    Returns:
        metrics: {
            'mean_ms': Mean latency in milliseconds,
            'std_ms': Std latency in milliseconds,
            'p50_ms': Median latency,
            'p95_ms': 95th percentile latency,
            'p99_ms': 99th percentile latency,
        }
    """
    import time
    import numpy as np
    
    vision_encoder.eval()
    vision_encoder.to(device)
    
    # Create dummy image (single global crop)
    dummy_image = torch.randn(1, 3, image_size[0], image_size[1], device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = vision_encoder(dummy_image)
    
    # Synchronize
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            _ = vision_encoder(dummy_image)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    metrics = {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Measure vision encoder overhead for single global crop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=10,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="Number of measurement runs"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[336, 336],
        help="Image size (H, W)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="vision_encoder_overhead.json",
        help="Output file for metrics"
    )
    
    args = parser.parse_args()
    
    log.info("Loading model...")
    experiment = BaseExperiment(model_path=args.model_path, device=args.device)
    
    # Get vision encoder
    vision_encoder = experiment.model.model.vision_backbone.image_vit
    
    log.info("Measuring vision encoder overhead...")
    log.info(f"Image size: {args.image_size}")
    log.info(f"Warmup runs: {args.num_warmup}")
    log.info(f"Measurement runs: {args.num_runs}")
    
    metrics = measure_vision_encoder_overhead(
        vision_encoder=vision_encoder,
        device=args.device,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        image_size=tuple(args.image_size),
    )
    
    log.info("=" * 80)
    log.info("Vision Encoder Overhead (Single Global Crop)")
    log.info("=" * 80)
    log.info(f"Mean: {metrics['mean_ms']:.2f} ms")
    log.info(f"Std:  {metrics['std_ms']:.2f} ms")
    log.info(f"P50:  {metrics['p50_ms']:.2f} ms")
    log.info(f"P95:  {metrics['p95_ms']:.2f} ms")
    log.info(f"P99:  {metrics['p99_ms']:.2f} ms")
    log.info(f"Min:  {metrics['min_ms']:.2f} ms")
    log.info(f"Max:  {metrics['max_ms']:.2f} ms")
    log.info("=" * 80)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    log.info(f"Results saved to {args.output_file}")
    
    # Analysis
    log.info("\nAnalysis:")
    log.info(f"- Overhead for single global crop: {metrics['mean_ms']:.2f} ms")
    log.info(f"- If using two-pass vision encoding (Variant 3), this is the overhead")
    log.info(f"- Compare with total inference latency to evaluate trade-off")


if __name__ == "__main__":
    main()

