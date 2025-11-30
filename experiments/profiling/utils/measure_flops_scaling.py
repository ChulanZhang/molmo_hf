"""
Simplified profiling to measure FLOPS impact of different top_k values.
Instead of using hooks (which caused CUDA errors), we'll:
1. Count theoretical FLOPs for different top_k
2. Profile actual GPU time with nvprof/torch.profiler
3. Check if computation scales linearly with top_k
"""

import sys
import os
sys.path.append(os.getcwd())

import torch
import logging
from PIL import Image
from transformers import AutoProcessor
from olmo import Molmo
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def measure_flops_scaling(model_path="hf:allenai/MolmoE-1B-0924", device="cuda:0"):
    """Measure if latency scales with top_k using simple timing"""
    
    log.info(f"Loading model from {model_path}...")
    model = Molmo.from_checkpoint(model_path, device=device)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        "allenai/MolmoE-1B-0924",
        trust_remote_code=True
    )
    
    # Prepare input
    image = Image.new('RGB', (336, 336), color='blue')
    prompt = "Describe this image."
    inputs = processor.process(text=prompt, images=image, return_tensors="pt")
    
    # Ensure batch dimension
    if inputs["input_ids"].ndim == 1:
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        if "images" in inputs and inputs["images"] is not None:
            inputs["images"] = inputs["images"].unsqueeze(0)
        if "image_masks" in inputs and inputs["image_masks"] is not None:
            inputs["image_masks"] = inputs["image_masks"].unsqueeze(0)
        if "image_input_idx" in inputs and inputs["image_input_idx"] is not None:
            inputs["image_input_idx"] = inputs["image_input_idx"].unsqueeze(0)
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Test with extreme values: 1 vs 8 (all experts)
    test_cases = [
        (1, "Minimal - 1 expert per token"),
        (8, "Default - 8 experts per token"),
    ]
    
    results = {}
    
    for top_k, description in test_cases:
        log.info(f"\n{'='*80}")
        log.info(f"Testing: {description} (top_k={top_k})")
        log.info(f"{'='*80}")
        
        # Set top_k for all MoE blocks
        moe_block_count = 0
        for i, block in enumerate(model.transformer.blocks):
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'args'):
                block.ffn.args.top_k = top_k
                moe_block_count += 1
                log.info(f"Block {i}: Set top_k to {top_k}")
        
        log.info(f"Total MoE blocks configured: {moe_block_count}")
        
        # Warmup
        log.info("Warmup (3 iterations)...")
        for _ in range(3):
            with torch.no_grad():
                try:
                    _ = model(**inputs)
                except Exception as e:
                    log.error(f"Error during warmup: {e}")
                    results[top_k] = {"error": str(e)}
                    break
        
        if top_k in results and "error" in results[top_k]:
            continue
        
        # Measure
        log.info("Measuring (10 iterations)...")
        torch.cuda.synchronize()
        times = []
        
        for i in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                try:
                    _ = model(**inputs)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    times.append(elapsed * 1000)  # Convert to ms
                except Exception as e:
                    log.error(f"Error during measurement iteration {i}: {e}")
                    results[top_k] = {"error": str(e)}
                    break
        
        if top_k in results and "error" in results[top_k]:
            continue
        
        # Compute statistics
        import numpy as np
        times_array = np.array(times)
        results[top_k] = {
            "mean": float(np.mean(times_array)),
            "median": float(np.median(times_array)),
            "std": float(np.std(times_array)),
            "min": float(np.min(times_array)),
            "max": float(np.max(times_array)),
        }
        
        log.info(f"Results for top_k={top_k}:")
        log.info(f"  Mean: {results[top_k]['mean']:.2f} ms")
        log.info(f"  Median: {results[top_k]['median']:.2f} ms")
        log.info(f"  Std: {results[top_k]['std']:.2f} ms")
    
    # Summary
    log.info(f"\n{'='*80}")
    log.info("SUMMARY")
    log.info(f"{'='*80}")
    
    if 1 in results and 8 in results and "error" not in results[1] and "error" not in results[8]:
        speedup = results[8]['median'] / results[1]['median']
        log.info(f"top_k=1: {results[1]['median']:.2f} ms")
        log.info(f"top_k=8: {results[8]['median']:.2f} ms")
        log.info(f"Ratio (8/1): {speedup:.3f}x")
        log.info(f"")
        log.info(f"Theoretical FLOPS ratio (assuming linear): 8.0x")
        log.info(f"Actual latency ratio: {speedup:.3f}x")
        
        if speedup < 1.5:
            log.info(f"")
            log.info(f"⚠️  FINDING: Minimal latency increase despite 8x more experts!")
            log.info(f"   This suggests computation is NOT the bottleneck.")
            log.info(f"   Likely causes:")
            log.info(f"   1. Memory bandwidth bottleneck")
            log.info(f"   2. Router overhead dominates")
            log.info(f"   3. GPU parallelism hides expert count")
        else:
            log.info(f"")
            log.info(f"✓  FINDING: Latency scales with top_k as expected!")
    
    return results

if __name__ == "__main__":
    results = measure_flops_scaling()
    
    # Save results
    import json
    output_file = "results/moe_topk/flops_scaling_analysis.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
