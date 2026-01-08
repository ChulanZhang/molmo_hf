"""
Analyze latency ranges for each tier from core experiment results.
This helps evaluate if budget-only prediction is feasible.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.controller.core_exp_data_loader import CoreExpDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def analyze_tier_latency_ranges(
    core_exp_results: list,
) -> dict:
    """
    Analyze latency ranges for each tier from core experiment results.
    
    IMPORTANT: We analyze prefill latency and decode per-token latency separately,
    NOT end-to-end latency, because:
    - Low configurations may produce longer outputs (hallucination), making
      end-to-end latency misleading
    - Prefill latency (vision encoder + projector + LLM prefill) is determined by config
    - Decode per-token latency is determined by config
    - End-to-end latency = prefill + decode_per_token * output_tokens
      (output_tokens is model-dependent, not config-dependent)
    
    Args:
        core_exp_results: List of core experiment result dictionaries
    
    Returns:
        tier_stats: {
            'low': {
                'prefill': {'mean': ..., 'std': ..., 'min': ..., 'max': ..., 'p50': ..., 'p95': ...},
                'decode_per_token': {'mean': ..., 'std': ..., 'min': ..., 'max': ..., 'p50': ..., 'p95': ...},
            },
            'medium': {...},
            'high': {...},
        }
    """
    import numpy as np
    from collections import defaultdict
    
    tier_prefill_latencies = defaultdict(list)
    tier_decode_per_token_latencies = defaultdict(list)
    
    for result in core_exp_results:
        tier = result.get('tier', 'medium')
        
        # Extract prefill latency components
        # Try multiple possible field names (per-sample or aggregate)
        T_vision_total = result.get('T_vision_total')
        if T_vision_total is None or T_vision_total == 0:
            T_vision_total = result.get('T_vision_total_mean', 0.0)
        
        T_LLM_prefill = result.get('T_LLM_prefill')
        if T_LLM_prefill is None or T_LLM_prefill == 0:
            T_LLM_prefill = result.get('T_LLM_prefill_mean', 0.0)
        
        # Prefill latency = vision total + LLM prefill
        prefill_latency = T_vision_total + T_LLM_prefill
        
        # Extract decode latency
        T_LLM_decode = result.get('T_LLM_decode', 0.0)
        output_tokens = result.get('output_tokens', 0)
        
        # Decode per-token latency
        # Priority: Use T_decode_per_token field directly if available and valid
        # Otherwise, calculate from T_LLM_decode / output_tokens
        decode_per_token = None
        
        # First, try to get per-token latency directly from the field
        decode_per_token = result.get('T_decode_per_token')
        
        # If not available or 0, try to calculate from T_LLM_decode and output_tokens
        if (decode_per_token is None or decode_per_token == 0) and output_tokens > 0 and T_LLM_decode > 0:
            decode_per_token = T_LLM_decode / output_tokens
        
        # If still not available, try aggregate mean (but this is less accurate)
        if (decode_per_token is None or decode_per_token == 0):
            decode_per_token = result.get('T_decode_per_token_mean')
        
        # Only add to statistics if we have valid decode_per_token
        # Note: We include all samples with valid decode_per_token, regardless of output_tokens
        # The T_decode_per_token field should already account for measurement precision
        if prefill_latency > 0:
            tier_prefill_latencies[tier].append(prefill_latency)
        
        # Include decode_per_token if it's valid (use the field value directly, trust the measurement)
        if decode_per_token is not None and decode_per_token > 0:
            tier_decode_per_token_latencies[tier].append(decode_per_token)
    
    # Debug: Print tier distribution and sample values
    log.info(f"Tier prefill distribution: {dict((k, len(v)) for k, v in tier_prefill_latencies.items())}")
    log.info(f"Tier decode_per_token distribution: {dict((k, len(v)) for k, v in tier_decode_per_token_latencies.items())}")
    
    # Debug: Print sample decode_per_token values for each tier
    for tier in ['low', 'medium', 'high']:
        if tier in tier_decode_per_token_latencies and len(tier_decode_per_token_latencies[tier]) > 0:
            sample_values = tier_decode_per_token_latencies[tier][:10]  # First 10 samples
            log.info(f"{tier} tier decode_per_token samples (first 10): {[f'{v:.2f}' for v in sample_values]}")
    
    def compute_stats(latencies):
        """Compute statistics for a list of latencies."""
        if len(latencies) == 0:
            return {}
        latencies = np.array(latencies)
        return {
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'count': len(latencies),
        }
    
    tier_stats = {}
    all_tiers = set(list(tier_prefill_latencies.keys()) + list(tier_decode_per_token_latencies.keys()))
    
    for tier in all_tiers:
        tier_stats[tier] = {
            'prefill': compute_stats(tier_prefill_latencies.get(tier, [])),
            'decode_per_token': compute_stats(tier_decode_per_token_latencies.get(tier, [])),
        }
    
    return tier_stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tier latency ranges",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing core experiment results"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=["text_vqa", "coco_2014_vqa"],
        help="Dataset names to load"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="tier_latency_ranges.json",
        help="Output file for analysis"
    )
    
    args = parser.parse_args()
    
    log.info("Loading core experiment results...")
    data_loader = CoreExpDataLoader(args.results_dir)
    samples = data_loader.load_multiple_datasets(args.dataset_names)
    
    if not samples:
        raise ValueError("No samples found!")
    
    log.info(f"Analyzing {len(samples)} samples...")
    
    # Debug: Check sample structure
    if samples:
        sample_keys = list(samples[0].keys())
        log.info(f"Sample keys: {sample_keys}")
        log.info(f"First sample tier: {samples[0].get('tier', 'NOT FOUND')}")
        log.info(f"First sample latency fields:")
        log.info(f"  T_vision_total: {samples[0].get('T_vision_total', 'NOT FOUND')}")
        log.info(f"  T_LLM_prefill: {samples[0].get('T_LLM_prefill', 'NOT FOUND')}")
        log.info(f"  T_LLM_decode: {samples[0].get('T_LLM_decode', 'NOT FOUND')}")
        log.info(f"  T_decode_per_token: {samples[0].get('T_decode_per_token', 'NOT FOUND')}")
        log.info(f"  output_tokens: {samples[0].get('output_tokens', 'NOT FOUND')}")
    
    tier_stats = analyze_tier_latency_ranges(samples)
    
    log.info(f"Tier stats keys: {list(tier_stats.keys())}")
    if tier_stats:
        for tier, stats in tier_stats.items():
            log.info(f"  {tier}: prefill_count={stats.get('prefill', {}).get('count', 0)}, decode_count={stats.get('decode_per_token', {}).get('count', 0)}")
    
    log.info("=" * 80)
    log.info("Tier Latency Ranges Analysis")
    log.info("=" * 80)
    log.info("\nNOTE: We analyze prefill latency and decode per-token latency separately,")
    log.info("NOT end-to-end latency, because low configs may produce longer outputs.")
    log.info("=" * 80)
    
    for tier in ['low', 'medium', 'high']:
        if tier in tier_stats:
            stats = tier_stats[tier]
            log.info(f"\n{tier.upper()} Tier:")
            
            # Prefill latency
            prefill_stats = stats.get('prefill', {})
            if prefill_stats:
                log.info(f"\n  Prefill Latency (vision encoder + projector + LLM prefill):")
                log.info(f"    Count: {prefill_stats.get('count', 0)}")
                log.info(f"    Mean:  {prefill_stats.get('mean', 0):.2f} ms")
                log.info(f"    Std:   {prefill_stats.get('std', 0):.2f} ms")
                log.info(f"    Min:   {prefill_stats.get('min', 0):.2f} ms")
                log.info(f"    Max:   {prefill_stats.get('max', 0):.2f} ms")
                log.info(f"    P50:   {prefill_stats.get('p50', 0):.2f} ms")
                log.info(f"    P95:   {prefill_stats.get('p95', 0):.2f} ms")
                log.info(f"    P99:   {prefill_stats.get('p99', 0):.2f} ms")
            
            # Decode per-token latency
            decode_stats = stats.get('decode_per_token', {})
            if decode_stats:
                log.info(f"\n  Decode Per-Token Latency:")
                log.info(f"    Count: {decode_stats.get('count', 0)}")
                log.info(f"    Mean:  {decode_stats.get('mean', 0):.2f} ms/token")
                log.info(f"    Std:   {decode_stats.get('std', 0):.2f} ms/token")
                log.info(f"    Min:   {decode_stats.get('min', 0):.2f} ms/token")
                log.info(f"    Max:   {decode_stats.get('max', 0):.2f} ms/token")
                log.info(f"    P50:   {decode_stats.get('p50', 0):.2f} ms/token")
                log.info(f"    P95:   {decode_stats.get('p95', 0):.2f} ms/token")
                log.info(f"    P99:   {decode_stats.get('p99', 0):.2f} ms/token")
    
    # Check overlap for prefill latency
    log.info("\n" + "=" * 80)
    log.info("Overlap Analysis - Prefill Latency (for budget-only prediction feasibility)")
    log.info("=" * 80)
    
    if all(tier in tier_stats for tier in ['low', 'medium', 'high']):
        low_prefill = tier_stats['low'].get('prefill', {})
        medium_prefill = tier_stats['medium'].get('prefill', {})
        high_prefill = tier_stats['high'].get('prefill', {})
        
        if low_prefill and medium_prefill and high_prefill:
            low_max = low_prefill.get('max', 0)
            low_p95 = low_prefill.get('p95', 0)
            medium_min = medium_prefill.get('min', 0)
            medium_max = medium_prefill.get('max', 0)
            medium_p95 = medium_prefill.get('p95', 0)
            high_min = high_prefill.get('min', 0)
            high_p95 = high_prefill.get('p95', 0)
            
            log.info(f"\nPrefill Latency:")
            log.info(f"  Low tier max:     {low_max:.2f} ms")
            log.info(f"  Low tier P95:     {low_p95:.2f} ms")
            log.info(f"  Medium tier min:  {medium_min:.2f} ms")
            log.info(f"  Medium tier max:  {medium_max:.2f} ms")
            log.info(f"  Medium tier P95:  {medium_p95:.2f} ms")
            log.info(f"  High tier min:    {high_min:.2f} ms")
            log.info(f"  High tier P95:    {high_p95:.2f} ms")
            
            # Check overlaps
            low_medium_overlap = low_max > medium_min
            medium_high_overlap = medium_max > high_min
            
            log.info(f"\nOverlap Analysis (Prefill):")
            log.info(f"  Low-Medium overlap: {low_medium_overlap}")
            if low_medium_overlap:
                overlap_range = min(low_max, medium_max) - max(low_p95, medium_min)
                log.info(f"    Overlap range: {overlap_range:.2f} ms")
            
            log.info(f"  Medium-High overlap: {medium_high_overlap}")
            if medium_high_overlap:
                overlap_range = min(medium_max, high_p95) - max(medium_p95, high_min)
                log.info(f"    Overlap range: {overlap_range:.2f} ms")
            
            # Feasibility assessment
            log.info(f"\nFeasibility Assessment for Budget-Only Prediction (Prefill):")
            if low_medium_overlap or medium_high_overlap:
                log.info("  ⚠️  There is overlap between tiers in prefill latency")
                log.info("  ⚠️  Budget-only prediction may have ambiguity")
                log.info("  💡  Consider using thresholds with confidence intervals")
                log.info("  💡  Or use budget + vision (Variant 3) for better accuracy")
            else:
                log.info("  ✅  Tiers are well-separated in prefill latency")
                log.info("  ✅  Budget-only prediction should work well for prefill")
    
    # Check overlap for decode per-token latency
    log.info("\n" + "=" * 80)
    log.info("Overlap Analysis - Decode Per-Token Latency")
    log.info("=" * 80)
    
    if all(tier in tier_stats for tier in ['low', 'medium', 'high']):
        low_decode = tier_stats['low'].get('decode_per_token', {})
        medium_decode = tier_stats['medium'].get('decode_per_token', {})
        high_decode = tier_stats['high'].get('decode_per_token', {})
        
        if low_decode and medium_decode and high_decode:
            low_max = low_decode.get('max', 0)
            low_p95 = low_decode.get('p95', 0)
            medium_min = medium_decode.get('min', 0)
            medium_max = medium_decode.get('max', 0)
            medium_p95 = medium_decode.get('p95', 0)
            high_min = high_decode.get('min', 0)
            high_p95 = high_decode.get('p95', 0)
            
            log.info(f"\nDecode Per-Token Latency:")
            log.info(f"  Low tier max:     {low_max:.2f} ms/token")
            log.info(f"  Low tier P95:     {low_p95:.2f} ms/token")
            log.info(f"  Medium tier min:  {medium_min:.2f} ms/token")
            log.info(f"  Medium tier max:  {medium_max:.2f} ms/token")
            log.info(f"  Medium tier P95:  {medium_p95:.2f} ms/token")
            log.info(f"  High tier min:    {high_min:.2f} ms/token")
            log.info(f"  High tier P95:    {high_p95:.2f} ms/token")
            
            # Check overlaps
            low_medium_overlap = low_max > medium_min
            medium_high_overlap = medium_max > high_min
            
            log.info(f"\nOverlap Analysis (Decode Per-Token):")
            log.info(f"  Low-Medium overlap: {low_medium_overlap}")
            if low_medium_overlap:
                overlap_range = min(low_max, medium_max) - max(low_p95, medium_min)
                log.info(f"    Overlap range: {overlap_range:.2f} ms/token")
            
            log.info(f"  Medium-High overlap: {medium_high_overlap}")
            if medium_high_overlap:
                overlap_range = min(medium_max, high_p95) - max(medium_p95, high_min)
                log.info(f"    Overlap range: {overlap_range:.2f} ms/token")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(tier_stats, f, indent=2)
    
    log.info(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()

