#!/usr/bin/env python3
"""
Merge rank-specific JSON result files from previous experiments.

This script finds all *_rank*.json files and merges them into final results,
then deletes the rank-specific files.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np
import sys
import os


def merge_config_results(gathered_configs: List[Dict], template_config: Dict) -> Dict:
    """
    Merge configuration results from all ranks.
    
    Args:
        gathered_configs: List of config results from all ranks (may contain None for failed ranks)
        template_config: Template config result (from rank 0) for structure reference
    
    Returns:
        Merged configuration result
    """
    # Collect all per_sample_results from all ranks
    all_per_sample = []
    for rank_config in gathered_configs:
        if rank_config is not None:
            all_per_sample.extend(rank_config.get("per_sample_results", []))
    
    if not all_per_sample:
        # No samples collected, return template with empty results
        return template_config
    
    # Recompute statistics from all samples
    merged_config = template_config.copy()
    
    # Accuracy statistics
    accuracy_values = [s["accuracy"] for s in all_per_sample if "accuracy" in s]
    if accuracy_values:
        merged_config["accuracy"] = float(np.mean(accuracy_values))
        merged_config["accuracy_std"] = float(np.std(accuracy_values))
        merged_config["num_samples"] = len(all_per_sample)
    
    # Recompute aggregate stats
    stage_keys = ["T_vision_encoder", "T_projector", "T_vision_total", 
                 "T_LLM_prefill", "T_LLM_decode", "T_total", "T_decode_per_token"]
    aggregate_stats = {}
    for key in stage_keys:
        values = [s[key] for s in all_per_sample if key in s]
        if values:
            aggregate_stats[f"{key}_mean"] = float(np.mean(values))
            aggregate_stats[f"{key}_std"] = float(np.std(values))
            aggregate_stats[f"{key}_p50"] = float(np.percentile(values, 50))
            aggregate_stats[f"{key}_p95"] = float(np.percentile(values, 95))
            aggregate_stats[f"{key}_p99"] = float(np.percentile(values, 99))
    
    # Vision tokens statistics
    vision_token_values = [s["actual_vision_tokens"] for s in all_per_sample if "actual_vision_tokens" in s]
    if vision_token_values:
        aggregate_stats["vision_tokens_mean"] = float(np.mean(vision_token_values))
        aggregate_stats["vision_tokens_std"] = float(np.std(vision_token_values))
    
    # Actual num_crops statistics
    actual_num_crops_list = [s.get("actual_num_crops", 0) for s in all_per_sample]
    if actual_num_crops_list:
        aggregate_stats["selected_crops_mean"] = float(np.mean(actual_num_crops_list))
        aggregate_stats["selected_crops_std"] = float(np.std(actual_num_crops_list))
    
    # Target vision tokens statistics
    target_vision_tokens_list = [s.get("target_vision_tokens", 0) for s in all_per_sample]
    if target_vision_tokens_list:
        aggregate_stats["target_vision_tokens_mean"] = float(np.mean(target_vision_tokens_list))
        aggregate_stats["target_vision_tokens_std"] = float(np.std(target_vision_tokens_list))
    
    # Selected crops distribution
    selected_crops_distribution = {}
    for crops in actual_num_crops_list:
        selected_crops_distribution[crops] = selected_crops_distribution.get(crops, 0) + 1
    
    # Update merged_config with merged statistics
    merged_config["selected_crops_distribution"] = selected_crops_distribution
    merged_config["selected_crops_mean"] = aggregate_stats.get("selected_crops_mean", 0.0)
    merged_config["selected_crops_std"] = aggregate_stats.get("selected_crops_std", 0.0)
    merged_config["target_vision_tokens_mean"] = aggregate_stats.get("target_vision_tokens_mean", 0.0)
    merged_config["target_vision_tokens_std"] = aggregate_stats.get("target_vision_tokens_std", 0.0)
    merged_config["actual_vision_tokens_mean"] = aggregate_stats.get("vision_tokens_mean", 0.0)
    merged_config["actual_vision_tokens_std"] = aggregate_stats.get("vision_tokens_std", 0.0)
    merged_config["aggregate_stats"] = aggregate_stats
    merged_config["per_sample_results"] = all_per_sample
    
    return merged_config


def parse_rank_filename(filename: str) -> Dict[str, Any]:
    """
    Parse rank-specific filename to extract configuration info.
    
    Format: {task_name}_imgsizetier-{tier}_crops{crops}_topk{top_k}_blocks{blocks}_rank{rank}.json
    
    Returns:
        Dict with keys: task_name, tier, crops, top_k, blocks, rank, base_name
    """
    # Remove .json extension
    base = filename.replace(".json", "")
    
    # Extract rank
    rank_match = re.search(r'_rank(\d+)$', base)
    if not rank_match:
        return None
    rank = int(rank_match.group(1))
    base_name = base.replace(f"_rank{rank}", "")
    
    # Parse: {task_name}_imgsizetier-{tier}_crops{crops}_topk{top_k}_blocks{blocks}
    pattern = r'^(.+?)_imgsizetier-([^_]+)_crops(\d+)_topk(\d+)_blocks(\d+)$'
    match = re.match(pattern, base_name)
    if not match:
        return None
    
    return {
        "task_name": match.group(1),
        "tier": match.group(2),
        "crops": int(match.group(3)),
        "top_k": int(match.group(4)),
        "blocks": int(match.group(5)),
        "rank": rank,
        "base_name": base_name,
        "full_path": filename,
    }


def find_rank_files(base_dir: str) -> Dict[str, List[Path]]:
    """
    Find all rank-specific JSON files and group them by configuration.
    
    Returns:
        Dict mapping config_key (tuple) to list of file paths
    """
    base_path = Path(base_dir)
    rank_files = defaultdict(list)
    
    # Find all rank-specific files
    for rank_file in base_path.rglob("*_rank*.json"):
        if "_rank" not in rank_file.name:
            continue
        
        parsed = parse_rank_filename(rank_file.name)
        if parsed is None:
            print(f"Warning: Could not parse filename: {rank_file.name}")
            continue
        
        # Group by configuration (task_name, tier, top_k, blocks)
        # Note: crops may vary, so we use the base_name pattern
        config_key = (rank_file.parent, parsed["base_name"])
        rank_files[config_key].append((rank_file, parsed))
    
    return rank_files


def merge_rank_files_for_config(rank_files: List[tuple], output_dir: Path) -> bool:
    """
    Merge rank files for a single configuration.
    
    Args:
        rank_files: List of (file_path, parsed_info) tuples
        output_dir: Output directory for merged file
    
    Returns:
        True if successful, False otherwise
    """
    # Load all rank files
    gathered_configs = []
    template_config = None
    
    # Sort by rank to ensure consistent ordering
    rank_files_sorted = sorted(rank_files, key=lambda x: x[1]["rank"])
    
    for file_path, parsed_info in rank_files_sorted:
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            if template_config is None:
                template_config = config_data
            
            gathered_configs.append(config_data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            gathered_configs.append(None)
    
    if template_config is None:
        print(f"Warning: No valid config files found for {rank_files[0][0]}")
        return False
    
    # Merge results
    try:
        merged_config = merge_config_results(gathered_configs, template_config)
    except Exception as e:
        print(f"Error merging configs: {e}")
        return False
    
    # Generate output filename (without rank suffix)
    parsed = rank_files_sorted[0][1]
    output_filename = f"{parsed['base_name']}.json"
    output_file = output_dir / output_filename
    
    # Check if merged file already exists
    if output_file.exists():
        print(f"Warning: Merged file already exists: {output_file}")
        print(f"  Skipping merge to avoid overwriting. Delete existing file to re-merge.")
        return False
    
    # Save merged result
    try:
        with open(output_file, 'w') as f:
            json.dump(merged_config, f, indent=2)
        print(f"✓ Merged {len(rank_files_sorted)} rank files -> {output_file.name}")
    except Exception as e:
        print(f"Error saving merged file {output_file}: {e}")
        return False
    
    # Delete rank-specific files
    deleted_count = 0
    for file_path, _ in rank_files_sorted:
        try:
            file_path.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")
    
    if deleted_count > 0:
        print(f"  Deleted {deleted_count} rank-specific file(s)")
    
    return True


def main():
    """Main function to merge all rank-specific files"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge rank-specific JSON result files")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./results/core_exp_h100",
        help="Base directory to search for rank files (default: ./results/core_exp_h100)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run: show what would be merged without actually merging"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return
    
    print(f"Searching for rank-specific files in: {base_dir}")
    print("=" * 60)
    
    # Find all rank files grouped by configuration
    rank_files_by_config = find_rank_files(str(base_dir))
    
    if not rank_files_by_config:
        print("No rank-specific files found.")
        return
    
    print(f"Found {len(rank_files_by_config)} configuration(s) with rank files")
    print()
    
    # Process each configuration
    merged_count = 0
    skipped_count = 0
    error_count = 0
    
    for config_key, rank_files in sorted(rank_files_by_config.items()):
        output_dir, base_name = config_key
        ranks = sorted([f[1]["rank"] for f in rank_files])
        
        print(f"Config: {base_name}")
        print(f"  Ranks: {ranks} ({len(rank_files)} files)")
        print(f"  Directory: {output_dir}")
        
        if args.dry_run:
            print(f"  [DRY RUN] Would merge to: {output_dir / f'{base_name}.json'}")
            print()
            continue
        
        # Check if merged file already exists
        merged_file = output_dir / f"{base_name}.json"
        if merged_file.exists():
            print(f"  ⚠ Merged file already exists, skipping")
            skipped_count += 1
            print()
            continue
        
        # Merge files
        success = merge_rank_files_for_config(rank_files, output_dir)
        if success:
            merged_count += 1
        else:
            error_count += 1
        print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print(f"  Configurations processed: {len(rank_files_by_config)}")
    if not args.dry_run:
        print(f"  Successfully merged: {merged_count}")
        print(f"  Skipped (already exists): {skipped_count}")
        print(f"  Errors: {error_count}")
    else:
        print(f"  [DRY RUN] Would merge: {len(rank_files_by_config)} configuration(s)")


if __name__ == "__main__":
    main()

