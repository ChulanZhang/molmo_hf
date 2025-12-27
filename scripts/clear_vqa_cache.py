#!/usr/bin/env python3
"""
Clear corrupted VQA dataset cache.
This script helps fix the "TypeError: must be called with a dataclass type or instance" error
that occurs when HuggingFace datasets cache is corrupted.
"""

import os
import shutil
import glob
import argparse

def clear_vqa_cache():
    """Clear VQA-related cache directories."""
    # Determine cache directory
    if "HF_HOME" in os.environ:
        hf_home = os.environ["HF_HOME"]
        cache_dir = os.path.join(hf_home, "datasets")
        print(f"Using HF_HOME={hf_home}")
    else:
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        print(f"Using default cache location: {cache_dir}")
    
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    # Find all VQA-related cache directories
    patterns = [
        os.path.join(cache_dir, "*vqa*"),
        os.path.join(cache_dir, "*VQA*"),
        os.path.join(cache_dir, "*vqa_v2*"),
        os.path.join(cache_dir, "*VQAv2*"),
    ]
    
    all_cache_dirs = set()
    for pattern in patterns:
        cache_dirs = glob.glob(pattern)
        all_cache_dirs.update(cache_dirs)
    
    if all_cache_dirs:
        print(f"\nFound {len(all_cache_dirs)} VQA cache directory(ies):")
        for cache_dir_path in sorted(all_cache_dirs):
            print(f"  - {cache_dir_path}")
        
        response = input("\nDo you want to delete these cache directories? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            for cache_dir_path in all_cache_dirs:
                try:
                    shutil.rmtree(cache_dir_path)
                    print(f"✓ Removed: {cache_dir_path}")
                except Exception as e:
                    print(f"✗ Failed to remove {cache_dir_path}: {e}")
            print("\nCache cleared! Please re-run your experiment.")
        else:
            print("Cache clearing cancelled.")
    else:
        print("No VQA cache directories found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear corrupted VQA dataset cache")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    
    if args.force:
        # Auto-clear without prompt
        if "HF_HOME" in os.environ:
            hf_home = os.environ["HF_HOME"]
            cache_dir = os.path.join(hf_home, "datasets")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        
        patterns = [
            os.path.join(cache_dir, "*vqa*"),
            os.path.join(cache_dir, "*VQA*"),
            os.path.join(cache_dir, "*vqa_v2*"),
            os.path.join(cache_dir, "*VQAv2*"),
        ]
        
        all_cache_dirs = set()
        for pattern in patterns:
            all_cache_dirs.update(glob.glob(pattern))
        
        for cache_dir_path in all_cache_dirs:
            try:
                shutil.rmtree(cache_dir_path)
                print(f"Removed: {cache_dir_path}")
            except Exception as e:
                print(f"Failed to remove {cache_dir_path}: {e}")
    else:
        clear_vqa_cache()
