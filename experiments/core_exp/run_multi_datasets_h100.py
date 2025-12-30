#!/usr/bin/env python3
"""
Script to run combined profiling on multiple datasets.
Supports both short-answer VQA datasets and long-answer datasets.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# ANSI color codes (same as in acc_lat_profiling.py)
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BRIGHT_CYAN = '\033[1;36m'
    BRIGHT_MAGENTA = '\033[1;35m'
    BRIGHT_YELLOW = '\033[1;33m'
    BRIGHT_WHITE = '\033[1;37m'
    BRIGHT_GREEN = '\033[1;32m'
    BRIGHT_BLUE = '\033[1;34m'
    CYAN = '\033[0;36m'
    YELLOW = '\033[0;33m'
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'


def setup_logging():
    """Setup colored logging"""
    try:
        import colorlog
        handler = colorlog.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers = []
        root_logger.addHandler(handler)
        return logging.getLogger(__name__)
    except ImportError:
        # Fallback to basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)


def detect_num_gpus() -> int:
    """Auto-detect number of GPUs"""
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, check=True)
        return len(result.stdout.strip().split('\n'))
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            return len(cuda_visible.split(','))
        return 1


def run_combined_profiling(
    dataset_name: str,
    split: str,
    max_new_tokens: int,
    model_path: str,
    base_output_dir: str,
    num_gpus: int,
    tier_list: List[str],
    top_k_list: List[int],
    num_active_blocks_list: List[int],
    sampling_strategy: str,
    num_samples: int,
    num_runs_per_sample: int,
    log: logging.Logger,
) -> None:
    """Run combined profiling for a single dataset"""
    output_dir = base_output_dir
    
    # Concise dataset header (detailed config is logged in acc_lat_profiling.py)
    log.info(f"{Colors.BRIGHT_CYAN}Dataset: {Colors.BRIGHT_MAGENTA}{dataset_name}{Colors.RESET} ({split}) | "
             f"{Colors.CYAN}Max tokens:{Colors.RESET} {max_new_tokens}")
    
    # Ensure log directory exists
    log_dir = Path(base_output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"combined_profiling_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # Don't log log file location - it's not essential information
    
    # Build command
    cmd = [
        "torchrun",
        f"--nproc-per-node={num_gpus}",
        "experiments/core_exp/acc_lat_profiling.py",
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--dataset_name", dataset_name,
        "--split", split,
        "--max_new_tokens", str(max_new_tokens),
        "--tier_list"] + tier_list + [
        "--top_k_list"] + [str(k) for k in top_k_list] + [
        "--num_active_blocks_list"] + [str(b) for b in num_active_blocks_list] + [
        "--sampling_strategy", sampling_strategy,
        "--num_samples", str(num_samples),
        "--num_runs_per_sample", str(num_runs_per_sample),
    ]
    
    # Run command and tee output to log file
    # IMPORTANT: Don't redirect stdout to PIPE, as it breaks tqdm's TTY detection
    # Instead, use unbuffered mode and let the subprocess write directly to terminal
    # We'll capture output using a custom tee-like approach
    import threading
    
    with open(log_file, 'w') as f:
        # Create environment with warnings suppressed
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
        env['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'  # Suppress torchrun warnings (must be OFF, INFO, or DETAIL)
        
        # IMPORTANT: Don't redirect stderr to stdout - let tqdm write directly to stderr
        # This allows tqdm to detect TTY and use single-line mode
        # We'll capture stdout for logging, but stderr goes directly to terminal
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,  # Capture stdout for logging
            stderr=None,  # Let stderr go directly to terminal (for tqdm)
            text=True,
            bufsize=1,
            env=env
        )
        
        # Stream stdout to both terminal and log file, filtering out torchrun warnings
        for line in process.stdout:
            # Filter out torchrun OMP_NUM_THREADS warnings
            if 'OMP_NUM_THREADS' in line or 'Setting OMP_NUM_THREADS' in line or '*****************************************' in line:
                # Still write to log file but not to stdout
                f.write(line)
                f.flush()
                continue
            sys.stdout.write(line)  # Write to actual stdout
            sys.stdout.flush()
            f.write(line)
            f.flush()
        
        process.wait()
    
    if process.returncode != 0:
        log.error(f"Command failed with return code {process.returncode}")
        sys.exit(process.returncode)
    
    # Don't log completion here - it's already logged in acc_lat_profiling.py


def main():
    """Main execution"""
    log = setup_logging()
    
    # ============================================================================
    # Experiment Configuration
    # ============================================================================
    # Modify these values directly to change experiment settings
    
    # Model and output paths
    model_path = "checkpoints"
    base_output_dir = "./results/core_exp_h100"
    
    # Sampling configuration
    num_samples = 36
    sampling_strategy = "balanced"
    num_runs_per_sample = 1
    
    # Tier-based vision token control
    tier_list = ["low", "medium", "high"]  # Available: "low", "medium", "high"
    top_k_list = [4, 8, 12]  # MoE top-k values
    num_active_blocks_list = [12, 14, 16]  # Number of active transformer blocks
    
    # Auto-detect number of GPUs (can be overridden with NUM_GPUS_OVERRIDE env var)
    num_gpus = detect_num_gpus()
    num_gpus = int(os.environ.get("NUM_GPUS_OVERRIDE", num_gpus))
    
    # Dataset configurations
    # Format: (dataset_name, split, max_new_tokens)
    datasets = [
        ("coco_2014_vqa", "validation", 16),
        ("text_vqa", "validation", 64),
        ("okvqa", "validation", 16),
        ("science_qa_img", "validation", 16),
        ("st_qa", "validation", 32),
        ("doc_qa", "validation", 32),
        ("tally_qa", "test", 16),
        ("mmmu", "validation", 16),
        ("coco_caption", "validation", 64),
    ]
    
    # Parse command line arguments (optional: run only specific dataset)
    specific_dataset = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Print concise header
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}Multi-Dataset Combined Profiling{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_YELLOW}Config:{Colors.RESET} {Colors.BRIGHT_WHITE}{num_samples} samples{Colors.RESET} | "
             f"{Colors.CYAN}Tiers:{Colors.RESET} {tier_list} | "
             f"{Colors.CYAN}Top K:{Colors.RESET} {top_k_list} | "
             f"{Colors.CYAN}Blocks:{Colors.RESET} {num_active_blocks_list} | "
             f"{Colors.CYAN}GPUs:{Colors.RESET} {num_gpus}")
    log.info(f"{Colors.BRIGHT_YELLOW}Output:{Colors.RESET} {Colors.BRIGHT_WHITE}{base_output_dir}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    # Run experiments
    for dataset_name, split, max_new_tokens in datasets:
        if specific_dataset and dataset_name != specific_dataset:
            continue
        
        run_combined_profiling(
            dataset_name=dataset_name,
            split=split,
            max_new_tokens=max_new_tokens,
            model_path=model_path,
            base_output_dir=base_output_dir,
            num_gpus=num_gpus,
            tier_list=tier_list,
            top_k_list=top_k_list,
            num_active_blocks_list=num_active_blocks_list,
            sampling_strategy=sampling_strategy,
            num_samples=num_samples,
            num_runs_per_sample=num_runs_per_sample,
            log=log,
        )
    
    # Concise completion message
    log.info(f"{Colors.BRIGHT_GREEN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_GREEN}All experiments completed!{Colors.RESET}")
    # List actual dataset names that were run
    completed_datasets = []
    for dataset_name, split, max_new_tokens in datasets:
        if specific_dataset and dataset_name != specific_dataset:
            continue
        completed_datasets.append(dataset_name)
    if completed_datasets:
        dataset_dirs = [d.replace("_", "-") for d in completed_datasets]
        log.info(f"{Colors.CYAN}Results: {', '.join(dataset_dirs)}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_GREEN}{'='*60}{Colors.RESET}")


if __name__ == "__main__":
    main()

