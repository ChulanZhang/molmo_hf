#!/usr/bin/env python3
"""
Script to run Beam Search experiments for block combination exploration.
Tests removing up to 4 blocks using beam search algorithm.

Usage:
    python experiments/profiling/knob3_layers/run_beam_search_experiment.py
    python experiments/profiling/knob3_layers/run_beam_search_experiment.py coco_2014_vqa
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# ANSI color codes
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
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            return len(cuda_visible.split(','))
        return 1


def run_beam_search(
    dataset_name: str,
    split: str,
    max_new_tokens: int,
    model_path: str,
    base_output_dir: str,
    num_gpus: int,
    batch_size: int,
    num_samples: int,
    beam_width: int,
    max_blocks_to_remove: int,
    log: logging.Logger = None,
) -> bool:
    """Run beam search experiment for a single dataset
    
    Returns:
        bool: True if successful, False if failed
    """
    # Create dataset-specific output directory
    dataset_suffix = dataset_name.replace("_", "-")
    output_dir = os.path.join(base_output_dir, dataset_suffix)
    
    log.info(f"{Colors.BRIGHT_CYAN}Dataset: {Colors.BRIGHT_MAGENTA}{dataset_name}{Colors.RESET} ({split}) | "
             f"{Colors.CYAN}Max tokens:{Colors.RESET} {max_new_tokens} | "
             f"{Colors.CYAN}Beam width:{Colors.RESET} {beam_width} | "
             f"{Colors.CYAN}Max blocks to remove:{Colors.RESET} {max_blocks_to_remove}")
    
    # Ensure log directory exists
    log_dir = Path(base_output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "torchrun",
        f"--nproc-per-node={num_gpus}",
        "experiments/profiling/knob3_layers/exp3_accuracy_sensitivity_v2.py",
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--dataset_name", dataset_name,
        "--split", split,
        "--batch_size", str(batch_size),
        "--max_new_tokens", str(max_new_tokens),
        "--num_samples", str(num_samples),
        "--beam_width", str(beam_width),
        "--max_blocks_to_remove", str(max_blocks_to_remove),
        "--auto_adjust_batch_size",
    ]
    
    # Create log file
    log_file = log_dir / f"beam_search_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Run command with real-time output streaming
    try:
        with open(log_file, 'w') as f:
            env = dict(os.environ)
            env['PYTHONUNBUFFERED'] = '1'
            env['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=None,  # Let stderr go directly to terminal (for tqdm)
                text=True,
                bufsize=1,
                env=env
            )
            
            # Stream stdout to both terminal and log file
            for line in process.stdout:
                # Filter out torchrun OMP_NUM_THREADS warnings
                if 'OMP_NUM_THREADS' in line or 'Setting OMP_NUM_THREADS' in line or '*****************************************' in line:
                    f.write(line)
                    f.flush()
                    continue
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                f.flush()
            
            process.wait()
        
        # Check if successful
        if process.returncode == 0:
            return True
        else:
            log.error(f"{Colors.RED}Command failed for {dataset_name} (return code {process.returncode}){Colors.RESET}")
            log.error(f"Check log file: {log_file}")
            return False
            
    except Exception as e:
        log.error(f"{Colors.RED}Unexpected error running {dataset_name}: {e}{Colors.RESET}")
        log.error(f"Check log file: {log_file}")
        return False


def main():
    """Main execution"""
    log = setup_logging()
    
    # ============================================================================
    # Experiment Configuration
    # ============================================================================
    
    # Model and output paths
    model_path = "checkpoints"
    base_output_dir = "./results/profiling/exp3_beam_search"
    
    # Beam search configuration
    beam_width = 3  # Number of top candidates to keep at each step
    max_blocks_to_remove = 4  # Maximum number of blocks to remove
    
    # Sampling configuration
    num_samples = 20000  # Use 20K samples for beam search (can be adjusted)
    batch_size = 16  # Initial batch size (will be auto-adjusted)
    
    # Auto-detect number of GPUs
    num_gpus = detect_num_gpus()
    num_gpus = int(os.environ.get("NUM_GPUS_OVERRIDE", num_gpus))
    
    # Dataset configurations
    # Format: (dataset_name, split, max_new_tokens)
    # Using same datasets as run_multi_datasets_h100.py for consistency
    datasets = [
        ("coco_2014_vqa", "train", 16),
        ("text_vqa", "train", 64),
        ("okvqa", "train", 16),
        ("science_qa_img", "train", 16),
        ("st_qa", "train", 32),
        ("doc_qa", "train", 32),
        ("tally_qa", "train", 16),  # Note: tally_qa uses "test" in core_exp, but we use "train" for sensitivity
        ("mmmu", "train", 16),
        ("coco_caption", "train", 64),
    ]
    
    # Parse command line arguments (optional: run only specific dataset)
    specific_dataset = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Print header
    log.info(f"{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}Beam Search Experiment: Block Combination Exploration{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_YELLOW}Config:{Colors.RESET} {Colors.BRIGHT_WHITE}{num_samples} samples{Colors.RESET} | "
             f"{Colors.CYAN}Beam width:{Colors.RESET} {beam_width} | "
             f"{Colors.CYAN}Max blocks to remove:{Colors.RESET} {max_blocks_to_remove} | "
             f"{Colors.CYAN}GPUs:{Colors.RESET} {num_gpus}")
    log.info(f"{Colors.BRIGHT_YELLOW}Output:{Colors.RESET} {Colors.BRIGHT_WHITE}{base_output_dir}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}")
    log.info("")
    log.info(f"{Colors.CYAN}Beam Search Algorithm:{Colors.RESET}")
    log.info(f"  Step 1: Remove 1 block, test all 16, keep top {beam_width}")
    log.info(f"  Step 2: Remove 2 blocks, test ~{beam_width * 15} configs, keep top {beam_width}")
    log.info(f"  Step 3: Remove 3 blocks, test ~{beam_width * 14} configs, keep top {beam_width}")
    log.info(f"  Step 4: Remove 4 blocks, test ~{beam_width * 13} configs, keep top {beam_width}")
    log.info(f"  Total: ~{16 + beam_width * (15 + 14 + 13)} configurations tested")
    log.info("")
    
    # Run experiments
    failed_datasets = []
    for dataset_name, split, max_new_tokens in datasets:
        if specific_dataset and dataset_name != specific_dataset:
            continue
        
        success = run_beam_search(
            dataset_name=dataset_name,
            split=split,
            max_new_tokens=max_new_tokens,
            model_path=model_path,
            base_output_dir=base_output_dir,
            num_gpus=num_gpus,
            batch_size=batch_size,
            num_samples=num_samples,
            beam_width=beam_width,
            max_blocks_to_remove=max_blocks_to_remove,
            log=log,
        )
        
        if not success:
            failed_datasets.append(dataset_name)
            log.warning(f"{Colors.YELLOW}Continuing to next dataset...{Colors.RESET}")
    
    # Completion message
    log.info(f"{Colors.BRIGHT_GREEN}{'='*80}{Colors.RESET}")
    if failed_datasets:
        log.warning(f"{Colors.BRIGHT_YELLOW}Experiments completed with {len(failed_datasets)} failed dataset(s){Colors.RESET}")
        log.warning(f"{Colors.YELLOW}Failed datasets: {', '.join(failed_datasets)}{Colors.RESET}")
    else:
        log.info(f"{Colors.BRIGHT_GREEN}All beam search experiments completed successfully!{Colors.RESET}")
    
    # List successful datasets
    completed_datasets = []
    for dataset_name, split, max_new_tokens in datasets:
        if specific_dataset and dataset_name != specific_dataset:
            continue
        if dataset_name not in failed_datasets:
            completed_datasets.append(dataset_name)
    if completed_datasets:
        dataset_dirs = [d.replace("_", "-") for d in completed_datasets]
        log.info(f"{Colors.CYAN}Successful datasets: {', '.join(dataset_dirs)}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_GREEN}{'='*80}{Colors.RESET}")


if __name__ == "__main__":
    main()

