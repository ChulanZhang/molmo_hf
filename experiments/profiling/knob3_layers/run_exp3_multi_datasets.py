#!/usr/bin/env python3
"""
Script to run Exp3 Accuracy Sensitivity V2 (Beam Search) on multiple datasets.
Tests block importance consistency across different datasets and task types.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

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


def run_exp3_v2(
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
    skip_sensitivity: bool = False,
    importance_scores_file: Optional[str] = None,
    log: logging.Logger = None,
) -> bool:
    """Run Exp3 V2 for a single dataset
    
    Returns:
        bool: True if successful, False if failed
    """
    # Create dataset-specific output directory
    dataset_suffix = dataset_name.replace("_", "-")
    output_dir = os.path.join(base_output_dir, dataset_suffix)
    
    log.info(f"{Colors.BRIGHT_CYAN}Dataset: {Colors.BRIGHT_MAGENTA}{dataset_name}{Colors.RESET} ({split}) | "
             f"{Colors.CYAN}Max tokens:{Colors.RESET} {max_new_tokens}")
    
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
        "--max_new_tokens", str(max_new_tokens),
        "--batch_size", str(batch_size),
        "--num_samples", str(num_samples),
        "--beam_width", str(beam_width),
        "--max_blocks_to_remove", str(max_blocks_to_remove),
    ]
    
    if skip_sensitivity:
        cmd.append("--skip_sensitivity")
    
    if importance_scores_file and Path(importance_scores_file).exists():
        cmd.extend(["--importance_scores_file", importance_scores_file])
        if log:
            log.debug(f"Using importance scores from: {importance_scores_file}")
    elif importance_scores_file:
        if log:
            log.warning(f"Importance scores file not found: {importance_scores_file}")
    
    # Create log file
    log_file = log_dir / f"exp3_v2_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Run command
    try:
        with open(log_file, 'w') as f:
            env = dict(os.environ)
            env['PYTHONUNBUFFERED'] = '1'
            env['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                env=env
            )
            
            # Stream stdout to both terminal and log file
            for line in process.stdout:
                if 'OMP_NUM_THREADS' in line or 'Setting OMP_NUM_THREADS' in line or '*****************************************' in line:
                    f.write(line)
                    f.flush()
                    continue
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                f.flush()
            
            process.wait()
        
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
    base_output_dir = "./results/profiling/exp3_accuracy_sensitivity_v2"
    
    # Experiment parameters
    batch_size = 32  # Initial batch size (will be optimized, lower to avoid OOM)
    num_samples = None  # Samples for evaluation (None = use all samples, recommended for sensitivity analysis)
    beam_width = 3  # Top-K candidates to keep
    max_blocks_to_remove = 4  # Maximum blocks to remove
    
    # Skip sensitivity analysis if importance scores already exist
    # Set to True to reuse existing importance scores
    skip_sensitivity = False
    importance_scores_file = None  # Optional: path to existing importance scores
    
    # Auto-detect number of GPUs
    num_gpus = detect_num_gpus()
    num_gpus = int(os.environ.get("NUM_GPUS_OVERRIDE", num_gpus))
    
    # Dataset configurations
    # Selected to cover short, medium, and long answer types
    # Format: (dataset_name, split, max_new_tokens, answer_type)
    # Note: Using "train" split for sensitivity analysis (recommended)
    # For datasets without train set, use validation set
    datasets = [
        # Short answer (1-2 words)
        ("coco_2014_vqa", "train", 16, "short"),
        ("okvqa", "train", 16, "short"),
        ("tally_qa", "train", 16, "short"),  # Use train if available, otherwise test
        
        # Medium answer (phrase/sentence)
        ("text_vqa", "train", 64, "medium"),
        ("st_qa", "train", 32, "medium"),
        ("doc_qa", "train", 32, "medium"),
        
        # Long answer (multiple sentences)
        ("coco_caption", "train", 64, "long"),
    ]
    
    # Parse command line arguments (optional: run only specific dataset)
    specific_dataset = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Print header
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}Exp3 Accuracy Sensitivity V2: Multi-Dataset Beam Search{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_YELLOW}Config:{Colors.RESET} "
             f"{Colors.BRIGHT_WHITE}{num_samples} samples{Colors.RESET} | "
             f"{Colors.CYAN}Beam width:{Colors.RESET} {beam_width} | "
             f"{Colors.CYAN}Max blocks to remove:{Colors.RESET} {max_blocks_to_remove} | "
             f"{Colors.CYAN}GPUs:{Colors.RESET} {num_gpus}")
    log.info(f"{Colors.BRIGHT_YELLOW}Output:{Colors.RESET} {Colors.BRIGHT_WHITE}{base_output_dir}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_YELLOW}Datasets:{Colors.RESET} {len(datasets)} datasets covering short/medium/long answers")
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    
    # Run experiments
    failed_datasets = []
    completed_datasets = []
    
    for dataset_name, split, max_new_tokens, answer_type in datasets:
        if specific_dataset and dataset_name != specific_dataset:
            continue
        
        log.info(f"\n{Colors.BRIGHT_BLUE}Processing {dataset_name} ({answer_type} answer type){Colors.RESET}")
        
        success = run_exp3_v2(
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
            skip_sensitivity=skip_sensitivity,
            importance_scores_file=importance_scores_file,
            log=log,
        )
        
        if success:
            completed_datasets.append((dataset_name, answer_type))
        else:
            failed_datasets.append((dataset_name, answer_type))
            log.warning(f"{Colors.YELLOW}Continuing to next dataset...{Colors.RESET}")
    
    # Summary
    log.info(f"\n{Colors.BRIGHT_GREEN}{'='*60}{Colors.RESET}")
    if failed_datasets:
        log.warning(f"{Colors.BRIGHT_YELLOW}Experiments completed with {len(failed_datasets)} failed dataset(s){Colors.RESET}")
        log.warning(f"{Colors.YELLOW}Failed datasets: {', '.join([d[0] for d in failed_datasets])}{Colors.RESET}")
    else:
        log.info(f"{Colors.BRIGHT_GREEN}All experiments completed successfully!{Colors.RESET}")
    
    if completed_datasets:
        log.info(f"{Colors.CYAN}Completed datasets:{Colors.RESET}")
        for dataset_name, answer_type in completed_datasets:
            log.info(f"  - {dataset_name} ({answer_type})")
    
    log.info(f"{Colors.BRIGHT_GREEN}{'='*60}{Colors.RESET}")


if __name__ == "__main__":
    main()

