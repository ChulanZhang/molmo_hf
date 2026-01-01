#!/usr/bin/env python3
"""
Script to run combined profiling on multiple datasets.
Supports both short-answer VQA datasets and long-answer datasets.
Optimized for A100-40GB memory constraints.
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
    enable_memory_optimization: bool,
    importance_scores_file: Optional[str] = None,
    log: logging.Logger = None,
) -> None:
    """Run combined profiling for a single dataset"""
    output_dir = base_output_dir
    
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}Running Combined Profiling on {Colors.BRIGHT_MAGENTA}{dataset_name}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_YELLOW}Dataset:{Colors.RESET} {Colors.BRIGHT_WHITE}{dataset_name}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_YELLOW}Split:{Colors.RESET} {Colors.BRIGHT_WHITE}{split}{Colors.RESET}")
    log.info(f"{Colors.CYAN}Output dir:{Colors.RESET} {output_dir} (will be adjusted to include dataset name)")
    log.info(f"{Colors.CYAN}Batch size:{Colors.RESET} 1 (always used for per-sample measurement)")
    log.info(f"{Colors.CYAN}Sequence length:{Colors.RESET} Dynamic (uses actual length per sample, no padding/truncation)")
    log.info(f"{Colors.CYAN}Num samples:{Colors.RESET} {num_samples} (total across all ranks)")
    log.info(f"{Colors.CYAN}Max new tokens:{Colors.RESET} {max_new_tokens} (upper limit, EOS token will stop early)")
    log.info(f"{Colors.CYAN}Number of GPUs:{Colors.RESET} {num_gpus}")
    log.info(f"{Colors.CYAN}Tiers:{Colors.RESET} {tier_list}")
    log.info(f"{Colors.CYAN}Top K:{Colors.RESET} {top_k_list}")
    log.info(f"{Colors.CYAN}Active blocks:{Colors.RESET} {num_active_blocks_list}")
    log.info(f"{Colors.CYAN}Memory optimization:{Colors.RESET} {enable_memory_optimization}")
    log.info("")
    log.info(f"{Colors.CYAN}Note: use_eos_token=True, so max_new_tokens is just an upper limit.{Colors.RESET}")
    log.info(f"{Colors.CYAN}      Model will stop when EOS token is generated, even if max_new_tokens is larger.{Colors.RESET}")
    log.info("")
    
    # Ensure log directory exists
    log_dir = Path(base_output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"combined_profiling_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log.info(f"Saving terminal log to: {log_file}")
    
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
    
    # Add importance scores file if provided and exists
    if importance_scores_file and Path(importance_scores_file).exists():
        cmd.extend(["--importance_scores_file", importance_scores_file])
        if log:
            log.debug(f"Using importance scores from: {importance_scores_file}")
    elif importance_scores_file:
        if log:
            log.warning(f"Importance scores file not found: {importance_scores_file}, will use prefix blocks")
    
    if enable_memory_optimization:
        cmd.append("--enable_memory_optimization")
    
    # Run command and tee output to log file
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output to both stdout and log file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        process.wait()
    
    if process.returncode != 0:
        log.error(f"Command failed with return code {process.returncode}")
        sys.exit(process.returncode)
    
    log.info("")
    log.info(f"Combined profiling completed for {dataset_name}")
    log.info(f"Results saved to: {Path(output_dir) / dataset_name.replace('_', '-')}")
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    log.info("")


def main():
    """Main execution"""
    log = setup_logging()
    
    # Set umask for group-writable files (important for shared directories)
    os.umask(0o002)
    
    # Memory optimization for A100-40GB
    # Set PyTorch CUDA allocator to use expandable segments to reduce fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Memory-optimized dataloader settings for A100-40GB
    # Reduce num_workers and prefetch_factor to save memory
    if "DATALOADER_NUM_WORKERS" not in os.environ:
        os.environ["DATALOADER_NUM_WORKERS"] = "2"  # Reduced from default 4
    if "DATALOADER_PREFETCH_FACTOR" not in os.environ:
        os.environ["DATALOADER_PREFETCH_FACTOR"] = "1"  # Reduced from default 2
    
    # ============================================================================
    # Experiment Configuration (Optimized for A100-40GB)
    # ============================================================================
    # Modify these values directly to change experiment settings
    
    # Model and output paths
    model_path = "checkpoints"
    base_output_dir = "./results/core_exp_a100"
    
    # Importance scores file (for block selection)
    importance_scores_file = "./results/layer_importance_scores.json"
    
    # Sampling configuration (reduced for A100 memory constraints)
    num_samples = 1000
    sampling_strategy = "balanced"
    num_runs_per_sample = 1
    
    # Tier-based vision token control (A100 defaults: more tiers for comprehensive profiling)
    tier_list = ["low", "medium", "high"]  # Available: "low", "medium", "high"
    top_k_list = [4, 6, 8]  # MoE top-k values
    num_active_blocks_list = [12, 14, 16]  # Number of active transformer blocks
    
    # Memory optimization (enabled by default for A100)
    enable_memory_optimization = True
    
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
    
    # Print header
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}Multi-Dataset Combined Profiling (A100 Optimized){Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_YELLOW}Model path:{Colors.RESET} {Colors.BRIGHT_WHITE}{model_path}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_YELLOW}Base output dir:{Colors.RESET} {Colors.BRIGHT_WHITE}{base_output_dir}{Colors.RESET}")
    log.info(f"{Colors.CYAN}Batch size:{Colors.RESET} 1 (always used for per-sample measurement)")
    log.info(f"{Colors.CYAN}Sequence length:{Colors.RESET} Dynamic (uses actual length per sample, no padding/truncation)")
    log.info(f"{Colors.CYAN}Num samples:{Colors.RESET} {num_samples}")
    log.info(f"{Colors.CYAN}Sampling strategy:{Colors.RESET} {sampling_strategy}")
    log.info(f"{Colors.CYAN}Number of GPUs:{Colors.RESET} {num_gpus} (auto-detected, can override with NUM_GPUS_OVERRIDE env var)")
    log.info("")
    log.info(f"{Colors.CYAN}Knob ranges:{Colors.RESET}")
    log.info(f"  {Colors.CYAN}Tiers:{Colors.RESET} {tier_list}")
    log.info(f"  {Colors.CYAN}Top K:{Colors.RESET} {top_k_list}")
    log.info(f"  {Colors.CYAN}Active blocks:{Colors.RESET} {num_active_blocks_list}")
    log.info(f"  {Colors.CYAN}Memory optimization:{Colors.RESET} {enable_memory_optimization}")
    if importance_scores_file and Path(importance_scores_file).exists():
        log.info(f"  {Colors.CYAN}Importance Scores:{Colors.RESET} {Colors.BRIGHT_WHITE}{importance_scores_file}{Colors.RESET} {Colors.GREEN}✓{Colors.RESET}")
    elif importance_scores_file:
        log.warning(f"  {Colors.YELLOW}Importance scores file not found: {importance_scores_file}{Colors.RESET} (will use prefix blocks)")
    log.info("")
    log.info(f"{Colors.CYAN}Note: Using tier-based control allows select_tiling to adaptively select best crop count{Colors.RESET}")
    log.info(f"{Colors.CYAN}      within each tier range based on image aspect ratio. Actual crops and vision tokens{Colors.RESET}")
    log.info(f"{Colors.CYAN}      are recorded per image in the experiment results.{Colors.RESET}")
    log.info(f"{Colors.CYAN}      for minimal distortion. Results will be saved in dataset-specific subdirectories{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    log.info("")
    
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
            enable_memory_optimization=enable_memory_optimization,
            importance_scores_file=importance_scores_file,
            log=log,
        )
    
    log.info("")
    log.info(f"{Colors.BRIGHT_GREEN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_GREEN}All experiments completed!{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_GREEN}{'='*60}{Colors.RESET}")
    log.info(f"{Colors.CYAN}Results saved in dataset-specific subdirectories under:{Colors.RESET}")
    # List actual dataset names that were run (only active datasets, not commented ones)
    for dataset_name, split, max_new_tokens in datasets:
        if specific_dataset and dataset_name != specific_dataset:
            continue  # Skip if filtered by specific_dataset
        dataset_dir = dataset_name.replace("_", "-")
        log.info(f"  {Colors.BRIGHT_WHITE}{base_output_dir}/{dataset_dir}/{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_GREEN}{'='*60}{Colors.RESET}")


if __name__ == "__main__":
    main()

