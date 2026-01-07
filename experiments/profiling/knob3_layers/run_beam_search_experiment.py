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
from typing import List, Optional, Tuple

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


def check_results_exist(
    output_dir: str,
    split: str,
    log: logging.Logger = None,
    beam_width: int = 3,
    max_blocks_to_remove: int = 4,
) -> Tuple[bool, int, int]:
    """Check if beam search results already exist for the given split.
    
    Args:
        output_dir: Base output directory
        split: Dataset split (train/validation)
        log: Logger instance
        beam_width: Beam width used in experiment
        max_blocks_to_remove: Maximum blocks to remove
    
    Returns:
        tuple: (is_complete, num_completed, total_expected)
            - is_complete: True if all results exist and are valid
            - num_completed: Number of completed configurations
            - total_expected: Total number of expected configurations
    """
    import json
    from pathlib import Path
    
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return False, 0, 0
    
    # Check final results file first
    results_file = results_dir / "exp3_accuracy_sensitivity_v2_results.json"
    
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Check if results contain expected structure
            if "config" not in results:
                if log:
                    log.warning(f"Results file exists but missing 'config' field: {results_file}")
                return False, 0, 0
            
            # Check if split matches
            config = results.get("config", {})
            result_split = config.get("split", "")
            
            if result_split != split:
                if log:
                    log.warning(f"Results file exists but split mismatch: expected '{split}', found '{result_split}'")
                return False, 0, 0
            
            # Check if results contain beam search data
            if "summary" not in results:
                if log:
                    log.warning(f"Results file exists but missing 'summary' field: {results_file}")
                return False, 0, 0
            
            summary = results.get("summary", [])
            if not summary or len(summary) == 0:
                if log:
                    log.warning(f"Results file exists but summary is empty: {results_file}")
                return False, 0, 0
            
            # Calculate expected number of configurations
            # Step 1: 16 configs (remove 1 block)
            # Step 2: beam_width * 15 configs (remove 2 blocks)
            # Step 3: beam_width * 14 configs (remove 3 blocks)
            # Step 4: beam_width * 13 configs (remove 4 blocks)
            total_expected = 16 + beam_width * (15 + 14 + 13)
            
            if log:
                log.info(f"✅ Found existing results for {split} split: {len(summary)}/{total_expected} configurations")
                log.info(f"   Results file: {results_file}")
            
            return len(summary) >= total_expected * 0.9, len(summary), total_expected  # 90% complete is considered done
            
        except json.JSONDecodeError as e:
            if log:
                log.warning(f"Results file exists but is invalid JSON: {results_file}, error: {e}")
        except Exception as e:
            if log:
                log.warning(f"Error checking results file: {results_file}, error: {e}")
    
    # Check individual configuration files for partial completion
    # Expected pattern: beam_search_step{step}_blocks{num_active}_removed{blocks}.json
    total_blocks = 16
    total_expected = 16 + beam_width * (15 + 14 + 13)
    num_completed = 0
    
    # Count completed configurations by checking individual files
    for step in range(1, max_blocks_to_remove + 1):
        num_active = total_blocks - step
        # For each step, we need to check beam_width candidates
        if step == 1:
            # Step 1: test all 16 blocks
            for block_idx in range(total_blocks):
                removed_str = str(block_idx)
                config_file = results_dir / f"beam_search_step{step}_blocks{num_active}_removed{removed_str}.json"
                if config_file.exists():
                    num_completed += 1
        else:
            # Steps 2-4: test beam_width * (total_blocks - step + 1) configurations
            # We can't know exactly which combinations were tested, so we'll estimate
            # by checking if we have at least some files for this step
            step_files = list(results_dir.glob(f"beam_search_step{step}_blocks{num_active}_removed*.json"))
            num_completed += len(step_files)
    
    if num_completed > 0:
        completion_rate = num_completed / total_expected
        if log:
            log.info(f"📊 Partial results found: {num_completed}/{total_expected} configurations ({completion_rate*100:.1f}%)")
        
        # Consider complete if 90% done
        return completion_rate >= 0.9, num_completed, total_expected
    
    return False, 0, total_expected


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
    skip_if_exists: bool = True,
    max_retries: int = 3,
    retry_delay: int = 60,
) -> bool:
    """Run beam search experiment for a single dataset with automatic retry
    
    Args:
        skip_if_exists: If True, skip if results already exist for this split
        max_retries: Maximum number of retry attempts on failure
        retry_delay: Delay in seconds between retries
    
    Returns:
        bool: True if successful or skipped, False if failed after all retries
    """
    # Create dataset-specific output directory
    # Include split in path to separate train/validation results
    dataset_suffix = dataset_name.replace("_", "-")
    output_dir = os.path.join(base_output_dir, dataset_suffix, split)
    
    # Check if results already exist
    if skip_if_exists:
        is_complete, num_completed, total_expected = check_results_exist(
            output_dir, split, log, beam_width, max_blocks_to_remove
        )
        if is_complete:
            log.info(f"{Colors.BRIGHT_GREEN}⏭️  Skipping {dataset_name} ({split}): Results already exist ({num_completed}/{total_expected} configs){Colors.RESET}")
            return True
        elif num_completed > 0:
            log.info(f"{Colors.BRIGHT_YELLOW}📊 Resuming {dataset_name} ({split}): {num_completed}/{total_expected} configs completed ({num_completed/total_expected*100:.1f}%){Colors.RESET}")
    
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
    
    # Run with retry logic
    for attempt in range(max_retries + 1):
        if attempt > 0:
            log.warning(f"{Colors.YELLOW}⏳ Retry attempt {attempt}/{max_retries} for {dataset_name} ({split}) after {retry_delay}s delay...{Colors.RESET}")
            import time
            time.sleep(retry_delay)
        
        # Create log file (append attempt number for retries)
        log_suffix = f"beam_search_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if attempt > 0:
            log_suffix += f"_retry{attempt}"
        log_file = log_dir / f"{log_suffix}.log"
        
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
                # Verify results were actually created
                is_complete, num_completed, total_expected = check_results_exist(
                    output_dir, split, None, beam_width, max_blocks_to_remove
                )
                if is_complete:
                    if attempt > 0:
                        log.info(f"{Colors.BRIGHT_GREEN}✅ Successfully completed {dataset_name} ({split}) on retry {attempt}{Colors.RESET}")
                    return True
                else:
                    log.warning(f"{Colors.YELLOW}⚠️  Process exited successfully but results incomplete ({num_completed}/{total_expected}){Colors.RESET}")
                    if attempt < max_retries:
                        continue  # Retry
                    else:
                        log.error(f"{Colors.RED}Results incomplete after {max_retries} retries{Colors.RESET}")
                        return False
            else:
                log.error(f"{Colors.RED}Command failed for {dataset_name} (return code {process.returncode}){Colors.RESET}")
                log.error(f"Check log file: {log_file}")
                if attempt < max_retries:
                    continue  # Retry
                else:
                    return False
                
        except Exception as e:
            log.error(f"{Colors.RED}Unexpected error running {dataset_name}: {e}{Colors.RESET}")
            log.error(f"Check log file: {log_file}")
            if attempt < max_retries:
                continue  # Retry
            else:
                return False
    
    return False


def main():
    """Main execution"""
    log = setup_logging()
    
    # ============================================================================
    # Experiment Configuration
    # ============================================================================
    
    # Model and output paths
    model_path = "checkpoints"
    base_output_dir = "./results/profiling/exp3_beam_search_multi_dataset"
    
    # Beam search configuration
    beam_width = 3  # Number of top candidates to keep at each step
    max_blocks_to_remove = 4  # Maximum number of blocks to remove
    
    # Sampling configuration
    # Using 1000 samples per dataset (from combined train+validation splits)
    # This balances statistical significance with experiment time
    # With 8 datasets, we get 8000 total samples, providing good cross-dataset validation
    num_samples = 1000  # Per dataset, sampled from combined train+validation
    batch_size = 16  # Initial batch size (will be auto-adjusted)
    
    # Auto-detect number of GPUs
    num_gpus = detect_num_gpus()
    num_gpus = int(os.environ.get("NUM_GPUS_OVERRIDE", num_gpus))
    
    # Dataset configurations
    # Format: (dataset_name, split, max_new_tokens)
    # Excluding mmmu due to low correlation (0.2558) and small sample size (900)
    # Using "train+validation" to combine both splits for better statistical power
    # Reduced num_samples per dataset since we're using both splits
    datasets = [
        ("coco_2014_vqa", "train+validation", 16),
        ("text_vqa", "train+validation", 64),
        ("okvqa", "train+validation", 16),
        ("science_qa_img", "train+validation", 16),
        ("st_qa", "train+validation", 32),
        ("doc_qa", "train+validation", 32),
        ("tally_qa", "train+test", 16),  # tally_qa uses "test" instead of "validation"
        ("coco_caption", "train+validation", 64),
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
    skipped_datasets = []
    completed_datasets = []
    
    for dataset_name, split, max_new_tokens in datasets:
        if specific_dataset and dataset_name != specific_dataset:
            continue
        
        # Check if results already exist before running
        dataset_suffix = dataset_name.replace("_", "-")
        output_dir = os.path.join(base_output_dir, dataset_suffix, split)
        is_complete, num_completed, total_expected = check_results_exist(
            output_dir, split, log, beam_width, max_blocks_to_remove
        )
        if is_complete:
            skipped_datasets.append((dataset_name, split))
            continue
        
        # Run experiment with automatic retry
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
            skip_if_exists=False,  # Already checked above
            max_retries=3,  # Retry up to 3 times
            retry_delay=60,  # Wait 60 seconds between retries
        )
        
        if success:
            completed_datasets.append((dataset_name, split))
        else:
            failed_datasets.append((dataset_name, split))
            log.warning(f"{Colors.YELLOW}Continuing to next dataset...{Colors.RESET}")
    
    # Completion message
    log.info(f"{Colors.BRIGHT_GREEN}{'='*80}{Colors.RESET}")
    
    if skipped_datasets:
        log.info(f"{Colors.BRIGHT_CYAN}⏭️  Skipped {len(skipped_datasets)} dataset(s) (results already exist):{Colors.RESET}")
        for dataset_name, split in skipped_datasets:
            log.info(f"   - {dataset_name} ({split})")
    
    if completed_datasets:
        log.info(f"{Colors.BRIGHT_GREEN}✅ Completed {len(completed_datasets)} dataset(s):{Colors.RESET}")
        for dataset_name, split in completed_datasets:
            log.info(f"   - {dataset_name} ({split})")
    
    if failed_datasets:
        log.warning(f"{Colors.BRIGHT_YELLOW}❌ Failed {len(failed_datasets)} dataset(s):{Colors.RESET}")
        for dataset_name, split in failed_datasets:
            log.warning(f"   - {dataset_name} ({split})")
    
    if not failed_datasets:
        log.info(f"{Colors.BRIGHT_GREEN}All beam search experiments completed successfully!{Colors.RESET}")
    
    log.info(f"{Colors.BRIGHT_GREEN}{'='*80}{Colors.RESET}")


if __name__ == "__main__":
    main()

