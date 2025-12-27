import argparse
import logging
import time
import os
import glob
import shutil

from molmo.data.academic_datasets import ChartQa, ScienceQAImageOnly, TextVqa, OkVqa, DocQa, \
    InfoQa, AOkVqa, Vqa2, PlotQa, FigureQa, DvQa, SceneTextQa, TabWMPDirectAnswer, \
    AndroidControl, TallyQa, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU, ClockBench, CocoCaption
from molmo.data.pixmo_datasets import (
    PixMoPointsEval, PixMoDocs, PixMoCount, PixMoPoints,
    PixMoCapQa, PixMoCap, PixMoPointExplanations, PixMoAskModelAnything
)
from molmo.util import prepare_cli_environment

ACADEMIC_EVAL = [
    ChartQa, TextVqa, DocQa, InfoQa, Vqa2,
    AndroidControl, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU
]

ACADEMIC_DATASETS = [
    ChartQa, ScienceQAImageOnly, TextVqa, OkVqa, DocQa,
    InfoQa, AOkVqa, PlotQa, FigureQa, DvQa, SceneTextQa, TabWMPDirectAnswer,
    TallyQa, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU,
    Vqa2, AndroidControl, CocoCaption
]

PIXMO_DATASETS = [
    PixMoDocs, PixMoCount, PixMoPoints, PixMoCapQa, PixMoCap, PixMoPointExplanations,
    PixMoPointsEval, PixMoAskModelAnything
]

DATASETS = ACADEMIC_DATASETS + PIXMO_DATASETS


DATASET_MAP = {
    x.__name__.lower(): x for x in DATASETS
}


def clean_dataset_cache(dataset_name):
    """Clean partial downloads from HuggingFace datasets cache."""
    # Use HF_HOME if set (from activate_env.sh), otherwise use default location
    if "HF_HOME" in os.environ:
        hf_home = os.environ["HF_HOME"]
        # HuggingFace datasets cache is typically in HF_HOME/datasets
        cache_dir = os.path.join(hf_home, "datasets")
        logging.info(f"Using HF_HOME={hf_home} for cache cleanup")
    else:
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        logging.info(f"HF_HOME not set, using default cache location: {cache_dir}")
    
    if not os.path.exists(cache_dir):
        logging.info(f"Cache directory does not exist: {cache_dir}")
        return
    
    # Find all cache directories matching the dataset name
    pattern = os.path.join(cache_dir, f"*{dataset_name.lower()}*")
    cache_dirs = glob.glob(pattern)
    
    if cache_dirs:
        logging.info(f"Cleaning {len(cache_dirs)} cache directory(ies) for {dataset_name}...")
        for cache_dir_path in cache_dirs:
            try:
                shutil.rmtree(cache_dir_path)
                logging.info(f"Removed: {cache_dir_path}")
            except Exception as e:
                logging.warning(f"Failed to remove {cache_dir_path}: {e}")
    else:
        logging.info(f"No cache directories found matching pattern: {pattern}")


def download():
    parser = argparse.ArgumentParser(prog="Download Molmo datasets")
    parser.add_argument("dataset",
                        help="Datasets to download, can be a name or one of: all, pixmo, academic or academic_eval")
    parser.add_argument("--n_procs", type=int, default=1,
                        help="Number of processes to download with")
    parser.add_argument("--ignore_errors", action="store_true",
                        help="If dataset fails to download, skip it and continue with the remaining")
    args = parser.parse_args()

    prepare_cli_environment()

    if args.dataset == "all":
        to_download = DATASETS
    elif args.dataset == "academic":
        to_download = ACADEMIC_DATASETS
    elif args.dataset == "pixmo":
        to_download = PIXMO_DATASETS
    elif args.dataset == "academic_eval":
        to_download = ACADEMIC_EVAL
    elif args.dataset.lower().replace("_", "") in DATASET_MAP:
        to_download = [DATASET_MAP[args.dataset.lower().replace("_", "")]]
    else:
        raise NotImplementedError(args.dataset)

    for ix, dataset in enumerate(to_download):
        t0 = time.perf_counter()
        logging.info(f"Starting download for {dataset.__name__} ({ix+1}/{len(to_download)})")
        logging.info(f"Calling dataset.download(n_procs={args.n_procs})")
        # Try downloading with automatic retry and concurrency reduction
        # For network errors, we'll retry multiple times with cache cleanup
        n_procs_to_try = [args.n_procs, max(1, args.n_procs // 2), 1] if args.n_procs > 1 else [1]
        download_success = False
        last_error = None
        max_cache_cleanup_retries = 3  # Additional retries for network errors with cache cleanup
        
        # First, try with concurrency reduction
        for attempt, n_proc in enumerate(n_procs_to_try):
            try:
                if attempt > 0:
                    # Add delay before retry to allow worker processes to clean up
                    # This is important because multiprocessing downloads may have
                    # background processes still running that show progress bars
                    logging.info(
                        f"Retrying {dataset.__name__} with lower concurrency (n_procs={n_proc}). "
                        f"Waiting 5 seconds for previous download processes to clean up..."
                    )
                    time.sleep(5)
                dataset.download(n_procs=n_proc)
                download_success = True
                break
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Check if it's a network/SSL/timeout error that might benefit from retry
                is_network_error = (
                    "SSL" in error_msg or 
                    "ContentLength" in error_msg or 
                    "SSLError" in error_msg or 
                    "ClientPayloadError" in error_msg or
                    "Connection" in error_msg or
                    "timeout" in error_msg.lower() or
                    "TimeoutError" in error_msg or
                    "FSTimeoutError" in error_msg or
                    type(e).__name__ in ["TimeoutError", "FSTimeoutError", "asyncio.TimeoutError"]
                )
                
                # If it's a network error and we have more concurrency attempts, continue
                if is_network_error and attempt < len(n_procs_to_try) - 1:
                    logging.warning(
                        f"Network error detected for {dataset.__name__} with n_procs={n_proc}. "
                        f"Will retry with lower concurrency after cleanup delay.\n"
                        f"Note: If you see progress bars continuing, they are from background "
                        f"worker processes that are being cleaned up."
                    )
                    # Add delay to allow worker processes to clean up before retry
                    time.sleep(5)
                    continue
                
                # If it's not a network error or we're out of concurrency attempts, break
                break
        
        # If all concurrency attempts failed with network errors, try cache cleanup retries
        if not download_success and last_error is not None:
            error_msg = str(last_error)
            is_network_error = (
                "SSL" in error_msg or 
                "ContentLength" in error_msg or 
                "SSLError" in error_msg or 
                "ClientPayloadError" in error_msg or
                "Connection" in error_msg or
                "timeout" in error_msg.lower() or
                "TimeoutError" in error_msg or
                "FSTimeoutError" in error_msg or
                type(last_error).__name__ in ["TimeoutError", "FSTimeoutError", "asyncio.TimeoutError"]
            )
            
            if is_network_error:
                for cache_retry in range(1, max_cache_cleanup_retries + 1):
                    logging.warning(
                        f"Network error detected for {dataset.__name__} after concurrency reduction attempts. "
                        f"Cleaning cache and retrying (attempt {cache_retry}/{max_cache_cleanup_retries})..."
                    )
                    # Clean cache to remove partial downloads
                    clean_dataset_cache(dataset.__name__)
                    # Wait a bit before retry
                    time.sleep(5)
                    # Retry with single process
                    try:
                        dataset.download(n_procs=1)
                        download_success = True
                        break
                    except Exception as retry_error:
                        last_error = retry_error
                        if cache_retry < max_cache_cleanup_retries:
                            continue
                        # Out of retries
                        break
        
        if not download_success:
            error_msg = str(last_error)
            # Provide helpful message for CRC errors
            if "CRC" in error_msg or "Bad CRC" in error_msg or "corrupted" in error_msg.lower():
                logging.error(
                    f"CRC/corruption error detected for {dataset.__name__}. "
                    f"This usually means the downloaded file is corrupted.\n"
                    f"To fix this, clean the cache and retry:\n"
                    f"  rm -rf ~/.cache/huggingface/datasets/*{dataset.__name__.lower()}*\n"
                    f"  python scripts/download_data.py {dataset.__name__.lower()} --n_procs 1"
                )
            # Provide helpful message for SSL/ContentLength/Timeout errors
            elif ("SSL" in error_msg or "ContentLength" in error_msg or "SSLError" in error_msg or 
                  "ClientPayloadError" in error_msg or "timeout" in error_msg.lower() or 
                  "TimeoutError" in error_msg or "FSTimeoutError" in error_msg):
                error_type = "Timeout" if ("timeout" in error_msg.lower() or "TimeoutError" in error_msg or "FSTimeoutError" in error_msg) else "SSL/ContentLength"
                logging.error(
                    f"{error_type} error detected for {dataset.__name__} after all retry attempts. "
                    f"This usually means network issues, slow connection, or corrupted download.\n"
                    f"\n"
                    f"Note: If you see progress bars continuing after this error, they are from "
                    f"background worker processes that are still cleaning up. The download has failed.\n"
                    f"\n"
                    f"Suggestions:\n"
                    f"  1. Clean cache and retry with single process (recommended for large files):\n"
                    f"     rm -rf ~/.cache/huggingface/datasets/*{dataset.__name__.lower()}*\n"
                    f"     python scripts/download_data.py {dataset.__name__.lower()} --n_procs 1\n"
                    f"  2. For very large files like TallyQA, the download may take several hours.\n"
                    f"     Consider running in a screen/tmux session to avoid interruption.\n"
                    f"  3. Check your network connection and try again later"
                )
            # Provide helpful message for ImportError (missing dependencies)
            elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg or "Unable to import" in error_msg:
                logging.error(
                    f"Import error detected for {dataset.__name__}. "
                    f"This usually means missing dependencies.\n"
                    f"Error: {error_msg[:500]}\n"
                    f"To fix this, install the required dependencies as shown in the error message above.\n"
                    f"Or use --ignore_errors to skip this dataset."
                )
            if args.ignore_errors:
                logging.warning(f"Error downloading {dataset.__name__}: {last_error}")
                continue
            else:
                raise last_error
        logging.info(f"Done with {dataset.__name__} in {time.perf_counter()-t0:0.1f} seconds")


if __name__ == '__main__':
    download()
