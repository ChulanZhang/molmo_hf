"""
Standard COCO Caption evaluation using pycocoevalcap.

This module provides standard COCO Caption evaluation metrics:
- CIDEr
- BLEU (1-4)
- METEOR
- ROUGE-L
- SPICE (optional, requires Stanford CoreNLP)

Reference: https://github.com/tylin/coco-caption
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


def evaluate_coco_caption_standard(
    predictions: List[Dict],
    references: List[Dict],
    image_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Evaluate COCO Caption predictions using standard pycocoevalcap metrics.
    
    This function uses the official COCO Caption evaluation code from:
    https://github.com/tylin/coco-caption
    
    Args:
        predictions: List of prediction dicts, each with:
            - "image_id": int (image ID)
            - "caption": str (predicted caption)
        references: List of reference dicts, each with:
            - "image_id": int (image ID)
            - "caption": str (reference caption)
            Note: Multiple references per image are supported (typically 5 for COCO)
        image_ids: Optional list of image IDs to evaluate. If None, uses all image IDs
                   present in predictions.
    
    Returns:
        Dictionary with evaluation metrics:
        - "CIDEr": CIDEr score
        - "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4": BLEU scores
        - "METEOR": METEOR score
        - "ROUGE_L": ROUGE-L score
        - "SPICE": SPICE score (if available)
    
    Raises:
        ImportError: If pycocoevalcap is not installed
        ValueError: If predictions or references are empty
    """
    try:
        from pycocoevalcap.eval import COCOEvalCap
        from pycocotools.coco import COCO
    except ImportError:
        raise ImportError(
            "pycocoevalcap is required for standard COCO Caption evaluation. "
            "Install it with: pip install pycocoevalcap pycocotools"
        )
    
    if not predictions or not references:
        raise ValueError("Predictions and references cannot be empty")
    
    # Group references by image_id
    refs_by_image = {}
    for ref in references:
        image_id = ref["image_id"]
        if image_id not in refs_by_image:
            refs_by_image[image_id] = []
        refs_by_image[image_id].append(ref["caption"])
    
    # Filter predictions if image_ids is specified
    if image_ids is not None:
        image_ids_set = set(image_ids)
        predictions = [p for p in predictions if p["image_id"] in image_ids_set]
        refs_by_image = {img_id: refs for img_id, refs in refs_by_image.items() 
                         if img_id in image_ids_set}
    
    if not predictions:
        raise ValueError("No predictions after filtering")
    
    # Create COCO format reference file
    # Format: {"images": [...], "annotations": [{"image_id": ..., "caption": ..., "id": ...}, ...]}
    coco_refs = {
        "images": [],
        "annotations": []
    }
    
    annotation_id = 0
    seen_image_ids = set()
    for image_id, captions in refs_by_image.items():
        if image_id not in seen_image_ids:
            coco_refs["images"].append({"id": image_id})
            seen_image_ids.add(image_id)
        
        for caption in captions:
            coco_refs["annotations"].append({
                "image_id": image_id,
                "caption": caption,
                "id": annotation_id
            })
            annotation_id += 1
    
    # Create COCO format result file
    # Format: [{"image_id": ..., "caption": ...}, ...]
    coco_results = []
    for pred in predictions:
        coco_results.append({
            "image_id": pred["image_id"],
            "caption": pred["caption"]
        })
    
    # Write to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        ref_file = tmpdir_path / "refs.json"
        res_file = tmpdir_path / "res.json"
        
        with open(ref_file, 'w') as f:
            json.dump(coco_refs, f)
        
        with open(res_file, 'w') as f:
            json.dump(coco_results, f)
        
        # Suppress Stanford CoreNLP and pycocoevalcap verbose output
        # Java processes write directly to file descriptors, so we need to redirect at FD level
        import sys
        import os
        
        # Suppress logging from Stanford CoreNLP Java library
        logging.getLogger('edu.stanford.nlp').setLevel(logging.ERROR)
        logging.getLogger('edu.stanford.nlp.pipeline').setLevel(logging.ERROR)
        
        # Set Java logging environment variables BEFORE any Java process starts
        # This suppresses "Picked up JAVA_TOOL_OPTIONS" messages and all Java logging
        old_java_opts = os.environ.get('JAVA_TOOL_OPTIONS', '')
        old_java_logging = os.environ.get('JAVA_LOGGING', '')
        
        # Completely suppress Java logging
        os.environ['JAVA_TOOL_OPTIONS'] = '-Djava.util.logging.config.file=/dev/null -Djava.util.logging.ConsoleHandler.level=SEVERE -Djava.util.logging.SimpleFormatter.format=""'
        os.environ['JAVA_LOGGING'] = 'SEVERE'
        
        # Save original file descriptors
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
        
        # Open /dev/null for writing
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
        except (OSError, AttributeError):
            # Fallback: create a temporary file if /dev/null is not available
            import tempfile
            devnull_fd = os.open(tempfile.mktemp(), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        
        # Save copies of original FDs
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)
        
        try:
            # Redirect stdout and stderr to /dev/null at file descriptor level
            # This will catch Java process output as well
            os.dup2(devnull_fd, original_stdout_fd)
            os.dup2(devnull_fd, original_stderr_fd)
            
            # Load COCO objects
            coco = COCO(str(ref_file))
            cocoRes = coco.loadRes(str(res_file))
            
            # Create evaluator
            cocoEval = COCOEvalCap(coco, cocoRes)
            
            # Evaluate (all output including Java will be suppressed)
            cocoEval.evaluate()
        finally:
            # Restore file descriptors first
            try:
                os.dup2(saved_stdout_fd, original_stdout_fd)
                os.dup2(saved_stderr_fd, original_stderr_fd)
            except:
                pass
            
            # Close file descriptors
            try:
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)
                os.close(devnull_fd)
            except:
                pass
            
            # Restore Java environment variables
            if old_java_opts:
                os.environ['JAVA_TOOL_OPTIONS'] = old_java_opts
            elif 'JAVA_TOOL_OPTIONS' in os.environ:
                del os.environ['JAVA_TOOL_OPTIONS']
            
            if old_java_logging:
                os.environ['JAVA_LOGGING'] = old_java_logging
            elif 'JAVA_LOGGING' in os.environ:
                del os.environ['JAVA_LOGGING']
        
        # Extract results
        results = {}
        for metric, score in cocoEval.eval.items():
            # Convert to float and handle different metric formats
            if isinstance(score, (int, float)):
                results[metric] = float(score)
            elif isinstance(score, (list, tuple)) and len(score) > 0:
                # Some metrics return lists (e.g., BLEU returns [bleu1, bleu2, bleu3, bleu4])
                if metric == "Bleu":
                    for i, s in enumerate(score, 1):
                        results[f"BLEU_{i}"] = float(s)
                else:
                    results[metric] = float(score[0]) if len(score) == 1 else [float(s) for s in score]
            else:
                results[metric] = float(score) if score is not None else 0.0
        
        return results


def evaluate_coco_caption_from_batch_results(
    per_sample_results: List[Dict],
) -> Dict[str, float]:
    """
    Evaluate COCO Caption from batch evaluation results.
    
    This is a convenience function that extracts predictions and references
    from the per_sample_results format used by BaseExperiment.compute_accuracy().
    
    Args:
        per_sample_results: List of per-sample result dicts, each with:
            - "pred": str (predicted caption)
            - "captions": List[str] (reference captions)
            - "metadata": dict with "image_id"
    
    Returns:
        Dictionary with evaluation metrics (same format as evaluate_coco_caption_standard)
    """
    predictions = []
    references = []
    
    for sample in per_sample_results:
        image_id = sample.get("metadata", {}).get("image_id")
        if image_id is None:
            log.warning(f"Sample missing image_id, skipping: {sample}")
            continue
        
        pred_caption = sample.get("pred", "")
        ref_captions = sample.get("captions", [])
        
        if not pred_caption:
            log.warning(f"Sample {image_id} has empty prediction, skipping")
            continue
        
        if not ref_captions:
            log.warning(f"Sample {image_id} has no reference captions, skipping")
            continue
        
        predictions.append({
            "image_id": image_id,
            "caption": pred_caption
        })
        
        # Add all reference captions
        for ref_caption in ref_captions:
            references.append({
                "image_id": image_id,
                "caption": ref_caption
            })
    
    if not predictions or not references:
        log.error("No valid predictions or references found")
        return {}
    
    return evaluate_coco_caption_standard(predictions, references)

