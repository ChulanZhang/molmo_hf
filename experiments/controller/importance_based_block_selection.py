"""
Importance-based block selection for Knob3.

Simplifies Knob3 from mask prediction to num_blocks prediction.
Importance scores are data-agnostic but task-dependent:
- Data-agnostic: Same task type across different datasets have similar importance scores
- Task-dependent: Different task types (e.g., VQA vs ScienceQA) have different importance patterns
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

log = logging.getLogger(__name__)


def load_importance_scores(importance_file: str) -> Dict[int, float]:
    """
    Load importance scores from file.
    
    Args:
        importance_file: Path to importance scores JSON file
    
    Returns:
        importance_scores: Dict mapping block index to importance score
    """
    with open(importance_file, 'r') as f:
        data = json.load(f)
    
    # Try different possible keys
    if 'merged_scores' in data:
        scores = data['merged_scores']
    elif 'train_scores' in data:
        scores = data['train_scores']
    elif 'importance_scores' in data:
        scores = data['importance_scores']
    else:
        # Assume the whole dict is scores
        scores = data
    
    # Convert string keys to int
    importance_scores = {}
    for k, v in scores.items():
        try:
            block_idx = int(k)
            importance_scores[block_idx] = float(v)
        except (ValueError, TypeError):
            continue
    
    return importance_scores


def select_blocks_by_importance(
    importance_scores: Dict[int, float],
    num_blocks: int,
    ensure_first_last: bool = True,
) -> List[int]:
    """
    Select top-N most important blocks based on importance scores.
    
    This is the core function for Knob3: instead of learning a mask,
    we simply select the top-N blocks by pre-computed importance scores.
    
    Args:
        importance_scores: Dict mapping block index to importance score
        num_blocks: Number of blocks to select
        ensure_first_last: If True, always include first and last blocks
    
    Returns:
        selected_blocks: List of block indices (sorted)
    """
    total_blocks = len(importance_scores)
    
    if num_blocks >= total_blocks:
        return list(range(total_blocks))
    
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    
    # Sort blocks by importance (descending: most important first)
    sorted_blocks = sorted(
        importance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Select top-N
    selected_blocks = [block_idx for block_idx, _ in sorted_blocks[:num_blocks]]
    
    # Ensure first and last blocks are always included
    if ensure_first_last:
        if 0 not in selected_blocks:
            # Replace least important with first block
            selected_blocks = [0] + selected_blocks[:-1]
        if (total_blocks - 1) not in selected_blocks:
            # Replace least important with last block
            if 0 in selected_blocks and len(selected_blocks) == num_blocks:
                # If first block was just added, replace second least important
                selected_blocks = [selected_blocks[0]] + [total_blocks - 1] + selected_blocks[2:]
            else:
                selected_blocks = selected_blocks[:-1] + [total_blocks - 1]
    
    return sorted(selected_blocks)


def create_block_mask_from_selection(
    selected_blocks: List[int],
    total_blocks: int,
) -> torch.Tensor:
    """
    Create block mask from selected block indices.
    
    Args:
        selected_blocks: List of selected block indices
        total_blocks: Total number of blocks
    
    Returns:
        block_mask: Boolean tensor of shape (total_blocks,)
    """
    block_mask = torch.zeros(total_blocks, dtype=torch.bool)
    for block_idx in selected_blocks:
        if 0 <= block_idx < total_blocks:
            block_mask[block_idx] = True
    
    return block_mask


def get_block_selection_for_knob3(
    importance_scores: Dict[int, float],
    knob3_value: int,  # num_blocks: 8, 10, 12, 14, 16
) -> List[int]:
    """
    Get block selection for given Knob3 value.
    
    This is the main interface for using importance-based block selection.
    
    Args:
        importance_scores: Pre-computed importance scores
        knob3_value: Number of blocks to select (8, 10, 12, 14, or 16)
    
    Returns:
        selected_blocks: List of selected block indices
    """
    if knob3_value not in [8, 10, 12, 14, 16]:
        raise ValueError(f"knob3_value must be one of [8, 10, 12, 14, 16], got {knob3_value}")
    
    return select_blocks_by_importance(importance_scores, knob3_value)


class ImportanceBasedBlockSelector:
    """
    Wrapper class for importance-based block selection.
    
    Supports task-dependent importance scores:
    - Can load task-specific importance scores
    - Falls back to merged/universal scores if task-specific not available
    """
    
    def __init__(
        self,
        importance_scores: Optional[Dict[int, float]] = None,
        importance_file: Optional[str] = None,
        task_type: Optional[str] = None,
    ):
        """
        Initialize block selector.
        
        Args:
            importance_scores: Dict of importance scores (if provided directly)
            importance_file: Path to importance scores file (if loading from file)
            task_type: Task type for task-dependent selection (e.g., "vqa", "science_qa")
        """
        self.task_type = task_type
        
        if importance_scores is not None:
            self.importance_scores = importance_scores
        elif importance_file is not None:
            # Try to load task-specific scores if task_type is provided
            if task_type is not None:
                self.importance_scores = self._load_task_specific_scores(importance_file, task_type)
            else:
                self.importance_scores = load_importance_scores(importance_file)
        else:
            raise ValueError("Either importance_scores or importance_file must be provided")
        
        # Validate
        if len(self.importance_scores) == 0:
            raise ValueError("Importance scores are empty")
        
        log.info(f"Loaded importance scores for {len(self.importance_scores)} blocks"
                f"{f' (task_type={task_type})' if task_type else ''}")
    
    def _load_task_specific_scores(
        self,
        importance_file: str,
        task_type: str,
    ) -> Dict[int, float]:
        """
        Load task-specific importance scores.
        
        Tries to load task-specific scores first, falls back to merged scores.
        
        Args:
            importance_file: Path to importance scores file
            task_type: Task type (e.g., "vqa", "science_qa")
        
        Returns:
            importance_scores: Dict mapping block index to importance score
        """
        with open(importance_file, 'r') as f:
            data = json.load(f)
        
        # Try task-specific scores first
        task_key = f"{task_type}_scores"
        if task_key in data:
            scores = data[task_key]
            log.info(f"Using task-specific scores for {task_type}")
        elif 'merged_scores' in data:
            scores = data['merged_scores']
            log.info(f"Using merged scores (task-specific not available for {task_type})")
        elif 'train_scores' in data:
            scores = data['train_scores']
            log.info(f"Using train scores (task-specific not available for {task_type})")
        else:
            scores = data
        
        # Convert string keys to int
        importance_scores = {}
        for k, v in scores.items():
            try:
                block_idx = int(k)
                importance_scores[block_idx] = float(v)
            except (ValueError, TypeError):
                continue
        
        return importance_scores
    
    def select_blocks(self, num_blocks: int) -> List[int]:
        """
        Select blocks for given number.
        
        Args:
            num_blocks: Number of blocks to select
        
        Returns:
            selected_blocks: List of selected block indices
        """
        return select_blocks_by_importance(self.importance_scores, num_blocks)
    
    def create_mask(self, num_blocks: int, total_blocks: Optional[int] = None) -> torch.Tensor:
        """
        Create block mask for given number of blocks.
        
        Args:
            num_blocks: Number of blocks to select
            total_blocks: Total number of blocks (default: len(importance_scores))
        
        Returns:
            block_mask: Boolean tensor
        """
        if total_blocks is None:
            total_blocks = len(self.importance_scores)
        
        selected_blocks = self.select_blocks(num_blocks)
        return create_block_mask_from_selection(selected_blocks, total_blocks)
    
    def get_ranking(self) -> List[Tuple[int, float]]:
        """
        Get block ranking by importance.
        
        Returns:
            ranking: List of (block_idx, importance_score) tuples, sorted by importance
        """
        return sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )


def test_importance_based_selection():
    """Test function for importance-based selection."""
    # Example importance scores (block 0 is most important)
    importance_scores = {
        0: 0.6,
        1: 0.04,
        2: 0.025,
        3: 0.01,
        4: 0.015,
        5: 0.044,
        6: 0.024,
        7: 0.02,
        8: 0.022,
        9: 0.031,
        10: 0.025,
        11: 0.027,
        12: 0.025,
        13: 0.026,
        14: 0.024,
        15: 0.034,
    }
    
    selector = ImportanceBasedBlockSelector(importance_scores=importance_scores)
    
    # Test different num_blocks
    for num_blocks in [8, 10, 12, 14, 16]:
        selected = selector.select_blocks(num_blocks)
        mask = selector.create_mask(num_blocks)
        log.info(f"num_blocks={num_blocks}: selected={selected}, mask_sum={mask.sum().item()}")


if __name__ == "__main__":
    test_importance_based_selection()

