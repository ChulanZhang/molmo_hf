#!/usr/bin/env python3
"""
Create importance scores file based on EXP3 beam search recommendations.

This ensures that when num_active_blocks is set, the blocks removed match
EXP3 recommendations:
- Remove 1: Block 4
- Remove 2: Block 4, 13
- Remove 3: Block 4, 10, 13
- Remove 4: Block 2, 4, 10, 13
"""

import json
from pathlib import Path

# EXP3 推荐的移除顺序（按重要性从低到高）
# 这些 blocks 应该被优先移除
EXP3_REMOVAL_ORDER = [
    4,   # 最不重要，第一个移除
    13,  # 第二个移除
    10,  # 第三个移除
    2,   # 第四个移除
]

# 其他 blocks 的相对重要性（基于原始 scores，但调整以确保 EXP3 推荐的 blocks 最低）
# 保持 Block 0 和 15 的重要性（它们总是被保留）
BASE_SCORES = {
    0: 0.56,   # 最高，总是保留
    1: 0.08,
    2: 0.06,   # 会被移除（第4个）
    3: 0.03,
    4: 0.00,   # 最低，第一个移除
    5: 0.08,
    6: 0.04,
    7: 0.05,
    8: 0.07,
    9: 0.06,
    10: 0.00,  # 会被移除（第3个）
    11: 0.04,
    12: 0.06,
    13: 0.00,  # 会被移除（第2个）
    14: 0.06,
    15: 0.06,  # 总是保留
}

def create_exp3_importance_scores() -> dict:
    """
    Create importance scores that ensure EXP3 recommended blocks are removed first.
    
    Strategy:
    1. Set blocks to be removed (4, 13, 10, 2) to very low scores
    2. Keep other blocks with reasonable scores
    3. Ensure Block 0 and 15 have high scores (always kept)
    """
    scores = {}
    
    # Set removal order: lower score = less important = removed first
    removal_scores = {
        4: 0.001,   # First to remove (lowest score)
        13: 0.002,  # Second to remove
        10: 0.003,  # Third to remove
        2: 0.004,   # Fourth to remove
    }
    
    # Create scores for all blocks
    for block_idx in range(16):
        if block_idx in removal_scores:
            scores[block_idx] = removal_scores[block_idx]
        elif block_idx == 0:
            scores[block_idx] = 0.56  # Always keep first block
        elif block_idx == 15:
            scores[block_idx] = 0.06  # Always keep last block
        else:
            # Use base scores for other blocks, ensuring they're higher than removal blocks
            scores[block_idx] = BASE_SCORES.get(block_idx, 0.05)
    
    return scores


def verify_removal_order(scores: dict) -> dict:
    """Verify that removal order matches EXP3 recommendations."""
    # Sort blocks by score (ascending = least important first)
    sorted_blocks = sorted(scores.items(), key=lambda x: x[1])
    
    # Get blocks to be removed (excluding 0 and 15 which are always kept)
    removable_blocks = [(idx, score) for idx, score in sorted_blocks if idx not in [0, 15]]
    
    # Expected removal order
    expected_order = [4, 13, 10, 2]
    
    # Check if first 4 removable blocks match expected order
    actual_order = [idx for idx, _ in removable_blocks[:4]]
    
    verification = {
        "expected_order": expected_order,
        "actual_order": actual_order,
        "matches": actual_order == expected_order,
    }
    
    return verification


def main():
    """Main execution"""
    # Create importance scores
    scores = create_exp3_importance_scores()
    
    # Verify removal order
    verification = verify_removal_order(scores)
    
    # Print verification
    print("=" * 80)
    print("EXP3 Importance Scores Generation")
    print("=" * 80)
    print()
    print("Removal Order Verification:")
    print(f"  Expected: {verification['expected_order']}")
    print(f"  Actual:   {verification['actual_order']}")
    print(f"  Match:    {'✓' if verification['matches'] else '✗'}")
    print()
    
    # Print scores
    print("Importance Scores (lower = less important = removed first):")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    for block_idx, score in sorted_scores:
        marker = " ← REMOVE" if block_idx in EXP3_REMOVAL_ORDER else ""
        always_keep = " (always keep)" if block_idx in [0, 15] else ""
        print(f"  Block {block_idx:2d}: {score:.6f}{marker}{always_keep}")
    print()
    
    # Save to file
    output_file = Path("results/layer_importance_scores_exp3_recommended.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(scores, f, indent=2)
    
    print(f"✅ Importance scores saved to: {output_file}")
    print()
    print("Usage in run_multi_datasets_*.py:")
    print(f'  importance_scores_file = "{output_file}"')
    print()
    print("Expected behavior:")
    print("  - num_active_blocks = 15 → Remove Block 4")
    print("  - num_active_blocks = 14 → Remove Block 4, 13")
    print("  - num_active_blocks = 13 → Remove Block 4, 10, 13")
    print("  - num_active_blocks = 12 → Remove Block 2, 4, 10, 13")
    print("=" * 80)


if __name__ == "__main__":
    main()

