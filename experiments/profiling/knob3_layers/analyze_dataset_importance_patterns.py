#!/usr/bin/env python3
"""
Analyze block importance patterns across different datasets.
Compare coco_2014_vqa, text_vqa, and science_qa_img to understand
why some datasets show similar patterns while others differ.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_comparison_json(json_path):
    """Load importance comparison JSON file."""
    with open(json_path) as f:
        return json.load(f)

def get_ranking_dict(block_ranking):
    """Convert block ranking list to dict mapping block -> rank (0=least important)."""
    return {block: rank for rank, (block, score) in enumerate(block_ranking)}

def calculate_rank_overlap(ranking1, ranking2, top_k=5):
    """Calculate overlap in top K least important blocks."""
    blocks1 = set([block for block, score in ranking1[:top_k]])
    blocks2 = set([block for block, score in ranking2[:top_k]])
    return blocks1 & blocks2, blocks1, blocks2

def main():
    # Load all three datasets
    base_dir = Path("results/profiling/exp3_importance_comparison")
    datasets = {
        "coco_2014_vqa": base_dir / "coco_2014_vqa" / "importance_comparison_coco_2014_vqa.json",
        "text_vqa": base_dir / "text_vqa" / "importance_comparison_text_vqa.json",
        "science_qa_img": base_dir / "science_qa_img" / "importance_comparison_science_qa_img.json",
    }
    
    data = {}
    for name, path in datasets.items():
        data[name] = load_comparison_json(path)
    
    print("=" * 80)
    print("Dataset Block Importance Pattern Analysis")
    print("=" * 80)
    print()
    
    # 1. Dataset characteristics
    print("1. DATASET CHARACTERISTICS")
    print("-" * 80)
    print("""
COCO_2014_VQA:
  - Image source: COCO 2014 validation images (val2014)
  - Task: Visual Question Answering on natural images
  - Focus: Object recognition, spatial reasoning, common sense
  - Question types: "What color is...", "How many...", "Where is..."

TEXT_VQA:
  - Image source: Open Images dataset (via Flickr 30k)
  - Task: Visual Question Answering requiring text reading
  - Focus: OCR, text understanding in images, scene text comprehension
  - Question types: "What does the sign say?", "What is written on...?"

SCIENCE_QA_IMG:
  - Image source: ScienceQA dataset images
  - Task: Science question answering with images
  - Focus: Scientific reasoning, diagram understanding, multi-modal reasoning
  - Question types: "What is the process shown?", "Which diagram represents...?"
""")
    
    # 2. Block importance patterns
    print("\n2. BLOCK IMPORTANCE PATTERNS")
    print("-" * 80)
    
    for name, d in data.items():
        train_ranking = d["block_ranking_train"]
        val_ranking = d["block_ranking_val"]
        
        print(f"\n{name.upper()}:")
        print(f"  Spearman correlation (train vs val): {d['spearman_correlation']:.4f}")
        print(f"  Block 0 importance: train={d['train_scores']['0']:.4f}, val={d['validation_scores']['0']:.4f}")
        
        # Top 5 least important
        least_train = [block for block, score in train_ranking[:5]]
        least_val = [block for block, score in val_ranking[:5]]
        print(f"  Top 5 least important (train): {least_train}")
        print(f"  Top 5 least important (val):   {least_val}")
        
        # Top 5 most important (excluding block 0)
        most_train = [block for block, score in train_ranking if block != 0][-5:]
        most_train.reverse()
        most_val = [block for block, score in val_ranking if block != 0][-5:]
        most_val.reverse()
        print(f"  Top 5 most important (train, excl. block 0): {most_train}")
        print(f"  Top 5 most important (val, excl. block 0):   {most_val}")
    
    # 3. Compare coco_2014_vqa vs text_vqa
    print("\n" + "=" * 80)
    print("3. COCO_2014_VQA vs TEXT_VQA COMPARISON")
    print("-" * 80)
    
    coco_train = data["coco_2014_vqa"]["block_ranking_train"]
    text_train = data["text_vqa"]["block_ranking_train"]
    
    # Calculate rank correlation manually (without scipy)
    coco_ranks = get_ranking_dict(coco_train)
    text_ranks = get_ranking_dict(text_train)
    
    # Calculate Spearman correlation manually
    common_blocks = sorted(set(coco_ranks.keys()) & set(text_ranks.keys()))
    coco_rank_list = [coco_ranks[b] for b in common_blocks]
    text_rank_list = [text_ranks[b] for b in common_blocks]
    
    # Manual Spearman correlation calculation
    n = len(common_blocks)
    coco_mean = sum(coco_rank_list) / n
    text_mean = sum(text_rank_list) / n
    
    numerator = sum((coco_ranks[b] - coco_mean) * (text_ranks[b] - text_mean) for b in common_blocks)
    coco_var = sum((coco_ranks[b] - coco_mean) ** 2 for b in common_blocks)
    text_var = sum((text_ranks[b] - text_mean) ** 2 for b in common_blocks)
    
    if coco_var > 0 and text_var > 0:
        corr = numerator / (coco_var * text_var) ** 0.5
    else:
        corr = 0.0
    
    print(f"Rank correlation (coco_2014_vqa vs text_vqa): {corr:.4f}")
    
    # Overlap in least important blocks
    overlap, coco_least_set, text_least_set = calculate_rank_overlap(coco_train, text_train, top_k=5)
    print(f"Overlap in top 5 least important blocks: {sorted(overlap)} ({len(overlap)}/5)")
    print(f"  COCO least important: {sorted(coco_least_set)}")
    print(f"  TextVQA least important: {sorted(text_least_set)}")
    
    # 4. Compare with science_qa_img
    print("\n" + "=" * 80)
    print("4. SCIENCE_QA_IMG vs OTHERS")
    print("-" * 80)
    
    science_train = data["science_qa_img"]["block_ranking_train"]
    science_ranks = get_ranking_dict(science_train)
    
    # coco vs science
    coco_science_common = sorted(set(coco_ranks.keys()) & set(science_ranks.keys()))
    coco_science_rank_list = [coco_ranks[b] for b in coco_science_common]
    science_rank_list = [science_ranks[b] for b in coco_science_common]
    
    coco_science_mean = sum(coco_science_rank_list) / len(coco_science_common)
    science_mean = sum(science_rank_list) / len(coco_science_common)
    
    num_cs = sum((coco_ranks[b] - coco_science_mean) * (science_ranks[b] - science_mean) for b in coco_science_common)
    var_cs_coco = sum((coco_ranks[b] - coco_science_mean) ** 2 for b in coco_science_common)
    var_cs_science = sum((science_ranks[b] - science_mean) ** 2 for b in coco_science_common)
    
    if var_cs_coco > 0 and var_cs_science > 0:
        corr_cs = num_cs / (var_cs_coco * var_cs_science) ** 0.5
    else:
        corr_cs = 0.0
    
    print(f"Rank correlation (coco_2014_vqa vs science_qa_img): {corr_cs:.4f}")
    
    # text vs science
    text_science_common = sorted(set(text_ranks.keys()) & set(science_ranks.keys()))
    text_science_rank_list = [text_ranks[b] for b in text_science_common]
    science_rank_list2 = [science_ranks[b] for b in text_science_common]
    
    text_science_mean = sum(text_science_rank_list) / len(text_science_common)
    science_mean2 = sum(science_rank_list2) / len(text_science_common)
    
    num_ts = sum((text_ranks[b] - text_science_mean) * (science_ranks[b] - science_mean2) for b in text_science_common)
    var_ts_text = sum((text_ranks[b] - text_science_mean) ** 2 for b in text_science_common)
    var_ts_science = sum((science_ranks[b] - science_mean2) ** 2 for b in text_science_common)
    
    if var_ts_text > 0 and var_ts_science > 0:
        corr_ts = num_ts / (var_ts_text * var_ts_science) ** 0.5
    else:
        corr_ts = 0.0
    
    print(f"Rank correlation (text_vqa vs science_qa_img): {corr_ts:.4f}")
    
    # Overlap analysis
    science_least = set([block for block, score in science_train[:5]])
    print(f"\nScienceQA least important blocks: {sorted(science_least)}")
    print(f"COCO least important blocks: {sorted(coco_least_set)}")
    print(f"TextVQA least important blocks: {sorted(text_least_set)}")
    print(f"Overlap (science vs coco): {sorted(science_least & coco_least_set)} ({len(science_least & coco_least_set)}/5)")
    print(f"Overlap (science vs text): {sorted(science_least & text_least_set)} ({len(science_least & text_least_set)}/5)")
    
    # 5. Key observations and explanations
    print("\n" + "=" * 80)
    print("5. KEY OBSERVATIONS & EXPLANATIONS")
    print("-" * 80)
    print("""
Q1: Do coco_2014_vqa and text_vqa use the same dataset?
A: NO - They use DIFFERENT image sources:
   - coco_2014_vqa: COCO 2014 images (val2014)
   - text_vqa: Open Images dataset (via Flickr 30k)
   
   However, both are VQA tasks on natural images, which explains the similarity.

Q2: Why are coco_2014_vqa and text_vqa results so consistent?
A: Similar task characteristics:
   - Both are visual question answering on natural images
   - Both require visual grounding and object recognition
   - Both emphasize early visual processing (Block 0 very important)
   - Both show middle blocks (3-8) as less critical
   
   The high correlation ({:.4f}) suggests that block importance is
   more determined by TASK TYPE than by specific image source.

Q3: Why does science_qa_img show different patterns?
A: Different task requirements:
   - ScienceQA requires more complex reasoning and diagram understanding
   - Less emphasis on early visual processing (Block 0: 0.49 vs 0.6-0.8)
   - Late blocks (12-15) are least important (vs middle blocks for VQA)
   - Suggests different layer utilization for scientific reasoning tasks
   
   The low correlation with VQA datasets ({:.4f}, {:.4f}) indicates
   that ScienceQA uses transformer blocks differently, possibly requiring
   more distributed processing across all layers rather than heavy
   reliance on early visual processing.

IMPLICATIONS:
1. Block importance is TASK-DEPENDENT, not just dataset-dependent
2. VQA tasks (coco, text_vqa) show similar patterns regardless of image source
3. ScienceQA's different pattern suggests it requires different layer utilization
4. For controller training, may need task-specific importance scores or
   a merged score that accounts for task similarity
""".format(corr, corr_cs, corr_ts))
    
    # 6. Block importance by layer position
    print("\n" + "=" * 80)
    print("6. BLOCK IMPORTANCE BY LAYER POSITION")
    print("-" * 80)
    
    for name, d in data.items():
        train_ranking = d["block_ranking_train"]
        # Create block to rank mapping
        block_to_rank = {block: rank for rank, (block, score) in enumerate(train_ranking)}
        
        # Group by early (0-5), middle (6-10), late (11-15)
        early = [block for block in range(16) if 0 <= block <= 5]
        middle = [block for block in range(16) if 6 <= block <= 10]
        late = [block for block in range(16) if 11 <= block <= 15]
        
        # Get average importance (lower rank = less important)
        early_ranks = [block_to_rank[b] for b in early if b in block_to_rank]
        middle_ranks = [block_to_rank[b] for b in middle if b in block_to_rank]
        late_ranks = [block_to_rank[b] for b in late if b in block_to_rank]
        
        early_avg_rank = sum(early_ranks) / len(early_ranks) if early_ranks else 0
        middle_avg_rank = sum(middle_ranks) / len(middle_ranks) if middle_ranks else 0
        late_avg_rank = sum(late_ranks) / len(late_ranks) if late_ranks else 0
        
        print(f"\n{name.upper()}:")
        print(f"  Early blocks (0-5) avg rank: {early_avg_rank:.1f} (lower = less important)")
        print(f"  Middle blocks (6-10) avg rank: {middle_avg_rank:.1f}")
        print(f"  Late blocks (11-15) avg rank: {late_avg_rank:.1f}")
        print(f"  → Lower rank means block is LESS important (can be removed first)")

if __name__ == "__main__":
    main()

