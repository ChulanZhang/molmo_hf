#!/usr/bin/env python3
"""
Analyze data requirements for all 9 datasets.
Checks which datasets need raw data and which only need HF cache.
"""

import os
from pathlib import Path

# Configuration for the 9 datasets
DATASETS = [
    ("coco_2014_vqa", "validation"),
    ("text_vqa", "validation"),
    ("okvqa", "validation"),
    ("science_qa_img", "validation"),
    ("st_qa", "validation"),
    ("doc_qa", "validation"),
    ("tally_qa", "test"),
    ("mmmu", "validation"),
    ("coco_caption", "validation"),
]

def analyze_dataset_requirements():
    """Analyze data needs for each dataset."""
    print("=" * 80)
    print("Data requirements for 9 datasets")
    print("=" * 80)
    print()
    
    molmo_data_dir = os.environ.get("MOLMO_DATA_DIR", "")
    hf_home = os.environ.get("HF_HOME", "")
    
    print(f"MOLMO_DATA_DIR: {molmo_data_dir or '(not set)'}")
    print(f"HF_HOME: {hf_home or '(not set)'}")
    print()
    
    # Dataset categorization
    hf_cache_only = []  # Only needs HF cache
    needs_raw_data = []  # Needs raw data
    
    print("=" * 80)
    print("Dataset analysis")
    print("=" * 80)
    print()
    
    for dataset_name, split in DATASETS:
        print(f"Dataset: {dataset_name} (split: {split})")
        print("-" * 80)
        
        if dataset_name == "coco_2014_vqa":
            print("Type: Vqa2 - uses VQAv2BuilderMultiQA")
            print("Needs: COCO 2014 images (val2014)")
            print("HF cache: {HF_HOME}/datasets/vqa_v2/ or equivalent")
            print("Raw data: {MOLMO_DATA_DIR}/torch_datasets/downloads/val2014/")
            needs_raw_data.append(("coco_2014_vqa", "val2014"))
            
        elif dataset_name == "text_vqa":
            print("Type: TextVqa - HfDataset")
            print("Source: facebook/textvqa (HuggingFace)")
            print("Needs: HF cache only")
            print("HF cache: {HF_HOME}/datasets/facebook___textvqa/")
            hf_cache_only.append(("text_vqa", "facebook___textvqa"))
            
        elif dataset_name == "okvqa":
            print("Type: OkVqa")
            print("Source: HuggingFaceM4/OK-VQA (HuggingFace)")
            print("Needs: HF cache only")
            print("HF cache: {HF_HOME}/datasets/HuggingFaceM4___ok-vqa/")
            hf_cache_only.append(("okvqa", "HuggingFaceM4___ok-vqa"))
            
        elif dataset_name == "science_qa_img":
            print("Type: ScienceQAImageOnly")
            print("Source: derek-thomas/ScienceQA (HuggingFace)")
            print("Needs: HF cache only")
            print("HF cache: {HF_HOME}/datasets/derek-thomas___science_qa/")
            hf_cache_only.append(("science_qa_img", "derek-thomas___science_qa"))
            
        elif dataset_name == "st_qa":
            print("Type: SceneTextQa - DatasetBase")
            print("Needs: manually downloaded JSON files and images")
            print("Location: {MOLMO_DATA_DIR}/torch_datasets/scene-text/")
            print("Files: train_task_3.json, test_task_3.json, image files")
            needs_raw_data.append(("st_qa", "scene-text"))
            
        elif dataset_name == "doc_qa":
            print("Type: DocQa - HfDataset")
            print("Source: HuggingFaceM4/DocumentVQA (HuggingFace)")
            print("Needs: HF cache only")
            print("HF cache: {HF_HOME}/datasets/HuggingFaceM4___document_vqa/")
            hf_cache_only.append(("doc_qa", "HuggingFaceM4___document_vqa"))
            
        elif dataset_name == "tally_qa":
            print("Type: TallyQa - uses TallyQaBuilder")
            print("Needs: COCO 2014 images (train2014, val2014)")
            print("HF cache: {HF_HOME}/datasets/tally_qa/")
            print("Raw data: {MOLMO_DATA_DIR}/torch_datasets/downloads/train2014/ and val2014/")
            needs_raw_data.append(("tally_qa", "train2014,val2014"))
            
        elif dataset_name == "mmmu":
            print("Type: MMMU")
            print("Source: MMMU/MMMU (HuggingFace)")
            print("Needs: HF cache only")
            print("HF cache: {HF_HOME}/datasets/MMMU___mmmu/")
            hf_cache_only.append(("mmmu", "MMMU___mmmu"))
            
        elif dataset_name == "coco_caption":
            print("Type: CocoCaption - uses CocoCaptionBuilder")
            print("Needs: COCO 2014 images (val2014)")
            print("HF cache: {HF_HOME}/datasets/coco_caption/")
            print("Raw data: {MOLMO_DATA_DIR}/torch_datasets/downloads/val2014/")
            needs_raw_data.append(("coco_caption", "val2014"))
        
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    
    print(f"HF-cache-only datasets ({len(hf_cache_only)}):")
    for ds_name, cache_name in hf_cache_only:
        print(f"  - {ds_name}: {cache_name}")
    print()
    
    print(f"Datasets needing raw data ({len(needs_raw_data)}):")
    for ds_name, data_type in needs_raw_data:
        print(f"  - {ds_name}: {data_type}")
    print()
    
    # Sync suggestions
    print("=" * 80)
    print("Sync suggestions")
    print("=" * 80)
    print()
    
    print("1. HF cache (all datasets):")
    print(f"   {hf_home}/datasets/")
    print("   Sync the following directories:")
    for ds_name, cache_name in hf_cache_only:
        print(f"   - {cache_name}/")
    print("   And also:")
    print("   - coco_caption/")
    print("   - vqa_v2/ or equivalent (coco_2014_vqa)")
    print("   - tally_qa/")
    print()
    
    print("2. Raw data (datasets needing images):")
    if molmo_data_dir:
        print(f"   {molmo_data_dir}/torch_datasets/downloads/")
        print("   Sync:")
        print("   - val2014/  (required by coco_2014_vqa, coco_caption)")
        print("   - train2014/  (required by tally_qa)")
        print()
        print(f"   {molmo_data_dir}/torch_datasets/scene-text/")
        print("   - train_task_3.json, test_task_3.json")
        print("   - image files")
    print()
    
    # Check actual files
    print("=" * 80)
    print("Check actual files")
    print("=" * 80)
    print()
    
    if hf_home:
        hf_path = Path(hf_home) / "datasets"
        print("HF cache directory check:")
        for ds_name, cache_name in hf_cache_only:
            cache_path = hf_path / cache_name
            if cache_path.exists():
                size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file()) / (1024**2)
                print(f"  ✓ {cache_name}: {size:.2f} MB")
            else:
                print(f"  ✗ {cache_name}: missing")
        
        # Check HF caches for datasets needing raw data
        for cache_name in ["coco_caption", "vqa_v2", "tally_qa"]:
            cache_path = hf_path / cache_name
            if cache_path.exists():
                size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file()) / (1024**2)
                print(f"  ✓ {cache_name}: {size:.2f} MB")
            else:
                print(f"  ✗ {cache_name}: missing")
    print()
    
    if molmo_data_dir:
        molmo_path = Path(molmo_data_dir) / "torch_datasets"
        print("Raw data directory check:")
        
        # Check COCO images
        for dir_name in ["val2014", "train2014"]:
            dir_path = molmo_path / "downloads" / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*.jpg")))
                print(f"  ✓ downloads/{dir_name}/: {file_count} image files")
            else:
                print(f"  ✗ downloads/{dir_name}/: missing")
        
        # Check scene-text
        st_path = molmo_path / "scene-text"
        if st_path.exists():
            json_files = list(st_path.glob("*_task_3.json"))
            print(f"  ✓ scene-text/: {len(json_files)} JSON files")
        else:
            print(f"  ✗ scene-text/: missing")
    print()
    
    print("=" * 80)

if __name__ == "__main__":
    analyze_dataset_requirements()

