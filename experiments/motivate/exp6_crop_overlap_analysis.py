"""
Experiment 6: Crop Overlap Analysis for Small Images

This experiment analyzes how small images are divided into crops when max_crops=12,
and quantifies the overlap and redundant computation.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from experiments.motivate.base_experiment import BaseExperiment, Timer

log = logging.getLogger(__name__)


class CropOverlapAnalysis(BaseExperiment):
    """Analyze crop overlap for small images."""
    
    def __init__(self, model_path: str, output_dir: str = "./results/motivation/exp6", device: str = "cuda"):
        super().__init__(model_path, device, output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_crop_overlap(
        self,
        image: np.ndarray,
        image_id: int = None,
        example_id: int = None
    ) -> Dict[str, Any]:
        """
        Analyze crop overlap for a given image.
        
        Returns:
            Dictionary with crop analysis results.
        """
        original_h, original_w = image.shape[:2]
        
        # Get preprocessing configuration
        # Use the preprocessor from build_dataloader if available, otherwise create one
        if hasattr(self, '_mm_preprocessor'):
            mm_preprocessor = self._mm_preprocessor
        else:
            from molmo.data.model_preprocessor import MultiModalPreprocessor
            mm_preprocessor = MultiModalPreprocessor(
                tokenizer=self.tokenizer,
                crop_mode=self.model.config.crop_mode,
                max_crops=self.model.config.max_crops,
                overlap_margins=self.model.config.overlap_margins,
                image_padding_mask=bool(self.model.config.image_padding_embed),
            )
        max_crops = self.model.config.max_crops
        overlap_margins = self.model.config.overlap_margins
        base_image_input_size = mm_preprocessor.base_image_input_size
        image_patch_size = mm_preprocessor.image_patch_size
        image_pooling_h = mm_preprocessor.image_pooling_h
        image_pooling_w = mm_preprocessor.image_pooling_w
        
        # Calculate crop parameters (same logic as in MultiModalPreprocessor)
        base_image_input_d = image_patch_size
        left_margin, right_margin = overlap_margins
        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d  # 336 // 14 = 24
        crop_window_patches = crop_patches - (right_margin + left_margin)  # 24 - 8 = 16
        crop_window_size = crop_window_patches * base_image_input_d  # 16 * 14 = 224
        
        # Select tiling (same logic as select_tiling)
        from molmo.data.model_preprocessor import select_tiling
        tiling = select_tiling(
            original_h - total_margin_pixels,
            original_w - total_margin_pixels,
            crop_window_size,
            max_crops
        )
        
        n_crops = int(tiling[0] * tiling[1])
        
        # Calculate resized image dimensions
        resized_h = int(tiling[0] * crop_window_size + total_margin_pixels)
        resized_w = int(tiling[1] * crop_window_size + total_margin_pixels)
        
        # Calculate crop coordinates and overlap
        crop_info = []
        total_patch_area = 0
        unique_patch_area = 0
        
        # Track unique patches (using patch coordinates in original image space)
        patch_coverage = {}  # (patch_y, patch_x) -> count
        
        for i in range(tiling[0]):
            for j in range(tiling[1]):
                # Calculate crop boundaries in resized image
                y0 = i * crop_window_size
                x0 = j * crop_window_size
                
                # Calculate crop boundaries in patch space
                if i == 0:
                    crop_y0 = 0
                else:
                    crop_y0 = left_margin // image_pooling_h
                
                if j == 0:
                    crop_x0 = 0
                else:
                    crop_x0 = left_margin // image_pooling_w
                
                crop_h = crop_patches - (right_margin + left_margin)
                if i == 0:
                    crop_h += left_margin
                if i == (tiling[0] - 1):
                    crop_h += right_margin
                
                crop_w = crop_patches - (right_margin + left_margin)
                if j == 0:
                    crop_w += left_margin
                if j == (tiling[1] - 1):
                    crop_w += right_margin
                
                # Calculate pooled dimensions
                pooled_w = (crop_w + image_pooling_w - 1) // image_pooling_w
                pooled_h = (crop_h + image_pooling_h - 1) // image_pooling_h
                
                # Calculate actual patch coverage in original image space
                # Each crop covers a region in the resized image
                # Map this back to original image coordinates
                scale_h = original_h / resized_h
                scale_w = original_w / resized_w
                
                # Crop boundaries in original image
                orig_y0 = int(y0 * scale_h)
                orig_y1 = int((y0 + crop_window_size) * scale_h)
                orig_x0 = int(x0 * scale_w)
                orig_x1 = int((x0 + crop_window_size) * scale_w)
                
                # Calculate patches in this crop
                crop_patches_count = pooled_h * pooled_w
                total_patch_area += crop_patches_count
                
                # Track unique patches (simplified: use crop index as patch identifier)
                # In reality, we need to track the actual patch ordering
                for py in range(pooled_h):
                    for px in range(pooled_w):
                        # Map to original image patch coordinates
                        orig_patch_y = int((orig_y0 + py * image_pooling_h * base_image_input_d) / base_image_input_d)
                        orig_patch_x = int((orig_x0 + px * image_pooling_w * base_image_input_d) / base_image_input_d)
                        patch_key = (orig_patch_y, orig_patch_x)
                        patch_coverage[patch_key] = patch_coverage.get(patch_key, 0) + 1
                
                crop_info.append({
                    "crop_idx": int(i * tiling[1] + j),
                    "tiling_pos": (int(i), int(j)),
                    "resized_coords": {
                        "y0": int(y0),
                        "y1": int(y0 + crop_window_size),
                        "x0": int(x0),
                        "x1": int(x0 + crop_window_size),
                    },
                    "original_coords": {
                        "y0": int(orig_y0),
                        "y1": int(orig_y1),
                        "x0": int(orig_x0),
                        "x1": int(orig_x1),
                    },
                    "patch_coords": {
                        "y0": int(crop_y0),
                        "h": int(crop_h),
                        "x0": int(crop_x0),
                        "w": int(crop_w),
                    },
                    "pooled_dims": {
                        "h": int(pooled_h),
                        "w": int(pooled_w),
                    },
                    "num_patches": int(crop_patches_count),
                })
        
        # Calculate overlap statistics
        unique_patch_area = len(patch_coverage)
        overlap_ratio = (total_patch_area - unique_patch_area) / total_patch_area if total_patch_area > 0 else 0
        redundancy_ratio = total_patch_area / unique_patch_area if unique_patch_area > 0 else 1.0
        
        # Calculate average overlap per patch
        overlapping_patches = sum(1 for count in patch_coverage.values() if count > 1)
        avg_overlap_count = np.mean(list(patch_coverage.values())) if patch_coverage else 1.0
        
        return {
            "image_id": image_id,
            "example_id": example_id,
            "original_size": {
                "height": int(original_h),
                "width": int(original_w),
            },
            "resized_size": {
                "height": int(resized_h),
                "width": int(resized_w),
            },
            "tiling": {
                "rows": int(tiling[0]),
                "cols": int(tiling[1]),
                "total_crops": int(n_crops),
            },
            "crop_config": {
                "max_crops": int(max_crops),
                "overlap_margins": tuple(int(x) for x in overlap_margins),
                "base_image_input_size": tuple(int(x) for x in base_image_input_size),
                "crop_window_size": int(crop_window_size),
                "crop_patches": int(crop_patches),
                "crop_window_patches": int(crop_window_patches),
            },
            "crops": crop_info,
            "overlap_statistics": {
                "total_patch_area": int(total_patch_area),
                "unique_patch_area": int(unique_patch_area),
                "overlap_ratio": float(overlap_ratio),
                "redundancy_ratio": float(redundancy_ratio),
                "overlapping_patches": int(overlapping_patches),
                "avg_overlap_count": float(avg_overlap_count),
            },
        }
    
    def find_small_images(
        self,
        dataset_name: str,
        split: str,
        num_samples: int = 100,
        max_size: int = 500
    ) -> List[Dict]:
        """Find small images from the dataset."""
        log.info(f"Finding small images (max_size={max_size}) from {dataset_name}/{split}...")
        
        dataloader = self.build_dataloader(dataset_name, split, batch_size=1, max_steps=num_samples)
        dataloader_iter = iter(dataloader)
        
        small_images = []
        for i in range(num_samples):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
            
            metadata = batch.get("metadata", [{}])[0] if "metadata" in batch else {}
            image_size = metadata.get("image_size", [0, 0])
            
            if len(image_size) == 2:
                h, w = image_size[1], image_size[0]  # image_size is (width, height)
                if h <= max_size or w <= max_size:
                    small_images.append({
                        "index": i,
                        "image_id": metadata.get("image_id"),
                        "example_id": metadata.get("example_id"),
                        "image_size": (h, w),
                        "area": h * w,
                    })
        
        # Sort by area (smallest first)
        small_images.sort(key=lambda x: x["area"])
        
        log.info(f"Found {len(small_images)} small images (max_size={max_size})")
        return small_images
    
    def find_images_from_exp2(
        self,
        exp2_results_path: str,
        num_images: int = 5,
        target_max_crops: int = None
    ) -> List[Dict]:
        """Find images from exp2 results with different crop counts.
        
        Args:
            exp2_results_path: Path to exp2_component_profiling.json
            num_images: Number of images to select (evenly distributed across crop count range)
        
        Returns:
            List of image info dictionaries with index, image_id, example_id, num_vision_tokens
        """
        log.info(f"Loading exp2 results from {exp2_results_path}...")
        with open(exp2_results_path, 'r') as f:
            exp2_data = json.load(f)
        
        results = exp2_data.get("results", [])
        
        # Extract crop information
        crop_data = []
        for i, r in enumerate(results):
            num_tokens = r.get("num_vision_tokens", 0)
            num_crops = r.get("num_crops", 0)
            if num_tokens > 0 and num_crops > 0:
                crop_data.append({
                    "index": i,
                    "image_id": r.get("image_id"),
                    "example_id": r.get("example_id"),
                    "num_vision_tokens": num_tokens,
                    "num_crops": num_crops,
                })
        
        # Sort by crop count
        crop_data.sort(key=lambda x: x["num_crops"])
        
        # Select evenly distributed images across the crop count range
        if len(crop_data) == 0:
            log.warning("No valid crop data found in exp2 results")
            return []
        
        min_crops = crop_data[0]["num_crops"]
        max_crops = crop_data[-1]["num_crops"]
        log.info(f"Crop count range: {min_crops} to {max_crops} crops")
        
        # If target_max_crops is specified, prefer images close to that value
        if target_max_crops is not None and target_max_crops > max_crops:
            log.info(f"Target max crops ({target_max_crops}) is higher than available max ({max_crops}), using available max")
            target_max_crops = max_crops
        elif target_max_crops is not None:
            log.info(f"Targeting images with up to {target_max_crops} crops")
        
        # Group by crop count
        crops_by_count = {}
        for item in crop_data:
            count = item["num_crops"]
            if count not in crops_by_count:
                crops_by_count[count] = []
            crops_by_count[count].append(item)
        
        # Always include the maximum crops count (or target_max_crops if specified)
        unique_counts = sorted(crops_by_count.keys())
        target_counts = []
        
        # Determine the target max to include
        effective_max = target_max_crops if target_max_crops is not None else max_crops
        # Find the closest available count to effective_max
        closest_max = min(unique_counts, key=lambda x: abs(x - effective_max))
        
        # Always include the closest to target max
        if closest_max not in target_counts:
            target_counts.append(closest_max)
            log.info(f"Including image with {closest_max} crops (closest to target {effective_max})")
        
        # If we need more images, select evenly distributed from the rest
        if len(unique_counts) <= num_images:
            # If we have fewer unique counts than requested, select one from each
            for count in unique_counts:
                if count not in target_counts:
                    target_counts.append(count)
        else:
            # Select evenly distributed counts, but ensure we include max
            remaining_slots = num_images - len(target_counts)
            if remaining_slots > 0:
                # Exclude max from the distribution
                other_counts = [c for c in unique_counts if c != max_crops]
                if len(other_counts) > 0:
                    indices = np.linspace(0, len(other_counts) - 1, remaining_slots, dtype=int)
                    for i in indices:
                        count = other_counts[i]
                        if count not in target_counts:
                            target_counts.append(count)
        
        # Sort target counts
        target_counts.sort()
        
        # Select one image from each target crop count
        selected = []
        for count in target_counts:
            candidates = crops_by_count[count]
            # Select the first one (or could randomize)
            selected.append(candidates[0])
            log.info(f"Selected image with {count} crops ({candidates[0]['num_vision_tokens']} vision tokens)")
        
        log.info(f"Found {len(selected)} images with crop counts: {[s['num_crops'] for s in selected]}")
        return selected
    
    def run(
        self,
        dataset_name: str,
        split: str,
        num_samples: int = 100,
        max_size: int = 500,
        analyze_top_k: int = 5,
        exp2_results_path: str = None,
        num_images: int = 5,
        target_max_crops: int = None,
        save_crop_images: bool = True
    ):
        """Run Experiment 6: Crop Overlap Analysis.
        
        Args:
            dataset_name: Dataset name
            split: Dataset split
            num_samples: Number of samples to search (if not using exp2)
            max_size: Maximum image size threshold (if not using exp2)
            analyze_top_k: Number of images to analyze (if not using exp2)
            exp2_results_path: Path to exp2_component_profiling.json (if using exp2 results)
            num_small_crops: Number of images with smallest crops to select (if using exp2)
            num_large_crops: Number of images with largest crops to select (if using exp2)
            save_crop_images: Whether to save individual crop images
        """
        with Timer("Exp 6: Crop Overlap Analysis"):
            log.info(f"Starting Exp 6: Crop Overlap Analysis...")
            
            # Select images from exp2 results or find small images
            if exp2_results_path:
                log.info(f"Selecting {num_images} images from exp2 results...")
                analyze_images = self.find_images_from_exp2(
                    exp2_results_path,
                    num_images,
                    target_max_crops
                )
            else:
                log.info(f"Finding small images (max_size={max_size})...")
                small_images = self.find_small_images(dataset_name, split, num_samples, max_size)
                if len(small_images) == 0:
                    log.warning("No small images found! Try increasing max_size or num_samples.")
                    return []
                analyze_images = small_images[:analyze_top_k]
            
            if len(analyze_images) == 0:
                log.warning("No images to analyze!")
                return []
            
            log.info(f"Analyzing {len(analyze_images)} images...")
            
            # Get raw dataset (without preprocessor) to access original images
            from molmo.data import get_dataset_by_name
            from molmo.data.model_preprocessor import MultiModalPreprocessor, load_image
            from molmo.data.data_formatter import DataFormatter
            from molmo.data.collator import MMCollator
            from molmo.data.dataset import DeterministicDataset
            from torch.utils.data import DataLoader
            
            # Get raw dataset
            raw_dataset = get_dataset_by_name(dataset_name, split=split)
            
            # Create preprocessor for analysis
            mm_preprocessor = MultiModalPreprocessor(
                tokenizer=self.tokenizer,
                crop_mode=self.model.config.crop_mode,
                max_crops=self.model.config.max_crops,
                overlap_margins=self.model.config.overlap_margins,
                image_padding_mask=bool(self.model.config.image_padding_embed),
            )
            
            # Store mm_preprocessor for analysis
            self._mm_preprocessor = mm_preprocessor
            
            # Create crops directory
            crops_dir = self.output_dir / "crops"
            if save_crop_images:
                crops_dir.mkdir(parents=True, exist_ok=True)
            
            results = []
            for img_info in analyze_images:
                idx = img_info['index']
                image_id = img_info.get('image_id')
                example_id = img_info.get('example_id')
                
                log.info(f"Analyzing image {idx}: image_id={image_id}, example_id={example_id}")
                
                # Get raw example from dataset
                raw_example = raw_dataset.get(idx, np.random.RandomState(42))
                
                # Load image
                if "image" in raw_example:
                    if isinstance(raw_example["image"], str):
                        image = load_image(raw_example["image"])
                    elif isinstance(raw_example["image"], np.ndarray):
                        image = raw_example["image"]
                    else:
                        log.warning(f"Image {idx} has unsupported image type, skipping...")
                        continue
                else:
                    log.warning(f"Image {idx} has no image data, skipping...")
                    continue
                
                # Analyze crop overlap
                analysis = self.analyze_crop_overlap(
                    image,
                    image_id=image_id,
                    example_id=example_id
                )
                
                # Add information about how num_crops is controlled
                analysis["crop_control_info"] = self.analyze_crop_control(
                    image,
                    analysis
                )
                
                # Save original image
                original_image_path = self.save_original_image(
                    image,
                    crops_dir,
                    image_id=image_id,
                    example_id=example_id
                )
                analysis["original_image_path"] = original_image_path
                
                # Save crop images if requested
                if save_crop_images:
                    crop_images = self.save_crop_images(
                        image,
                        analysis,
                        crops_dir,
                        image_id=image_id,
                        example_id=example_id
                    )
                    analysis["crop_image_paths"] = crop_images
                
                analysis["search_info"] = img_info
                results.append(analysis)
                
                log.info(f"  Tiling: {analysis['tiling']}")
                log.info(f"  Num crops: {analysis['tiling']['total_crops']}")
                log.info(f"  Overlap ratio: {analysis['overlap_statistics']['overlap_ratio']:.2%}")
                log.info(f"  Redundancy ratio: {analysis['overlap_statistics']['redundancy_ratio']:.2f}x")
            
            # Save results
            self.save_results(
                {
                    "results": results,
                    "summary": {
                        "num_analyzed": len(results),
                        "max_size_threshold": max_size if not exp2_results_path else None,
                        "num_samples_searched": num_samples if not exp2_results_path else None,
                        "exp2_results_path": exp2_results_path,
                        "num_images": num_images if exp2_results_path else None,
                    }
                },
                "exp6_crop_overlap_analysis.json"
            )
            
            # Generate visualization
            self.plot_crop_overlap(results)
            
            log.info(f"Exp 6 completed! Results saved to {self.output_dir}")
            return results
    
    def analyze_crop_control(
        self,
        image: np.ndarray,
        analysis: Dict
    ) -> Dict[str, Any]:
        """Analyze how num_crops is controlled and what methods can be used to control vision token count.
        
        Args:
            image: Original image array
            analysis: Crop analysis results
        
        Returns:
            Dictionary with crop control analysis
        """
        from molmo.data.model_preprocessor import select_tiling
        
        original_h, original_w = image.shape[:2]
        max_crops = self.model.config.max_crops
        overlap_margins = self.model.config.overlap_margins
        base_image_input_size = self._mm_preprocessor.base_image_input_size
        image_patch_size = self._mm_preprocessor.image_patch_size
        
        # Calculate crop parameters
        base_image_input_d = image_patch_size
        left_margin, right_margin = overlap_margins
        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d
        
        # Current tiling
        current_tiling = analysis["tiling"]
        current_crops = current_tiling["total_crops"]
        
        # Analyze all possible tilings
        all_tilings = []
        for i in range(1, max_crops + 1):
            for j in range(1, max_crops + 1):
                if i * j <= max_crops:
                    all_tilings.append((i, j, i * j))
        
        # Calculate required scaling for each tiling
        effective_h = original_h - total_margin_pixels
        effective_w = original_w - total_margin_pixels
        
        tiling_analysis = []
        for i, j, n_crops in all_tilings:
            required_h = i * crop_window_size
            required_w = j * crop_window_size
            
            # Calculate required scale
            scale_h = required_h / effective_h if effective_h > 0 else float('inf')
            scale_w = required_w / effective_w if effective_w > 0 else float('inf')
            required_scale = max(scale_h, scale_w)
            
            # Calculate resulting image size after scaling
            scaled_h = int(original_h * required_scale)
            scaled_w = int(original_w * required_scale)
            
            tiling_analysis.append({
                "tiling": (int(i), int(j)),
                "num_crops": int(n_crops),
                "required_scale": float(required_scale),
                "scaled_size": (int(scaled_h), int(scaled_w)),
                "is_current": (i, j) == (current_tiling["rows"], current_tiling["cols"]),
            })
        
        # Sort by num_crops
        tiling_analysis.sort(key=lambda x: x["num_crops"])
        
        # Find alternative methods to control vision tokens
        # Method 1: Change max_crops
        alternative_max_crops = []
        for test_max in [6, 9, 12, 15, 18]:
            if test_max != max_crops:
                test_tiling = select_tiling(
                    effective_h,
                    effective_w,
                    crop_window_size,
                    test_max
                )
                test_crops = test_tiling[0] * test_tiling[1]
                alternative_max_crops.append({
                    "max_crops": int(test_max),
                    "resulting_tiling": (int(test_tiling[0]), int(test_tiling[1])),
                    "resulting_crops": int(test_crops),
                })
        
        # Method 2: Analyze how image size affects tiling
        # Calculate what image sizes would trigger different crop counts
        size_requirements = {}
        for i, j, n_crops in all_tilings:
            if n_crops not in size_requirements:
                size_requirements[n_crops] = []
            # Calculate what image size would fit exactly (no scaling)
            req_h = i * crop_window_size + total_margin_pixels
            req_w = j * crop_window_size + total_margin_pixels
            size_requirements[n_crops].append({
                "tiling": (int(i), int(j)),
                "ideal_size": (int(req_h), int(req_w)),
                "aspect_ratio": float(req_w / req_h if req_h > 0 else 0),
            })
        
        return {
            "current_config": {
                "max_crops": int(max_crops),
                "tiling": (int(current_tiling["rows"]), int(current_tiling["cols"])),
                "num_crops": int(current_crops),
                "image_size": (int(original_h), int(original_w)),
                "effective_size": (int(effective_h), int(effective_w)),
            },
            "all_possible_tilings": tiling_analysis,
            "alternative_max_crops": alternative_max_crops,
            "size_requirements": {str(k): v for k, v in size_requirements.items()},
            "control_methods": {
                "method_1_change_max_crops": "Adjust max_crops parameter to limit maximum number of crops",
                "method_2_resize_image": "Resize image to trigger different tiling configurations",
                "method_3_change_overlap": "Adjust overlap_margins to change crop_window_size",
                "method_4_change_base_size": "Adjust base_image_input_size to change crop_window_size",
            },
        }
    
    def save_original_image(
        self,
        image: np.ndarray,
        crops_dir: Path,
        image_id: int = None,
        example_id: int = None
    ) -> str:
        """Save original image.
        
        Args:
            image: Original image array
            crops_dir: Directory to save images
            image_id: Image ID for naming
            example_id: Example ID for naming
        
        Returns:
            Path to saved image (relative to output_dir)
        """
        from PIL import Image
        
        # Save original image
        prefix = f"img_{image_id}" if image_id else f"ex_{example_id}"
        original_filename = f"{prefix}_original.png"
        original_path = crops_dir / original_filename
        
        Image.fromarray(image).save(original_path)
        return str(original_path.relative_to(self.output_dir))
    
    def save_crop_images(
        self,
        image: np.ndarray,
        analysis: Dict,
        crops_dir: Path,
        image_id: int = None,
        example_id: int = None
    ) -> List[str]:
        """Save individual crop images.
        
        Args:
            image: Original image array
            analysis: Crop analysis results
            crops_dir: Directory to save crops
            image_id: Image ID for naming
            example_id: Example ID for naming
        
        Returns:
            List of saved crop image paths
        """
        original_h, original_w = image.shape[:2]
        resized_h = analysis["resized_size"]["height"]
        resized_w = analysis["resized_size"]["width"]
        
        # Resize image
        from PIL import Image
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((resized_w, resized_h), Image.BILINEAR)
        resized_array = np.array(resized_image)
        
        crop_paths = []
        for crop_info in analysis["crops"]:
            crop_idx = crop_info["crop_idx"]
            coords = crop_info["resized_coords"]
            
            # Extract crop from resized image
            y0, y1 = coords["y0"], coords["y1"]
            x0, x1 = coords["x0"], coords["x1"]
            crop = resized_array[y0:y1, x0:x1]
            
            # Save crop
            prefix = f"img_{image_id}" if image_id else f"ex_{example_id}"
            crop_filename = f"{prefix}_crop_{crop_idx:02d}.png"
            crop_path = crops_dir / crop_filename
            
            Image.fromarray(crop).save(crop_path)
            crop_paths.append(str(crop_path.relative_to(self.output_dir)))
        
        return crop_paths
    
    def plot_crop_overlap(self, results: List[Dict]):
        """Plot crop overlap visualization."""
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            image_id = result.get("image_id", "unknown")
            original_size = result["original_size"]
            tiling = result["tiling"]
            crops = result["crops"]
            overlap_stats = result["overlap_statistics"]
            
            # Plot 1: Original image with crop boundaries (separate figure)
            fig1 = plt.figure(figsize=(10, 8))
            ax1 = fig1.add_subplot(111)
            ax1.set_xlim(0, original_size["width"])
            ax1.set_ylim(0, original_size["height"])
            ax1.set_aspect('equal')
            ax1.invert_yaxis()
            ax1.set_title(f"Original Size {original_size['width']}×{original_size['height']}\n"
                         f"Tiling: {tiling['rows']}×{tiling['cols']} = {tiling['total_crops']} crops\n"
                         f"Overlap: {overlap_stats['overlap_ratio']:.1%}, "
                         f"Redundancy: {overlap_stats['redundancy_ratio']:.2f}x",
                         fontsize=18)
            ax1.set_xlabel("Width (pixels)", fontsize=16)
            ax1.set_ylabel("Height (pixels)", fontsize=16)
            ax1.tick_params(labelsize=14)
            
            # Unified color palette (ggthemes Classic_10 - high saturation, colorblind-friendly)
            # Official Classic_10: https://emilhvitfeldt.github.io/r-color-palettes/discrete/ggthemes/Classic_10/
            # Full Classic_10: #1F77B4, #FF7F0E, #2CA02C, #D62728, #9467BD, #8C564B, #E377C2, #7F7F7F, #BCBD22, #17BECF
            color_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']
            # Cycle through colors if we have more crops than colors
            colors = [color_palette[i % len(color_palette)] for i in range(len(crops))]
            for crop, color in zip(crops, colors):
                coords = crop["original_coords"]
                rect = patches.Rectangle(
                    (coords["x0"], coords["y0"]),
                    coords["x1"] - coords["x0"],
                    coords["y1"] - coords["y0"],
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none'
                )
                ax1.add_patch(rect)
                ax1.text(
                    (coords["x0"] + coords["x1"]) / 2,
                    (coords["y0"] + coords["y1"]) / 2,
                    f"C{crop['crop_idx']}",
                    ha='center',
                    va='center',
                    fontsize=12,
                    color=color,
                    weight='bold'
                )
            
            plt.tight_layout()
            plt.savefig(
                fig_dir / f"exp6_crop_overlap_image_{image_id}_crops.png",
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            # Plot 2: Statistics (separate figure)
            fig2 = plt.figure(figsize=(8, 6))
            ax2 = fig2.add_subplot(111)
            stats = overlap_stats
            categories = ["Total\nPatches", "Unique\nPatches", "Overlapping\nPatches"]
            values = [
                stats["total_patch_area"],
                stats["unique_patch_area"],
                stats["overlapping_patches"]
            ]
            # Unified color palette (ggthemes Classic_10 - high saturation, colorblind-friendly)
            colors_bar = ['#1F77B4', '#FF7F0E', '#2CA02C']  # Classic_10 colors 1, 2, 3
            bars = ax2.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1.5)
            ax2.set_ylabel("Number of Patches", fontsize=16)
            ax2.set_title("Patch Coverage Statistics", fontsize=18)
            ax2.tick_params(labelsize=14)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(val)}',
                        ha='center', va='bottom', fontsize=14, weight='bold')
            
            # Add text statistics
            stats_text = (
                f"Overlap Ratio: {stats['overlap_ratio']:.2%}\n"
                f"Redundancy Ratio: {stats['redundancy_ratio']:.2f}x\n"
                f"Avg Overlap Count: {stats['avg_overlap_count']:.2f}"
            )
            ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes,
                    fontsize=14, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(
                fig_dir / f"exp6_crop_overlap_image_{image_id}_statistics.png",
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            # Plot 3: Crops stitched comparison with original (separate figure)
            self.plot_crops_comparison(result, fig_dir)
        
        log.info(f"Saved {len(results)} visualization(s) to {fig_dir}")
    
    def plot_crops_comparison(self, result: Dict, fig_dir: Path):
        """Plot stitched crops compared with original image."""
        image_id = result.get("image_id", "unknown")
        original_image_path = result.get("original_image_path")
        
        if not original_image_path:
            log.warning(f"No original image path for image {image_id}, skipping comparison plot")
            return
        
        # Load original image
        original_path = self.output_dir / original_image_path
        if not original_path.exists():
            log.warning(f"Original image not found: {original_path}, skipping comparison plot")
            return
        
        from PIL import Image
        original_img = Image.open(original_path)
        original_array = np.array(original_img)
        
        # Load and stitch crops
        crop_image_paths = result.get("crop_image_paths", [])
        if not crop_image_paths:
            log.warning(f"No crop images for image {image_id}, skipping comparison plot")
            return
        
        # Sort crops by index
        crops_data = []
        for crop_path in crop_image_paths:
            full_path = self.output_dir / crop_path
            if full_path.exists():
                crop_img = Image.open(full_path)
                crop_array = np.array(crop_img)
                # Extract crop index from filename
                crop_idx = int(full_path.stem.split('_')[-1])
                crops_data.append((crop_idx, crop_array))
        
        crops_data.sort(key=lambda x: x[0])
        
        # Get tiling info
        tiling = result["tiling"]
        rows, cols = tiling["rows"], tiling["cols"]
        crops = result["crops"]
        
        # Create stitched image
        # Use crop window size from analysis for consistent sizing
        crop_window_size = result["crop_config"]["crop_window_size"]
        crop_h = crop_w = crop_window_size
        
        # Handle crops that might have different sizes (edge crops with margins)
        stitched_h = rows * crop_h
        stitched_w = cols * crop_w
        stitched_image = np.zeros((stitched_h, stitched_w, 3), dtype=np.uint8)
        
        # Place crops in stitched image
        for crop_idx, crop_array in crops_data:
            # Find crop position in tiling
            crop_info = next((c for c in crops if c["crop_idx"] == crop_idx), None)
            if crop_info:
                tiling_pos = crop_info["tiling_pos"]
                i, j = tiling_pos[0], tiling_pos[1]
                y0 = i * crop_h
                x0 = j * crop_w
                
                # Resize crop to fit if needed (crops at edges might be different sizes)
                actual_h, actual_w = crop_array.shape[:2]
                if actual_h != crop_h or actual_w != crop_w:
                    # Resize to match expected size
                    from PIL import Image
                    crop_pil = Image.fromarray(crop_array)
                    crop_resized = crop_pil.resize((crop_w, crop_h), Image.BILINEAR)
                    crop_array = np.array(crop_resized)
                
                y1 = y0 + crop_h
                x1 = x0 + crop_w
                stitched_image[y0:y1, x0:x1] = crop_array
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Original image with crop boundaries
        ax1 = axes[0]
        ax1.imshow(original_array)
        ax1.set_title(f"Original Image\n{result['original_size']['width']}×{result['original_size']['height']} pixels",
                     fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Draw crop boundaries on original
        color_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']
        colors = [color_palette[i % len(color_palette)] for i in range(len(crops))]
        
        for crop, color in zip(crops, colors):
            coords = crop["original_coords"]
            rect = patches.Rectangle(
                (coords["x0"], coords["y0"]),
                coords["x1"] - coords["x0"],
                coords["y1"] - coords["y0"],
                linewidth=3,
                edgecolor=color,
                facecolor='none'
            )
            ax1.add_patch(rect)
            # Add crop label
            ax1.text(
                (coords["x0"] + coords["x1"]) / 2,
                coords["y0"] + 20,
                f"C{crop['crop_idx']}",
                ha='center',
                va='top',
                fontsize=12,
                color=color,
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color, linewidth=2)
            )
        
        # Right: Stitched crops with boundaries
        ax2 = axes[1]
        ax2.imshow(stitched_image)
        ax2.set_title(f"Stitched Crops\n{rows}×{cols} tiling = {tiling['total_crops']} crops\n"
                     f"Overlap: {result['overlap_statistics']['overlap_ratio']:.1%}, "
                     f"Redundancy: {result['overlap_statistics']['redundancy_ratio']:.2f}x",
                     fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        # Draw grid lines on stitched image
        for i in range(rows + 1):
            y = i * crop_h
            ax2.axhline(y, color='red', linewidth=2, linestyle='--', alpha=0.7)
        for j in range(cols + 1):
            x = j * crop_w
            ax2.axvline(x, color='red', linewidth=2, linestyle='--', alpha=0.7)
        
        # Add crop labels on stitched image
        for crop_idx, crop_array in crops_data:
            crop_info = next((c for c in crops if c["crop_idx"] == crop_idx), None)
            if crop_info:
                tiling_pos = crop_info["tiling_pos"]
                i, j = tiling_pos[0], tiling_pos[1]
                color = colors[crop_idx % len(colors)]
                x_center = j * crop_w + crop_w / 2
                y_center = i * crop_h + crop_h / 2
                ax2.text(
                    x_center, y_center,
                    f"C{crop_idx}",
                    ha='center',
                    va='center',
                    fontsize=14,
                    color=color,
                    weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=color, linewidth=2)
                )
        
        plt.tight_layout()
        plt.savefig(
            fig_dir / f"exp6_crop_overlap_image_{image_id}_comparison.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Exp 6: Crop Overlap Analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="coco_2014_vqa", help="Dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to search (if not using exp2)")
    parser.add_argument("--max_size", type=int, default=500, help="Maximum image size (height or width) to consider (if not using exp2)")
    parser.add_argument("--analyze_top_k", type=int, default=5, help="Number of smallest images to analyze in detail (if not using exp2)")
    parser.add_argument("--exp2_results", type=str, default=None, help="Path to exp2_component_profiling.json (if using exp2 results)")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to select (evenly distributed across crop count range, if using exp2)")
    parser.add_argument("--target_max_crops", type=int, default=None, help="Target maximum crops to include (e.g., 12 or 13). Will select closest available if not found.")
    parser.add_argument("--save_crop_images", action="store_true", default=True, help="Save individual crop images")
    parser.add_argument("--no_save_crop_images", dest="save_crop_images", action="store_false", help="Don't save individual crop images")
    parser.add_argument("--output_dir", type=str, default="./results/motivation/exp6", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    experiment = CropOverlapAnalysis(args.model_path, args.output_dir, args.device)
    experiment.run(
        args.dataset,
        args.split,
        args.num_samples,
        args.max_size,
        args.analyze_top_k,
        args.exp2_results,
        args.num_images,
        args.target_max_crops,
        args.save_crop_images
    )


if __name__ == "__main__":
    main()

