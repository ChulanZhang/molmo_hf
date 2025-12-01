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
        super().__init__(model_path, output_dir, device)
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
        
        n_crops = tiling[0] * tiling[1]
        
        # Calculate resized image dimensions
        resized_h = tiling[0] * crop_window_size + total_margin_pixels
        resized_w = tiling[1] * crop_window_size + total_margin_pixels
        
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
                    "crop_idx": i * tiling[1] + j,
                    "tiling_pos": (i, j),
                    "resized_coords": {
                        "y0": int(y0),
                        "y1": int(y0 + crop_window_size),
                        "x0": int(x0),
                        "x1": int(x0 + crop_window_size),
                    },
                    "original_coords": {
                        "y0": orig_y0,
                        "y1": orig_y1,
                        "x0": orig_x0,
                        "x1": orig_x1,
                    },
                    "patch_coords": {
                        "y0": crop_y0,
                        "h": crop_h,
                        "x0": crop_x0,
                        "w": crop_w,
                    },
                    "pooled_dims": {
                        "h": pooled_h,
                        "w": pooled_w,
                    },
                    "num_patches": crop_patches_count,
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
                "total_crops": n_crops,
            },
            "crop_config": {
                "max_crops": max_crops,
                "overlap_margins": overlap_margins,
                "base_image_input_size": base_image_input_size,
                "crop_window_size": int(crop_window_size),
                "crop_patches": crop_patches,
                "crop_window_patches": crop_window_patches,
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
    
    def run(
        self,
        dataset_name: str,
        split: str,
        num_samples: int = 100,
        max_size: int = 500,
        analyze_top_k: int = 5
    ):
        """Run Experiment 6: Crop Overlap Analysis."""
        with Timer("Exp 6: Crop Overlap Analysis"):
            log.info(f"Starting Exp 6: Crop Overlap Analysis...")
            log.info(f"Finding small images (max_size={max_size})...")
            
            # Find small images
            small_images = self.find_small_images(dataset_name, split, num_samples, max_size)
            
            if len(small_images) == 0:
                log.warning("No small images found! Try increasing max_size or num_samples.")
                return []
            
            # Analyze top K smallest images
            analyze_images = small_images[:analyze_top_k]
            log.info(f"Analyzing {len(analyze_images)} smallest images...")
            
            # Reload dataset to get actual images
            from molmo.data import get_dataset_by_name
            from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
            from molmo.data.data_formatter import DataFormatter
            from molmo.data.collator import MMCollator
            from molmo.data.deterministic_dataset import DeterministicDataset
            from torch.utils.data import DataLoader
            
            dataset = get_dataset_by_name(dataset_name, split=split)
            mm_preprocessor = MultiModalPreprocessor(
                tokenizer=self.tokenizer,
                crop_mode=self.model.config.crop_mode,
                max_crops=self.model.config.max_crops,
                overlap_margins=self.model.config.overlap_margins,
                image_padding_mask=bool(self.model.config.image_padding_embed),
            )
            formatter = DataFormatter(
                prompt_templates=self.model.config.prompt_type,
                message_format=self.model.config.message_formatting,
                system_prompt=self.model.config.system_prompt_kind,
                always_start_with_space=self.model.config.always_start_with_space,
            )
            preprocessor = Preprocessor(
                formater=formatter,
                mm_preprocessor=mm_preprocessor,
                for_inference=True,
            )
            det_dataset = DeterministicDataset(dataset, preprocessor, seed=42)
            
            # Store mm_preprocessor for analysis
            self._mm_preprocessor = mm_preprocessor
            
            results = []
            for img_info in analyze_images:
                log.info(f"Analyzing image {img_info['index']}: size={img_info['image_size']}, "
                        f"image_id={img_info.get('image_id')}")
                
                # Get the example
                example = det_dataset[img_info['index']]
                
                # Load image
                from molmo.data.utils import load_image
                if "image" in example:
                    if isinstance(example["image"], str):
                        image = load_image(example["image"])
                    else:
                        image = example["image"]
                else:
                    log.warning(f"Image {img_info['index']} has no image data, skipping...")
                    continue
                
                # Analyze crop overlap
                analysis = self.analyze_crop_overlap(
                    image,
                    image_id=img_info.get("image_id"),
                    example_id=img_info.get("example_id")
                )
                
                analysis["search_info"] = img_info
                results.append(analysis)
                
                log.info(f"  Tiling: {analysis['tiling']}")
                log.info(f"  Overlap ratio: {analysis['overlap_statistics']['overlap_ratio']:.2%}")
                log.info(f"  Redundancy ratio: {analysis['overlap_statistics']['redundancy_ratio']:.2f}x")
            
            # Save results
            self.save_results(
                {
                    "results": results,
                    "summary": {
                        "num_analyzed": len(results),
                        "max_size_threshold": max_size,
                        "num_samples_searched": num_samples,
                    }
                },
                "exp6_crop_overlap_analysis.json"
            )
            
            # Generate visualization
            self.plot_crop_overlap(results)
            
            log.info(f"Exp 6 completed! Results saved to {self.output_dir}")
            return results
    
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
        
        log.info(f"Saved {len(results)} visualization(s) to {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description="Exp 6: Crop Overlap Analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="coco_2014_vqa", help="Dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to search")
    parser.add_argument("--max_size", type=int, default=500, help="Maximum image size (height or width) to consider")
    parser.add_argument("--analyze_top_k", type=int, default=5, help="Number of smallest images to analyze in detail")
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
        args.analyze_top_k
    )


if __name__ == "__main__":
    main()

