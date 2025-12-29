import dataclasses
import math
import warnings
from typing import List, Optional, Union, Any, Tuple, Dict, TYPE_CHECKING

import PIL
from PIL import ImageFile
from PIL import ImageOps

from molmo.tokenizer import get_special_token_ids, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_COL_TOKEN, IMAGE_PROMPT

if TYPE_CHECKING:
    from molmo.data.data_formatter import DataFormatter


def setup_pil():
    PIL.Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import convert_image_dtype
import einops

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    is_valid_image,
)
from transformers.processing_utils import ImagesKwargs
from transformers.image_processing_utils import BaseImageProcessor
from transformers.utils import logging

# Note: This file was moved from molmo/data/model_preprocessor.py to molmo/preprocessors/
# to better organize preprocessing code separate from dataset management code.
# DataFormatter is imported locally in Preprocessor.__call__ to avoid circular imports.


def load_image(image_path):
    setup_pil()  # Call here so the setting is applied in multi-processing contexts
    if isinstance(image_path, PIL.Image.Image):
        # Avoid annoying palette transparency warnings filling up the logs
        with warnings.catch_warnings(record=True) as w:
            image = image_path.convert("RGB")
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            pass
        return np.array(image)
    elif isinstance(image_path, np.ndarray):
        assert len(image_path.shape) == 3, "Image should have 3 dimensions"
        assert image_path.shape[2] == 3, "Image should have 3 channels"
        assert image_path.dtype == np.uint8, "Image should have uint8 type"
        return image_path
    else:
        with PIL.Image.open(image_path) as image:
            return load_image(image)


def resize_and_pad(
    image,
    desired_output_size,
    is_training=False,
    resize_method="torch-bilinear",
    pad_value=0,
    rng=np.random
):
    """Resize an image while padding to preserve uts aspect ratio."""
    desired_height, desired_width = desired_output_size
    height, width = image.shape[:2]

    # Cast into float32 since the training code did this in float32 and it (very rarely) effects
    # the results after rounding.
    image_scale_y = np.array(desired_height, np.float32) / np.array(height, np.float32)
    image_scale_x = np.array(desired_width, np.float32) / np.array(width, np.float32)
    image_scale = min(image_scale_x, image_scale_y)
    scaled_height = int(np.array(height, np.float32) * image_scale)
    scaled_width = int(np.array(width, np.float32) * image_scale)

    if resize_method in ["tensorflow", "tensorflow-random"]:
        # This how the original training code did resizing, it can produce slightly different
        # results then using torch resize so we keep it just in case
        import tensorflow as tf
        if resize_method == "tensorflow-random" and is_training:
            resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
            mode = resize_methods[rng.randint(len(resize_methods))]
            mode = getattr(tf.image.ResizeMethod, mode)
        else:
            mode = tf.image.ResizeMethod.BILINEAR
        image = tf.image.convert_image_dtype(tf.constant(image), dtype=tf.float32)
        image = tf.image.resize(
            image,
            [scaled_height, scaled_width],
            method=mode,
            antialias=True,
        )
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = image.numpy()
    elif resize_method in ["torch-bilinear", "torch-rng"]:
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
        image = convert_image_dtype(image)  # resize in float32 to match the training code
        if resize_method == "torch-rng"  and is_training:
            options = [InterpolationMode.BILINEAR, InterpolationMode.NEAREST_EXACT,
                       InterpolationMode.BICUBIC, InterpolationMode.LANCZOS, InterpolationMode.HAMMING]
            mode = options[rng.randint(len(options))]
        else:
            mode = InterpolationMode.BILINEAR
        image = torchvision.transforms.Resize([scaled_height, scaled_width], mode, antialias=True)(image)
        image = torch.clip(image, 0.0, 1.0)
        image = torch.permute(image, [1, 2, 0]).numpy()
    else:
        raise NotImplementedError(resize_method)

    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    padding = [
        [top_pad, desired_height - scaled_height - top_pad],
        [left_pad, desired_width - scaled_width - left_pad],
        [0, 0]
    ]
    image_mask = np.pad(np.ones_like(image[:, :, 0], dtype=bool), padding[:2])
    image = np.pad(image, padding, constant_values=pad_value)
    return image, image_mask


def resize_and_crop_to_fill(
    image,
    desired_output_size,
    is_training=False,
    resize_method="torch-bilinear",
    pad_value=0,
    rng=np.random
):
    """
    Resize an image to cover the desired_output_size (no letterbox shrink),
    then center-crop (or pad if still smaller). This ensures small images
    are upscaled to the target canvas, keeping aspect ratio and minimizing
    padding.
    """
    desired_height, desired_width = desired_output_size
    height, width = image.shape[:2]

    # Scale to cover the target canvas (use max ratio, upsample if needed)
    image_scale_y = np.array(desired_height, np.float32) / np.array(height, np.float32)
    image_scale_x = np.array(desired_width, np.float32) / np.array(width, np.float32)
    image_scale = max(image_scale_x, image_scale_y)

    scaled_height = int(np.ceil(np.array(height, np.float32) * image_scale))
    scaled_width = int(np.ceil(np.array(width, np.float32) * image_scale))
    scaled_height = max(scaled_height, desired_height)
    scaled_width = max(scaled_width, desired_width)

    # Resize (reuse torch/tf paths for consistency with training)
    if resize_method in ["tensorflow", "tensorflow-random"]:
        import tensorflow as tf
        if resize_method == "tensorflow-random" and is_training:
            resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
            mode = resize_methods[rng.randint(len(resize_methods))]
            mode = getattr(tf.image.ResizeMethod, mode)
        else:
            mode = tf.image.ResizeMethod.BILINEAR
        image = tf.image.convert_image_dtype(tf.constant(image), dtype=tf.float32)
        image = tf.image.resize(
            image,
            [scaled_height, scaled_width],
            method=mode,
            antialias=True,
        )
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = image.numpy()
    elif resize_method in ["torch-bilinear", "torch-rng"]:
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
        image = convert_image_dtype(image)  # float32
        if resize_method == "torch-rng" and is_training:
            options = [InterpolationMode.BILINEAR, InterpolationMode.NEAREST_EXACT,
                       InterpolationMode.BICUBIC, InterpolationMode.LANCZOS, InterpolationMode.HAMMING]
            mode = options[rng.randint(len(options))]
        else:
            mode = InterpolationMode.BILINEAR
        image = torchvision.transforms.Resize([scaled_height, scaled_width], mode, antialias=True)(image)
        image = torch.clip(image, 0.0, 1.0)
        image = torch.permute(image, [1, 2, 0]).numpy()
    else:
        raise NotImplementedError(resize_method)

    # Center-crop if larger; pad if still smaller (rare)
    h, w = image.shape[:2]
    crop_top = max(0, (h - desired_height) // 2)
    crop_left = max(0, (w - desired_width) // 2)
    crop_bottom = crop_top + desired_height
    crop_right = crop_left + desired_width

    cropped = image[crop_top:crop_bottom, crop_left:crop_right]

    # If any dim is still short (due to rounding), pad symmetrically
    pad_h = desired_height - cropped.shape[0]
    pad_w = desired_width - cropped.shape[1]
    if pad_h > 0 or pad_w > 0:
        top_pad = pad_h // 2
        bottom_pad = pad_h - top_pad
        left_pad = pad_w // 2
        right_pad = pad_w - left_pad
        padding = [
            [top_pad, bottom_pad],
            [left_pad, right_pad],
            [0, 0]
        ]
        image_mask = np.pad(np.ones_like(cropped[:, :, 0], dtype=bool), padding[:2])
        cropped = np.pad(cropped, padding, constant_values=pad_value)
    else:
        image_mask = np.ones_like(cropped[:, :, 0], dtype=bool)

    return cropped, image_mask


def metaclip_resize(image, desired_output_size):
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    if torch.is_floating_point(image):
        image = torchvision.transforms.Resize(
            desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
        image = torch.clip(image, 0.0, 1.0)
    else:
        assert image.dtype == torch.uint8, "Expected float images or uint8 images, but got {}".format(image.dtype)
        image = torchvision.transforms.Resize(
            desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
        image = image.to(torch.float32)
        image = torch.clip(image, 0, 255)
        image = image / 255.0
    resized = torch.permute(image, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)
    return resized, image_mask


def siglip_resize_and_pad(
    image: np.ndarray,
    desired_output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    dtype = image.dtype
    if torch.is_floating_point(image):
        in_min = 0.0
        in_max = 1.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BILINEAR,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(dtype)
    else:
        assert image.dtype == torch.uint8, "SigLIP expects float images or uint8 images, but got {}".format(image.dtype)
        in_min = 0.0
        in_max = 255.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BILINEAR,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0, 255).to(dtype)

    resized = resized.to(torch.float32)
    resized = (resized - in_min) / (in_max - in_min)

    resized = torch.permute(resized, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)
    
    return resized, image_mask


def dino_resize_and_pad(
    image: np.ndarray,
    desired_output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    dtype = image.dtype
    if torch.is_floating_point(image):
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BICUBIC,
            antialias=True,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(torch.float32)
    else:
        assert image.dtype == torch.uint8, "DINOv2 expects float images or uint8 images, but got {}".format(image.dtype)
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BICUBIC,
            antialias=True,
        )(image)
        resized = torch.clip(resized, 0, 255).to(torch.float32)
        resized = resized / 255.0
    
    resized = torch.permute(resized, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)

    return resized, image_mask


def select_tiling(h, w, patch_size, max_num_crops, tier=None):
    """
    Select optimal tiling configuration for an image.
    
    Args:
        h: Image height (after subtracting margins)
        w: Image width (after subtracting margins)
        patch_size: Crop window size (e.g., 224)
        max_num_crops: Maximum number of crops allowed
        tier: Optional dict with keys:
            - min_crops: Minimum number of crops (default: 1)
            - max_crops: Maximum number of crops (default: max_num_crops)
            - preferred_crops: List of preferred crop counts to try first
            - mismatch_threshold: Maximum acceptable mismatch for preferred crops (default: 0.3)
            If provided, selects best crop count within tier range based on aspect ratio.
            If None, uses adaptive selection based on image size.
    
    Returns:
        (rows, cols) tiling configuration
    """
    aspect_ratio = w / h if h > 0 else 1.0
    
    # Tier-based selection: find best crop count within tier range
    if tier is not None:
        min_crops = tier.get("min_crops", 1)
        max_crops_in_tier = min(tier.get("max_crops", max_num_crops), max_num_crops)
        preferred_crops = tier.get("preferred_crops", [])
        mismatch_threshold = tier.get("mismatch_threshold", 0.3)
        
        # Helper function to find best tiling for a given crop count
        def find_best_tiling_for_crops(crops):
            """Find best tiling for a specific crop count based on aspect ratio."""
            tilings = []
            for i in range(1, crops + 1):
                if crops % i == 0:
                    j = crops // i
                    tilings.append((i, j))
            
            if not tilings:
                return None, float('inf')
            
            best_tiling = None
            best_mismatch = float('inf')
            for rows, cols in tilings:
                tiling_h = rows * patch_size
                tiling_w = cols * patch_size
                tiling_aspect = tiling_w / tiling_h if tiling_h > 0 else 1.0
                mismatch = abs(tiling_aspect - aspect_ratio)
                if mismatch < best_mismatch:
                    best_mismatch = mismatch
                    best_tiling = (rows, cols)
            return best_tiling, best_mismatch
        
        best_crops = None
        best_tiling = None
        best_mismatch = float('inf')
        
        # First, try preferred crop counts
        for crops in preferred_crops:
            if crops < min_crops or crops > max_crops_in_tier:
                continue
            tiling, mismatch = find_best_tiling_for_crops(crops)
            if tiling is not None and mismatch < best_mismatch:
                best_mismatch = mismatch
                best_crops = crops
                best_tiling = tiling
        
        # If mismatch is acceptable, return preferred crop
        if best_mismatch < mismatch_threshold and best_tiling is not None:
            return best_tiling
        
        # Otherwise, try all crop counts in tier range
        for crops in range(min_crops, max_crops_in_tier + 1):
            if crops in preferred_crops:
                continue  # Already tried
            tiling, mismatch = find_best_tiling_for_crops(crops)
            if tiling is not None and mismatch < best_mismatch:
                best_mismatch = mismatch
                best_crops = crops
                best_tiling = tiling
        
        # Return best found, or fallback to smallest tiling in tier
        if best_tiling is not None:
            return best_tiling
        # Fallback: use smallest tiling in tier
        fallback_tiling, _ = find_best_tiling_for_crops(min_crops)
        if fallback_tiling is not None:
            return fallback_tiling
        # Last resort: use (1, min_crops)
        return (1, min_crops)
    
    # Original adaptive selection logic (when tier is None)
    # Generate all possible tilings up to max_num_crops
    tilings = []
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i * j <= max_num_crops:
                tilings.append((i, j))
    
    # Sort so argmin and argmax favour smaller tilings in the event of a tie
    tilings.sort(key=lambda x: (x[0]*x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)  # [n_resolutions, 2]
    candidate_resolutions = candidate_tilings * patch_size  # [n_resolutions, 2]

    # How much we would need to scale the image to fit exactly in each tiling
    original_size = np.stack([h, w], dtype=np.float32)  # [1, 2]

    # The original size can be zero in rare cases if the image is smaller than the margin
    # In those cases letting the scale become infinite means the tiling is based on the
    # other side, or falls back to the smallest tiling
    with np.errstate(divide='ignore'):
        required_scale_d = candidate_resolutions.astype(np.float32) / original_size
        required_scale = np.min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        ix = np.argmax(required_scale)
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    return candidate_tilings[ix]


def pixels_to_patches(array, patch_size):
    """Reshape an image of [h, w, 3] -> [n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        w, h, c = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 2, 1, 3, 4])
        array = np.reshape(array, [h_patches*w_patches, patch_size*patch_size*c])
    else:
        w, h = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 2, 1, 3])
        array = np.reshape(array, [h_patches*w_patches, patch_size*patch_size])
    return array


def batch_pixels_to_patches(array, patch_size):
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, w, h = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size])
        return array
    else:
        n_crops, w, h, c = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size*c])
        return array


@dataclasses.dataclass
class MultiModalPreprocessor:
    """
    Converts text/images inputs into tensors that can be used in the forward method
    for the a model
    """
    tokenizer: Any
    loss_token_weighting: Optional[str] = None

    # How to crops/resize images
    normalize: str = "openai"
    crop_mode: str = "resize"
    # max_crops: int = 6
    max_crops: int = 12
    tier: Optional[Dict[str, Any]] = None  # Tier configuration for adaptive crop selection
    overlap_margins: Tuple[int, int] = (4, 4)
    resize: str = "default"
    use_col_tokens: bool = True

    # Data about the ViT and connector we need when deciding the crops
    base_image_input_size: Tuple[int, int] = (336, 336)
    image_pooling_w: int = 2
    image_pooling_h: int = 2
    image_token_length_w: int = 12
    image_token_length_h: int = 12
    image_patch_size: int = 14
    image_padding_mask: Union[bool, int] = False
    # If True, keep padded patches as valid tokens (no -100), so vision token
    # count matches theoretical (num_crops+1)*144 even with overlap.
    force_full_tokens: bool = False
    pad_value: float = 0
    # If True, resize to fill the target canvas (upscale small images),
    # then center-crop/pad to the desired size. Useful to better match
    # the target tiling and reduce invalid tokens when force_full_tokens=False.
    resize_to_fill: bool = False

    image_patch_token_id: int = dataclasses.field(init=False)
    image_col_token_id: int = dataclasses.field(init=False)
    image_start_token_id: int = dataclasses.field(init=False)
    image_end_token_id: int = dataclasses.field(init=False)

    def __post_init__(self):
        special_tokens = get_special_token_ids(self.tokenizer)
        self.image_end_token_id = special_tokens[DEFAULT_IM_END_TOKEN]
        self.image_start_token_id = special_tokens[DEFAULT_IM_START_TOKEN]
        self.image_col_token_id = special_tokens[DEFAULT_IM_COL_TOKEN]
        self.image_patch_token_id = special_tokens[DEFAULT_IMAGE_PATCH_TOKEN]
        self.image_prompt_token_id = special_tokens[IMAGE_PROMPT]

    def _normalize(self, image):
        if self.normalize == "openai":
            image -= np.array(OPENAI_CLIP_MEAN, dtype=np.float32)[None, None, :]
            image /= np.array(OPENAI_CLIP_STD, dtype=np.float32)[None, None, :]
        elif self.normalize == "siglip":
            image = np.asarray(-1.0, dtype=np.float32) + image * np.asarray(2.0, dtype=np.float32)
        elif self.normalize == "dino":
            image -= np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
            image /= np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
        else:
            raise NotImplementedError(self.normalize)
        return image

    def resize_image(self, image, output_size, is_training, rng):
        if self.resize == "siglip":
            return siglip_resize_and_pad(image, output_size)
        elif self.resize == "dino":
            return dino_resize_and_pad(image, output_size)
        elif self.resize == "metaclip":
            return metaclip_resize(image, output_size)
        else:
            resize = "torch-bilinear" if self.resize == "default" else self.resize
            if self.resize_to_fill:
                return resize_and_crop_to_fill(
                    image, output_size, pad_value=self.pad_value, rng=rng, is_training=is_training,
                    resize_method=resize)
            else:
                return resize_and_pad(
                    image, output_size, pad_value=self.pad_value, rng=rng, is_training=is_training,
                    resize_method=resize)

    def image_to_patches_and_tokens(
        self,
        image: ImageInput,
        is_training=False,
        rng=None
    ):
        max_crops = self.max_crops
        overlap_margins = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        image_token_length_w = self.image_token_length_w
        image_token_length_h = self.image_token_length_h
        image_patch_size = self.image_patch_size

        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        tokens_per_image = image_token_length_w * image_token_length_h
        image_base_patch_w = base_image_input_size[1] // base_image_input_d
        image_base_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        if self.crop_mode == "resize":
            resized, img_mask = self.resize_image(image, base_image_input_size, is_training, rng)
            resized = self._normalize(resized)
            patches = pixels_to_patches(resized, image_patch_size)
            img_mask = pixels_to_patches(img_mask, image_patch_size)

            per_row = np.full(
                (image_token_length_w,),
                self.image_patch_token_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.image_col_token_id]], 0, dtype=np.int32)
            extra_tokens = np.tile(per_row, [image_token_length_h])
            joint = [
                [self.image_start_token_id],
                extra_tokens,
                [self.image_end_token_id],
            ]
            joint = np.concatenate(joint, 0, dtype=np.int32)
            return np.expand_dims(patches, 0), joint, None, img_mask
        if self.crop_mode == "overlap-and-resize-c2":
            # Discard this many patches from the (left/top, right/bottom) of crops
            left_margin, right_margin = overlap_margins
            # Required for compatibility with image pooling
            assert left_margin % self.image_pooling_w == 0 and right_margin % self.image_pooling_w == 0
            assert left_margin % self.image_pooling_h == 0 and right_margin % self.image_pooling_h == 0
            total_margin_pixels = base_image_input_d*(right_margin + left_margin)  # pixels removed per dim
            crop_patches = base_image_input_size[0] // base_image_input_d  # patches per crop dim
            crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
            crop_window_size = crop_window_patches * base_image_input_d

            # Decide how to tile the image, to account for the overlap margins we compute the tiling
            # as if we had an image without the margins and were using a crop size without the margins
            tiling = select_tiling(
                original_image_h - total_margin_pixels,
                original_image_w - total_margin_pixels,
                crop_window_size,
                max_crops,
                tier=self.tier
            )
            src, img_mask = self.resize_image(
                image,
                [tiling[0]*crop_window_size+total_margin_pixels, tiling[1]*crop_window_size+total_margin_pixels],
                is_training,
                rng
            )
            src = self._normalize(src)

            # Now we have to split the image into crops, while keeping track of how each patch in the
            # each crop should be ordered in the global image, this require a lot of tricky booking
            n_crops = tiling[0] * tiling[1]
            patches_arr = []
            mask_arr = []
            patch_ordering_arr = []

            # We assume hxw pooling, but can allow padding the right/bottom with extra
            # patches if the number of patches per side is not divisible by h/w
            assert (crop_patches + self.image_pooling_h - 1) // self.image_pooling_h == image_token_length_h
            assert (crop_patches + self.image_pooling_w - 1) // self.image_pooling_w == image_token_length_w
            on = 0
            on_patch = 0
            for i in range(tiling[0]):
                y0 = i*crop_window_size
                if i == 0:
                    crop_y0 = 0
                else:
                    crop_y0 = left_margin // self.image_pooling_h

                crop_h = image_base_patch_h - (right_margin + left_margin)
                if i == 0:
                    crop_h += left_margin
                if i == (tiling[0]-1):
                    crop_h += right_margin
                for j in range(tiling[1]):
                    x0 = j*crop_window_size
                    if j == 0:
                        crop_x0 = 0
                    else:
                        crop_x0 = left_margin // self.image_pooling_w

                    crop_w = image_base_patch_w - (right_margin + left_margin)
                    if j == 0:
                        crop_w += left_margin
                    if j == (tiling[1]-1):
                        crop_w += right_margin

                    pooled_w = (crop_w + self.image_pooling_w - 1) // self.image_pooling_w
                    pooled_h = (crop_h + self.image_pooling_h - 1) // self.image_pooling_h
                    after_padding_width = image_token_length_w - pooled_w - crop_x0
                    after_padding_height = image_token_length_h - pooled_h - crop_y0
                    if self.force_full_tokens:
                        # Force full 12x12 tokens: assign indices to padded positions too
                        full_tokens = image_token_length_h * image_token_length_w  # 12*12=144
                        patch_ordering_arr.append(
                            np.reshape(
                                np.arange(on, on + full_tokens, dtype=np.int32),
                                (image_token_length_h, image_token_length_w)
                            )
                        )
                        # For masks, mark all tokens as valid (padding treated as valid tokens)
                        mask_arr.append(np.ones((crop_size, crop_size), dtype=np.float32))
                        patches_arr.append(src[y0:y0+crop_size, x0:x0+crop_size])
                        on += full_tokens
                    else:
                        patch_ordering_arr.append(
                            np.pad(
                                np.reshape(
                                    np.arange(on, on+pooled_h*pooled_w, dtype=np.int32),
                                    (pooled_h, pooled_w)),
                                [[crop_y0, after_padding_height], [crop_x0, after_padding_width]],
                                constant_values=-1, mode='constant'
                            )
                        )
                        patches_arr.append(src[y0:y0+crop_size, x0:x0+crop_size])
                        mask_arr.append(img_mask[y0:y0+crop_size, x0:x0+crop_size])
                        on += pooled_h*pooled_w
                    on_patch += 1
            patches = np.stack(patches_arr)
            patch_ordering = np.stack(patch_ordering_arr)
            img_mask = np.stack(mask_arr)

            # Switch to [n_crops, n_patches, pixels_per_patch] format
            image_layout_impatch_w, image_layout_impatch_h = tiling[0], tiling[1]

            patches = batch_pixels_to_patches(patches, image_patch_size)
            img_mask = batch_pixels_to_patches(img_mask, image_patch_size)
            img_mask = img_mask.astype(np.float32).mean(axis=-1)
            patch_ordering = np.reshape(patch_ordering, [-1])
            valid = patch_ordering >= 0

            # Path order numbers the patches crop-by-crop, here we transpose
            # it to get left-to-right order
            patch_ordering_rh = np.reshape(
                patch_ordering,
                [tiling[0], tiling[1], image_token_length_h, image_token_length_w]
            )
            patch_ordering_rh = np.transpose(patch_ordering_rh, [0, 2, 1, 3])
            patch_ordering_rh = np.reshape(patch_ordering_rh, [-1])

            # The transpose will screw up which patches are masked, project the
            # new order into sparse structure of `patch_ordering` to fix it
            patch_ordering[valid] = patch_ordering_rh[patch_ordering_rh >= 0]

            def get_num_patches(num_tiles: int, pooling_size: int) -> int:
                if num_tiles > 1:
                    left_crop_window_patches = (crop_window_patches + left_margin + pooling_size - 1) // pooling_size * pooling_size
                    middle_crop_window_patches = (crop_window_patches + pooling_size - 1) // pooling_size * pooling_size
                    right_crop_window_patches = (crop_window_patches + right_margin + pooling_size - 1) // pooling_size * pooling_size
                    return left_crop_window_patches + (num_tiles - 2) * middle_crop_window_patches + right_crop_window_patches
                else:
                    single_crop_window_patches = (crop_patches + pooling_size - 1) // pooling_size * pooling_size
                    return single_crop_window_patches

            # Now build the output tokens
            if self.force_full_tokens:
                # Force exact token grid: each crop contributes exactly 12x12 tokens
                tokens_w = image_token_length_w * tiling[1]
                tokens_h = image_token_length_h * tiling[0]
                per_row = np.full(
                    (tokens_w,),
                    self.image_patch_token_id,
                    dtype=np.int32
                )
                if self.use_col_tokens:
                    per_row = np.concatenate([per_row, [self.image_col_token_id]], 0)
                joint = np.tile(per_row, [tokens_h])
                joint = [
                    [self.image_start_token_id],
                    joint,
                    [self.image_end_token_id]
                ]
            else:
                h = get_num_patches(tiling[0], self.image_pooling_h)
                w = get_num_patches(tiling[1], self.image_pooling_w)
                per_row = np.full(
                    (w // self.image_pooling_w,),
                    self.image_patch_token_id,
                    dtype=np.int32
                )
                if self.use_col_tokens:
                    per_row = np.concatenate([per_row, [self.image_col_token_id]], 0)

                joint = np.tile(per_row, [h // self.image_pooling_h])
                joint = [
                    [self.image_start_token_id],
                    joint,
                    [self.image_end_token_id]
                ]

            # Finally do the same for the global image
            resized, _ = self.resize_image(image, base_image_input_size, is_training, rng)
            resized = self._normalize(resized)
            resized = pixels_to_patches(resized, image_patch_size)
            patches = np.concatenate([np.expand_dims(resized, 0), patches], 0)

            # Global image goes first, so the order of patches in previous crops gets increased
            patch_ordering = np.where(
                patch_ordering >= 0,
                patch_ordering + tokens_per_image,
                -1
            )
            patch_ordering = np.concatenate([np.arange(0, tokens_per_image), patch_ordering], 0)
            per_row = np.full(
                (image_token_length_w,),
                self.image_patch_token_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [image_token_length_h])
            joint = [
                        [self.image_start_token_id],
                        extra_tokens,
                        [self.image_end_token_id],
                    ] + joint

            joint = np.concatenate(joint, 0)
            img_mask = np.pad(img_mask, [[0, 1], [0, 0]], constant_values=-1)
            return patches, joint, patch_ordering, img_mask
        else:
            raise NotImplementedError(self.crop_mode)

    def build_image_input_idx(
        self,
        image_tokens: np.ndarray,
        patch_order: np.ndarray,
    ):
        """Converts `patch_order` into an array mapping patch_id -> token_position"""
        tokens_per_image = self.image_token_length_w * self.image_token_length_h

        image_input_idx = image_tokens == self.image_patch_token_id
        image_input_idx = np.nonzero(image_input_idx)[0].astype(np.int32)

        n_tokens = image_input_idx.shape[0]

        # Fast path for force_full_tokens: we already constructed a dense grid of
        # tokens_per_image per (global + crop), so just reshape directly.
        if self.force_full_tokens:
            # Ensure length is multiple of tokens_per_image
            n_images = max(1, len(image_input_idx) // tokens_per_image)
            total_tokens = n_images * tokens_per_image
            image_input_idx = image_input_idx[:total_tokens]
            image_input_idx = np.reshape(image_input_idx, [-1, tokens_per_image])
            return image_input_idx

        if patch_order is not None:
            patch_order = np.reshape(patch_order, [-1])
            n_patches = patch_order.shape[0]

            valid = patch_order >= 0
            n_valid_patches = valid.sum()
            assert len(image_input_idx) == n_valid_patches

            # Get the reversed mapping of patch order (so instead of sorted position->patch_idx we
            # want patch_idx->sorted position)
            # We have to be careful to preserve the sparse structure of `patch_order` where -1 means
            # a patch is skipped
            sorted_patch_ixs = np.zeros([n_tokens], np.int32)
            sorted_patch_ixs[patch_order[valid]] = np.arange(n_valid_patches, dtype=np.int32)
            sorted_patch_ixs_ex = np.full(np.shape(patch_order), -1)
            sorted_patch_ixs_ex[valid] = sorted_patch_ixs

            # Now go from patch_idx->sorted position to patch_idx->tokens position, we need to do
            # this since the `image_tokens`` will contain special tokens interleave with the
            # tokens that will become image features
            valid = (sorted_patch_ixs_ex >= 0).astype(np.int32)
            image_input_idx = image_input_idx[sorted_patch_ixs_ex*valid]
            image_input_idx = image_input_idx*valid - 100*(1 - valid)

        image_input_idx = np.reshape(image_input_idx, [-1, tokens_per_image])
        return image_input_idx

    def preprocess(self, image, is_training: bool, rng=None):
        """Preprocesses a single image

        Returns:
            crops: (n_crops, n_patches, patch_dim) individual crops, `n_crops` might
                   change between images but the other dimension are fixed
            tokens: (n_tokens,) int32 tokens, pad tokens indicate where to insert the
                                patch features, might include other special tokens as well
            image_idx: (n_crops, n_patches) index in `tokens` to put the patch features from the
                       crops after pooling, negative values indicates patches features to exclude
            padding_mask: (n_crops, n_patches) what percent of each crop is padding, can be None
                          if the image mask is not being used.
        """
        crops, image_tokens, patch_ordering, img_mask = self.image_to_patches_and_tokens(
            image, is_training, rng)
        patch_idx = self.build_image_input_idx(
            image_tokens,
            patch_ordering,
        )
        return crops, image_tokens, patch_idx, img_mask

    def __call__(
        self,
        images,
        messages: Union[List[str], List[List[str]]],
        weight=None,
        is_training=False,
        rng=None,
        require_image_features=False
    ):
        """Interleave images and text tokens into multi-modal features for the model"""
        if len(messages) == 0:
            raise ValueError("Given empty messages")
        if not isinstance(messages[0], str) and len(messages) == 1:
            messages = messages[0]
        if isinstance(messages[0], str):
            # List of user/system/user/system ect. prompts
            loss_masks = []
            token_ids = []
            for msg_ix, message in enumerate(messages):
                has_loss = msg_ix % 2 == 1
                message_ids = self.tokenizer.encode(message)
                if has_loss:
                    message_ids.append(self.tokenizer.eos_token_id)
                token_ids += message_ids
                if weight is None:
                    loss_masks += [has_loss]*len(message_ids)
                else:
                    loss_masks += [weight if has_loss else 0]*len(message_ids)
            tokens = np.array(token_ids, dtype=np.int32)
            loss_masks = np.array(loss_masks, dtype=np.float32)
            subsegments = None
        else:
            if weight is not None:
                raise NotImplementedError("Multi-messages with weights")
            # List of lists of user/system/user/system ect. prompts
            subsegments = []
            loss_masks = []
            token_ids = []
            for message_set_ix, message_set in enumerate(messages):
                n = 0
                for msg_ix, message in enumerate(message_set):
                    has_loss = msg_ix % 2 == 1
                    message_ids = self.tokenizer.encode(message)
                    if has_loss:
                        message_ids.append(self.tokenizer.eos_token_id)
                    token_ids += message_ids
                    loss_masks.append(np.full(len(message_ids), has_loss, dtype=np.bool_))
                    n += len(message_ids)
                subsegments.append(np.full(n, message_set_ix+1, dtype=np.int32))
            tokens = np.array(token_ids, dtype=np.int32)
            loss_masks = np.concatenate(loss_masks, dtype=np.float32)
            subsegments = np.concatenate(subsegments, dtype=np.int32)
            if weight is not None:
                loss_masks *= weight
            if self.loss_token_weighting == "root_subsegments":
                loss_masks *= math.sqrt(1/len(messages))
            elif self.loss_token_weighting is not None:
                raise NotImplementedError(self.loss_token_weighting)

        if images is None or (
            isinstance(images, (list, tuple)) and len(images) == 0
        ):
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            decoder_input_tokens = np.pad(tokens, [[1, 0]], constant_values=bos)[:-1]
            data = {"input_tokens": tokens, "loss_masks": loss_masks, "target_tokens": tokens}
            if subsegments is not None:
                subsegments = np.pad(subsegments, [[1, 0]], constant_values=subsegments[0])[:-1]
                data["subsegments"] = subsegments
            if require_image_features:
                # Add size-zero image features, this can be useful to make sure all devices
                # get an image input when the image ViT is FSDP wrapped
                tokens_per_image = self.image_token_length_w * self.image_token_length_h
                n_pixels = self.image_patch_size ** 2 * 3
                h, w = self.base_image_input_size
                image_num_patch = (h//self.image_patch_size * w//self.image_patch_size)
                crops = np.zeros((0, image_num_patch, n_pixels), dtype=np.float32)
                image_idx = np.zeros((0, tokens_per_image), np.int32)
                data.update(dict(
                    images=crops,
                    image_input_idx=image_idx,
                ))
                if self.image_padding_mask:
                    data["image_masks"] = np.zeros((0, image_num_patch), dtype=np.float32)
            return data

        if not isinstance(images, (list, tuple)):
            images = [images]
        image_idx = np.argwhere(tokens == self.image_prompt_token_id)
        if len(image_idx) == 0:
            image_idx = [-1] * len(images)
        else:
            image_idx = image_idx[:, 0]
            assert len(image_idx) == len(images)

        max_total_crops = self.max_crops
        image_token_length_w = self.image_token_length_w
        image_token_length_h = self.image_token_length_h
        image_patch_size = self.image_patch_size
        base_image_input_size = self.base_image_input_size
        image_num_patch = (
            base_image_input_size[0] // image_patch_size,
            base_image_input_size[1] // image_patch_size,
        )
        image_padding_mask = self.image_padding_mask

        tokens_per_image = image_token_length_w * image_token_length_h
        n_pixels = image_patch_size * image_patch_size * 3
        n_patches = image_num_patch[0] * image_num_patch[1]

        n = len(images)
        all_crops = []
        all_image_idx = []
        out_tokens = []
        all_crop_masks = []
        all_subsegments = []
        all_loss_masks = []

        for ix in range(n):
            token_ix = image_idx[ix]
            crops, image_tokens, patch_idx, img_mask = self.preprocess(images[ix], is_training, rng)

            if token_ix == -1:  # -1 is an image inserted at the very start
                start = 0
                token_ix = 0
                end = 0
            else:
                start = 0 if ix == 0 else image_idx[ix-1] + 1
                end = token_ix + 1

            all_image_idx.append(patch_idx + token_ix)
            all_crops.append(crops)

            out_tokens.append(tokens[start:token_ix])
            all_loss_masks.append(loss_masks[start:token_ix])

            out_tokens.append(image_tokens)
            all_loss_masks.append(np.zeros(image_tokens.shape[0], dtype=np.float32))
            if subsegments is not None:
                all_subsegments.append(subsegments[start:token_ix])
                image_subsegment = 10000 if image_idx[ix] == -1 else token_ix
                all_subsegments.append(np.full(len(image_tokens), image_subsegment, np.int32))
            if image_padding_mask:
                all_crop_masks.append(img_mask)

        end = image_idx[-1] + 1
        out_tokens.append(tokens[end:])
        all_loss_masks.append(loss_masks[end:])
        if subsegments is not None:
            all_subsegments.append(subsegments[end:])

        input_ids = np.concatenate(out_tokens, 0)
        images = np.concatenate(all_crops, 0)
        image_input_idx = np.concatenate(all_image_idx, 0)
        all_loss_masks = np.concatenate(all_loss_masks, 0)

        target_tokens = input_ids
        ends_with_eos = input_ids[-1] == self.tokenizer.eos_token_id
        if not ends_with_eos and loss_masks[-1]:
            raise RuntimeError("EOS should not be masked")

        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        input_ids = np.pad(input_ids, [[1, 0]], constant_values=bos)
        if ends_with_eos:
            input_ids = input_ids[:-1]
        else:
            # We are presumably doing inference since the messages end with user response instead
            # of a target response, so these fields should not be used, but pad them anyway
            # just so everything is a consistent length
            all_loss_masks = np.pad(all_loss_masks, [[0, 1]], constant_values=-1)
            target_tokens = np.pad(target_tokens, [[0, 1]], constant_values=-1)

        image_input_idx = np.where(image_input_idx < 0, image_input_idx, image_input_idx + 1)
        out = {
            "input_tokens": input_ids,
            "images": images,
            "image_input_idx": image_input_idx,
            "loss_masks": all_loss_masks,
            "target_tokens": target_tokens,
        }
        if image_padding_mask:
            out["image_masks"] = np.concatenate(all_crop_masks, 0)
        if subsegments is not None:
            all_subsegments = np.concatenate(all_subsegments, 0)
            all_subsegments = np.pad(all_subsegments, [[1, 0]], constant_values=all_subsegments[0])
            if ends_with_eos:
                all_subsegments = all_subsegments[:-1]
            out["subsegment_ids"] = all_subsegments
            position_ids = np.zeros_like(all_subsegments)
            for subsegment_id in np.unique(all_subsegments):
                segment_position_ids = np.cumsum(all_subsegments >= subsegment_id) - 1
                position_ids = np.where(all_subsegments == subsegment_id, segment_position_ids, position_ids)
            out["position_ids"] = position_ids
        else:
            out["position_ids"] = np.arange(len(input_ids), dtype=np.int64)
        return out


@dataclasses.dataclass
class Preprocessor:
    formater: "DataFormatter"  # Use string annotation to avoid circular import
    mm_preprocessor: MultiModalPreprocessor
    for_inference: bool = False
    is_training: bool = False
    shuffle_messages: bool = False
    include_image: bool = False
    require_image_features: bool = False

    def __call__(self, example, rng=np.random):
        # Import locally to avoid circular import with molmo.data.__init__
        from molmo.data.data_formatter import DataFormatter
        
        example = dict(example)
        if "image" in example:
            try:
                image = load_image(example["image"])
            except Exception as e:
                raise ValueError(f"Could not load image: {example['image']}")
            else:
                example["image"] = image
        else:
            image = None

        messages, formatter_metadata = self.formater(example, self.is_training, self.for_inference, rng)
        if self.shuffle_messages and isinstance(messages[0], list):
            # If there are multiple conversations for this example, shuffle their order
            # This might matter if we truncate the tokens to a max sequence length
            rng.shuffle(messages)
        batch = self.mm_preprocessor(
            image,
            messages,
            weight=example.get("weight"),
            rng=rng,
            is_training=self.is_training,
            require_image_features=self.require_image_features
        )
        if formatter_metadata is None:
            formatter_metadata = {}
        if self.include_image and image is not None:
            formatter_metadata["image"] = image
        if image is not None:
            h, w = image.shape[:2]
            formatter_metadata["image_size"] = (w, h)
        if "metadata" in example or formatter_metadata:
            metadata = example.get("metadata", {})
            if formatter_metadata:
                metadata.update(formatter_metadata)
            batch["metadata"] = metadata
        
        # Preserve important fields from example that might not be in metadata.
        # This is needed for profiling / evaluation so that we can later record
        # rich sample-level information (e.g., image id, question text, options)
        # into the JSON results.
        if "metadata" not in batch:
            batch["metadata"] = {}
        
        # 1) Options (e.g., MMMU, ScienceQA-style multiple choice)
        if "options" in example and "options" not in batch["metadata"]:
            batch["metadata"]["options"] = example["options"]
        
        # 2) Question text (so profiling results can include it)
        if "question" in example and "question" not in batch["metadata"]:
            batch["metadata"]["question"] = example["question"]
        
        return batch

    @property
    def tokenizer(self):
        return self.mm_preprocessor.tokenizer


# ============================================================================
# Helper functions for MolmoImageProcessor (HuggingFace compatibility layer)
# ============================================================================

def pad_to_bounding_box(
    image, offset_height, offset_width, target_height,
    target_width, value=0
):
    """Pad image to bounding box (used by MolmoImageProcessor)"""
    height, width = image.shape[:2]
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height
    return np.pad(image, [
        [offset_height, after_padding_height],
        [offset_width, after_padding_width],
        [0, 0]
    ], constant_values=value)


def normalize_image(image, offset, scale):
    """Normalize image (used by MolmoImageProcessor)"""
    image -= np.array(offset, dtype=np.float32)[None, None, :]
    image /= np.array(scale, dtype=np.float32)[None, None, :]
    return image


def resize_and_pad_with_normalize(
    image,
    desired_output_size,
    resize_method="torch-bilinear",
    pad_value=0,
    normalize=True,
    image_mean=OPENAI_CLIP_MEAN,
    image_std=OPENAI_CLIP_STD,
):
    """
    Resize and pad with normalization (for MolmoImageProcessor compatibility).
    This wraps resize_and_pad to support normalize parameter.
    """
    # Call the shared resize_and_pad (without normalize)
    image, image_mask = resize_and_pad(
        image=image,
        desired_output_size=desired_output_size,
        is_training=False,
        resize_method=resize_method,
        pad_value=pad_value,
        rng=np.random
    )
    
    # Apply normalization if requested
    if normalize:
        image = normalize_image(image, offset=image_mean, scale=image_std)
    
    return image, image_mask


# ============================================================================
# HuggingFace-compatible image processor (merged from image_preprocessing_molmo.py)
# ============================================================================

class MolmoImagesKwargs(ImagesKwargs, total=False):
    """Type hints for MolmoImageProcessor kwargs"""
    max_crops: Optional[int]
    overlap_margins: Optional[List[int]]
    base_image_input_size: Optional[List[int]]
    image_token_length_w: Optional[int]
    image_token_length_h: Optional[int]
    image_patch_size: Optional[int]
    image_padding_mask: Optional[bool]


class MolmoImageProcessor(BaseImageProcessor):
    """Preprocess images and multi-model inputs (HuggingFace-compatible)"""

    def __init__(
        self,
        max_crops: int = 12,
        overlap_margins: List[int] = (4, 4),
        base_image_input_size: List[int] = (336, 336),
        image_token_length_w: int = 12,
        image_token_length_h: int = 12,
        image_patch_size: int = 14,
        image_padding_mask: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_crops = max_crops
        self.overlap_margins = overlap_margins
        self.base_image_input_size = base_image_input_size
        self.image_token_length_w = image_token_length_w
        self.image_token_length_h = image_token_length_h
        self.image_patch_size = image_patch_size
        self.image_padding_mask = image_padding_mask
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

    def image_to_patches_and_tokens(
        self,
        image: ImageInput,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        max_crops: Optional[int] = None,
        overlap_margins: Optional[List[int]] = None,
        base_image_input_size: Optional[Union[int, List[int]]] = None,
        image_token_length_w: Optional[int] = None,
        image_token_length_h: Optional[int] = None,
        image_patch_size: Optional[int] = None,
    ):
        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        tokens_per_image = image_token_length_w * image_token_length_h
        image_base_patch_w = base_image_input_size[1] // base_image_input_d
        image_base_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        # Discard this many patches from the (left/top, right/bottom) of crops
        left_margin, right_margin = overlap_margins
        assert left_margin % 2 == 0  # Required for compatibility with 2x2 pooling
        total_margin_pixels = base_image_input_d*(right_margin + left_margin)  # pixels removed per dim
        crop_patches = base_image_input_size[0] // base_image_input_d  # patches per crop dim
        crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
        crop_window_size = crop_window_patches * base_image_input_d
        # Use select_tiling (now in same file)
        # Note: MolmoImageProcessor doesn't support tier-based selection, so we pass None
        tiling = select_tiling(
            original_image_h - total_margin_pixels,
            original_image_w - total_margin_pixels,
            crop_window_size,
            max_crops,  # max_num_crops parameter
            tier=None  # MolmoImageProcessor uses adaptive selection
        )
        # Use resize_and_pad_with_normalize (now in same file)
        src, img_mask = resize_and_pad_with_normalize(
            image,
            [tiling[0]*crop_window_size+total_margin_pixels, tiling[1]*crop_window_size+total_margin_pixels],
            normalize=True  # MolmoImageProcessor always normalizes
        )

        # Now we have to split the image into crops, while keeping track of how each patch in the
        # each crop should be ordered in the global image, this require a lot of tricky booking
        n_crops = tiling[0] * tiling[1]
        patches_arr = []
        mask_arr = []
        patch_ordering_arr = []

        # We assume 2x2 pooling, but can allow padding the right/bottom with extra
        # patches if the number of patches per side is not even
        assert (crop_patches+1)//2 == image_token_length_h
        assert (crop_patches+1)//2 == image_token_length_w
        on = 0
        on_patch = 0
        for i in range(tiling[0]):
            y0 = i*crop_window_size
            if i == 0:
                crop_y0 = 0
            else:
                crop_y0 = left_margin // 2

            crop_h = image_base_patch_h - (right_margin + left_margin)
            if i == 0:
                crop_h += left_margin
            if i == (tiling[0]-1):
                crop_h += right_margin
            for j in range(tiling[1]):
                x0 = j*crop_window_size
                if j == 0:
                    crop_x0 = 0
                else:
                    crop_x0 = left_margin // 2

                crop_w = image_base_patch_w - (right_margin + left_margin)
                if j == 0:
                    crop_w += left_margin
                if j == (tiling[1]-1):
                    crop_w += right_margin

                pooled_w = (crop_w + 1) // 2
                pooled_h = (crop_h + 1) // 2
                patch_ordering_arr.append(
                    pad_to_bounding_box(
                        np.reshape(np.arange(on, on+pooled_h*pooled_w, dtype=np.int32), (pooled_h, pooled_w, 1)),
                        crop_y0, crop_x0, image_token_length_h, image_token_length_w, value=-1
                    )[:, :, 0]
                )
                patches_arr.append(src[y0:y0+crop_size, x0:x0+crop_size])
                mask_arr.append(img_mask[y0:y0+crop_size, x0:x0+crop_size])

                on += pooled_h*pooled_w
                on_patch += 1
        patches = np.stack(patches_arr)
        patch_ordering = np.stack(patch_ordering_arr)
        img_mask = np.stack(mask_arr)

        # Switch to [n_crops, n_patches, pixels_per_patch] format
        image_layout_impatch_w, image_layout_impatch_h = tiling[0], tiling[1]
        patches = einops.rearrange(
            patches, 'p (h dh) (w dw) c -> p (h w) (dh dw c)',
            dh=base_image_input_d,
            dw=base_image_input_d,
            h=image_base_patch_h,
            w=image_base_patch_w
        )
        img_mask = einops.rearrange(
            img_mask, 'p (h dh) (w dw) -> p (h w) (dh dw)',
            dh=base_image_input_d,
            dw=base_image_input_d,
            h=image_base_patch_h,
            w=image_base_patch_w
        )

        img_mask = img_mask.astype(np.float32).mean(axis=-1)
        patch_ordering = np.reshape(patch_ordering, [-1])
        valid = patch_ordering >= 0

        # Transpose order, to get left-to-right order instead of crop-by-crop order
        patch_ordering_rh = np.reshape(
            patch_ordering,
            [tiling[0], tiling[1], image_token_length_h, image_token_length_w]
        )
        patch_ordering_rh = np.transpose(patch_ordering_rh, [0, 2, 1, 3])
        patch_ordering_rh = np.reshape(patch_ordering_rh, [-1])

        # The transpose will screw up which patches are masked, project the
        # new order into sparse structure of `patch_ordering` to fix this
        patch_ordering[valid] = patch_ordering_rh[patch_ordering_rh >= 0]

        # Now build the output tokens
        h = tiling[0] * crop_window_patches + (right_margin+left_margin)
        w = tiling[1] * crop_window_patches + (right_margin+left_margin)
        per_row = np.full(
            ((w+1)//2,),
            image_patch_token_id,
        )
        per_row = np.concatenate([per_row, [image_col_token_id]], 0)

        joint = np.tile(per_row, [(h+1)//2])
        joint = [
            [image_start_token_id],
            joint,
            [image_end_token_id]
        ]

        # Finally do the same for the global image
        resized, _ = resize_and_pad_with_normalize(image, base_image_input_size, normalize=True)
        resized = einops.rearrange(
            resized, '(h dh) (w dw) c -> (h w) (dh dw c)',
            dh=base_image_input_d,
            dw=base_image_input_d,
            h=image_base_patch_h,
            w=image_base_patch_w
        )
        patches = np.concatenate([np.expand_dims(resized, 0), patches], 0)

        # Global image goes first, so the order of patches in previous crops gets increased
        patch_ordering = np.where(
            patch_ordering >= 0,
            patch_ordering + tokens_per_image,
            -1
        )
        patch_ordering = np.concatenate([np.arange(0, tokens_per_image), patch_ordering], 0)
        per_row = np.full(
            (image_token_length_w,),
            image_patch_token_id,
        )
        per_row = np.concatenate([per_row, [image_col_token_id]], 0)
        extra_tokens = np.tile(per_row, [image_token_length_h])
        joint = [
                    [image_start_token_id],
                    extra_tokens,
                    [image_end_token_id],
                ] + joint

        joint = np.concatenate(joint, 0)
        img_mask = np.pad(img_mask, [[0, 1], [0, 0]], constant_values=-1)
        return patches, joint, patch_ordering, img_mask

    def build_image_input_idx(
        self,
        image_tokens: np.ndarray,
        patch_order: np.ndarray,
        image_patch_token_id: int,
        no_image: Optional[bool] = None,
        image_token_length_w: Optional[int] = None,
        image_token_length_h: Optional[int] = None,
    ):
        """Converts `patch_order` into a mapping of token_id -> patch_id"""

        tokens_per_image = image_token_length_w * image_token_length_h
        if no_image is not None and no_image:
            return np.zeros((0, tokens_per_image), np.int32)

        # Indices to insert the patches
        image_input_idx = image_tokens == image_patch_token_id
        image_input_idx = np.nonzero(image_input_idx)[0].astype(np.int32)

        if patch_order is not None:
            n_tokens = image_input_idx.shape[0]
            patch_order = np.reshape(patch_order, [-1])
            n_patches = patch_order.shape[0]

            valid = patch_order >= 0
            n_valid_patches = valid.sum()
            assert len(image_input_idx) == n_valid_patches

            sorted_patch_ixs = np.zeros([n_tokens], np.int32)
            sorted_patch_ixs[patch_order[valid]] = np.arange(n_valid_patches, dtype=np.int32)

            # Project the inverted mapping into same sparse structure
            sorted_patch_ixs_ex = np.full(np.shape(patch_order), -1)
            sorted_patch_ixs_ex[valid] = sorted_patch_ixs

            # Do the gather and then re-masked outputs that were masked in `sorted_patch_ixs`
            valid = (sorted_patch_ixs_ex >= 0).astype(np.int32)
            image_input_idx = image_input_idx[sorted_patch_ixs_ex*valid]
            image_input_idx = image_input_idx*valid - 100*(1 - valid)
            image_input_idx = np.reshape(image_input_idx, [-1, tokens_per_image])
        return image_input_idx

    def preprocess(
        self,
        image: np.ndarray,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        max_crops: Optional[int] = None,
        overlap_margins: Optional[List[int]] = None,
        base_image_input_size: Optional[Union[int, List[int]]] = None,
        image_token_length_w: Optional[int] = None,
        image_token_length_h: Optional[int] = None,
        image_patch_size: Optional[int] = None,
        **kwargs,
    ):
        """Preprocesses an image

        Returns:
            crops: (n_crops, n_patches, patch_dim) individual crops, `n_crops` might
                   change between images but the other dimension are fixed
            tokens: (n_tokens,) int32 tokens, pad tokens indicate where to insert the
                                patch features, might include other special tokens as well
            image_idx: (n_crops, n_patches) index in `tokens` to put the patch features from the
                       crops after pooling, negative values indicates patches features to exclude
            padding_mask: (n_crops, n_patches) what percent of each crop is padding, can be None
                          if the image mask is not being used.
        """

        max_crops = max_crops or self.max_crops
        overlap_margins = overlap_margins or self.overlap_margins
        base_image_input_size = base_image_input_size or self.base_image_input_size
        image_token_length_w = image_token_length_w or self.image_token_length_w
        image_token_length_h = image_token_length_h or self.image_token_length_h
        image_patch_size = image_patch_size or self.image_patch_size

        crops, image_tokens, patch_ordering, img_mask = self.image_to_patches_and_tokens(
            image,
            image_patch_token_id,
            image_col_token_id,
            image_start_token_id,
            image_end_token_id,
            max_crops,
            overlap_margins,
            base_image_input_size,
            image_token_length_w,
            image_token_length_h,
            image_patch_size,
        )
        patch_idx = self.build_image_input_idx(
            image_tokens,
            patch_ordering,
            image_patch_token_id,
            image_token_length_w=image_token_length_w,
            image_token_length_h=image_token_length_h,
        )
        return crops, image_tokens, patch_idx, img_mask

    def multimodal_preprocess(
        self,
        images: np.ndarray,
        tokens: List[int],
        image_idx: np.ndarray,
        sequence_length: int,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        **kwargs,
    ):
        """Merge images and text tokens into multi-modal features for the model

        :param images: images to use as input
        :param tokens: input text tokens
        :param image_idx: where to insert the images into `tokens`
        :params image_patch_token_id: id to use of tokens that will contain image features
        :params image_col_token_id: token id for image column special tokens
        :params image_start_token_id: token id for image start special tokens
        :params image_end_token_id: token id for image end special tokens
        :params kwargs: override preprocessor default args
        """
        max_total_crops = kwargs.get("max_crops") or self.max_crops
        image_token_length_w = kwargs.get("image_token_length_w") or self.image_token_length_w
        image_token_length_h = kwargs.get("image_token_length_h") or self.image_token_length_h
        image_patch_size = kwargs.get("image_patch_size") or self.image_patch_size
        base_image_input_size = kwargs.get("base_image_input_size") or self.base_image_input_size
        image_num_patch = (
            base_image_input_size[0] // image_patch_size,
            base_image_input_size[1] // image_patch_size,
        )
        image_padding_mask = kwargs.get("image_padding_mask") or self.image_padding_mask

        tokens_per_image = image_token_length_w * image_token_length_h
        n_pixels = image_patch_size * image_patch_size * 3
        n_patches = image_num_patch[0] * image_num_patch[1]

        if images is None:
            return {
                "input_ids": tokens,
            }
        else:
            n = len(images)
            all_crops = []
            all_image_idx = []
            out_tokens = []
            all_crop_masks = []

            for ix in range(n):
                token_ix = image_idx[ix]
                crops, image_tokens, patch_idx, img_mask = self.preprocess(
                    images[ix],
                    image_patch_token_id,
                    image_col_token_id,
                    image_start_token_id,
                    image_end_token_id,
                    **kwargs,
                )

                if token_ix == -1:  # -1 is an image inserted at the very start
                    start = 0
                    token_ix = 0
                    end = 0
                else:
                    start = 0 if ix == 0 else image_idx[ix-1] + 1
                    end = token_ix + 1

                all_image_idx.append(patch_idx + token_ix)
                all_crops.append(crops)
                out_tokens.append(tokens[start:token_ix])
                out_tokens.append(image_tokens)
                if ix == (n - 1):
                    out_tokens.append(tokens[end:])
                if image_padding_mask:
                    all_crop_masks.append(img_mask)

            input_ids = np.concatenate(out_tokens, 0)
            images = np.concatenate(all_crops, 0)
            image_input_idx = np.concatenate(all_image_idx, 0)
            if image_padding_mask:
                image_masks = np.concatenate(all_crop_masks, 0)
            else:
                image_masks = None

        out = {
            "input_ids": input_ids,
            "images": images,
            "image_input_idx": image_input_idx
        }
        if image_masks is not None:
            out["image_masks"] = image_masks
        return out


MolmoImageProcessor.register_for_auto_class()