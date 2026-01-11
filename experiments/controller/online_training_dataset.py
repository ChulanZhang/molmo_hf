"""
Dataset for online training of joint controller.
Loads actual dataset samples (images + prompts) for online execution.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import random
import numpy as np
from PIL import Image as PILImage

log = logging.getLogger(__name__)


class OnlineTrainingDataset(Dataset):
    """
    Dataset for online training (Joint GRPO).
    
    Loads actual dataset samples (images + prompts) for online execution.
    Each sample is paired with a randomly sampled latency budget from [170ms, 380ms].
    
    Features:
    - Eager loading: Loads all samples into memory for faster access
    - Random budget sampling: Each sample gets a random latency budget (uniform)
    - Supports multiple datasets (text_vqa, coco_2014_vqa, okvqa, etc.)
    - Returns: image, prompt, metadata, sample_id, latency_budget
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "validation",
        model_path: str = None,
        device: str = "cuda",
        num_samples: Optional[int] = None,
        latency_budget_min: float = 170.0,
        latency_budget_max: float = 380.0,
        seed: int = 42,
    ):
        """
        Args:
            dataset_name: Dataset name (text_vqa, coco_2014_vqa, etc.)
            split: Dataset split (validation, train, etc.)
            model_path: Path to model (for processor)
            device: Device to use
            num_samples: Number of samples to load (None = all)
            latency_budget_min: Minimum latency budget (ms)
            latency_budget_max: Maximum latency budget (ms)
            seed: Random seed for budget sampling
        """
        self.dataset_name = dataset_name
        self.split = split
        self.device = device
        self.num_samples = num_samples
        self.latency_budget_min = latency_budget_min
        self.latency_budget_max = latency_budget_max
        
        # Store model_path for later use (processor will be passed from outside)
        # Don't load model here to avoid repeated loading
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Load dataset
        log.info(f"Loading dataset {dataset_name}/{split}...")
        self.samples = self._load_dataset(dataset_name, split, num_samples)
        
        # Set seed for budget sampling
        random.seed(seed)
        np.random.seed(seed)
        
        log.info(f"Loaded {len(self.samples)} samples from {dataset_name}/{split}")
    
    def _load_dataset(
        self,
        dataset_name: str,
        split: str,
        num_samples: Optional[int],
    ) -> List[Dict]:
        """Load dataset samples."""
        from molmo.data import get_dataset_by_name
        
        try:
            dataset = get_dataset_by_name(dataset_name, split=split)
        except Exception as e:
            log.error(f"Error loading dataset {dataset_name}: {e}")
            return []
        
        samples = []
        total_items = len(dataset) if hasattr(dataset, '__len__') else None
        
        from tqdm import tqdm
        iterator = tqdm(enumerate(dataset), total=num_samples or total_items, desc=f"Loading {dataset_name}/{split}")
        
        for i, item in iterator:
            if num_samples and i >= num_samples:
                break
            
            # Extract image and prompt
            image = item.get('image')
            prompt = item.get('question', item.get('prompt', ''))
            metadata = item.get('metadata', {})
            
            # Extract answers - they may be at top level (not in metadata)
            # For VQA datasets, answers are typically at top level
            answers = item.get('answers', None)
            
            # Also check in metadata if not found at top level
            if answers is None and isinstance(metadata, dict):
                answers = metadata.get('answers', None)
            
            if image is None or not prompt:
                continue
            
            # Merge answers into metadata for easier access
            # This ensures answers are available when computing accuracy
            if answers is not None:
                if isinstance(metadata, dict):
                    metadata['answers'] = answers
                else:
                    # If metadata is not a dict, create one
                    metadata = {'answers': answers}
            
            # DEBUG: Log first few samples to verify answers are being loaded
            if i < 3:
                import logging
                log = logging.getLogger(__name__)
                log.info(f"[OnlineTrainingDataset] Sample {i}: answers={answers[:3] if isinstance(answers, list) and len(answers) > 3 else answers}, "
                        f"metadata_keys={list(metadata.keys()) if isinstance(metadata, dict) else 'N/A'}")
            
            # Also store dataset name/style in metadata for metric selection
            style = item.get('style', None)
            if style and isinstance(metadata, dict):
                metadata['dataset_name'] = style
                metadata['style'] = style
            
            samples.append({
                'image': image,
                'prompt': prompt,
                'metadata': metadata,
                'sample_id': metadata.get('question_id', metadata.get('image_id', metadata.get('example_id', i))),
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Sample latency budget from range
        latency_budget = random.uniform(self.latency_budget_min, self.latency_budget_max)
        
        return {
            'image': sample['image'],
            'prompt': sample['prompt'],
            'metadata': sample['metadata'],
            'sample_id': sample['sample_id'],
            'latency_budget': latency_budget,
        }


def collate_online_training_batch(
    batch: List[Dict],
    processor,  # Processor passed from outside
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Collate batch for online training.
    
    Args:
        batch: List of samples
        processor: MolmoProcessor instance
        device: Device to use
    
    Returns:
        Batched data ready for training
    """
    images = [item['image'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    metadatas = [item['metadata'] for item in batch]
    sample_ids = [item['sample_id'] for item in batch]
    latency_budgets = torch.tensor([item['latency_budget'] for item in batch], dtype=torch.float32)
    
    # Process each sample individually and then batch them
    # MolmoProcessor.process() expects single text/image, not lists
    processed_batch = []
    for prompt, image in zip(prompts, images):
        # Convert image to PIL Image if it's a string path
        # processor.process() expects PIL Image or numpy array, not string paths
        if isinstance(image, str):
            # String path - load as PIL Image
            image = PILImage.open(image).convert("RGB")
        elif isinstance(image, PILImage.Image):
            # Already a PIL Image - ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            # Already a numpy array - use as-is
            pass
        else:
            # Unknown type - try to load as image path
            try:
                image = PILImage.open(image).convert("RGB")
            except Exception as e:
                log.warning(f"Could not load image (type: {type(image)}): {e}")
                # Skip this sample or use a placeholder
                continue
        
        processed = processor.process(
            text=prompt,  # Single string, not list
            images=image,  # Single image (PIL Image or numpy array), not list
            text_kwargs={
                "message_format": "role",  # Add "User:" and "Assistant:" prefixes
                "always_start_with_space": True,
                "sequence_length": 1536,  # Max sequence length
                "padding": False,
            },
            images_kwargs={
                "max_crops": 12,  # Default max crops
                "overlap_margins": [4, 4],
                "base_image_input_size": [336, 336],
                "image_token_length_w": 12,
                "image_token_length_h": 12,
                "image_patch_size": 14,
                "image_padding_mask": True,
            },
        )
        processed_batch.append(processed)
    
    # Use MMCollator to handle variable-sized images and sequences
    from molmo.data.collator import MMCollator
    
    # Convert processed items to format expected by MMCollator
    # MMCollator expects numpy arrays with keys: input_tokens, images, image_masks, image_input_idx
    collator_batch = []
    for p in processed_batch:
        item = {
            'input_tokens': p['input_ids'].numpy() if isinstance(p['input_ids'], torch.Tensor) else p['input_ids'],
        }
        if p.get('images') is not None:
            item['images'] = p['images'].numpy() if isinstance(p['images'], torch.Tensor) else p['images']
        if p.get('image_masks') is not None:
            item['image_masks'] = p['image_masks'].numpy() if isinstance(p['image_masks'], torch.Tensor) else p['image_masks']
        if p.get('image_input_idx') is not None:
            item['image_input_idx'] = p['image_input_idx'].numpy() if isinstance(p['image_input_idx'], torch.Tensor) else p['image_input_idx']
        collator_batch.append(item)
    
    # Use MMCollator to handle padding
    collator = MMCollator(
        max_sequence_length=1536,
        max_crops=12,  # Max crops per image
        pad=None,  # Pad to max in batch
        include_metadata=False,
    )
    
    processed = collator(collator_batch)
    
    # MMCollator returns CPU tensors, keep them on CPU for pin_memory
    # DataLoader will handle moving to device
    result = {
        'input_ids': processed['input_ids'],  # Keep on CPU for pin_memory
        'images': processed.get('images'),  # Keep on CPU
        'image_masks': processed.get('image_masks'),  # Keep on CPU
        'image_input_idx': processed.get('image_input_idx'),  # Keep on CPU
        'prompts': prompts,
        'metadata': metadatas,
        'sample_id': sample_ids,
        'latency_budget': latency_budgets,  # Keep on CPU for pin_memory
    }
    
    return result

