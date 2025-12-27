"""
Data preparation: extract training data from exp5 and exp6 results.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)


class ExpDataLoader:
    """Load data from exp5 and exp6 result files."""
    
    def __init__(self, exp5_dir: str, exp6_dir: str):
        self.exp5_dir = Path(exp5_dir)
        self.exp6_dir = Path(exp6_dir)
    
    def load_exp5_results(self, dataset_name: str = "text_vqa") -> List[Dict]:
        """Load exp5 accuracy results."""
        results = []
        
        # Find all exp5 result files
        pattern = f"exp5_accuracy_results*.json"
        if dataset_name != "coco_2014_vqa":
            dataset_suffix = dataset_name.replace("_", "-")
            exp5_dataset_dir = self.exp5_dir / f"exp5_accuracy_{dataset_suffix}"
        else:
            exp5_dataset_dir = self.exp5_dir
        
        json_files = list(exp5_dataset_dir.glob(pattern))
        
        if not json_files:
            log.warning(f"No exp5 results found in {exp5_dataset_dir}")
            return results
        
        log.info(f"Loading {len(json_files)} exp5 result files from {exp5_dataset_dir}")
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Extract summary and all_samples
                if 'summary' in data:
                    for summary in data['summary']:
                        results.append({
                            'max_crops': summary.get('max_crops'),
                            'top_k': summary.get('top_k'),
                            'num_active_blocks': summary.get('num_active_blocks'),
                            'accuracy': summary.get('accuracy'),
                            'num_samples': summary.get('num_samples'),
                            'config': summary,
                        })
                
                # Per-sample data
                if 'all_samples' in data:
                    for sample in data['all_samples']:
                        # Parse config from filename
                        config = self._parse_config_from_filename(json_file.name)
                        if config:
                            results.append({
                                'sample_id': sample.get('sample_id'),
                                'max_crops': config['max_crops'],
                                'top_k': config['top_k'],
                                'num_active_blocks': config['num_active_blocks'],
                                'accuracy': sample.get('score', 0.0),
                                'pred': sample.get('pred'),
                                'answers': sample.get('answers'),
                            })
        
        log.info(f"Loaded {len(results)} exp5 records")
        return results
    
    def load_exp6_results(self, dataset_name: str = "text_vqa") -> List[Dict]:
        """Load exp6 latency + accuracy results."""
        results = []
        
        # Find all exp6 result files
        pattern = f"exp6_latency*.json"
        if dataset_name != "coco_2014_vqa":
            dataset_suffix = dataset_name.replace("_", "-")
            exp6_dataset_dir = self.exp6_dir / f"exp6_latency_{dataset_suffix}"
        else:
            exp6_dataset_dir = self.exp6_dir
        
        json_files = list(exp6_dataset_dir.glob(pattern))
        
        if not json_files:
            log.warning(f"No exp6 results found in {exp6_dataset_dir}")
            return results
        
        log.info(f"Loading {len(json_files)} exp6 result files from {exp6_dataset_dir}")
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Extract summary
                if 'summary' in data:
                    for summary in data['summary']:
                        results.append({
                            'max_crops': summary.get('max_crops'),
                            'top_k': summary.get('top_k'),
                            'num_active_blocks': summary.get('num_active_blocks'),
                            'accuracy': summary.get('accuracy'),
                            'latency_mean': summary.get('latency_mean'),
                            'latency_std': summary.get('latency_std'),
                            'latency_p50': summary.get('latency_p50'),
                            'latency_p95': summary.get('latency_p95'),
                            'latency_p99': summary.get('latency_p99'),
                            'num_samples': summary.get('num_samples'),
                            'config': summary,
                        })
                
                # Per-sample data
                if 'all_samples' in data:
                    for sample in data['all_samples']:
                        config = self._parse_config_from_filename(json_file.name)
                        if config:
                            results.append({
                                'sample_id': sample.get('sample_id'),
                                'max_crops': config['max_crops'],
                                'top_k': config['top_k'],
                                'num_active_blocks': config['num_active_blocks'],
                                'accuracy': sample.get('score', 0.0),
                                'latency': sample.get('latency_total', 0.0),
                                'latency_prefill': sample.get('latency_prefill', 0.0),
                                'latency_decode': sample.get('latency_decode', 0.0),
                                'pred': sample.get('pred'),
                                'answers': sample.get('answers'),
                            })
        
        log.info(f"Loaded {len(results)} exp6 records")
        return results
    
    def _parse_config_from_filename(self, filename: str) -> Optional[Dict]:
        """Parse config from filename.

        Format example: exp5_accuracy_results_crops12_topk32_blocks12_rank0.json
        """
        try:
            parts = filename.replace('.json', '').split('_')
            config = {}
            
            for part in parts:
                if part.startswith('crops'):
                    config['max_crops'] = int(part[5:])
                elif part.startswith('topk'):
                    config['top_k'] = int(part[4:])
                elif part.startswith('blocks'):
                    config['num_active_blocks'] = int(part[6:])
            
            if len(config) == 3:
                return config
        except:
            pass
        
        return None
    
    def merge_exp5_exp6(self, exp5_results: List[Dict], exp6_results: List[Dict]) -> List[Dict]:
        """Merge exp5 and exp6 data.

        Prefer exp6 records when available (has latency); otherwise use exp5.
        """
        merged = []
        
        # Build lookup dict for exp6 by (sample_id, config)
        exp6_dict = {}
        for r in exp6_results:
            if 'sample_id' in r:
                key = (r.get('sample_id'), r.get('max_crops'), r.get('top_k'), r.get('num_active_blocks'))
                exp6_dict[key] = r
        
        # Merge
        for r5 in exp5_results:
            if 'sample_id' in r5:
                key = (r5.get('sample_id'), r5.get('max_crops'), r5.get('top_k'), r5.get('num_active_blocks'))
                if key in exp6_dict:
                    # Use exp6 (has latency)
                    merged.append(exp6_dict[key])
                else:
                    # Use exp5; latency None
                    r5_copy = r5.copy()
                    r5_copy['latency'] = None
                    merged.append(r5_copy)
            else:
                # summary rows without sample_id
                merged.append(r5)
        
        # Add exp6-only entries
        for r6 in exp6_results:
            if 'sample_id' in r6:
                key = (r6.get('sample_id'), r6.get('max_crops'), r6.get('top_k'), r6.get('num_active_blocks'))
                if key not in exp6_dict:
                    merged.append(r6)
        
        log.info(f"Merged {len(merged)} records from exp5 and exp6")
        return merged


class TrainingDataBuilder:
    """Build training data."""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def extract_features(
        self,
        dataset,
        sample_ids: List[int],
        batch_size: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract image and language features from the dataset.

        Returns:
        {
            'image_features': (N, D_img),
            'language_features': (N, D_lang),
            'sample_ids': (N,),
        }
        """
        from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
        from molmo.data.data_formatter import DataFormatter
        from molmo.data.collator import MMCollator
        from molmo.data.dataset import DeterministicDataset
        
        # Build dataloader
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
        dataloader = torch.utils.data.DataLoader(
            det_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=MMCollator(
                max_sequence_length=1536,
                include_metadata=True,
                pad=True,
                max_crops=self.model.config.max_crops
            ),
            num_workers=4,
            pin_memory=True,
        )
        
        # Extract features
        image_features_list = []
        language_features_list = []
        sample_ids_list = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Image feature
                if batch.get('images') is not None:
                    image_features, cls_embed = self.model.model.vision_backbone(
                        batch['images'], 
                        batch.get('image_masks')
                    )
                    # Use CLS token or mean pooling
                    if cls_embed is not None:
                        image_feat = cls_embed.mean(dim=1)  # (B, D)
                    else:
                        image_feat = image_features.mean(dim=(1, 2))  # (B, D)
                else:
                    image_feat = torch.zeros(batch['input_ids'].shape[0], 768, device=self.device)
                
                # Language feature
                input_embeds = self.model.model.transformer.wte(batch['input_ids'])
                lang_feat = input_embeds.mean(dim=1)  # (B, D)
                
                image_features_list.append(image_feat.cpu())
                language_features_list.append(lang_feat.cpu())
                
                # Collect sample_ids
                if 'metadata' in batch:
                    batch_sample_ids = [m.get('sample_id', batch_idx * batch_size + i) 
                                      for i, m in enumerate(batch['metadata'])]
                    sample_ids_list.extend(batch_sample_ids)
                else:
                    batch_sample_ids = list(range(batch_idx * batch_size, 
                                                 batch_idx * batch_size + batch['input_ids'].shape[0]))
                    sample_ids_list.extend(batch_sample_ids)
        
        image_features = torch.cat(image_features_list, dim=0)
        language_features = torch.cat(language_features_list, dim=0)
        sample_ids = torch.tensor(sample_ids_list, dtype=torch.long)
        
        return {
            'image_features': image_features,
            'language_features': language_features,
            'sample_ids': sample_ids,
        }
    
    def build_training_data(
        self,
        merged_results: List[Dict],
        features: Dict[str, torch.Tensor],
        latency_budgets: List[float] = None,
    ) -> List[Dict]:
        """
        Build training data.

        Args:
            merged_results: merged exp5/exp6 results
            features: extracted features
            latency_budgets: latency budgets (if None, use multiples of actual latency)

        Returns:
            list of training records
        """
        training_data = []
        
        # Map sample_id to features
        sample_id_to_idx = {int(sid): idx for idx, sid in enumerate(features['sample_ids'])}
        
        # If no budgets provided, use multiples of actual latency
        if latency_budgets is None:
            latencies = [r.get('latency', 0.0) for r in merged_results if r.get('latency') is not None]
            if latencies:
                mean_latency = np.mean(latencies)
                latency_budgets = [mean_latency * 0.8, mean_latency * 1.0, mean_latency * 1.2]
            else:
                latency_budgets = [0.5, 1.0, 1.5]  # defaults
        
        for result in merged_results:
            sample_id = result.get('sample_id')
            if sample_id is None:
                continue
            
            # Get features
            if sample_id in sample_id_to_idx:
                idx = sample_id_to_idx[sample_id]
                image_feat = features['image_features'][idx]
                lang_feat = features['language_features'][idx]
            else:
                continue  # skip samples without features
            
            # Create one record per latency budget
            for budget in latency_budgets:
                training_data.append({
                    'sample_id': sample_id,
                    'image_feature': image_feat,
                    'language_feature': lang_feat,
                    'latency_budget': budget,
                    'config': {
                        'max_crops': result.get('max_crops'),
                        'top_k': result.get('top_k'),
                        'num_active_blocks': result.get('num_active_blocks'),
                    },
                    'accuracy': result.get('accuracy', 0.0),
                    'latency': result.get('latency', 0.0),
                    'pred': result.get('pred'),
                    'answers': result.get('answers'),
                })
        
        log.info(f"Built {len(training_data)} training samples")
        return training_data


def save_training_data(training_data: List[Dict], output_path: str):
    """Save training data to file."""
    # Convert tensor to numpy
    data_to_save = []
    for item in training_data:
        item_copy = item.copy()
        item_copy['image_feature'] = item['image_feature'].numpy().tolist()
        item_copy['language_feature'] = item['language_feature'].numpy().tolist()
        data_to_save.append(item_copy)
    
    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    log.info(f"Saved training data to {output_path}")


def load_training_data(input_path: str) -> List[Dict]:
    """Load training data from file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Convert numpy back to tensor
    training_data = []
    for item in data:
        item['image_feature'] = torch.tensor(item['image_feature'])
        item['language_feature'] = torch.tensor(item['language_feature'])
        training_data.append(item)
    
    log.info(f"Loaded {len(training_data)} training samples from {input_path}")
    return training_data

