"""
Load and process core experiment results for controller training.
Handles the JSON format from core_exp experiments.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict

log = logging.getLogger(__name__)


class CoreExpDataLoader:
    """
    Load data from core experiment results (acc_lat_profiling.py output).
    """
    
    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: Directory containing core experiment results
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory does not exist: {results_dir}")
    
    def load_dataset_results(
        self,
        dataset_name: str,
        pattern: str = "*.json",
    ) -> List[Dict]:
        """
        Load all result files for a dataset.
        
        Args:
            dataset_name: Name of dataset (e.g., "text_vqa", "coco_2014_vqa")
            pattern: File pattern to match
        
        Returns:
            List of result dictionaries
        """
        # Find dataset directory
        dataset_suffix = dataset_name.replace("_", "-")
        dataset_dir = self.results_dir / dataset_suffix
        
        if not dataset_dir.exists():
            log.warning(f"Dataset directory not found: {dataset_dir}")
            return []
        
        # Find all JSON files
        json_files = list(dataset_dir.glob(pattern))
        
        if not json_files:
            log.warning(f"No result files found in {dataset_dir} with pattern {pattern}")
            return []
        
        log.info(f"Loading {len(json_files)} result files from {dataset_dir}")
        
        all_results = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract per_sample_results
                if 'per_sample_results' in data:
                    for sample in data['per_sample_results']:
                        # Add config info from summary
                        config_info = data.get('summary', [{}])[0] if 'summary' in data else {}
                        
                        result = {
                            'sample_id': sample.get('sample_id'),
                            'tier': sample.get('tier', config_info.get('tier')),
                            'top_k': sample.get('top_k', config_info.get('top_k')),
                            'num_active_blocks': sample.get('num_active_blocks', config_info.get('num_active_blocks')),
                            'target_vision_tokens': sample.get('target_vision_tokens'),
                            'actual_vision_tokens': sample.get('actual_vision_tokens'),
                            'actual_text_tokens': sample.get('actual_text_tokens'),
                            'output_tokens': sample.get('output_tokens'),
                            'accuracy': sample.get('accuracy', 0.0),
                            'T_vision_total': sample.get('T_vision_total', 0.0),
                            'T_LLM_prefill': sample.get('T_LLM_prefill', 0.0),
                            'T_LLM_decode': sample.get('T_LLM_decode', 0.0),
                            'T_total': sample.get('T_total', 0.0),
                            'metadata': sample.get('metadata', {}),
                        }
                        
                        # Compute total latency if not present
                        if result['T_total'] == 0.0:
                            result['T_total'] = (
                                result['T_vision_total'] +
                                result['T_LLM_prefill'] +
                                result['T_LLM_decode']
                            )
                        
                        all_results.append(result)
                
            except Exception as e:
                log.error(f"Error loading {json_file}: {e}")
                continue
        
        log.info(f"Loaded {len(all_results)} sample results from {dataset_name}")
        return all_results
    
    def convert_tier_to_max_crops(self, tier: str) -> int:
        """
        Convert tier to max_crops value.
        
        Args:
            tier: "low", "medium", or "high"
        
        Returns:
            max_crops: Representative max_crops value
        """
        tier_map = {
            'low': 3,      # 1-3 crops, use 3 as representative
            'medium': 6,   # 4-8 crops, use 6 as representative
            'high': 12,   # 9-15 crops, use 12 as representative
        }
        return tier_map.get(tier.lower(), 6)
    
    def build_training_samples(
        self,
        results: List[Dict],
        latency_budgets: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Build training samples from results.
        
        Args:
            results: List of result dictionaries
            latency_budgets: Optional list of latency budgets to use.
                           If None, uses actual latencies as budgets.
        
        Returns:
            List of training samples
        """
        training_samples = []
        
        # Group by sample_id
        samples_by_id = defaultdict(list)
        for result in results:
            sample_id = result.get('sample_id')
            if sample_id is not None:
                samples_by_id[sample_id].append(result)
        
        log.info(f"Found {len(samples_by_id)} unique samples")
        
        # Build training samples
        for sample_id, sample_results in samples_by_id.items():
            # Get metadata from first result
            first_result = sample_results[0]
            metadata = first_result.get('metadata', {})
            
            # Use actual latency as budget if not provided
            if latency_budgets is None:
                # Use actual latencies from results
                budgets = [r['T_total'] for r in sample_results]
            else:
                # Use provided budgets
                budgets = latency_budgets
            
            # Create training samples for each configuration
            for result, budget in zip(sample_results, budgets):
                # Convert tier to max_crops
                tier = result.get('tier', 'medium')
                max_crops = self.convert_tier_to_max_crops(tier)
                
                training_sample = {
                    'sample_id': sample_id,
                    'tier': tier,
                    'max_crops': max_crops,
                    'top_k': result.get('top_k', 8),
                    'num_active_blocks': result.get('num_active_blocks', 16),
                    'accuracy': result.get('accuracy', 0.0),
                    'latency': result.get('T_total', 0.0),
                    'latency_budget': budget,
                    'vision_tokens': result.get('actual_vision_tokens', 0),
                    'text_tokens': result.get('actual_text_tokens', 0),
                    'output_tokens': result.get('output_tokens', 0),
                    # Preserve all latency components for analysis
                    'T_vision_total': result.get('T_vision_total', 0.0),
                    'T_LLM_prefill': result.get('T_LLM_prefill', 0.0),
                    'T_LLM_decode': result.get('T_LLM_decode', 0.0),
                    'T_decode_per_token': result.get('T_decode_per_token', 0.0),
                    'T_total': result.get('T_total', 0.0),
                    'metadata': metadata,
                }
                
                training_samples.append(training_sample)
        
        log.info(f"Built {len(training_samples)} training samples")
        return training_samples
    
    def load_multiple_datasets(
        self,
        dataset_names: List[str],
        latency_budgets: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Load results from multiple datasets.
        
        Args:
            dataset_names: List of dataset names
            latency_budgets: Optional latency budgets
        
        Returns:
            Combined list of training samples
        """
        all_samples = []
        
        for dataset_name in dataset_names:
            log.info(f"Loading results for {dataset_name}...")
            results = self.load_dataset_results(dataset_name)
            
            if not results:
                log.warning(f"No results found for {dataset_name}")
                continue
            
            samples = self.build_training_samples(results, latency_budgets)
            all_samples.extend(samples)
        
        log.info(f"Total training samples: {len(all_samples)}")
        return all_samples
    
    def get_statistics(self, samples: List[Dict]) -> Dict[str, Any]:
        """
        Get statistics about training samples.
        
        Args:
            samples: List of training samples
        
        Returns:
            Statistics dictionary
        """
        if not samples:
            return {}
        
        accuracies = [s['accuracy'] for s in samples]
        latencies = [s['latency'] for s in samples]
        budgets = [s['latency_budget'] for s in samples]
        
        stats = {
            'num_samples': len(samples),
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
            },
            'latency': {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
            },
            'budget': {
                'mean': np.mean(budgets),
                'std': np.std(budgets),
                'min': np.min(budgets),
                'max': np.max(budgets),
            },
            'config_distribution': {
                'tiers': defaultdict(int),
                'top_k': defaultdict(int),
                'num_active_blocks': defaultdict(int),
            },
        }
        
        # Count configurations
        for sample in samples:
            stats['config_distribution']['tiers'][sample['tier']] += 1
            stats['config_distribution']['top_k'][sample['top_k']] += 1
            stats['config_distribution']['num_active_blocks'][sample['num_active_blocks']] += 1
        
        return stats

