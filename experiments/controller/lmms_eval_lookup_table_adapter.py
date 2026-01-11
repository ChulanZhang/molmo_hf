"""
LMms-Eval Adapter for Lookup Table Baseline Controller.

Wraps the lookup table baseline controller to be compatible with lmms-eval framework.
Based on AdaLLaVA's implementation approach.
"""

import logging
import torch
from typing import Dict, List, Optional, Any, Union
from PIL import Image
import numpy as np

log = logging.getLogger(__name__)


class LookupTableLMMSEvalAdapter:
    """
    Adapter that wraps Lookup Table Baseline Controller to work with lmms-eval.
    
    This adapter implements the interface expected by lmms-eval's model API,
    allowing the lookup table baseline controller to be evaluated on standard benchmarks.
    """
    
    def __init__(
        self,
        engine,
        latency_budget: float = 200.0,
        max_new_tokens: int = 512,
        deterministic: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize the adapter.
        
        Args:
            engine: LookupTableInferenceEngine instance
            latency_budget: Latency budget in milliseconds (can be overridden per request)
            max_new_tokens: Maximum number of tokens to generate
            deterministic: If True, use deterministic (argmax) actions (always True for lookup table)
            device: Device to use
        """
        self.engine = engine
        self.latency_budget = latency_budget
        self.max_new_tokens = max_new_tokens
        self.deterministic = deterministic  # Always True for lookup table
        self.device = device
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "total_latency": 0.0,
            "knob_distribution": {
                "tier": {"low": 0, "medium": 0, "high": 0},
                "top_k": {},
                "num_active_blocks": {},
            }
        }
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        latency_budget: Optional[float] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompts and images.
        
        Args:
            prompts: Single prompt string or list of prompts
            images: Single image or list of images (PIL Image)
            latency_budget: Optional latency budget override
            **kwargs: Additional arguments
        
        Returns:
            Generated text (string or list of strings)
        """
        # Use provided budget or default
        budget = latency_budget if latency_budget is not None else self.latency_budget
        
        # Handle single vs batch
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
            if images is not None and not isinstance(images, list):
                images = [images]
        
        results = []
        
        for i, prompt in enumerate(prompts):
            image = images[i] if images is not None and i < len(images) else None
            
            try:
                # Run inference
                result = self.engine.infer(
                    prompt=prompt,
                    images=image,
                    latency_budget=budget,
                    max_new_tokens=self.max_new_tokens,
                    return_knobs=True,
                )
                
                generated_text = result.get("text", "")
                knobs = result.get("knobs", {})
                latency = result.get("latency", 0.0)
                
                # Update statistics
                self.stats["total_requests"] += 1
                self.stats["total_latency"] += latency
                
                if knobs:
                    tier = knobs.get("tier", "unknown")
                    if tier in self.stats["knob_distribution"]["tier"]:
                        self.stats["knob_distribution"]["tier"][tier] += 1
                    
                    top_k = knobs.get("top_k", 0)
                    if top_k not in self.stats["knob_distribution"]["top_k"]:
                        self.stats["knob_distribution"]["top_k"][top_k] = 0
                    self.stats["knob_distribution"]["top_k"][top_k] += 1
                    
                    num_blocks = knobs.get("num_active_blocks", 0)
                    if num_blocks not in self.stats["knob_distribution"]["num_active_blocks"]:
                        self.stats["knob_distribution"]["num_active_blocks"][num_blocks] = 0
                    self.stats["knob_distribution"]["num_active_blocks"][num_blocks] += 1
                
                results.append(generated_text)
                
            except Exception as e:
                log.error(f"Error generating for prompt {i}: {e}")
                results.append("")  # Return empty string on error
        
        # Return single result if input was single
        if is_single:
            return results[0] if results else ""
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        stats = self.stats.copy()
        if stats["total_requests"] > 0:
            stats["avg_latency"] = stats["total_latency"] / stats["total_requests"]
        else:
            stats["avg_latency"] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_requests": 0,
            "total_latency": 0.0,
            "knob_distribution": {
                "tier": {"low": 0, "medium": 0, "high": 0},
                "top_k": {},
                "num_active_blocks": {},
            }
        }


def create_lookup_table_lmms_eval_adapter(
    model_path: str,
    lookup_table_path: str,
    latency_budget: float = 200.0,
    max_new_tokens: int = 512,
    deterministic: bool = True,
    device: str = "cuda",
) -> LookupTableLMMSEvalAdapter:
    """
    Factory function to create an lmms-eval adapter for lookup table baseline.
    
    Args:
        model_path: Path to model checkpoint
        lookup_table_path: Path to lookup table JSON file
        latency_budget: Latency budget in milliseconds
        max_new_tokens: Maximum number of tokens to generate
        deterministic: If True, use deterministic actions (always True for lookup table)
        device: Device to use
    
    Returns:
        LookupTableLMMSEvalAdapter instance
    """
    from experiments.controller.evaluate_lookup_table_baseline import create_lookup_table_inference_engine
    
    # Create lookup table inference engine
    engine = create_lookup_table_inference_engine(
        model_path=model_path,
        lookup_table_path=lookup_table_path,
        device=device,
    )
    
    # Create adapter
    adapter = LookupTableLMMSEvalAdapter(
        engine=engine,
        latency_budget=latency_budget,
        max_new_tokens=max_new_tokens,
        deterministic=deterministic,
        device=device,
    )
    
    return adapter

