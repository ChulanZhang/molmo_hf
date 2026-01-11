"""
LMms-Eval Adapter for Adaptive Inference Engine
Wraps the adaptive inference engine to be compatible with lmms-eval framework.
Based on AdaLLaVA's implementation approach.
"""

import logging
import torch
from typing import Dict, List, Optional, Any, Union
from PIL import Image
import numpy as np

log = logging.getLogger(__name__)


class AdaptiveInferenceLMMSEvalAdapter:
    """
    Adapter that wraps AdaptiveInferenceEngine to work with lmms-eval.
    
    This adapter implements the interface expected by lmms-eval's model API,
    allowing the adaptive inference engine to be evaluated on standard benchmarks.
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
            engine: AdaptiveInferenceEngine instance
            latency_budget: Latency budget in milliseconds (can be overridden per request)
            max_new_tokens: Maximum number of tokens to generate
            deterministic: If True, use deterministic (argmax) actions
            device: Device to use
        """
        self.engine = engine
        self.latency_budget = latency_budget
        self.max_new_tokens = max_new_tokens
        self.deterministic = deterministic
        self.device = device
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "total_latency": 0.0,
            "total_flops": 0.0,
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
        Generate text from prompts and optional images.
        
        This is the main interface method expected by lmms-eval.
        
        Args:
            prompts: Single prompt string or list of prompts
            images: Optional single image or list of images (PIL Image)
            latency_budget: Optional latency budget override (in milliseconds)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text (string or list of strings)
        """
        # Handle single vs batch inputs
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
            if images is not None and not isinstance(images, list):
                images = [images]
        
        # Use provided latency budget or default
        budget = latency_budget if latency_budget is not None else self.latency_budget
        
        # Override max_new_tokens if provided
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        
        results = []
        for i, prompt in enumerate(prompts):
            image = images[i] if images and i < len(images) else None
            
            try:
                # Call adaptive inference engine
                result = self.engine.infer(
                    prompt=prompt,
                    images=image,
                    latency_budget=budget,
                    max_new_tokens=max_new_tokens,
                    deterministic=self.deterministic,
                    return_knobs=True,
                )
                
                generated_text = result.get("text", "")
                results.append(generated_text)
                
                # Update statistics
                self.stats["total_requests"] += 1
                if "knobs" in result:
                    knobs = result["knobs"]
                    # Track tier distribution
                    tier = knobs.get("tier", "unknown")
                    if tier in self.stats["knob_distribution"]["tier"]:
                        self.stats["knob_distribution"]["tier"][tier] += 1
                    
                    # Track top_k distribution
                    top_k = knobs.get("top_k", 0)
                    self.stats["knob_distribution"]["top_k"][top_k] = \
                        self.stats["knob_distribution"]["top_k"].get(top_k, 0) + 1
                    
                    # Track num_active_blocks distribution
                    num_blocks = knobs.get("num_active_blocks", 0)
                    self.stats["knob_distribution"]["num_active_blocks"][num_blocks] = \
                        self.stats["knob_distribution"]["num_active_blocks"].get(num_blocks, 0) + 1
                
                if "latency" in result:
                    self.stats["total_latency"] += result["latency"]
                
            except Exception as e:
                log.error(f"Error generating for prompt {i}: {e}")
                results.append("")  # Return empty string on error
        
        # Return single result if input was single
        if is_single:
            return results[0] if results else ""
        return results
    
    def generate_from_batch(
        self,
        batch: Dict[str, Any],
        latency_budget: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate from a batch dictionary (used by some lmms-eval tasks).
        
        Args:
            batch: Batch dictionary with 'prompt' and optionally 'image' keys
            latency_budget: Optional latency budget override
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated text strings
        """
        prompts = batch.get("prompt", [])
        images = batch.get("image", None)
        
        return self.generate(
            prompts=prompts,
            images=images,
            latency_budget=latency_budget,
            **kwargs
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get accumulated statistics."""
        stats = self.stats.copy()
        
        # Compute averages
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
            "total_flops": 0.0,
            "knob_distribution": {
                "tier": {"low": 0, "medium": 0, "high": 0},
                "top_k": {},
                "num_active_blocks": {},
            }
        }


def create_lmms_eval_adapter(
    model_path: str,
    controller_path: str,
    latency_budget: float = 200.0,
    max_new_tokens: int = 512,
    deterministic: bool = True,
    device: str = "cuda",
) -> AdaptiveInferenceLMMSEvalAdapter:
    """
    Factory function to create an lmms-eval adapter.
    
    Args:
        model_path: Path to model checkpoint
        controller_path: Path to controller checkpoint
        latency_budget: Latency budget in milliseconds
        max_new_tokens: Maximum number of tokens to generate
        deterministic: If True, use deterministic actions
        device: Device to use
    
    Returns:
        AdaptiveInferenceLMMSEvalAdapter instance
    """
    from experiments.controller.adaptive_inference import create_adaptive_inference_engine
    
    # Create adaptive inference engine
    engine = create_adaptive_inference_engine(
        model_path=model_path,
        controller_path=controller_path,
        device=device,
    )
    
    # Create adapter
    adapter = AdaptiveInferenceLMMSEvalAdapter(
        engine=engine,
        latency_budget=latency_budget,
        max_new_tokens=max_new_tokens,
        deterministic=deterministic,
        device=device,
    )
    
    return adapter

