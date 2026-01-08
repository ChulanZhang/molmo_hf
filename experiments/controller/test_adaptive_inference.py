"""
Test script for adaptive inference with two-stage controller.
Tests the complete inference pipeline and validates functionality.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.base_experiment import BaseExperiment
from experiments.controller.adaptive_inference import AdaptiveInferenceEngine, create_adaptive_inference_engine
from experiments.controller.controller import TwoStageController, Knob1PredictorBudgetLanguage, Knob2Knob3Predictor
from experiments.controller.feature_extractors import LanguageFeatureExtractor, LatencyBudgetEncoder
from experiments.controller.latency_estimator import LatencyEstimator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def test_stage1_prediction(
    knob1_predictor: Knob1Predictor,
    lang_extractor: LanguageFeatureExtractor,
    budget_encoder: LatencyBudgetEncoder,
    device: str = "cuda",
):
    """Test Stage 1 (Knob1) prediction."""
    log.info("=" * 80)
    log.info("Testing Stage 1: Knob1 Prediction")
    log.info("=" * 80)
    
    # Test cases
    test_cases = [
        {"prompt": "What is in this image?", "budget": 100.0},
        {"prompt": "Describe this image in detail.", "budget": 300.0},
        {"prompt": "What color is the sky?", "budget": 200.0},
    ]
    
    for i, case in enumerate(test_cases):
        prompt = case["prompt"]
        budget = case["budget"]
        
        # Extract features
        lang_feat = lang_extractor.extract(prompt).to(device)
        budget_feat = budget_encoder(torch.tensor([[budget]], device=device)).squeeze(0)
        
        # Predict
        knob1_predictor.eval()
        with torch.no_grad():
            logits = knob1_predictor(lang_feat, budget_feat.unsqueeze(0))
            tier_idx, tier_values = knob1_predictor.sample(logits, deterministic=True)
        
        log.info(f"Test {i+1}:")
        log.info(f"  Prompt: {prompt}")
        log.info(f"  Budget: {budget}ms")
        log.info(f"  Predicted tier: {tier_values[0]}")
        log.info(f"  Logits: {logits.cpu().numpy()}")
        log.info("")
    
    log.info("Stage 1 test completed!\n")


def test_stage2_prediction(
    knob2_knob3_predictor: Knob2Knob3Predictor,
    lang_extractor: LanguageFeatureExtractor,
    budget_encoder: LatencyBudgetEncoder,
    device: str = "cuda",
):
    """Test Stage 2 (Knob2 & Knob3) prediction."""
    log.info("=" * 80)
    log.info("Testing Stage 2: Knob2 & Knob3 Prediction")
    log.info("=" * 80)
    
    # Test cases
    test_cases = [
        {"budget": 100.0},
        {"budget": 200.0},
        {"budget": 300.0},
    ]
    
    # Placeholder vision feature
    vision_feat = torch.zeros(1, 2048, device=device)
    
    for i, case in enumerate(test_cases):
        budget = case["budget"]
        prompt = "What is in this image?"
        
        # Extract features
        lang_feat = lang_extractor.extract(prompt).to(device)
        budget_feat = budget_encoder(torch.tensor([[budget]], device=device)).squeeze(0)
        
        # Predict
        knob2_knob3_predictor.eval()
        with torch.no_grad():
            knob2_logits, knob3_logits = knob2_knob3_predictor(
                vision_feat, lang_feat, budget_feat.unsqueeze(0)
            )
            actions = knob2_knob3_predictor.sample(
                knob2_logits, knob3_logits, deterministic=True
            )
        
        log.info(f"Test {i+1}:")
        log.info(f"  Budget: {budget}ms")
        log.info(f"  Predicted top_k: {actions['knob2'][0].item()}")
        log.info(f"  Predicted num_active_blocks: {actions['knob3'][0].item()}")
        log.info(f"  Knob2 logits: {knob2_logits.cpu().numpy()}")
        log.info(f"  Knob3 logits: {knob3_logits.cpu().numpy()}")
        log.info("")
    
    log.info("Stage 2 test completed!\n")


def test_latency_estimator(
    latency_estimator: LatencyEstimator,
    device: str = "cuda",
):
    """Test latency estimator."""
    log.info("=" * 80)
    log.info("Testing Latency Estimator")
    log.info("=" * 80)
    
    # Test cases
    test_cases = [
        {
            "vision_tokens": 1008,
            "text_tokens": 45,
            "output_tokens": 8,
            "tier": "medium",
            "top_k": 8,
            "num_active_blocks": 12,
        },
        {
            "vision_tokens": 1440,
            "text_tokens": 50,
            "output_tokens": 10,
            "tier": "high",
            "top_k": 12,
            "num_active_blocks": 16,
        },
    ]
    
    for i, case in enumerate(test_cases):
        tier_map = {'low': 0, 'medium': 1, 'high': 2}
        tier_idx = tier_map.get(case['tier'], 1)
        
        latency_estimator.eval()
        with torch.no_grad():
            latencies = latency_estimator(
                vision_tokens=torch.tensor([case['vision_tokens']], device=device),
                text_tokens=torch.tensor([case['text_tokens']], device=device),
                tier_idx=torch.tensor([tier_idx], device=device),
                top_k=torch.tensor([case['top_k']], device=device),
                num_active_blocks=torch.tensor([case['num_active_blocks']], device=device),
                output_tokens=torch.tensor([case['output_tokens']], device=device),
            )
        
        log.info(f"Test {i+1}:")
        log.info(f"  Config: tier={case['tier']}, top_k={case['top_k']}, blocks={case['num_active_blocks']}")
        log.info(f"  T_vision_total: {latencies['T_vision_total'].item():.2f}ms")
        log.info(f"  T_LLM_prefill: {latencies['T_LLM_prefill'].item():.2f}ms")
        log.info(f"  T_LLM_decode_per_token: {latencies['T_LLM_decode_per_token'].item():.2f}ms")
        log.info(f"  T_total: {latencies['T_total'].item():.2f}ms")
        log.info("")
    
    log.info("Latency estimator test completed!\n")


def test_full_pipeline(
    model_path: str,
    controller_path: str,
    device: str = "cuda",
):
    """Test full adaptive inference pipeline."""
    log.info("=" * 80)
    log.info("Testing Full Adaptive Inference Pipeline")
    log.info("=" * 80)
    
    try:
        # Create engine
        log.info("Creating adaptive inference engine...")
        engine = create_adaptive_inference_engine(
            model_path=model_path,
            controller_path=controller_path,
            device=device,
        )
        log.info("Engine created successfully!")
        
        # Test inference
        test_prompts = [
            "What is in this image?",
            "Describe this image.",
        ]
        
        for i, prompt in enumerate(test_prompts):
            log.info(f"\nTest {i+1}: {prompt}")
            try:
                result = engine.infer(
                    prompt=prompt,
                    latency_budget=200.0,
                    max_new_tokens=128,
                    deterministic=True,
                    return_knobs=True,
                )
                
                log.info(f"  Generated text: {result['text'][:100]}...")
                if 'knobs' in result:
                    log.info(f"  Knobs: {result['knobs']}")
            except Exception as e:
                log.error(f"  Error: {e}")
        
        log.info("\nFull pipeline test completed!")
        
    except Exception as e:
        log.error(f"Error creating engine: {e}")
        log.error("This is expected if checkpoints are not available")
        log.error("Please train the controller first")


def main():
    parser = argparse.ArgumentParser(
        description="Test Adaptive Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--controller_path",
        type=str,
        default=None,
        help="Path to controller checkpoint (for full pipeline test)"
    )
    parser.add_argument(
        "--latency_estimator_path",
        type=str,
        default=None,
        help="Path to latency estimator checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--test_stage1",
        action="store_true",
        help="Test Stage 1 prediction"
    )
    parser.add_argument(
        "--test_stage2",
        action="store_true",
        help="Test Stage 2 prediction"
    )
    parser.add_argument(
        "--test_latency_estimator",
        action="store_true",
        help="Test latency estimator"
    )
    parser.add_argument(
        "--test_full_pipeline",
        action="store_true",
        help="Test full adaptive inference pipeline"
    )
    parser.add_argument(
        "--test_all",
        action="store_true",
        help="Run all tests"
    )
    
    args = parser.parse_args()
    
    if args.test_all:
        args.test_stage1 = True
        args.test_stage2 = True
        args.test_latency_estimator = True
        args.test_full_pipeline = True
    
    # Load model for feature extraction
    experiment = BaseExperiment(model_path=args.model_path, device=args.device)
    tokenizer = experiment.tokenizer
    wte_layer = experiment.model.model.transformer.wte
    
    # Initialize feature extractors
    lang_extractor = LanguageFeatureExtractor(tokenizer, wte_layer, max_length=512)
    budget_encoder = LatencyBudgetEncoder(hidden_dim=256, use_sinusoidal=False).to(args.device)
    
    # Test Stage 1
    if args.test_stage1:
        knob1_predictor = Knob1Predictor().to(args.device)
        # Load checkpoint if available
        if args.controller_path:
            try:
                checkpoint = torch.load(args.controller_path, map_location=args.device)
                if 'model_state_dict' in checkpoint:
                    knob1_predictor.load_state_dict(checkpoint['model_state_dict'])
                    log.info(f"Loaded Stage 1 checkpoint from {args.controller_path}")
            except Exception as e:
                log.warning(f"Could not load Stage 1 checkpoint: {e}")
        
        test_stage1_prediction(knob1_predictor, lang_extractor, budget_encoder, args.device)
    
    # Test Stage 2
    if args.test_stage2:
        knob2_knob3_predictor = Knob2Knob3Predictor().to(args.device)
        # Load checkpoint if available
        if args.controller_path:
            try:
                checkpoint = torch.load(args.controller_path, map_location=args.device)
                if 'model_state_dict' in checkpoint:
                    knob2_knob3_predictor.load_state_dict(checkpoint['model_state_dict'])
                    log.info(f"Loaded Stage 2 checkpoint from {args.controller_path}")
            except Exception as e:
                log.warning(f"Could not load Stage 2 checkpoint: {e}")
        
        test_stage2_prediction(knob2_knob3_predictor, lang_extractor, budget_encoder, args.device)
    
    # Test Latency Estimator
    if args.test_latency_estimator:
        if args.latency_estimator_path:
            latency_estimator = LatencyEstimator().to(args.device)
            checkpoint = torch.load(args.latency_estimator_path, map_location=args.device)
            latency_estimator.load_state_dict(checkpoint['model_state_dict'])
            log.info(f"Loaded latency estimator from {args.latency_estimator_path}")
            test_latency_estimator(latency_estimator, args.device)
        else:
            log.warning("Latency estimator path not provided - skipping test")
    
    # Test Full Pipeline
    if args.test_full_pipeline:
        if args.controller_path:
            test_full_pipeline(args.model_path, args.controller_path, args.device)
        else:
            log.warning("Controller path not provided - skipping full pipeline test")
    
    log.info("All tests completed!")


if __name__ == "__main__":
    main()

