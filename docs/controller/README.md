# Controller Design Documentation

> **Status**: Unified design documentation index  
> **Last Updated**: 2026-01-01  
> **Version**: 2.0

## 📚 Documentation Navigation

### 🎯 Core Documents (Must Read)

1. **[DESIGN.md](DESIGN.md)** - **Unified Design Document** ⭐
   - Complete design architecture
   - Detailed design of three knobs
   - Two-stage prediction architecture
   - Input feature design
   - Training methods
   - Overhead analysis
   - Key design decisions

2. **[ANALYSIS.md](ANALYSIS.md)** - **Technical Analysis Document** ⭐
   - Feature extraction analysis
   - Controller architecture analysis
   - Training method analysis
   - Latency estimator design
   - Feasibility analysis

3. **[EXPERIMENTS.md](EXPERIMENTS.md)** - **Experiments Document** ⭐
   - Detailed description of all experiments
   - Experiment purpose, scripts, expected outputs
   - Experiment execution order
   - Troubleshooting guide

### 🔬 Specialized Documents

4. **[SEMANTIC_ROUTER_INTEGRATION.md](SEMANTIC_ROUTER_INTEGRATION.md)** - **Semantic Router Integration Research**
   - Semantic Router introduction
   - Integration schemes (three approaches)
   - Implementation recommendations
   - Advantages analysis

5. **[ADALORA_DESIGNS.md](ADALORA_DESIGNS.md)** - **AdaLoRA-Inspired Designs (Two Approaches)**
   - Approach 1: Two-stage prediction (current implementation)
   - Approach 2: One-stage prediction (alternative)
   - Comparison of both approaches
   - Implementation plan

## 🚀 Quick Start

### 1. Train Latency Estimator

```bash
python experiments/controller/train_latency_estimator.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa coco_2014_vqa \
    --output_dir checkpoints/latency_estimator \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 1e-3 \
    --device cuda
```

### 2. Train Two-Stage Controller

**Stage 1 (Knob1)**:
```bash
python experiments/controller/train_two_stage_controller.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa coco_2014_vqa \
    --model_path checkpoints/molmo \
    --output_dir checkpoints/two_stage_controller \
    --stage stage1 \
    --batch_size 64 \
    --num_epochs_stage1 50 \
    --device cuda
```

**Stage 2 (Knob2 & Knob3)**:
```bash
python experiments/controller/train_two_stage_controller.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa coco_2014_vqa \
    --model_path checkpoints/molmo \
    --latency_estimator_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --output_dir checkpoints/two_stage_controller \
    --stage stage2 \
    --batch_size 32 \
    --num_epochs_stage2 100 \
    --group_size 5 \
    --device cuda
```

### 3. Test Adaptive Inference

```bash
python experiments/controller/test_adaptive_inference.py \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0 \
    --device cuda
```

## 📋 System Architecture

### Two-Stage Prediction Architecture

```
Stage 1 (Before Vision Encoder):
  Input: Language Feature + Budget Feature
  Output: Knob1 (Vision Tokens Tier: low/medium/high)
  
Stage 2 (After Vision Encoder + Projector):
  Input: Vision Feature + Language Feature + Budget Feature
  Output: Knob2 (MoE Top-K: 4/6/8/10/12) + Knob3 (Transformer Blocks: 8/10/12/14/16)
```

### Three Knobs

| Knob | Control Content | Decision Timing | Implementation | Output Space |
|------|----------------|-----------------|----------------|--------------|
| **Knob1** | Vision tokens tier | Before vision encoder | Stage 1 predictor | 3 choices |
| **Knob2** | MoE top-K | After vision encoder | Stage 2 predictor | 5 choices |
| **Knob3** | Transformer blocks count | After vision encoder | **Importance-based pruning** | 5 choices |

## 🔑 Key Design Decisions

### 1. Why Two-Stage?

**Knob1 must be determined before vision encoder** because:
- Crop count determines how images are processed (tiling, resize)
- Once images enter vision encoder, crop count is fixed
- Cannot change vision token count after vision encoding

### 2. Importance-Based Pruning

**Knob3 uses importance-based pruning**:
- Controller only predicts block count (5 choices)
- Uses pre-computed importance scores to select most important blocks
- Importance scores are **Data-Agnostic but Task-Dependent**
- Significantly simplifies controller output space (from 2^16 to 5)

### 3. Three Variants of Knob1

1. **Budget-Only** (minimal overhead): Only uses latency budget
2. **Budget + Language** (current default): Uses budget and language features
3. **Budget + Language + Vision** (highest accuracy): Uses all features, but higher overhead

### 4. Latency Estimator

**Why do we need Latency Estimator?**
- Avoids actual model execution during RL training
- Enables larger batch sizes
- Accelerates training process
- Can train different estimators for different hardware

## 📊 Performance Targets (SIGMETRICS Standards)

- **Overhead**: Controller overhead <0.1% of total inference
- **Decision Time**: <0.2ms
- **Effectiveness**: Significantly improves accuracy-latency trade-off
- **Simplicity**: Simple design, easy to deploy

## 🗂️ Code Structure

```
experiments/controller/
├── controller.py                    # Unified two-stage controller
├── feature_extractors.py           # Feature extraction
├── importance_based_block_selection.py  # Block selection
├── latency_estimator.py            # Latency estimator
├── core_exp_data_loader.py         # Data loading
├── train_latency_estimator.py      # Train latency estimator
├── train_two_stage_controller.py   # Train two-stage controller
├── test_adaptive_inference.py      # Test inference pipeline
└── adaptive_inference.py           # Inference engine
```

## 📝 Implementation Status

### ✅ Completed

- [x] Unified Controller implementation (supports three Knob1 variants)
- [x] Importance-based block selection (supports task-dependent)
- [x] Latency estimator model and training
- [x] Stage 1 training (supervised learning)
- [x] Stage 2 model (GRPO ready)
- [x] Complete documentation consolidation

### ⏳ In Progress

- [ ] Complete Stage 2 training (requires online execution)
- [ ] Semantic Router integration research
- [ ] One-stage AdaLoRA design implementation

### 📋 To Do

- [ ] Complete testing and validation
- [ ] Performance evaluation and optimization
- [ ] Hardware-specific optimization

## 🔗 Related Documentation

- **Importance Score Analysis**: `docs/profiling/`
- **Core Experiment**: `docs/core_exp/`
- **Code Updates**: See `DESIGN.md` and `ANALYSIS.md`

---

**Maintainer**: Controller Team  
**Issues**: Please refer to detailed descriptions in each document
