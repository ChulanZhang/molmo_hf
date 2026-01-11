# Controller Design Documentation

> **Status**: Unified design documentation index  
> **Last Updated**: 2026-01-10  
> **Version**: 3.0 (Joint Training Only)

## 📚 Documentation Navigation

> **New to Controller Design?** Start with **[OVERVIEW.md](OVERVIEW.md)** for a comprehensive introduction!

### 🎯 Core Documents (Must Read)

1. **[OVERVIEW.md](OVERVIEW.md)** - **Quick Start Guide** ⭐⭐⭐
   - Quick introduction to controller design
   - Core architecture overview
   - Key design decisions
   - Quick start commands

2. **[DESIGN.md](DESIGN.md)** - **Unified Design Document** ⭐⭐⭐
   - Complete design architecture
   - Detailed design of three knobs
   - Two-stage prediction architecture
   - Input feature design
   - Training methods
   - Overhead analysis
   - Key design decisions

3. **[training_guide.md](training_guide.md)** - **Training Guide** ⭐⭐
   - Complete training workflow
   - Step-by-step instructions
   - Hyperparameter tuning
   - Troubleshooting

4. **[EXPERIMENTS.md](EXPERIMENTS.md)** - **Experiments Document** ⭐⭐
   - Detailed description of all experiments
   - Experiment purpose, scripts, expected outputs
   - Experiment execution order
   - Troubleshooting guide

### 🔬 Specialized Documents

5. **[JOINT_TRAINING.md](JOINT_TRAINING.md)** - **Joint Training Design** ⭐
   - Joint training architecture
   - GRPO algorithm details
   - Reward function design
   - Training process

6. **[ANALYSIS.md](ANALYSIS.md)** - **Technical Analysis** ⭐
   - Input feature design analysis
   - Controller architecture analysis
   - Training method analysis
   - Overhead and feasibility analysis

7. **[DECODE_PHASE_DESIGN.md](DECODE_PHASE_DESIGN.md)** - **Decode Phase Design** ⭐
   - Decode phase implementation
   - Configuration preservation
   - Budget token handling

8. **[BUDGET_ENCODER_TRAINING.md](BUDGET_ENCODER_TRAINING.md)** - **Budget Encoder Training** ⭐
   - Budget encoder architecture
   - Training strategy
   - Sinusoidal encoding vs MLP

9. **[LATENCY_BUDGET_ANALYSIS.md](LATENCY_BUDGET_ANALYSIS.md)** - **Latency Budget Analysis**
   - Budget range determination
   - Pareto frontier analysis
   - Budget sampling strategy

10. **[LATENCY_BUDGET_ENCODING.md](LATENCY_BUDGET_ENCODING.md)** - **Budget Encoding Implementation**
    - AdaLLaVA-style encoding
    - Sinusoidal positional encoding
    - Two-layer MLP design
    - Budget token integration

11. **[REWARD_DESIGN_EXPLANATION.md](REWARD_DESIGN_EXPLANATION.md)** - **Reward Function Design**
    - Reward components
    - Accuracy reward
    - Latency penalty
    - Budget violation penalty

12. **[TRAINING_PRINCIPLE.md](TRAINING_PRINCIPLE.md)** - **Training Principles**
    - GRPO training principles
    - Reward function design
    - Training optimization

13. **[TRAINING_FAQ.md](TRAINING_FAQ.md)** - **Training FAQ**
    - Common questions
    - Troubleshooting
    - Best practices

14. **[TRAINING_MODULES.md](TRAINING_MODULES.md)** - **Training Modules Status**
    - Trainable modules
    - Frozen modules
    - Module dependencies

15. **[TRAINING_ISSUES_FIXED.md](TRAINING_ISSUES_FIXED.md)** - **Training Issues Fixed**
    - Common issues and solutions
    - Error fixes
    - Best practices

16. **[evaluation_guide.md](evaluation_guide.md)** - **Evaluation Guide**
    - Evaluation methods
    - Metrics calculation
    - Best practices

17. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - **Implementation Summary**
    - Current implementation status
    - Key design decisions
    - Update history

### 📚 Additional Documents

18. **[GRPO_EXPLANATION.md](GRPO_EXPLANATION.md)** - **GRPO Algorithm Explanation**
    - GRPO core concepts
    - Implementation details
    - Loss function

19. **[lookup_table_baseline.md](lookup_table_baseline.md)** - **Lookup Table Baseline**
    - Baseline controller design
    - Usage and comparison

20. **[WANDB_USAGE.md](WANDB_USAGE.md)** - **Weights & Biases Usage**
    - Wandb integration guide
    - Logging setup

21. **[EXPANDED_BATCH_SIZE_EXPLANATION.md](EXPANDED_BATCH_SIZE_EXPLANATION.md)** - **Expanded Batch Size Explanation**
    - GRPO batch size concept
    - Multi-GPU training notes

22. **[LOGGING_TOOL_COMPARISON.md](LOGGING_TOOL_COMPARISON.md)** - **Logging Tool Comparison**
    - TensorBoard vs wandb comparison
    - Usage recommendations

## 🚀 Quick Start

### 1. Train Joint Controller (Stage1 + Stage2)

```bash
./experiments/controller/run_training.sh
```

Or manually:

```bash
python experiments/controller/train_joint_controller.py \
    --results_dir results/core_exp_h100/5run_2000samples_w_new_importance_score \
    --dataset_names text_vqa coco_2014_vqa okvqa \
    --model_path checkpoints \
    --output_dir checkpoints/joint_controller \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --stage1_lr_ratio 1.0 \
    --group_size 5 \
    --device cuda \
    --seed 42 \
    --use_multi_gpu
```

**Key Points**:
- **Joint Training**: Stage1 and Stage2 are trained together end-to-end
- **Direct Latency Measurement**: Uses hooks to measure actual latency (batch_size=1 per sample)
- **Budget Range**: Latency budgets sampled from [170ms, 380ms] uniformly
- **No Latency Estimator**: Direct measurement is used for both training and validation

### 2. Test Adaptive Inference

```bash
python experiments/controller/test_adaptive_inference.py \
    --model_path checkpoints \
    --controller_path checkpoints/joint_controller/joint_checkpoint_epoch_100.pt \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0 \
    --device cuda
```

## 📋 System Architecture

### Two-Stage Prediction Architecture

```
Stage 1 (Before Vision Encoder):
  Input: Language Feature + Budget Token (encoded)
  Output: Knob1 (Vision Tokens Tier: low/medium/high) + Insertion Position (1-5)
  
Stage 2 (After Insertion Position):
  Input: Latency Token (from LLM after insertion position)
  Output: Knob2 (MoE Top-K: 4/5/6/7/8) + Knob3 (Total Blocks: 12/13/14/15/16)
```

### Three Knobs

| Knob | Control Content | Decision Timing | Implementation | Output Space |
|------|----------------|-----------------|----------------|--------------|
| **Knob1** | Vision tokens tier + Stage2 insertion position | Before vision encoder | Stage 1 predictor | 3 tiers × 5 positions |
| **Knob2** | MoE top-K | After insertion position | Stage 2 predictor | 5 choices (4,5,6,7,8) |
| **Knob3** | Transformer blocks count | After insertion position | **Importance-based pruning** | 5 choices (12-16 total blocks) |

**Key Design**:
- **First block fixed**: Top-K=8, always included
- **Dynamic insertion**: Stage1 decides where to insert Stage2 (after block 1-5)
- **Importance-based selection**: Knob3 uses pre-computed importance scores
- **Budget token**: Encoded as d_model-dimensional token, concatenated to input sequence

## 🔑 Key Design Decisions

### 1. Joint Training Only

**Current Implementation**: Only joint training is supported
- Stage1 and Stage2 are trained together end-to-end
- Both stages share the same reward signal
- GRPO algorithm optimizes both stages simultaneously

### 2. Direct Latency Measurement

**No Latency Estimator**: Direct measurement using hooks
- Batch size = 1 per sample (for accurate latency measurement)
- Uses PyTorch hooks to measure prefill and decode latency
- More accurate than estimator, but slower training

### 3. Budget Token Integration

**Budget as Token**: Latency budget encoded as token and concatenated
- Encoded using sinusoidal encoding (256-D) + MLP (to d_model)
- Concatenated to input sequence in prefill phase only
- Budget encoder MLP is trainable (sinusoidal encoding is fixed)

### 4. Decode Phase

**Configuration Preservation**: Decode uses prefill configuration
- Controller runs only in prefill phase
- Decode phase uses prefill-determined configuration (top_k, num_blocks)
- Budget token not added in decode phase

### 5. Importance-Based Block Selection

**Knob3 Simplification**: Uses pre-computed importance scores
- Controller predicts total block count (5 choices: 12-16)
- Blocks selected based on importance scores (task-dependent)
- Always includes first block and blocks before insertion position

## 📊 Training Configuration

### Current Settings

- **Latency Budget Range**: [170ms, 380ms] (uniform sampling)
- **Knob2 Options**: [4, 5, 6, 7, 8]
- **Knob3 Options**: [12, 13, 14, 15, 16] (total blocks)
- **First Block**: Fixed top_k=8, always included
- **Insertion Positions**: [1, 2, 3, 4, 5] (after block 1-5)
- **Max New Tokens**: 64
- **Batch Size**: 8 (samples processed one by one for latency measurement)

### Training Modules

**Trainable**:
- Stage1 Controller (knob1_predictor)
- Stage2 Controller (knob2_knob3_predictor)
- Budget Encoder MLP (budget_encoder.mlp)

**Frozen**:
- LLM Model
- Budget Encoder Sinusoidal Encoding
- Language Feature Extractor (wte_layer)

## 🗂️ Code Structure

```
experiments/controller/
├── controller.py                    # Controller models (Stage1 & Stage2)
├── feature_extractors.py           # Feature extraction (Language, Budget)
├── importance_based_block_selection.py  # Block selection
├── joint_grpo_trainer.py          # Joint GRPO trainer ⭐
├── train_joint_controller.py      # Main training script ⭐
├── online_training_dataset.py     # Online training dataset
├── model_loader.py                # Model loading utility
├── model_forward_with_dynamic_stage2.py  # Dynamic forward pass
├── test_adaptive_inference.py     # Test inference pipeline
├── adaptive_inference.py          # Inference engine
└── run_training.sh                 # Training script
```

## 📝 Implementation Status

### ✅ Completed

- [x] Joint Training implementation (Stage1 + Stage2)
- [x] Direct latency measurement (hooks)
- [x] Budget token integration
- [x] Dynamic insertion position
- [x] Importance-based block selection
- [x] Decode phase configuration preservation
- [x] Budget encoder MLP training

### ⏳ In Progress

- [ ] Performance evaluation and optimization
- [ ] Hyperparameter tuning
- [ ] Multi-dataset evaluation

### 📋 To Do

- [ ] Complete testing and validation
- [ ] Hardware-specific optimization
- [ ] Production deployment guide

## 🔗 Related Documentation

- **Importance Score Analysis**: `docs/profiling/`
- **Core Experiment**: `docs/core_exp/`
- **Code Updates**: See `DESIGN.md` and `JOINT_TRAINING.md`

---

**Maintainer**: Controller Team  
**Last Updated**: 2026-01-10  
**Version**: 3.0 (Joint Training Only)
