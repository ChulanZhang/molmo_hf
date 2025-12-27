# GRPO Controller Design

## 1. Goal
Train a lightweight controller that, given image features, language features, and a latency budget, selects max_crops, top_k, and num_active_blocks to maximize accuracy under latency constraints.

## 2. GRPO intuition
Group Relative Policy Optimization compares trajectories within groups:
- Advantage per sample: `A_rel = R_i - R_group_mean`
- No value network is needed; relative rewards stabilize training and suit offline data (exp5/exp6).

## 3. System architecture
1) Feature extraction (vision backbone, language encoder) → pooled features
2) Controller (policy net) → action distribution (max_crops, top_k, num_active_blocks)
3) Apply config → run model → measure latency/accuracy
4) Reward: `R = α·accuracy - β·latency_penalty - γ·budget_violation`

## 4. Feature design
- Image: CLS/pooled vision features (recommended) or stats over patch features.
- Language: CLS or mean-pool token embeddings (mean pooling recommended for simplicity).
- Budget: encode scalar budget via a small MLP or sinusoidal encoding.

## 5. Controller network (sketch)
- Inputs: image_feat_dim, lang_feat_dim, budget_dim (default 32)
- Hidden: ~256 dims
- Heads: logits for max_crops, top_k, num_active_blocks (options must match training data)

## 6. Placement
Pre-inference controller (recommended): decide config before model forward; low coupling, easy to train/deploy.

## 7. Notes
- Keep controller overhead small.
- Ensure action spaces align with training data (option lists).
- Use deterministic sampling for validation; stochastic (temperature) for training.
