# Molmo Inference Flow

## Overview
This document explains the Molmo inference pipeline end to end, including key stages, code locations, and practical tips for debugging and optimization.

Pipeline:
```
Input preprocessing → Vision encoding → Text embedding fusion → Transformer prefill → Autoregressive decode → Output decoding
```

## 1. High-level timeline
```
1) Input preprocessing (processor)
   - Text tokenization
   - Image resize / crop / patchify
   Code: molmo/preprocessors/preprocessing_molmo.py

2) Vision encoding (prefill - vision)
   - ViT encodes image patches
   - Image pooling (e.g., 2x2 → 12x12 tokens)
   - Project to LLM hidden size
   Code: molmo/models/modeling_molmoe.py:1594-1709

3) Embedding fusion (prefill - embedding)
   - Text embeddings
   - Insert image features at positions from image_input_idx
   - Positional handling
   Code: molmo/models/modeling_molmoe.py:1899-1944

4) Transformer prefill
   - Run all decoder blocks
   - Build KV cache for decode
   Code: molmo/models/modeling_molmoe.py:1996-2007

5) Decode loop
   - One-step autoregressive generation with KV cache
   - Stop on EOS or max_new_tokens
   Code: molmo/models/modeling_molmoe.py:2428-2481 (generate)

6) Output decoding
   - Token IDs → text
   - Strip prompt tokens
   Code: experiments/base_experiment.py:862-871
```

## 2. Key stages

### 2.1 Input preprocessing
- Text tokenization to IDs.
- Image resize/crop per `max_crops`, patchify (patch size typically 14x14).
- Build `image_input_idx` indicating where vision features enter the sequence.

Outputs:
```python
{
    "input_ids": torch.Tensor,       # (batch, seq_len)
    "images": torch.Tensor,          # (batch, num_crops, num_patches, patch_dim)
    "image_masks": torch.Tensor,     # (batch, num_crops)
    "image_input_idx": torch.Tensor, # (batch, num_crops, num_patches)
}
```

### 2.2 Vision encoding
- ViT encodes patches.
- Pooling reduces spatial tokens (e.g., 24x24 → 12x12).
- Project to LLM hidden size.
- Timers: `T_vision_encoder`, `T_projector`, `T_vision_total`.

### 2.3 Embedding fusion
- Text embeddings from `wte`.
- Vision features inserted by index: `x[batch_idx[valid], image_input_idx[valid]] += image_features[valid]`.
- Optional CLS feature prepended if enabled.
- Apply embedding dropout; optional input scaling (`normalize_input_embeds`).

### 2.4 Transformer prefill
- Build causal mask/bias, optionally add attention mask.
- Run all decoder blocks; optionally collect KV cache for decode.
- A block = self-attention + MLP (or MoE) with residuals.

### 2.5 Decode loop
- Autoregressive generation over `max_new_tokens`.
- Use only the last token as input each step; reuse KV cache.
- Sample/argmax logits for next token; append to output.
- Stop early if EOS is emitted (when provided).

### 2.6 Output decoding
- Convert generated IDs to text; drop the prompt portion per sample.

## 3. Timing and metrics
- `T_vision_encoder`, `T_projector`, `T_vision_total`
- `T_prefill`
- `T_decode_per_token`
- `T_total`

Use these to profile bottlenecks and compare hardware.

## 4. Critical configs and tips
- `use_cache=True` is required for fast decode with KV cache.
- `max_new_tokens` bounds generation length.
- `eos_token_id` enables early stop; omit/disable for fixed-length benchmarks.
- `image_input_idx` controls where vision tokens join the sequence.
- `max_crops` controls how many image crops → vision token count.

## 5. Quick code map
- Preprocessing: `molmo/preprocessors/preprocessing_molmo.py`
- Model: `molmo/models/modeling_molmoe.py`
- Prefill/decode timers: `experiments/base_experiment.py`
- Generation config: `transformers.GenerationConfig`

## 6. FAQ
1. **Why image_input_idx?**  To align vision features with the text sequence positions.
2. **Why full-sequence prefill?**  Prefill computes hidden states for the prompt and builds KV cache for decoding.
3. **Why decode one token at a time?**  Autoregressive generation expands step by step, reusing KV cache.
4. **How to measure decode performance?**  Track `T_decode_per_token`; disable EOS to force fixed-length generation for benchmarking.
