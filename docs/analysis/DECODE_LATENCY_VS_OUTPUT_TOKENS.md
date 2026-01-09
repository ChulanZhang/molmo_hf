# Decode Per-Token Latency vs Output Tokens Analysis

## Observation

From the output tokens distribution analysis, we observe that **decode per-token latency increases as the number of output tokens increases**:

| Output Tokens Range | Decode Per-Token Latency (ms/token) | Std Dev (ms/token) |
|---------------------|-------------------------------------|-------------------|
| 2 tokens            | 24.62                               | 3.40              |
| 3-5 tokens          | 34.42                               | 5.36              |
| 6-10 tokens         | 41.09                               | 5.77              |
| 11-20 tokens        | 41.47                               | 5.07              |
| 21+ tokens          | 45.17                               | 5.91              |

This is a **significant increase** (~83% from 2 tokens to 21+ tokens), which suggests that the decode latency is not constant per token as one might expect.

## Hypothesis: KV Cache Size Impact

The most likely explanation is that **KV (Key-Value) cache size grows with the number of generated tokens**, leading to:

1. **Increased Memory Access Time**: As the KV cache grows, each decode step needs to:
   - Read more key-value pairs from memory (growing linearly with sequence length)
   - Write new key-value pairs to memory
   - This increases memory bandwidth requirements

2. **Memory Bandwidth Bottleneck**: 
   - GPU memory bandwidth is finite
   - Longer sequences require more data to be transferred per decode step
   - This can become a bottleneck, especially for large models

3. **Cache Efficiency Degradation**:
   - Longer sequences may exceed GPU cache capacity
   - Cache misses increase, leading to slower memory access
   - Memory access patterns become less predictable

4. **Attention Computation Overhead**:
   - In attention mechanisms, the computation involves all previous tokens
   - Longer sequences mean more attention computations per token
   - Even with optimized implementations, there's overhead

## Mathematical Model

If we assume decode latency per token is a function of sequence length:

```
T_decode_per_token = T_base + T_kv_cache(L)
```

Where:
- `T_base`: Base computation time per token (constant)
- `T_kv_cache(L)`: KV cache access time as a function of sequence length L
- `L`: Current sequence length (prefill + generated tokens so far)

The KV cache access time typically grows with sequence length:
- **Linear growth**: `T_kv_cache(L) = α * L` (memory bandwidth limited)
- **Sub-linear growth**: `T_kv_cache(L) = α * log(L)` (cache effects)
- **Constant + linear**: `T_kv_cache(L) = β + α * L` (base overhead + linear growth)

From our data, we observe approximately **linear growth** in the early stages (2-10 tokens), then **slower growth** (10-21+ tokens), suggesting:
- Initial rapid growth due to KV cache initialization and memory bandwidth limits
- Later slower growth as cache effects stabilize or other bottlenecks dominate

## Expected Behavior

### Theoretical Expectation
In an ideal scenario with unlimited memory bandwidth and perfect caching:
- Decode per-token latency should be **constant** regardless of sequence length
- Each token generation should take the same time

### Observed Behavior
In practice, we observe:
- **Non-constant** decode per-token latency
- **Increasing** latency with sequence length
- **Variability** increases with sequence length (higher std dev)

## Implications for Latency Estimator

### Current Design
Our Latency Estimator predicts `T_decode_per_token` as a **single value** based on:
- `vision_tokens`
- `text_tokens`
- `tier_idx`
- `top_k`
- `num_active_blocks`

**But NOT** based on:
- `output_tokens` (unknown at inference time)

### Problem
The estimator predicts a **constant** decode per-token latency, but in reality:
- Decode per-token latency **varies** with the number of output tokens generated so far
- Early tokens (2-5) are faster (~25-35 ms/token)
- Later tokens (21+) are slower (~45 ms/token)

### Solutions

#### Option 1: Predict Average Decode Per-Token Latency
- Predict the **average** decode per-token latency across all output token positions
- Use this for budget checking with a `safety_factor`
- **Pros**: Simple, works with current design
- **Cons**: Less accurate for very short or very long outputs

#### Option 2: Predict Decode Per-Token Latency as Function of Position
- Predict decode per-token latency as a function of token position
- Model: `T_decode_per_token(pos) = f(vision_tokens, text_tokens, tier_idx, top_k, num_active_blocks, pos)`
- **Pros**: More accurate
- **Cons**: More complex, requires position information

#### Option 3: Use Conservative Estimate
- Use the **maximum** observed decode per-token latency (e.g., 45 ms/token for 21+ tokens)
- Apply this conservatively for all outputs
- **Pros**: Ensures budget adherence
- **Cons**: May be too conservative for short outputs

#### Option 4: Hybrid Approach (Recommended)
- Predict **two values**:
  1. `T_decode_per_token_early`: For first 2-5 tokens (faster)
  2. `T_decode_per_token_late`: For tokens after 5 (slower)
- Use weighted average based on expected output length distribution
- **Pros**: Balances accuracy and simplicity
- **Cons**: Still requires some estimate of output length

## Verification Methods

To verify that KV cache size is the primary cause:

1. **Profile Memory Access**:
   - Use NVIDIA Nsight Compute to profile memory bandwidth
   - Compare memory bandwidth usage for short vs. long sequences
   - Check if memory bandwidth saturates for longer sequences

2. **Measure KV Cache Size**:
   - Calculate actual KV cache size: `KV_cache_size = 2 * num_layers * hidden_size * sequence_length * dtype_size`
   - Plot decode latency vs. KV cache size
   - Check for correlation

3. **Compare Different Sequence Lengths**:
   - Measure decode latency for fixed output tokens but varying prefill lengths
   - If latency increases with prefill length, KV cache is likely the cause

4. **Model Architecture Analysis**:
   - Check if the model uses efficient attention mechanisms (e.g., Flash Attention)
   - Verify KV cache implementation efficiency

## Current Recommendation

Given the observed behavior:

1. **Keep current design** (predict single `T_decode_per_token`)
2. **Use conservative estimate** with `safety_factor`:
   - Use higher estimate (e.g., 45 ms/token) for budget checking
   - Apply `safety_factor = 1.2-1.3` to account for variability
3. **Document the limitation**: Decode per-token latency increases with output length
4. **Future improvement**: Consider Option 4 (hybrid approach) if more accuracy is needed

## References

- Transformer KV cache: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Memory bandwidth analysis: GPU memory bandwidth specifications
- Flash Attention: [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)


