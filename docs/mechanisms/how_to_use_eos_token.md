# How to use EOS Token for early stopping

## Overview

The EOS (End-of-Sequence) token enables early stopping during generation. When the model emits EOS, generation stops immediately even if `max_new_tokens` has not been reached.

## Code examples

### 1. Set EOS in `GenerationConfig`

```python
from transformers import GenerationConfig

# Get EOS token ID
eos_token_id = tokenizer.eos_token_id
if eos_token_id is None:
    eos_token_id = getattr(model.config, "eos_token_id", None)

# Get PAD token ID (optional but recommended to avoid warnings)
pad_token_id = tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = getattr(model.config, "pad_token_id", None)

# Build GenerationConfig
generation_config = GenerationConfig(
    max_new_tokens=128,
    use_cache=True,  # required for Molmo
    eos_token_id=eos_token_id,  # enable early stop
    pad_token_id=pad_token_id,  # avoid warnings
)

# Generate
outputs = model.generate(
    input_ids=input_ids,
    images=images,
    generation_config=generation_config,
)
```

### 2. Use in `BaseExperiment.measure_inference_latency`

`BaseExperiment` exposes `use_eos_token` to toggle EOS:

```python
# Enable EOS (default for normal tasks)
latency_results = experiment.measure_inference_latency(
    batch,
    max_new_tokens=128,
    use_eos_token=True,  # stop when EOS is generated
)

# Disable EOS (for perf studies, force fixed-length generation)
latency_results = experiment.measure_inference_latency(
    batch,
    max_new_tokens=128,
    use_eos_token=False,  # generate full max_new_tokens tokens
)
```

## Examples in the codebase

### Example 1: `exp1_accuracy.py` (regular VQA task)

```python
# experiments/profiling/knob1_tokens/exp1_accuracy.py:475-494

# Get EOS token ID
eos_token_id = self.tokenizer.eos_token_id
if eos_token_id is None:
    eos_token_id = getattr(self.model.config, "eos_token_id", None)

pad_token_id = self.tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = getattr(self.model.config, "pad_token_id", None)

# Build GenerationConfig with EOS enabled
generation_config = GenerationConfig(
    max_new_tokens=vqa_max_tokens,
    do_sample=False,
    use_cache=True,
    eos_token_id=eos_token_id,  # enable early stop
    pad_token_id=pad_token_id,
)
```

### Example 2: `exp4_language_tokens_vs_latency.py` (performance study)

```python
# experiments/motivate/exp4_language_tokens_vs_latency.py:81-87

# Disable EOS, force fixed-length generation
latency_results = self.measure_inference_latency(
    batch,
    max_new_tokens=max_tokens,
    measure_components=True,
    num_runs=1,
    return_output=True,
    use_eos_token=False,  # disable early stop to measure fixed decode length
)
```

## When to use EOS

### ✅ Use EOS when

1. **Normal VQA tasks**: stop naturally after the answer.
2. **Dialog tasks**: stop when the reply finishes.
3. **Any task needing natural termination**: avoid extra meaningless tokens.

### ❌ Avoid EOS when

1. **Performance studies** (e.g., exp4): need fixed token count to measure decode time.
2. **Benchmarks**: need exact token counts.
3. **Generation debugging**: want to observe full generation up to limit.

## Behavior comparison

### With EOS (`use_eos_token=True`)

```python
max_new_tokens = 128
# Model outputs: "down" (2 tokens) + EOS
# Actual new tokens: 3
# Time: Prefill + 3 × decode_per_token
```

### Without EOS (`use_eos_token=False`)

```python
max_new_tokens = 128
# Model outputs: "down" + 126 extra tokens (could be repetitive/noisy)
# Actual new tokens: 128
# Time: Prefill + 128 × decode_per_token
```

## Special notes for exp4

exp4 measures decode time at different token counts, so:

- **Set `use_eos_token=False`** to force full `max_new_tokens`.
- This makes per-token decode timing accurate (≈130 ms/token).
- Generated text may be nonsensical—that is expected for the experiment.

## Related code

- **BaseExperiment implementations**:
  - `experiments/base_experiment.py:305-612`
  - `experiments/motivate/base_experiment.py:264-571`
- **Examples**:
  - `experiments/profiling/knob1_tokens/exp1_accuracy.py:475-494`
  - `experiments/motivate/exp4_language_tokens_vs_latency.py:81-87`
- **Docs**:
  - `docs/mechanisms/max_new_tokens_knob.md`
