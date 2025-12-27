# `max_new_tokens` knob

## Overview
`max_new_tokens` caps how many new tokens the model may generate (excluding the prompt). This doc explains its behavior in Molmo, defaults, early stopping, VQA considerations, and performance impact.

## 1. Definition and defaults
- Specifies the maximum number of new tokens to generate.
- Molmo requires it to be set explicitly via `GenerationConfig`; there is no built-in default.
- Transformers fallback: if neither `max_new_tokens` nor `max_length` is set, `max_length=20` applies, but Molmo asserts that `max_new_tokens` is provided.

Example:
```python
from transformers import GenerationConfig

eos_id = tokenizer.eos_token_id or getattr(model.config, "eos_token_id", None)
pad_id = tokenizer.pad_token_id or getattr(model.config, "pad_token_id", None)

gen_cfg = GenerationConfig(
    max_new_tokens=128,  # required
    do_sample=False,
    use_cache=True,      # required for Molmo
    eos_token_id=eos_id, # enables early stop
    pad_token_id=pad_id, # avoid warnings
)
outputs = model.generate(input_ids=input_ids, generation_config=gen_cfg)
```

## 2. Generation behavior
- Generation may stop **before** reaching `max_new_tokens` due to:
  1) EOS token emission
  2) stop strings
  3) other stopping criteria (e.g., `max_length` if used)
- Batch generation stops per sequence; shorter sequences stop first.
- Actual generated length ≤ `max_new_tokens`.

### Early stopping internals
Molmo calls `super().generate(...)`; Transformers checks `eos_token_id` each step and applies `StoppingCriteria`. You can add custom criteria as needed.

### Short answers are not truncated
If the answer is short (e.g., 5 tokens), the output contains exactly those generated tokens (plus EOS if emitted); no extra truncation occurs.

## 3. VQA-specific guidance
- VQA answers are typically very short (1–10 tokens).
- A safe setting is `max_new_tokens=128`: covers all cases while allowing early stop on EOS.
- Typical VQA token length distribution (val): ~60% (1–5), ~30% (6–10), ~8% (11–20), ~2% (21+).

## 4. Handling generated tokens in code
- Outputs include prompt + generated tokens. Slice off the prompt length to get only generated tokens.
- Postprocess text (e.g., split on "Answer:" or last line) to extract final answers.

## 5. Performance impact
- Latency cost scales with **actual** generated tokens, not the `max_new_tokens` cap.
- Total time ≈ `T_prefill` + (#generated_tokens × `T_decode_per_token`).
- Longer caps do not hurt if EOS triggers early, but very large caps can mask config mistakes; choose reasonable bounds per dataset.

## 6. Best practices
- Always set `max_new_tokens` explicitly.
- Provide `eos_token_id` and `pad_token_id` to avoid warnings and enable early stop.
- Use smaller caps for short-answer datasets to save decode time; keep larger caps for long-form tasks.
- For fixed-length benchmarking, disable EOS/stop strings so the model generates the full `max_new_tokens`.
