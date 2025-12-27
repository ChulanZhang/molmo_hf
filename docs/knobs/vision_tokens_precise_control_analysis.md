# Vision tokens: precise control and image-size knob

## Goals
- Control vision token counts precisely while retaining overlap benefits.
- Prefer image size as the primary knob; derive tiling/num_crops/vision_tokens from resolution.
- Document mappings and behaviors for both `force_full_tokens` on/off.

## Using image size as the knob
- Provide `--image_size_list` (e.g., `560x336 560x784 784x784`) to experiments.
- Each size maps to tiling via `image_size_to_tiling`, num_crops = h_tiles × w_tiles, and target vision tokens = (num_crops + 1 global) × 144.
- Scripts updated: `run_combined_profiling.sh`, `run_multi_datasets_a100.sh`, `run_multi_datasets_h100.sh`.

### Reference sizes (example)
- 560x336 → tiling 2x1 → 2 crops + global → 432 tokens
- 560x784 → tiling 2x2 → 4 crops + global → 720 tokens
- 784x784 → tiling 3x3 → 9 crops + global → 1440 tokens
(Adjust according to your tiling function.)

## `force_full_tokens`
- `False` (default): padding tokens may be dropped; actual tokens can be lower than theoretical (padding and pooling effects).
- `True`: padding treated as valid tokens; actual tokens match theoretical. Used for precise control and analysis.
- Implementation: `MultiModalPreprocessor(force_full_tokens=True)` ensures padded regions get valid indices and masks.

## Theoretical vs actual values
- Theoretical (from target size/tiling): `theoretical_num_crops`, `theoretical_tiling`, `theoretical_vision_tokens`.
- Actual (measured via hooks): `actual_num_crops`, `actual_tiling`, `actual_vision_tokens`.
- When `force_full_tokens=True` and a mismatch is detected, actual values are overridden to match theoretical for consistency in logs/outputs.

## Logging/output expectations
- Primary knob logged as `image_size` with tiling, num_crops, target_vision_tokens.
- Secondary info (actual_* metrics: vision_tokens_mean, num_crops, max_crops) is kept from runtime hooks.
- Results are aggregated per knob combination (not per rank).

## Legacy vision-token knob
- `vision_tokens_list` is kept for compatibility when `image_size_list` is absent, but the recommended path is to drive experiments by image size.

## Notes
- Overlap and pooling can reduce usable tokens when `force_full_tokens=False`; this is expected.
- To benchmark fixed token counts, set `force_full_tokens=True` and choose image sizes that map cleanly to the desired tiling.
- See code: `experiments/core_exp/combined_profiling.py`, `molmo/data/model_preprocessor.py` for the preprocessing and counting logic.
