# Why Set `max_crops = num_crops` When Using `exact_num_crops`?

## Question

When using `exact_num_crops` to force selection of exactly `num_crops` crops, why do we also need to set `max_crops = num_crops`? 

## Answer

### Technical Analysis

Looking at the `select_tiling` function implementation:

```python
def select_tiling(h, w, patch_size, max_num_crops, exact_num_crops=None):
    if exact_num_crops is not None:
        # When exact_num_crops is set, it generates candidates independently
        exact_tilings = []
        for i in range(1, exact_num_crops + 1):  # Uses exact_num_crops, NOT max_num_crops
            if exact_num_crops % i == 0:
                j = exact_num_crops // i
                exact_tilings.append((i, j))
        # ... selects best tiling from exact_tilings
        return best_tiling
```

**Key observation**: When `exact_num_crops` is set, `select_tiling` **completely ignores** `max_num_crops` and generates candidates based solely on `exact_num_crops`.

### So Why Set `max_crops = num_crops`?

Even though `max_num_crops` is ignored when `exact_num_crops` is set, setting `max_crops = num_crops` serves several important purposes:

#### 1. **Consistency and Clarity**
- Makes the code intent explicit: "We want exactly `num_crops` crops"
- Ensures `max_crops` and `exact_num_crops` are aligned
- Prevents confusion if someone reads the code later

#### 2. **Fallback Protection**
- If `exact_num_crops` is not properly passed or used in some code path, `max_crops` still acts as an upper bound
- Provides a safety net in case of bugs or code changes

#### 3. **Model Configuration Consistency**
- `model.config.max_crops` is set via `_set_max_crops(max_crops)`
- This value may be used elsewhere in the codebase (e.g., for memory allocation, shape inference)
- Setting it to `num_crops` ensures the model config reflects the actual crop count being used

#### 4. **Fallback Estimation** (in `_calculate_vision_tokens`)
```python
# In combined_profiling.py
if "image_input_idx" not in batch:
    # Fallback: estimate from max_crops
    max_crops = self.model.config.max_crops
    num_vision_tokens = (max_crops + 1) * 144
```
- If `image_input_idx` is unavailable, the code falls back to estimating from `max_crops`
- Setting `max_crops = num_crops` ensures this fallback gives the correct estimate

### Conclusion

**Technically**: `max_crops` is not strictly necessary when `exact_num_crops` is set, because `select_tiling` ignores it.

**Practically**: Setting `max_crops = num_crops` is still **recommended** because:
- It ensures consistency across the codebase
- It provides fallback protection
- It makes the code intent clear
- It ensures model config reflects actual usage

**Best Practice**: Always set both `max_crops = num_crops` and `exact_num_crops = num_crops` when you want precise crop count control.

