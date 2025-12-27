# Vision Tokens Control Knob: Q&A

## Q1: Why are some result files still using "imgsize" naming instead of "visiontoken"?

### Answer

The result files with `imgsize560x336` naming are **old experimental results** generated using the `image_size_list` mode (legacy approach). The current code correctly uses `visiontoken` naming when `vision_tokens_list` is used.

### Evidence

**Old files (image_size_list mode)**:
- Filename: `coco-2014-vqa_imgsize560x336_topk8_blocks14.json`
- Content: Has `target_image_size` field
- Generated when: `--image_size_list` was used

**New files (vision_tokens_list mode)**:
- Filename: `coco-2014-vqa_visiontoken432_topk8_blocks14.json`
- Content: Has `target_crops` field (no `target_image_size`)
- Generated when: `--vision_tokens_list` is used (current default)

### Code Logic

The filename generation correctly detects the mode:

```python
# In acc_lat_profiling.py:1478
use_image_size_list=bool(image_size_list)  # True only if image_size_list is provided

# In _generate_config_filename:348-360
if use_image_size_list:
    # Use theoretical_image_size → "imgsize560x336"
else:
    # Use target_vision_tokens → "visiontoken432"
```

**Conclusion**: The naming is correct. The `imgsize*` files are old results from previous experiments using `image_size_list`. New experiments using `vision_tokens_list` correctly generate `visiontoken*` filenames.

## Q2: Is tier-based design better than fixed vision token targets?

### Short Answer

**Tier-based design has advantages** (better aspect ratio matching) but also **disadvantages** (harder to compare results, more complex). 

**Recommendation**: 
- **Short term**: Keep fixed targets but **avoid perfect square crop counts** (4, 9, 16)
- **Long term**: Consider hybrid tier-based approach if aspect ratio matching is critical

### Detailed Analysis

#### Current Problem: Perfect Square Crop Counts

**The Issue**:
- 1440 tokens = 9 crops → only 3 tilings: (1,9), (3,3), (9,1)
- For wide images (aspect 1.33), forced to use (3,3) with aspect 1.0
- **Result**: Significant aspect ratio mismatch (0.33), causing distortion

**Why This Happens**:
- Perfect squares (4, 9, 16) have fewer factorizations
- Fewer factorizations = fewer tiling options
- Fewer tiling options = worse aspect ratio matching

#### Solution 1: Avoid Perfect Squares (Immediate Fix)

**Change**: Use composite numbers instead of perfect squares

**Current**: `[432, 720, 1008, 1440]` → crops: [2, 4, 6, 9]
- Problem: 4 and 9 are perfect squares

**Recommended**: `[432, 720, 1008, 1296, 1584]` → crops: [2, 4, 6, 8, 10]
- Better: All composite numbers, more tiling options
- 6 crops: 4 tilings (excellent coverage)
- 8 crops: 4 tilings (good coverage)
- 10 crops: 4 tilings (good coverage)

**Benefits**:
- ✅ Simple (still fixed targets)
- ✅ Comparable results (same vision tokens)
- ✅ Better aspect ratio coverage
- ✅ No code changes needed (just change values)

**Trade-off**: Still fixed, cannot adapt crop count per image

#### Solution 2: Tier-Based Design (Long-Term)

**Concept**: Define tiers (ranges) that allow adaptive crop selection

**Example**:
```
Tier 1 (Low):    1-3 crops   → 288-576 tokens
Tier 2 (Medium): 4-8 crops   → 720-1296 tokens
Tier 3 (High):   9-15 crops  → 1440-2304 tokens
```

**How It Works**:
1. For each image and tier, try all crop counts in range
2. For each crop count, find best tiling for image's aspect ratio
3. Select crop count with best aspect ratio match
4. Calculate actual vision tokens: `(selected_crops + 1) × 144`

**Example: Wide Image (aspect 1.33) in Tier 2**

```
Tier 2: 4-8 crops

Try 4 crops: best tiling (2,2), mismatch = 0.33
Try 5 crops: best tiling (1,5), mismatch = 3.67
Try 6 crops: best tiling (2,3), mismatch = 0.17 ✓ (best!)
Try 7 crops: best tiling (1,7), mismatch = 5.67
Try 8 crops: best tiling (2,4), mismatch = 0.67

Selected: 6 crops (best aspect ratio match)
Actual vision tokens: 1008
```

**Benefits**:
- ✅ Better aspect ratio matching (adaptive selection)
- ✅ Avoids limitations of perfect square crop counts
- ✅ More flexible, can optimize per image

**Disadvantages**:
- ❌ Vision token count varies within tier (harder to compare)
- ❌ More complex implementation
- ❌ More complex configuration
- ❌ Results harder to interpret (variable vision tokens)

#### Solution 3: Hybrid Approach (Recommended Long-Term)

**Concept**: Tiers with **preferred crop counts** (non-primes, composites)

```
Tier 1 (Low):    1-3 crops, prefer [2, 3]
Tier 2 (Medium): 4-8 crops, prefer [4, 6, 8]
Tier 3 (High):   9-15 crops, prefer [9, 12, 15]
```

**Selection Strategy**:
1. First try preferred crop counts (better tiling options)
2. If aspect ratio match is poor (mismatch > 0.3), try other crop counts
3. Select crop count with best aspect ratio match

**Benefits**:
- ✅ Combines benefits of both approaches
- ✅ Prefers crop counts with good tiling options
- ✅ Falls back to other crop counts if needed
- ✅ More predictable (tends to use preferred counts)

**Trade-off**: Still has some vision token variation, but more predictable

### Comparison Table

| Aspect | Fixed Targets | Tier-Based | Hybrid Tier |
|--------|---------------|------------|-------------|
| **Aspect Ratio Matching** | Limited (squares) | Better | Good |
| **Vision Token Consistency** | High | Low | Medium |
| **Result Comparability** | Easy | Harder | Medium |
| **Configuration Complexity** | Simple | Complex | Medium |
| **Implementation Complexity** | Simple | Complex | Medium |

### Recommendation

**Immediate (No Code Changes)**:
- Change `VISION_TOKENS_LIST` from `[432, 720, 1008, 1440]` to `[432, 720, 1008, 1296, 1584]`
- This avoids perfect squares (4, 9) and uses composites (2, 4, 6, 8, 10)
- Better aspect ratio coverage with zero code changes

**Short Term (Minor Code Changes)**:
- Keep fixed targets but add validation to warn about perfect squares
- Document recommended values in scripts

**Long Term (If Aspect Ratio Matching is Critical)**:
- Implement hybrid tier-based approach
- Define tiers with preferred crop counts
- Record both tier and selected crops in results
- Provide comparison tools for tier-based results

### Implementation Considerations

#### Result Recording

**Current (Fixed)**:
```json
{
  "target_vision_tokens": 1008,
  "target_crops": 6,
  "actual_vision_tokens": 1002
}
```

**Tier-Based**:
```json
{
  "tier": "medium",
  "tier_range": {"min_crops": 4, "max_crops": 8},
  "selected_crops": 6,
  "selected_vision_tokens": 1008,
  "aspect_ratio_mismatch": 0.17,
  "selection_reason": "best_aspect_ratio_match"
}
```

#### Filename Options

**Option A**: Use tier name
```
coco-2014-vqa_imgsizetier-medium_topk8_blocks14.json
```

**Option B**: Use selected crops
```
coco-2014-vqa_imgsizecrops6_topk8_blocks14.json
```

**Option C**: Use tier + crops (most informative)
```
coco-2014-vqa_imgsizetier-medium-crops6_topk8_blocks14.json
```

### Open Questions

1. **Vision token variation**: Is 576 token range (720-1296) acceptable for Tier 2?
   - **Answer**: Depends on experiment goals. For accuracy-latency tradeoff analysis, some variation may be acceptable if we aggregate by tier.

2. **Comparison method**: How to compare results across tiers?
   - **Answer**: Compare by "typical_vision_tokens" (reference value) or aggregate statistics (mean, median) within tier.

3. **Backward compatibility**: Can we support both fixed targets and tier-based?
   - **Answer**: Yes, add a flag `--use_tier_based` to switch between modes.

4. **Edge cases**: How to handle very small/large images or extreme aspect ratios?
   - **Answer**: Add constraints: minimum crops for small images, maximum crops for large images, special handling for extreme aspect ratios.

## Conclusion

**For Q1 (Naming)**: The `imgsize*` files are old results. Current code correctly uses `visiontoken*` naming.

**For Q2 (Tier Design)**: 
- **Immediate**: Avoid perfect squares in fixed targets (use 2, 4, 6, 8, 10 instead of 2, 4, 6, 9)
- **Long-term**: Consider hybrid tier-based approach if better aspect ratio matching is critical

The tier-based design is **better for aspect ratio matching** but **more complex for experiments**. The hybrid approach provides a good balance.

