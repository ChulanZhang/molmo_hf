# Vision Tokens Control Knob: Tier-Based Design Discussion

## Current Approach: Fixed Vision Token Targets

### Current Implementation

The current system uses **fixed vision token targets** (e.g., `[432, 720, 1008, 1440]`), which map to specific crop counts:

| Vision Tokens | Crops | Possible Tilings | Aspect Ratio Coverage |
|---------------|-------|------------------|----------------------|
| 432 | 2 | (1,2), (2,1) | Wide/Tall only |
| 720 | 4 | (1,4), (2,2), (4,1) | Wide/Square/Tall |
| 1008 | 6 | (1,6), (2,3), (3,2), (6,1) | Excellent coverage |
| 1440 | 9 | (1,9), (3,3), (9,1) | Limited (square only) |

### Problems with Fixed Targets

#### 1. **Perfect Square Crop Counts Have Limited Tiling Options**

**Example: 1440 tokens (9 crops)**
```
Possible tilings: (1,9), (3,3), (9,1)
- (1,9): aspect = 9.0 (extremely wide)
- (3,3): aspect = 1.0 (square only)
- (9,1): aspect = 0.11 (extremely tall)

For a wide image (aspect 1.33):
- Best match: (3,3) with aspect 1.0
- Aspect ratio mismatch: |1.0 - 1.33| = 0.33
- Result: Significant distortion
```

**Example: 720 tokens (4 crops)**
```
Possible tilings: (1,4), (2,2), (4,1)
- (1,4): aspect = 4.0 (very wide)
- (2,2): aspect = 1.0 (square)
- (4,1): aspect = 0.25 (very tall)

For a moderately wide image (aspect 1.5):
- Best match: (2,2) with aspect 1.0
- Aspect ratio mismatch: |1.0 - 1.5| = 0.5
- Result: Significant distortion
```

#### 2. **Non-Square Crop Counts Provide Better Coverage**

**Example: 1008 tokens (6 crops)**
```
Possible tilings: (1,6), (2,3), (3,2), (6,1)
- (1,6): aspect = 6.0 (very wide)
- (2,3): aspect = 1.5 (wide)
- (3,2): aspect = 0.67 (tall)
- (6,1): aspect = 0.17 (very tall)

For a wide image (aspect 1.33):
- Best match: (2,3) with aspect 1.5
- Aspect ratio mismatch: |1.5 - 1.33| = 0.17
- Result: Good match, minimal distortion ✓
```

**Insight**: Non-square crop counts (6, 8, 10, 12) provide better aspect ratio coverage than perfect squares (4, 9, 16).

## Proposed Approach: Tier-Based Design

### Concept

Instead of fixing exact vision token targets, define **tiers** (ranges) that allow adaptive crop selection within each tier:

```
Tier 1 (Low):    1-3 crops   → 288-576 tokens
Tier 2 (Medium): 4-8 crops   → 720-1296 tokens
Tier 3 (High):   9-15 crops   → 1440-2304 tokens
```

### How It Would Work

For each image and tier:
1. **Calculate tier's crop range**: e.g., Tier 2 = 4-8 crops
2. **For each possible crop count in range**: Find best tiling for image's aspect ratio
3. **Select crop count with best aspect ratio match**: e.g., wide image might use 6 crops (2×3) instead of 4 crops (2×2)
4. **Calculate actual vision tokens**: `(selected_crops + 1) × 144`

### Example: Wide Image (aspect 1.33) in Tier 2

**Tier 2: 4-8 crops**

```
Crop count = 4:
  Tilings: (1,4), (2,2), (4,1)
  Best: (2,2) with aspect 1.0
  Mismatch: |1.0 - 1.33| = 0.33

Crop count = 5:
  Tilings: (1,5), (5,1)
  Best: (1,5) with aspect 5.0
  Mismatch: |5.0 - 1.33| = 3.67

Crop count = 6:
  Tilings: (1,6), (2,3), (3,2), (6,1)
  Best: (2,3) with aspect 1.5
  Mismatch: |1.5 - 1.33| = 0.17 ✓ (best!)

Crop count = 7:
  Tilings: (1,7), (7,1)
  Best: (1,7) with aspect 7.0
  Mismatch: |7.0 - 1.33| = 5.67

Crop count = 8:
  Tilings: (1,8), (2,4), (4,2), (8,1)
  Best: (2,4) with aspect 2.0
  Mismatch: |2.0 - 1.33| = 0.67

Selected: 6 crops (best aspect ratio match)
Actual vision tokens: (6 + 1) × 144 = 1008
```

**Result**: Within Tier 2, the system selects 6 crops instead of 4, achieving better aspect ratio match (0.17 vs 0.33 mismatch).

## Detailed Tier Design Options

### Option 1: Coarse Tiers (3 tiers)

```
Tier 1 (Low):    1-3 crops   → 288-576 tokens   (small images)
Tier 2 (Medium): 4-8 crops   → 720-1296 tokens  (medium images)
Tier 3 (High):   9-15 crops  → 1440-2304 tokens (large images)
```

**Pros**:
- Simple, easy to understand
- Clear separation between small/medium/large
- Good coverage within each tier

**Cons**:
- Large range in Tier 3 (9-15 crops = 6 different crop counts)
- May have significant vision token variation within tier

### Option 2: Fine Tiers (5-6 tiers)

```
Tier 1 (Very Low):  1-2 crops   → 288-432 tokens
Tier 2 (Low):       3-4 crops   → 576-720 tokens
Tier 3 (Medium):    5-7 crops   → 864-1152 tokens
Tier 4 (High):      8-10 crops  → 1296-1584 tokens
Tier 5 (Very High): 11-15 crops → 1728-2304 tokens
```

**Pros**:
- More granular control
- Smaller vision token variation within each tier
- Better for fine-grained experiments

**Cons**:
- More complex configuration
- More tiers to test in experiments

### Option 3: Aspect-Ratio-Aware Tiers

Instead of fixed crop ranges, define tiers based on **aspect ratio coverage**:

```
Tier 1: 2-3 crops   → Covers wide/tall (aspect 0.5-2.0)
Tier 2: 4-6 crops   → Covers wide/square/tall (aspect 0.33-3.0)
Tier 3: 7-10 crops  → Covers wide/square/tall (aspect 0.3-3.3)
Tier 4: 11-15 crops  → Covers wide/square/tall (aspect 0.27-3.7)
```

**Pros**:
- Explicitly considers aspect ratio coverage
- Ensures each tier can handle various image types

**Cons**:
- More complex to define and understand
- May not align with natural "small/medium/large" intuition

## Comparison: Fixed Targets vs Tier-Based

### Fixed Targets (Current)

**Configuration**:
```python
vision_tokens_list = [432, 720, 1008, 1440]
```

**For each target**:
- Calculate exact crop count: `num_crops = (target // 144) - 1`
- Force exact crop count: `exact_num_crops = num_crops`
- Select best tiling from limited options

**Pros**:
- ✅ Precise control over vision tokens
- ✅ Consistent vision token count across images
- ✅ Easy to compare results (same vision tokens)
- ✅ Simple configuration

**Cons**:
- ❌ Limited tiling options for perfect square crop counts (4, 9, 16)
- ❌ May cause aspect ratio mismatch for some images
- ❌ Cannot adapt crop count to image aspect ratio

### Tier-Based (Proposed)

**Configuration**:
```python
vision_token_tiers = [
    {"name": "low", "min_crops": 1, "max_crops": 3},
    {"name": "medium", "min_crops": 4, "max_crops": 8},
    {"name": "high", "min_crops": 9, "max_crops": 15},
]
```

**For each tier**:
- Try all crop counts in range: `[min_crops, min_crops+1, ..., max_crops]`
- For each crop count, find best tiling for image's aspect ratio
- Select crop count with best aspect ratio match
- Calculate actual vision tokens: `(selected_crops + 1) × 144`

**Pros**:
- ✅ Better aspect ratio matching (adaptive crop selection)
- ✅ Avoids limitations of perfect square crop counts
- ✅ More flexible, can optimize per image
- ✅ Natural "small/medium/large" categorization

**Cons**:
- ❌ Vision token count varies within tier (harder to compare)
- ❌ More complex implementation
- ❌ More complex configuration
- ❌ Results may be harder to interpret (variable vision tokens)

## Hybrid Approach: Tier with Preferred Crop Counts

### Concept

Define tiers, but **prefer specific crop counts** within each tier:

```
Tier 1 (Low):    1-3 crops, prefer [2, 3]
Tier 2 (Medium): 4-8 crops, prefer [4, 6, 8]
Tier 3 (High):   9-15 crops, prefer [9, 12, 15]
```

**Selection strategy**:
1. First, try preferred crop counts
2. If aspect ratio match is poor (mismatch > threshold), try other crop counts in tier
3. Select crop count with best aspect ratio match

**Example**:
```
Tier 2: 4-8 crops, prefer [4, 6, 8]

Wide image (aspect 1.33):
  Try 4 crops: best tiling (2,2), mismatch = 0.33
  Try 6 crops: best tiling (2,3), mismatch = 0.17 ✓ (best!)
  Try 8 crops: best tiling (2,4), mismatch = 0.67
  
  Selected: 6 crops (preferred, best match)
```

**Pros**:
- ✅ Combines benefits of both approaches
- ✅ Prefers crop counts with good tiling options (non-squares)
- ✅ Falls back to other crop counts if needed
- ✅ More predictable (tends to use preferred counts)

**Cons**:
- ❌ Still has some vision token variation
- ❌ More complex than fixed targets

## Implementation Considerations

### 1. Result Recording

**Current (Fixed Targets)**:
```json
{
  "target_vision_tokens": 1008,
  "target_crops": 6,
  "actual_vision_tokens": 1002,
  ...
}
```

**Tier-Based**:
```json
{
  "tier": "medium",
  "tier_range": {"min_crops": 4, "max_crops": 8},
  "selected_crops": 6,
  "selected_vision_tokens": 1008,
  "actual_vision_tokens": 1002,
  "selection_reason": "best_aspect_ratio_match",
  "aspect_ratio_mismatch": 0.17,
  ...
}
```

### 2. Filename Generation

**Current**:
```
coco-2014-vqa_visiontoken1008_topk8_blocks14.json
```

**Tier-Based Options**:
```
# Option A: Use tier name
coco-2014-vqa_imgsizetier-medium_topk8_blocks14.json

# Option B: Use selected crops
coco-2014-vqa_imgsizecrops6_topk8_blocks14.json

# Option C: Use tier + selected crops
coco-2014-vqa_imgsizetier-medium-crops6_topk8_blocks14.json
```

### 3. Experiment Design

**Current**: Test 4 fixed targets = 4 configurations

**Tier-Based**: Test 3 tiers, but each tier may produce different crop counts for different images
- Need to aggregate results by tier
- May need to analyze distribution of crop counts within tier

### 4. Comparison Across Configurations

**Current**: Easy to compare - same vision tokens across all images

**Tier-Based**: Harder to compare - vision tokens vary within tier
- Need to compare by tier (aggregate statistics)
- May need to analyze crop count distribution
- May need to compare "typical" vision tokens per tier

## Recommendation: Hybrid Approach with Smart Defaults

### Proposed Design

```python
vision_token_tiers = [
    {
        "name": "low",
        "min_crops": 1,
        "max_crops": 3,
        "preferred_crops": [2, 3],  # Avoid 1 crop (limited tiling options)
        "typical_vision_tokens": 432,  # For comparison/reference
    },
    {
        "name": "medium",
        "min_crops": 4,
        "max_crops": 8,
        "preferred_crops": [4, 6, 8],  # Avoid 5, 7 (prime numbers, limited tilings)
        "typical_vision_tokens": 1008,  # For comparison/reference
    },
    {
        "name": "high",
        "min_crops": 9,
        "max_crops": 15,
        "preferred_crops": [9, 12, 15],  # Avoid primes, prefer composites
        "typical_vision_tokens": 1872,  # For comparison/reference
    },
]
```

**Selection Algorithm**:
```python
def select_crops_for_tier(tier, image_aspect_ratio):
    best_crops = None
    best_mismatch = float('inf')
    
    # First, try preferred crop counts
    for crops in tier["preferred_crops"]:
        tilings = get_all_tilings(crops)
        best_tiling, mismatch = find_best_tiling(tilings, image_aspect_ratio)
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_crops = crops
    
    # If mismatch is acceptable (< 0.3), use preferred crop
    if best_mismatch < 0.3:
        return best_crops
    
    # Otherwise, try all crop counts in tier
    for crops in range(tier["min_crops"], tier["max_crops"] + 1):
        if crops in tier["preferred_crops"]:
            continue  # Already tried
        tilings = get_all_tilings(crops)
        best_tiling, mismatch = find_best_tiling(tilings, image_aspect_ratio)
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_crops = crops
    
    return best_crops
```

**Benefits**:
- ✅ Prefers crop counts with good tiling options (non-primes, composites)
- ✅ Falls back to other crop counts if aspect ratio match is poor
- ✅ More predictable (tends to use preferred counts)
- ✅ Better aspect ratio matching than fixed targets
- ✅ Still allows comparison via "typical_vision_tokens"

## Open Questions

1. **How much vision token variation is acceptable?**
   - Tier 2 (4-8 crops): 720-1296 tokens (576 token range, ~80% variation)
   - Is this acceptable for experiments?

2. **How to compare results across tiers?**
   - Compare by "typical_vision_tokens"?
   - Compare by actual vision token distribution?
   - Compare by crop count distribution?

3. **Should we allow tier overlap?**
   - Current: Non-overlapping tiers
   - Alternative: Overlapping tiers (e.g., 3-5, 4-8, 7-12)

4. **How to handle edge cases?**
   - Very small images: May need to force minimum crops
   - Very large images: May need to force maximum crops
   - Extreme aspect ratios: May need special handling

5. **Backward compatibility?**
   - Can we support both fixed targets and tier-based?
   - How to migrate existing experiments?

## Conclusion

The tier-based approach offers **better aspect ratio matching** by allowing adaptive crop selection within each tier. However, it introduces **complexity** in result comparison and experiment design.

**Recommendation**: 
- **Short term**: Keep fixed targets, but **avoid perfect square crop counts** (4, 9, 16)
  - Use: `[432, 720, 1008, 1296, 1584]` (2, 4, 6, 8, 10 crops)
  - Better tiling options, still simple and comparable

- **Long term**: Consider hybrid approach with smart defaults
  - Define tiers with preferred crop counts
  - Prefer non-prime, composite numbers (better tiling options)
  - Fall back to other crop counts if aspect ratio match is poor
  - Record both tier and selected crops in results

This provides a good balance between **flexibility** (better aspect ratio matching) and **simplicity** (predictable, comparable results).

