# Vision Tokens Control Knob: Tier-Based Design Summary

## Quick Answer

**Current Problem**: Fixed vision token targets (e.g., 1440 = 9 crops) limit tiling options, especially for perfect square crop counts (4, 9, 16).

**Proposed Solution**: Tier-based design allows adaptive crop selection within each tier, improving aspect ratio matching.

**Recommendation**: 
- **Short term**: Keep fixed targets but **avoid perfect squares** (use 2, 4, 6, 8, 10, 12 crops instead of 4, 9, 16)
- **Long term**: Consider hybrid tier-based approach with preferred crop counts

## The Core Issue

### Perfect Square Crop Counts Have Limited Tiling Options

**Example: 9 crops (1440 tokens)**
- Only 3 tilings: (1,9), (3,3), (9,1)
- For wide images (aspect 1.33), best match is (3,3) with aspect 1.0
- **Mismatch**: 0.33 (significant distortion)

**Example: 4 crops (720 tokens)**
- Only 3 tilings: (1,4), (2,2), (4,1)
- For moderately wide images (aspect 1.5), best match is (2,2) with aspect 1.0
- **Mismatch**: 0.5 (significant distortion)

### Non-Square Crop Counts Provide Better Coverage

**Example: 6 crops (1008 tokens)**
- 4 tilings: (1,6), (2,3), (3,2), (6,1)
- For wide images (aspect 1.33), best match is (2,3) with aspect 1.5
- **Mismatch**: 0.17 (good match, minimal distortion) ✓

**Insight**: Composite numbers (6, 8, 10, 12) have more factorizations, providing better aspect ratio coverage.

## Tier-Based Design Options

### Option 1: Simple Tiers (3 tiers)

```
Tier 1 (Low):    1-3 crops   → 288-576 tokens
Tier 2 (Medium): 4-8 crops   → 720-1296 tokens
Tier 3 (High):   9-15 crops  → 1440-2304 tokens
```

**Selection**: For each image, try all crop counts in tier, select best aspect ratio match.

**Pros**: Simple, natural categorization
**Cons**: Large vision token variation within tier (harder to compare)

### Option 2: Hybrid with Preferred Crop Counts (Recommended)

```
Tier 1 (Low):    1-3 crops, prefer [2, 3]
Tier 2 (Medium): 4-8 crops, prefer [4, 6, 8]
Tier 3 (High):   9-15 crops, prefer [9, 12, 15]
```

**Selection Strategy**:
1. First try preferred crop counts (non-primes, composites)
2. If aspect ratio match is poor (mismatch > 0.3), try other crop counts in tier
3. Select crop count with best aspect ratio match

**Benefits**:
- ✅ Prefers crop counts with good tiling options
- ✅ Falls back to other crop counts if needed
- ✅ More predictable (tends to use preferred counts)
- ✅ Better aspect ratio matching than fixed targets

## Comparison Table

| Aspect | Fixed Targets | Tier-Based |
|--------|---------------|------------|
| **Aspect Ratio Matching** | Limited (especially for squares) | Better (adaptive selection) |
| **Vision Token Consistency** | High (exact targets) | Lower (varies within tier) |
| **Result Comparability** | Easy (same tokens) | Harder (aggregate by tier) |
| **Configuration Complexity** | Simple | More complex |
| **Implementation Complexity** | Simple | More complex |

## Immediate Recommendation

### Short-Term Fix: Avoid Perfect Squares

**Current**: `[432, 720, 1008, 1440]` → crops: [2, 4, 6, 9]
- Problem: 4 and 9 are perfect squares, limited tiling options

**Recommended**: `[432, 720, 1008, 1296, 1584]` → crops: [2, 4, 6, 8, 10]
- Better: All composite numbers, more tiling options
- Still simple, comparable results
- Better aspect ratio coverage

**Why this works**:
- 2 crops: (1,2), (2,1) - covers wide/tall
- 4 crops: (1,4), (2,2), (4,1) - covers wide/square/tall (but square is limited)
- 6 crops: (1,6), (2,3), (3,2), (6,1) - excellent coverage ✓
- 8 crops: (1,8), (2,4), (4,2), (8,1) - good coverage ✓
- 10 crops: (1,10), (2,5), (5,2), (10,1) - good coverage ✓

### Long-Term: Consider Hybrid Tier Approach

If better aspect ratio matching is critical, implement hybrid tier-based approach with preferred crop counts.

## Implementation Considerations

### Result Recording

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

### Filename Options

**Option A**: Use tier name
```
coco-2014-vqa_imgsizetier-medium_topk8_blocks14.json
```

**Option B**: Use selected crops
```
coco-2014-vqa_imgsizecrops6_topk8_blocks14.json
```

**Option C**: Use tier + crops
```
coco-2014-vqa_imgsizetier-medium-crops6_topk8_blocks14.json
```

## Open Questions

1. **Vision token variation**: Is 576 token range (720-1296) acceptable for Tier 2?
2. **Comparison method**: How to compare results across tiers? By typical tokens or actual distribution?
3. **Backward compatibility**: Can we support both fixed targets and tier-based?
4. **Edge cases**: How to handle very small/large images or extreme aspect ratios?

## Conclusion

**For now**: Use fixed targets but **avoid perfect squares** (4, 9, 16 crops). Use composite numbers (2, 4, 6, 8, 10, 12) for better tiling options.

**Future**: If aspect ratio matching is critical, consider hybrid tier-based approach with preferred crop counts. This provides better flexibility while maintaining some predictability.

