# Safe Deletion Guide for Deprecated Documents

## Can deprecated/ documents be safely deleted?

**Answer: YES, with conditions**

## Safety Analysis

### ✅ Safe to Delete Because:

1. **No active code references**: No code or scripts directly reference these document paths
2. **Content fully integrated**: All content has been moved to new unified documents:
   - `image_resolution_vision_tokens_mapping.md` → `knobs/vision_tokens_knob.md`
   - `exp3_transformer_blocks_mask.md` → `knobs/transformer_blocks_knob.md`
   - `exp6_crop_overlap_analysis.md` → `knobs/vision_tokens_knob.md`
   - `auto_batch_size_logic.md` → `mechanisms/batch_size_optimization.md`
   - `dynamic_batch_size_guide.md` → `mechanisms/batch_size_optimization.md`
   - `max_crops_limits.md` → `knobs/vision_tokens_knob.md`

3. **Only referenced in migration docs**: The only references are in:
   - `deprecated/README.md` (migration guide - should be kept)
   - `MIGRATION_SUMMARY.md` (summary - mentions they exist)
   - `CONSOLIDATION_COMPLETE.md` (completion summary)

4. **Git history available**: If using Git, documents can be recovered from history

### ⚠️ Considerations:

1. **Keep `deprecated/README.md`**: This file explains migration paths and should be kept
2. **Script names similar but unrelated**: Scripts like `exp3_transformer_blocks_mask.py` have similar names but are not references to the deprecated documents
3. **Historical reference**: Some may want to see how documentation evolved

## Recommendation

### Option 1: Delete Now (Recommended if using Git)

If you're using Git version control:

```bash
# Delete deprecated documents (keep README.md)
cd docs/deprecated
rm image_resolution_vision_tokens_mapping.md
rm exp3_transformer_blocks_mask.md
rm exp6_crop_overlap_analysis.md
rm auto_batch_size_logic.md
rm dynamic_batch_size_guide.md
rm max_crops_limits.md

# Keep deprecated/README.md for migration reference
```

**Why safe**: Can recover from Git history if needed

### Option 2: Keep for Transition Period

Keep for 1-2 months, then delete:
- Allows team members to find migration paths
- Gives time to update any external references
- Then delete after transition period

### Option 3: Keep Only README.md

Delete all deprecated documents except `deprecated/README.md`:

```bash
cd docs/deprecated
# Keep README.md, delete others
rm *.md
# Then restore README.md from git or recreate it
```

## What to Keep

**Must keep**: `deprecated/README.md`
- Explains migration paths
- Helps users find new document locations
- Documents what was consolidated

## Verification Before Deletion

Before deleting, verify:
1. ✅ All content integrated into new documents
2. ✅ No active code references (verified - none found)
3. ✅ Migration paths documented in `deprecated/README.md`
4. ✅ Git history available (if using Git)

## After Deletion

If you delete the documents:
1. Update `MIGRATION_SUMMARY.md` to note documents were deleted
2. Keep `deprecated/README.md` for historical reference
3. Optionally add note in `deprecated/README.md`: "Documents deleted on [date]. Content available in new unified documents and Git history."

## Conclusion

**Safe to delete** if:
- Using Git (can recover from history)
- All content confirmed integrated
- Team notified of new document locations

**Recommendation**: Delete now if using Git, or keep for 1-2 month transition period.

