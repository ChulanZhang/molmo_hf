# Groundtruth Answer Length Analysis Summary

Based on analysis of 1000 samples per dataset (using tokenizer from `configs/tokenizer`):

## Results

| Dataset | Split | Mean | P95 | P99 | P99.9 | Max | Current Setting | Recommended |
|---------|-------|------|-----|-----|-------|-----|----------------|-------------|
| coco_2014_vqa | validation | 1.4 | 3.0 | 5.0 | 9.0 | 12 | 64 | **32** (covers 100%) |
| text_vqa | validation | 2.8 | 7.0 | 11.0 | 24.0 | 54 | 64 | **64** (covers 100%) |
| okvqa | validation | 2.2 | 4.0 | 5.0 | 8.0 | 15 | 64 | **32** (covers 100%) |
| science_qa_img | validation | N/A | N/A | N/A | N/A | N/A | 64 | **16** (multiple choice, single token) |
| st_qa | validation | 3.0 | 7.0 | 10.0 | 15.7 | 16 | 512 | **32** (covers 100%) |
| doc_qa | validation | 4.6 | 11.0 | 16.0 | 24.4 | 28 | 512 | **64** (covers 100%) |
| tally_qa | test | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 64 | **16** (numeric answers, all 1 token) |
| mmmu | validation | 1.1 | 1.0 | 3.0 | 10.5 | 15 | 1024 | **32** (mostly single-token answers; max 15) |

## Notes

1. **coco_2014_vqa**: Very short answers (mostly 1 token). Current setting of 64 is safe but could be reduced to 32.
2. **text_vqa**: Has some longer answers (max 54 tokens). Current setting of 64 is appropriate.
3. **okvqa**: Similar to coco_2014_vqa, mostly short answers. Current setting of 64 is safe but could be reduced to 32.
4. **science_qa_img**: Multiple choice dataset - answers are single option indices (typically 1 token). Current setting of 64 is more than sufficient.
5. **st_qa**: Short answers (max 16 tokens). Current setting of 512 is excessive, 32 would be sufficient.
6. **doc_qa**: Slightly longer answers (max 28 tokens). Current setting of 512 is excessive, 64 would be sufficient.
7. **tally_qa**: Counting answers; all answers are single-token numerals (e.g., \"2\", \"5\"). Current 64 is enough; 16 is sufficient.\n8. **mmmu**: Mostly single-token answers (MC options or short text) but can reach 15 tokens (P99.9=10.5). Current 1024 is excessive; 32 is sufficient.

## Recommendations for `run_multi_datasets.sh`

Based on P99.9 + 20% margin, rounded to nearest power of 2:

```bash
DATASETS=(
    "coco_2014_vqa:validation:32"      # Reduced from 64
    "text_vqa:validation:64"            # Keep 64 (covers max 54)
    "okvqa:validation:32"              # Reduced from 64
    "science_qa_img:validation:16"     # Reduced from 64 (MC, single token)
    "st_qa:validation:32"               # Reduced from 512
    "doc_qa:validation:64"              # Reduced from 512
    "tally_qa:test:16"                 # Reduced from 64 (all answers are 1 token)
    "mmmu:validation:32"               # Reduced from 1024 (max 15 tokens, P99.9=10.5)
)
```

## Important Notes

- These values are **upper limits** - the model will stop early when EOS token is generated
- Setting larger values is safe but may waste compute during generation
- For datasets with very short answers (coco_2014_vqa, okvqa), reducing max_new_tokens can speed up inference
- For long-answer datasets (mmmu), keeping high values (1024) is appropriate

