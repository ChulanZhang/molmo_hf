# Dataset Size Estimation

High-level guidance for estimating dataset size and runtime for profiling experiments.

## What to track
- Number of samples per dataset/split.
- Avg image size and tokenized text length (affects prefill/decode cost).
- Disk footprint of raw data and HF cache.

## Estimating runtime
- Runtime ≈ (prefill_time + decode_time_per_token × generated_tokens) × num_samples.
- Use small pilot runs (e.g., 100–500 samples) to measure per-sample latency, then extrapolate.
- Account for dataloader overhead and warmup cost.

## Storage checklist
- HF datasets cache location/size.
- Raw image directories (e.g., COCO 2014 val/train, scene-text).
- Optional zips can be removed once extracted to save space.

## Tips
- For multi-dataset sweeps, precompute sample counts and shard sizes to balance workloads.
- Use `scripts/sync_all_9_datasets.sh` for full sync when needed.
- Keep notes of which splits are required for each experiment to avoid unnecessary downloads.
