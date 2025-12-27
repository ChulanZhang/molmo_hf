# COCO Caption Standard Evaluation

This document explains how to use the standard COCO Caption evaluation based on the official [coco-caption](https://github.com/tylin/coco-caption) repository.

## Install dependencies

Standard COCO Caption evaluation requires `pycocoevalcap` and `pycocotools`:

```bash
pip install pycocoevalcap pycocotools
```

**Note**: If these packages are missing, the system falls back to a simplified CIDEr evaluation (token overlap).

## Metrics

Standard COCO Caption evaluation provides:

- **CIDEr**: Consensus-based Image Description Evaluation (primary metric)
- **BLEU_1, BLEU_2, BLEU_3, BLEU_4**: BLEU scores
- **METEOR**: METEOR score
- **ROUGE_L**: ROUGE-L score
- **SPICE**: SPICE score (optional; requires Stanford CoreNLP)

## Usage

### Within experiments

The COCO Caption dataset automatically uses standard evaluation. Specify the dataset name:

```python
from experiments.base_experiment import BaseExperiment

experiment = BaseExperiment(
    model_path="checkpoints",
    output_dir="./results"
)

# Load COCO Caption dataset
from molmo.data import get_dataset_by_name
dataset = get_dataset_by_name("coco_caption", split="validation")

# Run evaluation (standard COCO evaluation will be used)
results = experiment.compute_accuracy(
    batch=batch,
    predictions=outputs,
    metric_name="cider_score"  # automatically uses standard evaluation
)
```

### Result format

The evaluation result includes:

```python
{
    "accuracy": float,  # CIDEr score (primary metric)
    "per_sample_scores": List[Dict],  # details per sample
    "num_samples": int,  # sample count
    "coco_metrics": {  # all COCO metrics
        "CIDEr": float,
        "BLEU_1": float,
        "BLEU_2": float,
        "BLEU_3": float,
        "BLEU_4": float,
        "METEOR": float,
        "ROUGE_L": float,
        # ... other metrics
    }
}
```

### Direct evaluation

You can call the evaluator directly:

```python
from molmo.eval.coco_caption_eval import evaluate_coco_caption_standard

# Prepare predictions and references
predictions = [
    {"image_id": 1, "caption": "A man riding a horse on a beach."},
    {"image_id": 2, "caption": "A dog playing in the park."},
    # ...
]

references = [
    {"image_id": 1, "caption": "A man riding a horse on a beach."},
    {"image_id": 1, "caption": "A person on a horse by the sea."},
    {"image_id": 2, "caption": "A dog playing in the park."},
    # ... typically 5 references per image
]

# Evaluate
metrics = evaluate_coco_caption_standard(predictions, references)
print(f"CIDEr: {metrics['CIDEr']:.4f}")
print(f"BLEU-4: {metrics['BLEU_4']:.4f}")
print(f"METEOR: {metrics['METEOR']:.4f}")
```

### Evaluate from batch results

If you already have batch results from `BaseExperiment.compute_accuracy()`:

```python
from molmo.eval.coco_caption_eval import evaluate_coco_caption_from_batch_results

# per_sample_results returned by compute_accuracy()
metrics = evaluate_coco_caption_from_batch_results(per_sample_results)
```

## Implementation details

### Evaluation flow

1. **Collect predictions and references** from batch results.
2. **Convert format** to COCO-style JSON (reference and result files).
3. **Call pycocoevalcap** to compute all metrics.
4. **Return results** as a dictionary.

### Fallback

If `pycocoevalcap` is not installed, the system falls back to a simplified CIDEr:

- Uses token overlap (Jaccard similarity).
- Computes max overlap between each prediction and all references.
- Returns a score in [0, 1].

### Corpus-level vs per-sample

Standard COCO Caption evaluation is **corpus-level** because:

- CIDEr, BLEU, etc. need corpus statistics.
- These metrics consider all reference captions collectively.
- Per-sample evaluation cannot accurately reflect performance.

Therefore `compute_accuracy()`:
1. Collects predictions and references for all samples.
2. Runs corpus-level evaluation at the end.
3. Assigns the corpus score to each sample (as an approximation).

## References

- [Official COCO Caption evaluation](https://github.com/tylin/coco-caption)
- [COCO dataset](https://cocodataset.org/)
- [CIDEr paper](https://arxiv.org/abs/1411.5726)


