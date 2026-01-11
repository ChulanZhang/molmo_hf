# AdaLLaVA Evaluation Datasets

This document lists the datasets used by AdaLLaVA for evaluation, based on the paper and code repository.

## Datasets Used in AdaLLaVA Paper

According to the AdaLLaVA paper (https://arxiv.org/pdf/2503.10905), the following datasets were used for evaluation:

### 1. TextVQA (`textvqa_val`)
- **Full Name**: Text-based Visual Question Answering
- **Task**: Answer questions about text in images
- **Split**: Validation set
- **Metric**: VQA accuracy (standard VQA score)

### 2. VQAv2 (`vqav2_test`, `vqav2_val`)
- **Full Name**: Visual Question Answering v2
- **Task**: Answer questions about images
- **Splits**: Validation and test sets
- **Metric**: VQA accuracy
- **Note**: Test set requires submission to official server

### 3. MME (`mme`)
- **Full Name**: Multimodal Evaluation
- **Task**: Comprehensive multimodal understanding
- **Metrics**: Perception score and Cognition score

### 4. POPE (`pope`)
- **Full Name**: Polling-based Object Probing Evaluation
- **Task**: Evaluate object hallucination
- **Metric**: Accuracy, Precision, Recall, F1

### 5. MMBench (`mmbench_en_dev`)
- **Full Name**: Multimodal Benchmark
- **Task**: Comprehensive multimodal understanding
- **Split**: English dev set
- **Metric**: Accuracy

### 6. ScienceQA (`scienceqa_img`)
- **Full Name**: Science Question Answering
- **Task**: Answer science questions with images
- **Metric**: Multiple choice accuracy

### 7. OK-VQA (`okvqa_val`)
- **Full Name**: Outside Knowledge Visual Question Answering
- **Task**: Answer questions requiring external knowledge
- **Split**: Validation set
- **Metric**: VQA accuracy

## Evaluation Settings

### Latency Measurement
- **Batch Size**: **batch_size=1** (single sample per inference)
- **Reason**: To accurately measure per-sample latency for real-world scenarios
- **Hardware**: Typically measured on specific GPUs (e.g., NVIDIA V100, A100, H100)

### Latency Budget Constraints
AdaLLaVA evaluates with different latency budget constraints:
- **85% budget**: 85% of baseline latency
- **90% budget**: 90% of baseline latency
- **100% budget**: Full baseline latency (no constraint)

Example:
```bash
# 85% latency budget
--latency_budget 0.85  # Relative to baseline

# Or absolute latency budget in milliseconds
--latency_budget 170.0  # 170ms
```

## Dataset Preparation

### Automatic Download
Most datasets are automatically downloaded via HuggingFace when first used.

### Manual Download (if needed)
```bash
# Download specific datasets
python scripts/download_data.py textvqa --n_procs 1
python scripts/download_data.py okvqa --n_procs 1
python scripts/download_data.py coco_2014_vqa --n_procs 1
```

### Dataset Paths
- **HuggingFace Cache**: `~/.cache/huggingface/datasets/` (default)
- **Custom Path**: Set `MOLMO_DATA_DIR` environment variable
- **Images**: Some datasets require separate image downloads (e.g., COCO images)

## Using These Datasets in Our Evaluation

### Direct Evaluation
```bash
python experiments/controller/evaluate_adaptive_inference.py \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0
```

### LMms-Eval Framework
```bash
python -m experiments.controller.run_lmms_eval \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --tasks textvqa_val,mme,pope,mmbench_en_dev,scienceqa_img \
    --latency_budget 200.0
```

## Dataset Mapping

Our internal dataset names map to lmms-eval task names:

| Internal Name | LMms-Eval Task | Description |
|--------------|---------------|-------------|
| `text_vqa` | `textvqa_val` | TextVQA validation |
| `okvqa` | `okvqa_val` | OK-VQA validation |
| `coco_2014_vqa` | `vqav2_val` | VQA v2 validation |
| `science_qa_img` | `scienceqa_img` | ScienceQA with images |
| - | `mme` | Multimodal Evaluation |
| - | `pope` | POPE evaluation |
| - | `mmbench_en_dev` | MMBench English dev |

## References

- [AdaLLaVA Paper](https://arxiv.org/pdf/2503.10905)
- [AdaLLaVA GitHub](https://github.com/zhuoyan-xu/AdaLLaVA)
- [LMms-Eval GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)

