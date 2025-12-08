# Dataset Support Analysis for Exp5 and Exp6

## Summary

This document analyzes whether various datasets can be used with exp5 (accuracy measurement) and exp6 (latency measurement) experiments.

## Datasets Analyzed

1. ScienceQA
2. TextVQA
3. VizWiz
4. OK-VQA
5. ST-VQA (SceneTextQa)
6. TallyQA
7. DocVQA

## 1. ScienceQA (`science_qa_img`)

### ✅ Dataset Loading
- **Status**: ✅ Supported
- **Dataset Name**: `science_qa_img`
- **Class**: `ScienceQAImageOnly` in `molmo/data/academic_datasets.py`
- **Validation Split**: ✅ Supported (`split in ["train", "validation", "test"]`)
- **Loading**: Uses HuggingFace dataset `derek-thomas/ScienceQA`

### ⚠️ Accuracy Calculation
- **Current Status**: ⚠️ **Requires modification**
- **Issue**: `BaseExperiment.compute_accuracy()` only supports `metric_name="vqa_score"`, but ScienceQA requires multiple choice (MC) evaluation
- **Required Metric**: `mc` (multiple choice)
- **Data Format**: 
  - `style="science_qa"`
  - `answer_idx`: integer index of correct answer
  - `options`: list of choice strings

### 🔧 Required Changes
1. Modify `BaseExperiment.compute_accuracy()` to support `metric_name="mc"`
2. Extract `answer_idx` and `options` from metadata
3. Use `select_mc_option()` function from `molmo/eval/vqa.py` to parse prediction
4. Compare predicted index with `answer_idx`

### ✅ Latency Measurement
- **Status**: ✅ Should work (no accuracy calculation needed for latency-only measurement)

---

## 2. TextVQA (`text_vqa`)

### ✅ Dataset Loading
- **Status**: ✅ **Fully Supported**
- **Dataset Name**: `text_vqa`
- **Class**: `TextVqa` in `molmo/data/academic_datasets.py`
- **Validation Split**: ✅ Supported (loaded from HuggingFace `facebook/textvqa`)
- **Loading**: Uses HuggingFace dataset

### ✅ Accuracy Calculation
- **Status**: ✅ **Fully Supported**
- **Metric**: Uses `vqa_score` (same as VQA v2)
- **Data Format**:
  - `style="text_vqa"`
  - `answers`: list of answer strings (same format as VQA v2)
  - `metadata`: contains `image_id`, `question_id`, etc.

### ✅ Latency Measurement
- **Status**: ✅ **Fully Supported**

### 📝 Usage
```python
# For exp5
experiment.run(
    dataset_name="text_vqa",
    split="validation",
    ...
)

# For exp6
experiment.run(
    dataset_name="text_vqa",
    split="validation",
    ...
)
```

---

## 3. VizWiz

### ❌ Dataset Loading
- **Status**: ❌ **Not Found**
- **Search Results**: No VizWiz-related code found in the codebase
- **Possible Reasons**:
  1. Not yet implemented
  2. Different dataset name
  3. Part of another dataset class

### 🔧 Required Implementation
If VizWiz needs to be added:
1. Create `VizWiz` class in `molmo/data/academic_datasets.py` or `academic_datasets_manual.py`
2. Add dataset loading logic (likely from HuggingFace or manual download)
3. Implement `get()` method with appropriate data format
4. Add to `get_dataset_by_name()` in `molmo/data/__init__.py`
5. Determine appropriate evaluation metric (likely `vqa_score`)

### 📝 Recommendation
- Check if VizWiz is available under a different name
- Or implement VizWiz dataset support first before using it in exp5/exp6

---

## 4. OK-VQA (`okvqa`)

### ✅ Dataset Loading
- **Status**: ✅ **Fully Supported**
- **Dataset Name**: `okvqa`
- **Class**: `OkVqa` in `molmo/data/academic_datasets.py`
- **Validation Split**: ✅ Supported (`split in ["train", "validation", "test"]`)
- **Loading**: Uses HuggingFace dataset `HuggingFaceM4/OK-VQA`

### ✅ Accuracy Calculation
- **Status**: ✅ **Fully Supported**
- **Metric**: Uses `vqa_score` (same as VQA v2)
- **Data Format**:
  - `style="okvqa"`
  - `answers`: list of answer strings (same format as VQA v2)
  - `metadata`: contains `example_id` (question_id)

### ✅ Latency Measurement
- **Status**: ✅ **Fully Supported**

### 📝 Usage
```python
# For exp5
experiment.run(
    dataset_name="okvqa",
    split="validation",
    ...
)

# For exp6
experiment.run(
    dataset_name="okvqa",
    split="validation",
    ...
)
```

---

## 5. ST-VQA (`st_qa`)

### ⚠️ Dataset Loading
- **Status**: ⚠️ **Supported but requires manual download**
- **Dataset Name**: `st_qa`
- **Class**: `SceneTextQa` in `molmo/data/academic_datasets.py`
- **Validation Split**: ✅ Supported (custom split from train data)
- **Loading**: Requires manual download from https://rrc.cvc.uab.es/?ch=11
- **Note**: Validation split is created by splitting train data (first 1024 samples)

### ⚠️ Accuracy Calculation
- **Status**: ⚠️ **Requires modification**
- **Issue**: `BaseExperiment.compute_accuracy()` only supports `vqa_score`, but ST-VQA requires `ansl,em` evaluation
- **Required Metrics**: 
  - `ansl`: Average Normalized Levenshtein Similarity (from `molmo/eval/vqa.py`)
  - `em`: Exact Match
- **Data Format**:
  - `style="st_qa"`
  - `answers`: list of answer strings

### ✅ Latency Measurement
- **Status**: ✅ Should work (no accuracy calculation needed for latency-only measurement)

### 🔧 Required Changes
1. Modify `BaseExperiment.compute_accuracy()` to support `metric_name="ansl"` and `metric_name="em"`
2. Import `anls_metric` from `molmo/eval/vqa.py`
3. For `ansl`: use `max(anls_metric(ref, pred) for ref in answers)`
4. For `em`: use `pred.lower() in [x.lower() for x in answers]`

---

## 6. TallyQA (`tally_qa`)

### ❌ Dataset Loading
- **Status**: ❌ **No validation split**
- **Dataset Name**: `tally_qa`
- **Class**: `TallyQa` in `molmo/data/academic_datasets.py`
- **Validation Split**: ❌ Only supports `["train", "test"]` - **no validation split**
- **Loading**: Uses HuggingFace dataset

### ⚠️ Accuracy Calculation
- **Status**: ⚠️ **Requires modification**
- **Issue**: `BaseExperiment.compute_accuracy()` only supports `vqa_score`, but TallyQA requires `em` (exact match) evaluation
- **Required Metric**: `em` (exact match)
- **Data Format**:
  - `style="tally_qa"`
  - `answer`: string (not `answers` list)
  - `message_list`: list of questions with answers

### ⚠️ Latency Measurement
- **Status**: ⚠️ Can use `split="test"` for latency measurement, but no validation set

### 📝 Recommendation
- **Not recommended for exp5/exp6** because there's no validation split
- Could use test set, but that's not ideal for validation

---

## 7. DocVQA (`doc_qa`)

### ✅ Dataset Loading
- **Status**: ✅ **Fully Supported**
- **Dataset Name**: `doc_qa`
- **Class**: `DocQa` in `molmo/data/academic_datasets.py`
- **Validation Split**: ✅ Supported (`split in ["train", "validation", "test"]`)
- **Loading**: Uses HuggingFace dataset `HuggingFaceM4/DocumentVQA`

### ⚠️ Accuracy Calculation
- **Status**: ⚠️ **Requires modification**
- **Issue**: `BaseExperiment.compute_accuracy()` only supports `vqa_score`, but DocVQA requires `ansl,em` evaluation
- **Required Metrics**: 
  - `ansl`: Average Normalized Levenshtein Similarity
  - `em`: Exact Match
- **Data Format**:
  - `style="doc_qa"`
  - `answers`: list of answer strings
  - `metadata`: contains `doc_id`, `question_types`, `example_id`

### ✅ Latency Measurement
- **Status**: ✅ Should work (no accuracy calculation needed for latency-only measurement)

### 🔧 Required Changes
Same as ST-VQA: need to add support for `ansl` and `em` metrics in `compute_accuracy()`

---

## Recommendations

### Immediate Use (No Code Changes)
1. **TextVQA**: ✅ Ready to use for both exp5 and exp6
2. **OK-VQA**: ✅ Ready to use for both exp5 and exp6

### Requires Code Changes
3. **ScienceQA**: 
   - ⚠️ Need to modify `BaseExperiment.compute_accuracy()` to support `mc` metric
   - Can be used for exp6 (latency) without changes
   - For exp5 (accuracy), need to add MC evaluation support

4. **ST-VQA**: 
   - ⚠️ Need to modify `BaseExperiment.compute_accuracy()` to support `ansl` and `em` metrics
   - Requires manual dataset download
   - Can be used for exp6 (latency) without changes

5. **DocVQA**: 
   - ⚠️ Need to modify `BaseExperiment.compute_accuracy()` to support `ansl` and `em` metrics
   - Can be used for exp6 (latency) without changes

### Not Recommended
6. **TallyQA**: 
   - ❌ No validation split available (only train/test)
   - Not suitable for exp5/exp6 validation experiments

7. **VizWiz**: 
   - ❌ Need to implement dataset loading first
   - Then determine evaluation metric

---

## Implementation Plan for ScienceQA Support

### Step 1: Modify `BaseExperiment.compute_accuracy()`

Add support for `metric_name="mc"`:

```python
def compute_accuracy(
    self,
    batch: Dict[str, torch.Tensor],
    predictions: torch.Tensor,
    metric_name: str = "vqa_score",
) -> Dict[str, float]:
    # ... existing code ...
    
    for i, pred_text in enumerate(pred_texts):
        # ... existing prediction extraction ...
        
        metadata = metadatas[i] if i < len(metadatas) else {}
        
        # Compute score
        if metric_name == "vqa_score":
            # ... existing vqa_score logic ...
        elif metric_name == "mc":
            # New: Multiple choice evaluation
            if "answer_idx" not in metadata or "options" not in metadata:
                log.warning(f"Sample {i} missing answer_idx or options for MC evaluation")
                scores.append(0.0)
                continue
            
            from molmo.eval.vqa import select_mc_option
            options = metadata["options"]
            predicted_idx = select_mc_option(pred_text, options)
            correct_idx = metadata["answer_idx"]
            score = 1.0 if predicted_idx == correct_idx else 0.0
        else:
            raise NotImplementedError(f"Metric {metric_name} not implemented")
        
        scores.append(score)
        # ... rest of the code ...
```

### Step 2: Update exp5 and exp6 to use correct metric

For ScienceQA:
```python
batch_accuracy = self.compute_accuracy(
    batch=batch,
    predictions=outputs,
    metric_name="mc",  # Use "mc" for ScienceQA
)
```

---

## Testing Checklist

### TextVQA
- [x] Dataset loads with `split="validation"`
- [x] Accuracy calculation works with `vqa_score`
- [x] Can be used in exp5 and exp6

### ScienceQA
- [x] Dataset loads with `split="validation"`
- [ ] Accuracy calculation with `mc` metric (needs implementation)
- [x] Can be used in exp6 (latency only)

### VizWiz
- [ ] Dataset loading (needs implementation)
- [ ] Accuracy calculation (depends on dataset format)
- [ ] Can be used in exp5/exp6 (after implementation)

