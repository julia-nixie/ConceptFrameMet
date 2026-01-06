# Adaptive Source QA MelBERT - Usage Guide

## Overview

`AdaptiveSourceQAMelBert` provides configurable strategies for incorporating source domain information into metaphor detection.

## Configuration Flags

### 1. `--source_blend_mode` (Default: 'replacement')

Controls how source and target embeddings are combined:

- **`replacement`**: Original soft confidence approach
  ```
  blended = confidence * source + (1 - confidence) * target
  ```
  - Replaces target with source based on confidence
  - Higher confidence → more source influence
  
- **`additive`**: Keeps target, adds source as enhancement
  ```
  enhanced = target + alpha * confidence * source
  ```
  - Maintains baseline target strength
  - Adds source as supplementary information
  - Less likely to hurt performance if source is wrong

### 2. `--source_use_mode` (Default: 'all')

Controls when to apply source information:

- **`all`**: Use source for all samples (literals and metaphors)
  
- **`metaphor_only`**: Only use source for likely metaphors
  - Computes preliminary metaphor score
  - Only applies source blending when score > threshold
  - Rationale: Source domains only exist for metaphors

### 3. `--source_alpha` (Default: 0.3)

Scaling factor for additive mode (only used when `--source_blend_mode additive`):
- Controls how much source information is added
- Range: [0.0, 1.0]
- Lower values (0.1-0.3): Conservative, safer
- Higher values (0.4-0.6): More aggressive

### 4. `--metaphor_threshold` (Default: 0.5)

Threshold for metaphor-only mode (only used when `--source_use_mode metaphor_only`):
- Minimum metaphor probability to apply source
- Range: [0.0, 1.0]
- Lower values (0.3-0.4): Apply source more liberally
- Higher values (0.6-0.7): Only apply to confident metaphors

### 5. `--unfreeze_source_qa` (Default: False)

Whether to fine-tune the source QA model:
- `False`: Freeze source QA model (faster, less memory)
- `True`: Unfreeze for joint training (slower, more memory, potentially better)

---

## Usage Examples

### Example 1: Baseline Replacement (Same as SoftConfidenceSourceQAMelBert)

```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode replacement \
  --source_use_mode all \
  --do_train --do_eval
```

**Expected**: Similar to original soft confidence model

---

### Example 2: Additive + All Samples (Recommended Starting Point)

```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode additive \
  --source_use_mode all \
  --source_alpha 0.3 \
  --do_train --do_eval
```

**Rationale**:
- Keeps target strength (baseline performance)
- Adds source as enhancement (can only help)
- Conservative alpha (0.3) is safe

**Expected**: Should beat or match baseline

---

### Example 3: Additive + Metaphor-Only (Best Generalization)

```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode additive \
  --source_use_mode metaphor_only \
  --source_alpha 0.3 \
  --metaphor_threshold 0.5 \
  --do_train --do_eval
```

**Rationale**:
- Keeps target strength
- Only uses source for metaphors (where it's relevant)
- Avoids noise from using source for literal uses

**Expected**: Best zero-shot generalization (MOH, Trofi)

---

### Example 4: Aggressive Additive

```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode additive \
  --source_use_mode all \
  --source_alpha 0.5 \
  --do_train --do_eval
```

**Rationale**:
- More source influence (alpha=0.5)
- Try if conservative approach doesn't help enough

**Expected**: Higher risk, higher reward

---

### Example 5: Metaphor-Only with Low Threshold

```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode additive \
  --source_use_mode metaphor_only \
  --source_alpha 0.3 \
  --metaphor_threshold 0.3 \
  --do_train --do_eval
```

**Rationale**:
- Applies source more liberally (threshold=0.3)
- May help recall

---

### Example 6: Replacement + Metaphor-Only

```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode replacement \
  --source_use_mode metaphor_only \
  --metaphor_threshold 0.5 \
  --do_train --do_eval
```

**Rationale**:
- Original soft blending
- But only for metaphors
- Removes noise from literal uses

**Expected**: Better than full soft confidence

---

### Example 7: Additive + Unfrozen Source Model

```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode additive \
  --source_use_mode metaphor_only \
  --source_alpha 0.3 \
  --metaphor_threshold 0.5 \
  --unfreeze_source_qa True \
  --do_train --do_eval
```

**Rationale**:
- Joint training of source and metaphor models
- Source model adapts to metaphor task
- More parameters, more training time

**Expected**: Potentially best performance (if enough data)

---

## Recommended Experiment Order

1. **Start with Example 3** (Additive + Metaphor-Only)
   - Most likely to beat baseline
   - Good balance of safety and enhancement

2. **If not satisfactory, try Example 2** (Additive + All)
   - Maybe metaphor filtering is too restrictive
   
3. **If still not working, try Example 4** (Aggressive Alpha)
   - More source influence
   
4. **If replacement works better, try Example 6** (Replacement + Metaphor-Only)
   - At least remove literal noise

5. **Final attempt: Example 7** (Unfrozen)
   - Most complex, but joint optimization may help

---

## Full Configuration Example

```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode additive \
  --source_use_mode metaphor_only \
  --source_alpha 0.3 \
  --metaphor_threshold 0.5 \
  --unfreeze_source_qa False \
  --model_name roberta-base \
  --do_train \
  --do_eval \
  --learning_rate 2e-5 \
  --num_train_epoch 10 \
  --early_stopping_patience 5 \
  --per_gpu_train_batch_size 16 \
  --logging_dir saves/AdaptiveSourceQA_additive_metaphor \
  --seed 42
```

---

## Expected Performance Targets

Based on baseline performance:
- **Baseline MELBERT**: VUA18=0.782, MOH=0.806, Trofi=0.631

**Target with Additive + Metaphor-Only:**
- VUA18: **0.784** (+0.2%)
- VUA20: **0.707** (+0.2%)
- MOH: **0.812** (+0.6%)
- Trofi: **0.636** (+0.5%)

**Goal**: Consistently beat or match baseline across ALL datasets

---

## Troubleshooting

### Performance worse than baseline?

1. **Try lower alpha**: `--source_alpha 0.2` or `0.1`
2. **Try metaphor-only**: `--source_use_mode metaphor_only`
3. **Check early stopping**: May be overfitting, use best checkpoint

### No improvement over soft confidence?

1. **Try additive mode**: `--source_blend_mode additive`
2. **The issue is replacement, not threshold**

### Source seems to hurt?

1. **Use metaphor-only**: Likely adding noise to literals
2. **Lower alpha**: Reduce source influence
3. **Check source QA accuracy**: May need better source model

---

## Model Comparison

| Configuration | VUA (expected) | MOH (expected) | Trofi (expected) | Use Case |
|---------------|----------------|----------------|------------------|----------|
| Baseline | 0.782 | 0.806 | 0.631 | Standard |
| Soft Conf | 0.773 | 0.789 | 0.637 | Original approach |
| Hard Thresh | 0.762 | 0.809 | 0.630 | Simple threshold |
| Additive+All | 0.783 | 0.810 | 0.635 | Safe enhancement |
| **Additive+Metaphor** | **0.784** | **0.812** | **0.636** | **Best overall** |
| Replacement+Metaphor | 0.775 | 0.805 | 0.635 | Filtered soft |

---

## Notes

- All modes use soft confidence scores (not hard 0/1)
- Additive mode is THEORETICALLY safer (can't hurt baseline much)
- Metaphor-only mode is THEORETICALLY better (source only for metaphors)
- Actual results may vary - empirical testing needed!
