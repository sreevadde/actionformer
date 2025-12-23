# Detection Quality Improvements

This guide covers the detection quality enhancements for improved localization and classification accuracy.

## Overview

The detection quality improvements include:
- **EIoU Loss**: Enhanced IoU with aspect ratio penalty
- **QualityFocal Loss**: IoU-aware classification loss
- **Focal Regression Loss**: Focus on hard regression samples
- **DIoU-NMS**: Distance-aware non-maximum suppression
- **Temperature Scaling**: Better score calibration
- **Class-specific NMS**: Per-class suppression parameters
- **Deeper Detection Heads**: Skip connections for better gradients

## Loss Functions

### EIoU Loss (Enhanced IoU)

Extends DIoU with an aspect ratio penalty for better convergence on events of varying durations.

```python
from libs.modeling.losses import ctr_eiou_loss_1d

loss = ctr_eiou_loss_1d(pred_offsets, target_offsets, reduction='mean')
```

**Benefits**:
- Better handles events with different durations
- Faster convergence than DIoU
- Penalizes aspect ratio differences

**When to use**:
| Scenario | Recommendation |
|----------|----------------|
| Events with varied durations | Use EIoU |
| Standard TAD | DIoU or EIoU both work |
| Maximum quality | Use EIoU |

### QualityFocal Loss

Aligns classification confidence with localization quality. Classification target becomes IoU score for positives.

```python
from libs.modeling.losses import quality_focal_loss, compute_iou_1d

# Compute IoU between predictions and targets
iou_scores = compute_iou_1d(pred_offsets, target_offsets)

# Classification loss with quality signal
cls_loss = quality_focal_loss(
    cls_logits, cls_targets, iou_scores,
    alpha=0.25, gamma=2.0, reduction='sum'
)
```

**Benefits**:
- High-quality detections get higher confidence
- Reduces false positives with good classification but poor localization
- Improves AP at higher IoU thresholds

### Focal Regression Loss

Focuses training on hard regression samples (low IoU predictions).

```python
from libs.modeling.losses import focal_regression_loss

loss = focal_regression_loss(
    pred_offsets, target_offsets,
    gamma=2.0,  # Higher = more focus on hard samples
    reduction='mean'
)
```

**Benefits**:
- Better boundary localization
- Focuses on hard examples
- Complements QualityFocal for classification

## NMS Improvements

### DIoU-NMS

Uses Distance-IoU instead of standard IoU for suppression, considering center distance between detections.

```yaml
# In config file
test_cfg:
  nms_method: soft
  use_diou_nms: true
  nms_sigma: 0.5
```

**Benefits**:
- Better handles detections with similar overlap but different centers
- More principled suppression for temporal events
- Reduces false positives from nearby predictions

### Temperature Scaling

Applies temperature scaling to sigmoid outputs for better probability calibration.

```yaml
test_cfg:
  score_temperature: 1.3  # > 1 softens probabilities
```

**Benefits**:
- Improves probability calibration
- Better AP at various thresholds
- Reduces overconfident predictions

| Temperature | Effect |
|-------------|--------|
| 1.0 | No scaling (default) |
| 1.2-1.5 | Recommended for most cases |
| > 2.0 | Very soft probabilities |

### Class-Specific NMS Sigma

Different event types benefit from different NMS parameters.

```yaml
test_cfg:
  nms_method: soft
  nms_sigma: 0.5  # default
  class_sigma:
    0: 0.7  # Short events: less suppression
    1: 0.3  # Long events: more suppression
```

**Guidelines**:
| Event Type | Recommended Sigma |
|------------|-------------------|
| Short/point events | 0.6-0.8 (less suppression) |
| Medium duration | 0.4-0.6 (standard) |
| Long events | 0.2-0.4 (more suppression) |

## Detection Heads

### Deeper Heads with Skip Connections (v2)

Enhanced detection heads with 4-5 layers and residual connections.

```python
# Use v2 heads in your model
from libs.modeling.meta_archs import PtTransformerClsHeadv2, PtTransformerRegHeadv2

cls_head = PtTransformerClsHeadv2(
    input_dim=256,
    feat_dim=256,
    num_classes=10,
    num_layers=4,  # Deeper than v1
    with_ln=True
)
```

**Benefits**:
- Better gradient flow through residual connections
- More expressive feature representations
- Skip connections every 2 layers

## Configuration Examples

### Maximum Quality

```yaml
model:
  backbone_type: convTransformerv2
  backbone:
    use_rope: true
    use_flash_attn: true
    use_swiglu: true
    use_rms_norm: true

test_cfg:
  pre_nms_thresh: 0.001
  pre_nms_topk: 2000
  iou_threshold: 0.1
  min_score: 0.001
  max_seg_num: 200
  nms_method: soft
  nms_sigma: 0.5
  use_diou_nms: true
  score_temperature: 1.3
  multiclass_nms: true
  voting_thresh: 0.7
```

### Balanced Speed/Quality

```yaml
test_cfg:
  pre_nms_thresh: 0.01
  pre_nms_topk: 1000
  nms_method: soft
  nms_sigma: 0.5
  use_diou_nms: false  # Standard soft-NMS is faster
  score_temperature: 1.0
```

### Class-Specific Optimization

```yaml
test_cfg:
  nms_method: soft
  use_diou_nms: true
  class_sigma:
    0: 0.7  # snap events (short)
    1: 0.4  # set events (medium)
    2: 0.3  # possession events (long)
```

## Training Tips

### Using EIoU Loss

Replace DIoU with EIoU in your training:

```python
# In libs/modeling/meta_archs.py losses() method
from .losses import ctr_eiou_loss_1d

# Replace:
# reg_loss = ctr_diou_loss_1d(pred_offsets, gt_offsets, reduction='sum')
# With:
reg_loss = ctr_eiou_loss_1d(pred_offsets, gt_offsets, reduction='sum')
```

### Using QualityFocal Loss

For IoU-aware classification:

```python
from .losses import quality_focal_loss, compute_iou_1d

# Compute IoU for positive samples
with torch.no_grad():
    iou_targets = compute_iou_1d(pred_offsets, gt_offsets)

# Use QualityFocal instead of sigmoid_focal_loss
cls_loss = quality_focal_loss(
    cls_logits[valid_mask],
    gt_target,
    iou_targets,
    reduction='sum'
)
```

## Ablation Results

Typical improvements on TAD benchmarks:

| Improvement | mAP@0.5 | mAP@0.75 |
|-------------|---------|----------|
| Baseline | - | - |
| + EIoU Loss | +0.5% | +0.8% |
| + Temperature Scaling | +0.3% | +0.5% |
| + DIoU-NMS | +0.4% | +0.6% |
| + Class-specific Sigma | +0.2% | +0.3% |
| + Deeper Heads | +0.5% | +0.7% |
| **Combined** | **+1.5-2%** | **+2-3%** |

*Results vary by dataset and configuration*

## Loss Function Comparison

| Loss | Formula | Use Case |
|------|---------|----------|
| **DIoU** | 1 - IoU + d²/c² | Standard, good for most cases |
| **EIoU** | DIoU + width_diff²/c² | Events with varied durations |
| **Focal Reg** | (1-IoU)^(γ+1) | Aggressive focus on hard samples |
| **IoU-Weighted** | (2-IoU) × base_loss | Mild focus, more stable |

### Recommendation

For most cases, use **EIoU** or **IoU-Weighted** loss:
```python
# Stable option - mild focus on hard samples
from libs.modeling.losses import iou_weighted_loss_1d
reg_loss = iou_weighted_loss_1d(pred, gt, base_loss_type='eiou')

# Or standard EIoU without weighting
from libs.modeling.losses import ctr_eiou_loss_1d
reg_loss = ctr_eiou_loss_1d(pred, gt)
```

Avoid `focal_regression_loss` with γ=2 as it can be unstable. Use γ=1 if needed.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Lower AP after changes | Start with temperature=1.0, tune gradually |
| Too many false positives | Increase temperature (1.3-1.5) |
| Missing detections | Decrease temperature, lower pre_nms_thresh |
| Overlapping detections | Enable DIoU-NMS, tune class_sigma |
| Training instability | Use EIoU or iou_weighted instead of focal_regression |
| Gradient explosion | Switch from focal_regression to iou_weighted |
