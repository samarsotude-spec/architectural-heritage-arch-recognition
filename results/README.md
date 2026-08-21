# Reported Experimental Results

This directory records the final quantitative results reported for the fixed held-out test set used throughout the study.

## Test set

- Total test images: 1,496
- Has Arch: 877
- No Arch: 619

## Overall performance

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| ResNet50 | 91.98% | 0.9174 |
| DenseNet121 | 90.84% | 0.9063 |
| MobileNetV2 | 87.70% | 0.8746 |
| HOG + Linear SVM | 61.97% | 0.6110 |
| ResNet50 + architecture-aware augmentation | 91.91% | 0.9167 |

## Confusion matrices

Class order is [No Arch, Has Arch].

### ResNet50

```text
[[560, 59],
 [ 61,816]]
```

### DenseNet121

```text
[[566, 53],
 [ 84,793]]
```

### MobileNetV2

```text
[[552, 67],
 [117,760]]
```

### HOG + Linear SVM

```text
[[352,267],
 [302,575]]
```

### ResNet50 + architecture-aware augmentation

```text
[[560, 59],
 [ 62,815]]
```

## ResNet50 class-wise metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No Arch | 0.9018 | 0.9047 | 0.9032 | 619 |
| Has Arch | 0.9326 | 0.9304 | 0.9315 | 877 |

## DenseNet121 class-wise metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No Arch | 0.8708 | 0.9144 | 0.8920 | 619 |
| Has Arch | 0.9374 | 0.9042 | 0.9205 | 877 |

## MobileNetV2 class-wise metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No Arch | 0.8251 | 0.8918 | 0.8571 | 619 |
| Has Arch | 0.9190 | 0.8666 | 0.8920 | 877 |

## ResNet50 with architecture-aware augmentation

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No Arch | 0.9003 | 0.9047 | 0.9025 | 619 |
| Has Arch | 0.9325 | 0.9293 | 0.9309 | 877 |

## HOG + Linear SVM

- Accuracy: 61.97%
- Macro F1: 0.6110
- ROC AUC: 0.6689
- Average Precision: 0.7362
- Misclassifications: 569 / 1,496
- False positives: 267
- False negatives: 302

## Model selection details

### ResNet50
- Best epoch: 6
- Best validation loss: 0.3332
- Best validation accuracy: 93.45%
- Early stopping: epoch 11

### DenseNet121
- Best epoch by validation loss: 6
- Best validation loss: 0.3524
- Validation accuracy at selected epoch: 91.44%
- Early stopping: epoch 11

### MobileNetV2
- Best recorded epoch: 14
- Best validation loss: 0.3896
- Best validation accuracy: 88.64%

### ResNet50 + architecture-aware augmentation
- Best epoch: 6
- Best validation loss: 0.3374
- Validation accuracy at selected epoch: 93.18%
- Early stopping: epoch 11

## Misclassification analysis

The final ResNet50 baseline produced:

- 59 false positives
- 61 false negatives
- 120 total misclassifications

All 120 misclassified images were manually inspected qualitatively. The recurring patterns described in the manuscript are qualitative observations and should not be interpreted as a frequency-based error taxonomy.
