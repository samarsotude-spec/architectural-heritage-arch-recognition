# Automatic Arch Recognition in Architectural Heritage Images

This repository contains reproducible code and experimental resources for binary image-level **arch recognition** in architectural heritage imagery. The accompanying study evaluates transfer learning-based convolutional neural networks, a conventional HOG + Linear SVM baseline, and an architecture-aware data augmentation strategy.

## Study overview

The curated dataset contains **7,479 images** divided into two classes:

- **Has Arch:** 4,384 images
- **No Arch:** 3,095 images

A fixed stratified split with random seed **42** was used throughout all reported experiments:

| Split | Total | Has Arch | No Arch |
|---|---:|---:|---:|
| Training | 5,235 | 3,069 | 2,166 |
| Validation | 748 | 438 | 310 |
| Test | 1,496 | 877 | 619 |

The same fixed test set was retained for every reported model comparison.

## Models and experiments

The repository includes implementations of:

- ImageNet-pretrained **ResNet50** with partial fine-tuning
- ImageNet-pretrained **DenseNet121** with partial fine-tuning
- ImageNet-pretrained **MobileNetV2** with partial fine-tuning
- **HOG + Linear SVM** conventional baseline
- **ResNet50 with architecture-aware data augmentation**
- Misclassification analysis for the final ResNet50 baseline

## Final reported results

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| ResNet50 | **91.98%** | **0.9174** |
| DenseNet121 | 90.84% | 0.9063 |
| MobileNetV2 | 87.70% | 0.8746 |
| HOG + Linear SVM | 61.97% | 0.6110 |
| ResNet50 + architecture-aware augmentation | 91.91% | 0.9167 |

### ResNet50 baseline confusion matrix

Class order: **[No Arch, Has Arch]**

```text
[[560,  59],
 [ 61, 816]]
```

This corresponds to:

- True negatives: 560
- False positives: 59
- False negatives: 61
- True positives: 816
- Total misclassifications: 120

## Architecture-aware augmentation

Augmentation was applied **only to the training set**. Validation and test images were processed deterministically.

The final augmentation protocol used:

- Resize to 224 × 224
- Random horizontal flip: `p = 0.5`
- Random rotation: `-10° to +10°`
- Random scale: `0.95 to 1.05`
- No translation
- No shear
- Fill value: `(124, 116, 104)`
- Brightness jitter: `0.15`
- Contrast jitter: `0.15`
- ImageNet normalization

The purpose of this restrained augmentation strategy was to preserve the geometric and semantic identity of architectural arches.

## Training configuration

The CNN experiments used a common training protocol:

- Input size: `224 × 224`
- Batch size: `32`
- Maximum epochs: `20`
- Optimizer: `AdamW`
- Initial learning rate: `1e-4`
- Weight decay: `1e-4`
- Loss: cross-entropy with label smoothing `0.1`
- Scheduler: `ReduceLROnPlateau`
  - factor: `0.5`
  - patience: `2`
  - minimum learning rate: `1e-6`
- Early stopping patience: `5`
- Gradient clipping: `1.0`
- Random seed: `42`
- Mixed precision enabled when CUDA is available
- Model selection based on minimum validation loss
- Test data used only for final evaluation

No class weighting or resampling was used.

## Partial fine-tuning strategy

### ResNet50

Trainable components:

- `layer4`
- classification head

Parameter counts:

- Total: 23,512,130
- Trainable: 14,968,834
- Frozen: 8,543,296

### MobileNetV2

Trainable components:

- `features[18]`
- classifier

Parameter counts:

- Total: 2,226,434
- Trainable: 414,722
- Frozen: 1,811,712

### DenseNet121

Trainable components:

- `denseblock4`
- `norm5`
- classifier

Parameter counts:

- Total: 6,955,906
- Trainable: 2,162,178
- Frozen: 4,793,728

## HOG + Linear SVM baseline

The conventional baseline uses:

- Image size: `224 × 224`
- Grayscale HOG features
- Orientations: `9`
- Pixels per cell: `(16, 16)`
- Cells per block: `(2, 2)`
- Block normalization: `L2-Hys`
- HOG dimensionality: **6,084 features per image**
- `StandardScaler` fitted on the training features only
- Final validation-selected SVM parameter: `C = 0.001`

The validation set was used for parameter selection, while the test set remained untouched until final evaluation.

## Repository structure

```text
architectural-heritage-arch-recognition/
├── README.md
├── requirements.txt
├── data/
│   ├── fixed_train_split_public.csv
│   ├── fixed_validation_split_public.csv
│   └── fixed_test_split_public.csv
└── src/
    ├── resnet50_baseline.py
    ├── mobilenetv2_baseline.py
    ├── densenet121_baseline.py
    ├── hog_svm_baseline.py
    ├── resnet50_architecture_aware_augmentation.py
    └── failure_analysis.py
```

## Dataset files

The `data/` directory contains public-safe CSV files defining the exact experimental split membership.

Each split file includes:

- `sample_id`
- `original_filename`
- `class_name`
- `label`
- `split`
- `relative_reference`

The CSV files intentionally do **not** contain personal Google Drive paths.

## Raw image availability

The original images were collected principally from:

- Wikimedia Commons
- *Ganjnameh: Cyclopaedia of Iranian Islamic Architecture*

The raw image files are **not redistributed in this repository** because they originate from third-party documentary and published sources and remain subject to the reuse conditions of their respective sources.

The exact dataset metadata, class annotations, and fixed experimental partitions are being deposited separately in **Mendeley Data**. A persistent DOI will be cited after the dataset record is publicly released.

## Expected local dataset structure

The scripts accept a local dataset root containing the original images. The path resolver supports structures such as:

```text
dataset_root/
├── has_arch/
│   └── <images>
└── no_arch/
    └── <images>
```

and the original nested layout:

```text
dataset_root/
├── has_arch/
│   └── has_arch/
│       └── <images>
└── no_arch/
    └── no_arch/
        └── <images>
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Example usage

### ResNet50 baseline

```bash
python src/resnet50_baseline.py --dataset-root /path/to/dataset
```

### MobileNetV2 baseline

```bash
python src/mobilenetv2_baseline.py --dataset-root /path/to/dataset
```

### DenseNet121 baseline

```bash
python src/densenet121_baseline.py --dataset-root /path/to/dataset
```

### HOG + Linear SVM baseline

```bash
python src/hog_svm_baseline.py --dataset-root /path/to/dataset
```

### Architecture-aware augmentation experiment

```bash
python src/resnet50_architecture_aware_augmentation.py --dataset-root /path/to/dataset
```

### Misclassification analysis

After generating a test prediction CSV with the ResNet50 script:

```bash
python src/failure_analysis.py \
  --predictions-csv outputs/resnet50_baseline/test_predictions.csv \
  --verify-resnet50-final
```

## Reproducibility note

The repository uses the deposited split CSV files directly rather than regenerating the train/validation/test partition from directory order. This avoids accidental variation caused by filesystem ordering and preserves the exact experimental membership used in the study.

The fixed test set is never used for model selection.

## Data and code availability

The repository is currently being prepared for the public release associated with the research article. The Mendeley Data record and this code repository will be cross-linked after final release.

## Citation

Citation information will be added after publication of the associated article and final activation of the dataset DOI.

## License

A repository-level software license will be added after the final release decision. Third-party source images are not covered by any future code license and remain subject to their original reuse conditions.
