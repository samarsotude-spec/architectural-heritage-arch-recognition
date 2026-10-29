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

The same fixed test set was retained for every reported model comparison. A separate post-hoc ResNet50 sensitivity analysis excludes 21 exact-overlap test images; it does not change the original partitions.

## Models and experiments

The repository includes implementations of:

- ImageNet-pretrained **ResNet50** with partial fine-tuning
- ImageNet-pretrained **DenseNet121** with partial fine-tuning
- ImageNet-pretrained **MobileNetV2** with partial fine-tuning
- ImageNet-pretrained **ConvNeXt-Tiny** with partial fine-tuning (no augmentation)
- **HOG + Linear SVM** conventional baseline
- **ResNet50 with architecture-aware data augmentation**
- Misclassification analysis for the final ResNet50 baseline
- Saved-checkpoint **ResNet50 exact-overlap sensitivity evaluation without retraining**

## Final reported results

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| ResNet50 | **91.98%** | **0.9174** |
| ConvNeXt-Tiny | 91.98% | 0.9176 |
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

### ConvNeXt-Tiny baseline

The validation-selected checkpoint is from epoch **6** (validation loss: `0.35095349990110347`). On the original 1,496-image test set, accuracy is `0.9197860962566845` and macro F1 is `0.917556831228473`.

Confusion matrix, class order **[No Arch, Has Arch]**:

```text
[[565, 54],
 [ 66, 811]]
```

Results: [metrics](results/convnext_tiny/test_metrics.json) and [per-image predictions](results/convnext_tiny/test_predictions.csv).

### ResNet50 exact-overlap sensitivity analysis

The saved ResNet50 baseline checkpoint is evaluated **without retraining**, excluding 21 test images with byte-identical counterparts in training (17) or validation (4), as recorded in the SHA-256 audit.

| Evaluation | Test images | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Original fixed test set | 1,496 | 91.98% | 0.9174 |
| Excluding 21 exact overlaps | 1,475 | 92.61% | 0.9237 |

The sensitivity confusion matrix is `[[552, 54], [55, 814]]`, in class order **[No Arch, Has Arch]**. These results supplement the original evaluation; they are not an independent external test and do not establish building-level independence.

The [results table](results/sensitivity/sensitivity_results.csv) contains the original and 21-exclusion scenarios. The [integrity report](results/sensitivity/integrity_and_comparability.json) records the checkpoint hash, test-image hash agreement, and reproduction of the original reference results.

The public notebook uses the recorded audit pair annotations and rechecks test-image hashes. It does **not** rerun a complete train/validation duplicate search or SIFT audit. Additional scenarios retained in the notebook are for traceability; the 33- and 40-exclusion scenarios include exploratory SIFT candidates and must not be described as confirmed exact duplicates.

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

### ConvNeXt-Tiny

Trainable components:

- `features[7]`
- entire classifier, with the output layer replaced by a two-class linear layer

Parameter counts:

- Total: 27,821,666
- Trainable: 14,292,482
- Frozen: 13,529,184

Pretrained weights: `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`. No augmentation is used in this baseline.

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

Existing baseline scripts remain in `src/`. Revision files are organized as follows:

| Location | Contents |
|---|---|
| `data/fixed_*_split_public.csv` | Original train, validation, and test membership |
| `data/sensitivity_test_ids.csv` | IDs of the 1,475 retained sensitivity samples |
| `data/excluded_21_exact_overlaps.csv` | The 21 excluded samples and recorded audit evidence |
| `src/Arch_ConvNeXtTiny_PUBLIC.ipynb` | ConvNeXt-Tiny training and test evaluation |
| `src/Arch_ResNet50_Sensitivity_PUBLIC.ipynb` | Saved-checkpoint sensitivity evaluation without retraining |
| `results/convnext_tiny/test_metrics.json` | ConvNeXt-Tiny test metrics |
| `results/convnext_tiny/test_predictions.csv` | ConvNeXt-Tiny per-image predictions |
| `results/sensitivity/original_test_predictions.csv` | Original 1,496-image ResNet50 predictions |
| `results/sensitivity/sensitivity_test_predictions.csv` | Retained 1,475-image ResNet50 predictions |
| `results/sensitivity/sensitivity_results.csv` | Original and 21-exclusion results |
| `results/sensitivity/integrity_and_comparability.json` | Recorded integrity and comparability checks |
| `results/sensitivity/README.md` | Sensitivity documentation |

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

The associated dataset record is [Mendeley Data: t3txgj35sk](https://data.mendeley.com/datasets/t3txgj35sk). Source and reuse documentation belongs with the dataset metadata; image-specific unknowns should not be interpreted as permission to redistribute images. For reproducible citation, use the DOI of the published dataset version containing the files used in the study; this README does not confirm publication of the latest draft updates.

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

### ConvNeXt-Tiny notebook

Open [src/Arch_ConvNeXtTiny_PUBLIC.ipynb](src/Arch_ConvNeXtTiny_PUBLIC.ipynb) in Jupyter or Colab. Set `REPO_ROOT`, `DATASET_ROOT`, `SPLIT_ROOT`, and `OUTPUT_DIR` in its configuration cell. Supply the original images and fixed split CSVs, then run the cells in order.

**Running all cells trains a new model.** New outputs default to `runs/convnext_tiny/`, separate from the deposited reference results. The notebook saves a checkpoint, training history, training configuration, test metrics, and test predictions. It may download the ImageNet pretrained weights on first use.

### ResNet50 sensitivity notebook

Open [src/Arch_ResNet50_Sensitivity_PUBLIC.ipynb](src/Arch_ResNet50_Sensitivity_PUBLIC.ipynb). Configure `CKPT` and `DIRS` in the first code cell to point to the original saved ResNet50 checkpoint and image directories. Run cells in order; no training is performed. Use scenario `exclude_21_exact_any_overlap` for the manuscript sensitivity analysis.

The original images and saved checkpoint must be supplied separately; they are not embedded in the notebook. The recorded checkpoint hash is available in the integrity report. Download `best_resnet50_baseline.pth` from the **Assets** section of [Release v1.0.0](https://github.com/samarsotude-spec/architectural-heritage-arch-recognition/releases/tag/v1.0.0). Set `CKPT` in the sensitivity notebook to the downloaded file's local path.

Both notebooks require `torch`, `torchvision`, `numpy`, `pandas`, `Pillow`, and `scikit-learn`, plus a notebook interface. Exact original package versions were not recorded in the supplied ConvNeXt-Tiny notebook; it is not a pinned execution environment.

## Reproducibility note

The repository uses the deposited split CSV files directly rather than regenerating the train/validation/test partition from directory order. This avoids accidental variation caused by filesystem ordering and preserves the exact experimental membership used in the study.

The fixed test set is never used for model selection.

## Data and code availability

Code and evaluation outputs are organized at the paths listed above. Dataset metadata and fixed partitions are associated with the [Mendeley Data record](https://data.mendeley.com/datasets/t3txgj35sk). Check the selected published version for the required files; draft additions are not evidence of public availability.

## Citation

Article citation information will be added after publication. Cite the specific published Mendeley Data version used, rather than a draft URL.

## License

A repository-level software license will be added after the final release decision. Third-party source images are not covered by any future code license and remain subject to their original reuse conditions.
