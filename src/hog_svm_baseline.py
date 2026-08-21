"""
HOG + Linear SVM baseline for binary arch recognition.

Configuration aligned with the final reported experiment:
- Fixed Train/Validation/Test split supplied as CSV files
- Input resized to 224 x 224
- Grayscale HOG representation
- orientations = 9
- pixels_per_cell = (16, 16)
- cells_per_block = (2, 2)
- block_norm = "L2-Hys"
- HOG dimensionality = 6,084 features per image
- StandardScaler fitted on training features only
- Linear SVM regularization selected using validation data only
- Final selected C = 0.001
- Test set used only for final evaluation
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


IMG_SIZE = 224
HOG_ORIENTATIONS = 9
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (2, 2)
BLOCK_NORM = "L2-Hys"
EXPECTED_HOG_FEATURES = 6084
FINAL_C = 0.001


def load_split(csv_path):
    df = pd.read_csv(csv_path).reset_index(drop=True)
    required = {
        "original_filename",
        "class_name",
        "label",
        "split",
        "relative_reference",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    return df


def resolve_path(row, dataset_root):
    dataset_root = Path(dataset_root)

    relative = Path(str(row["relative_reference"]))
    candidate = dataset_root / relative
    if candidate.exists():
        return candidate

    class_dir = "has_arch" if int(row["label"]) == 1 else "no_arch"
    filename = str(row["original_filename"])

    candidates = [
        dataset_root / class_dir / filename,
        dataset_root / class_dir / class_dir / filename,
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Image not found for sample {row.get('sample_id', '')}: {filename}"
    )


def extract_hog_from_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    image = np.asarray(image, dtype=np.float32) / 255.0

    features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm=BLOCK_NORM,
        feature_vector=True,
    )

    if features.shape[0] != EXPECTED_HOG_FEATURES:
        raise RuntimeError(
            f"Unexpected HOG dimensionality: {features.shape[0]} "
            f"(expected {EXPECTED_HOG_FEATURES})"
        )

    return features.astype(np.float32)


def extract_split_features(df, dataset_root, split_name):
    features = []
    labels = []

    print(f"Extracting HOG features for {split_name}: {len(df)} images")

    for idx, row in df.iterrows():
        image_path = resolve_path(row, dataset_root)
        features.append(extract_hog_from_image(image_path))
        labels.append(int(row["label"]))

        if (idx + 1) % 250 == 0 or (idx + 1) == len(df):
            print(f"  {idx + 1}/{len(df)}")

    X = np.vstack(features)
    y = np.asarray(labels, dtype=np.int64)

    print(f"{split_name} feature matrix: {X.shape}")
    return X, y


def evaluate_model(model, X, y):
    pred = model.predict(X)
    scores = model.decision_function(X)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y,
            pred,
            labels=[0, 1],
            target_names=["No Arch", "Has Arch"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, pred, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-csv", default="data/fixed_train_split_public.csv")
    parser.add_argument("--val-csv", default="data/fixed_validation_split_public.csv")
    parser.add_argument("--test-csv", default="data/fixed_test_split_public.csv")
    parser.add_argument("--output-dir", default="outputs/hog_svm_baseline")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.train_csv)
    val_df = load_split(args.val_csv)
    test_df = load_split(args.test_csv)

    # Exact fixed split sizes used throughout the study.
    assert len(train_df) == 5235
    assert len(val_df) == 748
    assert len(test_df) == 1496

    X_train, y_train = extract_split_features(
        train_df, args.dataset_root, "Train"
    )
    X_val, y_val = extract_split_features(
        val_df, args.dataset_root, "Validation"
    )
    X_test, y_test = extract_split_features(
        test_df, args.dataset_root, "Test"
    )

    # IMPORTANT: fit the scaler using training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Final validation-selected setting reported in the study.
    model = LinearSVC(
        C=FINAL_C,
        random_state=42,
        max_iter=20000,
    )
    model.fit(X_train_scaled, y_train)

    val_pred = model.predict(X_val_scaled)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro")

    print(f"\nValidation Macro F1 at C={FINAL_C}: {val_macro_f1:.4f}")

    metrics, test_pred, test_scores = evaluate_model(
        model, X_test_scaled, y_test
    )

    print("\nFINAL HOG + LINEAR SVM TEST RESULTS")
    print("-----------------------------------")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"ROC AUC           : {metrics['roc_auc']:.4f}")
    print(f"Average Precision : {metrics['average_precision']:.4f}")
    print("Confusion Matrix [No Arch, Has Arch]:")
    print(np.asarray(metrics["confusion_matrix"]))

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions = test_df.copy()
    predictions["true_label"] = y_test
    predictions["predicted_label"] = test_pred
    predictions["decision_score"] = test_scores
    predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
