"""
Architecture-aware augmentation experiment for ResNet50.

This script reproduces the final augmentation protocol used in the study.

Key principles:
- Same fixed Train/Validation/Test partitions as all baseline experiments
- Augmentation is applied ONLY to the training set
- Validation and test sets use deterministic resize + ImageNet normalization
- Mild transformations are used to preserve arch semantics and geometry

Training augmentation:
- Resize to 224 x 224 using bilinear interpolation
- Random horizontal flip, p = 0.5
- Random affine:
    rotation = [-10, +10] degrees
    scale = [0.95, 1.05]
    translation = 0
    shear = 0
    fill = (124, 116, 104)
- Color jitter:
    brightness = 0.15
    contrast = 0.15
- ImageNet normalization

Model/training:
- ImageNet-pretrained ResNet50
- Partial fine-tuning: layer4 + classification head
- Dropout = 0.30
- Batch size = 32
- Max epochs = 20
- AdamW, lr = 1e-4, weight_decay = 1e-4
- Cross-entropy with label smoothing = 0.1
- ReduceLROnPlateau: factor = 0.5, patience = 2, min_lr = 1e-6
- Early stopping patience = 5
- Gradient clipping = 1.0
- Random seed = 42
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 5
GRAD_CLIP = 1.0
NUM_WORKERS = 2


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SplitDataset(Dataset):
    def __init__(self, csv_path, dataset_root, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        required = {
            "original_filename",
            "class_name",
            "label",
            "split",
            "relative_reference",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, row):
        relative = Path(str(row["relative_reference"]))
        candidate = self.dataset_root / relative
        if candidate.exists():
            return candidate

        class_dir = "has_arch" if int(row["label"]) == 1 else "no_arch"
        filename = str(row["original_filename"])

        candidates = [
            self.dataset_root / class_dir / filename,
            self.dataset_root / class_dir / class_dir / filename,
        ]
        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Image not found for sample {row.get('sample_id', '')}: {filename}"
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self._resolve_path(row)
        image = Image.open(image_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


def build_train_transform():
    return transforms.Compose([
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=(-10, 10),
            translate=None,
            scale=(0.95, 1.05),
            shear=None,
            interpolation=InterpolationMode.BILINEAR,
            fill=(124, 116, 104),
        ),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_model(device):
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1
    )

    for p in model.parameters():
        p.requires_grad = False

    for p in model.layer4.parameters():
        p.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(in_features, 2),
    )

    for p in model.fc.parameters():
        p.requires_grad = True

    return model.to(device)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    labels_all = []
    preds_all = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        use_amp = device.type == "cuda"

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        preds_all.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(labels_all, preds_all)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    labels_all = []
    preds_all = []
    probs_all = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        labels_all.extend(labels.numpy().tolist())
        preds_all.extend(preds.cpu().numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())

    labels_all = np.asarray(labels_all)
    preds_all = np.asarray(preds_all)
    probs_all = np.asarray(probs_all)

    metrics = {
        "accuracy": float(accuracy_score(labels_all, preds_all)),
        "macro_f1": float(f1_score(labels_all, preds_all, average="macro")),
        "confusion_matrix": confusion_matrix(
            labels_all, preds_all, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            labels_all,
            preds_all,
            labels=[0, 1],
            target_names=["No Arch", "Has Arch"],
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics, labels_all, preds_all, probs_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-csv", default="data/fixed_train_split_public.csv")
    parser.add_argument("--val-csv", default="data/fixed_validation_split_public.csv")
    parser.add_argument("--test-csv", default="data/fixed_test_split_public.csv")
    parser.add_argument(
        "--output-dir",
        default="outputs/resnet50_architecture_aware_augmentation",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = SplitDataset(
        args.train_csv,
        args.dataset_root,
        build_train_transform(),
    )
    val_ds = SplitDataset(
        args.val_csv,
        args.dataset_root,
        build_eval_transform(),
    )
    test_ds = SplitDataset(
        args.test_csv,
        args.dataset_root,
        build_eval_transform(),
    )

    assert len(train_ds) == 5235
    assert len(val_ds) == 748
    assert len(test_ds) == 1496

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(device)
    total, trainable, frozen = count_parameters(model)

    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {frozen:,}")

    assert total == 23_512_130
    assert trainable == 14_968_834
    assert frozen == 8_543_296

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_resnet50_augmented.pth"

    if args.evaluate_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required with --evaluate-only")

        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda"),
        )

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
            )

            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                device,
            )

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
            })

            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"Val Acc {val_acc:.4f} | "
                f"LR {current_lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "configuration": {
                        "seed": SEED,
                        "image_size": IMG_SIZE,
                        "batch_size": BATCH_SIZE,
                        "max_epochs": MAX_EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "label_smoothing": LABEL_SMOOTHING,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                        "gradient_clip": GRAD_CLIP,
                        "fine_tuning": "layer4 + classifier head",
                        "augmentation": {
                            "horizontal_flip_probability": 0.5,
                            "rotation_degrees": [-10, 10],
                            "scale": [0.95, 1.05],
                            "translation": None,
                            "shear": None,
                            "fill_rgb": [124, 116, 104],
                            "brightness": 0.15,
                            "contrast": 0.15,
                        },
                    },
                }, checkpoint_path)

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

        pd.DataFrame(history).to_csv(
            output_dir / "training_history.csv",
            index=False,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(
            f"Best checkpoint loaded from epoch "
            f"{checkpoint.get('epoch', 'unknown')}."
        )

    metrics, labels, preds, probs = evaluate(model, test_loader, device)

    print("\nARCHITECTURE-AWARE AUGMENTATION TEST RESULTS")
    print("--------------------------------------------")
    print("Test Accuracy :", f"{metrics['accuracy']:.4f}")
    print("Macro F1      :", f"{metrics['macro_f1']:.4f}")
    print("Confusion Matrix [No Arch, Has Arch]:")
    print(np.asarray(metrics["confusion_matrix"]))

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    prediction_df = test_ds.df.copy()
    prediction_df["true_label"] = labels
    prediction_df["predicted_label"] = preds
    prediction_df["prob_no_arch"] = probs[:, 0]
    prediction_df["prob_has_arch"] = probs[:, 1]

    prediction_df.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
