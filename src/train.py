"""
Two-Stage Transfer Learning Training Loop
Crop Disease Detection App for Farmers
Author: Khushi Yadav
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.model import CropDiseaseMobileNetV2
from src.dataset import create_stratified_dataloaders


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 5, delta: float = 0.001, path: str = "checkpoints/best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"[*] Validation loss improved to {val_loss:.4f}. Model saved to {self.path}")
        else:
            self.counter += 1
            print(f"[!] EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            total += targets.size(0)

            # Top-1 Accuracy
            _, preds = outputs.max(1)
            correct_top1 += preds.eq(targets).sum().item()

            # Top-3 Accuracy
            _, top3_preds = outputs.topk(3, dim=1)
            correct_top3 += top3_preds.eq(targets.view(-1, 1).expand_as(top3_preds)).sum().item()

    return running_loss / total, correct_top1 / total, correct_top3 / total


def train_model(
    data_dir: str,
    output_dir: str = "checkpoints",
    epochs_head: int = 5,
    epochs_fine_tune: int = 20,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_fine_tune: float = 1e-4,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Prepare data loaders
    train_loader, val_loader, test_loader, classes = create_stratified_dataloaders(
        root_dir=data_dir, batch_size=batch_size
    )
    num_classes = len(classes)

    # Initialize model
    model = CropDiseaseMobileNetV2(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stopping = EarlyStopping(patience=5, path=os.path.join(output_dir, "best_crop_mobilenetv2.pth"))

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc_top1": [],
        "val_acc_top3": [],
    }

    print("\n--- PHASE 1: Training Classification Head (Feature Extractor Frozen) ---")
    model.freeze_feature_extractor()
    optimizer_head = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head, weight_decay=1e-4)

    for epoch in range(1, epochs_head + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_head, device)
        v_loss, v_top1, v_top3 = validate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{epochs_head} [{elapsed:.1f}s] - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc*100:.2f}% | Val Loss: {v_loss:.4f}, Val Top-1: {v_top1*100:.2f}%, Val Top-3: {v_top3*100:.2f}%")
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc_top1"].append(v_top1)
        history["val_acc_top3"].append(v_top3)

    print("\n--- PHASE 2: Fine-Tuning Top Inverted Residual Blocks ---")
    model.unfreeze_top_layers(num_blocks=4)
    optimizer_ft = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_fine_tune, weight_decay=1e-4)
    scheduler_ft = CosineAnnealingLR(optimizer_ft, T_max=epochs_fine_tune, eta_min=1e-6)

    for epoch in range(1, epochs_fine_tune + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_ft, device)
        v_loss, v_top1, v_top3 = validate_epoch(model, val_loader, criterion, device)
        scheduler_ft.step()
        elapsed = time.time() - t0

        print(f"FineTune Epoch {epoch}/{epochs_fine_tune} [{elapsed:.1f}s] - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc*100:.2f}% | Val Loss: {v_loss:.4f}, Val Top-1: {v_top1*100:.2f}%, Val Top-3: {v_top3*100:.2f}%")
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc_top1"].append(v_top1)
        history["val_acc_top3"].append(v_top3)

        early_stopping(v_loss, model)
        if early_stopping.early_stop:
            print("[*] Early stopping triggered. Training completed!")
            break

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on PlantVillage")
    parser.add_argument("--data_dir", type=str, default="data/PlantVillage", help="Path to PlantVillage root directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_head", type=int, default=5)
    parser.add_argument("--epochs_fine_tune", type=int, default=20)
    args = parser.parse_args()

    if Path(args.data_dir).exists():
        train_model(args.data_dir, args.output_dir, args.epochs_head, args.epochs_fine_tune, args.batch_size)
    else:
        print(f"[Notice] Data directory '{args.data_dir}' not found. Download PlantVillage or use mock verification.")
