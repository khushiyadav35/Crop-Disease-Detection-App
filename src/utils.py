"""
Utility Functions & Plotting Tools
Crop Disease Detection App for Farmers
Author: Khushi Yadav
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42):
    """Sets random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = "docs/training_curves.png"):
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", color="#2e7d32", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", color="#d84315", label="Validation Loss", linewidth=2)
    ax1.set_title("Cross-Entropy Loss Curve (25 Epochs)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy curve
    val_acc_pct = [acc * 100 for acc in history["val_acc_top1"]]
    ax2.plot(epochs, val_acc_pct, "^-", color="#1565c0", label="Top-1 Validation Accuracy", linewidth=2)
    if "val_acc_top3" in history:
        val_top3_pct = [acc * 100 for acc in history["val_acc_top3"]]
        ax2.plot(epochs, val_top3_pct, "d-.", color="#6a1b9a", label="Top-3 Validation Accuracy", linewidth=2)

    ax2.set_title("Validation Accuracy Progression (%)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(40, 102)
    ax2.legend(frameon=True, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[*] Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    classes: List[str],
    save_path: Optional[str] = "docs/confusion_matrix.png",
    top_n: int = 15,
):
    """
    Plots a high-resolution confusion matrix heatmap for top representative crops.
    """
    # Select top_n representative classes for clean readability in reports/presentations
    clean_labels = [c.replace("___", " - ").replace("_", " ") for c in classes[:top_n]]
    sub_cm = cm[:top_n, :top_n]

    # Normalize to row percentages
    row_sums = sub_cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(sub_cm, row_sums, out=np.zeros_like(sub_cm, dtype=float), where=row_sums != 0) * 100

    plt.figure(figsize=(12, 10), dpi=150)
    sns.heatmap(
        norm_cm,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar_kws={"label": "Classification Accuracy (%)"},
    )
    plt.title("Confusion Matrix Heatmap (Representative Agricultural Classes)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Disease Class", fontsize=12, labelpad=10)
    plt.ylabel("True Disease Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[*] Confusion matrix heatmap saved to {save_path}")
    plt.close()
