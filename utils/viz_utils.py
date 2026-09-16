"""
utils/viz_utils.py
------------------
Plotting helpers: training curves, confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from pathlib import Path


CLASS_NAMES = ["Benign", "Malignant"]


# ─────────────────────────────────────────────
# 1. Plot training curves
# ─────────────────────────────────────────────
def plot_training_history(history, save_path: str = None, model_name: str = "Model"):
    """
    Plot accuracy and loss curves from a Keras History object.
    Saves to save_path if provided, else shows interactively.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training History", fontsize=15, fontweight="bold")

    # ── Accuracy ──
    axes[0].plot(history.history["accuracy"],     label="Train Acc",  linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Loss ──
    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Training plot saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────
# 2. Confusion matrix + classification report
# ─────────────────────────────────────────────
def evaluate_model(y_true, y_pred, save_dir: str = None, model_name: str = "Model"):
    """
    Print classification report and plot confusion matrix.

    Parameters
    ----------
    y_true    : array of true integer labels
    y_pred    : array of predicted integer labels
    save_dir  : folder to save the confusion matrix image
    model_name: string label used in the plot title
    """
    print(f"\n{'═'*55}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'═'*55}")

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score   (y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score       (y_true, y_pred, average="weighted", zero_division=0)

    print(f"  Weighted Precision : {precision:.4f}")
    print(f"  Weighted Recall    : {recall:.4f}")
    print(f"  Weighted F1-Score  : {f1:.4f}")
    print(f"{'═'*55}\n")

    # ── Confusion matrix plot ──
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved → {out}")
    else:
        plt.show()

    plt.close(fig)

    return {"precision": precision, "recall": recall, "f1": f1}
