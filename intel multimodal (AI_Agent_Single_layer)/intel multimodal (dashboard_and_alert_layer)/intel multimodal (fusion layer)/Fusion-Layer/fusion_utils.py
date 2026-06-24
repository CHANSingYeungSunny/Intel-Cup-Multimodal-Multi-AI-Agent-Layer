"""
fusion_utils.py — Metrics, logging, confusion matrix, CSV serialization, plotting.

Reuses patterns from the Physiological Layer utils.py with adaptations for
the Multimodal Fusion Layer (1024-dim concatenated features).
"""

import os
import json
import random
import logging
import numpy as np
import pandas as pd

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across numpy, torch, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "logs", name: str = "fusion") -> logging.Logger:
    """Create a logger that writes to both console and a timestamped file."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, loss: float = None) -> dict:
    """Compute classification metrics.

    Args:
        y_true: Ground-truth labels (list or numpy array).
        y_pred: Predicted labels (list or numpy array).
        loss: Optional loss value to include.

    Returns:
        Dict with accuracy, macro/weighted precision/recall/F1, and optional loss.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if loss is not None:
        metrics["loss"] = float(loss)

    return metrics


def confusion_matrix_dict(cm: np.ndarray, class_names: list = None) -> dict:
    """Flatten a confusion matrix into a dict with keys cm_{true}_to_{pred}.

    Args:
        cm: Confusion matrix as numpy array (N_classes x N_classes).
        class_names: List of class name strings.

    Returns:
        Dict with keys like "cm_healthy_to_healthy".
    """
    if class_names is None:
        class_names = ["healthy", "semi_healthy", "unhealthy"]

    n = len(class_names)
    result = {}
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            key = f"cm_{true_name}_to_{pred_name}"
            result[key] = int(cm[i, j]) if i < cm.shape[0] and j < cm.shape[1] else 0
    return result


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def list_to_json(arr) -> str:
    """Serialize a Python list/array to a JSON string."""
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    return json.dumps(arr)


def json_to_list(s: str):
    """Deserialize a JSON string back to a Python list."""
    return json.loads(s)


# ---------------------------------------------------------------------------
# CSV output helpers
# ---------------------------------------------------------------------------

def save_predictions_csv(predictions: list, path: str = "outputs/predictions.csv"):
    """Save predictions CSV with columns: filename, prediction, label, feature_vector.

    Args:
        predictions: List of dicts with keys:
            - filename (str)
            - prediction (int)
            - label (int)
            - feature_vector (list or numpy array, 256-dim fusion embedding)
        path: Output CSV path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows = []
    for p in predictions:
        fv = p["feature_vector"]
        if isinstance(fv, (np.ndarray, list)):
            fv = list_to_json(fv)
        rows.append({
            "filename": p["filename"],
            "prediction": p["prediction"],
            "label": p["label"],
            "feature_vector": fv,
        })

    df = pd.DataFrame(rows, columns=["filename", "prediction", "label", "feature_vector"])
    df.to_csv(path, index=False)
    return df


def save_experiment_csv(results: list, path: str = "outputs/experiment_results_with_accuracy.csv"):
    """Save the experiment results CSV with exactly 33 columns.

    Args:
        results: List of experiment result dicts, each containing all 33 columns.
        path: Output CSV path.

    Returns:
        DataFrame that was saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    column_order = [
        "exp_id", "config_label", "gamma", "epochs", "d_model", "n_layers",
        "batch_size", "label_mode", "best_epoch",
        "val_macro_f1_best", "accuracy", "test_accuracy",
        "test_precision_macro", "test_recall_macro", "test_f1_macro",
        "test_precision_weighted", "test_recall_weighted", "test_f1_weighted",
        "test_loss",
        "cm_healthy_to_healthy", "cm_healthy_to_semi", "cm_healthy_to_unhealthy",
        "cm_semi_to_healthy", "cm_semi_to_semi", "cm_semi_to_unhealthy",
        "cm_unhealthy_to_healthy", "cm_unhealthy_to_semi", "cm_unhealthy_to_unhealthy",
        "train_loss_curve", "val_loss_curve", "train_acc_curve", "val_acc_curve",
        "val_f1_curve",
    ]

    list_columns = {
        "train_loss_curve", "val_loss_curve", "train_acc_curve",
        "val_acc_curve", "val_f1_curve",
    }

    rows = []
    for r in results:
        row = {}
        for col in column_order:
            val = r.get(col)
            if col in list_columns and isinstance(val, (list, np.ndarray)):
                val = list_to_json(val)
            row[col] = val
        rows.append(row)

    df = pd.DataFrame(rows, columns=column_order)

    if "test_f1_weighted" in df.columns:
        df = df.sort_values("test_f1_weighted", ascending=False).reset_index(drop=True)

    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, save_path: str = "outputs/fusion_training_curves.png"):
    """Plot and save training curves: loss (left panel) and accuracy/F1 (right panel).

    Args:
        history: Dict with keys: train_loss, val_loss, train_acc, val_acc, val_macro_f1
        save_path: Output PNG path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    if "train_loss" in history:
        ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Fusion Layer — Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if "train_acc" in history:
        ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    if "val_acc" in history:
        ax2.plot(epochs, history["val_acc"], label="Val Acc", marker="s", markersize=3)
    if "val_macro_f1" in history:
        ax2.plot(epochs, history["val_macro_f1"], label="Val Macro F1", marker="^", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Fusion Layer — Accuracy / F1 Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    save_path: str = "outputs/fusion_confusion_matrix.png",
):
    """Plot and save a confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if class_names is None:
        class_names = ["healthy", "semi_healthy", "unhealthy"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Fusion Layer — Confusion Matrix")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Count model parameters
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
