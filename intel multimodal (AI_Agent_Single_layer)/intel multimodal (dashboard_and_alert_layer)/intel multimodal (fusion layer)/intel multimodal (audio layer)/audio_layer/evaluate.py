# =====================================================================
# Evaluation & Metrics
#
# Runs final test-set evaluation, computes all sklearn metrics, saves
# confusion matrix and training curves as PNGs, and writes the
# experiment_results_with_accuracy.csv file.
# =====================================================================

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from . import config as cfg


# ---------------------------------------------------------------------
# Test-set evaluation (collects CLS embeddings + filenames for export)
# ---------------------------------------------------------------------
def evaluate_test_set(model, test_loader, device=None):
    """
    Run inference on the test set, collecting predictions, ground truth,
    CLS embeddings, and filenames.

    Returns
    -------
    test_loss : float
    y_true, y_pred : list[int]
    cls_embeddings : np.ndarray  shape (N, d_model)
    filenames : list[str]
    """
    if device is None:
        device = cfg.DEVICE

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    y_true, y_pred = [], []
    all_cls = []
    all_filenames = []

    pbar = tqdm(test_loader, desc="Evaluating test set", leave=False)
    with torch.no_grad():
        for xb, yb, fnames in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            logits, cls_embedding = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)

            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            all_cls.append(cls_embedding.cpu().numpy())
            all_filenames.extend(fnames)

    test_loss = total_loss / len(test_loader.dataset)
    cls_embeddings = np.concatenate(all_cls, axis=0)

    return test_loss, y_true, y_pred, cls_embeddings, all_filenames


# ---------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------
def evaluate_model(
    model,
    test_loader,
    history: dict,
    device: str = None,
    output_dir: str = None,
):
    """
    Evaluate the model on the test set:
      - Print all metrics
      - Save confusion matrix PNG
      - Save training curves PNG
      - Write experiment_results_with_accuracy.csv

    Returns
    -------
    metrics : dict
        All computed metrics.
    eval_data : tuple
        (y_true, y_pred, cls_embeddings, filenames) for downstream export.
    """
    if device is None:
        device = cfg.DEVICE
    if output_dir is None:
        output_dir = cfg.OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    # --- Run test evaluation --------------------------------------------
    print("\n--- Loading Best Model for Test Set Evaluation ---")
    test_loss, y_true, y_pred, cls_embeddings, filenames = evaluate_test_set(
        model, test_loader, device
    )

    # --- Compute metrics ------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n[Overall Metrics]")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Macro Precision    : {prec_macro:.4f}")
    print(f"Macro Recall       : {rec_macro:.4f}")
    print(f"Macro F1-score     : {f1_macro:.4f}")
    print(f"Weighted Precision : {prec_weighted:.4f}")
    print(f"Weighted Recall    : {rec_weighted:.4f}")
    print(f"Weighted F1-score  : {f1_weighted:.4f}")
    print(f"Test Loss          : {test_loss:.4f}")

    # Classification report
    target_names = [cfg.ID2LABEL[i] for i in range(cfg.NUM_CLASSES)]
    print("\n[Classification Report]")
    report = classification_report(
        y_true, y_pred,
        labels=list(range(cfg.NUM_CLASSES)),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.NUM_CLASSES)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{cfg.ID2LABEL[i]}" for i in range(cfg.NUM_CLASSES)],
        columns=[f"Pred_{cfg.ID2LABEL[i]}" for i in range(cfg.NUM_CLASSES)],
    )
    print("\n[Confusion Matrix]")
    print(cm_df)

    # --- Save confusion matrix plot -------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(cfg.NUM_CLASSES))
        ax.set_yticks(range(cfg.NUM_CLASSES))
        ax.set_xticklabels(target_names, rotation=45, ha="right")
        ax.set_yticklabels(target_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        for i in range(cfg.NUM_CLASSES):
            for j in range(cfg.NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"\nConfusion matrix saved to: {cm_path}")
    except Exception as e:
        print(f"Confusion matrix plot skipped: {e}")

    # --- Save training curves -------------------------------------------
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(history["train_loss"], label="Train Loss")
        ax1.plot(history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("AST Loss Curve")
        ax1.legend()

        ax2.plot(history["train_acc"], label="Train Acc")
        ax2.plot(history["val_acc"], label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("AST Accuracy Curve")
        ax2.legend()

        plt.tight_layout()
        curves_path = os.path.join(output_dir, "training_curves.png")
        fig.savefig(curves_path, dpi=150)
        plt.close(fig)
        print(f"Training curves saved to: {curves_path}")
    except Exception as e:
        print(f"Training curves plot skipped: {e}")

    # --- Write experiment results CSV (vision-layer compatible format) ---
    import json as _json

    best_val_acc = max(history["val_acc"]) if history["val_acc"] else float("nan")
    best_epoch = (
        history["val_acc"].index(best_val_acc) + 1
        if history["val_acc"]
        else 0
    )

    # Build config_label matching vision layer convention
    config_label = (
        f"g0_e{cfg.EPOCHS}_d{cfg.D_MODEL}"
        f"_l{cfg.N_LAYERS}_b{cfg.BATCH_SIZE}_coughvid_AST"
    )

    # Flatten confusion matrix with vision-layer column names
    # Vision uses: healthy, semi, unhealthy
    cm_names = ["healthy", "semi", "unhealthy"]
    cm_flat = {}
    for i, true_name in enumerate(cm_names):
        for j, pred_name in enumerate(cm_names):
            col = f"cm_{true_name}_to_{pred_name}"
            cm_flat[col] = int(cm[i, j]) if i < cm.shape[0] and j < cm.shape[1] else 0

    # Training curves as JSON arrays
    train_loss_curve = _json.dumps(history["train_loss"])
    val_loss_curve = _json.dumps(history["val_loss"])
    train_acc_curve = _json.dumps(history["train_acc"])
    val_acc_curve = _json.dumps(history["val_acc"])
    val_f1_curve = _json.dumps(history["val_acc"])  # tracked via accuracy

    results = {
        # --- Vision-compatible columns ---
        "exp_id": 0,               # filled by run_ablation
        "config_label": config_label,
        "gamma": 0,                # no focal loss
        "epochs": cfg.EPOCHS,
        "d_model": cfg.D_MODEL,
        "n_layers": cfg.N_LAYERS,
        "batch_size": cfg.BATCH_SIZE,
        "label_mode": "coughvid",
        "best_epoch": best_epoch,
        "val_macro_f1_best": f1_macro,
        "accuracy": acc,
        "test_accuracy": acc,
        "test_precision_macro": prec_macro,
        "test_recall_macro": rec_macro,
        "test_f1_macro": f1_macro,
        "test_precision_weighted": prec_weighted,
        "test_recall_weighted": rec_weighted,
        "test_f1_weighted": f1_weighted,
        "test_loss": test_loss,
        **cm_flat,
        "train_loss_curve": train_loss_curve,
        "val_loss_curve": val_loss_curve,
        "train_acc_curve": train_acc_curve,
        "val_acc_curve": val_acc_curve,
        "val_f1_curve": val_f1_curve,
        # --- Audio-specific (kept for backward compat) ---
        "best_val_acc": best_val_acc,
        "seed": cfg.SEED,
        "weight_decay": cfg.WEIGHT_DECAY,
        "n_mels": cfg.N_MELS,
        "max_frames": cfg.MAX_FRAMES,
        "hop_length": cfg.HOP_LENGTH,
        "n_heads": cfg.N_HEADS,
        "d_ff": cfg.D_FF,
        "dropout": cfg.DROPOUT,
        "device": device,
    }

    # Write per-experiment CSV (kept for individual inspection)
    results_df = pd.DataFrame([results])
    results_csv = os.path.join(output_dir, "experiment_results_with_accuracy.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Experiment results saved to: {results_csv}")

    metrics = results
    eval_data = (y_true, y_pred, cls_embeddings, filenames)
    return metrics, eval_data
