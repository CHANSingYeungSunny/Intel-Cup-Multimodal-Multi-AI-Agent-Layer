"""
fusion_train.py — Training loop for the Multimodal Fusion Layer.

Features:
  - AMP (Automatic Mixed Precision) for memory efficiency
  - Gradient checkpointing for large batch sizes on 4GB GPU
  - Early stopping (patience=3) on validation macro F1
  - TensorBoard logging
  - 5 predefined experiment configurations
  - Reproducible train/val/test splits (seed=42)

Usage:
  python fusion_train.py --exp_id 1          # Baseline (3-class, bs=64, LR=3e-4)
  python fusion_train.py --exp_id 2          # Binary (bs=64, LR=3e-4) — expected best
  python fusion_train.py --exp_id 3          # Batch128 (3-class, bs=128, LR=3e-4)
  python fusion_train.py --exp_id 4          # Freeze encoder layer 0
  python fusion_train.py --exp_id 5          # Lower LR (3-class, bs=64, LR=1e-4)
  python fusion_train.py --exp_id all        # Run all 5 experiments
"""

import os
import sys
import json
import time
import argparse
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Project imports
from fusion_utils import (
    set_seed, setup_logging, get_device, count_parameters,
    compute_metrics, confusion_matrix_dict, list_to_json,
    save_predictions_csv, save_experiment_csv,
    plot_training_curves, plot_confusion_matrix,
)
from fusion_preprocess import (
    ID2LABEL_3CLASS, ID2LABEL_BINARY, to_binary_labels, get_class_names,
    split_data, normalize_features,
)
from fusion_loader import load_fused_data
from fusion_model import MultimodalFusionEncoder, FocalLoss, build_model

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    1: {
        "exp_id": 1,
        "config_label": "g2.0_e30_d256_l4_b64_3class",
        "gamma": 2.0,
        "epochs": 30,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff": 512,
        "batch_size": 64,
        "label_mode": "3class",
        "num_classes": 3,
        "lr": 3e-4,
        "freeze_layer_1": False,
        "description": "Baseline: 3-class, bs=64, LR=3e-4",
    },
    2: {
        "exp_id": 2,
        "config_label": "g2.0_e30_d256_l4_b64_binary",
        "gamma": 2.0,
        "epochs": 30,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff": 512,
        "batch_size": 64,
        "label_mode": "binary",
        "num_classes": 2,
        "lr": 3e-4,
        "freeze_layer_1": False,
        "description": "Binary: healthy vs symptomatic, bs=64, LR=3e-4 — expected best",
    },
    3: {
        "exp_id": 3,
        "config_label": "g2.0_e30_d256_l4_b128_3class",
        "gamma": 2.0,
        "epochs": 30,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff": 512,
        "batch_size": 128,
        "label_mode": "3class",
        "num_classes": 3,
        "lr": 3e-4,
        "freeze_layer_1": False,
        "description": "Batch128: 3-class, bs=128, LR=3e-4",
    },
    4: {
        "exp_id": 4,
        "config_label": "g2.0_e30_d256_l4_b64_freeze",
        "gamma": 2.0,
        "epochs": 30,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff": 512,
        "batch_size": 64,
        "label_mode": "3class",
        "num_classes": 3,
        "lr": 3e-4,
        "freeze_layer_1": True,
        "description": "Freeze encoder layer 0: 3-class, bs=64, LR=3e-4",
    },
    5: {
        "exp_id": 5,
        "config_label": "g2.0_e30_d256_l4_b64_lr1e4",
        "gamma": 2.0,
        "epochs": 30,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff": 512,
        "batch_size": 64,
        "label_mode": "3class",
        "num_classes": 3,
        "lr": 1e-4,
        "freeze_layer_1": False,
        "description": "Lower LR: 3-class, bs=64, LR=1e-4",
    },
}


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model, loader, criterion, optimizer, scaler, device, use_amp: bool,
    logger, epoch: int, writer: SummaryWriter = None, global_step: int = 0,
):
    """Run one training epoch with optional AMP."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=False)
    for batch_idx, (xb, yb) in enumerate(pbar):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1).detach().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.cpu().tolist())

        acc = (preds == yb.cpu()).float().mean().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

        if writer and batch_idx % 10 == 0:
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            global_step += 1

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, loss=avg_loss)
    return metrics, global_step


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_epoch(model, loader, criterion, device, use_amp: bool, logger):
    """Run one validation epoch."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for xb, yb in tqdm(loader, desc="Val", leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
        else:
            logits = model(xb)
            loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, loss=avg_loss)
    return metrics, all_labels, all_preds


# ---------------------------------------------------------------------------
# Test evaluation with feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_with_features(
    model, loader, criterion, device, use_amp: bool,
    filenames: np.ndarray, logger,
):
    """Run test evaluation and extract fusion embeddings for predictions.csv."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_features = [], [], []

    for xb, yb in tqdm(loader, desc="Test", leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits, feats = model(xb, return_features=True)
                loss = criterion(logits, yb)
        else:
            logits, feats = model(xb, return_features=True)
            loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.cpu().tolist())
        all_features.append(feats.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_features = np.concatenate(all_features, axis=0)
    metrics = compute_metrics(all_labels, all_preds, loss=avg_loss)

    # Build predictions list
    predictions = []
    for i in range(len(all_labels)):
        name = str(filenames[i]) if filenames is not None and i < len(filenames) else f"sample_{i:04d}"
        predictions.append({
            "filename": name,
            "prediction": int(all_preds[i]),
            "label": int(all_labels[i]),
            "feature_vector": all_features[i].tolist(),
        })

    return metrics, all_labels, all_preds, predictions


# ---------------------------------------------------------------------------
# Main training routine for one experiment
# ---------------------------------------------------------------------------

def run_experiment(exp_config: dict, args, logger) -> dict:
    """Run a single experiment and return its result dict.

    Returns:
        Result dict with all 33 columns for experiment_results_with_accuracy.csv.
    """
    exp_id = exp_config["exp_id"]
    config_label = exp_config["config_label"]
    num_classes = exp_config["num_classes"]
    binary = (num_classes == 2)
    label_mode = exp_config["label_mode"]

    logger.info(f"{'='*60}")
    logger.info(f"Experiment {exp_id}: {exp_config['description']}")
    logger.info(f"  Config: {config_label}")
    logger.info(f"  Classes: {num_classes} ({label_mode})")
    logger.info(f"  Batch size: {exp_config['batch_size']}, LR: {exp_config['lr']}")
    logger.info(f"  Freeze layer 1: {exp_config['freeze_layer_1']}")
    logger.info(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Setup devices and seed
    # ------------------------------------------------------------------
    set_seed(args.seed)
    device = get_device()
    use_amp = (device.type == "cuda")
    logger.info(f"Device: {device}, AMP: {use_amp}")

    # ------------------------------------------------------------------
    # 2. Load fused data
    # ------------------------------------------------------------------
    X, y, filenames = load_fused_data(seed=args.seed)
    logger.info(f"Fused data: {X.shape}, labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ------------------------------------------------------------------
    # 3. Convert to binary if needed
    # ------------------------------------------------------------------
    if binary:
        y = to_binary_labels(y)
        logger.info(f"Binary labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ------------------------------------------------------------------
    # 4. Train/val/test split
    # ------------------------------------------------------------------
    split = split_data(X, y, filenames, seed=args.seed)
    split = normalize_features(split)
    logger.info(f"Split: train={split['X_train'].shape[0]}, "
                f"val={split['X_val'].shape[0]}, test={split['X_test'].shape[0]}")

    # ------------------------------------------------------------------
    # 5. Build DataLoaders
    # ------------------------------------------------------------------
    # WeightedRandomSampler for class imbalance
    train_labels = split["y_train"]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_ds = TensorDataset(
        torch.from_numpy(split["X_train"]),
        torch.from_numpy(split["y_train"]).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(split["X_val"]),
        torch.from_numpy(split["y_val"]).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(split["X_test"]),
        torch.from_numpy(split["y_test"]).long(),
    )

    bs = exp_config["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"DataLoaders: train={len(train_loader)} batches, "
                f"val={len(val_loader)}, test={len(test_loader)}")

    # ------------------------------------------------------------------
    # 6. Build model
    # ------------------------------------------------------------------
    model = build_model(
        d_model=exp_config["d_model"],
        n_heads=exp_config["n_heads"],
        n_layers=exp_config["n_layers"],
        d_ff=exp_config["d_ff"],
        num_classes=num_classes,
        use_checkpoint=args.grad_checkpoint,
    )
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Freeze encoder layer 0 if requested
    if exp_config["freeze_layer_1"] and exp_config["n_layers"] >= 2:
        model.freeze_encoder_layer(0)

    # ------------------------------------------------------------------
    # 7. Loss and optimizer
    # ------------------------------------------------------------------
    alpha = None
    if args.use_class_weights:
        raw_weights = 1.0 / class_counts.astype(np.float32)
        normalized = raw_weights * len(class_counts) / raw_weights.sum()
        alpha = torch.tensor(normalized, dtype=torch.float32).to(device)
        logger.info(f"Class weights: {normalized.tolist()}")

    criterion = FocalLoss(alpha=alpha, gamma=exp_config["gamma"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp_config["lr"], weight_decay=1e-4,
    )

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ------------------------------------------------------------------
    # 8. TensorBoard
    # ------------------------------------------------------------------
    log_dir = os.path.join(args.log_dir, f"exp_{exp_id:02d}")
    writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------------------------------------------
    # 9. Training loop
    # ------------------------------------------------------------------
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    early_stop_patience = args.early_stop_patience
    max_epochs = exp_config["epochs"]
    global_step = 0

    history = defaultdict(list)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, use_amp, logger, epoch, writer, global_step,
        )
        # Validate
        val_metrics, val_labels, val_preds = val_epoch(
            model, val_loader, criterion, device, use_amp, logger,
        )

        epoch_time = time.time() - epoch_start

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["f1_macro"])

        # TensorBoard: epoch-level logging
        writer.add_scalars("loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["loss"],
        }, epoch)
        writer.add_scalars("accuracy", {
            "train": train_metrics["accuracy"],
            "val": val_metrics["accuracy"],
        }, epoch)
        writer.add_scalar("val/macro_f1", val_metrics["f1_macro"], epoch)

        # Log
        logger.info(
            f"Epoch {epoch:02d}/{max_epochs} | "
            f"T_loss={train_metrics['loss']:.4f} V_loss={val_metrics['loss']:.4f} | "
            f"T_acc={train_metrics['accuracy']:.4f} V_acc={val_metrics['accuracy']:.4f} | "
            f"V_F1={val_metrics['f1_macro']:.4f} | "
            f"Time={epoch_time:.1f}s"
        )

        # Early stopping
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  >>> New best model (val F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement (patience {patience_counter}/{early_stop_patience})")
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    writer.close()

    # ------------------------------------------------------------------
    # 10. Load best model and evaluate on test set
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)
    test_metrics, test_labels, test_preds, predictions = test_with_features(
        model, test_loader, criterion, device, use_amp,
        split.get("filenames_test"), logger,
    )

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    num_actual_classes = cm.shape[0]

    class_names = get_class_names(num_actual_classes)
    cm_dict = confusion_matrix_dict(cm, class_names[:3])

    # If binary, pad confusion matrix to 3x3
    if num_classes == 2:
        cm_3x3 = np.zeros((3, 3), dtype=int)
        cm_3x3[:2, :2] = cm
        cm_dict = confusion_matrix_dict(cm_3x3, ["healthy", "semi_healthy", "unhealthy"])

    logger.info(f"\nTest Results (best epoch={best_epoch}):")
    logger.info(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1:    {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Weighted F1: {test_metrics['f1_weighted']:.4f}")
    logger.info(f"  Loss:        {test_metrics['loss']:.4f}")

    # ------------------------------------------------------------------
    # 11. Save per-experiment outputs
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # Save checkpoint
    checkpoint_path = os.path.join(args.output_dir, f"best_fusion_exp{exp_id:02d}.pt")
    id2label = ID2LABEL_BINARY if binary else ID2LABEL_3CLASS
    torch.save({
        "model_state_dict": best_state,
        "config": {
            "d_model": exp_config["d_model"],
            "n_layers": exp_config["n_layers"],
            "n_heads": exp_config["n_heads"],
            "d_ff": exp_config["d_ff"],
            "num_classes": num_classes,
            "id2label": id2label,
            "label_mode": label_mode,
        },
        "best_epoch": best_epoch,
        "val_macro_f1_best": best_val_f1,
        "history": dict(history),
    }, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Save predictions
    pred_path = os.path.join(args.output_dir, f"predictions_exp{exp_id:02d}.csv")
    save_predictions_csv(predictions, pred_path)
    logger.info(f"Predictions saved: {pred_path} ({len(predictions)} rows)")

    # Save plots
    curves_path = os.path.join(args.output_dir, f"fusion_training_curves_exp{exp_id:02d}.png")
    plot_training_curves(dict(history), curves_path)
    logger.info(f"Training curves saved: {curves_path}")

    cm_path = os.path.join(args.output_dir, f"fusion_confusion_matrix_exp{exp_id:02d}.png")
    cm_plot = cm if num_classes > 2 else cm_3x3[:num_actual_classes, :num_actual_classes]
    plot_confusion_matrix(cm_plot, class_names[:num_actual_classes], cm_path)
    logger.info(f"Confusion matrix saved: {cm_path}")

    # ------------------------------------------------------------------
    # 12. Build result dict
    # ------------------------------------------------------------------
    result = {
        "exp_id": exp_id,
        "config_label": config_label,
        "gamma": exp_config["gamma"],
        "epochs": epoch,
        "d_model": exp_config["d_model"],
        "n_layers": exp_config["n_layers"],
        "batch_size": exp_config["batch_size"],
        "label_mode": "binary" if binary else label_mode,
        "best_epoch": best_epoch,
        "val_macro_f1_best": best_val_f1,
        "accuracy": test_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_precision_weighted": test_metrics["precision_weighted"],
        "test_recall_weighted": test_metrics["recall_weighted"],
        "test_f1_weighted": test_metrics["f1_weighted"],
        "test_loss": test_metrics["loss"],
        **cm_dict,
        "train_loss_curve": history["train_loss"],
        "val_loss_curve": history["val_loss"],
        "train_acc_curve": history["train_acc"],
        "val_acc_curve": history["val_acc"],
        "val_f1_curve": history["val_macro_f1"],
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fusion Layer — Multimodal Transformer training"
    )
    # Experiment selection
    parser.add_argument("--exp_id", type=str, default="1",
                        help="Experiment ID (1-5) or 'all' to run all experiments")
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory for output files")
    parser.add_argument("--predictions_path", type=str, default=None,
                        help="Path for predictions.csv (default: outputs/predictions.csv)")
    parser.add_argument("--log_dir", type=str, default="runs/fusion",
                        help="Directory for TensorBoard logs")
    # Training
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 = main process)")
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Early stopping patience")
    parser.add_argument("--grad_checkpoint", action="store_true", default=True,
                        help="Enable gradient checkpointing on encoder")
    parser.add_argument("--no_grad_checkpoint", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--use_class_weights", action="store_true", default=True,
                        help="Use class weights in focal loss")

    args = parser.parse_args()

    # Handle --no_grad_checkpoint
    if args.no_grad_checkpoint:
        args.grad_checkpoint = False

    # Setup logging
    logger = setup_logging(log_dir="logs", name="fusion_train")

    # Determine which experiments to run
    if args.exp_id == "all":
        exp_ids = [1, 2, 3, 4, 5]
    else:
        exp_ids = [int(args.exp_id)]

    logger.info(f"Running experiments: {exp_ids}")

    # Run experiments
    all_results = []
    for eid in exp_ids:
        if eid not in EXPERIMENTS:
            logger.error(f"Unknown experiment ID: {eid}. Valid: 1-5")
            continue

        result = run_experiment(EXPERIMENTS[eid], args, logger)
        all_results.append(result)

    # Save combined results CSV
    if all_results:
        csv_path = os.path.join(args.output_dir, "experiment_results_with_accuracy.csv")
        df = save_experiment_csv(all_results, csv_path)
        logger.info(f"\nExperiment results saved: {csv_path} ({len(df)} rows)")
        logger.info("\nResults summary:")
        cols = ["exp_id", "config_label", "best_epoch", "test_accuracy",
                "test_f1_macro", "test_f1_weighted", "val_macro_f1_best"]
        logger.info(f"\n{df[cols].to_string(index=False)}")

        # Select best experiment by weighted F1
        best_idx = df["test_f1_weighted"].idxmax()
        best_row = df.iloc[best_idx]
        best_exp_id = int(best_row["exp_id"])
        logger.info(f"\nBest experiment: exp_id={best_exp_id} "
                    f"(test_f1_weighted={best_row['test_f1_weighted']:.4f})")

        import shutil
        out = args.output_dir

        def _copy_if(src, dst):
            if os.path.exists(src):
                shutil.copy2(src, dst)
                logger.info(f"  Copied: {os.path.basename(dst)}")

        # Best checkpoint
        _copy_if(
            os.path.join(out, f"best_fusion_exp{best_exp_id:02d}.pt"),
            os.path.join(out, "best_fusion.pt"),
        )

        # Best predictions
        dst_pred = args.predictions_path or os.path.join(out, "predictions.csv")
        _copy_if(
            os.path.join(out, f"predictions_exp{best_exp_id:02d}.csv"),
            dst_pred,
        )

        # Best training curves
        _copy_if(
            os.path.join(out, f"fusion_training_curves_exp{best_exp_id:02d}.png"),
            os.path.join(out, "fusion_training_curves.png"),
        )

        # Best confusion matrix
        _copy_if(
            os.path.join(out, f"fusion_confusion_matrix_exp{best_exp_id:02d}.png"),
            os.path.join(out, "fusion_confusion_matrix.png"),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
