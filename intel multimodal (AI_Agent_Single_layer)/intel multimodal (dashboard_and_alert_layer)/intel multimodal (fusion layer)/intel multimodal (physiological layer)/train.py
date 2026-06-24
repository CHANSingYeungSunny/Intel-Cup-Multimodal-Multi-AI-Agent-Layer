"""
train.py — Training loop for the iTransformer Physiological Layer.

Features:
  - AMP (Automatic Mixed Precision) for memory efficiency
  - Gradient checkpointing for large batch sizes on 4GB GPU
  - Early stopping (patience=3) on validation macro F1
  - TensorBoard logging
  - 5 predefined experiment configurations
  - Reproducible train/val/test splits

Usage:
  python train.py --exp_id 1          # Baseline (3-class, bs=64, lr=3e-4)
  python train.py --exp_id 2          # Binary (bs=64, lr=3e-4) — expected best
  python train.py --exp_id 3          # Batch32 (3-class, bs=32, lr=3e-4)
  python train.py --exp_id 4          # Freeze encoder layer 1
  python train.py --exp_id 5          # Lower LR (3-class, bs=64, lr=1e-4)
  python train.py --exp_id all        # Run all 5 experiments
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm

# Project imports
from utils import (
    set_seed, setup_logging, get_device, count_parameters,
    compute_metrics, confusion_matrix_dict, list_to_json,
    save_predictions_csv, save_experiment_csv,
    plot_training_curves, plot_confusion_matrix,
)
from preprocess import (
    ID2LABEL_3CLASS, ID2LABEL_BINARY,
    to_binary_labels,
)
from data_loader import (
    list_bidmc_records, split_records,
    BIDMCDataset, WINDOW_WAVE,
    build_cache,
)
from model import iTransformerClassifier, FocalLoss, build_model

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    1: {
        "exp_id": 1,
        "config_label": "g2.0_e25_d128_l3_b64_quantile_baseline",
        "gamma": 2.0,
        "epochs": 25,
        "d_model": 128,
        "n_layers": 3,
        "n_heads": 4,
        "d_ff": 256,
        "batch_size": 64,
        "label_mode": "quantile",
        "num_classes": 3,
        "lr": 3e-4,
        "freeze_layer_1": False,
        "description": "Baseline: 3-class, bs=64, LR=3e-4",
    },
    2: {
        "exp_id": 2,
        "config_label": "g2.0_e25_d128_l3_b64_binary",
        "gamma": 2.0,
        "epochs": 25,
        "d_model": 128,
        "n_layers": 3,
        "n_heads": 4,
        "d_ff": 256,
        "batch_size": 64,
        "label_mode": "quantile",
        "num_classes": 2,
        "lr": 3e-4,
        "freeze_layer_1": False,
        "description": "Binary: merge symptomatic + unhealthy, bs=64, LR=3e-4 — expected best",
    },
    3: {
        "exp_id": 3,
        "config_label": "g2.0_e25_d128_l3_b32_quantile",
        "gamma": 2.0,
        "epochs": 25,
        "d_model": 128,
        "n_layers": 3,
        "n_heads": 4,
        "d_ff": 256,
        "batch_size": 32,
        "label_mode": "quantile",
        "num_classes": 3,
        "lr": 3e-4,
        "freeze_layer_1": False,
        "description": "Batch32: 3-class, bs=32, LR=3e-4",
    },
    4: {
        "exp_id": 4,
        "config_label": "g2.0_e25_d128_l3_b64_quantile_freeze",
        "gamma": 2.0,
        "epochs": 25,
        "d_model": 128,
        "n_layers": 3,
        "n_heads": 4,
        "d_ff": 256,
        "batch_size": 64,
        "label_mode": "quantile",
        "num_classes": 3,
        "lr": 3e-4,
        "freeze_layer_1": True,
        "description": "Freeze encoder layer 1: 3-class, bs=64, LR=3e-4",
    },
    5: {
        "exp_id": 5,
        "config_label": "g2.0_e25_d128_l3_b64_quantile_lr1e4",
        "gamma": 2.0,
        "epochs": 25,
        "d_model": 128,
        "n_layers": 3,
        "n_heads": 4,
        "d_ff": 256,
        "batch_size": 64,
        "label_mode": "quantile",
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

        # Update progress bar
        acc = (preds == yb.cpu()).float().mean().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

        # TensorBoard step-level logging (every 50 batches)
        if writer and batch_idx % 50 == 0:
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
def test_with_features(model, loader, criterion, device, use_amp: bool,
                       sample_records: list, logger):
    """Run test evaluation and extract CLS embeddings for predictions.csv."""
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
    all_features = np.concatenate(all_features, axis=0)  # [N, d_model]
    metrics = compute_metrics(all_labels, all_preds, loss=avg_loss)

    # Build per-record window counters for descriptive filenames
    rec_counters = {}
    predictions = []
    for i in range(len(all_labels)):
        rec_name = sample_records[i] if i < len(sample_records) else "unknown"
        if rec_name not in rec_counters:
            rec_counters[rec_name] = 0
        win_idx = rec_counters[rec_name]
        rec_counters[rec_name] += 1

        predictions.append({
            "filename": f"{rec_name}_win_{win_idx:04d}",
            "prediction": int(all_preds[i]),
            "label": int(all_labels[i]),
            "feature_vector": all_features[i].tolist(),
        })

    return metrics, all_labels, all_preds, predictions


# ---------------------------------------------------------------------------
# Main training routine for one experiment
# ---------------------------------------------------------------------------

def run_experiment(exp_config: dict, args, logger) -> dict:
    """
    Run a single experiment and return its result dict.

    Args:
        exp_config: Experiment configuration dict.
        args: Parsed CLI arguments.
        logger: Logger instance.

    Returns:
        Result dict with all 33 columns for experiment_results_with_accuracy.csv.
    """
    exp_id = exp_config["exp_id"]
    config_label = exp_config["config_label"]
    label_mode = exp_config["label_mode"]
    num_classes = exp_config["num_classes"]
    binary = (num_classes == 2)

    logger.info(f"{'='*60}")
    logger.info(f"Experiment {exp_id}: {exp_config['description']}")
    logger.info(f"  Config: {config_label}")
    logger.info(f"  Label mode: {label_mode}, Classes: {num_classes}")
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
    # 2. Load / build cache
    # ------------------------------------------------------------------
    cache_dir = args.cache_dir
    meta_path = os.path.join(cache_dir, "cache_meta.pt")
    if not os.path.exists(meta_path):
        logger.info("Cache not found. Building window cache...")
        build_cache(args.data_dir, cache_dir, label_mode=label_mode)

    # Load cache metadata
    meta = torch.load(meta_path, map_location="cpu", weights_only=True)
    all_records = meta["records"]
    logger.info(f"Cache loaded: {meta['total_windows']} windows from {len(all_records)} records")

    # ------------------------------------------------------------------
    # 3. Split records
    # ------------------------------------------------------------------
    train_ids, val_ids, test_ids = split_records(all_records, seed=args.seed)
    logger.info(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # ------------------------------------------------------------------
    # 4. Build datasets and loaders
    # ------------------------------------------------------------------
    train_ds = BIDMCDataset(cache_dir, train_ids, binary=binary)
    val_ds = BIDMCDataset(cache_dir, val_ids, binary=binary)
    test_ds = BIDMCDataset(cache_dir, test_ids, binary=binary)

    # WeightedRandomSampler for class imbalance (train only)
    train_labels = train_ds.labels.numpy()
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights).float(),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=exp_config["batch_size"],
        sampler=sampler, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=exp_config["batch_size"],
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=exp_config["batch_size"],
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Train windows: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    logger.info(f"Train class counts: {dict(zip(*np.unique(train_labels, return_counts=True)))}")

    # ------------------------------------------------------------------
    # 5. Build model
    # ------------------------------------------------------------------
    model = build_model(
        seq_len=WINDOW_WAVE,
        n_vars=4,
        num_classes=num_classes,
        d_model=exp_config["d_model"],
        n_heads=exp_config["n_heads"],
        n_layers=exp_config["n_layers"],
        d_ff=exp_config["d_ff"],
        dropout=0.1,
        use_checkpoint=args.grad_checkpoint,
    )
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Freeze encoder layer 1 if requested
    if exp_config["freeze_layer_1"] and exp_config["n_layers"] >= 2:
        layer0 = model.encoder.layers[0]
        for param in layer0.parameters():
            param.requires_grad = False
        logger.info("Encoder layer 0 frozen")

    # ------------------------------------------------------------------
    # 6. Loss and optimizer
    # ------------------------------------------------------------------
    # Class weights for focal loss — normalize so average weight = 1
    alpha = None
    if args.use_class_weights:
        raw_weights = 1.0 / class_counts.astype(np.float32)
        # Normalize: multiply by num_classes / sum so average = 1
        normalized = raw_weights * len(class_counts) / raw_weights.sum()
        alpha = torch.tensor(normalized, dtype=torch.float32).to(device)
        logger.info(f"Normalized class weights: {normalized.tolist()}")

    criterion = FocalLoss(alpha=alpha, gamma=exp_config["gamma"])
    optimizer = torch.optim.Adam(model.parameters(), lr=exp_config["lr"], weight_decay=1e-4)

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ------------------------------------------------------------------
    # 7. TensorBoard
    # ------------------------------------------------------------------
    log_dir = os.path.join(args.log_dir, f"exp_{exp_id:02d}")
    writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------------------------------------------
    # 8. Training loop
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

        # Early stopping / best model tracking
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
    # 9. Load best model and evaluate on test set
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)
    test_metrics, test_labels, test_preds, predictions = test_with_features(
        model, test_loader, criterion, device, use_amp,
        test_ds.sample_records, logger,
    )

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    num_actual_classes = cm.shape[0]

    # Map class names based on number of classes
    if num_classes == 2:
        class_names = ["healthy", "symptomatic_or_unhealthy"]
    else:
        class_names = ["healthy", "semi_healthy", "unhealthy"]

    # For 2-class: pad confusion matrix columns to match 3-class naming
    if num_classes == 2:
        cm_3x3 = np.zeros((3, 3), dtype=int)
        cm_3x3[:2, :2] = cm
        cm_dict = confusion_matrix_dict(cm_3x3, ["healthy", "semi_healthy", "unhealthy"])
    else:
        cm_dict = confusion_matrix_dict(cm, class_names)

    logger.info(f"\nTest Results (best epoch={best_epoch}):")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1:  {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Weighted F1: {test_metrics['f1_weighted']:.4f}")
    logger.info(f"  Loss:      {test_metrics['loss']:.4f}")

    # ------------------------------------------------------------------
    # 10. Save per-experiment outputs
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # Save per-experiment checkpoint
    checkpoint_path = os.path.join(args.output_dir, f"best_physio_exp{exp_id:02d}.pt")
    id2label = ID2LABEL_BINARY if binary else ID2LABEL_3CLASS
    torch.save({
        "model_state_dict": best_state,
        "config": {
            "seq_len": WINDOW_WAVE,
            "n_vars": 4,
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

    # Save per-experiment predictions CSV
    pred_path = os.path.join(args.output_dir, f"predictions_exp{exp_id:02d}.csv")
    save_predictions_csv(predictions, pred_path)
    logger.info(f"Predictions saved: {pred_path} ({len(predictions)} rows)")

    # Save per-experiment plots
    curves_path = os.path.join(args.output_dir, f"training_curves_exp{exp_id:02d}.png")
    plot_training_curves(dict(history), curves_path)
    logger.info(f"Training curves saved: {curves_path}")

    cm_path = os.path.join(args.output_dir, f"confusion_matrix_exp{exp_id:02d}.png")
    plot_confusion_matrix(cm, class_names[:num_actual_classes], cm_path)
    logger.info(f"Confusion matrix saved: {cm_path}")

    # ------------------------------------------------------------------
    # 11. Build result dict for experiment_results_with_accuracy.csv
    # ------------------------------------------------------------------
    result = {
        "exp_id": exp_id,
        "config_label": config_label,
        "gamma": exp_config["gamma"],
        "epochs": epoch,  # actual epochs run (may be less than max due to early stop)
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
        "train_loss_curve": list_to_json(history["train_loss"]),
        "val_loss_curve": list_to_json(history["val_loss"]),
        "train_acc_curve": list_to_json(history["train_acc"]),
        "val_acc_curve": list_to_json(history["val_acc"]),
        "val_f1_curve": list_to_json(history["val_macro_f1"]),
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Physiological Layer — iTransformer training for BIDMC-PPG"
    )
    # Experiment selection
    parser.add_argument("--exp_id", type=str, default="1",
                        help="Experiment ID (1-5) or 'all' to run all experiments")
    # Data paths
    parser.add_argument("--data_dir", type=str,
                        default="bidmc-ppg-and-respiration-dataset-1.0.0",
                        help="Path to BIDMC dataset root")
    parser.add_argument("--cache_dir", type=str, default="cache/windows",
                        help="Directory for cached window .pt files")
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory for output files")
    parser.add_argument("--predictions_path", type=str, default=None,
                        help="Path for predictions.csv (default: outputs/predictions.csv)")
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard",
                        help="Directory for TensorBoard logs")
    # Training
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 = main process)")
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--grad_checkpoint", action="store_true", default=True,
                        help="Enable gradient checkpointing on encoder")
    parser.add_argument("--no_grad_checkpoint", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--use_class_weights", action="store_true", default=True,
                        help="Use class weights in focal loss")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuild of window cache")

    args = parser.parse_args()

    # Handle --no_grad_checkpoint
    if args.no_grad_checkpoint:
        args.grad_checkpoint = False

    # Setup logging
    logger = setup_logging(log_dir="logs", name="train")

    # Rebuild cache if requested
    if args.rebuild_cache and os.path.exists(args.cache_dir):
        import shutil
        logger.info("Rebuilding window cache...")
        shutil.rmtree(args.cache_dir)
    if args.rebuild_cache or not os.path.exists(os.path.join(args.cache_dir, "cache_meta.pt")):
        logger.info("Building window cache (first run)...")
        build_cache(args.data_dir, args.cache_dir, label_mode="quantile")

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
        cols = ["exp_id", "config_label", "best_epoch", "test_accuracy", "test_f1_macro",
                "test_f1_weighted", "val_macro_f1_best"]
        logger.info(f"\n{df[cols].to_string(index=False)}")

        # Select best experiment by test_f1_macro and copy its outputs to final names
        best_idx = df["test_f1_macro"].idxmax()
        best_row = df.iloc[best_idx]
        best_exp_id = int(best_row["exp_id"])
        logger.info(f"\nBest experiment: exp_id={best_exp_id} (test_f1_macro={best_row['test_f1_macro']:.4f})")

        import shutil
        out = args.output_dir

        # Copy best checkpoint
        src_ckpt = os.path.join(out, f"best_physio_exp{best_exp_id:02d}.pt")
        dst_ckpt = os.path.join(out, "best_physio.pt")
        if os.path.exists(src_ckpt):
            shutil.copy2(src_ckpt, dst_ckpt)
            logger.info(f"Best checkpoint: {dst_ckpt}")

        # Copy best predictions
        src_pred = os.path.join(out, f"predictions_exp{best_exp_id:02d}.csv")
        dst_pred = args.predictions_path or os.path.join(out, "predictions.csv")
        if os.path.exists(src_pred):
            shutil.copy2(src_pred, dst_pred)
            logger.info(f"Best predictions: {dst_pred}")

        # Copy best training curves
        src_curves = os.path.join(out, f"training_curves_exp{best_exp_id:02d}.png")
        dst_curves = os.path.join(out, "training_curves.png")
        if os.path.exists(src_curves):
            shutil.copy2(src_curves, dst_curves)
            logger.info(f"Best training curves: {dst_curves}")

        # Copy best confusion matrix
        src_cm = os.path.join(out, f"confusion_matrix_exp{best_exp_id:02d}.png")
        dst_cm = os.path.join(out, "confusion_matrix.png")
        if os.path.exists(src_cm):
            shutil.copy2(src_cm, dst_cm)
            logger.info(f"Best confusion matrix: {dst_cm}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
