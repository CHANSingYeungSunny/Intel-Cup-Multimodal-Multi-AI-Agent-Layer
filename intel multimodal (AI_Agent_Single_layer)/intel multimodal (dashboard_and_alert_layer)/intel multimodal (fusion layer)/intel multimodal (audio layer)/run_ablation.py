#!/usr/bin/env python3
"""
5-Experiment Audio Layer Ablation Runner.

Runs all experiments sequentially, each in its own output subdirectory.
Saves a summary CSV comparing key metrics across all experiments.

Usage:
    python run_ablation.py

Monitor:
    powershell -c "Get-Content training.log -Wait"
"""

import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Tee output to training.log for live monitoring
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_ablation.log")

class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

_log_file = open(LOG_PATH, "w", buffering=1)
sys.stdout = _Tee(sys.stdout, _log_file)

from audio_layer import (
    set_seeds, ensure_output_dir, ensure_cache_dir,
    set_output_dir, apply_overrides,
    DEVICE, AudioSpectrogramTransformer,
    scan_audio_files, create_dataloaders,
    train_model, evaluate_model, export_predictions,
)
from audio_layer import config as cfg


# =====================================================================
# Experiment Definitions
# =====================================================================
EXPERIMENTS = [
    {
        "name": "1_baseline",
        "num_classes": 3,
        "batch_size": 32,
        "lr": 3e-4,
        "freeze_layer": None,
        "status_to_label": {"healthy": 0, "symptomatic": 1, "COVID-19": 2},
        "id2label": {0: "healthy", 1: "symptomatic", 2: "covid_19"},
    },
    {
        "name": "2_binary",
        "num_classes": 2,
        "batch_size": 32,
        "lr": 3e-4,
        "freeze_layer": None,
        "status_to_label": {"healthy": 0, "symptomatic": 1, "COVID-19": 1},
        "id2label": {0: "healthy", 1: "unhealthy"},
    },
    {
        "name": "3_batch16",
        "num_classes": 3,
        "batch_size": 16,
        "lr": 3e-4,
        "freeze_layer": None,
        "status_to_label": {"healthy": 0, "symptomatic": 1, "COVID-19": 2},
        "id2label": {0: "healthy", 1: "symptomatic", 2: "covid_19"},
    },
    {
        "name": "4_freeze_enc",
        "num_classes": 3,
        "batch_size": 32,
        "lr": 3e-4,
        "freeze_layer": 0,  # freeze first encoder layer
        "status_to_label": {"healthy": 0, "symptomatic": 1, "COVID-19": 2},
        "id2label": {0: "healthy", 1: "symptomatic", 2: "covid_19"},
    },
    {
        "name": "5_lr1e4",
        "num_classes": 3,
        "batch_size": 32,
        "lr": 1e-4,
        "freeze_layer": None,
        "status_to_label": {"healthy": 0, "symptomatic": 1, "COVID-19": 2},
        "id2label": {0: "healthy", 1: "symptomatic", 2: "covid_19"},
    },
]


# =====================================================================
# Run a single experiment
# =====================================================================
def run_experiment(exp: dict, cache_dir: str):
    """Run one experiment end-to-end. Returns metrics dict for summary."""
    exp_name = exp["name"]
    output_dir = os.path.join(cfg.PROJECT_ROOT, "outputs", exp_name)
    set_output_dir(output_dir)
    ensure_output_dir()

    print(f"\n{'#' * 60}")
    print(f"# Experiment: {exp_name}")
    print(f"# Output:    {output_dir}")
    print(f"{'#' * 60}")

    # --- Apply overrides ---
    apply_overrides(
        NUM_CLASSES=exp["num_classes"],
        BATCH_SIZE=exp["batch_size"],
        LR=exp["lr"],
        STATUS_TO_LABEL=exp["status_to_label"],
        ID2LABEL=exp["id2label"],
    )
    set_seeds()

    # --- 1. Scan & load data ---
    print("\n[1/5] Scanning audio files...")
    file_paths, labels = scan_audio_files(status_to_label=exp["status_to_label"])

    # --- 2. Create DataLoaders ---
    print("\n[2/5] Creating DataLoaders...")
    train_loader, val_loader, test_loader, split_data = create_dataloaders(
        file_paths, labels, cache_dir=cache_dir,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = split_data

    # --- 3. Build model ---
    print("\n[3/5] Building model...")
    model = AudioSpectrogramTransformer(
        n_mels=cfg.N_MELS,
        max_frames=cfg.MAX_FRAMES,
        patch_size=cfg.PATCH_SIZE,
        num_classes=exp["num_classes"],
        d_model=cfg.D_MODEL,
        n_heads=cfg.N_HEADS,
        n_layers=cfg.N_LAYERS,
        d_ff=cfg.D_FF,
        dropout=cfg.DROPOUT,
        use_checkpoint=cfg.GRADIENT_CHECKPOINTING,
    ).to(DEVICE)

    if exp["freeze_layer"] is not None:
        model.freeze_encoder_layer(exp["freeze_layer"])

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_p:,}  |  Trainable: {trainable_p:,}")

    # --- 4. Loss & Optimizer ---
    class_counts = np.bincount(y_train, minlength=exp["num_classes"])
    class_weights = len(y_train) / (exp["num_classes"] * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    print(f"  Class counts : {class_counts}")
    print(f"  Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )

    # --- 5. Train ---
    print("\n[4/5] Training...")
    model, history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=cfg.EPOCHS, patience=cfg.EARLY_STOP_PATIENCE,
        device=DEVICE, checkpoint_path=cfg.CHECKPOINT_PATH,
    )

    # --- 6. Evaluate ---
    print("\n[5/5] Evaluating & exporting...")
    metrics, eval_data = evaluate_model(
        model, test_loader, history, device=DEVICE, output_dir=cfg.OUTPUT_DIR,
    )
    y_true, y_pred, cls_embeddings, filenames = eval_data

    export_predictions(
        y_true, y_pred, cls_embeddings, filenames,
        output_path=cfg.PREDICTIONS_CSV_PATH,
    )

    # --- Return full vision-compatible metrics ---
    # Pick only vision-layer columns for the combined CSV
    vision_keys = [
        "exp_id", "config_label", "gamma", "epochs", "d_model", "n_layers",
        "batch_size", "label_mode", "best_epoch", "val_macro_f1_best",
        "accuracy", "test_accuracy",
        "test_precision_macro", "test_recall_macro", "test_f1_macro",
        "test_precision_weighted", "test_recall_weighted", "test_f1_weighted",
        "test_loss",
        # CM cells
        "cm_healthy_to_healthy", "cm_healthy_to_semi", "cm_healthy_to_unhealthy",
        "cm_semi_to_healthy", "cm_semi_to_semi", "cm_semi_to_unhealthy",
        "cm_unhealthy_to_healthy", "cm_unhealthy_to_semi", "cm_unhealthy_to_unhealthy",
        # Curves
        "train_loss_curve", "val_loss_curve",
        "train_acc_curve", "val_acc_curve", "val_f1_curve",
    ]
    result = {k: metrics.get(k) for k in vision_keys}
    result["epochs"] = len(history["train_loss"])  # actual epochs run
    # Add extras for debugging
    result["experiment"] = exp_name
    result["num_classes"] = exp["num_classes"]
    result["freeze_layer"] = exp["freeze_layer"]
    return result


# =====================================================================
# Main
# =====================================================================
def main():
    t_total = time.time()

    print("=" * 60)
    print("Audio Layer Ablation — 5 Experiments")
    print(f"Device: {cfg.DEVICE}")
    if cfg.DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input: {cfg.N_MELS}×{cfg.MAX_FRAMES}  |  "
          f"AMP: {cfg.USE_AMP}  |  "
          f"Checkpointing: {cfg.GRADIENT_CHECKPOINTING}")
    print(f"Max epochs: {cfg.EPOCHS}  |  Patience: {cfg.EARLY_STOP_PATIENCE}")
    print("=" * 60)

    ensure_cache_dir()
    cache_dir = cfg.CACHE_DIR

    # Count pre-cached files
    if os.path.isdir(cache_dir):
        n_cached = sum(1 for f in os.listdir(cache_dir) if f.endswith("_mel.pt"))
        print(f"\nCache status: {n_cached} files pre-cached")
        if n_cached > 10000:
            print("Cache is well-populated — all epochs will be fast.")

    all_results = []

    for i, exp in enumerate(EXPERIMENTS):
        t0 = time.time()
        print(f"\n{'=' * 60}")
        print(f"Experiment {i+1}/{len(EXPERIMENTS)}: {exp['name']}")
        print(f"{'=' * 60}")

        try:
            result = run_experiment(exp, cache_dir)
            result["exp_id"] = i + 1  # set experiment number
            all_results.append(result)
            elapsed = time.time() - t0
            print(f"\nExperiment {exp['name']} complete in {elapsed/60:.1f} min")
            print(f"  Best val F1: {result.get('val_macro_f1_best', 'N/A'):.4f}"
                  if isinstance(result.get('val_macro_f1_best'), float) else "")
            print(f"  Test accuracy: {result.get('accuracy', 'N/A')}")
        except Exception as e:
            print(f"\n[ERROR] Experiment {exp['name']} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"experiment": exp["name"], "error": str(e)})

    # --- Write combined experiment_results CSV (vision-layer compatible) ---
    print(f"\n{'#' * 60}")
    print(f"# All experiments complete!")
    print(f"# Total time: {(time.time() - t_total)/60:.1f} min")
    print(f"{'#' * 60}")

    # Vision-layer column order
    vision_cols = [
        "exp_id", "config_label", "gamma", "epochs", "d_model", "n_layers",
        "batch_size", "label_mode", "best_epoch", "val_macro_f1_best",
        "accuracy", "test_accuracy",
        "test_precision_macro", "test_recall_macro", "test_f1_macro",
        "test_precision_weighted", "test_recall_weighted", "test_f1_weighted",
        "test_loss",
        "cm_healthy_to_healthy", "cm_healthy_to_semi", "cm_healthy_to_unhealthy",
        "cm_semi_to_healthy", "cm_semi_to_semi", "cm_semi_to_unhealthy",
        "cm_unhealthy_to_healthy", "cm_unhealthy_to_semi", "cm_unhealthy_to_unhealthy",
        "train_loss_curve", "val_loss_curve",
        "train_acc_curve", "val_acc_curve", "val_f1_curve",
    ]

    # Build combined DataFrame (skip failed experiments)
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        # Ensure all vision columns exist
        for r in valid_results:
            for col in vision_cols:
                r.setdefault(col, None)
        combined_df = pd.DataFrame(valid_results)[vision_cols]
    else:
        combined_df = pd.DataFrame(columns=vision_cols)

    results_path = os.path.join(cfg.PROJECT_ROOT, "outputs",
                                 "experiment_results_with_accuracy.csv")
    combined_df.to_csv(results_path, index=False)
    print(f"\nCombined results saved to: {results_path}")
    print(f"  Rows: {len(combined_df)}  |  Columns: {len(combined_df.columns)}")
    print(combined_df[["exp_id", "config_label", "test_f1_macro",
                        "val_macro_f1_best"]].to_string(index=False))

    # Also save ablation_summary for quick reference
    summary_path = os.path.join(cfg.PROJECT_ROOT, "outputs", "ablation_summary.csv")
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # --- Pick best experiment and copy predictions.csv to top-level ---
    # Uses val_macro_f1_best (same metric as vision layer run_experiments.py)
    if valid_results:
        best = max(valid_results, key=lambda r: r.get("val_macro_f1_best", -1.0))
        best_name = best.get("experiment", "unknown")
        best_f1 = best.get("val_macro_f1_best", 0.0)
        print(f"\nBest experiment: {best_name} (val_macro_f1={best_f1:.4f})")

        # Copy predictions.csv
        import shutil
        src_pred = os.path.join(cfg.PROJECT_ROOT, "outputs", best_name,
                                "predictions.csv")
        dst_pred = os.path.join(cfg.PROJECT_ROOT, "outputs", "predictions.csv")
        if os.path.exists(src_pred):
            shutil.copy2(src_pred, dst_pred)
            print(f"  predictions.csv → {dst_pred}")

        # Copy best checkpoint
        src_ckpt = os.path.join(cfg.PROJECT_ROOT, "outputs", best_name,
                                "best_ast_coughvid_local.pt")
        dst_ckpt = os.path.join(cfg.PROJECT_ROOT, "outputs",
                                "best_ast_coughvid_local.pt")
        if os.path.exists(src_ckpt):
            shutil.copy2(src_ckpt, dst_ckpt)
            print(f"  best checkpoint → {dst_ckpt}")


if __name__ == "__main__":
    main()
