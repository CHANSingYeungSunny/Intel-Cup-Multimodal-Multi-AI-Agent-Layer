#!/usr/bin/env python3
"""
End-to-end training and evaluation pipeline for COUGHVID-v3 AST classifier.

Usage:
    python main.py

Outputs (in outputs/):
    best_ast_coughvid_local.pt               — model checkpoint
    predictions.csv                          — Fusion Layer input
    experiment_results_with_accuracy.csv     — metrics + hyperparameters
    confusion_matrix.png                     — confusion matrix plot
    training_curves.png                      — loss & accuracy curves
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Tee output: writes to BOTH stdout AND a log file simultaneously.
# The log file is always line-buffered so you can `type training.log`
# in a cmd window to see live progress.
# ------------------------------------------------------------------
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_main.log")

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

_log_file = open(LOG_PATH, "w", buffering=1)  # line-buffered
sys.stdout = _Tee(sys.stdout, _log_file)      # type: ignore

from audio_layer import (
    set_seeds,
    ensure_output_dir,
    ensure_cache_dir,
    DEVICE,
    CHECKPOINT_PATH,
    PREDICTIONS_CSV_PATH,
    AudioSpectrogramTransformer,
    scan_audio_files,
    create_dataloaders,
    train_model,
    evaluate_model,
    export_predictions,
)
from audio_layer import config as cfg


def main():
    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    ensure_output_dir()
    ensure_cache_dir()
    set_seeds()
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        mem_bytes = torch.cuda.get_device_properties(0).total_memory
        print(f"Memory : {mem_bytes / 1e9:.1f} GB")
        print(f"AMP    : {'enabled' if cfg.USE_AMP else 'disabled'}")
    else:
        print("GPU    : (none — running on CPU)")
    print(f"Input  : {cfg.N_MELS}×{cfg.MAX_FRAMES} Mel spectrograms")
    print(f"Cache  : {cfg.CACHE_DIR}")

    # ------------------------------------------------------------------
    # 1. Scan audio files & cross-reference with metadata
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1: Scanning audio files & loading metadata")
    print("=" * 60)

    file_paths, labels = scan_audio_files()

    # ------------------------------------------------------------------
    # 2. Stratified split & DataLoaders
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Creating DataLoaders")
    print("=" * 60)

    train_loader, val_loader, test_loader, split_data = create_dataloaders(
        file_paths, labels
    )
    X_train, X_val, X_test, y_train, y_val, y_test = split_data

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Building AST model")
    print("=" * 60)

    model = AudioSpectrogramTransformer(
        n_mels=cfg.N_MELS,
        max_frames=cfg.MAX_FRAMES,
        patch_size=cfg.PATCH_SIZE,
        num_classes=cfg.NUM_CLASSES,
        d_model=cfg.D_MODEL,
        n_heads=cfg.N_HEADS,
        n_layers=cfg.N_LAYERS,
        d_ff=cfg.D_FF,
        dropout=cfg.DROPOUT,
        use_checkpoint=cfg.GRADIENT_CHECKPOINTING,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    # ------------------------------------------------------------------
    # 4. Loss & Optimizer (with class-balanced weights)
    # ------------------------------------------------------------------
    class_counts = np.bincount(y_train, minlength=cfg.NUM_CLASSES)
    class_weights = len(y_train) / (cfg.NUM_CLASSES * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    print(f"\nClass counts  : {class_counts}")
    print(f"Class weights : {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Training")
    print("=" * 60)

    model, history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=cfg.EPOCHS, device=DEVICE, checkpoint_path=CHECKPOINT_PATH,
    )

    # ------------------------------------------------------------------
    # 6. Evaluate on test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Test-set Evaluation")
    print("=" * 60)

    metrics, eval_data = evaluate_model(
        model, test_loader, history, device=DEVICE
    )
    y_true, y_pred, cls_embeddings, filenames = eval_data

    # ------------------------------------------------------------------
    # 7. Export predictions.csv for Fusion Layer
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 7: Exporting predictions.csv")
    print("=" * 60)

    export_predictions(
        y_true, y_pred, cls_embeddings, filenames,
        output_path=PREDICTIONS_CSV_PATH,
    )

    # ------------------------------------------------------------------
    # Report cache statistics
    # ------------------------------------------------------------------
    for name, ds in [("Train", train_loader.dataset),
                      ("Val", val_loader.dataset),
                      ("Test", test_loader.dataset)]:
        hits, misses = ds.cache_stats
        total = hits + misses
        if total > 0:
            print(f"  {name} cache: {hits}/{total} hits ({100*hits/total:.0f}%)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"All outputs saved to: {cfg.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
