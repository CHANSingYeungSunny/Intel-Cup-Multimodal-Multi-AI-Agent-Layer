# =====================================================================
# Training Loop
#
# Provides run_epoch() and train_model() — the core training pipeline.
# Includes early stopping and clean line-by-line progress for easy
# monitoring via `type` or `tail -f` on the output file.
# =====================================================================

import time
import numpy as np
import torch
import torch.nn as nn

from . import config as cfg


# ---------------------------------------------------------------------
# Single epoch (train or eval) — clean logging, no \r overwrites
# ---------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None, device=None):
    """
    Run one epoch of training or evaluation.

    Returns:
        avg_loss, accuracy, y_true_list, y_pred_list
    """
    if device is None:
        device = cfg.DEVICE

    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    use_amp = train_mode and getattr(cfg, "USE_AMP", False)
    scaler = torch.amp.GradScaler(device) if use_amp else None

    total_loss = 0.0
    y_true, y_pred = [], []
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)  # print ~10 progress lines per epoch

    desc = "Train" if train_mode else "Eval "
    t0 = time.time()

    for i, (xb, yb, _) in enumerate(loader):  # filenames unused during training
        try:
            xb = xb.to(device)
            yb = yb.to(device)

            if train_mode:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train_mode):
                if use_amp:
                    with torch.amp.autocast(device):
                        logits, _ = model(xb)
                        loss = criterion(logits, yb)
                else:
                    logits, _ = model(xb)
                    loss = criterion(logits, yb)

                if train_mode:
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                        optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)

            y_true.extend(yb.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
        except Exception as e:
            print(f"  [{desc}] ERROR at batch {i}: {type(e).__name__}: {e}",
                  flush=True)
            print(f"  [{desc}] Skipping batch {i} and continuing...", flush=True)
            continue

        # Print a progress line every ~10% of the epoch (no \r, visible in logs)
        if (i + 1) % log_every == 0 or i == n_batches - 1:
            elapsed = time.time() - t0
            pct = 100.0 * (i + 1) / n_batches
            current_loss = total_loss / ((i + 1) * xb.size(0))
            current_acc = (np.array(y_true) == np.array(y_pred)).mean()
            eta = (elapsed / (i + 1)) * (n_batches - i - 1)
            print(f"  [{desc}] {i+1:4d}/{n_batches} ({pct:3.0f}%) | "
                  f"loss={current_loss:.4f} | acc={current_acc:.4f} | "
                  f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s", flush=True)

    avg_loss = total_loss / len(loader.dataset)
    acc = (np.array(y_true) == np.array(y_pred)).mean()
    return avg_loss, acc, y_true, y_pred


# ---------------------------------------------------------------------
# Full training routine with early stopping
# ---------------------------------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs: int = None,
    patience: int = None,
    device: str = None,
    checkpoint_path: str = None,
):
    """
    Train the AST model with early stopping.

    Stops early if validation accuracy does not improve for `patience`
    consecutive epochs. Saves the best checkpoint to disk.

    Returns:
        model        — loaded with best weights
        history      — dict of train/val loss and accuracy per epoch
        best_val_acc — best validation accuracy achieved
    """
    if epochs is None:
        epochs = cfg.EPOCHS
    if patience is None:
        patience = cfg.EARLY_STOP_PATIENCE
    if device is None:
        device = cfg.DEVICE
    if checkpoint_path is None:
        checkpoint_path = cfg.CHECKPOINT_PATH

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(f"\n{'='*60}")
    print(f"AST Training — max {epochs} epochs on {device}")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Train samples: {len(train_loader.dataset)} | "
          f"Val samples: {len(val_loader.dataset)}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t_start = time.time()

        # --- Training phase ---
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer=optimizer, device=device
        )
        # --- Validation phase ---
        val_loss, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - t_start
        improved = "*" if val_acc > best_val_acc else ""

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.0f}s {improved}", flush=True
        )

        # --- Checkpoint & early stopping logic ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"  >> New best model! (val_acc={best_val_acc:.4f})", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"  -- No improvement for {epochs_without_improvement}/{patience} epochs", flush=True)

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs "
                  f"(best was epoch {best_epoch} with val_acc={best_val_acc:.4f})")
            break

    # --- Save best checkpoint ---
    actual_epochs = len(history["train_loss"])
    torch.save(
        {
            "model_state_dict": best_state,
            "history": history,
            "id2label": cfg.ID2LABEL,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "early_stopped": epochs_without_improvement >= patience,
            "hyperparameters": {
                "n_mels": cfg.N_MELS,
                "max_frames": cfg.MAX_FRAMES,
                "n_fft": cfg.N_FFT,
                "hop_length": cfg.HOP_LENGTH,
                "patch_size": cfg.PATCH_SIZE,
                "d_model": cfg.D_MODEL,
                "n_heads": cfg.N_HEADS,
                "n_layers": cfg.N_LAYERS,
                "d_ff": cfg.D_FF,
                "dropout": cfg.DROPOUT,
                "batch_size": cfg.BATCH_SIZE,
                "epochs": actual_epochs,
                "max_epochs": cfg.EPOCHS,
                "lr": cfg.LR,
                "weight_decay": cfg.WEIGHT_DECAY,
                "early_stop_patience": patience,
                "seed": cfg.SEED,
            },
        },
        checkpoint_path,
    )
    print(f"\nBest checkpoint saved to: {checkpoint_path}")
    print(f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}/{actual_epochs}")

    # Load best weights back into model
    model.load_state_dict(best_state)
    return model, history, best_val_acc
