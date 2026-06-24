# Physiological Layer — BIDMC-PPG Health Classification

Clean Python refactor of Bailey's `IntelCup2026.ipynb` notebook for the
Intel Cup 2026 multimodal fusion pipeline.

## Overview

This project classifies physiological health status from PPG and ECG signals
using an iTransformer model (~575K parameters). It processes the BIDMC-PPG
dataset (53 ICU subjects, 8-minute recordings at 125 Hz) to predict whether
a patient is **healthy**, **semi-healthy (symptomatic)**, or **unhealthy** —
with an alternative **binary** mode merging the two unhealthy classes.

The output format matches the Audio Layer and Vision Layer for downstream
Fusion Layer integration.

## Project Structure

```
physiological_layer/
├── data_loader.py          # BIDMC-PPG loading, sliding window, .pt caching
├── preprocess.py           # Normalization, filtering, HR/RR features, labeling
├── model.py                # iTransformer (1D CNN proj + Transformer encoder)
├── train.py                # Training loop (AMP, grad ckpt, early stop, TensorBoard)
├── utils.py                # Metrics, logging, confusion matrix, plotting, CSV export
├── cache/windows/          # Pre-computed window tensors (~100 MB)
├── outputs/                # Generated artifacts
│   ├── predictions.csv
│   ├── experiment_results_with_accuracy.csv
│   ├── best_physio.pt
│   ├── training_curves.png
│   └── confusion_matrix.png
├── README.md               # This file
└── summary.md              # Refactoring & optimization summary
```

## Installation

```bash
# Required packages
pip install torch numpy pandas scipy scikit-learn wfdb tqdm matplotlib seaborn tensorboard
```

**System requirements:**
- Python 3.8+
- GPU with 4GB+ VRAM recommended (CPU-only also supported, just slower)
- ~100 MB disk for cached windows + ~200 MB for model checkpoints and outputs

## Dataset Setup

The BIDMC-PPG dataset must be extracted in the project root:

```
bidmc-ppg-and-respiration-dataset-1.0.0/
├── RECORDS
├── bidmc01.hea, bidmc01.dat, bidmc01.breath
├── bidmc01n.hea, bidmc01n.dat
├── ... (53 subjects: bidmc01–bidmc53)
└── bidmc_data.mat
```

The first run automatically builds the window cache in `cache/windows/`.

## Quick Start

```bash
# Build window cache (done automatically on first run)
python data_loader.py

# Run the best experiment (binary classification, exp_id=2)
python train.py --exp_id 2

# View outputs
ls outputs/
# predictions.csv  experiment_results_with_accuracy.csv
# best_physio.pt   training_curves.png  confusion_matrix.png
```

## Running Experiments

```bash
# Individual experiments
python train.py --exp_id 1    # Baseline (3-class, bs=64, lr=3e-4)
python train.py --exp_id 2    # Binary (bs=64, lr=3e-4) — expected best
python train.py --exp_id 3    # Batch32 (3-class, bs=32, lr=3e-4)
python train.py --exp_id 4    # Freeze encoder layer 1
python train.py --exp_id 5    # Lower LR (3-class, bs=64, lr=1e-4)

# Run all experiments
python train.py --exp_id all
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp_id` | `1` | Experiment ID: `1`-`5` or `all` |
| `--data_dir` | `bidmc-ppg-and-respiration-dataset-1.0.0` | BIDMC dataset path |
| `--cache_dir` | `cache/windows` | Cached window tensors directory |
| `--output_dir` | `outputs` | Output directory |
| `--log_dir` | `logs/tensorboard` | TensorBoard log directory |
| `--seed` | `42` | Random seed for reproducibility |
| `--num_workers` | `0` | DataLoader worker processes |
| `--early_stop_patience` | `3` | Epochs without improvement before stopping |
| `--grad_checkpoint` | (on) | Gradient checkpointing for memory savings |
| `--no_grad_checkpoint` | | Disable gradient checkpointing |
| `--rebuild_cache` | | Force rebuild of window cache |

## TensorBoard

```bash
tensorboard --logdir logs/tensorboard
# Open http://localhost:6006
```

## Output Files

### `predictions.csv`
| Column | Description |
|--------|-------------|
| `filename` | Window index (e.g. `window_000042`) |
| `prediction` | Predicted class (0=healthy, 1=symptomatic) |
| `label` | Ground-truth class |
| `feature_vector` | 128-dim CLS embedding (JSON array) |

### `experiment_results_with_accuracy.csv`
33 columns matching Audio/Vision Layer format:
- Config: `exp_id`, `config_label`, `gamma`, `epochs`, `d_model`, `n_layers`, `batch_size`, `label_mode`, `best_epoch`
- Performance: `accuracy`, `test_accuracy`, `test_f1_macro`, `test_f1_weighted`, etc.
- Confusion matrix: 9 columns (`cm_healthy_to_healthy`, ...)
- Curves: `train_loss_curve`, `val_loss_curve`, `train_acc_curve`, `val_acc_curve`, `val_f1_curve` (JSON arrays)

### `best_physio.pt`
PyTorch checkpoint containing:
- `model_state_dict` — trained weights
- `config` — model architecture and hyperparameters
- `best_epoch`, `val_macro_f1_best`, `history`

## Model Architecture

**iTransformerClassifier** — applies Transformer attention across variable tokens
(channels) rather than time steps:

```
Input: [B, 1250, 4]           (ECG_z, PPG_z, dECG_z, dPPG_z)
  → Transpose [B, 4, 1250]
  → Linear(1250, 128)        (project each variable)
  → + Variable Embedding
  → TransformerEncoder × 3   (d_model=128, nhead=4, d_ff=256)
  → Global Average Pool
  → LayerNorm → MLP Head
  → [B, num_classes]
```

~575K trainable parameters. Fits batch_size=64 on 4GB GPU with AMP + gradient checkpointing.

## Reproducibility

- Seed fixed at 42 across NumPy, PyTorch, and Python random
- Record-level train/val/test split (70/15/15) is deterministic
- Window cache is label-mode-specific; rebuild with `--rebuild_cache` if switching modes
- `python train.py --exp_id 2` regenerates identical `predictions.csv`

## License

BIDMC-PPG dataset: PhysioNet-derived research data.
Model code: MIT License.
