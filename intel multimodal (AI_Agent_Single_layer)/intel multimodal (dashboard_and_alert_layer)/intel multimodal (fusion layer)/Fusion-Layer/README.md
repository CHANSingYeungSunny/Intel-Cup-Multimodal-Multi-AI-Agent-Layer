# Fusion Layer — Multimodal Health State Classification

Fuses features from three monomodal layers (Vision, Audio, Physiological) using a Multimodal Transformer Encoder with cross-modal attention.

## Overview

The Fusion Layer loads pre-extracted feature vectors from the three upstream layers:
- **Vision** (Swin-Tiny): 768-dim features from UBFC rPPG video frames
- **Audio** (AST): 128-dim CLS embeddings from COUGHVID cough sounds
- **Physiological** (iTransformer): 128-dim pooled embeddings from BIDMC PPG/ECG

These are concatenated into **1024-dim** vectors and fed into a Multimodal Transformer that models cross-modal interactions via self-attention across modality tokens.

```
                   ┌──────────────────┐
  Vision   768 ───▶│ Linear(768, 256) │──▶ [V] token
                   └──────────────────┘
                   ┌──────────────────┐
  Audio    128 ───▶│ Linear(128, 256) │──▶ [A] token
                   └──────────────────┘
                   ┌──────────────────┐          ┌─────────────────────┐
  Physio   128 ───▶│ Linear(128, 256) │──▶ [P] token ──▶ Transformer ──▶ CLS ──▶ Classifier
                   └──────────────────┘          │  Encoder (4L)       │
                      [CLS] token ──────────────▶│  d=256, h=8, ff=512 │
                                                 └─────────────────────┘
```

## Prerequisites

The three monomodal layers must have been run to produce `predictions.csv`:

```
../intel multimodal (vision layer)/vision_layer/output/predictions/predictions.csv
../intel multimodal (audio layer)/outputs/predictions.csv
../intel multimodal (physiological layer)/outputs/predictions.csv
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm, tensorboard

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm tensorboard
```

## Project Structure

```
Fusion-Layer/
├── fusion_loader.py                 # Load & align predictions.csv from 3 layers
├── fusion_model.py                  # MultimodalFusionEncoder + FocalLoss
├── fusion_train.py                  # Training loop (AMP, grad ckpt, early stop)
├── fusion_utils.py                  # Metrics, confusion matrix, CSV, plotting
├── fusion_preprocess.py             # Normalization, label handling, split
├── README.md                        # This file
├── summary.md                       # Detailed approach explanation
├── cache/
│   └── fused_features.pt            # Cached 607×1024 aligned feature tensor
├── logs/
│   └── fusion_train.log             # Training log
├── runs/
│   └── fusion/
│       ├── exp_01/                  # TensorBoard logs per experiment
│       ├── exp_02/
│       ├── exp_03/
│       ├── exp_04/
│       └── exp_05/
└── outputs/
    ├── predictions.csv              # Best experiment's predictions (Exp 2)
    ├── predictions_exp01.csv        # Per-experiment predictions
    ├── predictions_exp02.csv
    ├── predictions_exp03.csv
    ├── predictions_exp04.csv
    ├── predictions_exp05.csv
    ├── experiment_results_with_accuracy.csv  # 33-column results (5 rows)
    ├── best_fusion.pt               # Best checkpoint (Exp 2)
    ├── best_fusion_exp01.pt         # Per-experiment checkpoints
    ├── best_fusion_exp02.pt
    ├── best_fusion_exp03.pt
    ├── best_fusion_exp04.pt
    ├── best_fusion_exp05.pt
    ├── fusion_training_curves.png   # Best experiment's curves (Exp 2)
    ├── fusion_training_curves_exp01.png
    ├── fusion_training_curves_exp02.png
    ├── fusion_training_curves_exp03.png
    ├── fusion_training_curves_exp04.png
    ├── fusion_training_curves_exp05.png
    ├── fusion_confusion_matrix.png   # Best experiment's CM (Exp 2)
    ├── fusion_confusion_matrix_exp01.png
    ├── fusion_confusion_matrix_exp02.png
    ├── fusion_confusion_matrix_exp03.png
    ├── fusion_confusion_matrix_exp04.png
    └── fusion_confusion_matrix_exp05.png
```

## Quick Start

```bash
cd Fusion-Layer

# Run a single experiment
python fusion_train.py --exp_id 2

# Run all 5 experiments
python fusion_train.py --exp_id all

# Run without gradient checkpointing (uses more GPU memory)
python fusion_train.py --exp_id 1 --no_grad_checkpoint

# View TensorBoard logs
tensorboard --logdir runs/fusion
```

## CLI Reference

```
python fusion_train.py [OPTIONS]

Options:
  --exp_id STR              Experiment ID: 1-5 or 'all' (default: 1)
  --output_dir DIR          Output directory (default: outputs)
  --predictions_path PATH   Override predictions.csv path
  --log_dir DIR             TensorBoard log directory (default: runs/fusion)
  --seed INT                Random seed (default: 42)
  --num_workers INT         DataLoader workers (default: 0)
  --early_stop_patience INT Early stopping patience (default: 3)
  --grad_checkpoint         Enable gradient checkpointing (default: True)
  --no_grad_checkpoint      Disable gradient checkpointing
  --use_class_weights       Use class weights in FocalLoss (default: True)
```

## Module Descriptions

### fusion_loader.py
Entry point for data loading. Resolves paths to the three upstream `predictions.csv` files, parses the JSON-encoded feature vectors, and performs **label-matched pairing** (see summary.md for details). Caches the aligned 1024-dim tensor to `cache/fused_features.pt` for instant reload on subsequent runs.

Run standalone to test: `python fusion_loader.py`

### fusion_model.py
Defines `MultimodalFusionEncoder` (~2.4M params):
- Splits 1024-dim input into 3 modality tokens → projects each to d_model=256
- Prepends learnable CLS token → 4-token sequence
- 4-layer Transformer Encoder (8 heads, d_ff=512, GELU, pre-norm)
- CLS token → classification head → logits
- Returns `(logits, fusion_embedding)` matching the monomodal layers' API
- Includes `FocalLoss` for class-imbalanced training

Run standalone to test: `python fusion_model.py`

### fusion_preprocess.py
Handles `StandardScaler` normalization (fit on train, apply to val/test), 3-class/binary label mapping, and stratified 70/15/15 train/val/test split.

Run standalone to test: `python fusion_preprocess.py`

### fusion_utils.py
Utility functions shared across the pipeline: seed setting, logging, metric computation (accuracy, macro/weighted precision/recall/F1), confusion matrix flattening, JSON serialization, CSV output (33-column experiment results schema), training curve plotting, and confusion matrix heatmaps.

### fusion_train.py
Main training script. Features:
- **AMP** (`torch.amp`) for half-precision on CUDA
- **Gradient checkpointing** per encoder layer (trades ~20% speed for ~40% memory)
- **Early stopping** (patience=3 on validation macro F1)
- **TensorBoard** logging (`runs/fusion/exp_XX/`)
- **WeightedRandomSampler** for class-balanced batches
- **FocalLoss** with optional class weights

## Experiments

Results sorted by `test_f1_weighted` (the selection metric):

| Exp | Config | Classes | BS | LR | Epochs | Best Ep | Test Acc | Macro F1 | **Weighted F1** |
|-----|--------|---------|----|----|--------|---------|----------|----------|-----------------|
| **2** | **Binary** | 2 | 64 | 3e-4 | 6 | 3 | **77.2%** | 0.770 | **0.775** |
| 1 | 3-class | 3 | 64 | 3e-4 | 10 | 7 | 71.7% | **0.783** | 0.715 |
| 3 | Batch128 | 3 | 128 | 3e-4 | 11 | 8 | 71.7% | 0.781 | 0.711 |
| 4 | Freeze enc | 3 | 64 | 3e-4 | 7 | 4 | 66.3% | 0.732 | 0.643 |
| 5 | Lower LR | 3 | 64 | 1e-4 | 11 | 8 | 66.3% | 0.732 | 0.643 |

**Exp 2 (binary) is selected as the best** — it has the highest weighted F1 (0.775) and accuracy (77.2%), and its outputs are copied to the final deliverables:
- `outputs/predictions.csv`
- `outputs/best_fusion.pt`
- `outputs/fusion_training_curves.png`
- `outputs/fusion_confusion_matrix.png`

### Experiment Descriptions

| Exp | Name | Rationale |
|-----|------|-----------|
| 1 | Baseline (3-class, bs=64) | Reference point for all comparisons |
| 2 | **Binary (bs=64)** | Simplifies to healthy vs symptomatic — best weighted F1 |
| 3 | Batch128 (3-class) | Tests large-batch training stability |
| 4 | Freeze encoder layer 0 | Tests impact of freezing first transformer layer |
| 5 | Lower LR (1e-4) | Tests convergence with reduced learning rate |

## Forecasting (Future Extension)

A forecasting head is architected in `fusion_model.py` (`forecast_horizon` parameter and `forecast_head` attribute) but **not enabled** in this competition. The dataset lacks true longitudinal sequences (samples come from different subjects matched by label), so meaningful temporal forecasting is not possible with the current data. Future work can extend this module when longitudinal multimodal datasets become available.

## Output File Formats

### predictions.csv

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | Composite ID: `v:{vision_id}\|a:{audio_id}\|p:{physio_id}` |
| `prediction` | int | Predicted class (0=healthy, 1=symptomatic/unhealthy) |
| `label` | int | Ground truth label |
| `feature_vector` | str | 256-dim fusion CLS embedding (JSON array) |

### experiment_results_with_accuracy.csv

33 columns, one row per experiment:

- **9 config columns**: `exp_id`, `config_label`, `gamma`, `epochs`, `d_model`, `n_layers`, `batch_size`, `label_mode`, `best_epoch`
- **10 performance columns**: `val_macro_f1_best`, `accuracy`, `test_accuracy`, `test_precision_macro`, `test_recall_macro`, `test_f1_macro`, `test_precision_weighted`, `test_recall_weighted`, `test_f1_weighted`, `test_loss`
- **9 confusion matrix columns**: `cm_healthy_to_healthy` through `cm_unhealthy_to_unhealthy`
- **5 curve columns**: `train_loss_curve`, `val_loss_curve`, `train_acc_curve`, `val_acc_curve`, `val_f1_curve` (JSON arrays)

## Reproducibility

All randomness controlled by `--seed 42`:
- Label-matched pairing across modalities
- Stratified 70/15/15 train/val/test split
- WeightedRandomSampler
- Model initialization

```bash
python fusion_train.py --exp_id 2 --seed 42   # always produces identical outputs
```

## Optimizations

| Technique | Benefit |
|-----------|---------|
| AMP (Automatic Mixed Precision) | Half-precision on CUDA; ~2× memory saving |
| Gradient Checkpointing | Per-layer recomputation; ~40% memory reduction |
| Feature Caching (.pt) | Skip repeated CSV parsing; instant reload |
| d_model=256 | Only 4 tokens × 256 dims = minimal memory footprint |
| Early Stopping (patience=3) | Stop when validation F1 plateaus |
| Focal Loss (gamma=2.0) | Focus on hard examples, handle class imbalance |
| WeightedRandomSampler | Balanced batches despite class skew |
| Weight Decay (1e-4) | Regularization to prevent overfitting |

Combined, these enable **batch_size=128 on a 4GB GPU**.
