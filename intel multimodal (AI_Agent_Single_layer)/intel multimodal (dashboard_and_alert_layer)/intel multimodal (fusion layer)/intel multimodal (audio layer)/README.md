# 🎙️ Intel Cup 2026 — Multimodal Audio Layer

**Audio Spectrogram Transformer (AST) for cough-based COVID-19 classification using the COUGHVID-v3 dataset.**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Instructions](#training-instructions)
5. [Fusion Layer Integration](#fusion-layer-integration)
6. [Optimization Notes](#optimization-notes)
7. [Usage Examples](#usage-examples)
8. [Contribution Notes](#contribution-notes)
9. [Fusion Layer Example](#fusion-layer-example)

---

## Overview

This repository implements the **Audio Layer** of the Intel Cup 2026 Multimodal pipeline. It classifies cough audio recordings into three categories:

| Label | Meaning |
|-------|---------|
| `0` | healthy |
| `1` | symptomatic |
| `2` | covid_19 |

### Architecture

The model is a lightweight **Audio Spectrogram Transformer (AST)** — a Vision Transformer (ViT) variant adapted for 2D Mel-spectrogram patches:

```
Raw Audio (.webm/.wav/.ogg)
    → Resample to 16 kHz
    → 128-bin Mel Spectrogram (192 frames)
    → 16×16 non-overlapping patches
    → Transformer Encoder (pre-norm, 3 layers, 4 heads, d=128)
    → CLS token → Classification Head
    → (logits, cls_embedding)
```

The **CLS token embedding** (128-dim) is exported alongside predictions as a JSON feature vector for downstream Fusion Layer use.

### Project Structure

```
├── main.py                  # End-to-end training & evaluation pipeline
├── run_ablation.py          # 5-experiment ablation runner
├── audio_layer/             # Core package
│   ├── __init__.py          # Public API exports
│   ├── config.py            # All hyperparameters, paths, seeds
│   ├── dataset.py           # Data loading, scanning, caching
│   ├── model.py             # AST architecture definition
│   ├── train.py             # Training loop + early stopping
│   ├── evaluate.py          # Metrics, confusion matrix, curves
│   └── export_predictions.py # predictions.csv generation
├── datasets/
│   └── public_dataset_v3/
│       └── coughvid_20211012/  # ← Place COUGHVID data here
│           ├── metadata_compiled.csv
│           ├── *.webm / *.wav / *.ogg
│           └── mel_cache/      # Auto-generated spectrogram cache
├── outputs/                 # All training outputs (auto-created)
│   ├── predictions.csv      # BEST experiment → Fusion Layer input
│   ├── best_ast_coughvid_local.pt   # BEST experiment checkpoint
│   ├── experiment_results_with_accuracy.csv  # Combined 5-experiment results
│   ├── ablation_summary.csv # Quick reference summary
│   ├── 1_baseline/          # Per-experiment: .pt, .csv, .png
│   ├── 2_binary/            #   (binary = best, F1=0.428)
│   ├── 3_batch16/
│   ├── 4_freeze_enc/
│   └── 5_lr1e4/
└── IntelCup2026_audio.ipynb # Original exploratory notebook
```

---

## Environment Setup

### Requirements

- **Python** >= 3.9
- **GPU** — NVIDIA GTX 1650 (4 GB VRAM) or higher recommended. The model uses ~2 GB VRAM with AMP enabled. CPU-only training is supported but slow (~3-5 min/epoch vs ~15 s/epoch on GPU).

### Install Dependencies

```bash
# Core ML stack
pip install torch>=2.0 torchaudio numpy pandas scikit-learn

# Visualization & utilities
pip install matplotlib tqdm

# Audio decoding (.webm/.ogg support — REQUIRED)
pip install av

# Optional: Jupyter for notebook exploration
pip install jupyter
```

> **💡 Why PyAV?** `torchaudio` 2.6+ requires `ffmpeg` on PATH for `.webm` containers. PyAV (`av`) bundles its own ffmpeg and is the primary audio decoder in this pipeline. Without it, `.webm` files will fail to load.

### Verify GPU Availability

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only')"
```

---

## Dataset Preparation

### Download COUGHVID-v3

1. Download the COUGHVID-v3 public dataset from:  
   [https://zenodo.org/record/7026208](https://zenodo.org/record/7026208)

2. Place the contents into the following directory structure:

```
datasets/public_dataset_v3/coughvid_20211012/
├── metadata_compiled.csv    # Master metadata (REQUIRED)
├── <uuid>.webm              # Audio files (.webm, .wav, .ogg)
├── <uuid>.wav
├── <uuid>.ogg
├── <uuid>.json              # Per-file metadata (not used by pipeline)
└── mel_cache/               # Auto-created on first run
```

3. **Verify:** The directory should contain `metadata_compiled.csv` plus ~29,000+ audio files. The pipeline scans for `.webm`, `.wav`, and `.ogg` extensions automatically.

### How Data Loading Works

1. **Scan** — The pipeline discovers all audio files on disk and builds a UUID→extension map.
2. **Cross-reference** — `metadata_compiled.csv` is loaded; rows are filtered to those with a valid `status` label (`healthy` / `symptomatic` / `COVID-19`) AND a matching audio file on disk.
3. **Split** — Stratified 70/15/15 split (train/val/test), seed-controlled for reproducibility.
4. **Cache** — On first epoch, Mel spectrograms are computed and saved to `mel_cache/`. Subsequent epochs load from cache (~100× faster).

---

## Training Instructions

### Quick Start: Single Run

Train the model with default settings (3-class, batch=32, lr=3e-4, max 10 epochs, early stopping patience=3):

```bash
python main.py
```

**Outputs** (all saved to `outputs/`):

| File | Description |
|------|-------------|
| `best_ast_coughvid_local.pt` | Best model checkpoint (state dict + hyperparameters) |
| `predictions.csv` | Sample-level predictions with JSON feature vectors |
| `experiment_results_with_accuracy.csv` | Per-run metrics in vision-compatible format |
| `confusion_matrix.png` | Confusion matrix visualization |
| `training_curves.png` | Loss & accuracy curves |
| `training_main.log` | Full console log (live-monitor with `tail -f`) |

### Ablation Study: 5 Experiments

Run all experiments sequentially:

```bash
python run_ablation.py
```

**Experiments run:**

| # | Name | Classes | Batch | LR | Special |
|---|------|---------|-------|-----|---------|
| 1 | `1_baseline` | 3 | 32 | 3e-4 | — |
| 2 | `2_binary` | 2 | 32 | 3e-4 | healthy vs unhealthy |
| 3 | `3_batch16` | 3 | 16 | 3e-4 | Smaller batch |
| 4 | `4_freeze_enc` | 3 | 32 | 3e-4 | Freeze 1st encoder layer |
| 5 | `5_lr1e4` | 3 | 32 | 1e-4 | Lower learning rate |

**Ablation results (GTX 1650, ~2.5 hours total):**

| exp_id | Name | Classes | Test F1 | Accuracy | Epochs | Notes |
|---|---|---|---|---|---|---|
| 1 | baseline | 3 | 0.286 | 74.9% | 3 | Majority-class dominant |
| 2 | **binary** | **2** | **0.428** | 74.9% | 3 | **Best — selected for Fusion Layer** |
| 3 | batch16 | 3 | 0.286 | 74.9% | 4 | Batch size insensitive |
| 4 | freeze_enc | 3 | 0.286 | 74.9% | 3 | Freezing layer 0 has no effect |
| 5 | lr1e4 | 3 | 0.288 | 72.6% | 3 | Lower LR, slight accuracy drop |

The best experiment (binary) is automatically copied to `outputs/predictions.csv` for Fusion Layer integration.

### Monitor Training Progress

In a separate terminal:

```bash
# Linux/macOS
tail -f training_ablation.log

# Windows PowerShell
Get-Content training_ablation.log -Wait
```

### Hyperparameter Reference

All configurable in `audio_layer/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_SR` | 16000 | Audio resample rate (Hz) |
| `N_MELS` | 128 | Mel filterbank bins |
| `MAX_FRAMES` | 192 | Fixed temporal dimension (~1.9 s) |
| `PATCH_SIZE` | (16, 16) | 2D patch dimensions |
| `D_MODEL` | 128 | Transformer embedding dim |
| `N_HEADS` | 4 | Multi-head attention heads |
| `N_LAYERS` | 3 | Transformer encoder layers |
| `D_FF` | 256 | Feed-forward hidden dim |
| `DROPOUT` | 0.1 | Dropout rate |
| `BATCH_SIZE` | 32 | Batch size |
| `EPOCHS` | 10 | Max epochs |
| `LR` | 3e-4 | Learning rate |
| `EARLY_STOP_PATIENCE` | 3 | Epochs without improvement |
| `USE_AMP` | True | Automatic Mixed Precision |
| `GRADIENT_CHECKPOINTING` | True | Memory-efficient training |

---

## Fusion Layer Integration

### 🔗 The `predictions.csv` Contract

For the Fusion Layer to work, **all three layers** (Vision, Audio, Physiological) must output a `predictions.csv` with exactly these four columns:

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | Sample identifier (e.g., `00014dcc-0f06-4c27-8c7b-737b18a2cf4c.webm`) |
| `prediction` | int | Predicted class label |
| `label` | int | Ground truth class label |
| `feature_vector` | str | JSON-serialized float array (e.g., `"[0.123, -0.456, ...]"`) |

### Audio Layer `predictions.csv` Example

```csv
filename,prediction,label,feature_vector
00014dcc-0f06-4c27-8c7b-737b18a2cf4c.webm,0,0,"[0.123, -0.456, 0.789, ...]"
00039425-7f3a-42aa-ac13-834aaa2b6b92.webm,1,1,"[-0.234, 0.567, -0.890, ...]"
```

- `feature_vector` = 128-dimensional CLS token embedding from the AST
- Stored as a **JSON array string** for CSV compatibility
- The Fusion Layer parses these JSON strings back into numpy arrays

### How the Audio Layer Generates This

In [`export_predictions.py`](audio_layer/export_predictions.py), each CLS embedding is serialized:

```python
feature_strings = [
    json.dumps(cls_embeddings[i].tolist())
    for i in range(len(cls_embeddings))
]
```

This format matches the Vision Layer convention — both layers use JSON arrays for `feature_vector`.

---

## Optimization Notes

These optimizations are already implemented in the Audio Layer and **should be applied to the Physiological Layer** for consistency:

### 1. Early Stopping
Training stops when validation accuracy does not improve for `EARLY_STOP_PATIENCE` consecutive epochs (default: 3). This prevents overfitting and saves compute time.

**Relevant code:** [`audio_layer/train.py`](audio_layer/train.py#L180-L193)

### 2. JSON Feature Vector Format
Feature vectors are serialized as JSON arrays (not space-separated or pickled). This is human-readable, CSV-safe, and trivially parsed by any language.

**Relevant code:** [`audio_layer/export_predictions.py`](audio_layer/export_predictions.py#L42-L45)

### 3. Vision-Compatible Combined Results
`experiment_results_with_accuracy.csv` uses a 33+ column schema matching the Vision Layer:
- Test metrics: accuracy, precision/recall/F1 (macro + weighted)
- Confusion matrix cells: `cm_healthy_to_healthy`, `cm_healthy_to_semi`, etc.
- Training curves as JSON arrays: `train_loss_curve`, `val_acc_curve`, etc.

This allows cross-layer comparison in a single table.

**Relevant code:** [`audio_layer/evaluate.py`](audio_layer/evaluate.py#L240-L286)

### 4. Mel Spectrogram Caching
Mel spectrograms are computed once on epoch 1 and saved to `mel_cache/`. Subsequent epochs load cached `.pt` files in ~0.01 s (vs ~2 s for audio decode + transform).

**Relevant code:** [`audio_layer/dataset.py`](audio_layer/dataset.py#L235-L256)

### 5. Automatic Mixed Precision (AMP)
AMP halves VRAM usage and speeds up training by ~30% on supported GPUs with negligible accuracy impact. Enabled by default; toggle via `USE_AMP` in config.

### 6. Gradient Checkpointing
Trades a small amount of compute for ~40% memory savings by recomputing intermediate activations during backpropagation. Enabled by default; toggle via `GRADIENT_CHECKPOINTING` in config.

---

## Usage Examples

### Train the Audio Layer (single run)

```bash
python main.py
```

### Run all ablation experiments

```bash
python run_ablation.py
```

### Check outputs after training

```bash
# List all output files
ls outputs/

# Preview predictions.csv
head -5 outputs/predictions.csv

# Check combined experiment results
python -c "import pandas as pd; df = pd.read_csv('outputs/experiment_results_with_accuracy.csv'); print(df[['exp_id', 'config_label', 'test_f1_macro']].to_string())"

# View confusion matrix
# Open outputs/1_baseline/confusion_matrix.png in any image viewer
```

### Check training progress during a run

```bash
# Linux/macOS
tail -f training_ablation.log

# Windows PowerShell
Get-Content training_ablation.log -Wait
```

### Run a single experiment manually

```python
# Custom experiment via Python
from audio_layer import (
    set_seeds, apply_overrides, AudioSpectrogramTransformer,
    scan_audio_files, create_dataloaders,
    train_model, evaluate_model, export_predictions,
)
from audio_layer import config as cfg

# Override config for custom experiment
apply_overrides(NUM_CLASSES=2, BATCH_SIZE=16, LR=1e-4,
                STATUS_TO_LABEL={"healthy": 0, "symptomatic": 1, "COVID-19": 1},
                ID2LABEL={0: "healthy", 1: "unhealthy"})

set_seeds()
file_paths, labels = scan_audio_files()
# ... continue with training pipeline (see main.py for full flow)
```

---

## Contribution Notes

### For Team Members

This repository is one of three layers in the Intel Cup 2026 Multimodal pipeline:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐
│   Vision Layer   │  │   Audio Layer   │  │  Physiological Layer    │
│  (CT/X-ray imgs) │  │ (cough sounds)  │  │  (vital signs, etc.)    │
└────────┬────────┘  └────────┬────────┘  └───────────┬─────────────┘
         │                    │                        │
         ▼                    ▼                        ▼
  predictions.csv      predictions.csv           predictions.csv
         │                    │                        │
         └────────────────────┼────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Fusion Layer    │
                    │  (concatenates    │
                    │   all 3 CSVs)     │
                    └──────────────────┘
```

### Physiological Layer Team

Apply the same optimizations used here:
- Output `predictions.csv` with columns: `filename`, `prediction`, `label`, `feature_vector`
- Use JSON arrays for `feature_vector`
- Output `experiment_results_with_accuracy.csv` with the vision-compatible 33+ columns
- Implement early stopping

### Prerequisites for Fusion Layer

The Fusion Layer needs **all three `predictions.csv` files** to exist in a known directory:

```
fusion_inputs/
├── vision_predictions.csv
├── audio_predictions.csv
└── physiological_predictions.csv
```

If any layer's `predictions.csv` is missing, the Fusion Layer cannot run.

---

## Fusion Layer Example

### Concatenating the Three Layers' Predictions

Below is a reference implementation showing how the Fusion Layer should concatenate the three `predictions.csv` files into a single input matrix:

```python
"""
fusion_layer.py — Example: Concatenate Vision, Audio, and Physiological
predictions.csv files into a unified feature matrix for the Fusion Layer.
"""
import json
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. Load all three layers' predictions
# ------------------------------------------------------------------
VISION_CSV = "fusion_inputs/vision_predictions.csv"
AUDIO_CSV  = "fusion_inputs/audio_predictions.csv"
PHYSIO_CSV = "fusion_inputs/physiological_predictions.csv"

vision_df  = pd.read_csv(VISION_CSV)
audio_df   = pd.read_csv(AUDIO_CSV)
physio_df  = pd.read_csv(PHYSIO_CSV)

print(f"Vision:         {len(vision_df)} samples")
print(f"Audio:          {len(audio_df)} samples")
print(f"Physiological:  {len(physio_df)} samples")

# ------------------------------------------------------------------
# 2. Parse JSON feature vectors into numpy arrays
# ------------------------------------------------------------------
def parse_features(df: pd.DataFrame, layer_name: str) -> np.ndarray:
    """Convert the 'feature_vector' JSON column to a 2D numpy array."""
    vectors = df["feature_vector"].apply(json.loads).tolist()
    X = np.array(vectors, dtype=np.float32)
    print(f"  [{layer_name}] feature dim: {X.shape[1]}")
    return X

X_vision = parse_features(vision_df, "Vision")
X_audio  = parse_features(audio_df,  "Audio")
X_physio = parse_features(physio_df, "Physiological")

# ------------------------------------------------------------------
# 3. Concatenate feature vectors horizontally
#    Each row = [vision_features | audio_features | physio_features]
# ------------------------------------------------------------------
# Assumption: rows are aligned by sample index (same ordering of subjects).
# If samples differ across layers, align by filename first.
if len(X_vision) == len(X_audio) == len(X_physio):
    X_fused = np.concatenate([X_vision, X_audio, X_physio], axis=1)
    y_fused = vision_df["label"].values  # ground truth (same across layers)
    print(f"\n[Fused] shape: {X_fused.shape}")
    print(f"[Fused] total feature dim: {X_fused.shape[1]}")
else:
    # --- Alignment by filename if sample counts differ ---
    print("\nSample counts differ — aligning by filename...")
    # Use vision as anchor; extend to common intersection
    common = (
        set(vision_df["filename"])
        & set(audio_df["filename"])
        & set(physio_df["filename"])
    )
    print(f"Common samples across all 3 layers: {len(common)}")

    def filter_and_sort(df, common_set):
        df = df[df["filename"].isin(common_set)].copy()
        df = df.sort_values("filename").reset_index(drop=True)
        return df

    vision_df  = filter_and_sort(vision_df, common)
    audio_df   = filter_and_sort(audio_df, common)
    physio_df  = filter_and_sort(physio_df, common)

    X_vision = parse_features(vision_df, "Vision")
    X_audio  = parse_features(audio_df,  "Audio")
    X_physio = parse_features(physio_df, "Physiological")

    X_fused = np.concatenate([X_vision, X_audio, X_physio], axis=1)
    y_fused = vision_df["label"].values
    print(f"[Fused] shape: {X_fused.shape}")

# ------------------------------------------------------------------
# 4. Save fused data for downstream training
# ------------------------------------------------------------------
np.savez("fusion_inputs/fused_features.npz",
         X=X_fused, y=y_fused,
         filenames=vision_df["filename"].values)
print("\nFused features saved to: fusion_inputs/fused_features.npz")
```

### What the Fusion Layer Gets

```
                    ┌─────────────────────────────────────────────┐
                    │          Fused Feature Vector               │
                    ├──────────────┬──────────────┬───────────────┤
                    │ Vision Feats │ Audio Feats  │ Physio Feats  │
                    │  (dim: N₁)   │  (dim: 128)  │   (dim: N₃)   │
                    └──────────────┴──────────────┴───────────────┘
                                     ↓
                            Fusion Classifier
                          (final prediction)
```

### Contract Checklist for All Layers

Before the Fusion Layer can run, verify each layer's `predictions.csv`:

- [ ] Contains columns: `filename`, `prediction`, `label`, `feature_vector`
- [ ] `feature_vector` is a valid JSON array string (e.g., `"[0.1, 0.2, ...]"`)
- [ ] `filename` values are consistent across layers (same naming convention)
- [ ] `label` values are consistent (same ground truth for the same subject)
- [ ] CSV is saved in a location accessible to the Fusion Layer

---

## Troubleshooting

### `metadata_compiled.csv` not found
Ensure the file is at `datasets/public_dataset_v3/coughvid_20211012/metadata_compiled.csv`. This is the expected path; update `DATA_DIR` in `audio_layer/config.py` if you use a different layout.

### `.webm` files fail to load
Install PyAV: `pip install av`. This bundles ffmpeg for `.webm` container support. Without it, `torchaudio` may fail on `.webm` files.

### Out of memory (OOM) on GPU
- Reduce `BATCH_SIZE` in config (e.g., from 32 → 16)
- Ensure `USE_AMP = True` (saves ~40% VRAM)
- Ensure `GRADIENT_CHECKPOINTING = True` (saves ~40% VRAM)
- Use `MAX_FRAMES = 128` instead of `192` (shorter temporal dimension)

### Zero matching audio files
The CSV UUIDs don't match your disk files. Check:
- Are audio files (`.webm`, `.wav`, `.ogg`) present in the data directory?
- Does `DATA_DIR` point to the correct folder?
- Run: `ls datasets/public_dataset_v3/coughvid_20211012/ | head -5` to verify

---

## Citation

**COUGHVID Dataset:**
> Orlandic, L., Teijeiro, T., & Atienza, D. (2021). The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. *Scientific Data*, 8(1), 156.  
> [https://doi.org/10.1038/s41597-021-00937-4](https://doi.org/10.1038/s41597-021-00937-4)

---

## License

MIT License — see [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Intel Cup 2026 · Multimodal Audio Layer</b><br>
  Maintained by <a href="https://github.com/CHANSingYeungSunny">CHANSingYeungSunny</a>
</p>
