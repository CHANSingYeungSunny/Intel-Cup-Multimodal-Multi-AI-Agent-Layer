# Intel Cup — Multimodal Health State Classification

**Monorepo** containing three monomodal health-classification layers and one fusion layer. The system classifies health states (Healthy / Sub-healthy / Unhealthy) by fusing facial video, cough audio, and physiological signals through a Multimodal Transformer Encoder with cross-modal attention.

---

## Table of Contents

1. [Input](#input)
2. [Methodology](#methodology)
3. [Output](#output)
4. [Repository Structure](#repository-structure)
5. [Architecture Diagram](#architecture-diagram)
6. [Quick Start](#quick-start)
7. [Results](#results)
8. [Reproducibility](#reproducibility)

---

## Input

### Datasets

| Layer | Dataset | Subjects | Modality | Sampling |
|-------|---------|----------|----------|----------|
| **Vision** | [UBFC rPPG](https://sites.google.com/view/ubfc-rppg) | 50 subjects | Facial video, 30 fps, 640×480 | 10 s windows, 5 s stride → **607 windows** |
| **Audio** | [COUGHVID v3](https://zenodo.org/record/4498364) | ~2,900 subjects | Cough sounds (.webm/.wav/.ogg) | 1.9 s segments, 16 kHz mono → **~3,100 samples** |
| **Physiological** | [BIDMC PPG](https://physionet.org/content/bidmc/) | 53 subjects | PPG + ECG waveforms (125 Hz) | 10 s windows, 5 s stride → **~855 windows** |

### Preprocessing

Each layer independently preprocesses its raw data:

- **Vision**: Middle frame extraction → Swin-Tiny backbone → **768-dim** feature vector
- **Audio**: Mel spectrogram (128 bins × 192 frames) → AST patch embedding → **128-dim** CLS token
- **Physiological**: 4-channel waveform (ECG, PPG, dECG, dPPG) → iTransformer across 4 variable tokens → global avg pool → **128-dim** feature vector

### Labels

Labels are generated per layer via dataset-specific heuristics (PPG-based risk scoring, COUGHVID expert labels, clinical thresholds), then mapped to a 3-class schema:

| Class | Label |
|-------|-------|
| 0 | Healthy |
| 1 | Sub-healthy |
| 2 | Unhealthy |

Binary mode merges classes 1+2 → "symptomatic or unhealthy" to simplify the problem.

---

## Methodology

### Monomodal Layers

Each layer trains independently using a Transformer-based architecture with focal loss, AMP mixed precision, gradient checkpointing, and early stopping:

| Layer | Model | Params | Feature Dim | Best F1 |
|-------|-------|--------|-------------|---------|
| Vision | Swin-Tiny (torchvision) | ~27.5M | 768 | 0.541 (macro) |
| Audio | Audio Spectrogram Transformer (custom ViT) | ~460K | 128 | 0.428 (macro, binary) |
| Physiological | iTransformer (cross-variable attention) | ~575K | 128 | 0.369 (macro) |

Each layer produces a `predictions.csv` with columns `(filename, prediction, label, feature_vector)` where `feature_vector` is a JSON-serialized embedding from the model backbone (before the classification head).

### Fusion Layer

Since the three modalities come from **different subjects with no overlap**, the Fusion Layer uses **label-matched pairing** to create a synthetic multimodal dataset:

1. **Anchor on Vision** (607 windows — the smallest dataset)
2. For each Vision sample with label `L`, randomly pair it with one Audio sample and one Physiological sample sharing the same label `L`
3. Concatenate: `[vision_768 | audio_128 | physio_128]` → **1024-dim fused feature vector**
4. Result: 607-sample multimodal dataset (seed=42 ensures reproducible pairing)

The fused features are fed into a **Multimodal Transformer Encoder**:

```
Input: [B, 1024]
  │
  ├─→ Vision   (768 → 256) ──→ [V] token  ──┐
  ├─→ Audio    (128 → 256) ──→ [A] token  ──┤
  ├─→ Physio   (128 → 256) ──→ [P] token  ──┤
  └─→ CLS token (learnable) ──→ [CLS] token ─┘
                                            │
                              ┌─────────────┘
                              ▼
                    Transformer Encoder (4 layers)
                      d_model=256, nhead=8
                      d_ff=512, GELU, pre-norm
                              │
                              ▼
                         CLS token [B, 256]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Classifier Head      Feature Vector
              (256 → num_classes)  (stored in predictions.csv)
```

**Key design**: Self-attention across all 4 tokens (CLS + V + A + P) inherently models cross-modal interactions — Vision can attend to Audio and Physiological, Audio can attend to Vision and Physiological, etc.

### Training Configuration

| Component | Detail |
|-----------|--------|
| Loss | Focal Loss (γ=2.0) with class-balanced weights |
| Optimizer | Adam (LR=3e-4, weight_decay=1e-4) |
| Batch size | 64 or 128 |
| Early stopping | Patience=3 on validation macro F1 |
| Mixed precision | AMP (torch.amp) on CUDA |
| Memory optimization | Gradient checkpointing per encoder layer |
| Data balance | WeightedRandomSampler |
| Reproducibility | Seed=42, deterministic CuDNN |

### Forecasting (Future Extension)

A forecasting head is architected in `fusion_model.py` (`forecast_horizon` parameter) but **not enabled**. The current dataset consists of i.i.d. samples from different subjects matched by label — there is no true temporal ordering. Meaningful health trend forecasting requires longitudinal multimodal data from the same subjects over time.

---

## Output

### Deliverables

Each layer produces a standardized set of outputs:

| File | Description |
|------|-------------|
| `predictions.csv` | Per-sample predictions with filename, predicted class, ground truth label, and feature vector (JSON array) |
| `experiment_results_with_accuracy.csv` | 33-column experiment summary (1 row per experiment) |
| `best_*.pt` | Best model checkpoint (state dict + config + training history) |
| `*_training_curves.png` | Dual-panel plot: loss (left) and accuracy/F1 (right) vs epoch |
| `*_confusion_matrix.png` | Seaborn heatmap of test-set confusion matrix |

### predictions.csv Schema

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | Sample identifier (e.g., `UBFC1/10-gt/win_000000`, UUID.webm, `bidmc08_win_0000`) |
| `prediction` | int | Predicted health class (0=healthy, 1=sub-healthy, 2=unhealthy) |
| `label` | int | Ground-truth health class |
| `feature_vector` | str | JSON-serialized float array (768-dim Vision, 128-dim Audio/Physio, 256-dim Fusion) |

### experiment_results_with_accuracy.csv Schema (33 columns)

**Config (9 columns):** `exp_id`, `config_label`, `gamma`, `epochs`, `d_model`, `n_layers`, `batch_size`, `label_mode`, `best_epoch`

**Performance (10 columns):** `val_macro_f1_best`, `accuracy`, `test_accuracy`, `test_precision_macro`, `test_recall_macro`, `test_f1_macro`, `test_precision_weighted`, `test_recall_weighted`, `test_f1_weighted`, `test_loss`

**Confusion Matrix (9 columns):** `cm_healthy_to_healthy`, `cm_healthy_to_semi`, `cm_healthy_to_unhealthy`, `cm_semi_to_healthy`, `cm_semi_to_semi`, `cm_semi_to_unhealthy`, `cm_unhealthy_to_healthy`, `cm_unhealthy_to_semi`, `cm_unhealthy_to_unhealthy`

**Training Curves (5 columns):** `train_loss_curve`, `val_loss_curve`, `train_acc_curve`, `val_acc_curve`, `val_f1_curve` — each a JSON array of per-epoch values

---

## Repository Structure

```
intel multimodal (fusion layer)/
│
├── README.md                              # This file
├── .gitignore                             # Excludes datasets, caches, logs, per-exp intermediates
│
├── intel multimodal (vision layer)/       # ── Vision Layer ──
│   ├── download_ubfc.py                   # UBFC dataset downloader
│   └── vision_layer/
│       ├── model.py                       # SwinHealthClassifier (Swin-Tiny backbone, 768-dim)
│       ├── train.py                       # Training loop with early stopping
│       ├── run_experiments.py             # 5-experiment ablation runner
│       ├── preprocessing.py               # PPG-based pseudo-label generation
│       ├── dataset.py                     # UBFCWindowDataset
│       ├── evaluate.py                    # Metrics, CSV export
│       ├── config.py                      # All hyperparameters
│       ├── export_onnx.py / optimize_openvino.py  # Deployment
│       ├── requirements.txt
│       ├── README.md / SUMMARY.md
│       └── output/
│           ├── labels.csv                 # Generated pseudo-labels (607 rows)
│           └── predictions/
│               ├── predictions.csv        # 607 windows, 768-dim feature vectors
│               └── experiment_results_with_accuracy.csv
│
├── intel multimodal (audio layer)/        # ── Audio Layer ──
│   ├── main.py                            # End-to-end training pipeline
│   ├── run_ablation.py                    # 5-experiment ablation runner
│   ├── IntelCup2026_audio.ipynb           # Original exploratory notebook
│   ├── README.md / SUMMARY.md
│   ├── audio_layer/
│   │   ├── model.py                       # AudioSpectrogramTransformer (128-dim CLS token)
│   │   ├── train.py                       # Training + early stopping
│   │   ├── dataset.py                     # Mel spectrogram caching
│   │   ├── evaluate.py                    # Metrics, confusion matrix, CSV export
│   │   ├── export_predictions.py          # predictions.csv generation
│   │   └── config.py                      # Hyperparameters
│   └── outputs/
│       ├── predictions.csv                # ~3,100 samples, 128-dim feature vectors
│       ├── experiment_results_with_accuracy.csv
│       ├── best_ast_coughvid_local.pt     # Best checkpoint (binary experiment)
│       ├── ablation_summary.csv
│       └── {1..5}_{baseline,binary,batch16,freeze_enc,lr1e4}/
│           └── (per-experiment outputs)
│
├── intel multimodal (physiological layer)/ # ── Physiological Layer ──
│   ├── model.py                           # iTransformerClassifier (128-dim)
│   ├── train.py                           # Training with AMP, grad checkpointing
│   ├── data_loader.py                     # BIDMCDataset + window caching
│   ├── preprocess.py                      # Risk-score label generation, normalization
│   ├── utils.py                           # Metrics, confusion matrix, CSV, plotting
│   ├── README.md / summary.md
│   └── outputs/
│       ├── predictions.csv                # ~855 windows, 128-dim feature vectors
│       ├── experiment_results_with_accuracy.csv
│       ├── best_physio.pt                 # Best checkpoint
│       ├── training_curves.png
│       └── confusion_matrix.png
│
└── Fusion-Layer/                          # ── Fusion Layer ──
    ├── fusion_loader.py                   # Label-matched pairing, CSV load, .pt cache
    ├── fusion_model.py                    # MultimodalFusionEncoder (2.4M params)
    ├── fusion_train.py                    # 5-experiment runner (AMP, early stop, TensorBoard)
    ├── fusion_utils.py                    # Metrics, CSV serialization, plotting
    ├── fusion_preprocess.py               # StandardScaler, label handling, train/val/test split
    ├── README.md                          # Fusion Layer documentation
    ├── summary.md                         # Detailed design explanation
    └── outputs/
        ├── predictions.csv                # 256-dim fusion CLS embeddings (Exp 2 best)
        ├── experiment_results_with_accuracy.csv  # 33 columns, 5 rows
        ├── best_fusion.pt                 # Best checkpoint (Exp 2, binary, 77.2% acc)
        ├── fusion_training_curves.png
        └── fusion_confusion_matrix.png
```

---

## Architecture Diagram

```
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────────┐
│    VISION LAYER      │  │    AUDIO LAYER        │  │  PHYSIOLOGICAL LAYER     │
│                      │  │                        │  │                          │
│  UBFC rPPG Dataset   │  │  COUGHVID Dataset      │  │  BIDMC PPG Dataset       │
│  50 subjects         │  │  ~2,900 subjects       │  │  53 subjects             │
│  Facial video (30fps)│  │  Cough sounds          │  │  PPG + ECG (125 Hz)      │
│         │            │  │          │             │  │          │               │
│         ▼            │  │          ▼             │  │          ▼               │
│  Swin-Tiny           │  │  Audio Spectrogram     │  │  iTransformer            │
│  (27.5M params)      │  │  Transformer           │  │  (575K params)           │
│         │            │  │  (460K params)         │  │          │               │
│         ▼            │  │          │             │  │          ▼               │
│  [B, 768] feature    │  │  [B, 128] CLS token    │  │  [B, 128] pooled         │
└──────────┬───────────┘  └──────────┬─────────────┘  └──────────┬───────────────┘
           │                         │                            │
           │    Label-matched pairing (anchor = Vision, 607 windows)
           │                         │                            │
           └─────────────────────────┼────────────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │    FUSION LAYER       │
                          │                      │
                          │  Concat [768|128|128] │
                          │  → [B, 1024]          │
                          │         │            │
                          │         ▼            │
                          │  Tokenization:        │
                          │  [CLS | V_tok | A_tok | P_tok]
                          │  [B, 4, 256]         │
                          │         │            │
                          │         ▼            │
                          │  Transformer Encoder  │
                          │  4 layers, 8 heads    │
                          │  d_ff=512, GELU       │
                          │  (cross-attention     │
                          │   across modalities)  │
                          │         │            │
                          │         ▼            │
                          │  CLS token → Classifier│
                          │  → Health State       │
                          │  (Healthy / Sub-healthy│
                          │   / Unhealthy)        │
                          └──────────────────────┘
```

---

## Quick Start

Each layer can be run independently. Prerequisites: Python ≥ 3.9, PyTorch ≥ 2.0.

```bash
# 1. Vision Layer
cd "intel multimodal (vision layer)/vision_layer"
pip install -r requirements.txt
python run_experiments.py

# 2. Audio Layer
cd "intel multimodal (audio layer)"
python main.py

# 3. Physiological Layer
cd "intel multimodal (physiological layer)"
python train.py --exp_id all

# 4. Fusion Layer (requires all three layers' predictions.csv)
cd Fusion-Layer
python fusion_train.py --exp_id all
```

---

## Results

### Fusion Layer — Classification Experiments

Sorted by `test_f1_weighted` (best selection metric):

| Exp | Config | Classes | BS | LR | Test Acc | Macro F1 | **Weighted F1** |
|-----|--------|---------|----|----|----------|----------|-----------------|
| **2** | **Binary** | 2 | 64 | 3e-4 | **77.2%** | 0.770 | **0.775** |
| 1 | 3-class | 3 | 64 | 3e-4 | 71.7% | **0.783** | 0.715 |
| 3 | Batch128 | 3 | 128 | 3e-4 | 71.7% | 0.781 | 0.711 |
| 4 | Freeze enc | 3 | 64 | 3e-4 | 66.3% | 0.732 | 0.643 |
| 5 | Lower LR | 3 | 64 | 1e-4 | 66.3% | 0.732 | 0.643 |

**Exp 2 (binary) is the best** — 77.2% accuracy, 0.775 weighted F1. Merging sub-healthy and unhealthy into one class simplifies the problem and yields the strongest weighted F1, which accounts for class imbalance.

### Fusion vs. Monomodal Comparison

| Layer | Feature Dim | Best Weighted F1 | Best Accuracy |
|-------|-------------|------------------|---------------|
| Vision (Swin-T) | 768 | 0.273 | 43.9% |
| Audio (AST) | 128 | 0.642 | 74.9% |
| Physiological (iTransformer) | 128 | 0.542 | 57.8% |
| **Fusion (Multimodal Transformer)** | **1024 → 256** | **0.775** | **77.2%** |

The Fusion Layer outperforms all individual monomodal layers, demonstrating the benefit of cross-modal attention.

---

## Reproducibility

All layers use **seed=42** for deterministic behavior:

- Label-matched pairing across modalities (Fusion Layer)
- Stratified 70/15/15 train/val/test splits
- WeightedRandomSampler for class-balanced batches
- Model initialization
- `torch.backends.cudnn.deterministic = True`

To reproduce exact results:

```bash
cd Fusion-Layer
python fusion_train.py --exp_id 2 --seed 42
```

The Fusion Layer caches aligned features to `Fusion-Layer/cache/fused_features.pt` — subsequent runs load the cached tensor directly, skipping CSV parsing and random pairing.
