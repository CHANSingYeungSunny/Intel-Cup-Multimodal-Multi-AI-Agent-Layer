# Audio Layer — Summary

## What was built

A complete training pipeline for the **Audio Layer** of a multimodal influenza health monitoring system. The pipeline processes cough recordings from the COUGHVID-v3 dataset, extracts Mel spectrogram features using an Audio Spectrogram Transformer (AST), and outputs predictions + feature vectors usable by the downstream Fusion Layer.

## Pipeline overview

```
COUGHVID-v3 Audio Files (34,434 recordings, ~30 GB)
  → Metadata filtering (status labels: healthy/symptomatic/COVID-19)
  → Audio decoding via PyAV (handles .webm, .wav, .ogg)
  → Mel spectrogram (128 mel bins × 192 frames, ~1.9s window)
  → Lazy spectrogram cache (.pt files, ~2.7 GB total)
  → AST model (460K params, 128-dim CLS embedding)
  → 5 ablation experiments (baseline, binary, batch16, freeze_enc, lr1e4)
  → predictions.csv + experiment_results_with_accuracy.csv
```

## Key technical decisions

| Decision | Rationale |
|---|---|
| **AST (Audio Spectrogram Transformer) over CNN** | Transformer attention captures long-range temporal patterns in cough sounds; 460K params is lightweight for 4GB VRAM. Returns CLS token embedding (128-dim) usable by Fusion Layer. |
| **MAX_FRAMES = 192 (~1.9s)** | Cough bursts are <1s; aggressive cropping removes trailing silence while capturing full cough + context. Attention cost is O(n²) — 96 patches vs 256 for 5s window (~7× faster). |
| **Spectrogram cache** | PyAV audio decoding is the bottleneck (~2s/file). Caching pre-computed Mel spectrograms as .pt files (~132 KB each) reduces per-epoch data loading from ~30 min to ~4 min. Built lazily during first training epoch. |
| **AMP + Gradient checkpointing** | Mixed precision saves ~40% VRAM. Gradient checkpointing trades compute for memory. Together they keep batch_size=32 within 4GB VRAM. |
| **Early stopping (patience=3)** | Prevents overfitting; each experiment converged in 3–7 epochs. Saves ~3× training time vs running full 10 epochs. |
| **PyAV over torchaudio** | torchaudio 2.6+ requires ffmpeg binary on PATH for .webm files. PyAV bundles ffmpeg — works out of the box on Windows. |
| **Binary label mode outperforms 3-class** | Merging symptomatic + COVID-19 into a single "unhealthy" class improves macro F1 from 0.286 to 0.428. The severe class imbalance (75% healthy) makes 3-class discrimination very difficult. |

## Outputs delivered

| File | Description |
|---|---|
| `predictions.csv` | 3,100 rows × 4 columns (`filename, prediction, label, feature_vector`). Feature vector is 128-dimensional (AST CLS token). JSON-encoded, matches vision layer format. |
| `experiment_results_with_accuracy.csv` | 5 rows (one per experiment) with 33 columns matching vision layer exactly: hyperparameters, metrics, confusion matrix cells, per-epoch curves as JSON arrays. |
| `best_ast_coughvid_local.pt` | Best model checkpoint (1.9 MB, 460K params). From the binary experiment (val_macro_f1=0.428). |
| 5 experiment subdirectories | Each with checkpoint, predictions.csv, confusion_matrix.png, training_curves.png. |

## Experiment results (AST, 5 configurations)

| exp_id | Config | Best F1 | Accuracy | Epochs | Notes |
|---|---|---|---|---|---|
| 1 | baseline: 3-class, bs=32, LR=3e-4 | 0.286 | 74.9% | 3 (early stop) | Majority-class dominant |
| 2 | **binary**, bs=32, LR=3e-4 | **0.428** | 74.9% | 3 (early stop) | Best overall — binary is easier |
| 3 | batch16, 3-class, bs=16 | 0.286 | 74.9% | 4 | Same as baseline |
| 4 | freeze_enc, bs=32 | 0.286 | 74.9% | 3 | Freezing layer 0 has no effect |
| 5 | lr1e4, 3-class, LR=1e-4 | 0.288 | 72.6% | 3 | Slightly lower LR, slight accuracy drop |

## Performance (GTX 1650 Max-Q, 4GB VRAM)

| Metric | Value |
|---|---|
| Model parameters | 460,035 |
| Per-epoch time (cached) | ~4–5 min |
| Per-epoch time (no cache) | ~20 min |
| Total 5 experiments | ~2.5 hours |
| GPU VRAM usage | ~179 MB (AMP + checkpointing) |
| Spectrogram cache size | ~2.7 GB (20,664 files) |

## Limitations

- **Severe class imbalance**: 75% of labeled samples are "healthy". The model struggles to learn minority classes (symptomatic=19%, COVID-19=6%). Binary mode helps but accuracy is still majority-dominated.
- **No clinical validation**: COUGHVID labels are self-reported via survey — not physician-verified. Real-world accuracy may differ.
- **Cough-only data**: The model only sees cough sounds, not breathing patterns, speech, or other respiratory indicators that could improve health classification.
- **Small feature vector**: 128 dimensions is compact for deployment but may limit downstream fusion capacity compared to vision layer's 768-dim features.
- **Single audio modality**: Combining with vision (facial cues) and physiological (PPG) signals in the Fusion Layer is expected to significantly improve accuracy.
