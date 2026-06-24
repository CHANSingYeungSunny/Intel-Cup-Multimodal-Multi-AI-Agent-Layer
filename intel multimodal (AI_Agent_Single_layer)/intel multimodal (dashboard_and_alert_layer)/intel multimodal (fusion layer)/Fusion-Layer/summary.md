# Fusion Layer — Summary

## What This Is

The Fusion Layer combines features from three independent health-monitoring modalities — **Vision** (facial rPPG/video), **Audio** (cough sounds), and **Physiological** (PPG/ECG signals) — into a unified multimodal representation for health state classification.

## Multimodal Fusion Approach

### The Challenge

The three modalities come from **different datasets with different subjects**:
- Vision: UBFC rPPG dataset (42+8 subjects, facial video)
- Audio: COUGHVID dataset (~2,900 subjects, cough recordings)
- Physiological: BIDMC PPG dataset (53 subjects, bedside monitoring)

There is no natural alignment — no subject appears in more than one dataset.

### Our Solution: Label-Matched Pairing

We use **label-matched pairing** to create a synthetic multimodal dataset:

1. **Anchor modality**: Vision (607 windows — the smallest dataset)
2. For each Vision sample with health label `L`:
   - Randomly select one Audio sample with the same label `L`
   - Randomly select one Physiological sample with the same label `L`
3. Concatenate features: `[768 (vision) | 128 (audio) | 128 (physio)]` = **1024-dim vector**
4. Result: 607-sample multimodal dataset

This is standard practice in multimodal learning when modalities come from independent sources. The underlying assumption is that features from subjects with the same health state carry similar information, regardless of which specific subject they come from.

### Model Architecture

**MultimodalFusionEncoder** — A Transformer that models cross-modal interactions:

1. **Tokenization**: Each modality's feature chunk is linearly projected to a shared 256-dim space:
   - Vision (768 → 256)
   - Audio (128 → 256)
   - Physiological (128 → 256)
2. **CLS Token**: A learnable classification token is prepended to the sequence
3. **Positional Encoding**: Learnable per-position embeddings (4 positions: CLS, V, A, P)
4. **Transformer Encoder**: 4 layers, 8 heads, d_model=256, d_ff=512, GELU, pre-norm
   - All 4 tokens attend to each other — this inherently models cross-modal interactions (Vision↔Audio, Vision↔Physio, Audio↔Physio)
5. **Classifier Head**: CLS token → LayerNorm → Linear → GELU → Dropout → Linear(num_classes)
6. **Output**: Returns `(logits, fusion_embedding)` where fusion_embedding is the 256-dim CLS token

**Parameters**: ~2–3M (lightweight, GPU-friendly)

## Optimizations

### Memory Efficiency (for 4GB GPU)

| Technique | Benefit |
|-----------|---------|
| AMP (Automatic Mixed Precision) | Half-precision where safe; ~2× memory saving |
| Gradient Checkpointing | Per-layer recomputation; ~40% memory saving |
| Feature Caching (.pt) | Skip repeated CSV parsing; instant reload |
| Small d_model (256) | Only 4 tokens × 256 dims = minimal memory |

Combined, these enable **batch_size=128 on a 4GB GPU**.

### Training Efficiency

| Technique | Benefit |
|-----------|---------|
| Early Stopping (patience=3) | Stop when validation F1 plateaus |
| Focal Loss (gamma=2.0) | Focus on hard examples, handle class imbalance |
| WeightedRandomSampler | Balanced batches despite class skew |
| Weight Decay (1e-4) | Regularization to prevent overfitting |

## Experiment Design

5 experiments test different configurations:

| Exp | Name | Rationale |
|-----|------|-----------|
| 1 | Baseline (3-class, bs=64) | Reference point for all comparisons |
| 2 | **Binary (bs=64)** | Simplifies to healthy vs symptomatic — expected best F1 |
| 3 | Batch128 (3-class) | Tests large-batch training stability |
| 4 | Freeze encoder layer 0 | Tests impact of frozen pre-trained-like representations |
| 5 | Lower LR (1e-4) | Tests convergence with smaller learning rate |

**Exp 2 (binary) is expected to perform best** because merging sub-healthy and unhealthy into one class reduces the problem to a binary distinction, which is inherently easier and yields higher F1 scores.

## Why This Works

1. **Cross-modal attention**: The Transformer's self-attention lets each modality "query" the others — e.g., the Vision token can attend to Audio and Physiological tokens to resolve ambiguous cases.

2. **Label-matched pairing is valid**: Subjects with the same health label have similar underlying physiology. While pairing different subjects introduces noise, the Transformer learns to focus on consistent health-relevant patterns across modalities.

3. **CLS token aggregation**: Rather than hand-designing a fusion rule (sum, concat, gating), the CLS token learns to gather the most relevant information from all three modality tokens through attention.

## Reproducibility

- Fixed seed (42) for all randomness: pairing, splitting, sampling, initialization
- Deterministic CuDNN: `torch.backends.cudnn.deterministic = True`
- Cached aligned features prevent re-pairing on subsequent runs
- `python fusion_train.py --exp_id 2` always produces identical results

## Output Format Compliance

The Fusion Layer produces outputs in the same format as the three monomodal layers:

- **predictions.csv**: `filename, prediction, label, feature_vector` (256-dim fusion embedding)
- **experiment_results_with_accuracy.csv**: 33 columns identical to Vision/Audio/Physio format
- **best_fusion.pt**: Full checkpoint with model weights, config, and training history
- **Plots**: Training curves (loss + accuracy/F1) and confusion matrix heatmap

## Limitations

1. **Cross-dataset noise**: Label-matched pairing across different datasets introduces variance that a truly multimodal dataset (same subjects, all modalities) would not have.
2. **Small fused dataset**: 607 samples (limited by Vision, the smallest modality) — data augmentation or semi-supervised approaches could help.
3. **No temporal alignment**: Each modality's features come from different time windows; temporal synchronization is not modeled.
4. **No longitudinal data for forecasting**: The fused dataset consists of i.i.d. samples from different subjects matched by label. There is no true temporal ordering, so meaningful health trend forecasting is not possible. A forecasting head is architected in the model (`forecast_horizon` parameter) but not enabled or trained.

## Forecasting (Future Extension)

A forecasting head is architected in `fusion_model.py` (`forecast_horizon` parameter and `forecast_head` attribute) but **not enabled** in this competition because the dataset lacks true longitudinal sequences. Pseudo-sequences created by same-label grouping produce trivial results (the model learns that labels never change within a sequence). Future work can extend this module when longitudinal multimodal datasets with real temporal progression become available.
