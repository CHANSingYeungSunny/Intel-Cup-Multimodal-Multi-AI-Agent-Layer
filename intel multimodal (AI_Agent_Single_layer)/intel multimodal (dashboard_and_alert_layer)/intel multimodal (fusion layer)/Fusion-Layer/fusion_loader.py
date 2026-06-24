"""
fusion_loader.py — Load and align predictions.csv from Vision, Audio, Physiological.

Strategy: Label-matched pairing across independent datasets.
- Vision (UBFC): 607 windows, 768-dim features
- Audio (COUGHVID): ~3100 samples, 128-dim features
- Physiological (BIDMC): ~5035 windows, 128-dim features

Since these are DIFFERENT datasets with no shared subjects, we anchor on the
smallest modality (Vision, ~607 samples) and pair each sample with randomly
selected Audio + Physiological samples sharing the same label.

Output: concatenated [vision_768 | audio_128 | physio_128] = 1024-dim vectors.
"""

import os
import json
import numpy as np
import pandas as pd
import torch

from fusion_utils import set_seed


# ---------------------------------------------------------------------------
# Paths (relative to Fusion-Layer/)
# ---------------------------------------------------------------------------

MONOREPO_ROOT = os.path.dirname(os.path.abspath(__file__))  # Fusion-Layer/
PARENT = os.path.dirname(MONOREPO_ROOT)  # monorepo root

VISION_PREDS = os.path.join(
    PARENT, "intel multimodal (vision layer)", "vision_layer",
    "output", "predictions", "predictions.csv",
)
AUDIO_PREDS = os.path.join(
    PARENT, "intel multimodal (audio layer)", "outputs", "predictions.csv",
)
PHYSIO_PREDS = os.path.join(
    PARENT, "intel multimodal (physiological layer)", "outputs", "predictions.csv",
)

CACHE_DIR = os.path.join(MONOREPO_ROOT, "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "fused_features.pt")


# ---------------------------------------------------------------------------
# Low-level CSV loading
# ---------------------------------------------------------------------------

def _load_csv(path: str, modality: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a predictions.csv and return (features, labels, filenames).

    Args:
        path: Path to predictions.csv.
        modality: 'vision', 'audio', or 'physio'.

    Returns:
        features:  [N, D] float32 array.
        labels:    [N] int64 array.
        filenames: [N] str array.
    """
    df = pd.read_csv(path)

    # Parse JSON feature vectors → numpy
    features = np.array([
        json.loads(fv) if isinstance(fv, str) else fv
        for fv in df["feature_vector"].values
    ], dtype=np.float32)

    labels = df["label"].values.astype(np.int64)
    filenames = df["filename"].values.astype(str)

    print(f"  [{modality}] Loaded {len(df)} samples, feature dim={features.shape[1]}")
    return features, labels, filenames


# ---------------------------------------------------------------------------
# Label-matched pairing
# ---------------------------------------------------------------------------

def _pair_by_label(
    anchor_feats: np.ndarray,
    anchor_labels: np.ndarray,
    anchor_names: np.ndarray,
    other_feats: np.ndarray,
    other_labels: np.ndarray,
    other_names: np.ndarray,
    rng: np.random.RandomState,
    other_modality: str,
) -> tuple[np.ndarray, np.ndarray]:
    """For each anchor sample, randomly select one other sample with the same label.

    Returns (paired_features, paired_names).
    """
    paired_feats = np.zeros((len(anchor_labels), other_feats.shape[1]), dtype=np.float32)
    paired_names = np.empty(len(anchor_labels), dtype=object)

    # Build label → index lookup
    label_to_indices = {}
    for i, lbl in enumerate(other_labels):
        label_to_indices.setdefault(int(lbl), []).append(i)

    skipped = 0
    for i, lbl in enumerate(anchor_labels):
        lbl = int(lbl)
        candidates = label_to_indices.get(lbl)
        if candidates is None or len(candidates) == 0:
            # No matching label in the other modality — use mean of all
            paired_feats[i] = other_feats.mean(axis=0)
            paired_names[i] = f"{other_modality}_mean"
            skipped += 1
        else:
            idx = rng.choice(candidates)
            paired_feats[i] = other_feats[idx]
            paired_names[i] = other_names[idx]

    if skipped > 0:
        print(f"  [{other_modality}] {skipped} samples had no label match (using mean)")

    return paired_feats, paired_names


# ---------------------------------------------------------------------------
# Main fused data loader
# ---------------------------------------------------------------------------

def load_fused_data(
    seed: int = 42,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and align multimodal features via label-matched pairing.

    Args:
        seed: Random seed for reproducible pairing.
        use_cache: If True and cache exists, load from cache.

    Returns:
        X_fused:   [N, 1024] concatenated feature matrix.
        y:         [N] labels from vision (anchor modality).
        filenames: [N] composite filenames.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if use_cache and os.path.exists(CACHE_PATH):
        print(f"[fusion_loader] Loading cached fused features from {CACHE_PATH}")
        data = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        return data["X"].numpy(), data["y"].numpy(), data["filenames"]

    set_seed(seed)
    rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # 1. Load all three CSVs
    # ------------------------------------------------------------------
    print("[fusion_loader] Loading predictions from monomodal layers...")
    print(f"  Vision:        {VISION_PREDS}")
    print(f"  Audio:         {AUDIO_PREDS}")
    print(f"  Physiological: {PHYSIO_PREDS}")

    for path, modality in [
        (VISION_PREDS, "vision"),
        (AUDIO_PREDS, "audio"),
        (PHYSIO_PREDS, "physio"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{modality} predictions.csv not found at: {path}\n"
                f"Please ensure the {modality} layer has been run and produces predictions.csv."
            )

    v_feats, v_labels, v_names = _load_csv(VISION_PREDS, "vision")
    a_feats, a_labels, a_names = _load_csv(AUDIO_PREDS, "audio")
    p_feats, p_labels, p_names = _load_csv(PHYSIO_PREDS, "physio")

    # ------------------------------------------------------------------
    # 2. Pair: Vision (anchor) → Audio + Physiological
    # ------------------------------------------------------------------
    print(f"\n[fusion_loader] Label-matched pairing (anchor=vision, {len(v_labels)} samples)...")

    a_paired, a_paired_names = _pair_by_label(
        v_feats, v_labels, v_names, a_feats, a_labels, a_names, rng, "audio"
    )
    p_paired, p_paired_names = _pair_by_label(
        v_feats, v_labels, v_names, p_feats, p_labels, p_names, rng, "physio"
    )

    # ------------------------------------------------------------------
    # 3. Concatenate → [N, 768 + 128 + 128] = [N, 1024]
    # ------------------------------------------------------------------
    X_fused = np.concatenate([v_feats, a_paired, p_paired], axis=1)
    y_fused = v_labels.copy()

    # Composite filenames for traceability
    composite_names = np.array([
        f"v:{vn}|a:{an}|p:{pn}"
        for vn, an, pn in zip(v_names, a_paired_names, p_paired_names)
    ])

    print(f"\n[fusion_loader] Fused dataset: {X_fused.shape[0]} samples, "
          f"{X_fused.shape[1]} features (=768+128+128)")
    print(f"  Label distribution: {dict(zip(*np.unique(y_fused, return_counts=True)))}")

    # ------------------------------------------------------------------
    # 4. Cache
    # ------------------------------------------------------------------
    torch.save({
        "X": torch.from_numpy(X_fused),
        "y": torch.from_numpy(y_fused),
        "filenames": composite_names.tolist(),
    }, CACHE_PATH)
    print(f"[fusion_loader] Cached fused features → {CACHE_PATH}")

    return X_fused, y_fused, composite_names


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Fusion Loader — Self-Test")
    print("=" * 60)
    X, y, names = load_fused_data(use_cache=False)
    print(f"\nX shape:  {X.shape}")
    print(f"y shape:  {y.shape}")
    print(f"Names:    {len(names)}")
    print(f"X dtype:  {X.dtype}")
    print(f"y dtype:  {y.dtype}")
    print(f"First 3 names: {names[:3]}")
    print(f"First feature vector (first 10 dims): {X[0, :10]}")
