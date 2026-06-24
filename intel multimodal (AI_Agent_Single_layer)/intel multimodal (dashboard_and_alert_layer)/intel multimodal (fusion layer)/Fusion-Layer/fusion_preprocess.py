"""
fusion_preprocess.py — Normalization, label handling, train/val/test split.

Prepares the fused multimodal dataset for training:
- StandardScaler normalization (fit on train, apply to val/test).
- Label mapping for 3-class and binary modes.
- Stratified train/val/test split (70/15/15).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

ID2LABEL_3CLASS = {0: "healthy", 1: "sub_healthy", 2: "unhealthy"}
ID2LABEL_BINARY = {0: "healthy", 1: "symptomatic_or_unhealthy"}

LABEL2ID_3CLASS = {v: k for k, v in ID2LABEL_3CLASS.items()}
LABEL2ID_BINARY = {v: k for k, v in ID2LABEL_BINARY.items()}


def to_binary_labels(y: np.ndarray) -> np.ndarray:
    """Merge classes 1 and 2 into class 1 (healthy vs symptomatic/unhealthy).

    Args:
        y: Array of integer labels (0, 1, 2).

    Returns:
        Binary labels (0, 1) where 1 = sub_healthy or unhealthy.
    """
    y_bin = y.copy()
    y_bin[y_bin >= 1] = 1
    return y_bin


def get_class_names(num_classes: int) -> list:
    """Return class name list for the given number of classes."""
    if num_classes == 2:
        return ["healthy", "symptomatic_or_unhealthy"]
    else:
        return ["healthy", "sub_healthy", "unhealthy"]


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    filenames: np.ndarray = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> dict:
    """Stratified train/val/test split.

    Args:
        X: Feature matrix [N, D].
        y: Labels [N].
        filenames: Optional filenames array [N].
        test_size: Fraction for test set (default 0.15).
        val_size: Fraction for validation set relative to (1 - test_size).
        seed: Random seed.
        stratify: If True, use stratified split.

    Returns:
        Dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        (filenames_train, filenames_val, filenames_test).
    """
    stratify_arr = y if stratify else None

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify_arr,
    )

    if filenames is not None:
        f_train_val, f_test = train_test_split(
            filenames, test_size=test_size, random_state=seed, stratify=stratify_arr,
        )
    else:
        f_train_val, f_test = None, None

    # Second split: train vs val
    val_frac = val_size / (1.0 - test_size)
    stratify_val = y_train_val if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_frac,
        random_state=seed, stratify=stratify_val,
    )

    if filenames is not None:
        f_train, f_val = train_test_split(
            f_train_val, test_size=val_frac, random_state=seed, stratify=stratify_val,
        )
    else:
        f_train, f_val = None, None

    result = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    if filenames is not None:
        result["filenames_train"] = f_train
        result["filenames_val"] = f_val
        result["filenames_test"] = f_test

    return result


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_features(split: dict) -> dict:
    """Fit StandardScaler on training features, transform all splits.

    Args:
        split: Dict from split_data() with X_train, X_val, X_test.

    Returns:
        Updated split dict with normalized features.
    """
    scaler = StandardScaler()
    split["X_train"] = scaler.fit_transform(split["X_train"]).astype(np.float32)
    split["X_val"] = scaler.transform(split["X_val"]).astype(np.float32)
    split["X_test"] = scaler.transform(split["X_test"]).astype(np.float32)
    split["scaler"] = scaler

    print(f"[preprocess] Features normalized (mean=0, std=1) on train set "
          f"({split['X_train'].shape[0]} samples)")
    return split


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fusion_loader import load_fused_data

    print("=" * 60)
    print("Fusion Preprocess — Self-Test")
    print("=" * 60)

    X, y, names = load_fused_data(use_cache=True)
    split = split_data(X, y, names, seed=42)
    split = normalize_features(split)

    print(f"\nTrain: {split['X_train'].shape}, labels: {dict(zip(*np.unique(split['y_train'], return_counts=True)))}")
    print(f"Val:   {split['X_val'].shape}, labels: {dict(zip(*np.unique(split['y_val'], return_counts=True)))}")
    print(f"Test:  {split['X_test'].shape}, labels: {dict(zip(*np.unique(split['y_test'], return_counts=True)))}")

    # Test binary conversion
    y_bin = to_binary_labels(y)
    print(f"\nBinary label distribution: {dict(zip(*np.unique(y_bin, return_counts=True)))}")
