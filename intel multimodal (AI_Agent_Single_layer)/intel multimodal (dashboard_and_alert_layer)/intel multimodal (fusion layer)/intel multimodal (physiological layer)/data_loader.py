"""
data_loader.py — BIDMC-PPG data loading, sliding window generation, and .pt caching.

Reads WFDB-format physiological signals (.hea/.dat) and numeric signals (*n.hea/*n.dat)
from the BIDMC-PPG dataset, applies sliding windows (10s, stride 5s), and caches
window tensors as .pt files to avoid recomputation across experiments.

Matches the logic from IntelCup2026.ipynb cells 2 and 4.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from preprocess import robust_zscore, derive_labels, to_binary_labels

# ---------------------------------------------------------------------------
# WFDB import — may fail if not installed; provide clear error
# ---------------------------------------------------------------------------
try:
    import wfdb
except ImportError:
    raise ImportError(
        "wfdb package is required. Install it with: pip install wfdb"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS_WAVE = 125          # ECG / PPG waveform sampling rate (Hz)
FS_NUM = 1             # HR / SpO2 numerics sampling rate (Hz)
WINDOW_SEC = 10        # Window duration in seconds
STRIDE_SEC = 5         # Stride between consecutive windows
WINDOW_WAVE = int(FS_WAVE * WINDOW_SEC)    # 1250 samples
STRIDE_WAVE = int(FS_WAVE * STRIDE_SEC)    # 625 samples
RECORD_DURATION_SEC = 480  # 8 minutes
EXPECTED_WAVEFORM_SAMPLES = FS_WAVE * RECORD_DURATION_SEC  # 60000


# ---------------------------------------------------------------------------
# Record discovery
# ---------------------------------------------------------------------------

def list_bidmc_records(data_dir: str) -> list:
    """
    Scan the BIDMC data directory for available waveform records.

    Filters out numeric-only records (*n.hea) and keeps base records
    that have both .hea and .dat files and their numeric counterparts.

    Args:
        data_dir: Path to the BIDMC dataset root directory.

    Returns:
        Sorted list of record names (e.g. ["bidmc01", "bidmc02", ...]).
    """
    hea_files = sorted(glob.glob(os.path.join(data_dir, "bidmc*.hea")))

    records = set()
    for hea_path in hea_files:
        basename = os.path.splitext(os.path.basename(hea_path))[0]
        # Skip numeric records (e.g. "bidmc01n")
        if basename.endswith("n"):
            continue
        # Verify both waveform and numeric files exist
        dat_path = os.path.join(data_dir, f"{basename}.dat")
        num_hea = os.path.join(data_dir, f"{basename}n.hea")
        num_dat = os.path.join(data_dir, f"{basename}n.dat")
        if os.path.exists(dat_path) and os.path.exists(num_hea) and os.path.exists(num_dat):
            records.add(basename)

    return sorted(records)


# ---------------------------------------------------------------------------
# Channel lookup
# ---------------------------------------------------------------------------

def find_channel(record, target_names: list) -> int:
    """
    Find the index of a signal channel in a wfdb Record by name.

    Performs case-insensitive exact match first, then substring match if
    exact match fails. Returns the first matching channel index.

    Args:
        record: wfdb Record object (has .sig_name attribute).
        target_names: List of candidate signal name strings (e.g. ["II", "ecg", "mlii"]).

    Returns:
        Integer channel index (0-based).

    Raises:
        ValueError: If no matching channel is found.
    """
    sig_names_lower = [s.lower().strip() for s in record.sig_name]

    for target in target_names:
        target_lower = target.lower().strip()
        # Exact match first
        for i, sn in enumerate(sig_names_lower):
            if sn == target_lower:
                return i
        # Substring match
        for i, sn in enumerate(sig_names_lower):
            if target_lower in sn:
                return i

    raise ValueError(
        f"Cannot find channel matching {target_names} in {record.sig_name}. "
        f"Available signals: {record.sig_name}"
    )


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def safe_fill_nan(arr: np.ndarray) -> np.ndarray:
    """
    Fill NaN values: forward-fill → backward-fill → median fill → zeros.

    Args:
        arr: 1-D numpy array possibly containing NaN values.

    Returns:
        1-D array with NaN values filled.
    """
    arr = np.asarray(arr, dtype=np.float64).copy()
    nan_mask = np.isnan(arr)

    if not nan_mask.any():
        return arr.astype(np.float32)

    # Forward fill
    arr = np.where(nan_mask, np.nan, arr)
    arr = pd_forward_fill_1d(arr)
    # Backward fill (reverse forward fill)
    arr = pd_forward_fill_1d(arr[::-1])[::-1]

    # Median fill for any remaining NaN
    still_nan = np.isnan(arr)
    if still_nan.any():
        median_val = np.nanmedian(arr)
        if np.isnan(median_val):
            median_val = 0.0
        arr[still_nan] = median_val

    # Zero fill as absolute fallback
    arr = np.nan_to_num(arr, nan=0.0)
    return arr.astype(np.float32)


def pd_forward_fill_1d(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values along a 1-D array."""
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    idx = np.where(~mask, np.arange(len(arr)), 0)
    idx = np.maximum.accumulate(idx)
    return arr[idx]


# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------

def upsample_numeric(num_arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Linearly interpolate a 1-Hz numeric signal to match waveform length (125 Hz).

    Args:
        num_arr: 1-D array at 1 Hz (length ~480 for 8 min).
        target_len: Target length (typically 60000 for 480 s at 125 Hz).

    Returns:
        1-D array of length target_len, upsampled via linear interpolation.
    """
    num_arr = np.asarray(num_arr, dtype=np.float64)
    orig_len = len(num_arr)
    x_old = np.linspace(0, target_len / FS_WAVE, orig_len)
    x_new = np.linspace(0, target_len / FS_WAVE, target_len)
    upsampled = np.interp(x_new, x_old, num_arr)
    return upsampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Single record loading
# ---------------------------------------------------------------------------

def load_record(rec_name: str, data_dir: str) -> dict:
    """
    Load a single BIDMC recording: waveform signals (ECG, PPG, RESP) and
    numerics (HR, SpO2).

    Args:
        rec_name: Record name (e.g. "bidmc01").
        data_dir: Path to the BIDMC dataset root.

    Returns:
        Dict with keys:
            ecg: 1-D float32 array (lead II ECG, length=N_samples)
            ppg: 1-D float32 array (PLETH/PPG)
            resp: 1-D float32 array (impedance respiration)
            hr_up: 1-D float32 array (HR upsampled to 125 Hz)
            spo2_up: 1-D float32 array (SpO2 upsampled to 125 Hz)
    """
    wave_path = os.path.join(data_dir, rec_name)
    num_path = os.path.join(data_dir, f"{rec_name}n")

    # --- Load waveform: ECG + PPG + RESP ---
    record = wfdb.rdrecord(wave_path, channels=None)

    # Find channels
    ecg_idx = find_channel(record, ["II", "ecg", "mlii"])
    ppg_idx = find_channel(record, ["PLETH", "ppg", "pleth"])
    resp_idx = find_channel(record, ["RESP", "resp", "resp_sig"])

    ecg_raw = record.p_signal[:, ecg_idx].astype(np.float32)
    ppg_raw = record.p_signal[:, ppg_idx].astype(np.float32)
    resp_raw = record.p_signal[:, resp_idx].astype(np.float32)

    # Truncate extra sample (60001 → 60000)
    if len(ecg_raw) > EXPECTED_WAVEFORM_SAMPLES:
        ecg_raw = ecg_raw[:EXPECTED_WAVEFORM_SAMPLES]
        ppg_raw = ppg_raw[:EXPECTED_WAVEFORM_SAMPLES]
        resp_raw = resp_raw[:EXPECTED_WAVEFORM_SAMPLES]

    n_samples = len(ecg_raw)

    # --- Load numerics: HR + SpO2 ---
    num_record = wfdb.rdrecord(num_path, channels=None)
    hr_idx = find_channel(num_record, ["HR", "hr", "heart rate", "Heart Rate"])
    spo2_idx = find_channel(num_record, ["SpO2", "spo2", "sp02", "sao2", "SAO2"])

    hr_raw = num_record.p_signal[:, hr_idx].astype(np.float64)
    spo2_raw = num_record.p_signal[:, spo2_idx].astype(np.float64)

    # Handle NaN in numerics
    hr_clean = safe_fill_nan(hr_raw)
    spo2_clean = safe_fill_nan(spo2_raw)

    # Upsample to 125 Hz
    hr_up = upsample_numeric(hr_clean, n_samples)
    spo2_up = upsample_numeric(spo2_clean, n_samples)

    return {
        "ecg": ecg_raw,
        "ppg": ppg_raw,
        "resp": resp_raw,
        "hr_up": hr_up,
        "spo2_up": spo2_up,
    }


# ---------------------------------------------------------------------------
# Sliding window generation
# ---------------------------------------------------------------------------

def sliding_windows(record_dict: dict) -> np.ndarray:
    """
    Generate sliding windows from a loaded record.

    Each window is 4-channel: [ecg_z, ppg_z, dECG_z, dPPG_z]
    where _z denotes robust z-score normalization and d denotes first derivative
    (gradient), also independently z-scored.

    This matches the notebook's channel layout exactly — no HR/SpO2 channels
    to avoid label leakage.

    Args:
        record_dict: Dict from load_record().

    Returns:
        Float32 numpy array of shape [N_windows, 1250, 4].
    """
    ecg = np.asarray(record_dict["ecg"], dtype=np.float32)
    ppg = np.asarray(record_dict["ppg"], dtype=np.float32)

    n_samples = min(len(ecg), len(ppg))

    # Robust z-score on full signals (per-notebook behavior)
    ecg_z = robust_zscore(ecg[:n_samples])
    ppg_z = robust_zscore(ppg[:n_samples])

    # First differences (gradients), then z-score
    decg_z = robust_zscore(np.gradient(ecg_z))
    dppg_z = robust_zscore(np.gradient(ppg_z))

    # Generate windows
    windows = []
    for start in range(0, n_samples - WINDOW_WAVE + 1, STRIDE_WAVE):
        end = start + WINDOW_WAVE
        win = np.stack([
            ecg_z[start:end],
            ppg_z[start:end],
            decg_z[start:end],
            dppg_z[start:end],
        ], axis=-1).astype(np.float32)
        windows.append(win)

    if not windows:
        return np.empty((0, WINDOW_WAVE, 4), dtype=np.float32)

    return np.stack(windows, axis=0)


# ---------------------------------------------------------------------------
# Cache building
# ---------------------------------------------------------------------------

def build_cache(data_dir: str, cache_dir: str = "cache/windows", label_mode: str = "quantile"):
    """
    Load all BIDMC records, generate sliding windows, derive labels, and
    save each record's windows + labels as .pt files.

    Cache format:
        cache/windows/{rec_name}.pt       → torch.Tensor [N_w, 1250, 4]
        cache/windows/{rec_name}_labels.pt → torch.Tensor [N_w] (int64)

    Args:
        data_dir: Path to the BIDMC dataset root.
        cache_dir: Directory to save cached .pt files.
        label_mode: "quantile" (default) or "clinical".
    """
    os.makedirs(cache_dir, exist_ok=True)

    records = list_bidmc_records(data_dir)
    print(f"Found {len(records)} BIDMC records in {data_dir}")

    # Phase 1: Load all records and generate windows
    records_data = {}
    for rec_name in tqdm(records, desc="Loading records"):
        try:
            rec_dict = load_record(rec_name, data_dir)
        except Exception as e:
            print(f"  [WARN] Skipping {rec_name}: {e}")
            continue

        windows = sliding_windows(rec_dict)
        records_data[rec_name] = {
            "windows": windows,
            "hr_up": rec_dict["hr_up"],
            "spo2_up": rec_dict["spo2_up"],
            "num_windows": windows.shape[0],
        }

    if not records_data:
        raise RuntimeError("No records loaded successfully. Check data_dir path.")

    # Phase 2: Derive global labels
    print(f"Deriving labels with mode='{label_mode}'...")
    labels_per_record = derive_labels(records_data, label_mode=label_mode)

    # Phase 3: Save to disk
    total_windows = 0
    for rec_name, data in tqdm(records_data.items(), desc="Saving cache"):
        windows_tensor = torch.from_numpy(data["windows"])
        labels_array = labels_per_record[rec_name]
        labels_tensor = torch.from_numpy(np.asarray(labels_array, dtype=np.int64))

        torch.save(windows_tensor, os.path.join(cache_dir, f"{rec_name}.pt"))
        torch.save(labels_tensor, os.path.join(cache_dir, f"{rec_name}_labels.pt"))
        total_windows += data["num_windows"]

    # Save metadata
    meta = {
        "records": list(records_data.keys()),
        "label_mode": label_mode,
        "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC,
        "window_samples": WINDOW_WAVE,
        "num_channels": 4,
        "channel_names": ["ecg_z", "ppg_z", "dECG_z", "dPPG_z"],
        "total_windows": total_windows,
    }
    torch.save(meta, os.path.join(cache_dir, "cache_meta.pt"))

    # Count class distribution
    all_labels = np.concatenate([labels_per_record[r] for r in meta["records"]])
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f"\nCache built successfully: {total_windows} total windows")
    print(f"Label distribution: {dict(zip(unique.astype(int), counts))}")
    print(f"Cache saved to: {os.path.abspath(cache_dir)}")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BIDMCDataset(Dataset):
    """
    PyTorch Dataset that loads pre-computed window tensors from .pt cache files.

    Args:
        cache_dir: Directory containing cached .pt files.
        record_ids: List of record names to include (e.g. training split).
        binary: If True, convert labels to binary (merge classes 1+2 → 1).
    """

    def __init__(self, cache_dir: str, record_ids: list, binary: bool = False):
        self.cache_dir = cache_dir
        self.record_ids = list(record_ids)
        self.binary = binary

        self.windows = []
        self.labels = []
        self.sample_records = []  # record name for each sample index

        for rec_name in self.record_ids:
            win_path = os.path.join(cache_dir, f"{rec_name}.pt")
            lbl_path = os.path.join(cache_dir, f"{rec_name}_labels.pt")

            if not os.path.exists(win_path) or not os.path.exists(lbl_path):
                print(f"  [WARN] Missing cache for {rec_name}, skipping")
                continue

            w = torch.load(win_path, map_location="cpu", weights_only=True)
            l = torch.load(lbl_path, map_location="cpu", weights_only=True)

            # Convert to binary labels if needed
            if self.binary:
                l = torch.from_numpy(to_binary_labels(l.numpy()))

            n_windows = w.shape[0]
            self.windows.append(w)
            self.labels.append(l)
            self.sample_records.extend([rec_name] * n_windows)

        if self.windows:
            self.windows = torch.cat(self.windows, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
        else:
            self.windows = torch.empty((0, WINDOW_WAVE, 4), dtype=torch.float32)
            self.labels = torch.empty((0,), dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        return self.windows[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Record-level train/val/test split
# ---------------------------------------------------------------------------

def split_records(records: list, train_ratio: float = 0.70, val_ratio: float = 0.15,
                  seed: int = 42) -> tuple:
    """
    Split record IDs into train/val/test sets at the record level
    (no data leakage across splits).

    Args:
        records: List of record name strings.
        train_ratio: Fraction of records for training.
        val_ratio: Fraction of records for validation.
        seed: Random seed for reproducible split.

    Returns:
        Tuple of (train_ids, val_ids, test_ids).
    """
    np.random.seed(seed)
    shuffled = np.random.permutation(records)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = sorted(shuffled[:n_train].tolist())
    val_ids = sorted(shuffled[n_train:n_train + n_val].tolist())
    test_ids = sorted(shuffled[n_train + n_val:].tolist())

    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Main: build cache when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BIDMC-PPG window cache")
    parser.add_argument("--data_dir", type=str,
                        default="bidmc-ppg-and-respiration-dataset-1.0.0",
                        help="Path to BIDMC dataset root")
    parser.add_argument("--cache_dir", type=str, default="cache/windows",
                        help="Output directory for .pt cache files")
    parser.add_argument("--label_mode", type=str, default="quantile",
                        choices=["quantile", "clinical"],
                        help="Label derivation mode")
    args = parser.parse_args()

    build_cache(args.data_dir, args.cache_dir, args.label_mode)
