"""
preprocess.py — Normalization, signal filtering, HR/RR feature extraction,
and label derivation for the BIDMC-PPG Physiological Layer.

Matches the logic from IntelCup2026.ipynb cells 2 and 4.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def robust_zscore(x: np.ndarray) -> np.ndarray:
    """
    Robust standardization using median and Median Absolute Deviation (MAD).

    This is the exact same function used throughout the notebook.
    Outlier-resistant: outliers don't dominate mean/std.

    Args:
        x: Input 1-D signal (or flattened array).

    Returns:
        Standardized array of same shape as x.
    """
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / (1.4826 * mad)


# ---------------------------------------------------------------------------
# Bandpass filter
# ---------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    fs: float = 125.0,
    lowcut: float = 0.1,
    highcut: float = 0.4,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter using second-order sections (SOS)
    for numerical stability.

    Typical use: filter respiration signal to 0.1–0.4 Hz (6–24 breaths/min).

    Args:
        signal: 1-D input signal.
        fs: Sampling frequency in Hz (default 125 for BIDMC waveforms).
        lowcut: Low cutoff frequency in Hz.
        highcut: High cutoff frequency in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered signal (same length as input).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, signal)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extraction: HR & RR
# ---------------------------------------------------------------------------

def compute_hr_features(hr_signal: np.ndarray) -> dict:
    """
    Extract scalar heart-rate features from the 1-Hz HR numerics signal.

    Args:
        hr_signal: 1-D HR signal sampled at 1 Hz (length = 480 for 8-min recording).

    Returns:
        Dict with keys: hr_mean, hr_std, hr_min, hr_max.
    """
    hr = np.asarray(hr_signal, dtype=np.float32)
    hr = hr[~np.isnan(hr)]  # drop NaN samples
    if len(hr) == 0:
        return {"hr_mean": 0.0, "hr_std": 0.0, "hr_min": 0.0, "hr_max": 0.0}
    return {
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "hr_min": float(np.min(hr)),
        "hr_max": float(np.max(hr)),
    }


def compute_rr_features(resp_signal: np.ndarray, fs: float = 125.0) -> dict:
    """
    Estimate respiratory rate features from impedance respiration signal.

    Uses simple peak detection on the bandpass-filtered signal.

    Args:
        resp_signal: 1-D respiration signal at 125 Hz.
        fs: Sampling frequency in Hz.

    Returns:
        Dict with keys: rr_mean_breaths_per_min, rr_std.
    """
    resp = np.asarray(resp_signal, dtype=np.float64)

    # Bandpass 0.1–0.4 Hz (typical adult respiratory rate: 6–24 breaths/min)
    filtered = bandpass_filter(resp, fs=fs, lowcut=0.1, highcut=0.4, order=4)

    # Simple peak detection
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(filtered, distance=int(fs * 1.5), prominence=0.1)

    if len(peaks) < 2:
        return {"rr_mean_breaths_per_min": 0.0, "rr_std": 0.0, "num_breaths": 0}

    # Inter-breath intervals in seconds
    ibi = np.diff(peaks) / fs
    rr_instantaneous = 60.0 / ibi  # breaths per minute

    return {
        "rr_mean_breaths_per_min": float(np.mean(rr_instantaneous)),
        "rr_std": float(np.std(rr_instantaneous)),
        "num_breaths": int(len(peaks)),
    }


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def clinical_label(mean_hr: float, mean_spo2: float) -> int:
    """
    Assign health label based on fixed clinical thresholds.

    From notebook cell 2:
      - unhealthy (2): SpO2 < 90  or  HR < 50  or  HR > 110
      - semi_healthy (1): SpO2 < 95  or  HR < 60  or  HR > 100
      - healthy (0): otherwise

    Args:
        mean_hr: Mean heart rate (bpm) over the recording.
        mean_spo2: Mean SpO2 (%) over the recording.

    Returns:
        Integer label: 0 = healthy, 1 = semi_healthy, 2 = unhealthy.
    """
    if mean_spo2 < 90 or mean_hr < 50 or mean_hr > 110:
        return 2  # unhealthy
    elif mean_spo2 < 95 or mean_hr < 60 or mean_hr > 100:
        return 1  # semi_healthy
    else:
        return 0  # healthy


def risk_score_from_hr_spo2(mean_hr: float, mean_spo2: float) -> float:
    """
    Compute a continuous risk score from HR and SpO2.
    Larger values indicate worse health.

    From notebook cell 2:
      hr_dev  = max(0, (60-hr)/20) if hr < 60 else max(0, (hr-100)/20)
      spo2_dev = max(0, (95-spo2)/5) if spo2 < 95 else 0
      risk = hr_dev + spo2_dev

    Args:
        mean_hr: Mean heart rate.
        mean_spo2: Mean SpO2.

    Returns:
        Risk score (float, >= 0).
    """
    hr_dev = 0.0
    if mean_hr < 60:
        hr_dev = max(0.0, (60.0 - mean_hr) / 20.0)
    elif mean_hr > 100:
        hr_dev = max(0.0, (mean_hr - 100.0) / 20.0)

    spo2_dev = 0.0
    if mean_spo2 < 95:
        spo2_dev = max(0.0, (95.0 - mean_spo2) / 5.0)

    return hr_dev + spo2_dev


def quantile_label(risk_scores: np.ndarray) -> np.ndarray:
    """
    Assign labels based on quantile thresholds of risk scores.

    Splits at 1/3 and 2/3 quantiles for balanced classes:
      - risk <= q1/3  → 0 (healthy)
      - risk <= q2/3  → 1 (semi_healthy)
      - risk >  q2/3  → 2 (unhealthy)

    Args:
        risk_scores: 1-D array of risk scores (one per record/window).

    Returns:
        1-D integer array of labels (0, 1, 2).
    """
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    q1 = np.quantile(risk_scores, 1.0 / 3.0)
    q2 = np.quantile(risk_scores, 2.0 / 3.0)

    labels = np.zeros(len(risk_scores), dtype=np.int64)
    labels[risk_scores > q1] = 1
    labels[risk_scores > q2] = 2
    return labels


def derive_labels(
    records_data: dict,
    label_mode: str = "quantile",
) -> np.ndarray:
    """
    Derive per-window labels for all records.

    When label_mode="quantile", risk scores are computed per record, then
    global quantile thresholds are applied across all records so class
    distribution is balanced across the entire dataset.

    When label_mode="clinical", fixed clinical thresholds are applied.

    Args:
        records_data: Dict mapping record_name -> dict with keys:
            hr_up, spo2_up (1-D arrays at 125 Hz, length = record_samples)
            num_windows (int, number of windows in this record)
        label_mode: "quantile" or "clinical".

    Returns:
        Dict mapping record_name -> 1-D numpy array of integer labels
        (length = num_windows for that record).
    """
    # Step 1: compute per-record mean HR / SpO2
    record_stats = {}
    for rec_name, data in records_data.items():
        hr_up = np.asarray(data["hr_up"], dtype=np.float32)
        spo2_up = np.asarray(data["spo2_up"], dtype=np.float32)
        mean_hr = float(np.nanmean(hr_up)) if len(hr_up) > 0 else 80.0
        mean_spo2 = float(np.nanmean(spo2_up)) if len(spo2_up) > 0 else 95.0
        record_stats[rec_name] = {
            "mean_hr": mean_hr,
            "mean_spo2": mean_spo2,
            "num_windows": data["num_windows"],
        }

    # Step 2: assign labels
    if label_mode == "clinical":
        labels_per_record = {}
        for rec_name, stats in record_stats.items():
            label = clinical_label(stats["mean_hr"], stats["mean_spo2"])
            labels_per_record[rec_name] = np.full(stats["num_windows"], label, dtype=np.int64)
        return labels_per_record

    elif label_mode == "quantile":
        # Compute risk score for each record
        rec_names = list(record_stats.keys())
        risks = np.array([
            risk_score_from_hr_spo2(
                record_stats[r]["mean_hr"],
                record_stats[r]["mean_spo2"],
            )
            for r in rec_names
        ])

        # Global quantile thresholds
        labels_per_record_idx = quantile_label(risks)

        labels_per_record = {}
        for i, rec_name in enumerate(rec_names):
            labels_per_record[rec_name] = np.full(
                record_stats[rec_name]["num_windows"],
                labels_per_record_idx[i],
                dtype=np.int64,
            )
        return labels_per_record

    else:
        raise ValueError(f"Unknown label_mode: {label_mode}. Use 'quantile' or 'clinical'.")


def to_binary_labels(labels_3class: np.ndarray) -> np.ndarray:
    """
    Convert 3-class labels to binary by merging classes 1+2 → 1.

    Args:
        labels_3class: Array of integer labels {0, 1, 2}.

    Returns:
        Array of binary labels {0, 1} as int64.
    """
    labels = np.asarray(labels_3class, dtype=np.int64)
    binary = np.where(labels > 0, 1, 0).astype(np.int64)
    return binary


# ---------------------------------------------------------------------------
# ID ↔ Label mappings
# ---------------------------------------------------------------------------

ID2LABEL_3CLASS = {0: "healthy", 1: "semi_healthy", 2: "unhealthy"}
LABEL2ID_3CLASS = {v: k for k, v in ID2LABEL_3CLASS.items()}

ID2LABEL_BINARY = {0: "healthy", 1: "symptomatic_or_unhealthy"}
LABEL2ID_BINARY = {v: k for k, v in ID2LABEL_BINARY.items()}
