# =====================================================================
# Data Loading & Preprocessing
#
# Handles:
#   - Scanning the audio directory for .webm, .wav, and .ogg files
#   - Cross-referencing with metadata_compiled.csv
#   - Stratified train/val/test split
#   - LocalCoughvidDataset with Mel-spectrogram transform
#   - DataLoader creation
# =====================================================================

import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from . import config as cfg

# PyAV is used as the primary audio loader because torchaudio 2.6+
# requires ffmpeg on PATH for .webm/.mp4 containers. PyAV bundles ffmpeg.
try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    print("[WARN] PyAV not installed. Falling back to torchaudio (may fail for .webm).")


# ---------------------------------------------------------------------
# Utility: pad or truncate a Mel spectrogram to a fixed time length
# ---------------------------------------------------------------------
def pad_or_truncate(mel: torch.Tensor, max_len: int) -> torch.Tensor:
    """Ensure all Mel spectrograms match the exact temporal dimension."""
    if mel.size(1) > max_len:
        return mel[:, :max_len]
    elif mel.size(1) < max_len:
        pad_amount = max_len - mel.size(1)
        return torch.nn.functional.pad(mel, (0, pad_amount))
    return mel


# ---------------------------------------------------------------------
# Disk scanner: discover audio files & cross-reference with metadata
# ---------------------------------------------------------------------
def scan_audio_files(
    audio_dir: str = None,
    extensions: tuple = None,
    status_to_label: dict = None,
    csv_path: str = None,
):
    """
    Scan the audio directory for matching files, cross-reference with
    the metadata CSV, and return file paths + integer labels.

    Returns
    -------
    file_paths : list of str
        Absolute paths to audio files that have a label.
    labels : list of int
        Integer class labels (0, 1, or 2).
    """
    if audio_dir is None:
        audio_dir = cfg.DATA_DIR
    if extensions is None:
        extensions = cfg.AUDIO_EXTENSIONS
    if status_to_label is None:
        status_to_label = cfg.STATUS_TO_LABEL
    if csv_path is None:
        csv_path = cfg.CSV_PATH

    # --- Validate paths -------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at: {csv_path}")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found at: {audio_dir}")

    print(f"Metadata CSV : {csv_path}")
    print(f"Audio dir    : {audio_dir}")

    # --- Scan disk for audio files (all supported extensions) ------------
    local_files = os.listdir(audio_dir)

    # Build {uuid: extension} mapping; prefer .wav > .webm > .ogg on duplicates
    uuid_to_ext: dict[str, str] = {}
    ext_priority = {".wav": 3, ".webm": 2, ".ogg": 1}

    for f in local_files:
        uuid_part, ext = os.path.splitext(f)
        ext_lower = ext.lower()
        if ext_lower in extensions:
            current_priority = ext_priority.get(ext_lower, 0)
            existing_priority = ext_priority.get(uuid_to_ext.get(uuid_part, ""), 0)
            if uuid_part not in uuid_to_ext or current_priority > existing_priority:
                uuid_to_ext[uuid_part] = ext_lower

    existing_uuids = set(uuid_to_ext.keys())
    print(f"Found {len(existing_uuids)} unique audio files on disk "
          f"(across {extensions}).")

    # --- Load and filter metadata ----------------------------------------
    print(f"Parsing metadata from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Drop rows missing a diagnostic status
    df = df.dropna(subset=["status"])
    df["label"] = df["status"].map(status_to_label)
    df = df.dropna(subset=["label"])

    # Cross-reference: keep only rows whose UUID exists on disk
    df["uuid_str"] = df["uuid"].astype(str)
    df_filtered = df[df["uuid_str"].isin(existing_uuids)].copy()

    if len(df_filtered) == 0:
        raise RuntimeError(
            "Zero matching audio files found! Verify your audio directory.\n"
            f"  Sample CSV uuid: {df['uuid'].iloc[0]}\n"
            f"  Sample disk file: {local_files[0] if local_files else 'None'}"
        )

    # --- Build file paths with the correct extension for each UUID --------
    file_paths = [
        os.path.join(audio_dir, f"{uid}{uuid_to_ext[uid]}")
        for uid in df_filtered["uuid_str"]
    ]
    labels = df_filtered["label"].values.astype(int)

    # Print class distribution
    vc = pd.Series(labels).value_counts().sort_index()
    items = [(k, vc[k]) for k in sorted(vc.index)]
    dist_str = ", ".join(f"{cfg.ID2LABEL.get(k, k)}={v}" for k, v in items)
    print(f"Matched {len(file_paths)} labeled audio files ({dist_str})")

    return file_paths, labels


# ---------------------------------------------------------------------
# PyTorch Dataset (with lazy spectrogram cache)
# ---------------------------------------------------------------------
class LocalCoughvidDataset(Dataset):
    """
    Lazy-loading dataset with optional spectrogram cache.

    - First epoch: decodes audio via PyAV, computes Mel spectrogram,
      normalizes, and saves to cache_dir for future epochs.
    - Subsequent epochs: loads pre-computed .pt files (~0.01s vs ~2s).

    Returns (mel_spectrogram, label, filename) for each sample.
    """

    def __init__(self, file_paths, labels, cache_dir=None):
        self.file_paths = list(file_paths)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.cache_dir = cache_dir
        self._cache_hits = 0
        self._cache_misses = 0

        # Mel transform (only used on cache misses)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=cfg.TARGET_SR,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS,
        )

    def __len__(self):
        return len(self.labels)

    # -----------------------------------------------------------------
    # Cache path helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _cache_key(filepath: str) -> str:
        """Derive a cache filename from the audio file path."""
        base = os.path.splitext(os.path.basename(filepath))[0]
        return f"{base}_mel.pt"

    def _cache_path(self, idx: int):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, self._cache_key(self.file_paths[idx]))

    # -----------------------------------------------------------------
    # Audio → Mel computation (the expensive part)
    # -----------------------------------------------------------------
    def _compute_mel(self, path: str) -> torch.Tensor:
        """Decode audio, convert to mono, resample, compute Mel, normalize."""
        if HAS_PYAV:
            try:
                container = av.open(path)
                audio_stream = container.streams.audio[0]
                sr = audio_stream.sample_rate or cfg.TARGET_SR
                frames = container.decode(audio=0)
                chunks = [frame.to_ndarray() for frame in frames]
                container.close()
                if not chunks:
                    raise RuntimeError(f"No audio frames decoded from {path}")
                audio = np.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]
                if audio.shape[0] > 1:
                    audio = audio.mean(axis=0, keepdims=True)
                waveform = torch.from_numpy(audio.copy()).float()
                if sr != cfg.TARGET_SR:
                    resampler = T.Resample(orig_freq=sr, new_freq=cfg.TARGET_SR)
                    waveform = resampler(waveform)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load {path} via PyAV.\n"
                    f"Original error: {type(e).__name__}: {e}"
                )
        else:
            try:
                waveform, sr = torchaudio.load(path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load {path}. Install PyAV (`pip install av`) "
                    f"for .webm/.mp4 support.\nOriginal error: {e}"
                )
            if sr != cfg.TARGET_SR:
                resampler = T.Resample(orig_freq=sr, new_freq=cfg.TARGET_SR)
                waveform = resampler(waveform)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel = self.mel_transform(waveform).squeeze(0)         # [n_mels, time]
        mel = pad_or_truncate(mel, cfg.MAX_FRAMES)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel

    # -----------------------------------------------------------------
    # Main accessor
    # -----------------------------------------------------------------
    def __getitem__(self, idx):
        label = self.labels[idx]
        path = self.file_paths[idx]
        filename = os.path.basename(path)

        # --- Try cache first ---
        cache_path = self._cache_path(idx)
        if cache_path and os.path.exists(cache_path):
            self._cache_hits += 1
            mel = torch.load(cache_path, weights_only=False)
            return mel, label, filename

        # --- Cache miss: compute from audio ---
        self._cache_misses += 1
        mel = self._compute_mel(path)

        # --- Save to cache for next epoch ---
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(mel, cache_path)

        return mel, label, filename

    @property
    def cache_stats(self) -> tuple:
        return self._cache_hits, self._cache_misses


# ---------------------------------------------------------------------
# Collate function (must be module-level for multiprocessing on Windows)
# ---------------------------------------------------------------------
def _collate_fn(batch):
    """Stack mels, labels, and collect filenames into a batch."""
    mels = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    filenames = [item[2] for item in batch]
    return mels, labels, filenames


# ---------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------
def create_dataloaders(
    file_paths,
    labels,
    batch_size: int = None,
    seed: int = None,
    test_split: float = None,
    val_split: float = None,
    cache_dir: str = None,
    num_workers: int = None,
):
    """
    Perform a stratified train/val/test split and return DataLoaders
    plus the raw split arrays (needed by export_predictions).

    Args:
        cache_dir: if set, datasets will lazily cache Mel spectrograms here
        num_workers: DataLoader workers (0 for cache-building, 4 for cached)

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    split_data : tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    if seed is None:
        seed = cfg.SEED
    if test_split is None:
        test_split = cfg.TEST_SPLIT
    if val_split is None:
        val_split = cfg.VAL_SPLIT
    if cache_dir is None:
        cache_dir = cfg.CACHE_DIR
    if num_workers is None:
        num_workers = cfg.NUM_WORKERS

    # Stratified split: train / (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        file_paths, labels,
        test_size=test_split,
        random_state=seed,
        stratify=labels,
    )

    # Split the remainder into val / test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_split,
        random_state=seed,
        stratify=y_temp,
    )

    print("\n--- Train / Val / Test Split ---")
    print(f"Train : {len(X_train)} samples")
    print(f"Val   : {len(X_val)} samples")
    print(f"Test  : {len(X_test)} samples")

    # Count pre-cached files
    if cache_dir and os.path.isdir(cache_dir):
        cached = sum(1 for f in os.listdir(cache_dir) if f.endswith("_mel.pt"))
        total = len(X_train) + len(X_val) + len(X_test)
        pct = 100 * cached / max(total, 1)
        print(f"Cache : {cached}/{total} files pre-cached ({pct:.0f}%)")
        # Use workers for fast loading from cache
        actual_workers = num_workers if cached > total * 0.5 else 0
    else:
        actual_workers = 0
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Cache : building during epoch 1 (dir: {cache_dir})")

    print(f"Workers: {actual_workers}")

    # Build datasets with cache support
    train_ds = LocalCoughvidDataset(X_train, y_train, cache_dir=cache_dir)
    val_ds = LocalCoughvidDataset(X_val, y_val, cache_dir=cache_dir)
    test_ds = LocalCoughvidDataset(X_test, y_test, cache_dir=cache_dir)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=actual_workers, collate_fn=_collate_fn,
        pin_memory=True if cfg.DEVICE == "cuda" else False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=actual_workers, collate_fn=_collate_fn,
        pin_memory=True if cfg.DEVICE == "cuda" else False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=actual_workers, collate_fn=_collate_fn,
        pin_memory=True if cfg.DEVICE == "cuda" else False,
    )

    split_data = (X_train, X_val, X_test, y_train, y_val, y_test)
    return train_loader, val_loader, test_loader, split_data
