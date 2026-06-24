# =====================================================================
# Centralized Configuration for Audio Layer AST Pipeline
# All constants, paths, seeds, and hyperparameters live here.
# Every other module imports from this single source of truth.
# =====================================================================

import os
import random
import numpy as np
import torch

# ---------------------------------------------------------------------
# Project Root & Paths (adapted to local COUGHVID-v3 dataset structure)
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Audio files AND metadata both live in this directory
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "public_dataset_v3", "coughvid_20211012")
CSV_PATH = os.path.join(DATA_DIR, "metadata_compiled.csv")

# Spectrogram cache (lazy-built during first training epoch)
CACHE_DIR = os.path.join(DATA_DIR, "mel_cache")

# All outputs go into a dedicated outputs/ folder
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_ast_coughvid_local.pt")
PREDICTIONS_CSV_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")
EXPERIMENT_CSV_PATH = os.path.join(OUTPUT_DIR, "experiment_results_with_accuracy.csv")
CONFUSION_MATRIX_PNG = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
TRAINING_CURVES_PNG = os.path.join(OUTPUT_DIR, "training_curves.png")

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# Audio / Spectrogram Hyperparameters
# ---------------------------------------------------------------------
TARGET_SR = 16000          # Resample all audio to 16 kHz
N_MELS = 128               # Number of Mel filterbank bins
N_FFT = 1024               # FFT window size
HOP_LENGTH = 156           # 156 samples @ 16kHz → ~2.5 s window
MAX_FRAMES = 192           # Aggressive crop: ~1.9 s (cough burst + tight context)

# ---------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------
PATCH_SIZE = (16, 16)      # Non-overlapping 2D patches over (mel, time)
D_MODEL = 128              # Transformer embedding dimension
N_HEADS = 4                # Multi-head attention heads
N_LAYERS = 3               # Transformer encoder layers
D_FF = 256                 # Feed-forward hidden dimension
DROPOUT = 0.1              # Dropout rate

# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------
NUM_CLASSES = 3            # healthy=0, symptomatic=1, covid_19=2
STATUS_TO_LABEL = {
    "healthy": 0,
    "symptomatic": 1,
    "COVID-19": 2,
}
ID2LABEL = {0: "healthy", 1: "symptomatic", 2: "covid_19"}

# ---------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------
BATCH_SIZE = 32              # Mixed precision (AMP) keeps VRAM usage low
EPOCHS = 10                 # Max epochs — hard cap for ablation throughput
LR = 3e-4                   # Baseline learning rate
EARLY_STOP_PATIENCE = 3     # Stop if no val_acc improvement for N epochs
USE_AMP = True              # Automatic Mixed Precision — saves ~40% VRAM
GRADIENT_CHECKPOINTING = True  # Trades compute for memory (enables larger batches)
NUM_WORKERS = 4             # DataLoader workers (used after cache is built)
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# ---------------------------------------------------------------------
# Data Split Ratios
# ---------------------------------------------------------------------
TEST_SPLIT = 0.30          # Fraction held out for test+val
VAL_SPLIT = 0.50           # Fraction of held-out used for validation

# ---------------------------------------------------------------------
# Supported Audio File Extensions (scan all three)
# ---------------------------------------------------------------------
AUDIO_EXTENSIONS = (".webm", ".wav", ".ogg")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def set_seeds():
    """Apply all random seeds for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_output_dir():
    """Create the outputs directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def ensure_cache_dir():
    """Create the spectrogram cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)

def set_output_dir(path: str):
    """Redirect all output paths to a subdirectory (used by ablation runner)."""
    global OUTPUT_DIR, CHECKPOINT_PATH, PREDICTIONS_CSV_PATH
    global EXPERIMENT_CSV_PATH, CONFUSION_MATRIX_PNG, TRAINING_CURVES_PNG
    OUTPUT_DIR = path
    CHECKPOINT_PATH = os.path.join(path, "best_ast_coughvid_local.pt")
    PREDICTIONS_CSV_PATH = os.path.join(path, "predictions.csv")
    EXPERIMENT_CSV_PATH = os.path.join(path, "experiment_results_with_accuracy.csv")
    CONFUSION_MATRIX_PNG = os.path.join(path, "confusion_matrix.png")
    TRAINING_CURVES_PNG = os.path.join(path, "training_curves.png")

def apply_overrides(**kwargs):
    """Apply experiment-specific config overrides (for ablation studies)."""
    for key, value in kwargs.items():
        if key in globals():
            globals()[key] = value
        else:
            print(f"[WARN] Unknown config key: {key}")
    # If OUTPUT_DIR was changed, recompute dependent paths
    if "OUTPUT_DIR" in kwargs:
        set_output_dir(kwargs["OUTPUT_DIR"])
