# =====================================================================
# Audio Layer Package — COUGHVID-v3 AST Classifier
# =====================================================================

from .config import (
    set_seeds,
    ensure_output_dir,
    ensure_cache_dir,
    set_output_dir,
    apply_overrides,
    DEVICE,
    OUTPUT_DIR,
    CHECKPOINT_PATH,
    PREDICTIONS_CSV_PATH,
    EXPERIMENT_CSV_PATH,
    CONFUSION_MATRIX_PNG,
    TRAINING_CURVES_PNG,
)

from .dataset import (
    scan_audio_files,
    LocalCoughvidDataset,
    create_dataloaders,
    pad_or_truncate,
)

from .model import AudioSpectrogramTransformer

from .train import run_epoch, train_model

from .evaluate import evaluate_test_set, evaluate_model

from .export_predictions import export_predictions
