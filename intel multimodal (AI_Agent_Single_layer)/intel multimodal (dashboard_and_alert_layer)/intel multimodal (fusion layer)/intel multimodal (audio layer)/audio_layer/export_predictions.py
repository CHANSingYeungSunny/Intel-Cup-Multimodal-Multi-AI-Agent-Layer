# =====================================================================
# Predictions Export
#
# Generates predictions.csv with columns:
#   filename       — base name of the audio file
#   prediction     — predicted class (0=healthy, 1=symptomatic, 2=covid_19)
#   label          — ground truth class
#   feature_vector — CLS token embedding (space-separated floats)
# =====================================================================

import json
import os
import numpy as np
import pandas as pd

from . import config as cfg


def export_predictions(
    y_true: list,
    y_pred: list,
    cls_embeddings: np.ndarray,
    filenames: list,
    output_path: str = None,
):
    """
    Write predictions.csv from the test-set evaluation results.

    Args:
        y_true:         ground truth labels (list of int)
        y_pred:         predicted labels (list of int)
        cls_embeddings: CLS token embeddings, shape (N, d_model)
        filenames:      audio file base names (list of str)
        output_path:    path to write predictions.csv
    """
    if output_path is None:
        output_path = cfg.PREDICTIONS_CSV_PATH

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Serialize each CLS embedding as a JSON array (matches vision layer format)
    feature_strings = [
        json.dumps(cls_embeddings[i].tolist())
        for i in range(len(cls_embeddings))
    ]

    df = pd.DataFrame({
        "filename": filenames,
        "prediction": y_pred,
        "label": y_true,
        "feature_vector": feature_strings,
    })

    df.to_csv(output_path, index=False)
    print(f"\nPredictions exported to: {output_path}")
    print(f"  Rows       : {len(df)}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Embed dim  : {cls_embeddings.shape[1]}")

    # Quick sanity check
    acc = (df["prediction"] == df["label"]).mean()
    print(f"  Accuracy   : {acc:.4f}")
