"""
DataStore: loads and caches Fusion Layer output CSVs.

predictions.csv columns: filename, prediction, label, feature_vector (256-dim JSON array)
experiment_results_with_accuracy.csv columns: 33 columns including metrics, confusion matrix,
    and training curves (JSON arrays).
"""
import json
import re
import pandas as pd
import numpy as np
from config import PREDICTIONS_CSV, EXPERIMENT_CSV, LABEL_NAMES


class DataStore:
    """Singleton-like cache for Fusion Layer outputs."""

    def __init__(self):
        self._predictions_df = None
        self._experiments_df = None
        self._feature_matrix = None        # shape (N, 256)
        self._feature_vectors = None       # list of lists
        self._subjects = None              # extracted subject IDs
        self._active_experiment_id = 1
        self._num_classes = 3

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_all(self):
        """Load both CSVs into memory. Call once at startup."""
        self._predictions_df = pd.read_csv(PREDICTIONS_CSV)
        self._experiments_df = pd.read_csv(EXPERIMENT_CSV)
        self._parse_features()
        self._parse_subjects()
        print(f"[DataStore] Loaded {len(self._predictions_df)} predictions, "
              f"{len(self._experiments_df)} experiments")

    def _parse_features(self):
        """Deserialize feature_vector JSON strings into numeric matrix."""
        vectors = []
        for fv_str in self._predictions_df["feature_vector"]:
            vec = json.loads(fv_str)
            vectors.append(vec)
        self._feature_vectors = vectors
        self._feature_matrix = np.array(vectors, dtype=np.float32)

    def _parse_subjects(self):
        """Extract vision subject IDs from composite filenames."""
        subjects = []
        for fname in self._predictions_df["filename"]:
            # Format: v:UBFC2/subject14/win_000292|a:...|p:...
            match = re.search(r"v:([^|]+)", str(fname))
            if match:
                # Extract subject name: UBFC2/subject14/win_000292 -> subject14
                path = match.group(1)
                subj_match = re.search(r"subject(\d+)", path)
                if subj_match:
                    subjects.append(f"subject{subj_match.group(1)}")
                else:
                    subjects.append(path.split("/")[-2] if "/" in path else path)
            else:
                subjects.append("unknown")
        self._subjects = subjects

    # ------------------------------------------------------------------
    # Predictions access
    # ------------------------------------------------------------------
    def get_predictions_df(self):
        return self._predictions_df

    def get_prediction_count(self):
        return len(self._predictions_df)

    def get_predictions_list(self):
        """Return list of dicts for API consumption."""
        records = []
        for _, row in self._predictions_df.iterrows():
            records.append({
                "filename": row["filename"],
                "prediction": int(row["prediction"]),
                "label": int(row["label"]),
                "prediction_name": LABEL_NAMES.get(int(row["prediction"]), "Unknown"),
                "label_name": LABEL_NAMES.get(int(row["label"]), "Unknown"),
                "correct": int(row["prediction"]) == int(row["label"]),
            })
        return records

    def get_counts(self):
        """Return per-class counts and percentages (always shows all 3 classes
        since predictions.csv is always 3-class)."""
        preds = self._predictions_df["prediction"].value_counts().to_dict()
        labels = self._predictions_df["label"].value_counts().to_dict()
        total = len(self._predictions_df)

        # Always show all 3 classes for health state display
        pred_counts = {LABEL_NAMES.get(k, str(k)): preds.get(k, 0) for k in range(3)}
        label_counts = {LABEL_NAMES.get(k, str(k)): labels.get(k, 0) for k in range(3)}
        pred_pct = {k: round(v / total * 100, 1) for k, v in pred_counts.items()}
        label_pct = {k: round(v / total * 100, 1) for k, v in label_counts.items()}

        mode_val = self._predictions_df["prediction"].mode()
        current_state = LABEL_NAMES.get(int(mode_val.iloc[0]), "Unknown") if len(mode_val) > 0 else "Unknown"

        return {
            "total_samples": total,
            "prediction_counts": pred_counts,
            "label_counts": label_counts,
            "prediction_percentages": pred_pct,
            "label_percentages": label_pct,
            "current_state": current_state,
            "num_classes": self._num_classes,
            "label_mode": "3class" if self._num_classes == 3 else "binary",
        }

    # ------------------------------------------------------------------
    # Feature access
    # ------------------------------------------------------------------
    def get_feature_matrix(self):
        return self._feature_matrix

    def get_feature_vector(self, index):
        return self._feature_vectors[index]

    # ------------------------------------------------------------------
    # Subject access
    # ------------------------------------------------------------------
    def get_subjects(self):
        return self._subjects

    def get_unique_subjects(self):
        return sorted(set(self._subjects))

    def get_predictions_for_subject(self, subject_id):
        """Return all prediction rows for a given subject, sorted by window."""
        mask = [s == subject_id for s in self._subjects]
        subset = self._predictions_df[mask].copy()
        # Sort by window number extracted from filename
        subset["_win"] = subset["filename"].apply(self._extract_win_number)
        subset = subset.sort_values("_win")
        return subset

    def get_feature_vectors_for_subject(self, subject_id):
        """Return feature vectors and metadata for one subject."""
        subset = self.get_predictions_for_subject(subject_id)
        indices = subset.index.tolist()
        vectors = self._feature_matrix[indices]
        return {
            "subject_id": subject_id,
            "windows": subset["_win"].tolist() if "_win" in subset.columns else list(range(len(subset))),
            "predictions": subset["prediction"].tolist(),
            "labels": subset["label"].tolist(),
            "feature_vectors": vectors.tolist(),
        }

    @staticmethod
    def _extract_win_number(filename):
        match = re.search(r"win_(\d+)", str(filename))
        return int(match.group(1)) if match else 0

    # ------------------------------------------------------------------
    # Experiment access
    # ------------------------------------------------------------------
    def get_experiments_df(self):
        return self._experiments_df

    def get_all_experiments(self):
        """List all experiments with summary fields, sorted by exp_id."""
        exps = []
        for _, row in self._experiments_df.iterrows():
            exps.append({
                "id": int(row["exp_id"]),
                "label": row["config_label"],
                "mode": row.get("label_mode", "3class"),
                "accuracy": round(float(row["test_accuracy"]) * 100, 2),
                "f1_macro": round(float(row["test_f1_macro"]) * 100, 2),
                "f1_weighted": round(float(row["test_f1_weighted"]) * 100, 2),
                "best_epoch": int(row["best_epoch"]),
            })
        exps.sort(key=lambda e: e["id"])
        return exps

    def get_experiment(self, exp_id):
        """Get full experiment row with deserialized list columns."""
        row = self._experiments_df[self._experiments_df["exp_id"] == exp_id]
        if row.empty:
            return None
        row = row.iloc[0].to_dict()

        # Deserialize JSON array columns
        list_cols = ["train_loss_curve", "val_loss_curve", "train_acc_curve",
                      "val_acc_curve", "val_f1_curve"]
        for col in list_cols:
            if col in row and isinstance(row[col], str):
                try:
                    row[col] = json.loads(row[col])
                except (json.JSONDecodeError, TypeError):
                    row[col] = []

        # Build confusion matrix
        cm_keys = ["cm_healthy_to_healthy", "cm_healthy_to_semi",
                    "cm_healthy_to_unhealthy", "cm_semi_to_healthy",
                    "cm_semi_to_semi", "cm_semi_to_unhealthy",
                    "cm_unhealthy_to_healthy", "cm_unhealthy_to_semi",
                    "cm_unhealthy_to_unhealthy"]
        cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for key in cm_keys:
            parts = key.replace("cm_", "").split("_to_")
            from_map = {"healthy": 0, "semi": 1, "unhealthy": 2}
            if parts[0] in from_map and parts[1] in from_map:
                val = row.get(key, 0)
                cm[from_map[parts[0]]][from_map[parts[1]]] = int(val) if pd.notna(val) else 0

        row["confusion_matrix"] = cm
        row["exp_id"] = int(row["exp_id"])

        return row

    # ------------------------------------------------------------------
    # Active experiment
    # ------------------------------------------------------------------
    def set_active_experiment(self, exp_id):
        exp = self.get_experiment(exp_id)
        if exp:
            self._active_experiment_id = exp_id
            self._num_classes = 3 if exp.get("label_mode") == "3class" else 2
            return True
        return False

    def get_active_experiment_id(self):
        return self._active_experiment_id

    def get_num_classes(self):
        return self._num_classes


# Singleton instance
store = DataStore()
