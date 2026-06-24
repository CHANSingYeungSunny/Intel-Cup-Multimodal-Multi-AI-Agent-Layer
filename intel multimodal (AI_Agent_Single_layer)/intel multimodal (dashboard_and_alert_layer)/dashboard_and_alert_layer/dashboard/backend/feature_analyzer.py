"""
FeatureAnalyzer: PCA / t-SNE dimensionality reduction and signal generation.

Generates simulated cough/respiratory waveforms and physiological trends
from the 256-dim fusion feature vectors, since raw modality data is not
available in the Fusion Layer outputs.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from config import PCA_N_COMPONENTS, TSNE_PERPLEXITY, TSNE_RANDOM_STATE


class FeatureAnalyzer:
    """Fits PCA/t-SNE on the feature matrix and generates pseudo-signals."""

    def __init__(self, feature_matrix):
        """
        Args:
            feature_matrix: np.ndarray of shape (N, 256)
        """
        self._features = feature_matrix
        self._n_samples, self._n_dims = feature_matrix.shape
        self._pca = None
        self._tsne = None
        self._pca_projections = None
        self._tsne_projections = None
        self._explained_variance = None

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------
    def fit_pca(self, n_components=PCA_N_COMPONENTS):
        """Fit PCA on the feature matrix."""
        n_comp = min(n_components, self._n_dims, self._n_samples)
        self._pca = PCA(n_components=n_comp, random_state=42)
        self._pca_projections = self._pca.fit_transform(self._features)
        self._explained_variance = self._pca.explained_variance_ratio_.tolist()
        print(f"[FeatureAnalyzer] PCA fitted: {n_comp} components, "
              f"explained variance: {[round(v, 3) for v in self._explained_variance]}")
        return self._pca_projections

    def get_pca_coordinates(self, n_components=2):
        """Return PCA coordinates for the first `n_components` dimensions."""
        if self._pca is None:
            self.fit_pca()
        n = min(n_components, self._pca_projections.shape[1])
        return self._pca_projections[:, :n].tolist()

    def get_explained_variance(self):
        if self._explained_variance is None:
            self.fit_pca()
        return self._explained_variance

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------
    def fit_tsne(self, n_components=2):
        """Fit t-SNE on the feature matrix. Computationally heavier."""
        n_comp = min(n_components, 3)
        self._tsne = TSNE(
            n_components=n_comp,
            perplexity=min(TSNE_PERPLEXITY, self._n_samples - 1),
            random_state=TSNE_RANDOM_STATE,
            init="pca",
        )
        self._tsne_projections = self._tsne.fit_transform(self._features)
        print(f"[FeatureAnalyzer] t-SNE fitted: {n_comp} components")
        return self._tsne_projections

    def get_tsne_coordinates(self, n_components=2):
        """Return t-SNE coordinates. Fits on first call."""
        if self._tsne is None:
            self.fit_tsne(n_components=n_components)
        n = min(n_components, self._tsne_projections.shape[1])
        return self._tsne_projections[:, :n].tolist()

    # ------------------------------------------------------------------
    # Signal Generation: Cough / Respiratory Waveform
    # ------------------------------------------------------------------
    @staticmethod
    def generate_cough_curve(feature_vector, sample_metadata=None, n_points=50, duration=2.0):
        """
        Generate a simulated respiratory waveform from a feature vector.

        Uses PC0-like projection (dot product with first principal direction)
        as the baseline offset, then adds sinusoidal components at respiratory
        frequencies (0.15–0.4 Hz for adults).

        Returns:
            dict with timestamps, amplitude, envelope_upper, envelope_lower,
            peak_count, and metadata.
        """
        vec = np.array(feature_vector, dtype=np.float32)
        # Use feature statistics as pseudo-PC scores
        pc0 = float(np.mean(vec))                  # DC offset proxy
        pc1 = float(np.std(vec))                   # amplitude proxy
        pc2 = float(np.percentile(vec, 90))        # irregularity proxy

        t = np.linspace(0, duration, n_points)

        # Base respiratory frequency: map pc0 from [-2, 2] to [0.15, 0.4] Hz
        freq = np.clip(0.275 + pc0 * 0.0625, 0.12, 0.5)
        # Amplitude: map pc1 from [0, 2] to [0.1, 1.0]
        amp = np.clip(pc1 * 0.5, 0.1, 1.0)
        # Noise level based on prediction (healthier = more regular)
        noise_level = np.clip(abs(pc2) * 0.05, 0.01, 0.2)

        fv_slice = np.array(feature_vector[:10], dtype=np.float32)
        rng = np.random.RandomState(hash(fv_slice.tobytes()) % (2**31))
        signal = amp * np.sin(2 * np.pi * freq * t)
        # Add harmonic
        signal += amp * 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        # Add noise
        signal += rng.normal(0, noise_level, n_points)
        # Add cough spikes for unhealthy subjects
        prediction = sample_metadata.get("prediction", 0) if sample_metadata else 0
        if prediction == 2:  # Unhealthy
            n_coughs = rng.randint(1, 4)
            for _ in range(n_coughs):
                cough_t = rng.uniform(0.2, duration - 0.2)
                cough_idx = int(cough_t / duration * n_points)
                # Exponential decay burst
                burst = np.exp(-np.abs(t - cough_t) * 15) * rng.uniform(1.5, 3.0)
                signal += burst

        # Envelopes
        envelope_upper = signal + amp * 0.3
        envelope_lower = signal - amp * 0.3
        # Count peaks
        from scipy import signal as scipy_signal
        try:
            peaks, _ = scipy_signal.find_peaks(signal, distance=n_points // 6)
            peak_count = len(peaks)
        except Exception:
            peak_count = int(freq * duration)

        return {
            "timestamps": t.tolist(),
            "amplitude": signal.tolist(),
            "envelope_upper": envelope_upper.tolist(),
            "envelope_lower": envelope_lower.tolist(),
            "peak_count": peak_count,
            "respiratory_rate_bpm": round(freq * 60, 1),
            "prediction": prediction,
            "label": sample_metadata.get("label", -1) if sample_metadata else -1,
        }

    # ------------------------------------------------------------------
    # Physio Trend Generation
    # ------------------------------------------------------------------
    @staticmethod
    def generate_physio_trends(subject_data):
        """
        Generate simulated multi-vital-sign trends from feature vectors.

        Maps PCA proxy scores to physiological ranges:
          - Heart Rate:  60–100 bpm
          - SpO2:        90–100 %
          - RR Interval: 0.6–1.2 s (50–100 bpm respiratory)

        Args:
            subject_data: dict from DataStore.get_feature_vectors_for_subject()
                with keys: subject_id, windows, predictions, labels, feature_vectors

        Returns:
            dict with arrays for heart_rate, spo2, rr_interval, plus annotations.
        """
        vectors = np.array(subject_data["feature_vectors"], dtype=np.float32)
        n_windows = len(vectors)

        if n_windows == 0:
            return {"subject_id": subject_data["subject_id"], "windows": [],
                    "heart_rate": [], "spo2": [], "rr_interval": [],
                    "predictions": [], "labels": []}

        # Compute proxy scores from feature vectors
        pc0 = vectors.mean(axis=1)       # mean as PC0 proxy
        pc1 = vectors.std(axis=1)        # std as PC1 proxy
        pc2 = np.percentile(vectors, 90, axis=1)  # 90th percentile as PC2 proxy

        # Map to physiological ranges
        hr = 80 + pc0 * 10             # center at 80 bpm, spread ±10
        hr = np.clip(hr, 60, 100)

        spo2 = 97 - abs(pc1) * 3       # center at 97%, drops with variability
        spo2 = np.clip(spo2, 88, 100)

        rr_interval = 0.85 - pc0 * 0.1  # center at 0.85s, shorter with stress
        rr_interval = np.clip(rr_interval, 0.5, 1.3)

        # For unhealthy predictions, degrade the signals
        predictions = np.array(subject_data["predictions"])
        hr[predictions == 2] += 10
        spo2[predictions == 2] -= 5
        rr_interval[predictions == 2] -= 0.15
        hr = np.clip(hr, 50, 120)
        spo2 = np.clip(spo2, 80, 100)

        return {
            "subject_id": subject_data["subject_id"],
            "windows": subject_data["windows"],
            "window_indices": list(range(n_windows)),
            "heart_rate_sim": hr.round(1).tolist(),
            "spo2_sim": spo2.round(1).tolist(),
            "rr_interval_sim": rr_interval.round(3).tolist(),
            "predictions": [int(p) for p in predictions],
            "labels": [int(l) for l in subject_data["labels"]],
            "prediction_names": [
                {0: "Healthy", 1: "Sub-healthy", 2: "Unhealthy"}.get(int(p), "Unknown")
                for p in predictions
            ],
        }
