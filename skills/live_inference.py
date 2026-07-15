"""
Real-Time Live Inference Engine.

Orchestrates the full hardware → AI → Agent pipeline:
  1. Capture camera frame  → Vision model (Swin-Tiny) → 768-dim features
  2. Capture microphone audio → Audio model (AST)      → 128-dim features
  3. Read sensor data       → Physiological model (iT) → 128-dim features
  4. Concatenate            → Fusion model            → 256-dim CLS + prediction
  5. POST to AI Agent       → Structured health advice

On platforms without hardware (Windows, no /dev/video0, no I2C sensors):
falls back to mock data producing plausible health predictions for demo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
_IS_LINUX = platform.system() == "Linux"
_IS_WINDOWS = platform.system() == "Windows"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SENSOR_SCRIPT = _PROJECT_ROOT / "tools" / "sensors" / "read_live_sensors.py"

# Try to import OpenCV (optional — only needed for real camera)
try:
    import cv2

    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False

# Try to import torch (optional — only needed for real model inference)
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA_DEVICES = ["/dev/video0", "/dev/video1"]
CAMERA_FRAME_WIDTH = 224
CAMERA_FRAME_HEIGHT = 224
VISION_FEATURE_DIM = 768
AUDIO_FEATURE_DIM = 128
PHYSIO_FEATURE_DIM = 128
FUSION_INPUT_DIM = VISION_FEATURE_DIM + AUDIO_FEATURE_DIM + PHYSIO_FEATURE_DIM  # 1024
FUSION_OUTPUT_DIM = 256
DEFAULT_INFERENCE_INTERVAL = 2.0  # seconds

HEALTH_STATES = {0: "Healthy", 1: "Sub-healthy", 2: "Unhealthy"}


# ===========================================================================
# LiveInferenceEngine
# ===========================================================================


class LiveInferenceEngine:
    """
    Real-time inference engine that bridges hardware capture → AI models →
    Multi-Agent Coordinator.

    Runs a background asyncio task that periodically:
    1. Captures camera frame + audio + sensor data
    2. Runs modality models (or mock if unavailable)
    3. Runs fusion model (or mock)
    4. Sends prediction to AI Agent Coordinator
    5. Caches the latest result

    Parameters
    ----------
    coordinator : AgentCoordinator or None
        Multi-Agent Coordinator for advice generation.
    interval : float
        Seconds between inference runs (default 2.0).
    agent_api_url : str
        Base URL for the AI Agent API (used when coordinator is None).
    """

    def __init__(
        self,
        coordinator=None,
        interval: float = DEFAULT_INFERENCE_INTERVAL,
        agent_api_url: str = "http://localhost:8000/api/v1",
    ):
        self._coordinator = coordinator
        self._interval = interval
        self._agent_api_url = agent_api_url
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Cached latest result
        self._latest: dict = {
            "prediction": None,
            "health_state": "Initializing...",
            "confidence": None,
            "advice": None,
            "anomalies": [],
            "features": None,
            "timestamp": None,
            "status": "initializing",
            "mode": "mock" if not self._hardware_available() else "live",
        }

        # Model placeholders (loaded lazily)
        self._vision_model = None
        self._audio_model = None
        self._physio_model = None
        self._fusion_model = None

        # Historical feature buffer (for trend analysis)
        self._feature_history: list[dict] = []

        logger.info(
            "LiveInferenceEngine initialized (mode=%s, interval=%.1fs)",
            self._latest["mode"],
            self._interval,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_latest(self) -> dict:
        """Return the most recent inference result (non-blocking)."""
        return dict(self._latest)

    async def start(self) -> None:
        """Start the background inference loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._inference_loop())
        logger.info("Live inference loop started")

    async def stop(self) -> None:
        """Stop the background inference loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Live inference loop stopped")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------

    async def _inference_loop(self) -> None:
        """Background loop: capture → infer → agent → cache."""
        while self._running:
            try:
                result = await self.run_inference()
                self._latest = result
                self._feature_history.append({
                    "prediction": result.get("prediction"),
                    "timestamp": result.get("timestamp"),
                })
                # Keep last 60 entries (~2 min at 2s interval)
                if len(self._feature_history) > 60:
                    self._feature_history = self._feature_history[-60:]
            except Exception as exc:
                logger.warning("Live inference cycle failed: %s", exc)
                self._latest["status"] = "degraded"
                self._latest["error"] = str(exc)

            await asyncio.sleep(self._interval)

    async def run_inference(self) -> dict:
        """
        Execute one full inference cycle.

        Returns dict with keys: prediction, health_state, confidence,
        advice, anomalies, features, timestamp, status, mode.
        """
        now = datetime.now(timezone.utc)
        status = "ok"
        mode = self._latest.get("mode", "mock")

        # --- 1. Capture frame → Vision features ---
        try:
            frame = await self._capture_frame()
            vision_features = await self._run_vision_model(frame)
        except Exception as exc:
            logger.debug("Vision capture/inference failed: %s — using mock", exc)
            vision_features = self._mock_vision_features()
            status = "degraded"

        # --- 2. Capture audio → Audio features ---
        try:
            audio_samples = await self._capture_audio()
            audio_features = await self._run_audio_model(audio_samples)
        except Exception as exc:
            logger.debug("Audio capture/inference failed: %s — using mock", exc)
            audio_features = self._mock_audio_features()
            status = "degraded"

        # --- 3. Read sensors → Physiological features ---
        try:
            sensor_data = await self._read_sensors()
            physio_features = await self._run_physio_model(sensor_data)
        except Exception as exc:
            logger.debug("Sensor read/inference failed: %s — using mock", exc)
            physio_features = self._mock_physio_features()
            status = "degraded"

        # --- 4. Fusion ---
        try:
            fused = await self._run_fusion(
                vision_features, audio_features, physio_features
            )
        except Exception as exc:
            logger.debug("Fusion failed: %s — using mock", exc)
            fused = self._mock_fusion()
            status = "degraded"

        prediction = fused["prediction"]
        health_state = HEALTH_STATES.get(prediction, "Unknown")
        confidence = fused.get("confidence", None)

        # --- 5. Get AI Agent advice ---
        advice = None
        anomalies = []
        try:
            advice, anomalies = await self._get_agent_advice(
                prediction=prediction,
                feature_vector=fused.get("features"),
                subject_id="live_subject",
            )
        except Exception as exc:
            logger.debug("Agent advice failed: %s", exc)
            status = "degraded"

        return {
            "prediction": prediction,
            "health_state": health_state,
            "confidence": confidence,
            "advice": advice,
            "anomalies": anomalies,
            "features": fused.get("features"),
            "timestamp": now.isoformat(),
            "status": status,
            "mode": mode,
        }

    # ------------------------------------------------------------------
    # Hardware capture (with mock fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _hardware_available() -> bool:
        """Check if real hardware is accessible."""
        if not _IS_LINUX:
            return False
        # Check for camera
        has_camera = any(
            os.path.exists(d) for d in CAMERA_DEVICES
        )
        # Check for sensor script
        has_sensors = _SENSOR_SCRIPT.exists()
        return has_camera or has_sensors

    async def _capture_frame(self) -> np.ndarray:
        """Capture a single frame from the camera (or mock)."""
        if not _IS_LINUX or not _OPENCV_AVAILABLE:
            return self._mock_frame()

        for device in CAMERA_DEVICES:
            if not os.path.exists(device):
                continue
            try:
                idx = int(device.replace("/dev/video", ""))
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    frame = cv2.resize(
                        frame, (CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT)
                    )
                    return frame
            except Exception:
                continue

        return self._mock_frame()

    async def _capture_audio(self) -> np.ndarray:
        """Capture 1 second of audio from USB microphone (or mock)."""
        if not _IS_LINUX:
            return self._mock_audio()

        import shutil

        arecord = shutil.which("arecord")
        if not arecord:
            return self._mock_audio()

        try:
            result = subprocess.run(
                [
                    arecord,
                    "-D", "plughw:2,0",
                    "-f", "S16_LE",
                    "-c", "1",
                    "-r", "16000",
                    "-d", "1",
                    "-t", "wav",
                    "-q",
                ],
                capture_output=True,
                timeout=4,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                import io
                import wave

                with wave.open(io.BytesIO(result.stdout), "rb") as wav:
                    frames = wav.readframes(wav.getnframes())
                    samples = np.frombuffer(frames, dtype=np.int16).astype(
                        np.float32
                    )
                    return samples / 32768.0  # Normalize to [-1, 1]
        except Exception:
            pass

        return self._mock_audio()

    async def _read_sensors(self) -> dict:
        """Read sensor data via subprocess (or mock)."""
        if _SENSOR_SCRIPT.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(_SENSOR_SCRIPT)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return json.loads(result.stdout)
            except Exception:
                pass
        return self._mock_sensor_data()

    # ------------------------------------------------------------------
    # AI Model inference (with mock fallback)
    # ------------------------------------------------------------------

    async def _run_vision_model(self, frame: np.ndarray) -> list[float]:
        """Run Swin-Tiny on a captured frame (or mock)."""
        if _TORCH_AVAILABLE and self._vision_model is not None:
            try:
                import torchvision.transforms as T

                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
                tensor = transform(frame).unsqueeze(0)
                with torch.no_grad():
                    features = self._vision_model(tensor)
                    if isinstance(features, tuple):
                        features = features[0]
                    return features.squeeze(0).cpu().numpy().tolist()
            except Exception:
                pass
        return self._mock_vision_features()

    async def _run_audio_model(self, samples: np.ndarray) -> list[float]:
        """Run AST on captured audio (or mock)."""
        if _TORCH_AVAILABLE and self._audio_model is not None:
            try:
                tensor = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    features = self._audio_model(tensor)
                    if isinstance(features, tuple):
                        features = features[0]
                    return features.squeeze(0).cpu().numpy().tolist()
            except Exception:
                pass
        return self._mock_audio_features()

    async def _run_physio_model(self, sensor_data: dict) -> list[float]:
        """Run iTransformer on sensor data (or mock)."""
        if _TORCH_AVAILABLE and self._physio_model is not None:
            try:
                # Extract relevant time-series channels
                hr = float(sensor_data.get("max30102", {}).get("heart_rate", 75) or 75)
                spo2 = float(sensor_data.get("max30102", {}).get("spo2", 97) or 97)
                temp = float(
                    sensor_data.get("scd40", {}).get("temperature_c", 25) or 25
                )
                co2 = float(sensor_data.get("scd40", {}).get("co2_ppm", 600) or 600)

                # Build a minimal 4-channel sequence (repeat 32 times for context)
                seq = np.tile([hr, spo2, temp, co2], (32, 1))
                tensor = torch.from_numpy(seq).float().unsqueeze(0)
                with torch.no_grad():
                    features = self._physio_model(tensor)
                    if isinstance(features, tuple):
                        features = features[0]
                    return features.squeeze(0).cpu().numpy().tolist()
            except Exception:
                pass
        return self._mock_physio_features()

    async def _run_fusion(
        self,
        vision_f: list[float],
        audio_f: list[float],
        physio_f: list[float],
    ) -> dict:
        """Run MultimodalFusionEncoder (or mock)."""
        if _TORCH_AVAILABLE and self._fusion_model is not None:
            try:
                combined = vision_f + audio_f + physio_f
                tensor = torch.tensor([combined], dtype=torch.float32)
                with torch.no_grad():
                    logits, fusion_emb = self._fusion_model(tensor)
                    pred = int(torch.argmax(logits, dim=1).item())
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    confidence = float(probs[pred])
                    return {
                        "prediction": pred,
                        "confidence": round(confidence, 4),
                        "features": fusion_emb.squeeze(0).cpu().numpy().tolist(),
                    }
            except Exception:
                pass
        return self._mock_fusion()

    # ------------------------------------------------------------------
    # Agent advice
    # ------------------------------------------------------------------

    async def _get_agent_advice(
        self,
        prediction: int,
        feature_vector: Optional[list[float]] = None,
        subject_id: str = "live_subject",
    ) -> tuple:
        """Get AI agent advice for the given prediction."""
        # Prefer coordinator (in-process)
        if self._coordinator:
            result = await self._coordinator.process_tick_multi(
                prediction=prediction,
                subject_id=subject_id,
                feature_vector=feature_vector,
            )
            advice = result.get("single_agent_advice") or result.get(
                "multi_agent_advice"
            )
            anomalies = result.get("anomalies", [])
            return advice, anomalies

        # Fall back to HTTP call
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self._agent_api_url}/tick",
                    json={
                        "prediction": prediction,
                        "subject_id": subject_id,
                        "feature_vector": feature_vector,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data, []
        except Exception:
            pass

        return None, []

    # ------------------------------------------------------------------
    # Mock data generators
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_frame() -> np.ndarray:
        """Generate a mock camera frame (random noise + gradient)."""
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add a simple gradient to simulate face-like structure
        for i in range(224):
            frame[i, :, 0] = np.clip(
                frame[i, :, 0] + int(30 * np.sin(i / 30)), 0, 255
            ).astype(np.uint8)
        return frame

    @staticmethod
    def _mock_audio() -> np.ndarray:
        """Generate mock audio samples (silence + slight noise)."""
        return np.random.randn(16000).astype(np.float32) * 0.01

    @staticmethod
    def _mock_sensor_data() -> dict:
        """Generate plausible mock sensor readings."""
        return {
            "scd40": {
                "co2_ppm": random.randint(400, 1200),
                "temperature_c": round(random.uniform(22.0, 28.0), 1),
                "humidity_percent": round(random.uniform(35.0, 65.0), 1),
                "status": "mock",
            },
            "max30102": {
                "heart_rate": random.randint(65, 100),
                "spo2": random.randint(93, 99),
                "finger_detected": True,
                "status": "mock",
            },
            "mlx90614": {
                "detected": False,
                "object_temperature_c": None,
                "status": "mock",
            },
            "camera": {"detected": True, "devices": ["mock"], "status": "mock"},
            "microphone": {"detected": True, "status": "mock"},
            "timestamp": time.time(),
        }

    @staticmethod
    def _mock_vision_features() -> list[float]:
        """Generate mock 768-dim vision features."""
        vec = np.random.randn(768).astype(np.float32) * 0.5
        vec[0:256] += 0.3  # Bias first third (simulates face detection)
        return (vec / np.linalg.norm(vec)).tolist()

    @staticmethod
    def _mock_audio_features() -> list[float]:
        """Generate mock 128-dim audio features."""
        vec = np.random.randn(128).astype(np.float32) * 0.5
        vec[0:40] += 0.2  # Bias lower frequencies
        return (vec / np.linalg.norm(vec)).tolist()

    @staticmethod
    def _mock_physio_features() -> list[float]:
        """Generate mock 128-dim physiological features."""
        vec = np.random.randn(128).astype(np.float32) * 0.5
        return (vec / np.linalg.norm(vec)).tolist()

    @staticmethod
    def _mock_fusion() -> dict:
        """Generate mock fusion output."""
        pred = random.choices([0, 0, 0, 1, 1, 2], k=1)[0]  # Weighted toward healthy
        conf = round(random.uniform(0.60, 0.95), 4)
        features = (
            np.random.randn(256).astype(np.float32) * 0.5
        ).tolist()
        return {
            "prediction": pred,
            "confidence": conf,
            "features": features,
        }
