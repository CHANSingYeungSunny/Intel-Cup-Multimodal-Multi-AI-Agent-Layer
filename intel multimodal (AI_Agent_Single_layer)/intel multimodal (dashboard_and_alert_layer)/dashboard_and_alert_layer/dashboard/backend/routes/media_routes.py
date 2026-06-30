"""REST endpoints for lightweight camera and microphone media status."""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
from flask import Blueprint, Response, jsonify, request

try:
    import cv2  # noqa: F401
    _OPENCV_AVAILABLE = True
    _OPENCV_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - defensive runtime fallback
    _OPENCV_AVAILABLE = False
    _OPENCV_IMPORT_ERROR = str(exc)

media_bp = Blueprint("media", __name__)
PROJECT_ROOT = Path(__file__).resolve().parents[5]
CAMERA_DEVICE_CANDIDATES = ["/dev/video0", "/dev/video1"]
CAMERA_TIMEOUT_SECONDS = 4
CAMERA_JPEG_QUALITY = 85
CAMERA_STREAM_FPS = 6
CAMERA_STREAM_BOUNDARY = "frame"
MICROPHONE_CAPTURE_TIMEOUT_SECONDS = 4
MICROPHONE_SAMPLE_SECONDS = 1
MICROPHONE_SAMPLE_RATE = 16000
MICROPHONE_MAX_RMS = 32767.0
CAMERA_CAPTURE_SNIPPET = r'''
import json
import re
import sys

try:
    import cv2
except Exception as exc:
    print(json.dumps({"error": f"opencv_import_failed: {exc}"}), file=sys.stderr)
    raise SystemExit(1)


def resolve_source(device_path):
    match = re.search(r"(\d+)$", device_path)
    if device_path.startswith("/dev/video") and match:
        return int(match.group(1))
    return device_path


device_path = sys.argv[1]
source = resolve_source(device_path)
cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
try:
    if not cap.isOpened():
        raise RuntimeError(f"unable_to_open_camera:{device_path}")

    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame = None
    for _ in range(3):
        ok, current = cap.read()
        if ok and current is not None:
            frame = current
            break

    if frame is None:
        raise RuntimeError(f"unable_to_read_frame:{device_path}")

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(sys.argv[2])],
    )
    if not ok:
        raise RuntimeError(f"unable_to_encode_jpeg:{device_path}")

    sys.stdout.buffer.write(encoded.tobytes())
finally:
    cap.release()
'''


def _camera_error(message, device=None, status="camera_capture_failed"):
    payload = {
        "detected": False,
        "status": status,
        "error": message,
    }
    if device:
        payload["device"] = device
    return payload



def _decode_subprocess_error(raw_bytes):
    text = raw_bytes.decode("utf-8", "ignore").strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
    except Exception:
        pass

    return text



def _existing_camera_devices():
    return [device for device in CAMERA_DEVICE_CANDIDATES if Path(device).exists()]



def _resolve_camera_source(device_path):
    match = re.search(r"(\d+)$", device_path)
    if device_path.startswith("/dev/video") and match:
        return int(match.group(1))
    return device_path



def _open_camera_capture():
    if not _OPENCV_AVAILABLE:
        return None, None, _camera_error(
            f"opencv_unavailable:{_OPENCV_IMPORT_ERROR}",
            status="camera_capture_unavailable",
        ), 503

    devices = _existing_camera_devices()
    if not devices:
        return None, None, _camera_error(
            "no_camera_device_found",
            status="camera_not_detected",
        ), 503

    last_error = None
    for device in devices:
        if not os.access(device, os.R_OK | os.W_OK):
            last_error = _camera_error(
                f"permission_denied_opening_{device}",
                device=device,
                status="camera_permission_denied",
            )
            continue

        capture = None
        try:
            capture = cv2.VideoCapture(_resolve_camera_source(device), cv2.CAP_V4L2)
            if not capture.isOpened():
                last_error = _camera_error(
                    f"unable_to_open_camera:{device}",
                    device=device,
                    status="camera_not_detected",
                )
                capture.release()
                continue

            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if hasattr(cv2, "CAP_PROP_FRAME_WIDTH"):
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            if hasattr(cv2, "CAP_PROP_FRAME_HEIGHT"):
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if hasattr(cv2, "CAP_PROP_FPS"):
                capture.set(cv2.CAP_PROP_FPS, CAMERA_STREAM_FPS)

            return capture, device, None, 200
        except Exception as exc:
            last_error = _camera_error(str(exc), device=device)
            if capture is not None:
                capture.release()

    return None, None, last_error or _camera_error("camera_capture_failed"), 503



def _capture_camera_snapshot_bytes():
    if not _OPENCV_AVAILABLE:
        return None, _camera_error(
            f"opencv_unavailable:{_OPENCV_IMPORT_ERROR}",
            status="camera_capture_unavailable",
        ), 503

    devices = _existing_camera_devices()
    if not devices:
        return None, _camera_error(
            "no_camera_device_found",
            status="camera_not_detected",
        ), 503

    last_error = None
    for device in devices:
        if not os.access(device, os.R_OK | os.W_OK):
            last_error = _camera_error(
                f"permission_denied_opening_{device}",
                device=device,
                status="camera_permission_denied",
            )
            continue

        try:
            result = subprocess.run(
                [sys.executable, "-c", CAMERA_CAPTURE_SNIPPET, device, str(CAMERA_JPEG_QUALITY)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                timeout=CAMERA_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, _camera_error(
                f"camera_capture_timeout_after_{CAMERA_TIMEOUT_SECONDS}_seconds",
                device=device,
                status="camera_capture_timeout",
            ), 504
        except Exception as exc:
            last_error = _camera_error(str(exc), device=device)
            continue

        if result.returncode == 0 and result.stdout:
            return result.stdout, None, 200

        error_message = _decode_subprocess_error(result.stderr) or _decode_subprocess_error(result.stdout)
        last_error = _camera_error(
            error_message or f"camera_capture_failed_returncode_{result.returncode}",
            device=device,
        )

    return None, last_error or _camera_error("camera_capture_failed"), 503



def _generate_camera_stream(capture):
    frame_interval = 1.0 / float(CAMERA_STREAM_FPS)
    consecutive_failures = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    break
                time.sleep(frame_interval)
                continue

            consecutive_failures = 0
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(CAMERA_JPEG_QUALITY)],
            )
            if not ok:
                time.sleep(frame_interval)
                continue

            frame_bytes = encoded.tobytes()
            yield (
                b"--" + CAMERA_STREAM_BOUNDARY.encode("ascii") + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            time.sleep(frame_interval)
    finally:
        capture.release()



def _find_usb_microphone_device():
    arecord_path = shutil.which("arecord")
    if not arecord_path:
        return None, "arecord_not_available"

    try:
        result = subprocess.run(
            [arecord_path, "-l"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "arecord_device_scan_timeout"
    except Exception as exc:
        return None, str(exc)

    if result.returncode != 0:
        return None, (result.stderr or result.stdout or "arecord_scan_failed").strip()

    for line in result.stdout.splitlines():
        lower = line.lower()
        if "card" not in lower or "device" not in lower:
            continue
        if "usb" not in lower and "composite" not in lower:
            continue

        match = re.search(r"card\s+(\d+):.*device\s+(\d+):", line, re.IGNORECASE)
        if match:
            card_index = int(match.group(1))
            device_index = int(match.group(2))
            return {
                "detected": True,
                "capture_device": f"plughw:{card_index},{device_index}",
                "card_index": card_index,
                "device_index": device_index,
                "description": line.strip(),
                "arecord_path": arecord_path,
            }, None

    return None, "usb_microphone_not_detected"



def _build_microphone_unavailable(error_message, capture_device=None):
    payload = {
        "detected": False,
        "level_rms": None,
        "level_percent": None,
        "status": "microphone_unavailable",
    }
    if capture_device:
        payload["capture_device"] = capture_device
    if error_message:
        payload["error"] = error_message
    return payload



def _classify_microphone_level(level_percent):
    if level_percent < 2.0:
        return "quiet"
    if level_percent < 12.0:
        return "normal"
    return "loud"



def _capture_microphone_level_payload():
    device_info, device_error = _find_usb_microphone_device()
    if not device_info:
        return _build_microphone_unavailable(device_error), 503

    capture_device = device_info["capture_device"]
    cmd = [
        device_info["arecord_path"],
        "-D",
        capture_device,
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        str(MICROPHONE_SAMPLE_RATE),
        "-d",
        str(MICROPHONE_SAMPLE_SECONDS),
        "-t",
        "wav",
        "-q",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=MICROPHONE_CAPTURE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _build_microphone_unavailable(
            f"microphone_capture_timeout_after_{MICROPHONE_CAPTURE_TIMEOUT_SECONDS}_seconds",
            capture_device=capture_device,
        ), 504
    except Exception as exc:
        return _build_microphone_unavailable(str(exc), capture_device=capture_device), 503

    if result.returncode != 0:
        error_text = result.stderr.decode("utf-8", "ignore").strip() or result.stdout.decode("utf-8", "ignore").strip()
        return _build_microphone_unavailable(error_text or "microphone_capture_failed", capture_device=capture_device), 503

    try:
        with wave.open(io.BytesIO(result.stdout), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16)
            if wav_file.getnchannels() > 1 and samples.size:
                samples = samples.reshape(-1, wav_file.getnchannels())[:, 0]
    except Exception as exc:
        return _build_microphone_unavailable(f"microphone_decode_failed:{exc}", capture_device=capture_device), 503

    if samples.size == 0:
        rms = 0.0
    else:
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    level_percent = min(100.0, (rms / MICROPHONE_MAX_RMS) * 100.0)
    return {
        "detected": True,
        "level_rms": round(rms, 2),
        "level_percent": round(level_percent, 2),
        "status": _classify_microphone_level(level_percent),
        "capture_device": capture_device,
    }, 200


@media_bp.route("/api/camera_snapshot", methods=["GET"])
def camera_snapshot():
    snapshot_bytes, error_payload, status_code = _capture_camera_snapshot_bytes()
    if snapshot_bytes is None:
        return jsonify(error_payload), status_code

    return Response(
        snapshot_bytes,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        },
    )


@media_bp.route("/api/camera_stream", methods=["GET", "HEAD"])
def camera_stream():
    capture, device, error_payload, status_code = _open_camera_capture()
    if capture is None:
        return jsonify(error_payload), status_code

    response_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "X-Camera-Device": device or "",
    }

    if request.method == "HEAD":
        capture.release()
        return Response(
            status=200,
            mimetype=f"multipart/x-mixed-replace; boundary={CAMERA_STREAM_BOUNDARY}",
            headers=response_headers,
        )

    return Response(
        _generate_camera_stream(capture),
        mimetype=f"multipart/x-mixed-replace; boundary={CAMERA_STREAM_BOUNDARY}",
        headers=response_headers,
        direct_passthrough=True,
    )


@media_bp.route("/api/microphone_level", methods=["GET"])
def microphone_level():
    payload, status_code = _capture_microphone_level_payload()
    return jsonify(payload), status_code

