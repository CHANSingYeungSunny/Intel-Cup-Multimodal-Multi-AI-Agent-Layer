#!/usr/bin/env python3

import json
import sys
import time
import glob
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
SCD40_SCRIPT = "tools/sensors/test_scd40.py"
MAX30102_SCRIPT = "tools/sensors/test_max30102.py"
SCD40_TIMEOUT = 12
MAX30102_TIMEOUT = 8
SCD40_MAX_ATTEMPTS = 2
SCD40_RETRY_DELAY_SECONDS = 0.35


def run_json_script(script_path, timeout=15):
    full_path = PROJECT_ROOT / script_path

    try:
        result = subprocess.run(
            [PYTHON, str(full_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout.strip()
        error_output = result.stderr.strip()

        if output:
            return json.loads(output)

        if error_output:
            try:
                return json.loads(error_output)
            except Exception:
                return {
                    "error": error_output,
                    "returncode": result.returncode,
                }

        return {
            "error": "no output",
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "error": f"timeout running {script_path}",
        }
    except Exception as e:
        return {
            "error": str(e),
        }


def _scd40_unavailable_payload(error_message=None, attempts=1):
    payload = {
        "co2_ppm": None,
        "temperature_c": None,
        "humidity_percent": None,
        "status": "temporarily_unavailable",
    }

    if error_message:
        payload["error"] = error_message

    payload["attempts"] = attempts
    return payload


def _extract_scd40_error(payload):
    if isinstance(payload, dict):
        nested = payload.get("scd40")
        if isinstance(nested, dict) and nested.get("error"):
            return str(nested.get("error"))
        if payload.get("error"):
            return str(payload.get("error"))
    return "unknown_scd40_error"


def _is_valid_scd40_payload(payload):
    return isinstance(payload, dict) and all(
        payload.get(key) is not None
        for key in ("co2_ppm", "temperature_c", "humidity_percent")
    )


def _is_transient_scd40_error(error_message):
    lower = (error_message or "").lower()
    return (
        "remote i/o error" in lower
        or "errno 121" in lower
        or "[errno 121]" in lower
        or "resource temporarily unavailable" in lower
    )


def read_scd40_status():
    last_error = None

    for attempt in range(1, SCD40_MAX_ATTEMPTS + 1):
        scd40_result = run_json_script(SCD40_SCRIPT, timeout=SCD40_TIMEOUT)
        scd40_payload = scd40_result.get("scd40", scd40_result)

        if _is_valid_scd40_payload(scd40_payload):
            return scd40_payload

        last_error = _extract_scd40_error(scd40_result)

        if attempt < SCD40_MAX_ATTEMPTS and _is_transient_scd40_error(last_error):
            time.sleep(SCD40_RETRY_DELAY_SECONDS)
            continue

        break

    return _scd40_unavailable_payload(last_error, attempts=attempt)


def read_camera_status():
    video_devices = sorted(glob.glob("/dev/video*"))

    return {
        "detected": len(video_devices) > 0,
        "devices": video_devices,
        "status": "camera_detected" if video_devices else "camera_not_detected",
    }


def read_microphone_status():
    try:
        cards_text = ""
        pcm_text = ""

        try:
            with open("/proc/asound/cards", "r") as f:
                cards_text = f.read()
        except Exception:
            cards_text = ""

        try:
            with open("/proc/asound/pcm", "r") as f:
                pcm_text = f.read()
        except Exception:
            pcm_text = ""

        combined = cards_text + "\n" + pcm_text
        lower = combined.lower()

        detected = (
            ("usb-audio" in lower or "usb composite device" in lower or "jieli" in lower)
            and "capture" in lower
        )

        if detected:
            status = "microphone_detected"
        elif "usb-audio" in lower or "usb composite device" in lower or "jieli" in lower:
            status = "usb_audio_detected_no_capture_pcm_found"
        else:
            status = "microphone_not_detected"

        return {
            "detected": detected,
            "status": status,
            "cards": cards_text.strip(),
            "pcm": pcm_text.strip(),
        }

    except Exception as e:
        return {
            "detected": False,
            "status": "microphone_check_failed",
            "error": str(e),
        }


def main():
    scd40_result = read_scd40_status()
    max30102_result = run_json_script(MAX30102_SCRIPT, timeout=MAX30102_TIMEOUT)

    output = {
        "scd40": scd40_result,
        "max30102": max30102_result.get("max30102", max30102_result),
        "mlx90614": {
            "detected": False,
            "address": "0x5A",
            "object_temperature_c": None,
            "ambient_temperature_c": None,
            "status": "sensor_not_detected_or_unplugged",
        },
        "camera": read_camera_status(),
        "microphone": read_microphone_status(),
        "timestamp": time.time(),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()