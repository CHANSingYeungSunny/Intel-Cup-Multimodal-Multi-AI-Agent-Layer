#!/usr/bin/env python3

import json
import sys
import time
import glob
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


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
    scd40_result = run_json_script("tools/sensors/test_scd40.py", timeout=12)
    max30102_result = run_json_script("tools/sensors/test_max30102.py", timeout=8)

    output = {
        "scd40": scd40_result.get("scd40", scd40_result),
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
