"""REST endpoints for health state summary."""
import json
import subprocess
import sys
import time
from pathlib import Path

from flask import Blueprint, jsonify
from dashboard.backend.data_loader import store

health_bp = Blueprint("health", __name__)
LIVE_SENSOR_TIMEOUT_SECONDS = 15


class LiveSensorProbeError(RuntimeError):
    """Raised when the live sensor probe script cannot return usable JSON."""

    def __init__(self, payload, status_code):
        super().__init__(payload.get("error", "live_sensor_probe_failed"))
        self.payload = payload
        self.status_code = status_code



def _find_live_sensor_script():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tools" / "sensors" / "read_live_sensors.py"
        if candidate.is_file():
            return candidate
    return None



def _scd40_unavailable_payload(error_message=None):
    payload = {
        "co2_ppm": None,
        "temperature_c": None,
        "humidity_percent": None,
        "status": "temporarily_unavailable",
    }
    if error_message:
        payload["error"] = error_message
    return payload



def _max30102_unavailable_payload(error_message=None):
    payload = {
        "detected": False,
        "address": "0x57",
        "finger_present": False,
        "bpm": None,
        "status": "sensor_unavailable",
    }
    if error_message:
        payload["error"] = error_message
    return payload



def _mlx90614_unavailable_payload():
    return {
        "detected": False,
        "address": "0x5A",
        "object_temperature_c": None,
        "ambient_temperature_c": None,
        "status": "sensor_not_detected_or_unplugged",
    }



def _camera_unavailable_payload(error_message=None):
    payload = {
        "detected": False,
        "devices": [],
        "status": "camera_not_detected",
    }
    if error_message:
        payload["error"] = error_message
    return payload



def _microphone_unavailable_payload(error_message=None):
    payload = {
        "detected": False,
        "status": "microphone_unavailable",
    }
    if error_message:
        payload["error"] = error_message
    return payload



def _fallback_live_sensors_payload(error_message=None):
    return {
        "scd40": _scd40_unavailable_payload(error_message),
        "max30102": _max30102_unavailable_payload(error_message),
        "mlx90614": _mlx90614_unavailable_payload(),
        "camera": _camera_unavailable_payload(error_message),
        "microphone": _microphone_unavailable_payload(error_message),
        "timestamp": time.time(),
    }



def _run_live_sensor_probe():
    script_path = _find_live_sensor_script()
    if script_path is None:
        raise LiveSensorProbeError({"error": "live_sensor_script_not_found"}, 500)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=LIVE_SENSOR_TIMEOUT_SECONDS,
            cwd=str(script_path.parents[2]),
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise LiveSensorProbeError({
            "error": "live_sensor_timeout",
            "timeout_seconds": LIVE_SENSOR_TIMEOUT_SECONDS,
        }, 504)
    except Exception as exc:
        raise LiveSensorProbeError({
            "error": "live_sensor_execution_failed",
            "details": str(exc),
        }, 500)

    if result.returncode != 0:
        raise LiveSensorProbeError({
            "error": "live_sensor_script_failed",
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }, 500)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LiveSensorProbeError({
            "error": "live_sensor_invalid_json",
            "details": str(exc),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }, 500)



def _build_status_item(label, status, message):
    return {
        "label": label,
        "status": status,
        "message": message,
    }



def _normalize_number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def _build_air_quality_item(scd40):
    co2 = _normalize_number(scd40.get("co2_ppm"))
    if co2 is None:
        return _build_status_item("Air quality", "unavailable", "CO2 reading is not currently available.")
    if co2 < 1000:
        return _build_status_item("Air quality", "normal", "CO2 is within an acceptable range.")
    if co2 <= 1500:
        return _build_status_item("Air quality", "attention", "CO2 is elevated. Consider improving ventilation.")
    return _build_status_item("Air quality", "warning", "CO2 is high. Ventilation is recommended.")



def _build_temperature_item(scd40):
    temperature = _normalize_number(scd40.get("temperature_c"))
    if temperature is None:
        return _build_status_item("Temperature", "unavailable", "Temperature reading is not currently available.")
    if 20 <= temperature <= 28:
        return _build_status_item("Temperature", "normal", "Temperature is comfortable.")
    return _build_status_item("Temperature", "attention", "Temperature is outside the comfortable range.")



def _build_humidity_item(scd40):
    humidity = _normalize_number(scd40.get("humidity_percent"))
    if humidity is None:
        return _build_status_item("Humidity", "unavailable", "Humidity reading is not currently available.")
    if 30 <= humidity <= 70:
        return _build_status_item("Humidity", "normal", "Humidity is within a normal indoor range.")
    return _build_status_item("Humidity", "attention", "Humidity is outside the normal indoor range.")



def _build_ppg_item(max30102):
    if not max30102.get("detected"):
        return _build_status_item("PPG sensor", "unavailable", "PPG sensor is not currently available.")
    if max30102.get("finger_present"):
        return _build_status_item("PPG sensor", "active", "Finger is detected and pulse reading is active.")
    return _build_status_item("PPG sensor", "waiting", "No finger is detected, so pulse reading is not available.")



def _build_camera_item(camera):
    if camera.get("detected"):
        return _build_status_item("Camera", "online", "Camera is available.")
    return _build_status_item("Camera", "unavailable", "Camera is not currently available.")



def _read_microphone_level_payload():
    try:
        from dashboard.backend.routes.media_routes import _capture_microphone_level_payload

        payload, _status_code = _capture_microphone_level_payload()
        return payload
    except Exception as exc:
        return {
            "detected": False,
            "status": "microphone_unavailable",
            "error": str(exc),
        }



def _build_microphone_item(microphone_level_payload, live_sensor_microphone):
    if microphone_level_payload.get("detected"):
        status = microphone_level_payload.get("status") or "normal"
        if status == "quiet":
            return _build_status_item("Microphone", "quiet", "Audio level is quiet.")
        if status == "normal":
            return _build_status_item("Microphone", "normal", "Audio level is within a normal range.")
        if status == "loud":
            return _build_status_item("Microphone", "loud", "Audio level is loud.")

    if live_sensor_microphone.get("detected"):
        return _build_status_item("Microphone", "normal", "Microphone is online.")

    return _build_status_item("Microphone", "unavailable", "Microphone is not currently available.")



def _compute_overall_status(items):
    statuses = {item["status"] for item in items}
    if "warning" in statuses:
        return "warning"
    if "attention" in statuses:
        return "attention"
    if {"normal", "active", "online", "quiet", "loud", "waiting"}.isdisjoint(statuses):
        return "unavailable"
    return "normal"



def _build_overall_summary(overall_status, items):
    item_map = {item["label"]: item for item in items}
    if overall_status == "warning":
        intro = "Room condition needs attention."
    elif overall_status == "attention":
        intro = "Room condition is mostly stable, with a few items to watch."
    elif overall_status == "unavailable":
        intro = "Live sensor data is currently limited."
    else:
        intro = "Room condition looks normal."

    air = item_map["Air quality"]
    temperature = item_map["Temperature"]
    humidity = item_map["Humidity"]
    ppg = item_map["PPG sensor"]
    camera = item_map["Camera"]
    microphone = item_map["Microphone"]

    details = []

    if air["status"] == "normal":
        details.append("CO2 is acceptable")
    elif air["status"] == "attention":
        details.append("CO2 is elevated")
    elif air["status"] == "warning":
        details.append("CO2 is high")
    else:
        details.append("CO2 is unavailable")

    if temperature["status"] == "normal" and humidity["status"] == "normal":
        details.append("Temperature and humidity are comfortable")
    else:
        if temperature["status"] == "attention":
            details.append("Temperature is outside the comfortable range")
        elif temperature["status"] == "unavailable":
            details.append("Temperature is unavailable")

        if humidity["status"] == "attention":
            details.append("Humidity is outside the normal indoor range")
        elif humidity["status"] == "unavailable":
            details.append("Humidity is unavailable")

    if ppg["status"] == "active":
        details.append("PPG reading is active")
    elif ppg["status"] == "waiting":
        details.append("PPG is waiting for finger placement")
    elif ppg["status"] == "unavailable":
        details.append("PPG is unavailable")

    if camera["status"] == "online" and microphone["status"] != "unavailable":
        details.append("Camera and microphone are online")
    elif camera["status"] == "online":
        details.append("Camera is online")
    elif microphone["status"] != "unavailable":
        details.append("Microphone is online")
    else:
        details.append("Camera and microphone are unavailable")

    return f"{intro} " + ". ".join(details) + "."



def _build_live_summary_payload(live_sensors_payload):
    scd40 = live_sensors_payload.get("scd40") or {}
    max30102 = live_sensors_payload.get("max30102") or {}
    camera = live_sensors_payload.get("camera") or {}
    microphone = live_sensors_payload.get("microphone") or {}
    microphone_level_payload = _read_microphone_level_payload()

    items = [
        _build_air_quality_item(scd40),
        _build_temperature_item(scd40),
        _build_humidity_item(scd40),
        _build_ppg_item(max30102),
        _build_camera_item(camera),
        _build_microphone_item(microphone_level_payload, microphone),
    ]
    overall_status = _compute_overall_status(items)

    return {
        "overall_status": overall_status,
        "summary": _build_overall_summary(overall_status, items),
        "items": items,
        "timestamp": live_sensors_payload.get("timestamp") or time.time(),
    }


@health_bp.route("/api/health_state", methods=["GET"])
def health_state():
    """Aggregated health state across all predictions."""
    exp = store.get_experiment(store.get_active_experiment_id())
    counts = store.get_counts()

    return jsonify({
        **counts,
        "active_experiment_id": store.get_active_experiment_id(),
        "active_experiment_label": exp.get("config_label", "") if exp else "",
        "accuracy": round(float(exp["test_accuracy"]) * 100, 2) if exp else None,
        "f1_macro": round(float(exp["test_f1_macro"]) * 100, 2) if exp else None,
    })


@health_bp.route("/api/health_history", methods=["GET"])
def health_history():
    """Returns the full list of predictions as a history log."""
    predictions = store.get_predictions_list()
    return jsonify({
        "total": len(predictions),
        "predictions": predictions,
    })


@health_bp.route("/api/live_sensors", methods=["GET"])
def live_sensors():
    """Runs the on-device sensor probe script and returns its JSON output."""
    try:
        payload = _run_live_sensor_probe()
    except LiveSensorProbeError as exc:
        return jsonify(exc.payload), exc.status_code

    return jsonify(payload)


@health_bp.route("/api/live_summary", methods=["GET"])
def live_summary():
    """Returns a user-facing summary derived from the current live sensor payload."""
    try:
        live_sensors_payload = _run_live_sensor_probe()
    except LiveSensorProbeError as exc:
        live_sensors_payload = _fallback_live_sensors_payload(exc.payload.get("error"))

    return jsonify(_build_live_summary_payload(live_sensors_payload))

