# Intel Cup 2025 Final Report Draft

## Project Title

DK2500 Multimodal Health Monitoring Dashboard with Multi-AI Agent Support

## Abstract

This project implements a prototype multimodal health monitoring system on the Intel DK2500 platform. The system combines environmental sensing, physiological sensing, camera streaming, microphone activity detection, AI-based health inference, and a web dashboard for live monitoring. The deployed stack includes a Flask plus SocketIO dashboard backend on port `5000`, a React frontend on port `3000`, and a FastAPI agent backend on port `8000`. The platform is accessible over the local network for demo use and is designed to remain stable even when some sensors are temporarily unavailable.

The implemented hardware integration uses the Intel DK2500 directly through Linux device interfaces rather than an external ESP32 or Raspberry Pi Pico bridge. The current working live hardware path includes `/dev/i2c-1` for I2C sensors, USB webcam detection on `/dev/video0` and `/dev/video1`, and ALSA-based USB microphone detection. The dashboard provides a current multimodal summary, live sensor cards, a large MJPEG camera panel, microphone level status, and health-state visualization. Safe fallback handling is included so that missing or unstable sensors do not crash the system.

This report documents the implemented system architecture, hardware integration status, dashboard design, endpoint behavior, testing results, limitations, and future improvement opportunities for the Intel Cup submission.

## 1. Introduction and Motivation

Multimodal health monitoring is valuable because no single signal is sufficient to describe a person's condition in real time. Environmental air quality, body-related optical signals, temperature sensing, camera observations, and audio activity each provide partial context. By integrating multiple modalities into one dashboard, the system can provide a more informative prototype monitoring interface for demos and future research.

The Intel DK2500 platform offers a compact edge-computing base for this type of integration. In this project, it serves as the host for sensor access, service orchestration, browser-based monitoring, and AI-agent-driven health interpretation. A major design goal of the final prototype is safe demo reliability: if one sensor is missing or unstable, the rest of the system should continue running and display honest status information instead of failing.

## 2. Problem Statement

The project goal is to build a prototype dashboard that can:

1. Collect live data from multiple hardware sources connected to the Intel DK2500.
2. Present those data in a unified, browser-accessible dashboard.
3. Connect live sensing outputs with an existing AI-based health monitoring pipeline.
4. Remain usable during demonstrations even when some hardware devices are absent, slow, or unreliable.

The system is not intended to be a medical-grade device. Instead, it demonstrates practical multimodal integration, real-time visualization, and safe prototype behavior on edge hardware.

## 3. System Overview

The implemented deployment contains three main software services:

1. React dashboard frontend on port `3000`.
2. Flask plus SocketIO dashboard backend on port `5000`.
3. FastAPI agent backend on port `8000`.

The services are started together through `./start_all.sh`, which supports SSH-oriented operation and NVM-based Node startup, and are stopped with `./stop_all.sh`. During demos, the dashboard can be opened from the LAN, for example:

- `http://192.168.137.84:3000`

At a high level, the dashboard backend aggregates live hardware state and health-simulation outputs, while the React frontend displays status cards, charts, and media panels. The FastAPI agent backend provides the AI-agent layer for health interpretation.

## 4. Hardware Platform

### 4.1 Intel DK2500

The Intel DK2500 is the central hardware platform used for this submission. It runs the backend and frontend services directly and accesses sensors through standard Linux interfaces.

### 4.2 I2C Sensor Integration

Direct DK2500 I2C is used in the current implementation. No ESP32 or Pico bridge is required. The working I2C bus is:

- `/dev/i2c-1`

### 4.3 USB Webcam

The camera subsystem currently detects:

- `/dev/video0`
- `/dev/video1`

The dashboard uses these devices for snapshot and MJPEG streaming support.

### 4.4 USB Microphone

When available, a USB microphone is detected through ALSA and sampled for audio activity level estimation. The microphone feature is used as a loudness/activity indicator rather than a speech-recognition subsystem.

## 5. Software Architecture

### 5.1 Flask Dashboard Backend

The Flask plus SocketIO backend on port `5000` is responsible for:

- Serving dashboard-oriented APIs.
- Providing live sensor and media endpoints.
- Exposing health-state and summary data to the frontend.
- Coordinating dashboard-safe fallback behavior when sensors are unavailable.

Important endpoints include:

- `/api/live_summary`
- `/api/live_sensors`
- `/api/health_state`
- `/api/camera_snapshot`
- `/api/camera_stream`
- `/api/microphone_level`

### 5.2 React Dashboard Frontend

The React frontend on port `3000` displays the live monitoring interface. The current dashboard layout includes:

- Current Multimodal Status summary card.
- Live Sensor Snapshot card.
- Large live camera preview panel.
- Microphone level and status card.
- Health-state card and existing health dashboard sections.

The frontend is designed to avoid crashes when upstream data are temporarily unavailable.

### 5.3 FastAPI Agent Backend

The FastAPI backend on port `8000` serves the existing AI-agent layer. It remains separate from the dashboard backend so that sensing, visualization, and AI logic are cleanly divided.

### 5.4 Startup and Shutdown Scripts

The repository includes operational scripts for demo control:

- `./start_all.sh`
- `./stop_all.sh`

These scripts simplify service management over SSH and reduce operator friction during live demonstrations.

## 6. Sensor and Media Integration

### 6.1 SCD40 / SCD41 Environmental Sensor

The SCD40/SCD41 is working at I2C address `0x62`. The system currently reads:

- CO2 concentration in ppm
- Temperature
- Humidity

Because SCD40 measurements have timing requirements and can show transient read delays, retry and fallback logic was added to reduce failures and prevent dashboard crashes.

### 6.2 MAX30102 Physiological Sensor

The MAX30102 is working at I2C address `0x57`. The implementation currently:

- Detects the sensor
- Reads red and IR raw values
- Computes average values
- Reports finger/PPG status
- Shows BPM only when the signal is reliable

This is appropriate for prototype visualization, but BPM should not be described as clinically accurate without formal validation.

### 6.3 MLX90614 / GY906 Infrared Temperature Sensor

The MLX90614 infrared temperature sensor is currently not reliably detected in the deployed setup. Instead of forcing unstable output, the dashboard safely presents this source as unavailable. This preserves demo stability and truthfulness.

### 6.4 Camera Integration

The dashboard supports both snapshot and stream-based camera access:

- `/api/camera_snapshot`
- `/api/camera_stream`

For demo use, the live camera now uses MJPEG streaming with:

- `multipart/x-mixed-replace`
- requested resolution `1280x720`
- target frame rate `6 FPS`
- fallback camera detection order `/dev/video0` then `/dev/video1`

The frontend presents this stream in a large 16:9 preview panel for visibility during demonstrations. Snapshot support is retained as a fallback path.

### 6.5 Microphone Integration

The `/api/microphone_level` endpoint records a short ALSA sample and reports activity as quiet, normal, loud, or unavailable. This feature is useful for indicating sound presence and microphone readiness, but it is not speech recognition and should not be presented as semantic audio analysis.

## 7. Dashboard Design

The dashboard was refined to support live demonstration needs, with emphasis on readability and resilience.

### 7.1 Current Multimodal Status

This card provides a concise textual summary of the overall system condition and highlights the state of major sensing channels.

### 7.2 Live Sensor Snapshot

This section gives a quick structured overview of current hardware readings and per-sensor availability.

### 7.3 Large Camera Panel

The camera preview is displayed as a large live panel with a 16:9 aspect ratio. This improves visibility during presentations and makes the system feel live rather than static.

### 7.4 Microphone Status

The microphone card provides a simple live loudness indicator and device status, which helps demonstrate that the audio input path is active.

### 7.5 Health Dashboard Sections

Existing health state cards and dashboard visualizations were preserved so that the new live hardware integration does not disrupt prior project functionality.

## 8. Testing and Validation

The system was validated with practical endpoint checks and browser-based verification.

### 8.1 Service Startup Validation

The services were started using:

```bash
./start_all.sh
```

and stopped with:

```bash
./stop_all.sh
```

Successful startup confirms that the React frontend, Flask backend, and FastAPI agent backend can be launched together in the current deployment flow.

### 8.2 Endpoint Checks

Representative checks include:

```bash
curl -I http://localhost:5000/api/camera_stream
curl -s http://localhost:5000/api/live_summary | python -m json.tool
curl -s http://localhost:5000/api/live_sensors | python -m json.tool
curl -s http://localhost:5000/api/microphone_level | python -m json.tool
curl -s http://localhost:5000/api/health_state | python -m json.tool
```

These checks confirm that the primary live dashboard APIs respond and that the stream endpoint can be opened successfully.

### 8.3 Browser Verification

The dashboard was also checked in the browser from the LAN-facing URL:

- `http://192.168.137.84:3000`

Verification focused on:

- Dashboard page loading
- Large camera stream visibility
- Live summary rendering
- Sensor cards rendering
- Microphone card rendering
- Health dashboard rendering
- No React crash during page load

### 8.4 Stability-Oriented Validation

Additional validation emphasized fallback behavior. The dashboard was checked to ensure that missing devices such as the MLX90614 do not cause the page or APIs to fail.

## 9. Results

The current implemented system successfully demonstrates the following:

- Live DK2500-hosted multimodal dashboard
- Working SCD40/SCD41 environmental sensing
- Working MAX30102 detection and signal-status reporting
- Graceful unavailable state for MLX90614
- Live MJPEG camera stream in a large preview panel
- Snapshot fallback retention
- Live microphone activity/status endpoint
- Browser-accessible dashboard over the LAN
- Continued operation when some sensors are missing or temporarily unstable

These results show that the project has moved beyond a static AI pipeline and now includes practical edge-device integration suitable for an Intel Cup demo.

## 10. Limitations

The following limitations should be stated clearly:

1. This is a prototype and not a medical-grade device.
2. MAX30102 PPG and BPM values should not be claimed as clinically accurate.
3. The MLX90614 infrared temperature sensor was not reliably detected and is currently shown as unavailable.
4. I2C sensors, especially the SCD40, may exhibit transient read delays or timing-sensitive behavior.
5. The current `/api/live_sensors` implementation still reads hardware on demand instead of using a persistent background cache.
6. Camera streaming uses MJPEG for demo simplicity rather than WebRTC or a lower-latency production streaming stack.
7. Microphone level is only an activity/loudness indicator, not speech recognition or medical audio diagnosis.

## 11. Future Work

Recommended next steps include:

1. Add a background sensor cache/service so hardware reads are decoupled from request timing.
2. Improve I2C scheduling and retry coordination for smoother SCD40 operation.
3. Perform more rigorous validation of MAX30102-derived BPM behavior.
4. Improve camera device selection and recovery logic for more varied hardware setups.
5. Explore WebRTC or more efficient streaming approaches for future low-latency video.
6. Expand the AI interpretation layer for richer contextual reasoning over live multimodal signals.

## 12. Conclusion

This Intel Cup prototype demonstrates a working multimodal dashboard system built on the Intel DK2500. It integrates live environmental sensing, physiological signal monitoring, camera streaming, microphone activity detection, AI-agent support, and browser-based visualization into a single edge-hosted demo platform.

The strongest practical outcome of the current implementation is not only that individual sensors work, but that the total system remains usable when some hardware components are unavailable or imperfect. That stability, combined with truthful limitation handling and LAN-accessible live visualization, makes the project suitable for final demonstration and submission as a prototype multimodal health monitoring platform.

## Appendix A. Useful Commands

```bash
cd ~/Intel-Cup-Multimodal-Multi-AI-Agent-Layer
source .venv/bin/activate
./start_all.sh
./stop_all.sh
```

## Appendix B. Dashboard URLs

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:5000`
- Agent backend: `http://localhost:8000/docs`
- LAN frontend example: `http://192.168.137.84:3000`

## Appendix C. Important Endpoints

- `GET /api/live_summary`
- `GET /api/live_sensors`
- `GET /api/health_state`
- `GET /api/camera_snapshot`
- `GET /api/camera_stream`
- `GET /api/microphone_level`

## Appendix D. Known Issues

- MLX90614 is currently unavailable in the deployed setup.
- SCD40 reads may occasionally require retry handling.
- Camera streaming is optimized for demo stability rather than minimal latency.
- Microphone availability may vary depending on ALSA device state.

