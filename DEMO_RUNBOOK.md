# Demo Runbook

## Purpose

This runbook provides the minimum commands and checks needed to operate the Intel DK2500 multimodal dashboard demo safely.

## Start

```bash
cd ~/Intel-Cup-Multimodal-Multi-AI-Agent-Layer
source .venv/bin/activate
./start_all.sh
```

Expected services:

- React dashboard frontend on port `3000`
- Flask plus SocketIO dashboard backend on port `5000`
- FastAPI agent backend on port `8000`

## Stop

```bash
cd ~/Intel-Cup-Multimodal-Multi-AI-Agent-Layer
./stop_all.sh
```

## Browser URLs

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:5000`
- Agent docs: `http://localhost:8000/docs`
- LAN frontend example: `http://192.168.137.84:3000`
- LAN backend example: `http://192.168.137.84:5000`

## Tested Endpoints

```bash
curl -I http://localhost:5000/api/camera_stream
curl -s http://localhost:5000/api/live_summary | python -m json.tool
curl -s http://localhost:5000/api/live_sensors | python -m json.tool
curl -s http://localhost:5000/api/microphone_level | python -m json.tool
curl -s http://localhost:5000/api/health_state | python -m json.tool
```

Key expected behavior:

- `/api/camera_stream` returns `200` when the camera is available
- `/api/live_summary` returns overall dashboard summary JSON
- `/api/live_sensors` returns current sensor availability and readings
- `/api/microphone_level` returns quiet/normal/loud/unavailable style status
- `/api/health_state` returns the current health classification payload

## Emergency Checks

If the dashboard does not look correct, run:

```bash
cd ~/Intel-Cup-Multimodal-Multi-AI-Agent-Layer
tail -n 80 logs/dashboard_backend.log
tail -n 80 logs/dashboard_frontend.log
tail -n 80 logs/agent_backend.log
```

If services are not responding, restart them:

```bash
./stop_all.sh
./start_all.sh
```

Quick port checks:

```bash
curl -I http://localhost:3000
curl -I http://localhost:5000/api/health_state
curl -I http://localhost:8000/docs
```

Hardware quick checks:

- SCD40/SCD41 should appear on `/dev/i2c-1` at `0x62`
- MAX30102 should appear at `0x57`
- Camera devices are expected on `/dev/video0` and/or `/dev/video1`
- USB microphone availability depends on ALSA device state

## Known Limitations

- This system is a prototype and not a medical-grade device.
- MAX30102 PPG/BPM output is not a clinically validated measurement.
- MLX90614 is currently not reliably detected and is shown as unavailable.
- SCD40 reads can have transient timing-related delays or errors.
- `/api/live_sensors` still reads hardware on demand; a background sensor cache would improve smoothness.
- Camera streaming uses MJPEG for demo simplicity rather than WebRTC.
- Microphone level is only an activity indicator, not speech recognition.

