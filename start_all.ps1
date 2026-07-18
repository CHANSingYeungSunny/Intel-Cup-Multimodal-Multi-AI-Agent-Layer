# One-click launcher for the full system (Windows)
# Usage: .\start_all.ps1

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = "$root\intel multimodal (AI_Agent_Single_layer)\.venv\Scripts\Activate.ps1"
$dashboard = "$root\intel multimodal (AI_Agent_Single_layer)\intel multimodal (dashboard_and_alert_layer)\dashboard_and_alert_layer"
$frontend = "$dashboard\dashboard\frontend"

if (-not (Test-Path $venv)) { Write-Host "ERROR: venv not found at $venv"; exit 1 }

Write-Host "Activating venv..."

# Terminal 1 — AI Agent Backend (:8000)
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "& '$venv'; cd '$root'; Write-Host '=== Multi AI Agent Backend (:8000) ==='; python run.py"

# Terminal 2 — Dashboard Backend (:5000)
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "& '$venv'; cd '$dashboard'; `$env:AGENT_API_URL='http://localhost:8000/api/v1'; Write-Host '=== Dashboard Backend (:5000) ==='; python run.py --no-agent"

# Terminal 3 — React Frontend (:3000)
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$frontend'; Write-Host '=== React Frontend (:3000) ==='; npm start"

Write-Host ""
Write-Host "All 3 services launching in separate windows..."
Write-Host "  Agent backend:     http://localhost:8000/docs"
Write-Host "  Dashboard backend: http://localhost:5000"
Write-Host "  Dashboard UI:      http://localhost:3000"
Write-Host ""
Write-Host "To stop: close the 3 PowerShell windows."
