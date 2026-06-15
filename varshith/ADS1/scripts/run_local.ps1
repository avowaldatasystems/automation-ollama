# Full local stack: dependencies check, micro-APIs, Streamlit. No Docker.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Create the venv first: python -m venv .venv; pip install -r requirements.txt"
}

.\.venv\Scripts\Activate.ps1

Write-Host "=== Step 1: Native dependencies (MySQL, Qdrant, Ollama) - no Docker ==="
& "$PSScriptRoot\start_qdrant.ps1"
& "$PSScriptRoot\start_dependencies.ps1"

Write-Host ""
Write-Host "=== Step 2: Micro-APIs ==="
& "$PSScriptRoot\start_apis.ps1"

Start-Sleep -Seconds 4

Write-Host ""
Write-Host "=== Step 3: Streamlit UI ==="
Write-Host "Open http://127.0.0.1:8501 after Streamlit starts."
streamlit run streamlit_app.py
