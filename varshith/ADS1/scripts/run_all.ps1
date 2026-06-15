# Start all micro-APIs and Streamlit (no Docker).
# For dependency checks first, use: .\scripts\run_local.cmd
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Create the venv first: python -m venv .venv"
}

.\.venv\Scripts\Activate.ps1

& "$PSScriptRoot\start_apis.ps1"

Start-Sleep -Seconds 4

Write-Host "Starting Streamlit ..."
streamlit run streamlit_app.py
