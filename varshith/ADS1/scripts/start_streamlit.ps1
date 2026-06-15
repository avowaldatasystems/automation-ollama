# Start Employee Office RAG UI (API-only, no Docker).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual env missing. Run: python -m venv .venv; pip install -r requirements.txt"
}

$check = & $python -c "from app.api_client import OfficeRagClient; c=OfficeRagClient(); exit(0 if c.all_services_ok() else 1)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warning "One or more micro-APIs are offline. Start them first:"
    Write-Host "  .\scripts\start_apis.cmd"
    Write-Host ""
}

Write-Host "Starting Streamlit at http://127.0.0.1:8501"
& $python -m streamlit run streamlit_app.py --server.port 8501
