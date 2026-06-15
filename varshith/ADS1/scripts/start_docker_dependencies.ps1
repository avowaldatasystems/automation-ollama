# Optional Docker dependency stack for MySQL and Qdrant.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual env missing. Run: python -m venv .venv; pip install -r requirements.txt"
}

Write-Host "Starting Docker MySQL and Qdrant..."
docker compose up -d mysql qdrant

Write-Host "Waiting for dependencies and creating schema..."
& "$PSScriptRoot\start_dependencies.ps1"
