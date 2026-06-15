# Quick verification: native deps + all micro-APIs (no Docker).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = ".\.venv\Scripts\python.exe"

Write-Host "System check..."
& $py scripts\check_system.py
Write-Host ""
Write-Host "API check..."
& $py -c @"
from app.api_client import OfficeRagClient
import json
c = OfficeRagClient()
h = c.health_all()
print(json.dumps(h, indent=2))
if not c.all_services_ok():
    raise SystemExit(1)
print('All micro-APIs OK')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "Start APIs: .\scripts\start_apis.cmd"
    exit 1
}

Write-Host ""
Write-Host "Ready. Open http://127.0.0.1:8501"
