# Start / verify native local dependencies (MySQL, Qdrant). No Docker.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual env missing. Run: python -m venv .venv; pip install -r requirements.txt"
}

Write-Host "Local dependencies only (no Docker)."
Write-Host ""
Write-Host "Before continuing, ensure these are installed and running on Windows:"
Write-Host "  1. MySQL Server 8  (service MySQL80, port 3306)"
Write-Host "  2. Qdrant          (qdrant.exe, port 6333)  -> optional helper: .\scripts\start_qdrant.cmd"
Write-Host "  3. Ollama          (port 11434, models pulled)"
Write-Host ""

function Test-ServiceCheck {
    param([string]$Name, [string]$Script)
    & $python -c $Script
    return $LASTEXITCODE -eq 0
}

$mysqlReady = $false
Write-Host "Waiting for MySQL..."
for ($i = 0; $i -lt 45; $i++) {
    if (Test-ServiceCheck "mysql" "from app.diagnostics import check_mysql; import sys; sys.exit(0 if check_mysql()['ok'] else 1)") {
        $mysqlReady = $true
        Write-Host "MySQL is ready."
        break
    }
    Start-Sleep -Seconds 2
}
if (-not $mysqlReady) {
    Write-Warning "MySQL not reachable. Install MySQL Server, start service MySQL80, set MYSQL_* in .env"
}

$qdrantReady = $false
Write-Host "Waiting for Qdrant..."
for ($i = 0; $i -lt 45; $i++) {
    if (Test-ServiceCheck "qdrant" "from app.diagnostics import check_qdrant; import sys; sys.exit(0 if check_qdrant()['ok'] else 1)") {
        $qdrantReady = $true
        Write-Host "Qdrant is ready."
        break
    }
    Start-Sleep -Seconds 2
}
if (-not $qdrantReady) {
    Write-Warning "Qdrant not reachable. Download qdrant.exe (see README) and run .\scripts\start_qdrant.cmd"
}

if ($mysqlReady) {
    Write-Host "Creating / updating MySQL schema..."
    & $python scripts\init_db.py
}

Write-Host ""
Write-Host "System check:"
& $python scripts\check_system.py
