# Start all local micro-APIs (one service per domain).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Create the venv first: python -m venv .venv"
}

$services = @(
    @{ Name = "health";    Module = "app.apis.health_api:app";    Port = 8000 },
    @{ Name = "employees"; Module = "app.apis.employees_api:app"; Port = 8001 },
    @{ Name = "ingestion"; Module = "app.apis.ingestion_api:app"; Port = 8002 },
    @{ Name = "retrieval"; Module = "app.apis.retrieval_api:app"; Port = 8003 },
    @{ Name = "chat";      Module = "app.apis.chat_api:app";      Port = 8004 }
)

foreach ($svc in $services) {
    $title = "ADS1 $($svc.Name) API :$($svc.Port)"
    Write-Host "Starting $title ..."
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd '$root'; $python -m uvicorn $($svc.Module) --reload --host 127.0.0.1 --port $($svc.Port)"
    ) -WindowStyle Normal
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "APIs started:"
Write-Host "  Health     http://127.0.0.1:8000/docs"
Write-Host "  Employees  http://127.0.0.1:8001/docs"
Write-Host "  Ingestion  http://127.0.0.1:8002/docs"
Write-Host "  Retrieval  http://127.0.0.1:8003/docs"
Write-Host "  Chat       http://127.0.0.1:8004/docs"
