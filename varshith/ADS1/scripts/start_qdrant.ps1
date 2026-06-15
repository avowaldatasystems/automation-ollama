# Start Qdrant natively on Windows (no Docker).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$config = Join-Path $root "tools\qdrant\config.yaml"
$qdrantExe = Join-Path $root "tools\qdrant\qdrant.exe"

function Test-QdrantUp {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:6333" -UseBasicParsing -TimeoutSec 3
        return $true
    } catch {
        return $false
    }
}

if (Test-QdrantUp) {
    Write-Host "Qdrant is already running at http://127.0.0.1:6333"
    exit 0
}

if (-not (Test-Path $qdrantExe)) {
    Write-Host "qdrant.exe not found. Download from:"
    Write-Host "https://github.com/qdrant/qdrant/releases/download/v1.12.6/qdrant-x86_64-pc-windows-msvc.zip"
    Write-Host "Extract to: $qdrantExe"
    exit 1
}

New-Item -ItemType Directory -Force -Path "$root\data\qdrant_storage" | Out-Null

Write-Host "Starting Qdrant: $qdrantExe"
Start-Process -FilePath $qdrantExe -ArgumentList @("--config-path", $config) -WorkingDirectory (Split-Path $qdrantExe) -WindowStyle Minimized

for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 2
    if (Test-QdrantUp) {
        Write-Host "Qdrant started at http://127.0.0.1:6333"
        exit 0
    }
}

Write-Warning "Qdrant started but port 6333 is not responding yet. Wait a few seconds and retry."
