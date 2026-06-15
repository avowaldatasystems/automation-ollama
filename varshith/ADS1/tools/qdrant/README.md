# Local Qdrant (no Docker)

`qdrant.exe` v1.12.6 is installed here for native Windows use.

Start:

```powershell
..\..\scripts\start_qdrant.ps1
```

Re-download manually:

```powershell
$url = "https://github.com/qdrant/qdrant/releases/download/v1.12.6/qdrant-x86_64-pc-windows-msvc.zip"
Invoke-WebRequest -Uri $url -OutFile qdrant-download.zip
Expand-Archive qdrant-download.zip -DestinationPath . -Force
```
