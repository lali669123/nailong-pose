param(
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CacheDir = Join-Path $ProjectRoot ".cache"
New-Item -ItemType Directory -Force -Path (Join-Path $CacheDir "mpl"), (Join-Path $CacheDir "tmp") | Out-Null

$env:MPLCONFIGDIR = Join-Path $CacheDir "mpl"
$env:TMP = Join-Path $CacheDir "tmp"
$env:TEMP = Join-Path $CacheDir "tmp"
$env:NAILONG_MODE = if ($Release) { "release" } else { "studio" }

Set-Location $ProjectRoot
Write-Host "==================================================" -ForegroundColor Yellow
if ($Release) {
    Write-Host "  奶龙家人鉴定 · 精简发布版 (Release) 正在启动..." -ForegroundColor Green
} else {
    Write-Host "  奶龙家人鉴定 · AI Pose Studio 正在启动..." -ForegroundColor Yellow
    Write-Host "  提示: 加 -Release 参数可启动精简发布版（隐藏训练功能）" -ForegroundColor DarkGray
}
Write-Host "  请在浏览器打开: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Yellow

& (Join-Path $ProjectRoot ".venv\Scripts\python.exe") -m uvicorn src.server:app --host 127.0.0.1 --port 8000
