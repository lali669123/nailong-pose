$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CacheRoot = Join-Path $ProjectRoot ".cache"
$BuildDir = Join-Path $CacheRoot "pyinstaller_build"
$SpecDir = Join-Path $CacheRoot "pyinstaller_spec"
$StageConfigDir = Join-Path $CacheRoot "release_config"
$DistDir = Join-Path $ProjectRoot "release"
$AppName = "奶龙家人鉴定"
$IconPath = Join-Path $ProjectRoot "assets\app_icon.ico"

# Pin ALL temp / PyInstaller caches strictly to <ProjectRoot>\.cache
New-Item -ItemType Directory -Force -Path $BuildDir, $SpecDir, $StageConfigDir, (Join-Path $CacheRoot "tmp"), $DistDir | Out-Null
$env:TMP = Join-Path $CacheRoot "tmp"
$env:TEMP = Join-Path $CacheRoot "tmp"
$env:PYINSTALLER_CONFIG_DIR = Join-Path $CacheRoot "pyinstaller_cfg"

if (-not (Test-Path (Join-Path $ProjectRoot "models\pose_clf.joblib"))) {
    throw "未找到已训练的模型 models\pose_clf.joblib，请先训练模型再打包！"
}
if (-not (Test-Path $IconPath)) {
    throw "未找到应用图标 $IconPath"
}

Write-Host "==================================================" -ForegroundColor Yellow
Write-Host "  [1/4] 生成纯净发布版配置 (mode=release, threshold=0.75)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Yellow

$ReleaseYaml = Join-Path $StageConfigDir "poses.yaml"
& $PythonExe -c @"
import yaml
from pathlib import Path

src = Path(r'$ProjectRoot\config\poses.yaml')
dst = Path(r'$ReleaseYaml')
cfg = yaml.safe_load(src.read_text(encoding='utf-8'))
cfg.setdefault('app', {})['mode'] = 'release'
cfg.setdefault('recognition', {})['confidence_threshold'] = 0.75
dst.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
print('Generated release poses.yaml -> mode:', cfg['app']['mode'], 'threshold:', cfg['recognition']['confidence_threshold'])
"@

# Clean previous build outputs cleanly
$AppDistDir = Join-Path $DistDir $AppName
if (Test-Path $AppDistDir) { Remove-Item -Recurse -Force $AppDistDir }
$ZipPath = Join-Path $DistDir "${AppName}_分享版.zip"
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }

Write-Host "==================================================" -ForegroundColor Yellow
Write-Host "  [2/4] 正在使用 PyInstaller 打包独立桌面版 .exe..." -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Yellow

Set-Location $ProjectRoot
& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name $AppName `
    --icon $IconPath `
    --distpath $DistDir `
    --workpath $BuildDir `
    --specpath $SpecDir `
    --add-data "${ReleaseYaml};config" `
    --add-data "${ProjectRoot}\web;web" `
    --add-data "${ProjectRoot}\assets\images;assets\images" `
    --add-data "${ProjectRoot}\assets\audio;assets\audio" `
    --add-data "${ProjectRoot}\models\pose_clf.joblib;models" `
    --collect-data mediapipe `
    --collect-data webview `
    --collect-submodules uvicorn `
    --collect-submodules starlette `
    --collect-submodules fastapi `
    --collect-submodules webview `
    --hidden-import sklearn.neural_network._multilayer_perceptron `
    --hidden-import sklearn.ensemble._forest `
    --hidden-import sklearn.utils._cython_blas `
    --hidden-import sklearn.neighbors._typedefs `
    --hidden-import sklearn.utils._typedefs `
    --hidden-import sklearn.tree._utils `
    --hidden-import scipy._lib.messagestream `
    --hidden-import websockets `
    --hidden-import websockets.legacy `
    --hidden-import websockets.legacy.server `
    --hidden-import websockets.asyncio `
    --hidden-import websockets.asyncio.server `
    --exclude-module pandas `
    --exclude-module pytest `
    --exclude-module edge_tts `
    --exclude-module IPython `
    --exclude-module jupyter `
    "src\desktop_app.py"

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller 打包失败，退出码：$LASTEXITCODE"
}

Write-Host "==================================================" -ForegroundColor Yellow
Write-Host "  [3/4] 创建 captures 目录与《使用说明.txt》" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Yellow

New-Item -ItemType Directory -Force -Path (Join-Path $AppDistDir "captures") | Out-Null

$ReadmeContent = @"
==================================================
      奶龙家人鉴定 · 奶蛙动作识别（纯净分享版）
==================================================

【怎么玩】
1. 解压整个文件夹到任意目录（不要直接在压缩包里双击运行）。
2. 确保电脑有摄像头并开启音量，双击「奶龙家人鉴定.exe」即可启动独立桌面窗口！
3. 点击画面下方大号金色「▶ 开始游戏」，3 秒倒计时后站在镜头前摆出任意奶蛙动作并保持 0.5 秒，即可弹出你和奶蛙的双栏合照与魔性原声！
4. 彩蛋：摆出「捧腹大笑奶蛙」动作时，有 30% 概率爆出 SSR 隐藏款神秘奶蛙（连非 8 次必出保底）！
5. 玩完点击「⏹ 停止游戏」，即可生成你的「奶家人纯度鉴定报告」并一键保存鉴定证书。

【合照存在哪里？】
- 每次鉴定成功拍下的同框合照和保存的鉴定证书，都会自动保存在本文件夹下的 captures\ 目录里，随时可取图发朋友圈！

【常见问题】
- 首次双击若提示 Windows 防护弹窗，点击「更多信息」->「仍要运行」即可（本软件完全离线运行在本地，不会上传任何摄像头画面）。
- 若画面提示“等待摄像头”，请检查是否被微信/腾讯会议等软件占用摄像头，关闭后程序会自动重连。
"@
Set-Content -Path (Join-Path $AppDistDir "使用说明.txt") -Value $ReadmeContent -Encoding UTF8

Write-Host "==================================================" -ForegroundColor Yellow
Write-Host "  [4/4] 压缩生成可分享 zip 包..." -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Yellow

Compress-Archive -Path $AppDistDir -DestinationPath $ZipPath -CompressionLevel Optimal

$ExeSize = [math]::Round((Get-Item (Join-Path $AppDistDir "${AppName}.exe")).Length / 1MB, 2)
$ZipSize = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)
Write-Host "打包完成！" -ForegroundColor Green
Write-Host "  独立程序: $AppDistDir\${AppName}.exe ($ExeSize MB)" -ForegroundColor Green
Write-Host "  分享压缩包: $ZipPath ($ZipSize MB)" -ForegroundColor Green
