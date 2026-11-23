# 奶龙家人鉴定 · 奶蛙动作识别（Nailong Pose Studio）

基于 **MediaPipe Pose + scikit-learn + FastAPI + 原生 HTML5/Canvas** 的本地离线全身动作识别互动应用。站在电脑摄像头前摆出经典「奶蛙」姿势并保持约 0.5 秒，即可触发霓虹骨骼蓄力特效、弹出你与奶蛙的双栏同框认证海报、播放专属魔性原声，并在本局结束时生成「奶龙家人血统鉴定报告」与可下载证书。

> **隐私与离线说明**：全部摄像头推理、动作分类、海报合成与证书保存均在**本机离线运行**（默认仅监听 `127.0.0.1`），不会向任何外部服务器上传摄像头画面或个人数据。

---

## 核心特性

- **实时骨骼识别与不变特征提取**：MediaPipe Pose 提取 33 个身体关键点，经骨盆中心平移与躯干尺度归一化后构建 46 维几何/坐标特征（[src/features.py](src/features.py)），对站位远近与身高差异保持鲁棒。
- **防泄露自训练闭环（Studio 模式）**：网页端内置 3 秒倒计时录制与样本管理，一键对比 **RandomForest** 与 **StandardScaler + MLP**，优先使用 `StratifiedGroupKFold` 防止相邻帧时序泄露，自动导出混淆矩阵并热加载模型（[src/train.py](src/train.py)）。
- **防误触状态机**：滑窗概率平滑 + 置信度门限 + 蓄力保持 + 瞬断容错 + 触发冷却（[src/state_machine.py](src/state_machine.py)）。
- **10 套奶蛙图鉴 + SSR 彩虹隐藏款**：支持 9 个常规动作 + 自然待机负样本 + 「捧腹大笑奶蛙」概率掉落 **SSR 彩虹奶蛙**（含连非保底机制）。
- **双栏同框海报 & 血统鉴定报告**：触发瞬间合成 `1280×680` 双栏对比海报（兼容 4K 与 RGBA 透明通道，[src/capture.py](src/capture.py)），结束游戏后弹出纯度评分动画并支持一键保存高清 PNG 鉴定证书。
- **Studio / Release 双模式 & 独立桌面打包**：Release 模式自动隐藏采集/训练界面并以 `403` 锁定训练接口；内置 PyInstaller + `pywebview` 打包脚本，可生成一键双击运行的 Windows 独立分享版。
- **可选硬件联动**：预留非阻塞 HTTP 音频触发桥接（[src/esp32_bridge.py](src/esp32_bridge.py)），可在 `config/poses.yaml` 中按需开启以对接 ESP32 局域网功放喇叭（默认关闭）。

---

## 快速开始

### 方式一：直接下载 Windows 免安装桌面版（推荐）

前往本仓库的 [Releases 页面](https://github.com/lali669123/nailong-pose/releases) 下载最新版的 `奶龙家人鉴定_分享版.zip`，完整解压后双击 `奶龙家人鉴定.exe` 即可开玩（已内置预训练模型，无需安装 Python）。

### 方式二：从源码运行（含动作录制与训练）

#### 环境要求

- Windows 10/11（摄像头可用）
- Python 3.11（推荐，兼容 `mediapipe==0.10.21`）

#### 1. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

#### 2. 启动应用

**开发训练版（Studio 模式，含动作录制与模型训练）：**

```powershell
.\start.ps1
```

**精简游玩版（Release 模式，隐藏并锁定训练接口）：**

```powershell
.\start_release.ps1
# 或：.\start.ps1 -Release
```

启动后在浏览器访问 `http://127.0.0.1:8000` 即可。

#### 3. 录制个人动作并训练模型（首次从源码运行）

仓库默认忽略个人样本 `data/samples.csv` 与本地模型 `models/pose_clf.joblib`（你也可以从 [Releases 页面](https://github.com/lali669123/nailong-pose/releases) 直接下载预训练的 `pose_clf.joblib` 放到 `models/` 目录下跳过录制）：

1. 以 Studio 模式启动，切换到右侧 **「采集与训练」** 标签页；
2. 先录制 **自然站立（idle）** 负样本，再依次录制 9 个常规奶蛙动作（每类建议分 2~3 轮共 160~240 帧；SSR 彩虹奶蛙无需单独录制）；
3. 点击 **「一键训练模型」**，训练完成后模型自动热加载；
4. 切回 **「实战图鉴」**，点击主舞台下方 **「开始游戏」** 即可开玩。

> 未训练模型时，也可以直接点击右侧任意图鉴卡片的 **「试播预览」** 体验弹窗、音效与同框海报合成。

---

## 打包可分享 Windows 独立桌面版

确认已训练出 `models/pose_clf.joblib` 后，安装打包依赖并执行：

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller pywebview
.\build_release.ps1
```

脚本会自动生成 `mode: release` 的纯净配置（不包含 `data/samples.csv` 与本地 `captures/`），在 `release/` 目录下输出独立程序与可直接分享的 `.zip` 压缩包。

---

## 运行单元测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

---

## 目录结构

```text
├── config/poses.yaml      # 动作图鉴、阈值、TTS 与可选 ESP32 配置
├── src/
│   ├── features.py        # 46 维平移/尺度不变骨骼特征提取
│   ├── train.py           # RF vs MLP 交叉验证对比与模型导出
│   ├── state_machine.py   # 蓄力/防抖/冷却状态机
│   ├── capture.py         # 1280×680 双栏同框海报合成
│   ├── config_loader.py   # 运行目录与配置加载（兼容 PyInstaller 冻结路径）
│   ├── tts_gen.py         # Edge-TTS 缺失配音补齐工具
│   ├── esp32_bridge.py    # 可选 ESP32 局域网音频触发桥接
│   ├── server.py          # FastAPI + MJPEG + 20Hz WebSocket 服务
│   └── desktop_app.py     # pywebview 独立桌面启动器
├── web/                   # 零构建前端（index.html / style.css / app.js）
├── assets/                # 图鉴图片、原声音频与桌面图标
├── tests/                 # 单元与端到端集成测试
├── start.ps1              # Studio / Release 启动脚本
└── build_release.ps1      # PyInstaller 纯净分享版打包脚本
```

---

## 免责声明与素材说明

本项目仅用于个人计算机视觉学习、技术演示与娱乐互动，非商业用途。`assets/` 中的表情包形象与梗音效素材版权均归原作者及相关权利人所有，如有侵权请联系删除。
