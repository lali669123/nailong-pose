/**
 * 奶龙家人鉴定 · 奶蛙演播室前端控制引擎
 * 零构建工具 · 纯原生 ES6 · 严格遵循 SPEC.md REST & WebSocket 契约
 */
(() => {
  "use strict";

  // =========================================================================
  // 常量与默认兜底配置
  // =========================================================================
  const CHARGE_RING_CIRCUMFERENCE = 2 * Math.PI * 76; // r = 76
  const SUMMARY_RING_CIRCUMFERENCE = 2 * Math.PI * 82; // r = 82
  const STORAGE_KEY_UNLOCKED = "nailong_studio_unlocked_v1";
  const STORAGE_KEY_MUTED = "nailong_studio_muted_v1";
  const DEFAULT_SKELETON_EDGES = [
    [1, 2], [1, 3], [3, 5], [5, 7],
    [2, 4], [4, 6], [6, 8],
    [1, 9], [2, 10], [9, 10],
    [9, 11], [11, 13], [13, 15],
    [10, 12], [12, 14], [14, 16],
    [0, 1], [0, 2],
  ];
  const RAINBOW_COLORS = [
    "#ff3b6b", "#ff9f1c", "#facc15", "#4ade80", "#38bdf8", "#818cf8", "#e879f9",
  ];

  // =========================================================================
  // 应用状态中心
  // =========================================================================
  const appState = {
    config: null,
    releaseMode: false,
    poses: [],
    poseMap: new Map(),
    skeletonEdges: DEFAULT_SKELETON_EDGES,
    sampleCounts: {},
    sampleTotal: 0,
    modelInfo: null,
    captures: [],

    // 解锁图鉴记录 { [pose_id]: count }
    unlocked: loadUnlockedMap(),
    // 音频状态
    muted: loadMutedPref(),
    audioUnlocked: false,
    currentAudio: null,
    esp32Enabled: false,

    // WS 实时状态
    ws: null,
    wsReconnectAttempt: 0,
    wsReconnectTimer: null,
    live: {
      fps: 0,
      mode: "infer",
      hasModel: false,
      personDetected: false,
      state: "IDLE",
      candidate: null,
      confidence: 0,
      chargeProgress: 0,
      cooldownRemaining: 0,
      collect: {
        active: false,
        phase: "idle",
        pose_id: null,
        countdown_left: 0,
        recorded: 0,
        target: 80,
      },
      game: {
        active: false,
        phase: "stopped",
        countdown_left: 0,
      },
    },
    wasCollecting: false,
    lastCollectPoseId: null,
    manualStopCollect: false,
    lastCountdownInt: null,
    lastGameCountdownInt: null,
    prevGamePhase: "stopped",
    gameGoTimer: null,

    // 本局游戏战绩与总结动画控制
    gameSession: {
      startedAt: 0,
      events: [],
      lastSummary: null,
    },
    previewEventKeys: new Set(),
    summaryVisible: false,
    summaryRafId: null,
    summaryStampTimer: null,

    // 骨骼 60fps 插值与粒子特效
    targetSkeleton: [],
    smoothSkeleton: [],
    particles: [],

    // 爆发弹窗控制
    burstVisible: false,
    burstSessionId: 0,
    burstTimer: null,
    burstExtraGuardTimer: null,
    burstEndedDelayTimer: null,
    burstAudioEl: null,
    burstAudioEndedHandler: null,
    burstCaptureUrl: "",
    lastEventDedupKey: "",
    lastEventTime: 0,

    // 灯箱状态
    lightboxItems: [],
    lightboxIndex: -1,

    // 按钮异步锁
    isTraining: false,
    isGeneratingTts: false,
  };

  // =========================================================================
  // DOM 元素缓存
  // =========================================================================
  const dom = {
    appTitle: document.getElementById("app-title"),
    audioUnlockPill: document.getElementById("audio-unlock-pill"),
    fpsValue: document.getElementById("fps-value"),
    modelBadge: document.getElementById("model-badge"),
    modelBadgeText: document.getElementById("model-badge-text"),
    wsBadge: document.getElementById("ws-badge"),
    wsBadgeText: document.getElementById("ws-badge-text"),
    btnSound: document.getElementById("btn-sound"),
    iconSoundOn: document.getElementById("icon-sound-on"),
    iconSoundOff: document.getElementById("icon-sound-off"),
    soundLabel: document.getElementById("sound-label"),
    btnFullscreen: document.getElementById("btn-fullscreen"),
    iconFsEnter: document.getElementById("icon-fs-enter"),
    iconFsExit: document.getElementById("icon-fs-exit"),
    fsLabel: document.getElementById("fs-label"),

    stageViewport: document.getElementById("stage-viewport"),
    videoFeed: document.getElementById("video-feed"),
    skeletonCanvas: document.getElementById("skeleton-canvas"),
    stageModePill: document.getElementById("stage-mode-pill"),
    stageModeText: document.getElementById("stage-mode-text"),
    personPill: document.getElementById("person-pill"),
    personPillText: document.getElementById("person-pill-text"),
    liveCandidatePill: document.getElementById("live-candidate-pill"),
    liveCandDot: document.getElementById("live-cand-dot"),
    liveCandName: document.getElementById("live-cand-name"),
    liveCandConf: document.getElementById("live-cand-conf"),
    cooldownPill: document.getElementById("cooldown-pill"),
    cooldownTime: document.getElementById("cooldown-time"),
    gameStandbyPill: document.getElementById("game-standby-pill"),
    gameCountdownOverlay: document.getElementById("game-countdown-overlay"),
    gameCdNum: document.getElementById("game-cd-num"),
    gameCdTip: document.getElementById("game-cd-tip"),
    btnGameStart: document.getElementById("btn-game-start"),
    btnGameStartLabel: document.getElementById("btn-game-start-label"),
    btnGameStartSub: document.getElementById("btn-game-start-sub"),
    btnGameStop: document.getElementById("btn-game-stop"),
    btnGameStopLabel: document.getElementById("btn-game-stop-label"),
    btnGameStopSub: document.getElementById("btn-game-stop-sub"),

    chargeHud: document.getElementById("charge-hud"),
    chargeArc: document.getElementById("charge-arc"),
    chargeAvatar: document.getElementById("charge-avatar"),
    chargePercent: document.getElementById("charge-percent"),
    chargeConfBadge: document.getElementById("charge-conf-badge"),
    chargePoseName: document.getElementById("charge-pose-name"),
    chargeLinearFill: document.getElementById("charge-linear-fill"),

    collectOverlay: document.getElementById("collect-overlay"),
    collectPhaseLabel: document.getElementById("collect-phase-label"),
    btnOverlayStopCollect: document.getElementById("btn-overlay-stop-collect"),
    collectRefImg: document.getElementById("collect-ref-img"),
    collectPoseName: document.getElementById("collect-pose-name"),
    collectPoseHint: document.getElementById("collect-pose-hint"),
    collectCountdownBox: document.getElementById("collect-countdown-box"),
    collectCountdownNum: document.getElementById("collect-countdown-num"),
    collectProgressText: document.getElementById("collect-progress-text"),
    collectProgressFill: document.getElementById("collect-progress-fill"),

    burstOverlay: document.getElementById("burst-overlay"),
    burstCard: document.getElementById("burst-card"),
    btnBurstClose: document.getElementById("btn-burst-close"),
    burstRainbowBanner: document.getElementById("burst-rainbow-banner"),
    burstPosterStage: document.getElementById("burst-poster-stage"),
    burstPosterImg: document.getElementById("burst-poster-img"),
    burstPosterConf: document.getElementById("burst-poster-conf"),
    burstPosterTitle: document.getElementById("burst-poster-title"),
    burstPosterSpeech: document.getElementById("burst-poster-speech"),
    burstPosterAudioStatus: document.getElementById("burst-poster-audio-status"),
    burstDownloadBtn: document.getElementById("burst-download-btn"),
    burstZoomBtn: document.getElementById("burst-zoom-btn"),
    burstFallbackContent: document.getElementById("burst-fallback-content"),
    burstImage: document.getElementById("burst-image"),
    burstConfVal: document.getElementById("burst-conf-val"),
    burstTitle: document.getElementById("burst-title"),
    burstSubtitle: document.getElementById("burst-subtitle"),
    burstSpeech: document.getElementById("burst-speech"),
    burstAudioStatus: document.getElementById("burst-audio-status"),
    burstCaptureChip: document.getElementById("burst-capture-chip"),

    stageTipText: document.getElementById("stage-tip-text"),
    footerSampleWrap: document.getElementById("footer-sample-wrap"),
    footerSampleTotal: document.getElementById("footer-sample-total"),
    footerUnlockedCount: document.getElementById("footer-unlocked-count"),

    brandSub: document.querySelector(".brand-sub"),
    tabNav: document.querySelector(".tab-nav"),
    tabBtnDex: document.getElementById("tab-btn-dex"),
    tabBtnTrain: document.getElementById("tab-btn-train"),
    tabButtons: document.querySelectorAll(".tab-btn"),
    panelDex: document.getElementById("panel-dex"),
    panelTrain: document.getElementById("panel-train"),
    tabDexBadge: document.getElementById("tab-dex-badge"),
    tabTrainBadge: document.getElementById("tab-train-badge"),

    dexProgressFill: document.getElementById("dex-progress-fill"),
    dexProgressLabel: document.getElementById("dex-progress-label"),
    btnResetDex: document.getElementById("btn-reset-dex"),
    dexGrid: document.getElementById("dex-grid"),
    galleryCount: document.getElementById("gallery-count"),
    btnRefreshGallery: document.getElementById("btn-refresh-gallery"),
    galleryGrid: document.getElementById("gallery-grid"),

    trainSampleTotal: document.getElementById("train-sample-total"),
    activeCollectBanner: document.getElementById("active-collect-banner"),
    activeCollectText: document.getElementById("active-collect-text"),
    btnStopCollect: document.getElementById("btn-stop-collect"),
    collectList: document.getElementById("collect-list"),
    evalSection: document.getElementById("eval-section"),
    evalCvChip: document.getElementById("eval-cv-chip"),
    evalCompareGrid: document.getElementById("eval-compare-grid"),
    evalMatrixWrap: document.getElementById("eval-matrix-wrap"),
    evalConfusionImg: document.getElementById("eval-confusion-img"),
    btnTrainModel: document.getElementById("btn-train-model"),
    btnTrainLabel: document.getElementById("btn-train-label"),
    btnTtsGen: document.getElementById("btn-tts-gen"),
    btnTtsLabel: document.getElementById("btn-tts-label"),

    lightbox: document.getElementById("lightbox"),
    lightboxBackdrop: document.getElementById("lightbox-backdrop"),
    lightboxTitle: document.getElementById("lightbox-title"),
    lightboxSub: document.getElementById("lightbox-sub"),
    lightboxDownload: document.getElementById("lightbox-download"),
    btnLightboxClose: document.getElementById("btn-lightbox-close"),
    btnLightboxPrev: document.getElementById("btn-lightbox-prev"),
    btnLightboxNext: document.getElementById("btn-lightbox-next"),
    lightboxImg: document.getElementById("lightbox-img"),

    summaryOverlay: document.getElementById("summary-overlay"),
    summaryBackdrop: document.getElementById("summary-backdrop"),
    summaryCard: document.getElementById("summary-card"),
    btnSummaryClose: document.getElementById("btn-summary-close"),
    summaryRingArc: document.getElementById("summary-ring-arc"),
    summaryPurityNum: document.getElementById("summary-purity-num"),
    summaryStamp: document.getElementById("summary-stamp"),
    summaryDuration: document.getElementById("summary-duration"),
    summaryActions: document.getElementById("summary-actions"),
    summaryUnique: document.getElementById("summary-unique"),
    summaryAvgConf: document.getElementById("summary-avg-conf"),
    summaryRankBadge: document.getElementById("summary-rank-badge"),
    summaryComment: document.getElementById("summary-comment"),
    summaryMvpCard: document.getElementById("summary-mvp-card"),
    summaryMvpImg: document.getElementById("summary-mvp-img"),
    summaryMvpName: document.getElementById("summary-mvp-name"),
    summaryMvpMeta: document.getElementById("summary-mvp-meta"),
    btnSummaryReplay: document.getElementById("btn-summary-replay"),
    btnSummarySave: document.getElementById("btn-summary-save"),

    toastContainer: document.getElementById("toast-container"),
  };

  const ctx = dom.skeletonCanvas.getContext("2d");

  // =========================================================================
  // 安全转义、容错与本地存储辅助函数
  // =========================================================================
  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function sanitizeUrl(url) {
    if (typeof url !== "string") return "";
    const trimmed = url.trim();
    if (!trimmed) return "";
    if (trimmed.startsWith("/") || trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
      return trimmed;
    }
    return "";
  }

  function sanitizeColor(color, fallback = "#facc15") {
    if (typeof color === "string" && /^#[0-9a-fA-F]{3,8}$/.test(color.trim())) {
      return color.trim();
    }
    return fallback;
  }

  function createFallbackSvgDataUri(label, accent = "#facc15") {
    const safeColor = sanitizeColor(accent);
    const shortText = String(label || "奶蛙").slice(0, 6);
    const svg =
      `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240" viewBox="0 0 320 240">` +
      `<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">` +
      `<stop offset="0%" stop-color="#172038"/><stop offset="100%" stop-color="#0b1020"/>` +
      `</linearGradient></defs>` +
      `<rect width="320" height="240" fill="url(#g)"/>` +
      `<circle cx="160" cy="100" r="46" fill="${safeColor}" fill-opacity="0.2" stroke="${safeColor}" stroke-width="3"/>` +
      `<text x="160" y="108" text-anchor="middle" fill="#fef08a" font-size="22" font-weight="bold" font-family="sans-serif">${escapeHtml(shortText)}</text>` +
      `<text x="160" y="182" text-anchor="middle" fill="#94a3b8" font-size="13" font-family="sans-serif">NAILONG STUDIO</text>` +
      `</svg>`;
    return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  }

  function bindImageFallback(imgEl, label, color) {
    if (!imgEl) return;
    imgEl.onerror = () => {
      imgEl.onerror = null;
      imgEl.src = createFallbackSvgDataUri(label, color);
    };
  }

  function loadUnlockedMap() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY_UNLOCKED);
      const parsed = raw ? JSON.parse(raw) : {};
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
      return {};
    }
  }

  function saveUnlockedMap() {
    try {
      localStorage.setItem(STORAGE_KEY_UNLOCKED, JSON.stringify(appState.unlocked));
    } catch {
      // 忽略无痕模式存储限制
    }
  }

  function loadMutedPref() {
    try {
      return localStorage.getItem(STORAGE_KEY_MUTED) === "1";
    } catch {
      return false;
    }
  }

  function saveMutedPref(muted) {
    try {
      localStorage.setItem(STORAGE_KEY_MUTED, muted ? "1" : "0");
    } catch {
      // ignore
    }
  }

  function clamp(v, min, max) {
    return Math.min(max, Math.max(min, v));
  }

  function formatPercent(val, digits = 0) {
    const num = Number(val);
    if (!Number.isFinite(num)) return "--";
    const pct = num <= 1.0001 ? num * 100 : num;
    return `${clamp(pct, 0, 100).toFixed(digits)}%`;
  }

  function showToast(message, type = "info", duration = 3200) {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    const icon = type === "success" ? "✅" : type === "error" ? "⚠️" : "✨";
    const iconSpan = document.createElement("span");
    iconSpan.textContent = icon;
    const textSpan = document.createElement("span");
    textSpan.textContent = String(message || "");
    toast.appendChild(iconSpan);
    toast.appendChild(textSpan);
    dom.toastContainer.appendChild(toast);

    window.setTimeout(() => {
      toast.classList.add("is-leaving");
      window.setTimeout(() => toast.remove(), 240);
    }, duration);
  }

  async function apiFetch(url, options = {}) {
    const resp = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    let payload = null;
    try {
      payload = await resp.json();
    } catch {
      payload = null;
    }
    if (!resp.ok) {
      const msg = (payload && (payload.detail || payload.message || payload.error)) || `请求失败 (${resp.status})`;
      throw new Error(msg);
    }
    return payload;
  }

  // =========================================================================
  // 声音播放 & 浏览器自动播放解锁机制（支持 MP3 / WAV）
  // =========================================================================
  function updateSoundUi() {
    const muted = appState.muted;
    dom.btnSound.classList.toggle("is-muted", muted);
    dom.iconSoundOn.classList.toggle("hidden", muted);
    dom.iconSoundOff.classList.toggle("hidden", !muted);
    dom.soundLabel.textContent = muted ? "已静音" : "原声开";

    const showUnlock = !muted && !appState.audioUnlocked && !appState.esp32Enabled;
    dom.audioUnlockPill.classList.toggle("hidden", !showUnlock);
  }

  function unlockAudioContext(silent = true) {
    if (appState.audioUnlocked) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        const audioCtx = new AudioCtx();
        if (audioCtx.state === "suspended") {
          audioCtx.resume().catch(() => {});
        }
        const buffer = audioCtx.createBuffer(1, 1, 22050);
        const src = audioCtx.createBufferSource();
        src.buffer = buffer;
        src.connect(audioCtx.destination);
        src.start(0);
      }
      appState.audioUnlocked = true;
      updateSoundUi();
      if (!silent) {
        showToast("已解锁奶蛙原声自动播放！", "success", 2200);
      }
    } catch {
      // 忽略异常
    }
  }

  function stopCurrentAudio() {
    if (appState.currentAudio) {
      try {
        appState.currentAudio.pause();
        appState.currentAudio.currentTime = 0;
      } catch {
        // ignore
      }
      appState.currentAudio = null;
    }
  }

  function setBurstAudioStatus(text) {
    if (dom.burstAudioStatus) dom.burstAudioStatus.textContent = text;
    if (dom.burstPosterAudioStatus) dom.burstPosterAudioStatus.textContent = text;
  }

  function playEventAudio(event) {
    stopCurrentAudio();
    const audioUrl = sanitizeUrl(event.audio_url);

    if (event.esp32_enabled) {
      setBurstAudioStatus("📡 已推送至 ESP32 硬件播报");
      return null;
    }
    if (appState.muted) {
      setBurstAudioStatus("🔇 当前已静音");
      return null;
    }
    if (!audioUrl) {
      setBurstAudioStatus(appState.releaseMode ? "🔈 暂无配音文件" : "🔈 暂无配音（可在训练面板点补齐配音）");
      return null;
    }

    const audio = new Audio();
    audio.preload = "auto";
    audio.src = audioUrl;
    audio.volume = 1.0;
    appState.currentAudio = audio;
    setBurstAudioStatus("🔊 正在播放奶蛙原声");

    const playPromise = audio.play();
    if (playPromise && typeof playPromise.then === "function") {
      playPromise
        .then(() => {
          appState.audioUnlocked = true;
          updateSoundUi();
        })
        .catch((err) => {
          if (err && err.name === "NotAllowedError") {
            appState.audioUnlocked = false;
            updateSoundUi();
            setBurstAudioStatus("🔔 请点顶部「点击解锁奶蛙原声」");
            showToast("浏览器拦截了自动发声，请点击页面任意处解锁奶蛙原声！", "info", 3800);
          } else {
            setBurstAudioStatus("⚠️ 音频文件加载失败");
          }
        });
    }

    audio.addEventListener("ended", () => {
      if (appState.currentAudio === audio) {
        setBurstAudioStatus("✨ 原声播放完成");
      }
    });
    audio.addEventListener("error", () => {
      if (appState.currentAudio === audio) {
        setBurstAudioStatus("⚠️ 原声加载失败");
      }
    });

    return audio;
  }

  // =========================================================================
  // 配置加载、模型状态解析与 UI 渲染
  // =========================================================================
  function extractModelSummary(modelInfo) {
    if (!modelInfo || typeof modelInfo !== "object") {
      return { loaded: false, name: "", accuracy: null, macroF1: null, cvStrategy: "", comparisons: null, confusionUrl: "" };
    }
    const name = modelInfo.best_model || modelInfo.model_name || "";
    const loaded = Boolean(
      modelInfo.loaded === true ||
      (modelInfo.loaded !== false && name)
    );
    const metrics = modelInfo.metrics && typeof modelInfo.metrics === "object" ? modelInfo.metrics : {};
    const comparisons = modelInfo.comparisons || metrics.comparisons || null;

    let accuracy = modelInfo.best_accuracy ?? metrics.best_accuracy ?? metrics.accuracy ?? null;
    let macroF1 = modelInfo.best_macro_f1 ?? metrics.best_macro_f1 ?? metrics.macro_f1 ?? null;
    if (accuracy == null && comparisons && name && comparisons[name]) {
      accuracy = comparisons[name].accuracy;
      macroF1 = comparisons[name].macro_f1;
    }

    const cvStrategy = modelInfo.cv_strategy || metrics.cv_strategy || "";
    const confusionUrl = sanitizeUrl(modelInfo.confusion_matrix_url || metrics.confusion_matrix_url || (loaded ? "/models/eval_confusion.png" : ""));

    return { loaded, name, accuracy, macroF1, cvStrategy, comparisons, confusionUrl };
  }

  function applyAppMode(isRelease) {
    appState.releaseMode = Boolean(isRelease);
    document.body.classList.toggle("is-release-mode", appState.releaseMode);

    if (dom.tabNav) {
      dom.tabNav.classList.toggle("hidden", appState.releaseMode);
    }
    if (dom.tabBtnTrain) {
      dom.tabBtnTrain.classList.toggle("hidden", appState.releaseMode);
    }
    if (dom.panelTrain) {
      dom.panelTrain.classList.toggle("hidden", appState.releaseMode);
      if (appState.releaseMode) {
        dom.panelTrain.classList.remove("active");
      }
    }
    if (appState.releaseMode && dom.panelDex) {
      dom.panelDex.classList.add("active");
      if (dom.tabBtnDex) {
        dom.tabBtnDex.classList.add("active");
        dom.tabBtnDex.setAttribute("aria-selected", "true");
      }
    }
    if (dom.footerSampleWrap) {
      dom.footerSampleWrap.classList.toggle("hidden", appState.releaseMode);
    }
    if (dom.brandSub) {
      dom.brandSub.textContent = appState.releaseMode
        ? "NAILONG POSE · LIVE EDITION"
        : "NAILONG POSE STUDIO · LIVE AI STAGE";
    }
    if (dom.stageTipText) {
      dom.stageTipText.textContent = appState.releaseMode
        ? "面向镜头摆出任意奶蛙姿势保持 0.5 秒，立即弹出你与奶蛙的双栏合照（捧腹大笑有概率爆出 SSR 彩虹奶蛙）！"
        : "站在摄像头前摆出奶蛙经典姿势并保持约 0.5 秒触发合照，捧腹大笑有概率爆出 SSR 隐藏款彩虹奶蛙！";
    }
  }

  function renderModelBadge() {
    const summary = extractModelSummary(appState.modelInfo);
    const hasModel = summary.loaded || appState.live.hasModel;

    dom.modelBadge.classList.toggle("status-ready", hasModel);
    dom.modelBadge.classList.toggle("status-untrained", !hasModel);

    if (hasModel) {
      if (appState.releaseMode) {
        dom.modelBadgeText.textContent = "AI 识别已就绪";
      } else {
        const modelName = summary.name || "已加载模型";
        const accText = summary.accuracy != null ? ` · ${formatPercent(summary.accuracy, 1)}` : "";
        dom.modelBadgeText.textContent = `已就绪 · ${modelName}${accText}`;
      }
    } else {
      dom.modelBadgeText.textContent = appState.releaseMode
        ? "未加载模型 · 仅预览模式"
        : "模型未训练 · 可试播体验";
    }
  }

  function renderEvalSection(evalPayload) {
    const summary = extractModelSummary(evalPayload);
    if (!summary.loaded && !summary.comparisons) {
      dom.evalSection.classList.add("hidden");
      return;
    }

    dom.evalSection.classList.remove("hidden");
    dom.evalCvChip.textContent = summary.cvStrategy || "交叉验证完成";
    dom.evalCompareGrid.innerHTML = "";

    const comparisons = summary.comparisons;
    if (comparisons && typeof comparisons === "object") {
      const entries = Object.entries(comparisons);
      entries.forEach(([modelName, scores]) => {
        const isWinner = summary.name && modelName === summary.name;
        const accPct = formatPercent(scores?.accuracy, 1);
        const f1Pct = formatPercent(scores?.macro_f1, 1);
        const accWidth = Number.isFinite(Number(scores?.accuracy)) ? clamp(Number(scores.accuracy) * 100, 0, 100) : 0;
        const f1Width = Number.isFinite(Number(scores?.macro_f1)) ? clamp(Number(scores.macro_f1) * 100, 0, 100) : 0;

        const card = document.createElement("div");
        card.className = `eval-model-card${isWinner ? " is-winner" : ""}`;
        card.innerHTML = `
          <div class="eval-model-head">
            <span class="eval-model-name">${escapeHtml(modelName)}</span>
            ${isWinner ? '<span class="winner-badge">👑 胜出</span>' : ""}
          </div>
          <div class="metric-row">
            <div class="metric-labels">
              <span>准确率 Accuracy</span>
              <strong class="hud-mono">${escapeHtml(accPct)}</strong>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" style="width:${accWidth.toFixed(1)}%"></div>
            </div>
          </div>
          <div class="metric-row">
            <div class="metric-labels">
              <span>宏平均 Macro-F1</span>
              <strong class="hud-mono">${escapeHtml(f1Pct)}</strong>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" style="width:${f1Width.toFixed(1)}%"></div>
            </div>
          </div>
        `;
        dom.evalCompareGrid.appendChild(card);
      });
    } else if (summary.name) {
      const card = document.createElement("div");
      card.className = "eval-model-card is-winner";
      card.style.gridColumn = "1 / -1";
      card.innerHTML = `
        <div class="eval-model-head">
          <span class="eval-model-name">${escapeHtml(summary.name)}</span>
          <span class="winner-badge">当前模型</span>
        </div>
        <div class="metric-labels">
          <span>验证准确率</span>
          <strong class="hud-mono">${escapeHtml(formatPercent(summary.accuracy, 1))}</strong>
        </div>
      `;
      dom.evalCompareGrid.appendChild(card);
    }

    const matrixUrl = summary.confusionUrl || "/models/eval_confusion.png";
    if (matrixUrl) {
      const bustedUrl = `${matrixUrl}${matrixUrl.includes("?") ? "&" : "?"}t=${Date.now()}`;
      dom.evalConfusionImg.onload = () => {
        dom.evalMatrixWrap.classList.remove("hidden");
      };
      dom.evalConfusionImg.onerror = () => {
        dom.evalMatrixWrap.classList.add("hidden");
      };
      dom.evalConfusionImg.src = bustedUrl;
    } else {
      dom.evalMatrixWrap.classList.add("hidden");
    }
  }

  function getTriggerablePoses() {
    return appState.poses.filter((p) => p.triggerable !== false);
  }

  function updateUnlockedSummaryUi() {
    const triggerables = getTriggerablePoses();
    const total = triggerables.length || 10;
    const unlockedCount = triggerables.filter((p) => (appState.unlocked[p.id] || 0) > 0).length;
    const pct = total > 0 ? (unlockedCount / total) * 100 : 0;

    dom.dexProgressFill.style.width = `${pct.toFixed(1)}%`;
    dom.dexProgressLabel.textContent = `${unlockedCount} / ${total}`;
    dom.tabDexBadge.textContent = `${unlockedCount}/${total}`;
    dom.footerUnlockedCount.textContent = `${unlockedCount}/${total}`;
  }

  function renderDexGrid() {
    const poses = getTriggerablePoses();
    updateUnlockedSummaryUi();

    if (!poses.length) {
      dom.dexGrid.innerHTML = '<div class="empty-state">暂未读取到奶蛙动作配置</div>';
      return;
    }

    const activeCandidate = appState.live.candidate;
    dom.dexGrid.innerHTML = "";

    poses.forEach((pose) => {
      const count = appState.unlocked[pose.id] || 0;
      const isUnlocked = count > 0;
      const isRainbow = Boolean(pose.is_hidden) || pose.id === "10_rainbow";
      const isMysteryLocked = isRainbow && !isUnlocked;
      const hv = pose.hidden_variant && typeof pose.hidden_variant === "object" ? pose.hidden_variant : null;
      const hvPct = hv ? Math.round(Number(hv.probability || 0.15) * 100) : 0;
      const color = sanitizeColor(pose.color, "#facc15");
      const imgUrl = sanitizeUrl(pose.image_url) || createFallbackSvgDataUri(pose.name, color);
      const displayName = isMysteryLocked ? "??? 神秘隐藏款" : pose.name;
      const displaySub = pose.subtitle || pose.speech || "";

      const card = document.createElement("article");
      card.className = [
        "dex-card",
        isUnlocked ? "is-unlocked" : "",
        isRainbow ? "is-rainbow-card" : "",
        isMysteryLocked ? "is-mystery-locked" : "",
        activeCandidate === pose.id ? "is-candidate" : "",
      ].filter(Boolean).join(" ");
      card.dataset.poseId = pose.id;
      card.style.setProperty("--pose-color", color);

      const thumbHtml = isMysteryLocked
        ? `
          <div class="dex-mystery-shadow" aria-label="未解锁的神秘隐藏款">
            <div class="dex-mystery-silhouette" aria-hidden="true"></div>
            <span class="dex-mystery-q" aria-hidden="true">?</span>
          </div>
        `
        : `<img class="dex-thumb" src="${escapeHtml(imgUrl)}" alt="${escapeHtml(pose.name)}" loading="lazy" />`;

      card.innerHTML = `
        <div class="dex-thumb-wrap">
          ${thumbHtml}
          <span class="dex-status-badge ${isUnlocked ? "unlocked" : "locked"}">
            ${isUnlocked ? `已解锁 ×${count}` : "待解锁"}
          </span>
          ${isRainbow ? '<span class="dex-ssr-tag">SSR 隐藏款</span>' : ""}
          ${!isRainbow && hv ? `<span class="dex-prob-tag">${hvPct}% 掉 SSR</span>` : ""}
        </div>
        <div class="dex-body">
          <div class="dex-name" title="${escapeHtml(displayName)}">${escapeHtml(displayName)}</div>
          <div class="dex-sub" title="${escapeHtml(displaySub)}">${escapeHtml(displaySub)}</div>
          <button class="dex-preview-btn" type="button" data-action="preview" data-pose-id="${escapeHtml(pose.id)}">
            <span>▶</span>
            <span>试播预览</span>
          </button>
        </div>
      `;

      const img = card.querySelector(".dex-thumb");
      if (img) bindImageFallback(img, pose.name, color);
      dom.dexGrid.appendChild(card);
    });
  }

  function highlightDexCandidate(candidateId) {
    const cards = dom.dexGrid.querySelectorAll(".dex-card");
    cards.forEach((card) => {
      card.classList.toggle("is-candidate", Boolean(candidateId) && card.dataset.poseId === candidateId);
    });
  }

  function renderCollectList() {
    const total = Number.isFinite(Number(appState.sampleTotal))
      ? Number(appState.sampleTotal)
      : Object.values(appState.sampleCounts).reduce((sum, n) => sum + (Number(n) || 0), 0);

    dom.trainSampleTotal.textContent = String(total);
    dom.tabTrainBadge.textContent = `${total} 帧`;
    dom.footerSampleTotal.textContent = String(total);

    if (!appState.poses.length) {
      dom.collectList.innerHTML = '<div class="empty-state">暂未读取到动作列表</div>';
      return;
    }

    const collect = appState.live.collect;
    const isAnyCollecting = Boolean(collect && collect.active);

    dom.collectList.innerHTML = "";

    appState.poses.forEach((pose) => {
      const count = Number(appState.sampleCounts[pose.id] || 0);
      const isIdle = pose.triggerable === false || pose.id === "idle";
      const isHiddenVariant = pose.trainable === false || Boolean(pose.is_hidden);
      const isRecordingThis = isAnyCollecting && collect.pose_id === pose.id;
      const color = sanitizeColor(pose.color, "#94a3b8");
      const imgUrl = sanitizeUrl(pose.image_url) || createFallbackSvgDataUri(pose.name, color);

      const pillClass = count >= 60 ? "is-enough" : count > 0 ? "has-some" : "";

      const row = document.createElement("div");
      row.className = [
        "collect-row",
        isIdle ? "is-idle-row" : "",
        isHiddenVariant ? "is-hidden-row" : "",
        isRecordingThis ? "is-recording" : "",
      ].filter(Boolean).join(" ");
      row.dataset.poseId = pose.id;

      const actionsHtml = isHiddenVariant
        ? `
          <div class="collect-hidden-note">✨ 隐藏款由「捧腹大笑奶蛙」按概率爆出，无需单独录制${count > 0 ? "（训练时已自动忽略旧帧）" : ""}</div>
          ${count > 0 ? `<button class="btn-clear" type="button" data-action="clear-samples" data-pose-id="${escapeHtml(pose.id)}" ${isAnyCollecting ? "disabled" : ""}>清空旧帧</button>` : ""}
        `
        : `
          ${
            isRecordingThis
              ? `<button class="btn-danger-sm" style="flex:1" type="button" data-action="stop-collect">⏹ 停止录制 (${collect.recorded}/${collect.target})</button>`
              : `<button class="btn-record" type="button" data-action="start-collect" data-pose-id="${escapeHtml(pose.id)}" ${isAnyCollecting ? "disabled" : ""}>● 录制 80 帧</button>`
          }
          <button class="btn-clear" type="button" data-action="clear-samples" data-pose-id="${escapeHtml(pose.id)}" ${count === 0 || isAnyCollecting ? "disabled" : ""}>清空</button>
        `;

      row.innerHTML = `
        <div class="collect-row-top">
          <img class="collect-thumb" src="${escapeHtml(imgUrl)}" alt="${escapeHtml(pose.name)}" loading="lazy" />
          <div class="collect-meta">
            <div class="collect-title-line">
              <span class="collect-name" title="${escapeHtml(pose.name)}">${escapeHtml(pose.name)}</span>
              ${isIdle ? '<span class="idle-tag">待机负样本</span>' : ""}
              ${isHiddenVariant ? '<span class="idle-tag" style="background:rgba(192,132,252,0.28);color:#f3e8ff">SSR 隐藏款</span>' : ""}
            </div>
            <div class="collect-hint" title="${escapeHtml(pose.hint || "")}">${escapeHtml(pose.hint || pose.subtitle || "")}</div>
          </div>
          <span class="frame-count-pill hud-mono ${pillClass}">${isHiddenVariant && count === 0 ? "免录" : `${count} 帧`}</span>
        </div>
        <div class="collect-row-actions">
          ${actionsHtml}
        </div>
      `;

      const img = row.querySelector(".collect-thumb");
      bindImageFallback(img, pose.name, color);
      dom.collectList.appendChild(row);
    });
  }

  function renderGallery() {
    const items = appState.captures || [];
    dom.galleryCount.textContent = `${items.length} 张`;

    if (!items.length) {
      dom.galleryGrid.innerHTML =
        '<div class="empty-state">触发任意奶蛙动作或点击「试播预览」，即可自动生成双栏同框海报！</div>';
      return;
    }

    dom.galleryGrid.innerHTML = "";
    items.forEach((item, idx) => {
      const safeUrl = sanitizeUrl(item.url);
      if (!safeUrl) return;
      const caption = formatCaptureCaption(item);

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "gallery-item";
      btn.dataset.index = String(idx);
      btn.title = `点击查看大图：${caption}`;
      btn.innerHTML = `
        <img src="${escapeHtml(safeUrl)}" alt="${escapeHtml(caption)}" loading="lazy" />
        <span class="gallery-caption">${escapeHtml(caption)}</span>
      `;
      const img = btn.querySelector("img");
      bindImageFallback(img, "同框照", "#facc15");
      dom.galleryGrid.appendChild(btn);
    });
  }

  function formatCaptureCaption(item) {
    const fname = String(item.filename || "");
    for (const pose of appState.poses) {
      if (fname.includes(pose.id)) {
        const timePart = formatShortTime(item.created_at);
        return timePart ? `${pose.name} · ${timePart}` : pose.name;
      }
    }
    return fname || "奶蛙同框照";
  }

  function formatShortTime(isoOrStr) {
    if (!isoOrStr) return "";
    const d = new Date(isoOrStr);
    if (Number.isNaN(d.getTime())) return String(isoOrStr).slice(0, 16);
    const pad = (n) => String(n).padStart(2, "0");
    return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  // =========================================================================
  // 后端 REST 数据同步
  // =========================================================================
  async function loadConfig(silent = false) {
    try {
      const cfg = await apiFetch("/api/config", { method: "GET" });
      appState.config = cfg;
      appState.poses = Array.isArray(cfg.poses) ? cfg.poses : [];
      appState.poseMap = new Map(appState.poses.map((p) => [p.id, p]));
      appState.skeletonEdges =
        Array.isArray(cfg.skeleton_edges) && cfg.skeleton_edges.length
          ? cfg.skeleton_edges
          : DEFAULT_SKELETON_EDGES;
      appState.sampleCounts = cfg.sample_counts && typeof cfg.sample_counts === "object" ? cfg.sample_counts : {};
      appState.sampleTotal =
        typeof cfg.sample_total === "number"
          ? cfg.sample_total
          : Object.values(appState.sampleCounts).reduce((a, b) => a + (Number(b) || 0), 0);
      appState.modelInfo = cfg.model_info || null;
      appState.esp32Enabled = Boolean(cfg.esp32 && cfg.esp32.enabled);

      if (cfg.app && cfg.app.title) {
        dom.appTitle.textContent = cfg.app.title;
        document.title = cfg.app.title;
      }

      const urlParams = new URLSearchParams(window.location.search);
      const urlMode = (urlParams.get("mode") || "").trim().toLowerCase();
      const serverRelease = Boolean(
        cfg.app && (cfg.app.release_mode === true || String(cfg.app.mode || "").toLowerCase() === "release")
      );
      applyAppMode(serverRelease || urlMode === "release");

      if (cfg.game && typeof cfg.game === "object") {
        appState.live.game = {
          active: Boolean(cfg.game.active),
          phase: cfg.game.phase || "stopped",
          countdown_left: Number(cfg.game.countdown_left || 0),
        };
      }

      updateSoundUi();
      renderModelBadge();
      renderDexGrid();
      updateGameUi(appState.live.game, Boolean(appState.live.collect && appState.live.collect.active));
      if (!appState.releaseMode) {
        renderCollectList();
        if (appState.modelInfo) {
          renderEvalSection(appState.modelInfo);
        }
      }
    } catch (err) {
      if (!silent) {
        showToast(`配置加载失败：${err.message}`, "error");
      }
    }
  }

  async function loadCaptures(silent = true) {
    try {
      const data = await apiFetch("/api/captures", { method: "GET" });
      const list = Array.isArray(data) ? data : Array.isArray(data?.captures) ? data.captures : [];
      appState.captures = list;
      renderGallery();
    } catch (err) {
      if (!silent) {
        showToast(`刷新同框照失败：${err.message}`, "error");
      }
    }
  }

  // =========================================================================
  // 爆发弹窗与粒子特效触发
  // =========================================================================
  function spawnBurstParticles(color, isRainbow) {
    const rect = dom.stageViewport.getBoundingClientRect();
    const cx = rect.width * 0.5;
    const cy = rect.height * 0.48;
    const count = isRainbow ? 125 : 78;

    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.35;
      const speed = (isRainbow ? 4.5 : 3.6) + Math.random() * 9.5;
      const palette = isRainbow
        ? RAINBOW_COLORS
        : [color, "#fef08a", "#ffffff", "#fb923c"];
      const pColor = palette[Math.floor(Math.random() * palette.length)];

      appState.particles.push({
        x: cx + (Math.random() - 0.5) * 30,
        y: cy + (Math.random() - 0.5) * 30,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - Math.random() * 2.4,
        size: 3 + Math.random() * 6.5,
        color: pColor,
        alpha: 1,
        decay: 0.011 + Math.random() * 0.014,
        shape: Math.random() > 0.45 ? "star" : "rect",
        rot: Math.random() * Math.PI * 2,
        vrot: (Math.random() - 0.5) * 0.22,
      });
    }
  }

  function clearBurstTimers() {
    if (appState.burstTimer) {
      window.clearTimeout(appState.burstTimer);
      appState.burstTimer = null;
    }
    if (appState.burstExtraGuardTimer) {
      window.clearTimeout(appState.burstExtraGuardTimer);
      appState.burstExtraGuardTimer = null;
    }
    if (appState.burstEndedDelayTimer) {
      window.clearTimeout(appState.burstEndedDelayTimer);
      appState.burstEndedDelayTimer = null;
    }
    if (appState.burstAudioEl && appState.burstAudioEndedHandler) {
      appState.burstAudioEl.removeEventListener("ended", appState.burstAudioEndedHandler);
    }
    appState.burstAudioEl = null;
    appState.burstAudioEndedHandler = null;
  }

  function closeBurstOverlay() {
    appState.burstSessionId += 1;
    clearBurstTimers();
    appState.burstVisible = false;
    dom.burstOverlay.classList.add("hidden");
    dom.burstOverlay.classList.remove("is-active", "is-rainbow");
    dom.burstCard.classList.remove("is-rainbow", "has-poster");
  }

  function showBurstPosterFallback() {
    dom.burstCard.classList.remove("has-poster");
    if (dom.burstPosterStage) dom.burstPosterStage.classList.add("hidden");
    if (dom.burstFallbackContent) dom.burstFallbackContent.classList.remove("hidden");
  }

  function handleTriggeredEvent(event, options = {}) {
    if (!event || !event.pose_id) return;

    const fromPreview = Boolean(options && options.fromPreview);
    // 防止 REST /preview 返回与 WS 推送在极短时间内重复触发同一次事件
    const dedupKey = `${event.pose_id}|${event.capture_url || ""}`;
    if (fromPreview) {
      appState.previewEventKeys.add(dedupKey);
    }
    const now = Date.now();
    if (dedupKey === appState.lastEventDedupKey && now - appState.lastEventTime < 900) {
      return;
    }
    appState.lastEventDedupKey = dedupKey;
    appState.lastEventTime = now;

    // 仅在真实游戏运行态记录本局战绩（试播预览不计入）
    if (
      !fromPreview &&
      !appState.previewEventKeys.has(dedupKey) &&
      appState.live.game &&
      appState.live.game.phase === "running" &&
      appState.gameSession.startedAt > 0
    ) {
      appState.gameSession.events.push({
        pose_id: event.pose_id,
        name: event.name || (appState.poseMap.get(event.pose_id) || {}).name || event.pose_id,
        image_url: event.image_url || (appState.poseMap.get(event.pose_id) || {}).image_url || "",
        color: event.color || (appState.poseMap.get(event.pose_id) || {}).color || "#facc15",
        confidence: clamp(Number(event.confidence ?? 0.95), 0, 1),
        is_hidden: Boolean(event.is_hidden) || event.pose_id === "10_rainbow",
        ts: now,
      });
    }

    const poseFallback = appState.poseMap.get(event.pose_id) || {};
    const isRainbow = Boolean(event.is_hidden) || event.pose_id === "10_rainbow";
    const color = sanitizeColor(event.color || poseFallback.color, "#facc15");
    const name = event.name || poseFallback.name || event.pose_id;
    const subtitle = event.subtitle || poseFallback.subtitle || "";
    const speech = event.speech || poseFallback.speech || "家人鉴定通过！";
    const imgUrl = sanitizeUrl(event.image_url || poseFallback.image_url) || createFallbackSvgDataUri(name, color);
    const captureUrl = sanitizeUrl(event.capture_url);
    const confText = formatPercent(event.confidence ?? 0.99, 0);

    // 1. 记录图鉴解锁计数并更新卡片
    appState.unlocked[event.pose_id] = (appState.unlocked[event.pose_id] || 0) + 1;
    saveUnlockedMap();
    renderDexGrid();

    // 2. 隐藏蓄力条并配置主题色与隐藏款彩虹特效
    dom.chargeHud.classList.add("hidden");
    dom.stageViewport.style.setProperty("--burst-color", color);

    dom.burstOverlay.classList.remove("is-active", "is-rainbow");
    dom.burstCard.classList.remove("is-rainbow", "has-poster");
    dom.burstRainbowBanner.classList.toggle("hidden", !isRainbow);

    if (isRainbow) {
      if (event.pity_triggered) {
        dom.burstRainbowBanner.textContent = "✨ 保底欧皇触发 · SSR 彩虹奶蛙降临 ✨";
      } else if (event.rolled_from) {
        dom.burstRainbowBanner.textContent = "✨ 捧腹大笑爆出 SSR 隐藏款 · 彩虹奶蛙降临 ✨";
      } else {
        dom.burstRainbowBanner.textContent = "✨ SSR 隐藏款 · 彩虹奶蛙降临 ✨";
      }
      dom.burstOverlay.classList.add("is-rainbow");
      dom.burstCard.classList.add("is-rainbow");
    }

    // 同时填充兜底单图卡与直出合照卡
    dom.burstTitle.textContent = name;
    dom.burstSubtitle.textContent = subtitle;
    dom.burstSpeech.textContent = speech;
    dom.burstConfVal.textContent = confText;
    dom.burstImage.src = imgUrl;
    bindImageFallback(dom.burstImage, name, color);

    appState.burstCaptureUrl = captureUrl;
    dom.burstCaptureChip.classList.toggle("hidden", !captureUrl);

    if (captureUrl && dom.burstPosterStage && dom.burstPosterImg) {
      dom.burstCard.classList.add("has-poster");
      dom.burstPosterStage.classList.remove("hidden");
      if (dom.burstFallbackContent) dom.burstFallbackContent.classList.add("hidden");

      dom.burstPosterTitle.textContent = name;
      if (dom.burstPosterConf) dom.burstPosterConf.textContent = confText;
      dom.burstPosterSpeech.textContent = speech ? `「${speech}」` : "";
      if (dom.burstDownloadBtn) {
        dom.burstDownloadBtn.href = captureUrl;
        dom.burstDownloadBtn.setAttribute("download", captureUrl.split("/").pop() || "nailong_match.jpg");
      }
      dom.burstPosterImg.onerror = () => {
        dom.burstPosterImg.onerror = null;
        showBurstPosterFallback();
      };
      dom.burstPosterImg.src = captureUrl;
    } else {
      showBurstPosterFallback();
    }

    // 3. 重启动画并展示爆发弹窗 + Canvas 粒子烟花
    dom.burstOverlay.classList.remove("hidden");
    void dom.burstOverlay.offsetWidth; // 强制重排以重放冲击波与弹跳动画
    dom.burstOverlay.classList.add("is-active");
    appState.burstVisible = true;
    spawnBurstParticles(color, isRainbow);

    // 4. 播放原声配音（MP3/WAV 自动支持）
    const audioEl = playEventAudio(event);

    // 5. 自动关闭时机（已减半加快节奏：普通合照 2.6s、彩虹隐藏款 3.75s，关闭后音频继续自然播完）
    clearBurstTimers();
    appState.burstSessionId += 1;
    const sessionId = appState.burstSessionId;
    const baseStayMs = isRainbow ? 3750 : (captureUrl ? 2600 : 1900);

    appState.burstTimer = window.setTimeout(() => {
      if (sessionId !== appState.burstSessionId) return;
      closeBurstOverlay();
    }, baseStayMs);

    // 6. 刷新最近同框照画廊
    if (captureUrl) {
      window.setTimeout(() => loadCaptures(true), 320);
    }
  }

  // =========================================================================
  // WebSocket 20Hz 实时推流与断线自动重连
  // =========================================================================
  function setWsBadge(status, text) {
    dom.wsBadge.classList.remove("ws-connected", "ws-connecting", "ws-disconnected");
    dom.wsBadge.classList.add(`ws-${status}`);
    dom.wsBadgeText.textContent = text;
  }

  function connectWebSocket() {
    if (appState.wsReconnectTimer) {
      window.clearTimeout(appState.wsReconnectTimer);
      appState.wsReconnectTimer = null;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    setWsBadge("connecting", appState.wsReconnectAttempt > 0 ? `重连中 (${appState.wsReconnectAttempt})...` : "连接中...");

    let ws;
    try {
      ws = new WebSocket(wsUrl);
    } catch {
      scheduleWsReconnect();
      return;
    }

    appState.ws = ws;

    ws.addEventListener("open", () => {
      const wasReconnect = appState.wsReconnectAttempt > 0;
      appState.wsReconnectAttempt = 0;
      setWsBadge("connected", "实时推流已连接");
      if (wasReconnect) {
        showToast("实时连接已恢复！", "success", 2000);
        loadConfig(true);
        loadCaptures(true);
      }
    });

    ws.addEventListener("message", (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        handleWsMessage(msg);
      } catch {
        // 忽略非法帧
      }
    });

    ws.addEventListener("close", () => {
      if (appState.ws === ws) {
        setWsBadge("disconnected", "连接已断开");
        scheduleWsReconnect();
      }
    });

    ws.addEventListener("error", () => {
      // close 事件会随后触发重连
    });
  }

  function scheduleWsReconnect() {
    if (appState.wsReconnectTimer) return;
    appState.wsReconnectAttempt += 1;
    const delayMs = Math.min(8000, 1000 * Math.pow(1.6, appState.wsReconnectAttempt - 1));
    setWsBadge("connecting", `${(delayMs / 1000).toFixed(0)}s 后重连...`);
    appState.wsReconnectTimer = window.setTimeout(() => {
      appState.wsReconnectTimer = null;
      connectWebSocket();
    }, delayMs);
  }

  function handleWsMessage(msg) {
    if (!msg || typeof msg !== "object") return;

    const live = appState.live;
    const prevCandidate = live.candidate;
    const prevHasModel = live.hasModel;

    live.fps = Number(msg.fps || 0);
    live.mode = msg.mode || "infer";
    live.hasModel = Boolean(msg.has_model);
    live.personDetected = Boolean(msg.person_detected);
    live.state = msg.state || "IDLE";
    live.candidate = msg.candidate || null;
    live.confidence = Number(msg.confidence || 0);
    live.chargeProgress = clamp(Number(msg.charge_progress || 0), 0, 1);
    live.cooldownRemaining = Math.max(0, Number(msg.cooldown_remaining || 0));
    live.collect = msg.collect || {
      active: false,
      phase: "idle",
      pose_id: null,
      countdown_left: 0,
      recorded: 0,
      target: 80,
    };
    live.game = msg.game || {
      active: false,
      phase: "stopped",
      countdown_left: 0,
    };

    // 骨骼目标点更新
    appState.targetSkeleton = Array.isArray(msg.skeleton) ? msg.skeleton : [];
    if (!live.personDetected || appState.targetSkeleton.length === 0) {
      appState.smoothSkeleton = [];
    }

    // 顶部 FPS 与模型状态同步
    dom.fpsValue.textContent = live.fps > 0 ? live.fps.toFixed(1) : "--";
    if (prevHasModel !== live.hasModel) {
      renderModelBadge();
    }

    // 舞台 HUD 状态更新
    const isCollectMode = Boolean(live.collect.active || live.mode === "collect");
    const gamePhase = String(live.game.phase || "stopped");
    dom.stageModePill.classList.toggle("pill-live", !isCollectMode && gamePhase === "running");
    dom.stageModePill.classList.toggle("pill-rec", isCollectMode);
    dom.stageModePill.classList.toggle("pill-standby", !isCollectMode && gamePhase !== "running");
    if (isCollectMode) {
      dom.stageModeText.textContent = "REC 动作采集中";
    } else if (gamePhase === "countdown") {
      dom.stageModeText.textContent = `READY 准备倒计时 ${Math.max(1, Math.ceil(Number(live.game.countdown_left || 0)))}s`;
    } else if (gamePhase === "running") {
      dom.stageModeText.textContent = "LIVE 实战鉴定中";
    } else {
      dom.stageModeText.textContent = "STANDBY 待机预览";
    }

    updateGameUi(live.game, isCollectMode);

    dom.personPill.classList.toggle("pill-detected", live.personDetected);
    dom.personPill.classList.toggle("pill-muted", !live.personDetected);
    dom.personPillText.textContent = live.personDetected ? "已锁定 17 点骨骼" : "请站入镜头中央";

    // 右上角实时候选指示
    if (!isCollectMode && live.candidate && live.confidence > 0.05) {
      const candPose = appState.poseMap.get(live.candidate);
      const candColor = sanitizeColor(candPose?.color, "#facc15");
      dom.liveCandidatePill.classList.remove("hidden");
      dom.liveCandDot.style.background = candColor;
      dom.liveCandDot.style.boxShadow = `0 0 10px ${candColor}`;
      dom.liveCandName.textContent = candPose ? candPose.name : live.candidate;
      dom.liveCandConf.textContent = formatPercent(live.confidence, 0);
    } else {
      dom.liveCandidatePill.classList.add("hidden");
    }

    if (prevCandidate !== live.candidate) {
      highlightDexCandidate(live.candidate);
    }

    // 采集遮罩 & 采集状态切换时同步更新列表（进入/切换录制态时重绘按钮与禁用态；自然完成时提示并刷新统计）
    const nowCollecting = Boolean(live.collect.active);
    const nowCollectPoseId = nowCollecting ? (live.collect.pose_id || null) : null;
    const collectStartedOrChanged =
      nowCollecting && (!appState.wasCollecting || appState.lastCollectPoseId !== nowCollectPoseId);

    if (collectStartedOrChanged) {
      appState.manualStopCollect = false;
      renderCollectList();
    }
    updateCollectOverlayUi(live.collect);
    if (appState.wasCollecting && !nowCollecting) {
      if (!appState.manualStopCollect) {
        showToast("本轮动作数据采集已完成！", "success");
      }
      appState.manualStopCollect = false;
      loadConfig(true);
    }
    appState.wasCollecting = nowCollecting;
    appState.lastCollectPoseId = nowCollectPoseId;

    // 蓄力环与冷却提示更新
    updateChargeAndCooldownUi(live);

    // 触发爆发弹窗
    if (msg.triggered_event) {
      handleTriggeredEvent(msg.triggered_event);
    }
  }

  function updateGameUi(game, isCollectMode) {
    const phase = String((game && game.phase) || "stopped");
    const cdLeft = Math.max(0, Number((game && game.countdown_left) || 0));
    const isCountdown = !isCollectMode && phase === "countdown";
    const isRunning = !isCollectMode && phase === "running";
    const isStopped = !isCollectMode && phase === "stopped";

    // 进入实战 running 的瞬间开始精确计时（不含 3 秒准备倒计时）
    if (isRunning && appState.gameSession.startedAt === 0) {
      appState.gameSession.startedAt = performance.now();
    }

    // 大按钮状态同步
    if (dom.btnGameStart && dom.btnGameStop) {
      if (isCollectMode) {
        dom.btnGameStart.disabled = true;
        dom.btnGameStop.disabled = true;
        dom.btnGameStart.classList.remove("is-active-state");
        dom.btnGameStop.classList.remove("is-active-state");
        if (dom.btnGameStartLabel) dom.btnGameStartLabel.textContent = "动作采集中";
        if (dom.btnGameStartSub) dom.btnGameStartSub.textContent = "请先完成或停止当前录制";
      } else if (isCountdown) {
        const sec = Math.max(1, Math.ceil(cdLeft));
        dom.btnGameStart.disabled = true;
        dom.btnGameStop.disabled = false;
        dom.btnGameStart.classList.add("is-active-state");
        dom.btnGameStop.classList.add("is-active-state");
        if (dom.btnGameStartLabel) dom.btnGameStartLabel.textContent = `准备倒计时 ${sec} 秒...`;
        if (dom.btnGameStartSub) dom.btnGameStartSub.textContent = "请退后站好并摆好姿势";
      } else if (isRunning) {
        dom.btnGameStart.disabled = true;
        dom.btnGameStop.disabled = false;
        dom.btnGameStart.classList.add("is-active-state");
        dom.btnGameStop.classList.add("is-active-state");
        if (dom.btnGameStartLabel) dom.btnGameStartLabel.textContent = "游戏进行中";
        if (dom.btnGameStartSub) dom.btnGameStartSub.textContent = "摆出任意奶蛙动作保持 0.5 秒触发";
      } else {
        dom.btnGameStart.disabled = false;
        dom.btnGameStop.disabled = true;
        dom.btnGameStart.classList.remove("is-active-state");
        dom.btnGameStop.classList.remove("is-active-state");
        if (dom.btnGameStartLabel) dom.btnGameStartLabel.textContent = "开始游戏";
        if (dom.btnGameStartSub) dom.btnGameStartSub.textContent = "点击后倒计时 3 秒准备";
      }
    }

    // 待机提示条（未开始游戏且未采集、未弹窗时显示）
    if (dom.gameStandbyPill) {
      dom.gameStandbyPill.classList.toggle("hidden", !isStopped || appState.burstVisible);
    }

    // 3 秒大字倒计时遮罩 + 结束瞬间 GO! 闪现
    if (dom.gameCountdownOverlay && dom.gameCdNum) {
      if (isCountdown) {
        if (appState.gameGoTimer) {
          window.clearTimeout(appState.gameGoTimer);
          appState.gameGoTimer = null;
        }
        dom.gameCountdownOverlay.classList.remove("hidden", "is-go");
        const num = Math.max(1, Math.ceil(cdLeft));
        if (appState.lastGameCountdownInt !== num) {
          appState.lastGameCountdownInt = num;
          dom.gameCdNum.textContent = String(num);
          if (dom.gameCdTip) dom.gameCdTip.textContent = "请退后站好，准备摆出奶蛙动作！";
          dom.gameCdNum.style.animation = "none";
          void dom.gameCdNum.offsetWidth;
          dom.gameCdNum.style.animation = "";
        }
      } else if (appState.prevGamePhase === "countdown" && isRunning) {
        // 刚刚倒计时归零 -> 闪现 GO! 650ms
        appState.lastGameCountdownInt = null;
        dom.gameCountdownOverlay.classList.remove("hidden");
        dom.gameCountdownOverlay.classList.add("is-go");
        dom.gameCdNum.textContent = "GO!";
        if (dom.gameCdTip) dom.gameCdTip.textContent = "鉴定开始，尽情摆姿势吧！";
        dom.gameCdNum.style.animation = "none";
        void dom.gameCdNum.offsetWidth;
        dom.gameCdNum.style.animation = "";
        if (appState.gameGoTimer) window.clearTimeout(appState.gameGoTimer);
        appState.gameGoTimer = window.setTimeout(() => {
          dom.gameCountdownOverlay.classList.add("hidden");
          dom.gameCountdownOverlay.classList.remove("is-go");
          appState.gameGoTimer = null;
        }, 680);
      } else if (!appState.gameGoTimer) {
        appState.lastGameCountdownInt = null;
        dom.gameCountdownOverlay.classList.add("hidden");
        dom.gameCountdownOverlay.classList.remove("is-go");
      }
    }

    appState.prevGamePhase = phase;
  }

  function updateChargeAndCooldownUi(live) {
    const isCollect = Boolean(live.collect && live.collect.active);
    const isCharging = !isCollect && !appState.burstVisible && live.state === "CHARGING" && live.chargeProgress > 0.01;

    if (isCharging) {
      const pose = appState.poseMap.get(live.candidate) || {};
      const color = sanitizeColor(pose.color, "#facc15");
      const pct = Math.round(live.chargeProgress * 100);
      const offset = CHARGE_RING_CIRCUMFERENCE * (1 - live.chargeProgress);

      dom.chargeHud.style.setProperty("--charge-color", color);
      dom.chargeArc.style.strokeDashoffset = offset.toFixed(2);
      dom.chargePercent.textContent = `${pct}%`;
      dom.chargePoseName.textContent = pose.name || live.candidate || "奶蛙蓄力";
      dom.chargeConfBadge.textContent = `置信度 ${formatPercent(live.confidence, 0)}`;
      dom.chargeLinearFill.style.width = `${pct}%`;

      const avatarUrl = sanitizeUrl(pose.image_url);
      if (avatarUrl && dom.chargeAvatar.dataset.src !== avatarUrl) {
        dom.chargeAvatar.dataset.src = avatarUrl;
        dom.chargeAvatar.src = avatarUrl;
        bindImageFallback(dom.chargeAvatar, pose.name, color);
      }

      dom.chargeHud.classList.remove("hidden");
    } else {
      dom.chargeHud.classList.add("hidden");
    }

    const isCooldown = !isCollect && !appState.burstVisible && live.state === "COOLDOWN" && live.cooldownRemaining > 0.05;
    if (isCooldown) {
      dom.cooldownTime.textContent = `${live.cooldownRemaining.toFixed(1)}s`;
      dom.cooldownPill.classList.remove("hidden");
    } else {
      dom.cooldownPill.classList.add("hidden");
    }
  }

  function updateCollectOverlayUi(collect) {
    const active = Boolean(collect && collect.active);
    dom.collectOverlay.classList.toggle("hidden", !active);
    dom.activeCollectBanner.classList.toggle("hidden", !active);

    if (!active) {
      appState.lastCountdownInt = null;
      return;
    }

    const pose = appState.poseMap.get(collect.pose_id) || {
      id: collect.pose_id,
      name: collect.pose_id || "未知姿势",
      hint: "请保持目标姿势在镜头中央",
      color: "#facc15",
    };

    const refUrl = sanitizeUrl(pose.image_url) || createFallbackSvgDataUri(pose.name, pose.color);
    if (dom.collectRefImg.dataset.poseId !== pose.id) {
      dom.collectRefImg.dataset.poseId = pose.id;
      dom.collectRefImg.src = refUrl;
      bindImageFallback(dom.collectRefImg, pose.name, pose.color);
    }

    dom.collectPoseName.textContent = pose.name;
    dom.collectPoseHint.textContent = pose.hint || pose.subtitle || "保持动作清晰可见";

    const countdownLeft = Math.max(0, Number(collect.countdown_left || 0));
    const isCountdown = collect.phase === "countdown" || countdownLeft > 0.05;
    const recorded = Number(collect.recorded || 0);
    const target = Math.max(1, Number(collect.target || 80));
    const pct = clamp((recorded / target) * 100, 0, 100);

    dom.collectPhaseLabel.textContent = isCountdown ? "准备倒计时" : "正在采集骨骼帧...";
    dom.activeCollectText.textContent = isCountdown
      ? `准备采集「${pose.name}」(${Math.ceil(countdownLeft)}s)`
      : `录制「${pose.name}」${recorded} / ${target} 帧`;

    if (isCountdown) {
      dom.collectCountdownBox.classList.remove("hidden");
      const num = Math.max(1, Math.ceil(countdownLeft));
      if (appState.lastCountdownInt !== num) {
        appState.lastCountdownInt = num;
        dom.collectCountdownNum.textContent = String(num);
        dom.collectCountdownNum.style.animation = "none";
        void dom.collectCountdownNum.offsetWidth;
        dom.collectCountdownNum.style.animation = "";
      }
    } else {
      dom.collectCountdownBox.classList.add("hidden");
    }

    dom.collectProgressText.textContent = `${recorded} / ${target} 帧`;
    dom.collectProgressFill.style.width = `${pct.toFixed(1)}%`;

    // 同步更新采集列表中正在录制行的按钮文字
    const recRowBtn = dom.collectList.querySelector(`.collect-row[data-pose-id="${CSS.escape(pose.id)}"] [data-action="stop-collect"]`);
    if (recRowBtn) {
      recRowBtn.textContent = `⏹ 停止录制 (${recorded}/${target})`;
    }
  }

  // =========================================================================
  // 60fps Canvas 霓虹火柴人骨骼 & 粒子特效渲染引擎
  // =========================================================================
  function resizeSkeletonCanvas() {
    const rect = dom.stageViewport.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const targetW = Math.max(1, Math.round(rect.width * dpr));
    const targetH = Math.max(1, Math.round(rect.height * dpr));

    if (dom.skeletonCanvas.width !== targetW || dom.skeletonCanvas.height !== targetH) {
      dom.skeletonCanvas.width = targetW;
      dom.skeletonCanvas.height = targetH;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /**
   * 精确计算 object-fit: contain 下 <img id="video-feed"> 的真实内容渲染矩形
   * 保证 17 点归一化坐标在任意窗口尺寸和全屏模式下零错位对齐
   */
  function getVideoContentRect() {
    const vw = dom.stageViewport.clientWidth;
    const vh = dom.stageViewport.clientHeight;
    const cfgApp = (appState.config && appState.config.app) || {};
    const nw = dom.videoFeed.naturalWidth || Number(cfgApp.frame_width) || 640;
    const nh = dom.videoFeed.naturalHeight || Number(cfgApp.frame_height) || 480;

    if (!vw || !vh || !nw || !nh) {
      return { offsetX: 0, offsetY: 0, drawW: vw, drawH: vh };
    }
    const scale = Math.min(vw / nw, vh / nh);
    const drawW = nw * scale;
    const drawH = nh * scale;
    const offsetX = (vw - drawW) * 0.5;
    const offsetY = (vh - drawH) * 0.5;
    return { offsetX, offsetY, drawW, drawH };
  }

  function updateSmoothedSkeleton() {
    const target = appState.targetSkeleton;
    if (!target || target.length === 0) {
      appState.smoothSkeleton = [];
      return;
    }
    if (appState.smoothSkeleton.length !== target.length) {
      appState.smoothSkeleton = target.map((pt) => ({ ...pt }));
      return;
    }
    const alpha = 0.45;
    for (let i = 0; i < target.length; i++) {
      const t = target[i];
      const s = appState.smoothSkeleton[i];
      if ((s.v || 0) <= 0.3 && (t.v || 0) > 0.3) {
        s.x = t.x;
        s.y = t.y;
      } else {
        s.x += (t.x - s.x) * alpha;
        s.y += (t.y - s.y) * alpha;
      }
      s.v = t.v;
      s.idx = t.idx;
    }
  }

  function drawSkeleton(nowMs) {
    const pts = appState.smoothSkeleton;
    if (!pts || pts.length === 0) return;

    const { offsetX, offsetY, drawW, drawH } = getVideoContentRect();
    const live = appState.live;
    const candPose = live.candidate ? appState.poseMap.get(live.candidate) : null;
    const isRainbow = live.candidate === "10_rainbow" && (live.state === "CHARGING" || live.state === "TRIGGERED");
    const isCharging = live.state === "CHARGING";
    const isCollecting = Boolean(live.collect && live.collect.active);

    let strokeColor = "#facc15";
    let glowBlur = 14;
    let lineWidth = 3.5;

    if (isCollecting) {
      strokeColor = "#fb7185";
      glowBlur = 16;
    } else if (isCharging) {
      strokeColor = sanitizeColor(candPose?.color, "#facc15");
      glowBlur = 16 + live.chargeProgress * 18;
      lineWidth = 3.8 + live.chargeProgress * 2.2;
    }

    const mapped = pts.map((pt) => ({
      x: offsetX + pt.x * drawW,
      y: offsetY + pt.y * drawH,
      v: Number(pt.v || 0),
    }));

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // 1. 绘制霓虹骨骼连线（仅两端点 v > 0.3）
    const edges = appState.skeletonEdges || DEFAULT_SKELETON_EDGES;
    for (let i = 0; i < edges.length; i++) {
      const [aIdx, bIdx] = edges[i];
      const a = mapped[aIdx];
      const b = mapped[bIdx];
      if (!a || !b || a.v <= 0.3 || b.v <= 0.3) continue;

      if (isRainbow) {
        const hue = ((nowMs * 0.16) + i * 24) % 360;
        const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
        grad.addColorStop(0, `hsl(${hue}, 95%, 64%)`);
        grad.addColorStop(1, `hsl(${(hue + 55) % 360}, 95%, 64%)`);
        ctx.strokeStyle = grad;
        ctx.shadowColor = `hsl(${hue}, 95%, 60%)`;
      } else {
        ctx.strokeStyle = strokeColor;
        ctx.shadowColor = strokeColor;
      }

      ctx.shadowBlur = glowBlur;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();

      // 细亮白内芯，增强赛博霓虹质感
      ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.72)";
      ctx.lineWidth = Math.max(1.2, lineWidth * 0.32);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    // 2. 绘制关节点（v > 0.3）
    for (let i = 0; i < mapped.length; i++) {
      const p = mapped[i];
      if (p.v <= 0.3) continue;

      const nodeColor = isRainbow
        ? `hsl(${((nowMs * 0.2) + i * 28) % 360}, 95%, 66%)`
        : i === 0
          ? "#38bdf8"
          : strokeColor;

      const baseRadius = i === 0 ? 6.5 : 4.6;
      const pulse = isCharging ? Math.sin(nowMs * 0.012 + i) * 1.1 : 0;
      const r = baseRadius + pulse;

      // 外光晕环
      ctx.shadowBlur = glowBlur;
      ctx.shadowColor = nodeColor;
      ctx.fillStyle = nodeColor;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();

      // 白色高光内核
      ctx.shadowBlur = 0;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(1.8, r * 0.45), 0, Math.PI * 2);
      ctx.fill();

      // 鼻尖 (index 0) 演播室追踪准星小环
      if (i === 0) {
        ctx.strokeStyle = isRainbow ? nodeColor : "rgba(56, 189, 248, 0.75)";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 13 + Math.sin(nowMs * 0.006) * 1.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  function updateAndDrawParticles() {
    if (appState.particles.length === 0) return;

    ctx.save();
    for (let i = appState.particles.length - 1; i >= 0; i--) {
      const p = appState.particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.965;
      p.vy = p.vy * 0.965 + 0.14; // 轻微重力
      p.rot += p.vrot;
      p.alpha -= p.decay;

      if (p.alpha <= 0.02) {
        appState.particles.splice(i, 1);
        continue;
      }

      ctx.save();
      ctx.globalAlpha = clamp(p.alpha, 0, 1);
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.shadowBlur = 10;
      ctx.shadowColor = p.color;
      ctx.fillStyle = p.color;

      if (p.shape === "star") {
        ctx.beginPath();
        const spikes = 4;
        const outerR = p.size;
        const innerR = p.size * 0.42;
        for (let s = 0; s < spikes * 2; s++) {
          const rad = s % 2 === 0 ? outerR : innerR;
          const a = (s * Math.PI) / spikes;
          if (s === 0) ctx.moveTo(Math.cos(a) * rad, Math.sin(a) * rad);
          else ctx.lineTo(Math.cos(a) * rad, Math.sin(a) * rad);
        }
        ctx.closePath();
        ctx.fill();
      } else {
        ctx.fillRect(-p.size * 0.7, -p.size * 0.38, p.size * 1.4, p.size * 0.76);
      }
      ctx.restore();
    }
    ctx.restore();
  }

  function renderLoop(nowMs) {
    const w = dom.stageViewport.clientWidth;
    const h = dom.stageViewport.clientHeight;
    ctx.clearRect(0, 0, w, h);

    updateSmoothedSkeleton();
    drawSkeleton(nowMs);
    updateAndDrawParticles();

    window.requestAnimationFrame(renderLoop);
  }

  // =========================================================================
  // 结算总结：纯度计算（全部下放一档）、动画渲染与高清鉴定证书导出
  // =========================================================================
  function getPurityTier(purity) {
    if (purity > 95) {
      return {
        rank: "奶龙始祖 · 血统溢出",
        stamp: "奶龙始祖",
        comment: "动作拿捏到极致，你就是奶龙本龙降世，建议直接去给奶龙当动捕演员！",
        color: "#f472b6",
        isRainbow: true,
      };
    }
    if (purity >= 75) {
      return {
        rank: "尊贵纯血奶家人",
        stamp: "纯血奶家人",
        comment: "这已经不是模仿了，你就是奶龙失散多年的亲戚！",
        color: "#facc15",
        isRainbow: false,
      };
    }
    if (purity >= 55) {
      return {
        rank: "高纯度奶龙后裔",
        stamp: "奶龙后裔",
        comment: "骨骼惊奇、神韵到位，离彻底奶龙化只差临门一脚！",
        color: "#fb923c",
        isRainbow: false,
      };
    }
    if (purity >= 30) {
      return {
        rank: "见习奶蛙学徒",
        stamp: "奶蛙学徒",
        comment: "有内味儿了！多解锁几个姿势，血统浓度还能飙升！",
        color: "#38bdf8",
        isRainbow: false,
      };
    }
    if (purity >= 1) {
      return {
        rank: "远房疑似亲戚",
        stamp: "远房亲戚",
        comment: "动作略显拘谨，放开点再大笑两声试试！",
        color: "#a78bfa",
        isRainbow: false,
      };
    }
    return {
      rank: "纯正人类围观群众",
      stamp: "纯正人类",
      comment: "一动不动是吧？鉴定为纯路人，快站起来摆个姿势！",
      color: "#94a3b8",
      isRainbow: false,
    };
  }

  function computeGameSummary() {
    const startedAt = appState.gameSession.startedAt;
    const durationSec = startedAt > 0 ? Math.max(0.1, (performance.now() - startedAt) / 1000) : 0;
    const events = appState.gameSession.events.slice();
    const totalActions = events.length;
    const totalPoseTypes = Math.max(1, getTriggerablePoses().length || 10);

    const uniqueMap = new Map();
    let confSum = 0;
    let hasRainbow = false;
    let mvpEvent = null;

    events.forEach((ev) => {
      confSum += ev.confidence;
      uniqueMap.set(ev.pose_id, (uniqueMap.get(ev.pose_id) || 0) + 1);
      if (ev.is_hidden || ev.pose_id === "10_rainbow") {
        hasRainbow = true;
      }
      if (!mvpEvent || ev.confidence >= mvpEvent.confidence) {
        mvpEvent = ev;
      }
    });

    const uniqueCount = uniqueMap.size;
    const avgConf = totalActions > 0 ? confSum / totalActions : 0;

    let purity = 0;
    if (totalActions > 0) {
      const accuracyScore = clamp(avgConf, 0, 1) * 45;
      const varietyScore = Math.min(1, uniqueCount / 5) * 35;
      const activityScore = Math.min(1, totalActions / 5) * 20;
      const ssrBonus = hasRainbow ? 8 : 0;
      purity = Math.min(108, Math.round(accuracyScore + varietyScore + activityScore + ssrBonus));
    }

    const tier = getPurityTier(purity);
    const mvp = mvpEvent
      ? {
          pose_id: mvpEvent.pose_id,
          name: mvpEvent.name,
          image_url: mvpEvent.image_url,
          color: mvpEvent.color,
          confidence: mvpEvent.confidence,
          count: uniqueMap.get(mvpEvent.pose_id) || 1,
        }
      : null;

    return {
      durationSec,
      totalActions,
      uniqueCount,
      totalPoseTypes,
      avgConf,
      purity,
      tier,
      mvp,
      finishedAt: new Date(),
    };
  }

  function clearSummaryAnimations() {
    if (appState.summaryRafId) {
      window.cancelAnimationFrame(appState.summaryRafId);
      appState.summaryRafId = null;
    }
    if (appState.summaryStampTimer) {
      window.clearTimeout(appState.summaryStampTimer);
      appState.summaryStampTimer = null;
    }
  }

  function closeSummaryModal() {
    clearSummaryAnimations();
    appState.summaryVisible = false;
    if (dom.summaryOverlay) {
      dom.summaryOverlay.classList.add("hidden");
      dom.summaryOverlay.classList.remove("is-active");
    }
    if (dom.summaryCard) {
      dom.summaryCard.classList.remove("is-rainbow", "is-shaking");
    }
  }

  function openSummaryModal(summary) {
    if (!dom.summaryOverlay || !summary) return;
    clearSummaryAnimations();
    appState.gameSession.lastSummary = summary;
    appState.summaryVisible = true;

    const { tier, mvp } = summary;
    dom.summaryCard.classList.toggle("is-rainbow", Boolean(tier.isRainbow));
    dom.summaryCard.classList.remove("is-shaking");
    dom.summaryCard.style.setProperty("--summary-accent", tier.color);

    if (dom.summaryRingArc) {
      dom.summaryRingArc.style.strokeDasharray = `${SUMMARY_RING_CIRCUMFERENCE.toFixed(2)}`;
      dom.summaryRingArc.style.strokeDashoffset = `${SUMMARY_RING_CIRCUMFERENCE.toFixed(2)}`;
      dom.summaryRingArc.setAttribute(
        "stroke",
        tier.isRainbow ? "url(#summaryRingRainbow)" : "url(#summaryRingGrad)"
      );
    }

    if (dom.summaryStamp) {
      dom.summaryStamp.classList.add("hidden");
      dom.summaryStamp.classList.remove("is-stamped");
      dom.summaryStamp.textContent = tier.stamp;
    }
    if (dom.summaryRankBadge) dom.summaryRankBadge.textContent = tier.rank;
    if (dom.summaryComment) dom.summaryComment.textContent = tier.comment;

    if (mvp && dom.summaryMvpCard) {
      const mvpImgUrl = sanitizeUrl(mvp.image_url) || createFallbackSvgDataUri(mvp.name, mvp.color);
      dom.summaryMvpImg.src = mvpImgUrl;
      bindImageFallback(dom.summaryMvpImg, mvp.name, mvp.color);
      dom.summaryMvpName.textContent = mvp.name;
      dom.summaryMvpMeta.textContent = `最高匹配度 ${formatPercent(mvp.confidence, 1)} · 本局触发 ${mvp.count} 次`;
      dom.summaryMvpCard.classList.remove("hidden");
    } else if (dom.summaryMvpCard) {
      dom.summaryMvpCard.classList.add("hidden");
    }

    // 初始清零准备滚动动画
    dom.summaryPurityNum.textContent = "0%";
    dom.summaryDuration.textContent = "0.0s";
    dom.summaryActions.textContent = "0 次";
    dom.summaryUnique.textContent = `0 / ${summary.totalPoseTypes}`;
    dom.summaryAvgConf.textContent = "0%";

    dom.summaryOverlay.classList.remove("hidden");
    void dom.summaryOverlay.offsetWidth;
    dom.summaryOverlay.classList.add("is-active");

    const animDuration = 1100;
    const animStart = performance.now();
    const targetRingRatio = clamp(summary.purity / 100, 0, 1);

    const tick = (nowTs) => {
      if (!appState.summaryVisible) return;
      const p = clamp((nowTs - animStart) / animDuration, 0, 1);
      const eased = 1 - Math.pow(1 - p, 3);

      const curPurity = Math.round(summary.purity * eased);
      const curDur = summary.durationSec * eased;
      const curAct = Math.round(summary.totalActions * eased);
      const curUniq = Math.round(summary.uniqueCount * eased);
      const curConf = summary.avgConf * 100 * eased;

      dom.summaryPurityNum.textContent = `${curPurity}%`;
      dom.summaryDuration.textContent = `${curDur.toFixed(1)}s`;
      dom.summaryActions.textContent = `${curAct} 次`;
      dom.summaryUnique.textContent = `${curUniq} / ${summary.totalPoseTypes}`;
      dom.summaryAvgConf.textContent = `${curConf.toFixed(1)}%`;

      if (dom.summaryRingArc) {
        const offset = SUMMARY_RING_CIRCUMFERENCE * (1 - targetRingRatio * eased);
        dom.summaryRingArc.style.strokeDashoffset = offset.toFixed(2);
      }

      if (p < 1) {
        appState.summaryRafId = window.requestAnimationFrame(tick);
      } else {
        appState.summaryRafId = null;
        // 数值定格后“咚！”一声盖下段位印章 + 震动 + 舞台粒子烟花
        appState.summaryStampTimer = window.setTimeout(() => {
          if (!appState.summaryVisible) return;
          if (dom.summaryStamp) {
            dom.summaryStamp.classList.remove("hidden");
            void dom.summaryStamp.offsetWidth;
            dom.summaryStamp.classList.add("is-stamped");
          }
          dom.summaryCard.classList.add("is-shaking");
          if (summary.purity > 0) {
            spawnBurstParticles(tier.color, Boolean(tier.isRainbow));
          }
        }, 80);
      }
    };
    appState.summaryRafId = window.requestAnimationFrame(tick);
  }

  async function saveSummaryCertificate() {
    const summary = appState.gameSession.lastSummary;
    if (!summary) return;

    const W = 1080;
    const H = summary.mvp ? 780 : 620;
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const c = canvas.getContext("2d");

    const drawRoundRect = (x, y, w, h, r) => {
      c.beginPath();
      c.moveTo(x + r, y);
      c.arcTo(x + w, y, x + w, y + h, r);
      c.arcTo(x + w, y + h, x, y + h, r);
      c.arcTo(x, y + h, x, y, r);
      c.arcTo(x, y, x + w, y, r);
      c.closePath();
    };

    // 1. 深空赛博背景
    const bgGrad = c.createLinearGradient(0, 0, W, H);
    bgGrad.addColorStop(0, "#0b1120");
    bgGrad.addColorStop(1, "#151d36");
    c.fillStyle = bgGrad;
    c.fillRect(0, 0, W, H);

    // 外框
    drawRoundRect(22, 22, W - 44, H - 44, 28);
    c.strokeStyle = summary.tier.isRainbow ? "#c084fc" : summary.tier.color;
    c.lineWidth = 4;
    c.stroke();

    // 2. 顶部标题与时间
    const d = summary.finishedAt || new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const tsStr = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;

    c.fillStyle = "#facc15";
    c.font = "700 18px 'Microsoft YaHei', sans-serif";
    c.fillText("NAILONG BLOODLINE REPORT", 58, 76);

    c.fillStyle = "#94a3b8";
    c.font = "500 18px 'Microsoft YaHei', sans-serif";
    c.textAlign = "right";
    c.fillText(tsStr, W - 58, 76);
    c.textAlign = "left";

    c.fillStyle = "#ffffff";
    c.font = "900 38px 'Microsoft YaHei', sans-serif";
    c.fillText("奶龙家人血统鉴定证书", 58, 126);

    // 3. 左侧纯度圆环
    const cx = 235;
    const cy = 315;
    const rad = 115;
    c.beginPath();
    c.arc(cx, cy, rad, 0, Math.PI * 2);
    c.strokeStyle = "rgba(148, 163, 184, 0.2)";
    c.lineWidth = 18;
    c.stroke();

    const pctRatio = clamp(summary.purity / 100, 0, 1);
    const arcGrad = c.createLinearGradient(cx - rad, cy - rad, cx + rad, cy + rad);
    if (summary.tier.isRainbow) {
      arcGrad.addColorStop(0, "#ff3b6b");
      arcGrad.addColorStop(0.5, "#facc15");
      arcGrad.addColorStop(1, "#38bdf8");
    } else {
      arcGrad.addColorStop(0, "#fef08a");
      arcGrad.addColorStop(1, summary.tier.color);
    }
    c.beginPath();
    c.arc(cx, cy, rad, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * pctRatio);
    c.strokeStyle = arcGrad;
    c.lineWidth = 18;
    c.lineCap = "round";
    c.stroke();

    c.textAlign = "center";
    c.fillStyle = "#cbd5e1";
    c.font = "700 18px 'Microsoft YaHei', sans-serif";
    c.fillText("奶家人纯度", cx, cy - 26);

    c.fillStyle = "#ffffff";
    c.font = "900 62px 'Microsoft YaHei', sans-serif";
    c.fillText(`${summary.purity}%`, cx, cy + 32);

    // 倾斜段位印章
    c.save();
    c.translate(cx + 88, cy + 78);
    c.rotate(-0.2);
    drawRoundRect(-84, -24, 168, 48, 10);
    c.fillStyle = "rgba(15, 23, 42, 0.92)";
    c.fill();
    c.strokeStyle = summary.tier.color;
    c.lineWidth = 3;
    c.stroke();
    c.fillStyle = summary.tier.color;
    c.font = "900 22px 'Microsoft YaHei', sans-serif";
    c.fillText(summary.tier.stamp, 0, 8);
    c.restore();

    // 4. 右侧 2x2 数据卡
    c.textAlign = "left";
    const cards = [
      { k: "本局用时", v: `${summary.durationSec.toFixed(1)}s` },
      { k: "完成动作", v: `${summary.totalActions} 次` },
      { k: "解锁种类", v: `${summary.uniqueCount} / ${summary.totalPoseTypes}` },
      { k: "平均识别率", v: `${(summary.avgConf * 100).toFixed(1)}%` },
    ];
    cards.forEach((item, idx) => {
      const col = idx % 2;
      const row = Math.floor(idx / 2);
      const x = 435 + col * 300;
      const y = 195 + row * 126;
      drawRoundRect(x, y, 280, 108, 18);
      c.fillStyle = "rgba(30, 41, 59, 0.78)";
      c.fill();
      c.strokeStyle = "rgba(148, 163, 184, 0.24)";
      c.lineWidth = 1.5;
      c.stroke();

      c.fillStyle = "#94a3b8";
      c.font = "700 18px 'Microsoft YaHei', sans-serif";
      c.fillText(item.k, x + 24, y + 38);

      c.fillStyle = "#fef08a";
      c.font = "900 36px 'Microsoft YaHei', sans-serif";
      c.fillText(item.v, x + 24, y + 86);
    });

    // 5. 评语框
    drawRoundRect(58, 472, W - 116, 98, 18);
    c.fillStyle = "rgba(30, 41, 59, 0.85)";
    c.fill();
    c.strokeStyle = summary.tier.color;
    c.lineWidth = 2;
    c.stroke();

    c.fillStyle = summary.tier.color;
    c.font = "900 22px 'Microsoft YaHei', sans-serif";
    c.fillText(`【${summary.tier.rank}】`, 82, 512);

    c.fillStyle = "#e2e8f0";
    c.font = "600 20px 'Microsoft YaHei', sans-serif";
    c.fillText(summary.tier.comment, 82, 548);

    // 6. 本命招牌奶蛙（MVP）
    if (summary.mvp) {
      drawRoundRect(58, 594, W - 116, 134, 18);
      c.fillStyle = "rgba(15, 23, 42, 0.82)";
      c.fill();
      c.strokeStyle = "rgba(250, 204, 21, 0.35)";
      c.lineWidth = 1.5;
      c.stroke();

      const imgUrl = sanitizeUrl(summary.mvp.image_url);
      if (imgUrl) {
        try {
          const mvpImg = await new Promise((resolve, reject) => {
            const im = new Image();
            im.crossOrigin = "anonymous";
            im.onload = () => resolve(im);
            im.onerror = reject;
            im.src = imgUrl;
          });
          const boxX = 80;
          const boxY = 609;
          const boxW = 80;
          const boxH = 104;
          const scale = Math.min(boxW / mvpImg.width, boxH / mvpImg.height);
          const dw = mvpImg.width * scale;
          const dh = mvpImg.height * scale;
          c.drawImage(mvpImg, boxX + (boxW - dw) / 2, boxY + (boxH - dh) / 2, dw, dh);
        } catch {
          // 忽略图片绘制失败
        }
      }

      c.fillStyle = "#facc15";
      c.font = "800 18px 'Microsoft YaHei', sans-serif";
      c.fillText("★ 本局本命形态（最高匹配）", 184, 636);

      c.fillStyle = "#ffffff";
      c.font = "900 30px 'Microsoft YaHei', sans-serif";
      c.fillText(summary.mvp.name, 184, 676);

      c.fillStyle = "#cbd5e1";
      c.font = "600 19px 'Microsoft YaHei', sans-serif";
      c.fillText(
        `最高匹配度 ${formatPercent(summary.mvp.confidence, 1)} · 本局触发 ${summary.mvp.count} 次`,
        184,
        710
      );
    }

    const dataUrl = canvas.toDataURL("image/png");
    let savedToCaptures = false;
    try {
      await apiFetch("/api/captures/save_certificate", {
        method: "POST",
        body: JSON.stringify({ data_url: dataUrl }),
      });
      savedToCaptures = true;
      loadCaptures(true);
    } catch {
      // 仅浏览器下载兜底
    }

    const link = document.createElement("a");
    const fname = `nailong_purity_${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.png`;
    link.download = fname;
    link.href = dataUrl;
    link.click();
    showToast(
      savedToCaptures ? "鉴定证书已保存到 captures 文件夹并开始下载！" : "鉴定证书已保存到本地！",
      "success",
      2800
    );
  }

  // =========================================================================
  // 业务操作：开始/停止游戏 / 试播预览 / 采集 / 清空 / 训练 / 补齐配音
  // =========================================================================
  async function startGame() {
    unlockAudioContext(true);
    closeBurstOverlay();
    closeSummaryModal();
    appState.gameSession = {
      startedAt: 0,
      events: [],
      lastSummary: appState.gameSession.lastSummary,
    };
    appState.previewEventKeys.clear();
    try {
      const res = await apiFetch("/api/game/start", {
        method: "POST",
        body: JSON.stringify({ countdown: 3.0 }),
      });
      if (res && res.game) {
        appState.live.game = res.game;
        updateGameUi(res.game, Boolean(appState.live.collect && appState.live.collect.active));
      }
    } catch (err) {
      showToast(`开始游戏失败：${err.message}`, "error");
    }
  }

  async function stopGame() {
    const hadStartedRunning = appState.gameSession.startedAt > 0;
    const summarySnapshot = hadStartedRunning ? computeGameSummary() : null;

    closeBurstOverlay();
    stopCurrentAudio();
    if (appState.gameGoTimer) {
      window.clearTimeout(appState.gameGoTimer);
      appState.gameGoTimer = null;
    }
    try {
      const res = await apiFetch("/api/game/stop", { method: "POST" });
      if (res && res.game) {
        appState.live.game = res.game;
        updateGameUi(res.game, Boolean(appState.live.collect && appState.live.collect.active));
      }
      appState.gameSession.startedAt = 0;
      appState.gameSession.events = [];

      if (summarySnapshot) {
        openSummaryModal(summarySnapshot);
      } else {
        showToast("已取消准备，切回安全待机预览", "info", 2000);
      }
    } catch (err) {
      showToast(`停止游戏失败：${err.message}`, "error");
    }
  }

  async function triggerPreview(poseId, btnEl) {
    unlockAudioContext(true);
    if (btnEl) {
      btnEl.disabled = true;
      btnEl.textContent = "试播中...";
    }
    try {
      const res = await apiFetch(`/api/preview/${encodeURIComponent(poseId)}`, {
        method: "POST",
      });
      // 若 WS 未连接或后端仅在 HTTP 返回中给出事件，兜底直接触发弹窗（标记 fromPreview 不计入游戏战绩）
      const maybeEvent = res && (res.triggered_event || res.event || (res.pose_id ? res : null));
      if (maybeEvent && maybeEvent.pose_id) {
        handleTriggeredEvent(maybeEvent, { fromPreview: true });
      }
    } catch (err) {
      showToast(`试播失败：${err.message}`, "error");
    } finally {
      if (btnEl) {
        btnEl.disabled = false;
        btnEl.innerHTML = "<span>▶</span><span>试播预览</span>";
      }
    }
  }

  async function startCollect(poseId) {
    unlockAudioContext(true);
    try {
      await apiFetch("/api/collect/start", {
        method: "POST",
        body: JSON.stringify({ pose_id: poseId, frames: 80, countdown: 3.0 }),
      });
      const pose = appState.poseMap.get(poseId);
      showToast(`即将录制「${pose ? pose.name : poseId}」80 帧，请做好准备！`, "info");
    } catch (err) {
      showToast(`启动采集失败：${err.message}`, "error");
    }
  }

  async function stopCollect() {
    appState.manualStopCollect = true;
    try {
      await apiFetch("/api/collect/stop", { method: "POST" });
      appState.wasCollecting = false;
      appState.lastCollectPoseId = null;
      appState.live.collect.active = false;
      renderCollectList();
      showToast("已停止动作录制，切回实战模式", "info");
      await loadConfig(true);
    } catch (err) {
      appState.manualStopCollect = false;
      showToast(`停止录制失败：${err.message}`, "error");
    }
  }

  async function clearSamples(poseId) {
    const pose = appState.poseMap.get(poseId);
    const name = pose ? pose.name : poseId;
    if (!window.confirm(`确定清空「${name}」的全部已录样本帧吗？`)) return;

    try {
      const res = await apiFetch(`/api/samples/${encodeURIComponent(poseId)}`, {
        method: "DELETE",
      });
      if (res && res.sample_counts && typeof res.sample_counts === "object") {
        appState.sampleCounts = res.sample_counts;
        appState.sampleTotal = Object.values(res.sample_counts).reduce((a, b) => a + (Number(b) || 0), 0);
        renderCollectList();
      } else {
        await loadConfig(true);
      }
      showToast(`已清空「${name}」的样本数据`, "success");
    } catch (err) {
      showToast(`清空样本失败：${err.message}`, "error");
    }
  }

  async function trainModel() {
    if (appState.isTraining) return;
    appState.isTraining = true;
    dom.btnTrainModel.disabled = true;
    dom.btnTrainLabel.textContent = "训练对比中 (RF vs MLP)...";

    try {
      const res = await apiFetch("/api/train", { method: "POST" });
      appState.modelInfo = { ...res, loaded: true };
      renderModelBadge();
      renderEvalSection(res);

      const bestName = res.best_model || "模型";
      const acc = formatPercent(res.best_accuracy, 1);
      showToast(`训练完成！优胜模型 ${bestName}（准确率 ${acc}），已热加载上线！`, "success", 4500);
      await loadConfig(true);
      renderEvalSection(res);
    } catch (err) {
      showToast(`训练失败：${err.message}`, "error", 4800);
    } finally {
      appState.isTraining = false;
      dom.btnTrainModel.disabled = false;
      dom.btnTrainLabel.textContent = "一键训练模型";
    }
  }

  async function generateMissingTts() {
    if (appState.isGeneratingTts) return;
    appState.isGeneratingTts = true;
    dom.btnTtsGen.disabled = true;
    dom.btnTtsLabel.textContent = "配音生成中...";

    try {
      const res = await apiFetch("/api/tts/generate", { method: "POST" });
      let detail = "配音检查与补齐完成！";
      let toastType = "success";
      if (res && typeof res === "object" && !Array.isArray(res) && ("generated" in res || "errors" in res || "skipped" in res)) {
        const generated = Number(res.generated || 0);
        const skipped = Number(res.skipped || 0);
        const errors = Number(res.errors || 0);
        if (errors > 0) {
          toastType = "error";
          detail = `配音补齐出现 ${errors} 条失败（已生成 ${generated} 条，跳过 ${skipped} 条），请检查网络后重试`;
        } else if (generated > 0) {
          detail = `成功补齐 ${generated} 条奶蛙配音（保留已有 ${skipped} 条原声）！`;
        } else {
          detail = `全部 ${skipped} 条奶蛙原声均已就绪，无需重复生成！`;
        }
      } else if (Array.isArray(res)) {
        const created = res.filter((r) => r && (r.status === "generated" || r.generated === true)).length;
        const errors = res.filter((r) => r && r.status === "error").length;
        if (errors > 0) {
          toastType = "error";
          detail = `配音补齐出现 ${errors} 条失败（已生成 ${created} 条）`;
        } else {
          detail = created > 0 ? `成功补齐 ${created} 条奶蛙配音！` : "全部 10 条奶蛙原声均已就绪，无需重复生成！";
        }
      } else if (res && typeof res.message === "string") {
        detail = res.message;
      }
      showToast(detail, toastType, 4000);
      await loadConfig(true);
    } catch (err) {
      showToast(`补齐配音失败：${err.message}`, "error");
    } finally {
      appState.isGeneratingTts = false;
      dom.btnTtsGen.disabled = false;
      dom.btnTtsLabel.textContent = "补齐配音";
    }
  }

  // =========================================================================
  // 灯箱画廊（支持同框照切换 + 混淆矩阵单图查看 + 一键下载）
  // =========================================================================
  function openLightbox(items, startIndex = 0) {
    if (!items || !items.length) return;
    appState.lightboxItems = items;
    appState.lightboxIndex = clamp(startIndex, 0, items.length - 1);
    renderLightboxSlide();
    dom.lightbox.classList.remove("hidden");
  }

  function renderLightboxSlide() {
    const items = appState.lightboxItems;
    const item = items[appState.lightboxIndex];
    if (!item) return;

    const safeUrl = sanitizeUrl(item.url);
    dom.lightboxImg.src = safeUrl;
    dom.lightboxTitle.textContent = item.title || "同框照预览";
    dom.lightboxSub.textContent = item.sub || "";
    dom.lightboxDownload.href = safeUrl;
    dom.lightboxDownload.setAttribute("download", item.downloadName || "nailong_snapshot.jpg");

    const hasMultiple = items.length > 1;
    dom.btnLightboxPrev.classList.toggle("hidden", !hasMultiple);
    dom.btnLightboxNext.classList.toggle("hidden", !hasMultiple);
  }

  function closeLightbox() {
    dom.lightbox.classList.add("hidden");
    appState.lightboxItems = [];
    appState.lightboxIndex = -1;
  }

  function stepLightbox(delta) {
    const len = appState.lightboxItems.length;
    if (len <= 1) return;
    appState.lightboxIndex = (appState.lightboxIndex + delta + len) % len;
    renderLightboxSlide();
  }

  // =========================================================================
  // 全屏切换与事件绑定
  // =========================================================================
  function updateFullscreenUi() {
    const isFs = Boolean(document.fullscreenElement);
    dom.iconFsEnter.classList.toggle("hidden", isFs);
    dom.iconFsExit.classList.toggle("hidden", !isFs);
    dom.fsLabel.textContent = isFs ? "退出全屏" : "全屏";
    window.setTimeout(resizeSkeletonCanvas, 80);
  }

  function bindEvents() {
    // 任意首次交互静默解锁音频权限
    const handleFirstGesture = () => {
      unlockAudioContext(true);
    };
    window.addEventListener("pointerdown", handleFirstGesture, { once: true });
    window.addEventListener("keydown", handleFirstGesture, { once: true });

    dom.audioUnlockPill.addEventListener("click", () => {
      unlockAudioContext(false);
    });

    // 游戏开始 / 停止超大主控按钮
    if (dom.btnGameStart) {
      dom.btnGameStart.addEventListener("click", startGame);
    }
    if (dom.btnGameStop) {
      dom.btnGameStop.addEventListener("click", stopGame);
    }

    // 声音开关
    dom.btnSound.addEventListener("click", () => {
      appState.muted = !appState.muted;
      saveMutedPref(appState.muted);
      if (appState.muted) {
        stopCurrentAudio();
      } else {
        unlockAudioContext(true);
      }
      updateSoundUi();
      showToast(appState.muted ? "已静音，不再自动播放配音" : "已开启奶蛙原声播报", "info", 1800);
    });

    // 全屏按钮
    dom.btnFullscreen.addEventListener("click", async () => {
      try {
        if (!document.fullscreenElement) {
          await document.documentElement.requestFullscreen();
        } else {
          await document.exitFullscreen();
        }
      } catch {
        showToast("当前浏览器不支持全屏切换", "error");
      }
    });
    document.addEventListener("fullscreenchange", updateFullscreenUi);

    // 控制台双标签切换
    dom.tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.dataset.tab;
        dom.tabButtons.forEach((b) => {
          const active = b === btn;
          b.classList.toggle("active", active);
          b.setAttribute("aria-selected", active ? "true" : "false");
        });
        dom.panelDex.classList.toggle("active", target === "dex");
        dom.panelTrain.classList.toggle("active", target === "train");
      });
    });

    // 图鉴事件委托（试播预览）
    dom.dexGrid.addEventListener("click", (e) => {
      const btn = e.target.closest('[data-action="preview"]');
      if (!btn) return;
      const poseId = btn.dataset.poseId;
      if (poseId) triggerPreview(poseId, btn);
    });

    // 重置图鉴解锁计数
    dom.btnResetDex.addEventListener("click", () => {
      if (!window.confirm("确定重置本机的奶蛙图鉴解锁记录吗？")) return;
      appState.unlocked = {};
      saveUnlockedMap();
      renderDexGrid();
      showToast("已重置图鉴解锁进度，准备新一轮鉴定！", "info");
    });

    // 同框照画廊点击 -> 打开灯箱
    dom.galleryGrid.addEventListener("click", (e) => {
      const itemEl = e.target.closest(".gallery-item");
      if (!itemEl) return;
      const idx = Number(itemEl.dataset.index || 0);
      const slides = appState.captures.map((c) => ({
        url: c.url,
        title: formatCaptureCaption(c),
        sub: c.filename || "",
        downloadName: c.filename || "nailong_match.jpg",
      }));
      openLightbox(slides, idx);
    });

    dom.btnRefreshGallery.addEventListener("click", () => {
      loadCaptures(false);
    });

    // 爆发弹窗关闭与查看/放大同框合照
    const openCurrentBurstPoster = () => {
      if (!appState.burstCaptureUrl) return;
      openLightbox(
        [
          {
            url: appState.burstCaptureUrl,
            title: `${dom.burstTitle.textContent} · 刚刚抓拍的同框合照`,
            sub: "双栏鉴定纪念海报",
            downloadName: appState.burstCaptureUrl.split("/").pop() || "nailong_match.jpg",
          },
        ],
        0
      );
    };

    dom.btnBurstClose.addEventListener("click", closeBurstOverlay);
    dom.burstOverlay.addEventListener("click", (e) => {
      if (e.target === dom.burstOverlay) closeBurstOverlay();
    });
    dom.burstCaptureChip.addEventListener("click", openCurrentBurstPoster);
    if (dom.burstZoomBtn) {
      dom.burstZoomBtn.addEventListener("click", openCurrentBurstPoster);
    }
    if (dom.burstPosterImg) {
      dom.burstPosterImg.addEventListener("click", openCurrentBurstPoster);
    }

    // 采集列表事件委托
    dom.collectList.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-action]");
      if (!btn) return;
      const action = btn.dataset.action;
      const poseId = btn.dataset.poseId;
      if (action === "start-collect" && poseId) startCollect(poseId);
      else if (action === "stop-collect") stopCollect();
      else if (action === "clear-samples" && poseId) clearSamples(poseId);
    });

    dom.btnStopCollect.addEventListener("click", stopCollect);
    dom.btnOverlayStopCollect.addEventListener("click", stopCollect);

    // 训练 & 配音按钮
    dom.btnTrainModel.addEventListener("click", trainModel);
    dom.btnTtsGen.addEventListener("click", generateMissingTts);

    // 混淆矩阵点击放大
    dom.evalConfusionImg.addEventListener("click", () => {
      if (!dom.evalConfusionImg.src) return;
      openLightbox(
        [
          {
            url: dom.evalConfusionImg.src,
            title: "交叉验证混淆矩阵（Confusion Matrix）",
            sub: dom.evalCvChip.textContent || "",
            downloadName: "eval_confusion.png",
          },
        ],
        0
      );
    });

    // 总结弹窗交互
    if (dom.btnSummaryClose) dom.btnSummaryClose.addEventListener("click", closeSummaryModal);
    if (dom.summaryBackdrop) dom.summaryBackdrop.addEventListener("click", closeSummaryModal);
    if (dom.btnSummaryReplay) {
      dom.btnSummaryReplay.addEventListener("click", () => {
        closeSummaryModal();
        startGame();
      });
    }
    if (dom.btnSummarySave) {
      dom.btnSummarySave.addEventListener("click", saveSummaryCertificate);
    }

    // 灯箱交互与快捷键
    dom.btnLightboxClose.addEventListener("click", closeLightbox);
    dom.lightboxBackdrop.addEventListener("click", closeLightbox);
    dom.btnLightboxPrev.addEventListener("click", () => stepLightbox(-1));
    dom.btnLightboxNext.addEventListener("click", () => stepLightbox(1));

    window.addEventListener("keydown", (e) => {
      if (!dom.lightbox.classList.contains("hidden")) {
        if (e.key === "Escape") closeLightbox();
        else if (e.key === "ArrowLeft") stepLightbox(-1);
        else if (e.key === "ArrowRight") stepLightbox(1);
      } else if (appState.summaryVisible && e.key === "Escape") {
        closeSummaryModal();
      } else if (appState.burstVisible && e.key === "Escape") {
        closeBurstOverlay();
      }
    });

    // 视频尺寸和视口尺寸同步监听
    dom.videoFeed.addEventListener("load", resizeSkeletonCanvas);
    dom.videoFeed.addEventListener("error", () => {
      // 视频流偶发中断时自动重连提示
      dom.personPillText.textContent = "等待视频流连接...";
    });

    if (typeof ResizeObserver !== "undefined") {
      const ro = new ResizeObserver(() => resizeSkeletonCanvas());
      ro.observe(dom.stageViewport);
    }
    window.addEventListener("resize", resizeSkeletonCanvas);
  }

  // =========================================================================
  // 启动入口
  // =========================================================================
  async function init() {
    updateSoundUi();
    updateFullscreenUi();
    bindEvents();
    resizeSkeletonCanvas();
    window.requestAnimationFrame(renderLoop);

    await Promise.all([loadConfig(false), loadCaptures(true)]);
    connectWebSocket();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
