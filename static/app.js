const elements = {
    folderPathInput: document.getElementById("folderPathInput"),
    brushCursor: document.getElementById("brushCursor"),
    scanFolderBtn: document.getElementById("scanFolderBtn"),
    redBandSelect: document.getElementById("redBandSelect"),
    greenBandSelect: document.getElementById("greenBandSelect"),
    blueBandSelect: document.getElementById("blueBandSelect"),
    loadSceneBtn: document.getElementById("loadSceneBtn"),
    sceneInfo: document.getElementById("sceneInfo"),
    classList: document.getElementById("classList"),
    newClassName: document.getElementById("newClassName"),
    newClassColor: document.getElementById("newClassColor"),
    addClassBtn: document.getElementById("addClassBtn"),
    brushToolBtn: document.getElementById("brushToolBtn"),
    eraserToolBtn: document.getElementById("eraserToolBtn"),
    brushSizeInput: document.getElementById("brushSizeInput"),
    brushSizeValue: document.getElementById("brushSizeValue"),
    labelOpacityInput: document.getElementById("labelOpacityInput"),
    labelOpacityValue: document.getElementById("labelOpacityValue"),
    clearLabelsBtn: document.getElementById("clearLabelsBtn"),
    copyPredictionBtn: document.getElementById("copyPredictionBtn"),
    trainBandsSelect: document.getElementById("trainBandsSelect"),
    neighborhoodSizeSelect: document.getElementById("neighborhoodSizeSelect"),
    nEstimatorsInput: document.getElementById("nEstimatorsInput"),
    maxDepthInput: document.getElementById("maxDepthInput"),
    maxSamplesInput: document.getElementById("maxSamplesInput"),
    confThresholdInput: document.getElementById("confThresholdInput"),
    confThresholdValue: document.getElementById("confThresholdValue"),
    trainBtn: document.getElementById("trainBtn"),
    inferProgressWrap: document.getElementById("inferProgressWrap"),
    inferProgressBar: document.getElementById("inferProgressBar"),
    inferProgressLabel: document.getElementById("inferProgressLabel"),
    showPredictionInput: document.getElementById("showPredictionInput"),
    showLabelsInput: document.getElementById("showLabelsInput"),
    predictionOpacityInput: document.getElementById("predictionOpacityInput"),
    predictionOpacityValue: document.getElementById("predictionOpacityValue"),
    downloadLabelsBtn: document.getElementById("downloadLabelsBtn"),
    trainStats: document.getElementById("trainStats"),
    fitViewBtn: document.getElementById("fitViewBtn"),
    zoomOutBtn: document.getElementById("zoomOutBtn"),
    zoomInBtn: document.getElementById("zoomInBtn"),
    zoomSlider: document.getElementById("zoomSlider"),
    zoomLabel: document.getElementById("zoomLabel"),
    statusPill: document.getElementById("statusPill"),
    cursorInfo: document.getElementById("cursorInfo"),
    viewport: document.getElementById("viewport"),
    canvasStack: document.getElementById("canvasStack"),
    imageCanvas: document.getElementById("imageCanvas"),
    predictionCanvas: document.getElementById("predictionCanvas"),
    labelCanvas: document.getElementById("labelCanvas"),
    scanBatchBtn: document.getElementById("scanBatchBtn"),
    folderNavRow: document.getElementById("folderNavRow"),
    folderNavInfo: document.getElementById("folderNavInfo"),
    prevFolderBtn: document.getElementById("prevFolderBtn"),
    nextFolderBtn: document.getElementById("nextFolderBtn")
};

const imageContext = elements.imageCanvas.getContext("2d");
const predictionContext = elements.predictionCanvas.getContext("2d", { willReadFrequently: true });
const labelContext = elements.labelCanvas.getContext("2d", { willReadFrequently: true });
imageContext.imageSmoothingEnabled = false;
predictionContext.imageSmoothingEnabled = false;
labelContext.imageSmoothingEnabled = false;

const state = {
    sceneBands: [],
    allBands: [],
    sessionId: null,
    sceneFolderPath: "",
    bandLabels: [],
    imageWidth: 0,
    imageHeight: 0,
    previewImage: null,
    labelMask: null,
    predictionMask: null,
    labelLookup: buildBlankLookup(),
    predictionLookup: buildBlankLookup(),
    classes: [
        { id: 1, name: "类别 1", color: "#ff6b4a" },
        { id: 2, name: "类别 2", color: "#1f8a70" },
        { id: 3, name: "类别 3", color: "#2b59c3" }
    ],
    nextClassId: 4,
    activeClassId: 1,
    tool: "brush",
    brushSize: 6,
    labelOpacity: 0.74,
    predictionOpacity: 0.48,
    showLabels: true,
    showPrediction: true,
    zoom: 1,
    drawing: false,
    lastPoint: null,
    isBusy: false,
    isInferring: false,
    predictionTifUrl: null,
    folderQueue: [],
    currentFolderIndex: -1,
    labelsDirty: false
};

// 平移状态：fromPointer=true 表示空格+左键（canvas capture），false 表示中键（window mouse）
const panState = { active: false, fromPointer: false, startX: 0, startY: 0, scrollX: 0, scrollY: 0 };
let spaceHeld = false;

// 每个层单独维护自己的 ImageData
const labelImageData = { data: null };
const predictionImageData = { data: null };

function buildBlankLookup() {
    return new Uint8ClampedArray(256 * 4);
}

function hexToRgb(hex) {
    const normalized = hex.replace("#", "");
    const value = normalized.length === 3
        ? normalized.split("").map((char) => char + char).join("")
        : normalized;
    return {
        r: Number.parseInt(value.slice(0, 2), 16),
        g: Number.parseInt(value.slice(2, 4), 16),
        b: Number.parseInt(value.slice(4, 6), 16)
    };
}

function refreshColorLookup() {
    state.labelLookup = buildBlankLookup();
    state.predictionLookup = buildBlankLookup();
    // 不再将 alpha 内嵌到象素，透明度由 canvas.style.opacity 控制
    for (const category of state.classes) {
        const { r, g, b } = hexToRgb(category.color);
        const offset = category.id * 4;
        state.labelLookup[offset] = r;
        state.labelLookup[offset + 1] = g;
        state.labelLookup[offset + 2] = b;
        state.labelLookup[offset + 3] = 255;

        state.predictionLookup[offset] = r;
        state.predictionLookup[offset + 1] = g;
        state.predictionLookup[offset + 2] = b;
        state.predictionLookup[offset + 3] = 255;
    }
}

function setStatus(message, tone = "") {
    elements.statusPill.textContent = message;
    elements.statusPill.className = `status-pill ${tone}`.trim();
}

function setBusy(busy, message) {
    state.isBusy = busy;
    const disabled = busy;
    elements.loadSceneBtn.disabled = disabled;
    elements.trainBtn.disabled = disabled;
    elements.scanFolderBtn.disabled = disabled;
    elements.copyPredictionBtn.disabled = disabled || !state.predictionMask;
    if (message) {
        setStatus(message, busy ? "busy" : "");
    }
    updateFolderNavUI();
}

function setInferring(active) {
    state.isInferring = active;
    // 推理中隐藏/显示进度条
    elements.inferProgressWrap.classList.toggle("hidden", !active);
    if (!active) {
        elements.inferProgressBar.style.width = "0%";
        elements.inferProgressLabel.textContent = "";
    }
}

function updateProgress(pct, message) {
    elements.inferProgressBar.style.width = `${pct}%`;
    elements.inferProgressLabel.textContent = message || "";
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(payload.error || `请求失败：${response.status}`);
    }
    return payload;
}

function formatNumber(value) {
    return Number(value || 0).toLocaleString("zh-CN");
}



function populateBandSelect(selectElement, bands, selectedKey) {
    selectElement.innerHTML = "";
    for (const band of bands) {
        const option = document.createElement("option");
        option.value = band.key;
        option.textContent = `${band.relativePath} · ${band.width}×${band.height}`;
        if (band.key === selectedKey) {
            option.selected = true;
        }
        selectElement.append(option);
    }
}

function populateTrainBandSelect(bands) {
    elements.trainBandsSelect.innerHTML = "";
    for (const band of bands) {
        const option = document.createElement("option");
        option.value = band.key;
        const bNum = band.bandNumber != null ? `B${band.bandNumber}` : "B?";
        option.textContent = `${bNum} · ${band.name} · ${band.width}×${band.height}`;
        option.selected = true;
        elements.trainBandsSelect.append(option);
    }
}

function updateSceneInfo(scene) {
    if (!scene) {
        elements.sceneInfo.textContent = "尚未选择场景。";
        return;
    }
    const previewKeys = (scene.defaultBandKeys || []).join(" / ") || "无默认推荐";
    elements.sceneInfo.innerHTML = [
        `<div><strong>${scene.name}</strong></div>`,
        `<div>文件夹：${scene.folderPath}</div>`,
        `<div>可选波段：${scene.bandCount}</div>`,
        `<div>推荐组合：${previewKeys}</div>`
    ].join("");
}



// 点击"扫描"时的智能入口：优先批量模式，否则直接扫描并加载
async function handleScanClick() {
    const targetPath = (elements.folderPathInput.value || "").trim();
    if (!targetPath) {
        setStatus("请先输入文件夹路径", "error");
        return;
    }

    setBusy(true, "正在检测子文件夹…");
    let batchPayload;
    try {
        batchPayload = await fetchJson("/api/list-subfolders", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ parentPath: targetPath })
        });
    } catch (err) {
        setStatus(err.message, "error");
        setBusy(false);
        return;
    }
    setBusy(false);

    if (batchPayload.subfolders && batchPayload.subfolders.length > 0) {
        // 批量模式：扫到子文件夹，加载第一个
        state.folderQueue = batchPayload.subfolders;
        state.currentFolderIndex = -1;
        updateFolderNavUI();
        await navigateToFolder(0);
    } else {
        // 单文件夹模式：直接扫描波段并加载影像
        await scanFolder(targetPath);
        if (state.sceneBands.length >= 3) {
            await loadScene();
        }
    }
}

async function scanFolder(folderPath) {
    const targetPath = (folderPath || elements.folderPathInput.value || "").trim();
    if (!targetPath) {
        setStatus("请先输入或选择一个文件夹", "error");
        return;
    }

    setBusy(true, "正在读取波段列表");
    try {
        const payload = await fetchJson("/api/scan-folder", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ folderPath: targetPath })
        });
        const scene = payload.scene;
        state.sceneBands = scene.bands || [];
        if (state.sceneBands.length < 3) {
            throw new Error("该文件夹内可用波段不足 3 个。")
        }
        const defaults = scene.defaultBandKeys || state.sceneBands.slice(0, 3).map((band) => band.key);
        populateBandSelect(elements.redBandSelect, state.sceneBands, defaults[0]);
        populateBandSelect(elements.greenBandSelect, state.sceneBands, defaults[1]);
        populateBandSelect(elements.blueBandSelect, state.sceneBands, defaults[2]);
        state.allBands = state.sceneBands;
        populateTrainBandSelect(state.allBands);
        updateSceneInfo(scene);
        state.sceneFolderPath = scene.folderPath;
        elements.folderPathInput.value = scene.folderPath;
        setStatus(`已读取 ${scene.bandCount} 个波段`, "success");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        setBusy(false);
    }
}

function createEmptyMask() {
    return new Uint8Array(state.imageWidth * state.imageHeight);
}

function configureCanvasSize() {
    const needResizeImage = elements.imageCanvas.width !== state.imageWidth || elements.imageCanvas.height !== state.imageHeight;
    const needResizeLabel = elements.labelCanvas.width !== state.imageWidth || elements.labelCanvas.height !== state.imageHeight;
    const needResizePrediction = elements.predictionCanvas.width !== state.imageWidth || elements.predictionCanvas.height !== state.imageHeight;

    if (needResizeImage) {
        elements.imageCanvas.width = state.imageWidth;
        elements.imageCanvas.height = state.imageHeight;
    }
    if (needResizeLabel) {
        elements.labelCanvas.width = state.imageWidth;
        elements.labelCanvas.height = state.imageHeight;
        labelImageData.data = null;
    }
    if (needResizePrediction) {
        elements.predictionCanvas.width = state.imageWidth;
        elements.predictionCanvas.height = state.imageHeight;
        predictionImageData.data = null;
    }

    const w = `${state.imageWidth * state.zoom}px`;
    const h = `${state.imageHeight * state.zoom}px`;
    elements.canvasStack.style.width = w;
    elements.canvasStack.style.height = h;
    elements.imageCanvas.style.width = w;
    elements.imageCanvas.style.height = h;
    elements.predictionCanvas.style.width = w;
    elements.predictionCanvas.style.height = h;
    elements.labelCanvas.style.width = w;
    elements.labelCanvas.style.height = h;

    if (needResizeImage && state.previewImage) { drawBaseImage(); }
    if (needResizeLabel && state.labelMask) { rebuildLabelLayer(); }
    if (needResizePrediction && state.predictionMask) { rebuildPredictionLayer(); }
}

function setZoom(zoomValue) {
    if (!state.imageWidth || !state.imageHeight) {
        return;
    }
    state.zoom = Math.max(0.05, Math.min(16, zoomValue));
    elements.zoomSlider.value = Math.round(state.zoom * 100);
    elements.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
    configureCanvasSize();
}

function fitView() {
    if (!state.imageWidth || !state.imageHeight) {
        return;
    }
    const availableWidth = Math.max(160, elements.viewport.clientWidth - 40);
    const availableHeight = Math.max(160, elements.viewport.clientHeight - 40);
    const zoom = Math.min(availableWidth / state.imageWidth, availableHeight / state.imageHeight, 1);
    setZoom(zoom);
    elements.viewport.scrollTop = 0;
    elements.viewport.scrollLeft = 0;
}

async function loadImage(url) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("图像加载失败"));
        image.src = url;
    });
}

function drawBaseImage() {
    imageContext.clearRect(0, 0, state.imageWidth, state.imageHeight);
    imageContext.drawImage(state.previewImage, 0, 0, state.imageWidth, state.imageHeight);
}

function ensureLabelImageData() {
    if (!labelImageData.data || labelImageData.data.width !== state.imageWidth || labelImageData.data.height !== state.imageHeight) {
        labelImageData.data = labelContext.createImageData(state.imageWidth, state.imageHeight);
    }
}

function ensurePredictionImageData() {
    if (!predictionImageData.data || predictionImageData.data.width !== state.imageWidth || predictionImageData.data.height !== state.imageHeight) {
        predictionImageData.data = predictionContext.createImageData(state.imageWidth, state.imageHeight);
    }
}

// 刷新人工标注层
function rebuildLabelLayer() {
    if (!state.labelMask || !state.imageWidth || !state.imageHeight) {
        labelContext.clearRect(0, 0, elements.labelCanvas.width, elements.labelCanvas.height);
        return;
    }
    ensureLabelImageData();
    const pixels = labelImageData.data.data;
    for (let i = 0; i < state.labelMask.length; i++) {
        const base = i * 4;
        const v = state.labelMask[i];
        if (v > 0) {
            const lut = v * 4;
            pixels[base]     = state.labelLookup[lut];
            pixels[base + 1] = state.labelLookup[lut + 1];
            pixels[base + 2] = state.labelLookup[lut + 2];
            pixels[base + 3] = 255;
        } else {
            pixels[base + 3] = 0;
        }
    }
    labelContext.putImageData(labelImageData.data, 0, 0);
}

// 刷新 AI 预测层
function rebuildPredictionLayer() {
    if (!state.predictionMask || !state.imageWidth || !state.imageHeight) {
        predictionContext.clearRect(0, 0, elements.predictionCanvas.width, elements.predictionCanvas.height);
        return;
    }
    ensurePredictionImageData();
    const pixels = predictionImageData.data.data;
    for (let i = 0; i < state.predictionMask.length; i++) {
        const base = i * 4;
        const v = state.predictionMask[i];
        // 99 为无效类，显示为透明
        if (v > 0 && v !== 99) {
            const lut = v * 4;
            pixels[base]     = state.predictionLookup[lut];
            pixels[base + 1] = state.predictionLookup[lut + 1];
            pixels[base + 2] = state.predictionLookup[lut + 2];
            pixels[base + 3] = 255;
        } else {
            pixels[base + 3] = 0;
        }
    }
    predictionContext.putImageData(predictionImageData.data, 0, 0);
}

// 局部刷新标注层的矩形块
function rebuildOverlay() {
    rebuildLabelLayer();
    rebuildPredictionLayer();
}

function updateOverlayPatch(x0, y0, x1, y1) {
    if (!labelImageData.data) {
        rebuildLabelLayer();
        return;
    }

    const minX = Math.max(0, Math.min(x0, x1));
    const minY = Math.max(0, Math.min(y0, y1));
    const maxX = Math.min(state.imageWidth - 1, Math.max(x0, x1));
    const maxY = Math.min(state.imageHeight - 1, Math.max(y0, y1));
    const pixels = labelImageData.data.data;

    for (let y = minY; y <= maxY; y++) {
        const rowOffset = y * state.imageWidth;
        for (let x = minX; x <= maxX; x++) {
            const base = (rowOffset + x) * 4;
            const v = state.labelMask[rowOffset + x];
            if (v > 0) {
                const lut = v * 4;
                pixels[base]     = state.labelLookup[lut];
                pixels[base + 1] = state.labelLookup[lut + 1];
                pixels[base + 2] = state.labelLookup[lut + 2];
                pixels[base + 3] = 255;
            } else {
                pixels[base + 3] = 0;
            }
        }
    }

    labelContext.putImageData(
        labelImageData.data,
        0, 0,
        minX, minY,
        maxX - minX + 1,
        maxY - minY + 1
    );
}

function updateCursorInfo(x, y) {
    if (x == null || y == null || !state.labelMask) {
        elements.cursorInfo.textContent = "x -, y -, 标注 -, AI -";
        return;
    }
    const index = y * state.imageWidth + x;
    const labelValue = state.labelMask[index] || 0;
    const predictionValue = state.predictionMask ? state.predictionMask[index] || 0 : 0;
    elements.cursorInfo.textContent = `x ${x}, y ${y}, 标注 ${labelValue || "-"}, AI ${predictionValue || "-"}`;
}

function clampToImage(value, maxValue) {
    return Math.max(0, Math.min(maxValue, value));
}

function paintDisk(cx, cy, targetValue) {
    const radius = state.brushSize;
    const x0 = clampToImage(cx - radius, state.imageWidth - 1);
    const y0 = clampToImage(cy - radius, state.imageHeight - 1);
    const x1 = clampToImage(cx + radius, state.imageWidth - 1);
    const y1 = clampToImage(cy + radius, state.imageHeight - 1);
    let changed = false;

    for (let y = y0; y <= y1; y += 1) {
        const dy = y - cy;
        const rowOffset = y * state.imageWidth;
        for (let x = x0; x <= x1; x += 1) {
            const dx = x - cx;
            if ((dx * dx) + (dy * dy) > radius * radius) {
                continue;
            }
            const index = rowOffset + x;
            if (state.labelMask[index] !== targetValue) {
                state.labelMask[index] = targetValue;
                changed = true;
            }
        }
    }

    return { changed, x0, y0, x1, y1 };
}

function paintStroke(fromX, fromY, toX, toY) {
    const targetValue = state.tool === "eraser" ? 0 : state.activeClassId;
    const distance = Math.max(1, Math.hypot(toX - fromX, toY - fromY));
    const step = Math.max(1, state.brushSize * 0.4);
    const steps = Math.max(1, Math.ceil(distance / step));
    let changed = false;
    let minX = state.imageWidth - 1;
    let minY = state.imageHeight - 1;
    let maxX = 0;
    let maxY = 0;

    for (let index = 0; index <= steps; index += 1) {
        const ratio = index / steps;
        const x = Math.round(fromX + ((toX - fromX) * ratio));
        const y = Math.round(fromY + ((toY - fromY) * ratio));
        const patch = paintDisk(x, y, targetValue);
        if (!patch.changed) {
            continue;
        }
        changed = true;
        minX = Math.min(minX, patch.x0);
        minY = Math.min(minY, patch.y0);
        maxX = Math.max(maxX, patch.x1);
        maxY = Math.max(maxY, patch.y1);
    }

    if (changed) {
        state.labelsDirty = true;
        updateOverlayPatch(minX, minY, maxX, maxY);
    }
}

function eventToPixel(event) {
    if (!state.imageWidth || !state.imageHeight) { return null; }
    const rect = elements.labelCanvas.getBoundingClientRect();
    const inside = event.clientX >= rect.left && event.clientX <= rect.right
        && event.clientY >= rect.top && event.clientY <= rect.bottom;
    if (!inside) { return null; }
    const x = Math.floor((event.clientX - rect.left) * (state.imageWidth / rect.width));
    const y = Math.floor((event.clientY - rect.top) * (state.imageHeight / rect.height));
    return {
        x: clampToImage(x, state.imageWidth - 1),
        y: clampToImage(y, state.imageHeight - 1)
    };
}

function renderClassList() {
    elements.classList.innerHTML = "";
    for (const category of state.classes) {
        const item = document.createElement("div");
        item.className = `class-item ${category.id === state.activeClassId ? "active" : ""}`.trim();

        const dot = document.createElement("span");
        dot.className = "class-dot";
        dot.style.background = category.color;

        const meta = document.createElement("div");
        meta.className = "class-meta";
        meta.innerHTML = `<span class="class-name">${category.name}</span><span class="class-subtitle">ID ${category.id}</span>`;

        const button = document.createElement("button");
        button.className = "class-select-btn";
        button.textContent = category.id === state.activeClassId ? "当前" : "切换";
        button.addEventListener("click", () => {
            state.activeClassId = category.id;
            renderClassList();
        });

        item.append(dot, meta, button);
        elements.classList.append(item);
    }
}

function updateToolButtons() {
    elements.brushToolBtn.classList.toggle("active", state.tool === "brush");
    elements.eraserToolBtn.classList.toggle("active", state.tool === "eraser");
}

function hasLabelPixels() {
    if (!state.labelMask) {
        return false;
    }
    for (let index = 0; index < state.labelMask.length; index += 1) {
        if (state.labelMask[index] > 0) {
            return true;
        }
    }
    return false;
}

function renderTrainStats(payload) {
    const trainLines = [];
    trainLines.push(`<div><strong>人工标注像素：</strong>${formatNumber(payload.labeledPixels)}</div>`);
    trainLines.push(`<div><strong>抽样训练像素：</strong>${formatNumber(payload.sampledPixels)}</div>`);
    trainLines.push(`<div><strong>训练精度：</strong>${(payload.trainAccuracy * 100).toFixed(2)}%</div>`);
    if (payload.oobScore != null) {
        trainLines.push(`<div><strong>OOB 评分：</strong>${(payload.oobScore * 100).toFixed(2)}%</div>`);
    }
    if (payload.labeledDistribution?.length) {
        const distribution = payload.labeledDistribution
            .map((item) => `类别 ${item.classId}: ${formatNumber(item.count)}`)
            .join("，");
        trainLines.push(`<div><strong>人工分布：</strong>${distribution}</div>`);
    }
    if (payload.predictionDistribution?.length) {
        const distribution = payload.predictionDistribution
            .map((item) => `类别 ${item.classId}: ${formatNumber(item.count)}`)
            .join("，");
        trainLines.push(`<div><strong>推理分布：</strong>${distribution}</div>`);
    }
    elements.trainStats.innerHTML = trainLines.join("");
}

async function loadScene() {
    if (state.sceneBands.length < 3) {
        setStatus("请先扫描一个包含至少 3 个波段的文件夹", "error");
        return;
    }

    const bandPaths = [
        elements.redBandSelect.value,
        elements.greenBandSelect.value,
        elements.blueBandSelect.value
    ];

    if (new Set(bandPaths).size !== 3) {
        setStatus("红绿蓝波段必须互不相同", "error");
        return;
    }

    setBusy(true, "正在生成三波段合成与 2% 拉伸图像");
    try {
        const payload = await fetchJson("/api/load-scene", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                folderPath: state.sceneFolderPath || elements.folderPathInput.value,
                bandPaths
            })
        });
        state.sessionId = payload.sessionId;
        state.bandLabels = payload.bandLabels || [];
        state.imageWidth = payload.width;
        state.imageHeight = payload.height;
        state.previewImage = await loadImage(`${payload.previewUrl}?v=${Date.now()}`);
        if (payload.allBands && payload.allBands.length > 0) {
            state.allBands = payload.allBands;
            populateTrainBandSelect(state.allBands);
        }
        state.labelMask = createEmptyMask();
        state.predictionMask = null;
        state.labelsDirty = false;
        labelImageData.data = null;
        predictionImageData.data = null;
        fitView();
        drawBaseImage();
        refreshColorLookup();
        rebuildLabelLayer();
        rebuildPredictionLayer();
        elements.trainStats.innerHTML = [
            `<div><strong>当前场景：</strong>${payload.sceneName}</div>`,
            `<div><strong>尺寸：</strong>${payload.width} × ${payload.height}</div>`,
            `<div><strong>有效像素：</strong>${formatNumber(payload.validPixels)}</div>`,
            `<div><strong>预览波段：</strong>${payload.bandLabels.join(" / ")}</div>`,
            `<div><strong>可用波段数：</strong>${(payload.allBands || []).length} 个（已全选用于训练）</div>`
        ].join("");
        elements.copyPredictionBtn.disabled = true;
        elements.downloadLabelsBtn.disabled = false;
        setStatus("合成图已加载，可以开始逐像素标注", "success");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        setBusy(false);
    }
}

async function readMaskFromImage(url) {
    const image = await loadImage(url);
    const decodeCanvas = document.createElement("canvas");
    decodeCanvas.width = image.width;
    decodeCanvas.height = image.height;
    const context = decodeCanvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(image, 0, 0);
    const imageData = context.getImageData(0, 0, image.width, image.height).data;
    const mask = new Uint8Array(image.width * image.height);
    for (let index = 0; index < mask.length; index += 1) {
        mask[index] = imageData[index * 4];
    }
    return mask;
}

async function maskToBlob(mask) {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement("canvas");
        canvas.width = state.imageWidth;
        canvas.height = state.imageHeight;
        const context = canvas.getContext("2d");
        const imageData = context.createImageData(state.imageWidth, state.imageHeight);
        for (let index = 0; index < mask.length; index += 1) {
            const offset = index * 4;
            const value = mask[index];
            imageData.data[offset] = value;
            imageData.data[offset + 1] = value;
            imageData.data[offset + 2] = value;
            imageData.data[offset + 3] = 255;
        }
        context.putImageData(imageData, 0, 0);
        canvas.toBlob((blob) => {
            if (!blob) {
                reject(new Error("无法生成掩膜文件"));
                return;
            }
            resolve(blob);
        }, "image/png");
    });
}

async function trainModel() {
    if (!state.sessionId || !state.labelMask) {
        setStatus("请先加载一个场景", "error");
        return;
    }
    if (!hasLabelPixels()) {
        setStatus("请先标注至少两个类别后再训练", "error");
        return;
    }

    setBusy(true, "正在训练随机森林并执行全图推理");
    setInferring(true);
    updateProgress(0, "准备中…");

    try {
        const formData = new FormData();
        formData.append("mask", await maskToBlob(state.labelMask), "labels.png");
        formData.append("nEstimators", elements.nEstimatorsInput.value || "120");
        formData.append("maxDepth", elements.maxDepthInput.value || "18");
        formData.append("maxSamplesPerClass", elements.maxSamplesInput.value || "15000");
        formData.append("confThreshold", elements.confThresholdInput.value || "0.95");
        formData.append("neighborhoodSize", elements.neighborhoodSizeSelect.value || "1");
        const selectedBands = [...elements.trainBandsSelect.selectedOptions].map((opt) => opt.value);
        for (const key of selectedBands) {
            formData.append("trainBandPaths", key);
        }

        const response = await fetch(`/api/sessions/${state.sessionId}/train`, {
            method: "POST",
            body: formData
        });

        if (!response.ok || !response.body) {
            const payload = await response.json().catch(() => ({}));
            throw new Error(payload.error || `训练失败：${response.status}`);
        }

        // 用 SSE（text/event-stream）读取进度
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split("\n\n");
            buffer = parts.pop(); // 保留不完整的最后一段

            for (const block of parts) {
                let eventType = "message";
                let dataStr = "";
                for (const line of block.split("\n")) {
                    if (line.startsWith("event: ")) {
                        eventType = line.slice(7).trim();
                    } else if (line.startsWith("data: ")) {
                        dataStr = line.slice(6).trim();
                    }
                }
                if (!dataStr) continue;
                let data;
                try { data = JSON.parse(dataStr); } catch { continue; }

                if (eventType === "progress") {
                    updateProgress(data.pct ?? 0, data.message ?? "");
                    if (data.pct != null) {
                        setStatus(data.message || "推理中…", "busy");
                    }
                } else if (eventType === "result") {
                    state.predictionMask = await readMaskFromImage(data.predictionUrl);
                    state.predictionTifUrl = data.predictionTifUrl || null;
                    refreshColorLookup();
                    rebuildPredictionLayer();
                    rebuildLabelLayer();
                    renderTrainStats(data);
                    elements.copyPredictionBtn.disabled = false;
                    setStatus(data.message || "推理完成", "success");
                } else if (eventType === "error") {
                    throw new Error(data.message || "训练失败");
                }
            }
        }
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        setBusy(false);
        setInferring(false);
    }
}

function clearEditableLayer() {
    if (!state.labelMask) { return; }
    state.labelMask.fill(0);
    rebuildLabelLayer();
    setStatus("可编辑层已清空", "success");
}

function copyPredictionToEditableLayer() {
    if (!state.predictionMask || !state.labelMask) {
        setStatus("当前没有 AI 结果可复制", "error");
        return;
    }
    for (let i = 0; i < state.predictionMask.length; i++) {
        state.labelMask[i] = state.predictionMask[i] === 99 ? 0 : state.predictionMask[i];
    }
    state.labelsDirty = true;
    // 重置 AI 推理状态，进入全新标注/训练流程
    state.predictionMask = null;
    state.predictionTifUrl = null;
    predictionImageData.data = null;
    predictionContext.clearRect(0, 0, elements.predictionCanvas.width, elements.predictionCanvas.height);
    elements.copyPredictionBtn.disabled = true;
    rebuildLabelLayer();
    setStatus("AI 结果已复制到可编辑层，已重置推理状态，可重新训练", "success");
}

// ── 文件夹批量导航 ────────────────────────────────────────────

function updateFolderNavUI() {
    if (!elements.folderNavRow) { return; }
    if (state.folderQueue.length === 0) {
        elements.folderNavRow.classList.add("hidden");
        return;
    }
    elements.folderNavRow.classList.remove("hidden");
    const idx = state.currentFolderIndex;
    const total = state.folderQueue.length;
    const name = idx >= 0 ? state.folderQueue[idx].split(/[\\/]/).pop() : "—";
    const pos = idx >= 0 ? `${idx + 1}/${total}` : `—/${total}`;
    elements.folderNavInfo.textContent = `${pos}  ${name}`;
    elements.prevFolderBtn.disabled = state.isBusy || idx <= 0;
    elements.nextFolderBtn.disabled = state.isBusy || idx >= total - 1;
}

async function scanBatchFolders() {
    const parentPath = (elements.folderPathInput.value || "").trim();
    if (!parentPath) {
        setStatus("请先输入父文件夹路径", "error");
        return;
    }
    setBusy(true, "正在扫描 DZV01 子文件夹…");
    try {
        const payload = await fetchJson("/api/list-subfolders", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ parentPath })
        });
        if (!payload.subfolders || payload.subfolders.length === 0) {
            throw new Error("当前文件夹下没有找到包含波段数据的子文件夹");
        }
        state.folderQueue = payload.subfolders;
        state.currentFolderIndex = -1;
        updateFolderNavUI();
        setStatus(`已找到 ${payload.count} 个 DZV01 场景，点击"下一张"开始加载`, "success");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        setBusy(false);
    }
}

async function navigateToFolder(targetIndex) {
    if (targetIndex < 0 || targetIndex >= state.folderQueue.length) { return; }
    if (state.labelsDirty) {
        if (!confirm("当前有未导出的标注，切换后将丢失，是否继续？")) {
            return;
        }
    }
    const folderPath = state.folderQueue[targetIndex];
    const parentPath = elements.folderPathInput.value;  // 保存父文件夹路径，scanFolder 会把它覆盖
    await scanFolder(folderPath);
    elements.folderPathInput.value = parentPath;         // 恢复父文件夹路径
    if (state.sceneBands.length >= 3) {
        await loadScene();
        state.currentFolderIndex = targetIndex;
    }
    updateFolderNavUI();
}

function downloadMask(mask, filename) {
    if (!mask) {
        setStatus("当前没有可导出的掩膜", "error");
        return;
    }
    maskToBlob(mask).then((blob) => {
        const anchor = document.createElement("a");
        anchor.href = URL.createObjectURL(blob);
        anchor.download = filename;
        anchor.click();
        URL.revokeObjectURL(anchor.href);
    }).catch((error) => {
        setStatus(error.message, "error");
    });
}

function addClass() {
    const name = elements.newClassName.value.trim() || `类别 ${state.nextClassId}`;
    if (state.nextClassId > 250) {
        setStatus("类别数量已达到上限 250", "error");
        return;
    }
    state.classes.push({
        id: state.nextClassId,
        name,
        color: elements.newClassColor.value
    });
    state.activeClassId = state.nextClassId;
    state.nextClassId += 1;
    elements.newClassName.value = "";
    renderClassList();
    refreshColorLookup();
    rebuildLabelLayer();
    setStatus(`已添加 ${name}`, "success");
}

function handlePointerDown(event) {
    if (event.button !== 0 || !state.labelMask || state.isBusy || state.isInferring) {
        return;
    }
    if (spaceHeld) {
        beginPan(event.clientX, event.clientY, true);
        elements.labelCanvas.setPointerCapture?.(event.pointerId);
        return;
    }
    const point = eventToPixel(event);
    if (!point) { return; }
    state.drawing = true;
    state.lastPoint = point;
    paintStroke(point.x, point.y, point.x, point.y);
    updateCursorInfo(point.x, point.y);
    elements.labelCanvas.setPointerCapture?.(event.pointerId);
}

function handlePointerMove(event) {
    if (panState.active && panState.fromPointer) {
        continuePan(event.clientX, event.clientY);
        return;
    }
    updateBrushCursor(event.clientX, event.clientY);
    const point = eventToPixel(event);
    if (!point) { updateCursorInfo(null, null); return; }
    updateCursorInfo(point.x, point.y);
    if (!state.drawing || !state.lastPoint) { return; }
    paintStroke(state.lastPoint.x, state.lastPoint.y, point.x, point.y);
    state.lastPoint = point;
}

function handlePointerUp(event) {
    if (panState.active && panState.fromPointer) {
        endPan();
        elements.labelCanvas.releasePointerCapture?.(event.pointerId);
        return;
    }
    if (!state.drawing) { return; }
    state.drawing = false;
    state.lastPoint = null;
    elements.labelCanvas.releasePointerCapture?.(event.pointerId);
}

function handleShortcuts(event) {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
        return;
    }
    if (event.key === "b" || event.key === "B") {
        state.tool = "brush";
        updateToolButtons();
        return;
    }
    if (event.key === "e" || event.key === "E") {
        state.tool = "eraser";
        updateToolButtons();
        return;
    }
    if (event.key === "[") {
        elements.brushSizeInput.value = Math.max(1, Number(elements.brushSizeInput.value) - 1);
        syncBrushSize();
        return;
    }
    if (event.key === "]") {
        elements.brushSizeInput.value = Math.min(80, Number(elements.brushSizeInput.value) + 1);
        syncBrushSize();
        return;
    }
    if (event.key === "+" || event.key === "=") {
        setZoom(state.zoom * 1.25);
        return;
    }
    if (event.key === "-" || event.key === "_") {
        setZoom(state.zoom / 1.25);
        return;
    }
    const numeric = Number.parseInt(event.key, 10);
    if (!Number.isNaN(numeric) && numeric >= 1 && numeric <= 9) {
        const category = state.classes[numeric - 1];
        if (category) {
            state.activeClassId = category.id;
            renderClassList();
        }
    }
}

function syncBrushSize() {
    state.brushSize = Number(elements.brushSizeInput.value);
    elements.brushSizeValue.textContent = state.brushSize;
    // 同步更新画笔光标大小（若正在悬停则即时更新）
    const diameter = (state.brushSize * 2 + 1) * state.zoom;
    elements.brushCursor.style.width = `${diameter}px`;
    elements.brushCursor.style.height = `${diameter}px`;
}

function syncLabelOpacity() {
    state.labelOpacity = Number(elements.labelOpacityInput.value) / 100;
    elements.labelOpacityValue.textContent = `${Math.round(state.labelOpacity * 100)}%`;
    elements.labelCanvas.style.opacity = state.labelOpacity;
}

function syncPredictionOpacity() {
    state.predictionOpacity = Number(elements.predictionOpacityInput.value) / 100;
    elements.predictionOpacityValue.textContent = `${Math.round(state.predictionOpacity * 100)}%`;
    elements.predictionCanvas.style.opacity = state.predictionOpacity;
}

function syncVisibility() {
    state.showLabels = elements.showLabelsInput.checked;
    state.showPrediction = elements.showPredictionInput.checked;
    elements.labelCanvas.style.display = state.showLabels ? "" : "none";
    elements.predictionCanvas.style.display = state.showPrediction ? "" : "none";
}

// ── 平移与缩放辅助函数 ────────────────────────────────────────

function updateCanvasCursor() {
    if (panState.active) {
        elements.labelCanvas.style.cursor = "grabbing";
        elements.brushCursor.style.display = "none";
    } else if (spaceHeld && state.sessionId) {
        elements.labelCanvas.style.cursor = "grab";
        elements.brushCursor.style.display = "none";
    } else {
        elements.labelCanvas.style.cursor = "none";
    }
}

function updateBrushCursor(clientX, clientY) {
    if (panState.active || spaceHeld || !state.sessionId) {
        elements.brushCursor.style.display = "none";
        return;
    }
    const rect = elements.canvasStack.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const diameter = (state.brushSize * 2 + 1) * state.zoom;
    elements.brushCursor.style.left = `${x}px`;
    elements.brushCursor.style.top = `${y}px`;
    elements.brushCursor.style.width = `${diameter}px`;
    elements.brushCursor.style.height = `${diameter}px`;
    elements.brushCursor.style.display = "block";
}

function beginPan(clientX, clientY, fromPointer) {
    panState.active = true;
    panState.fromPointer = fromPointer;
    panState.startX = clientX;
    panState.startY = clientY;
    panState.scrollX = elements.viewport.scrollLeft;
    panState.scrollY = elements.viewport.scrollTop;
    updateCanvasCursor();
}

function continuePan(clientX, clientY) {
    if (!panState.active) {
        return;
    }
    elements.viewport.scrollLeft = panState.scrollX - (clientX - panState.startX);
    elements.viewport.scrollTop = panState.scrollY - (clientY - panState.startY);
}

function endPan() {
    panState.active = false;
    updateCanvasCursor();
}

/**
 * 滚轮缩放：以鼠标所在图像像素为中心缩放，缩放后保持该像素不动。
 * 原理：viewport 内 canvasStack 距内容原点始终为 padding=20px，
 *   newScrollLeft = 20 + imgX * newZoom - cursorViewX
 */
function handleWheel(event) {
    if (!state.sessionId) {
        return;
    }
    event.preventDefault();
    if (state.drawing) {
        return;
    }
    const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZoom = Math.max(0.05, Math.min(16, state.zoom * factor));
    if (Math.abs(newZoom - state.zoom) < 1e-9) {
        return;
    }
    // 鼠标在图像中的坐标（图像像素，非 CSS 像素）
    const canvasRect = elements.labelCanvas.getBoundingClientRect();
    const imgX = (event.clientX - canvasRect.left) * state.imageWidth / canvasRect.width;
    const imgY = (event.clientY - canvasRect.top) * state.imageHeight / canvasRect.height;
    // 鼠标在 viewport 可视区内的偏移
    const viewportRect = elements.viewport.getBoundingClientRect();
    const cursorViewX = event.clientX - viewportRect.left;
    const cursorViewY = event.clientY - viewportRect.top;
    setZoom(newZoom);
    // viewport padding = 20px（与 styles.css 保持一致）
    elements.viewport.scrollLeft = 20 + imgX * newZoom - cursorViewX;
    elements.viewport.scrollTop  = 20 + imgY * newZoom - cursorViewY;
}

function bindEvents() {
    elements.scanFolderBtn.addEventListener("click", handleScanClick);
    elements.prevFolderBtn.addEventListener("click", () => navigateToFolder(state.currentFolderIndex - 1));
    elements.nextFolderBtn.addEventListener("click", () => navigateToFolder(state.currentFolderIndex + 1));
    elements.loadSceneBtn.addEventListener("click", loadScene);
    elements.addClassBtn.addEventListener("click", addClass);
    elements.brushToolBtn.addEventListener("click", () => {
        state.tool = "brush";
        updateToolButtons();
    });
    elements.eraserToolBtn.addEventListener("click", () => {
        state.tool = "eraser";
        updateToolButtons();
    });
    elements.brushSizeInput.addEventListener("input", syncBrushSize);
    elements.labelOpacityInput.addEventListener("input", syncLabelOpacity);
    elements.predictionOpacityInput.addEventListener("input", syncPredictionOpacity);
    elements.confThresholdInput.addEventListener("input", () => {
        elements.confThresholdValue.textContent = parseFloat(elements.confThresholdInput.value).toFixed(2);
    });
    elements.showLabelsInput.addEventListener("change", syncVisibility);
    elements.showPredictionInput.addEventListener("change", syncVisibility);
    elements.clearLabelsBtn.addEventListener("click", clearEditableLayer);
    elements.copyPredictionBtn.addEventListener("click", copyPredictionToEditableLayer);
    elements.trainBtn.addEventListener("click", trainModel);
    elements.downloadLabelsBtn.addEventListener("click", async () => {
        if (!state.labelMask || !state.sessionId) {
            setStatus("当前没有可导出的标注", "error");
            return;
        }
        try {
            setBusy(true, "正在导出标注 TIF…");
            const formData = new FormData();
            formData.append("mask", await maskToBlob(state.labelMask), "labels.png");
            const response = await fetch(`/api/sessions/${state.sessionId}/labels.tif`, {
                method: "POST",
                body: formData
            });
            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.error || `导出失败：${response.status}`);
            }
            const blob = await response.blob();
            const anchor = document.createElement("a");
            anchor.href = URL.createObjectURL(blob);
            anchor.download = "labels.tif";
            anchor.click();
            URL.revokeObjectURL(anchor.href);
            state.labelsDirty = false;
            setStatus("标注 TIF 导出成功", "success");
        } catch (error) {
            setStatus(error.message, "error");
        } finally {
            setBusy(false);
        }
    });
    elements.zoomSlider.addEventListener("input", () => setZoom(Number(elements.zoomSlider.value) / 100));
    elements.zoomOutBtn.addEventListener("click", () => setZoom(state.zoom - 0.1));
    elements.zoomInBtn.addEventListener("click", () => setZoom(state.zoom + 0.1));
    elements.fitViewBtn.addEventListener("click", fitView);

    elements.labelCanvas.addEventListener("pointerdown", handlePointerDown);
    elements.labelCanvas.addEventListener("pointermove", handlePointerMove);
    elements.labelCanvas.addEventListener("pointerup", handlePointerUp);
    elements.labelCanvas.addEventListener("pointerleave", () => {
        updateCursorInfo(null, null);
        elements.brushCursor.style.display = "none";
    });
    elements.labelCanvas.addEventListener("contextmenu", (event) => event.preventDefault());
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("keydown", handleShortcuts);
    window.addEventListener("resize", () => {
        if (state.sessionId) {
            configureCanvasSize();
        }
    });

    // ── 滚轮缩放 ────────────────────────────────────────────────
    elements.viewport.addEventListener("wheel", handleWheel, { passive: false });

    // ── 中键拖动平移（fromPointer=false，由 window mouse* 驱动）────
    window.addEventListener("mousedown", (event) => {
        if (event.button === 1 && state.sessionId && !panState.active) {
            event.preventDefault(); // 阻止浏览器自带自动滚动光标
            beginPan(event.clientX, event.clientY, false);
        }
    });
    window.addEventListener("mousemove", (event) => {
        // 仅处理中键平移；空格平移由 canvas pointermove（capture）处理
        if (panState.active && !panState.fromPointer) {
            continuePan(event.clientX, event.clientY);
        }
    });
    window.addEventListener("mouseup", (event) => {
        if (event.button === 1 && panState.active && !panState.fromPointer) {
            endPan();
        }
    });

    // ── 空格键切换平移模式 ────────────────────────────────────────
    window.addEventListener("keydown", (event) => {
        if (
            event.code === "Space" &&
            !(event.target instanceof HTMLInputElement) &&
            !(event.target instanceof HTMLSelectElement) &&
            !(event.target instanceof HTMLTextAreaElement)
        ) {
            spaceHeld = true;
            updateCanvasCursor();
            event.preventDefault(); // 防止页面被空格滚动
        }
    });
    window.addEventListener("keyup", (event) => {
        if (event.code === "Space") {
            spaceHeld = false;
            // 若平移仍在进行则等松手后再恢复光标
            if (!panState.active) {
                updateCanvasCursor();
            }
        }
    });
}

function initialize() {
    refreshColorLookup();
    renderClassList();
    updateToolButtons();
    syncBrushSize();
    syncLabelOpacity();
    syncPredictionOpacity();
    bindEvents();
    elements.copyPredictionBtn.disabled = true;
    elements.downloadLabelsBtn.disabled = true;
    elements.folderPathInput.value = window.APP_CONFIG.workspaceRoot || "";
    updateFolderNavUI();
}

initialize();