const elements = {
    folderPathInput: document.getElementById("folderPathInput"),
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
    trainBtn: document.getElementById("trainBtn"),
    showPredictionInput: document.getElementById("showPredictionInput"),
    showLabelsInput: document.getElementById("showLabelsInput"),
    predictionOpacityInput: document.getElementById("predictionOpacityInput"),
    predictionOpacityValue: document.getElementById("predictionOpacityValue"),
    downloadLabelsBtn: document.getElementById("downloadLabelsBtn"),
    downloadPredictionBtn: document.getElementById("downloadPredictionBtn"),
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
    overlayCanvas: document.getElementById("overlayCanvas")
};

const imageContext = elements.imageCanvas.getContext("2d");
const overlayContext = elements.overlayCanvas.getContext("2d", { willReadFrequently: true });
imageContext.imageSmoothingEnabled = false;
overlayContext.imageSmoothingEnabled = false;

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
    overlayImageData: null,
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
    isBusy: false
};

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
    const labelAlpha = Math.round(state.labelOpacity * 255);
    const predictionAlpha = Math.round(state.predictionOpacity * 255);
    for (const category of state.classes) {
        const { r, g, b } = hexToRgb(category.color);
        const labelOffset = category.id * 4;
        state.labelLookup[labelOffset] = r;
        state.labelLookup[labelOffset + 1] = g;
        state.labelLookup[labelOffset + 2] = b;
        state.labelLookup[labelOffset + 3] = labelAlpha;

        state.predictionLookup[labelOffset] = r;
        state.predictionLookup[labelOffset + 1] = g;
        state.predictionLookup[labelOffset + 2] = b;
        state.predictionLookup[labelOffset + 3] = predictionAlpha;
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
    elements.downloadPredictionBtn.disabled = disabled || !state.predictionMask;
    if (message) {
        setStatus(message, busy ? "busy" : "");
    }
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
    const imageCanvasNeedsResize = elements.imageCanvas.width !== state.imageWidth || elements.imageCanvas.height !== state.imageHeight;
    const overlayCanvasNeedsResize = elements.overlayCanvas.width !== state.imageWidth || elements.overlayCanvas.height !== state.imageHeight;

    if (imageCanvasNeedsResize) {
        elements.imageCanvas.width = state.imageWidth;
        elements.imageCanvas.height = state.imageHeight;
    }
    if (overlayCanvasNeedsResize) {
        elements.overlayCanvas.width = state.imageWidth;
        elements.overlayCanvas.height = state.imageHeight;
        state.overlayImageData = null;
    }

    elements.canvasStack.style.width = `${state.imageWidth * state.zoom}px`;
    elements.canvasStack.style.height = `${state.imageHeight * state.zoom}px`;
    elements.imageCanvas.style.width = `${state.imageWidth * state.zoom}px`;
    elements.imageCanvas.style.height = `${state.imageHeight * state.zoom}px`;
    elements.overlayCanvas.style.width = `${state.imageWidth * state.zoom}px`;
    elements.overlayCanvas.style.height = `${state.imageHeight * state.zoom}px`;

    if (imageCanvasNeedsResize && state.previewImage) {
        drawBaseImage();
    }
    if (overlayCanvasNeedsResize && state.labelMask) {
        rebuildOverlay();
    }
}

function setZoom(zoomValue) {
    if (!state.imageWidth || !state.imageHeight) {
        return;
    }
    state.zoom = Math.max(0.05, Math.min(3, zoomValue));
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

function ensureOverlayImageData() {
    if (!state.overlayImageData || state.overlayImageData.width !== state.imageWidth || state.overlayImageData.height !== state.imageHeight) {
        state.overlayImageData = overlayContext.createImageData(state.imageWidth, state.imageHeight);
    }
}

function applyVisiblePixel(data, pixelIndex) {
    const byteOffset = pixelIndex * 4;
    const labelValue = state.showLabels && state.labelMask ? state.labelMask[pixelIndex] : 0;
    if (labelValue > 0) {
        const lookupOffset = labelValue * 4;
        data[byteOffset] = state.labelLookup[lookupOffset];
        data[byteOffset + 1] = state.labelLookup[lookupOffset + 1];
        data[byteOffset + 2] = state.labelLookup[lookupOffset + 2];
        data[byteOffset + 3] = state.labelLookup[lookupOffset + 3];
        return;
    }

    const predictionValue = state.showPrediction && state.predictionMask ? state.predictionMask[pixelIndex] : 0;
    if (predictionValue > 0) {
        const lookupOffset = predictionValue * 4;
        data[byteOffset] = state.predictionLookup[lookupOffset];
        data[byteOffset + 1] = state.predictionLookup[lookupOffset + 1];
        data[byteOffset + 2] = state.predictionLookup[lookupOffset + 2];
        data[byteOffset + 3] = state.predictionLookup[lookupOffset + 3];
        return;
    }

    data[byteOffset] = 0;
    data[byteOffset + 1] = 0;
    data[byteOffset + 2] = 0;
    data[byteOffset + 3] = 0;
}

function rebuildOverlay() {
    if (!state.labelMask || !state.imageWidth || !state.imageHeight) {
        overlayContext.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
        return;
    }
    ensureOverlayImageData();
    const data = state.overlayImageData.data;
    for (let index = 0; index < state.labelMask.length; index += 1) {
        applyVisiblePixel(data, index);
    }
    overlayContext.putImageData(state.overlayImageData, 0, 0);
}

function updateOverlayPatch(x0, y0, x1, y1) {
    if (!state.overlayImageData) {
        rebuildOverlay();
        return;
    }

    const minX = Math.max(0, Math.min(x0, x1));
    const minY = Math.max(0, Math.min(y0, y1));
    const maxX = Math.min(state.imageWidth - 1, Math.max(x0, x1));
    const maxY = Math.min(state.imageHeight - 1, Math.max(y0, y1));
    const data = state.overlayImageData.data;

    for (let y = minY; y <= maxY; y += 1) {
        const rowOffset = y * state.imageWidth;
        for (let x = minX; x <= maxX; x += 1) {
            applyVisiblePixel(data, rowOffset + x);
        }
    }

    overlayContext.putImageData(
        state.overlayImageData,
        0,
        0,
        minX,
        minY,
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
        updateOverlayPatch(minX, minY, maxX, maxY);
    }
}

function eventToPixel(event) {
    if (!state.imageWidth || !state.imageHeight) {
        return null;
    }
    const rect = elements.overlayCanvas.getBoundingClientRect();
    const inside = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
    if (!inside) {
        return null;
    }
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
        state.overlayImageData = null;
        fitView();
        drawBaseImage();
        refreshColorLookup();
        rebuildOverlay();
        elements.trainStats.innerHTML = [
            `<div><strong>当前场景：</strong>${payload.sceneName}</div>`,
            `<div><strong>尺寸：</strong>${payload.width} × ${payload.height}</div>`,
            `<div><strong>有效像素：</strong>${formatNumber(payload.validPixels)}</div>`,
            `<div><strong>预览波段：</strong>${payload.bandLabels.join(" / ")}</div>`,
            `<div><strong>可用波段数：</strong>${(payload.allBands || []).length} 个（已全选用于训练）</div>`
        ].join("");
        elements.copyPredictionBtn.disabled = true;
        elements.downloadPredictionBtn.disabled = true;
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
    try {
        const formData = new FormData();
        formData.append("mask", await maskToBlob(state.labelMask), "labels.png");
        formData.append("nEstimators", elements.nEstimatorsInput.value || "120");
        formData.append("maxDepth", elements.maxDepthInput.value || "18");
        formData.append("maxSamplesPerClass", elements.maxSamplesInput.value || "15000");
        formData.append("neighborhoodSize", elements.neighborhoodSizeSelect.value || "1");
        const selectedBands = [...elements.trainBandsSelect.selectedOptions].map((opt) => opt.value);
        for (const key of selectedBands) {
            formData.append("trainBandPaths", key);
        }

        const response = await fetch(`/api/sessions/${state.sessionId}/train`, {
            method: "POST",
            body: formData
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.error || `训练失败：${response.status}`);
        }

        state.predictionMask = await readMaskFromImage(payload.predictionUrl);
        refreshColorLookup();
        rebuildOverlay();
        renderTrainStats(payload);
        elements.copyPredictionBtn.disabled = false;
        elements.downloadPredictionBtn.disabled = false;
        setStatus(payload.message || "推理完成", "success");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        setBusy(false);
    }
}

function clearEditableLayer() {
    if (!state.labelMask) {
        return;
    }
    state.labelMask.fill(0);
    rebuildOverlay();
    setStatus("可编辑层已清空", "success");
}

function copyPredictionToEditableLayer() {
    if (!state.predictionMask || !state.labelMask) {
        setStatus("当前没有 AI 结果可复制", "error");
        return;
    }
    state.labelMask.set(state.predictionMask);
    rebuildOverlay();
    setStatus("AI 结果已复制到可编辑层，可继续精修", "success");
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
    rebuildOverlay();
    setStatus(`已添加 ${name}`, "success");
}

function handlePointerDown(event) {
    if (event.button !== 0 || !state.labelMask || state.isBusy) {
        return;
    }
    const point = eventToPixel(event);
    if (!point) {
        return;
    }
    state.drawing = true;
    state.lastPoint = point;
    paintStroke(point.x, point.y, point.x, point.y);
    updateCursorInfo(point.x, point.y);
    elements.overlayCanvas.setPointerCapture?.(event.pointerId);
}

function handlePointerMove(event) {
    const point = eventToPixel(event);
    if (!point) {
        updateCursorInfo(null, null);
        return;
    }
    updateCursorInfo(point.x, point.y);
    if (!state.drawing || !state.lastPoint) {
        return;
    }
    paintStroke(state.lastPoint.x, state.lastPoint.y, point.x, point.y);
    state.lastPoint = point;
}

function handlePointerUp(event) {
    if (!state.drawing) {
        return;
    }
    state.drawing = false;
    state.lastPoint = null;
    elements.overlayCanvas.releasePointerCapture?.(event.pointerId);
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
}

function syncLabelOpacity() {
    state.labelOpacity = Number(elements.labelOpacityInput.value) / 100;
    elements.labelOpacityValue.textContent = `${Math.round(state.labelOpacity * 100)}%`;
    refreshColorLookup();
    rebuildOverlay();
}

function syncPredictionOpacity() {
    state.predictionOpacity = Number(elements.predictionOpacityInput.value) / 100;
    elements.predictionOpacityValue.textContent = `${Math.round(state.predictionOpacity * 100)}%`;
    refreshColorLookup();
    rebuildOverlay();
}

function syncVisibility() {
    state.showLabels = elements.showLabelsInput.checked;
    state.showPrediction = elements.showPredictionInput.checked;
    rebuildOverlay();
}

function bindEvents() {
    elements.scanFolderBtn.addEventListener("click", () => scanFolder(elements.folderPathInput.value));
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
    elements.showLabelsInput.addEventListener("change", syncVisibility);
    elements.showPredictionInput.addEventListener("change", syncVisibility);
    elements.clearLabelsBtn.addEventListener("click", clearEditableLayer);
    elements.copyPredictionBtn.addEventListener("click", copyPredictionToEditableLayer);
    elements.trainBtn.addEventListener("click", trainModel);
    elements.downloadLabelsBtn.addEventListener("click", () => downloadMask(state.labelMask, "editable_labels.png"));
    elements.downloadPredictionBtn.addEventListener("click", () => downloadMask(state.predictionMask, "prediction_labels.png"));

    elements.zoomSlider.addEventListener("input", () => setZoom(Number(elements.zoomSlider.value) / 100));
    elements.zoomOutBtn.addEventListener("click", () => setZoom(state.zoom - 0.1));
    elements.zoomInBtn.addEventListener("click", () => setZoom(state.zoom + 0.1));
    elements.fitViewBtn.addEventListener("click", fitView);

    elements.overlayCanvas.addEventListener("pointerdown", handlePointerDown);
    elements.overlayCanvas.addEventListener("pointermove", handlePointerMove);
    elements.overlayCanvas.addEventListener("pointerup", handlePointerUp);
    elements.overlayCanvas.addEventListener("pointerleave", () => updateCursorInfo(null, null));
    elements.overlayCanvas.addEventListener("contextmenu", (event) => event.preventDefault());
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("keydown", handleShortcuts);
    window.addEventListener("resize", () => {
        if (state.sessionId) {
            configureCanvasSize();
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
    elements.downloadPredictionBtn.disabled = true;
    elements.downloadLabelsBtn.disabled = true;
    elements.folderPathInput.value = window.APP_CONFIG.workspaceRoot || "";
}

initialize();