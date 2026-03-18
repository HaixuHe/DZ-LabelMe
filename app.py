from __future__ import annotations

import io
import json
import os
import re
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import rasterio
from flask import Flask, Response, jsonify, render_template, request, send_file
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

APP_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_DIR.parent
OUTPUT_DIR = APP_DIR / "outputs"
SESSION_LIMIT = 3
PREDICT_BATCH_SIZE = 500_000
MAX_CLASS_ID = 250
DEFAULT_PORT = 5000
DEFAULT_HOST = "127.0.0.1"
DEFAULT_CLASS_COLORS = ["#ff6b4a", "#1f8a70", "#2b59c3", "#c77d20", "#8f2d56"]
BAND_PATTERN = re.compile(r"_B(\d+)", re.IGNORECASE)


@dataclass
class SessionState:
    session_id: str
    scene_dir: Path
    band_paths: List[Path]
    band_labels: List[str]
    stack: np.ndarray
    valid_mask: np.ndarray
    preview_png: bytes
    prediction_mask: Optional[np.ndarray] = None
    prediction_png: Optional[bytes] = None
    feature_band_paths: Optional[List[Path]] = None
    feature_stack: Optional[np.ndarray] = None
    feature_valid_mask: Optional[np.ndarray] = None
    feature_neighborhood_size: int = 1
    # 记录训练时使用的第一个波段路径（用于读取坐标参考）
    train_reference_band: Optional[Path] = None


SESSIONS: Dict[str, SessionState] = {}
SESSION_ORDER: List[str] = []


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

    @app.get("/")
    def index():
        return render_template("index.html", workspace_root=str(WORKSPACE_ROOT))

    @app.get("/api/scenes")
    def api_scenes():
        scenes = [serialize_scene(scene_dir, include_bands=False) for scene_dir in discover_scene_dirs()]
        return jsonify({"workspaceRoot": str(WORKSPACE_ROOT), "scenes": scenes})

    @app.post("/api/scan-folder")
    def api_scan_folder():
        payload = request.get_json(silent=True) or {}
        folder_path = payload.get("folderPath", "")
        try:
            scene_dir = normalize_folder_path(folder_path)
            return jsonify({"scene": serialize_scene(scene_dir, include_bands=True)})
        except ValueError as exc:
            return error_response(str(exc), 400)

    @app.post("/api/load-scene")
    def api_load_scene():
        payload = request.get_json(silent=True) or {}
        folder_path = payload.get("folderPath", "")
        band_values = payload.get("bandPaths") or []

        if len(band_values) != 3:
            return error_response("请准确选择 3 个波段文件进行合成。", 400)

        try:
            scene_dir = normalize_folder_path(folder_path)
            band_paths = [resolve_band_path(scene_dir, value) for value in band_values]
            stack, valid_mask, preview_png = build_stack_and_preview(band_paths)
        except ValueError as exc:
            return error_response(str(exc), 400)
        except Exception as exc:
            return error_response(f"读取波段失败：{exc}", 500)

        session_id = uuid.uuid4().hex
        session = SessionState(
            session_id=session_id,
            scene_dir=scene_dir,
            band_paths=band_paths,
            band_labels=[path.relative_to(scene_dir).as_posix() for path in band_paths],
            stack=stack,
            valid_mask=valid_mask,
            preview_png=preview_png,
        )
        register_session(session)

        all_bands = list_band_files(scene_dir)
        height, width = valid_mask.shape
        return jsonify(
            {
                "sessionId": session_id,
                "sceneName": scene_dir.name,
                "folderPath": str(scene_dir),
                "width": width,
                "height": height,
                "bandLabels": session.band_labels,
                "validPixels": int(valid_mask.sum()),
                "previewUrl": f"/api/sessions/{session_id}/preview.png",
                "allBands": all_bands,
            }
        )

    @app.get("/api/sessions/<session_id>/preview.png")
    def api_preview_png(session_id: str):
        try:
            session = get_session(session_id)
        except KeyError as exc:
            return error_response(str(exc), 404)
        return send_png_bytes(session.preview_png, f"{session.scene_dir.name}_preview.png")

    @app.get("/api/sessions/<session_id>/prediction.png")
    def api_prediction_png(session_id: str):
        try:
            session = get_session(session_id)
        except KeyError as exc:
            return error_response(str(exc), 404)
        if session.prediction_png is None:
            return error_response("当前会话还没有推理结果。", 404)
        return send_png_bytes(session.prediction_png, f"{session.scene_dir.name}_prediction.png")

    @app.get("/api/sessions/<session_id>/prediction.tif")
    def api_prediction_tif(session_id: str):
        """导出带有地理参考的 GeoTIFF 推理结果。"""
        try:
            session = get_session(session_id)
        except KeyError as exc:
            return error_response(str(exc), 404)
        if session.prediction_mask is None:
            return error_response("当前会话还没有推理结果。", 404)
        try:
            tif_bytes = encode_prediction_tif(session.prediction_mask, session.train_reference_band)
        except Exception as exc:
            return error_response(f"导出 GeoTIFF 失败：{exc}", 500)
        buffer = io.BytesIO(tif_bytes)
        buffer.seek(0)
        response = send_file(
            buffer,
            mimetype="image/tiff",
            download_name=f"{session.scene_dir.name}_prediction.tif",
            max_age=0,
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/sessions/<session_id>/train")
    def api_train(session_id: str):
        """SSE 流式训练接口，实时推送进度事件，最终返回结果。"""
        try:
            session = get_session(session_id)
        except KeyError as exc:
            return error_response(str(exc), 404)
        if "mask" not in request.files:
            return error_response("训练请求缺少标注掩膜。", 400)

        # 同步读取所有表单数据（SSE 生成器中无法再访问 request）
        try:
            mask_bytes = request.files["mask"].read()
            n_estimators = clamp_int(request.form.get("nEstimators", "120"), 10, 500)
            max_depth = parse_optional_depth(request.form.get("maxDepth", "18"))
            max_samples_per_class = clamp_int(request.form.get("maxSamplesPerClass", "15000"), 50, 300000)
            neighborhood_size = clamp_int(request.form.get("neighborhoodSize", "1"), 1, 11)
            if neighborhood_size % 2 == 0:
                neighborhood_size += 1
            train_band_keys = request.form.getlist("trainBandPaths")
        except ValueError as exc:
            return error_response(str(exc), 400)

        def generate() -> Generator[str, None, None]:
            def sse(event: str, data: dict) -> str:
                return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

            try:
                import io as _io
                from PIL import Image as _Image

                buf = _io.BytesIO(mask_bytes)
                label_mask = parse_uploaded_mask_bytes(buf, session.valid_mask.shape)
                labeled_pixels = int((label_mask > 0).sum())
                labeled_classes = np.unique(label_mask[label_mask > 0])
                if labeled_pixels == 0:
                    yield sse("error", {"message": "至少需要标注一些像素后才能训练。"})
                    return
                if labeled_classes.size < 2:
                    yield sse("error", {"message": "随机森林至少需要 2 个类别才能训练。"})
                    return

                # ── 步骤 1：构建/复用特征栈 ─────────────────────────────
                yield sse("progress", {"step": "feature", "pct": 0, "message": "正在构建特征矩阵…"})
                if train_band_keys:
                    train_band_paths = [resolve_band_path(session.scene_dir, key) for key in train_band_keys]
                else:
                    train_band_paths = session.band_paths

                need_rebuild = (
                    session.feature_stack is None
                    or session.feature_neighborhood_size != neighborhood_size
                    or [str(p) for p in (session.feature_band_paths or [])] != [str(p) for p in train_band_paths]
                )
                if need_rebuild:
                    feat_stack, feat_valid_mask = build_feature_stack(train_band_paths, session.valid_mask.shape)
                    if neighborhood_size > 1:
                        feat_stack = add_texture_features(feat_stack, neighborhood_size)
                    session.feature_band_paths = train_band_paths
                    session.feature_stack = feat_stack
                    session.feature_valid_mask = feat_valid_mask
                    session.feature_neighborhood_size = neighborhood_size
                    session.train_reference_band = train_band_paths[0]

                feature_stack = session.feature_stack
                feature_valid_mask = session.feature_valid_mask
                yield sse("progress", {"step": "feature", "pct": 20, "message": "特征矩阵就绪"})

                # ── 步骤 2：采样与训练 ───────────────────────────────────
                yield sse("progress", {"step": "train", "pct": 25, "message": "正在采样训练样本…"})
                sampled_indices, labeled_distribution = sample_training_indices(label_mask, max_samples_per_class)
                flat_stack = feature_stack.reshape(-1, feature_stack.shape[-1])
                train_features = flat_stack[sampled_indices]
                train_labels = label_mask.reshape(-1)[sampled_indices]
                finite_mask = np.all(np.isfinite(train_features), axis=1)
                train_features = train_features[finite_mask]
                train_labels = train_labels[finite_mask]

                if np.unique(train_labels).size < 2:
                    yield sse("error", {"message": "有效训练样本不足，可能全部落在无效像素上。"})
                    return

                yield sse("progress", {"step": "train", "pct": 35, "message": "正在训练随机森林…"})
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1,
                    max_features=None,
                    class_weight="balanced_subsample",
                    bootstrap=True,
                    oob_score=n_estimators >= 50,
                )
                model.fit(train_features, train_labels)
                yield sse("progress", {"step": "train", "pct": 60, "message": "训练完成，开始全图推理…"})

                # ── 步骤 3：分批推理（带进度） ───────────────────────────
                flat_features = feature_stack.reshape(-1, feature_stack.shape[-1])
                flat_prediction = np.zeros(flat_features.shape[0], dtype=np.uint8)
                flat_prediction[~feature_valid_mask.reshape(-1)] = 99
                valid_indices = np.flatnonzero(feature_valid_mask.reshape(-1))
                total_valid = valid_indices.size
                done = 0
                for start in range(0, total_valid, PREDICT_BATCH_SIZE):
                    end = start + PREDICT_BATCH_SIZE
                    batch_indices = valid_indices[start:end]
                    batch_pred = model.predict(flat_features[batch_indices]).astype(np.uint8)
                    flat_prediction[batch_indices] = batch_pred
                    done += len(batch_indices)
                    pct = 60 + int(done / max(total_valid, 1) * 35)
                    yield sse("progress", {"step": "infer", "pct": pct,
                                           "message": f"推理中… {done}/{total_valid} 像素"})

                prediction_mask = flat_prediction.reshape(feature_valid_mask.shape)
                session.prediction_mask = prediction_mask
                session.prediction_png = encode_mask_png(prediction_mask)

                prediction_distribution = count_labels(prediction_mask)
                yield sse("progress", {"step": "done", "pct": 100, "message": "推理完成"})
                yield sse("result", {
                    "message": "训练和推理已完成，可以继续修改 AI 结果后再次迭代。",
                    "labeledPixels": labeled_pixels,
                    "sampledPixels": int(train_labels.size),
                    "trainAccuracy": float(model.score(train_features, train_labels)),
                    "oobScore": float(model.oob_score_) if hasattr(model, "oob_score_") else None,
                    "labeledDistribution": labeled_distribution,
                    "predictionDistribution": prediction_distribution,
                    "predictionUrl": f"/api/sessions/{session_id}/prediction.png?ts={uuid.uuid4().hex}",
                    "predictionTifUrl": f"/api/sessions/{session_id}/prediction.tif",
                })
            except Exception as exc:
                yield sse("error", {"message": f"训练失败：{exc}"})

        return Response(generate(), mimetype="text/event-stream",
                        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    return app


def error_response(message: str, status_code: int):
    response = jsonify({"error": message})
    response.status_code = status_code
    return response


def discover_scene_dirs() -> List[Path]:
    scenes: List[Path] = []
    for item in sorted(WORKSPACE_ROOT.iterdir()):
        if item == APP_DIR or not item.is_dir():
            continue
        if list_band_files(item):
            scenes.append(item)
    return scenes


def normalize_folder_path(folder_value: str) -> Path:
    folder_value = (folder_value or "").strip()
    if not folder_value:
        raise ValueError("请先选择或输入一个数据文件夹。")

    candidate = Path(folder_value)
    if not candidate.is_absolute():
        candidate = (WORKSPACE_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"找不到文件夹：{candidate}")
    return candidate


def serialize_scene(scene_dir: Path, include_bands: bool) -> Dict[str, object]:
    bands = list_band_files(scene_dir) if include_bands else []
    payload: Dict[str, object] = {
        "name": scene_dir.name,
        "folderPath": str(scene_dir),
        "bandCount": len(bands) if include_bands else count_band_candidates(scene_dir),
    }
    if include_bands:
        payload["bands"] = bands
        payload["defaultBandKeys"] = pick_default_band_keys(bands)
    return payload


def count_band_candidates(scene_dir: Path) -> int:
    return len([path for path in scene_dir.rglob("*.tif") if is_band_file(path)])


def list_band_files(scene_dir: Path) -> List[Dict[str, object]]:
    band_files: List[Dict[str, object]] = []
    for path in sorted(scene_dir.rglob("*.tif")):
        if not is_band_file(path):
            continue
        try:
            with rasterio.open(path) as dataset:
                band_files.append(
                    {
                        "key": path.relative_to(scene_dir).as_posix(),
                        "name": path.name,
                        "relativePath": path.relative_to(scene_dir).as_posix(),
                        "width": dataset.width,
                        "height": dataset.height,
                        "dtype": dataset.dtypes[0],
                        "bandNumber": extract_band_number(path.name),
                    }
                )
        except Exception:
            continue
    return sorted(band_files, key=lambda b: (b.get("bandNumber") or 999))


def is_band_file(path: Path) -> bool:
    if path.name.lower() == "cloud.tif":
        return False
    return extract_band_number(path.name) is not None


def extract_band_number(filename: str) -> Optional[int]:
    match = BAND_PATTERN.search(filename)
    if not match:
        return None
    return int(match.group(1))


def pick_default_band_keys(bands: List[Dict[str, object]]) -> List[str]:
    preferred_orders = ([3, 2, 1], [4, 3, 2], [9, 8, 7])
    by_band: Dict[int, str] = {}
    for band in bands:
        band_number = band.get("bandNumber")
        key = band.get("key")
        if isinstance(band_number, int) and isinstance(key, str) and band_number not in by_band:
            by_band[band_number] = key
    for order in preferred_orders:
        if all(number in by_band for number in order):
            return [by_band[number] for number in order]
    return [band["key"] for band in bands[:3] if isinstance(band.get("key"), str)]


def resolve_band_path(scene_dir: Path, band_value: str) -> Path:
    band_value = (band_value or "").strip()
    if not band_value:
        raise ValueError("存在空的波段路径。")

    candidate = Path(band_value)
    if not candidate.is_absolute():
        candidate = (scene_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists() or not candidate.is_file():
        raise ValueError(f"找不到波段文件：{candidate}")
    if candidate.suffix.lower() != ".tif":
        raise ValueError(f"不支持的文件类型：{candidate.name}")
    return candidate


def build_feature_stack(band_paths: List[Path], target_shape: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """从多个波段文件构建特征栈，不生成预览图。"""
    arrays: List[np.ndarray] = []
    reference_shape: Optional[Tuple[int, int]] = target_shape
    for path in band_paths:
        array = read_band(path, reference_shape)
        if reference_shape is None:
            reference_shape = array.shape
        arrays.append(array)
    stack = np.stack(arrays, axis=-1).astype(np.float32)
    valid_mask = np.all(np.isfinite(stack), axis=-1) & np.any(np.abs(stack) > 1e-6, axis=-1)
    if not np.any(valid_mask):
        raise ValueError("特征波段叠加后没有可用像素，请检查数据内容。")
    return stack, valid_mask


def add_texture_features(stack: np.ndarray, neighborhood_size: int) -> np.ndarray:
    """对每个波段计算邻域均值和标准差，拼接为纹理特征。"""
    n_bands = stack.shape[-1]
    mean_layers = np.empty_like(stack)
    std_layers = np.empty_like(stack)
    for i in range(n_bands):
        ch = stack[..., i]
        mean = uniform_filter(ch, size=neighborhood_size, mode="reflect")
        mean_sq = uniform_filter(ch * ch, size=neighborhood_size, mode="reflect")
        mean_layers[..., i] = mean
        std_layers[..., i] = np.sqrt(np.maximum(0.0, mean_sq - mean * mean))
    return np.concatenate([stack, mean_layers, std_layers], axis=-1)


def build_stack_and_preview(band_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, bytes]:
    arrays: List[np.ndarray] = []
    reference_shape: Optional[Tuple[int, int]] = None
    for path in band_paths:
        array = read_band(path, reference_shape)
        if reference_shape is None:
            reference_shape = array.shape
        arrays.append(array)

    stack = np.stack(arrays, axis=-1).astype(np.float32)
    valid_mask = np.all(np.isfinite(stack), axis=-1) & np.any(np.abs(stack) > 1e-6, axis=-1)
    if not np.any(valid_mask):
        raise ValueError("三波段叠加后没有可用像素，请检查数据内容。")

    preview_rgb = percent_stretch_rgb(stack, valid_mask, lower=2.0, upper=98.0)
    return stack, valid_mask, encode_rgb_png(preview_rgb)


def read_band(path: Path, target_shape: Optional[Tuple[int, int]]) -> np.ndarray:
    with rasterio.open(path) as dataset:
        nodata = dataset.nodata
        if target_shape is None or (dataset.height, dataset.width) == target_shape:
            raw = dataset.read(1)
        else:
            raw = dataset.read(1, out_shape=target_shape, resampling=Resampling.bilinear)
    # 不使用 masked=True，避免 uint16 等整型 masked array 无法表示 nan 的问题
    arr = raw.astype(np.float32)
    if nodata is not None:
        if np.isnan(float(nodata)):
            arr[np.isnan(arr)] = 0.0
        else:
            arr[raw == raw.dtype.type(nodata)] = 0.0
    return arr


def percent_stretch_rgb(stack: np.ndarray, valid_mask: np.ndarray, lower: float, upper: float) -> np.ndarray:
    rgb = np.zeros(stack.shape, dtype=np.uint8)
    for channel_index in range(stack.shape[-1]):
        channel = stack[..., channel_index]
        samples = channel[valid_mask]
        if samples.size == 0:
            continue
        p_low, p_high = np.percentile(samples, [lower, upper])
        if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
            p_low = float(np.nanmin(samples))
            p_high = float(np.nanmax(samples))
        if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
            continue
        stretched = np.clip(channel, p_low, p_high)
        stretched = (stretched - p_low) / (p_high - p_low)
        rgb[..., channel_index] = np.where(valid_mask, np.round(stretched * 255), 0).astype(np.uint8)
    return rgb


def encode_rgb_png(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def encode_mask_png(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def send_png_bytes(payload: bytes, download_name: str):
    buffer = io.BytesIO(payload)
    buffer.seek(0)
    response = send_file(buffer, mimetype="image/png", download_name=download_name, max_age=0)
    response.headers["Cache-Control"] = "no-store"
    return response


def register_session(session: SessionState) -> None:
    SESSIONS[session.session_id] = session
    SESSION_ORDER.append(session.session_id)
    while len(SESSION_ORDER) > SESSION_LIMIT:
        stale_session_id = SESSION_ORDER.pop(0)
        SESSIONS.pop(stale_session_id, None)


def get_session(session_id: str) -> SessionState:
    session = SESSIONS.get(session_id)
    if session is None:
        raise KeyError(f"未找到会话：{session_id}")
    return session


def parse_uploaded_mask(file_storage, expected_shape: Tuple[int, int]) -> np.ndarray:
    image = Image.open(file_storage.stream).convert("L")
    expected_height, expected_width = expected_shape
    if image.size != (expected_width, expected_height):
        raise ValueError(
            f"掩膜尺寸不匹配：收到 {image.size[0]}x{image.size[1]}，期望 {expected_width}x{expected_height}。"
        )
    mask = np.asarray(image, dtype=np.uint8)
    if mask.max() > MAX_CLASS_ID:
        raise ValueError(f"类别编号不能超过 {MAX_CLASS_ID}。")
    return mask


def parse_uploaded_mask_bytes(buf: io.BytesIO, expected_shape: Tuple[int, int]) -> np.ndarray:
    """从 BytesIO 对象解析掩膜（供 SSE 生成器使用）。"""
    image = Image.open(buf).convert("L")
    expected_height, expected_width = expected_shape
    if image.size != (expected_width, expected_height):
        raise ValueError(
            f"掩膜尺寸不匹配：收到 {image.size[0]}x{image.size[1]}，期望 {expected_width}x{expected_height}。"
        )
    mask = np.asarray(image, dtype=np.uint8)
    if mask.max() > MAX_CLASS_ID:
        raise ValueError(f"类别编号不能超过 {MAX_CLASS_ID}。")
    return mask


def encode_prediction_tif(prediction_mask: np.ndarray, reference_band: Optional[Path]) -> bytes:
    """将推理结果编码为与参考波段坐标投影完全一致的 GeoTIFF。"""
    height, width = prediction_mask.shape
    crs = None
    transform = None
    if reference_band is not None and reference_band.exists():
        try:
            with rasterio.open(reference_band) as ref:
                crs = ref.crs
                # 若参考波段尺寸与掩膜不一致（发生了重采样），需重新计算 transform
                if ref.height == height and ref.width == width:
                    transform = ref.transform
                elif ref.transform is not None:
                    # 根据边界重新算 transform
                    left, bottom, right, top = ref.bounds
                    transform = from_bounds(left, bottom, right, top, width, height)
        except Exception:
            pass  # 读取参考信息失败时退化为无投影

    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(prediction_mask.astype(np.uint8), 1)
    return buf.getvalue()


def sample_training_indices(label_mask: np.ndarray, max_samples_per_class: int) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    rng = np.random.default_rng(42)
    flat = label_mask.reshape(-1)
    sampled_groups: List[np.ndarray] = []
    distribution: List[Dict[str, int]] = []

    for class_id in np.unique(flat):
        if class_id == 0:
            continue
        indices = np.flatnonzero(flat == class_id)
        distribution.append({"classId": int(class_id), "count": int(indices.size)})
        if indices.size > max_samples_per_class:
            indices = rng.choice(indices, size=max_samples_per_class, replace=False)
        sampled_groups.append(indices)

    if not sampled_groups:
        raise ValueError("没有可用于训练的类别样本。")

    return np.concatenate(sampled_groups), distribution


def predict_full_image(model: RandomForestClassifier, stack: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    flat_features = stack.reshape(-1, stack.shape[-1])
    flat_prediction = np.zeros(flat_features.shape[0], dtype=np.uint8)
    # 不在 valid_mask 内的像素（含空值填0区域）赋值 99，不参与推理
    flat_prediction[~valid_mask.reshape(-1)] = 99
    valid_indices = np.flatnonzero(valid_mask.reshape(-1))
    for start in range(0, valid_indices.size, PREDICT_BATCH_SIZE):
        end = start + PREDICT_BATCH_SIZE
        batch_indices = valid_indices[start:end]
        batch_prediction = model.predict(flat_features[batch_indices]).astype(np.uint8)
        flat_prediction[batch_indices] = batch_prediction
    return flat_prediction.reshape(valid_mask.shape)


def count_labels(mask: np.ndarray) -> List[Dict[str, int]]:
    values, counts = np.unique(mask, return_counts=True)
    result: List[Dict[str, int]] = []
    for value, count in zip(values.tolist(), counts.tolist()):
        if value == 0:
            continue
        result.append({"classId": int(value), "count": int(count)})
    return result


def clamp_int(raw_value: str, minimum: int, maximum: int) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError("随机森林参数必须是整数。") from None
    return max(minimum, min(maximum, value))


def parse_optional_depth(raw_value: str) -> Optional[int]:
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return None
    return clamp_int(raw_value, 1, 128)


app = create_app()
print(app)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    host = os.environ.get("RF_LABEL_WEB_HOST", DEFAULT_HOST)
    port = int(os.environ.get("RF_LABEL_WEB_PORT", DEFAULT_PORT))
    app.run(host=host, port=port, debug=False)