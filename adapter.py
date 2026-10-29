"""Kokoro-82M TTS adapter（EntryPoint 模块契约：/health /info /predict/speak）。

推理经 kokoro-onnx（ONNX Runtime，纯 CPU）：Kokoro.create() 对长文本自动按
~510 token 音素块分块产出 (samples, sample_rate) 流，adapter 顺序拼接写
24kHz WAV——上层无需自行分句。

产物协议（MODULE_SPEC §5）：params.output_path（daemon 注入）优先落盘，
缺省落 EP_WORKSPACE/模块临时目录；响应 {status, output_type:"file",
result/output_path, metadata, elapsed_seconds}。
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kokoro-tts")

app = FastAPI(title="kokoro-tts", version="1.0.0")

MODULE_ID = "kokoro-tts"
DEFAULT_MODEL_FILE = "kokoro-v1.0.onnx"
DEFAULT_VOICES_FILE = "voices-v1.0.bin"
DEFAULT_LANG = "en-us"

# 全局引擎（startup 惰性加载；测试可注入 FakeKokoro）
_engine: Any = None
_engine_error: Optional[str] = None
_engine_meta: dict[str, Any] = {}


# ─── 路径解析 ────────────────────────────────────────────────────────────────


def _module_dir() -> Path:
    return Path(os.environ.get("MODULE_DIR", Path(__file__).resolve().parent))


def _candidate_roots() -> list[Path]:
    base = _module_dir()
    roots = [base, base.parent]
    for d in sorted(base.glob("*/")):
        roots.append(d)
    return roots


def _find_model_file(model_file: str, voices_file: str) -> Optional[tuple[Path, Path]]:
    """在候选根下定位 (model.onnx, voices.bin)；daemon 下载协议落位
    MODULE_DIR/<target_dir>/ 与 MODULE_DIR 同级均尝试。"""
    for root in _candidate_roots():
        m = root / model_file
        v = root / voices_file
        if m.is_file() and v.is_file():
            return m, v
    return None


def _workspace_dir() -> Path:
    ws = os.environ.get("EP_WORKSPACE")
    if ws:
        p = Path(ws)
        p.mkdir(parents=True, exist_ok=True)
        return p
    fallback = _module_dir() / "output"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# ─── 引擎加载 ────────────────────────────────────────────────────────────────


def load_engine(model_file: str = DEFAULT_MODEL_FILE, voices_file: str = DEFAULT_VOICES_FILE) -> bool:
    """加载 kokoro-onnx 引擎（startup 或首请求时调用）。成功返回 True。"""
    global _engine, _engine_error, _engine_meta
    if _engine is not None:
        return True
    try:
        from kokoro_onnx import Kokoro

        found = _find_model_file(model_file, voices_file)
        if found is None:
            _engine_error = (
                f"model files not found: {model_file} / {voices_file} "
                f"(searched {[_candidate_roots()][0]}; daemon 下载协议落位后重启)"
            )
            logger.error(_engine_error)
            return False
        model_path, voices_path = found
        # kokoro-onnx 0.4.x：构造函数直载（无 from_local）；get_voices() 返回
        # 音色名 → 风格向量字典
        _engine = Kokoro(str(model_path), str(voices_path))
        voices = []
        try:
            voices = sorted(_engine.get_voices().keys())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - 音色枚举失败不阻断
            pass
        _engine_meta = {
            "model_file": model_path.name,
            "voices_file": voices_path.name,
            "model_size_mb": round(model_path.stat().st_size / 1024 / 1024, 1),
            "voices": voices,
        }
        logger.info("kokoro engine loaded: %s", _engine_meta)
        return True
    except Exception as exc:  # noqa: BLE001
        _engine_error = f"{type(exc).__name__}: {exc}"
        logger.exception("kokoro engine load failed")
        return False


# ─── 参数校验（纯函数，可单测）──────────────────────────────────────────────


def validate_params(params: dict[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    """校验/归一 speak 参数；返回 (归一化参数, 错误消息)。"""
    text = (params.get("text") or "").strip() if isinstance(params.get("text"), str) else ""
    if not text:
        return {}, "缺少输入文本（multipart 'file' 文本文件或 JSON 'input'/'input_text'）"
    lang = params.get("lang") or DEFAULT_LANG
    if lang not in ("en-us", "en-gb", "ja"):
        return {}, f"lang 仅支持 en-us/en-gb/ja，收到: {lang}"
    try:
        speed = float(params.get("speed") or 1.0)
    except (TypeError, ValueError):
        return {}, f"speed 非法: {params.get('speed')}"
    if not 0.5 <= speed <= 2.0:
        return {}, f"speed 超出 [0.5, 2.0]: {speed}"
    voice = (params.get("voice") or "af_heart").strip() or "af_heart"
    model_file = (params.get("model_file") or "").strip() or DEFAULT_MODEL_FILE
    voices_file = (params.get("voices_file") or "").strip() or DEFAULT_VOICES_FILE
    norm = {
        "text": text, "lang": lang, "speed": speed, "voice": voice,
        "model_file": model_file, "voices_file": voices_file,
    }
    # 未声明键透传（daemon 产物协议的 output_path 等；openai-bridge 同惯例）
    for key, value in params.items():
        if key not in norm:
            norm[key] = value
    return norm, None


def resolve_output_path(params: dict[str, Any]) -> Path:
    injected = params.get("output_path")
    if injected:
        p = Path(str(injected))
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    ws = _workspace_dir()
    return ws / f"kokoro_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"


# ─── 合成 ────────────────────────────────────────────────────────────────────


def _read_text_file(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_params(request_body: Optional[dict], file_text: Optional[str], form_params: Optional[str]) -> dict:
    """双格式归一：JSON body 契约为 ``{"input"|"input_text", "params": {..}}``
    （params 嵌套字典为参数权威来源，qwen3-tts 同款）；multipart 为
    ``file``（文本内容）+ ``params`` 表单 JSON。file 文本在 params 未显式
    给出 text 时作为输入。"""
    import json

    merged: dict[str, Any] = {}
    if request_body:
        p = request_body.get("params")
        if isinstance(p, dict):
            merged.update(p)
        raw_input = request_body.get("input")
        if raw_input is None:
            raw_input = request_body.get("input_text")
        if isinstance(raw_input, str):
            if raw_input and "text" not in merged:
                merged["text"] = raw_input
        elif raw_input is not None:
            merged["text"] = json.dumps(raw_input, ensure_ascii=False)
        input_path = request_body.get("input_path")
        if isinstance(input_path, str) and input_path and "text" not in merged:
            try:
                merged["text"] = Path(input_path).read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("input_path 不可读: %s", exc)
    if file_text is not None and not merged.get("text"):
        merged["text"] = file_text
    if form_params:
        try:
            merged.update(json.loads(form_params))
        except ValueError:
            pass
    return merged


def synthesize_to_file(norm: dict[str, Any], output_path: Path) -> tuple[float, int, float]:
    """调用引擎合成并写 WAV；返回 (时长秒, 采样率, 合成耗时秒)。

    kokoro-onnx 0.4.2 的 create() 返回单 (samples, sample_rate) 元组；兼容
    生成器形态（分块 yield）以防上游改回流式。"""
    import soundfile as sf

    t0 = time.time()
    result = _engine.create(
        norm["text"],
        voice=norm["voice"],
        speed=norm["speed"],
        lang=norm["lang"],
    )
    import numpy as np

    samples_parts: list[Any] = []
    sample_rate = 24000
    if isinstance(result, tuple) and len(result) == 2 and np.isscalar(result[1]):
        samples, sample_rate = result
        samples_parts.append(np.asarray(samples, dtype=np.float32))
    else:
        for samples, sr in result:
            sample_rate = sr
            samples_parts.append(np.asarray(samples, dtype=np.float32))
    if not samples_parts:
        raise RuntimeError("engine produced no audio (empty input after phonemization?)")
    audio = samples_parts[0] if len(samples_parts) == 1 else np.concatenate(samples_parts)
    sf.write(str(output_path), audio, sample_rate)
    elapsed = time.time() - t0
    duration = len(audio) / sample_rate
    return duration, sample_rate, elapsed


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def ensure_engine(model_file: str, voices_file: str) -> Optional[str]:
    """确保引擎按请求的模型文件就绪；返回 None=就绪，否则错误消息。"""
    global _engine, _engine_error
    if _engine is not None and _engine_meta.get("model_file") == model_file:
        return None
    if _engine is not None:
        _engine = None  # 请求了不同模型文件（如 v1.1-zh 实验）：重载
    if not load_engine(model_file, voices_file):
        return _engine_error or "engine not loaded"
    return None


async def _predict(request: Request, file: Optional[UploadFile], params_form: Optional[str]) -> JSONResponse:
    body: Optional[dict] = None
    file_text: Optional[str] = None
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    elif file is not None:
        file_text = _read_text_file(await file.read())

    norm, err = validate_params(extract_params(body, file_text, params_form))
    if err:
        return _error(400, "INVALID_PARAMS", err)

    if err2 := ensure_engine(norm["model_file"], norm["voices_file"]):
        return _error(503, "MODEL_NOT_READY", err2)

    output_path = resolve_output_path(norm)
    try:
        duration, sample_rate, elapsed = synthesize_to_file(norm, output_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("speak synthesis failed")
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        return _error(500, "INFERENCE_ERROR", f"speak synthesis failed: {exc}")

    return JSONResponse({
        "status": "completed",
        "output_type": "file",
        "result": str(output_path.resolve()),
        "output_path": str(output_path.resolve()),
        "metadata": {
            "voice": norm["voice"],
            "lang": norm["lang"],
            "speed": norm["speed"],
            "duration_seconds": round(duration, 2),
            "sample_rate": sample_rate,
            "model_file": _engine_meta.get("model_file"),
        },
        "elapsed_seconds": round(elapsed, 2),
    })


def _make_predict_endpoint(cap: str):
    async def _predict_endpoint(
        request: Request,
        file: Optional[UploadFile] = File(None),
        params_form: Optional[str] = Form(None, alias="params"),
    ):
        return await _predict(request, file, params_form)

    _predict_endpoint.__name__ = f"predict_{cap}"
    return _predict_endpoint


app.post("/predict/speak")(_make_predict_endpoint("speak"))


@app.get("/health")
async def health():
    return {
        "status": "ok" if _engine is not None else ("degraded" if _engine_error else "starting"),
        "engine": "kokoro-onnx" if _engine is not None else None,
        "model_id": MODULE_ID,
        "error": _engine_error,
        "meta": _engine_meta or None,
    }


@app.get("/info")
async def info():
    return {
        "module_id": MODULE_ID,
        "name": "Kokoro TTS",
        "version": "1.0.0",
        "device": "cpu",
        "backend": "cpu",
        "capabilities": ["speak"],
        "meta": _engine_meta or None,
    }


@app.on_event("startup")
async def _startup():
    # 惰性加载：模型文件未就绪时不阻塞 health 探测（ready_timeout 语义交给
    # /health 的 degraded→ok 转变；首请求会再尝试加载）
    load_engine()
