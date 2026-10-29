"""kokoro-tts adapter 契约测试（FakeKokoro 注入，零真实推理）。"""

from __future__ import annotations

import json
import wave

from fastapi.testclient import TestClient

import adapter
from conftest import FakeKokoro


def _client() -> TestClient:
    return TestClient(adapter.app)


def test_health_contract(fake_engine):
    with _client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "kokoro-onnx"
    assert body["model_id"] == "kokoro-tts"


def test_info_contract(fake_engine):
    with _client() as c:
        r = c.get("/info")
    body = r.json()
    assert body["module_id"] == "kokoro-tts"
    assert body["capabilities"] == ["speak"]
    assert body["backend"] == "cpu"


def test_speak_json_body_writes_wav(fake_engine, tmp_path):
    out = tmp_path / "out.wav"
    with _client() as c:
        r = c.post(
            "/predict/speak",
            json={"input": "hello world", "params": {"output_path": str(out)}},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["output_type"] == "file"
    assert body["output_path"] == str(out)
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnframes() > 0
    # 引擎收到归一化参数：分块拼接（两次 yield）后单文件
    assert len(fake_engine.calls) == 1
    assert fake_engine.calls[0]["text"] == "hello world"
    assert fake_engine.calls[0]["voice"] == "af_heart"


def test_speak_default_params_filled(fake_engine, tmp_path):
    with _client() as c:
        r = c.post("/predict/speak", json={"input_text": "hi", "params": {"output_path": str(tmp_path / "b.wav")}})
    assert r.status_code == 200
    assert fake_engine.calls[0]["lang"] == "en-us"
    assert fake_engine.calls[0]["speed"] == 1.0


def test_speak_missing_text_is_400(fake_engine):
    with _client() as c:
        r = c.post("/predict/speak", json={"params": {}})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_PARAMS"


def test_speak_bad_lang_is_400(fake_engine):
    with _client() as c:
        r = c.post("/predict/speak", json={"input": "x", "params": {"lang": "fr"}})
    assert r.status_code == 400


def test_speak_bad_speed_is_400(fake_engine):
    with _client() as c:
        r = c.post("/predict/speak", json={"input": "x", "params": {"speed": 9.0}})
    assert r.status_code == 400


def test_speak_speed_boundary_values(fake_engine, tmp_path):
    # speed=2.0 上界合法；0.4 低于下界 400
    with _client() as c:
        r_ok = c.post("/predict/speak", json={"input": "x", "params": {"speed": 2.0, "output_path": str(tmp_path / "s2.wav")}})
        r_bad = c.post("/predict/speak", json={"input": "x", "params": {"speed": 0.4, "output_path": str(tmp_path / "s04.wav")}})
    assert r_ok.status_code == 200, r_ok.text
    assert r_bad.status_code == 400


def test_multipart_text_file(fake_engine, tmp_path):
    out = tmp_path / "m.wav"
    with _client() as c:
        r = c.post(
            "/predict/speak",
            files={"file": ("in.txt", b"multipart text", "text/plain")},
            data={"params": json.dumps({"output_path": str(out)})},
        )
    assert r.status_code == 200, r.text
    assert fake_engine.calls[0]["text"] == "multipart text"


def test_validate_params_pure():
    norm, err = adapter.validate_params({"text": " hi ", "lang": "ja", "speed": "1.5"})
    assert err is None
    assert norm["text"] == "hi"
    assert norm["lang"] == "ja"
    assert norm["speed"] == 1.5
    norm, err = adapter.validate_params({"text": ""})
    assert err and "缺少输入文本" in err


def test_engine_not_loaded_is_503(monkeypatch):
    monkeypatch.setattr(adapter, "_engine", None)
    monkeypatch.setattr(adapter, "_engine_error", "model files not found")
    monkeypatch.setattr(adapter, "load_engine", lambda *a, **k: False)
    with _client() as c:
        r = c.post("/predict/speak", json={"input": "x"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "MODEL_NOT_READY"


def test_engine_runtime_error_is_500(fake_engine, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("onnx exploded")

    fake_engine.create = boom
    with _client() as c:
        r = c.post("/predict/speak", json={"input": "x", "params": {"output_path": str(tmp_path / "e.wav")}})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "INFERENCE_ERROR"
