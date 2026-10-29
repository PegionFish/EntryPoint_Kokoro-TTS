"""kokoro-tts 测试夹具：注入 FakeKokoro（不发真实推理）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_DIR))

import adapter  # noqa: E402


class FakeKokoro:
    """引擎替身：按文本长度产已知正弦样本，记录调用参数。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.voices = ["af_heart", "af_bella", "jf_alpha"]

    def get_voice_names(self) -> list[str]:
        return self.voices

    def create(self, text: str, voice: str = "af_heart", speed: float = 1.0, lang: str = "en-us"):
        self.calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
        n = max(2400, len(text) * 240)
        yield np.linspace(-0.1, 0.1, n, dtype=np.float32), 24000
        yield np.linspace(0.1, -0.1, n // 2, dtype=np.float32), 24000


@pytest.fixture()
def fake_engine(monkeypatch, tmp_path):
    """装载 FakeKokoro 并把产物落点导到 tmp_path。"""
    fake = FakeKokoro()
    monkeypatch.setattr(adapter, "_engine", fake)
    monkeypatch.setattr(adapter, "_engine_error", None)
    monkeypatch.setattr(adapter, "_engine_meta", {"model_file": "kokoro-v1.0.onnx", "voices": fake.voices})
    monkeypatch.setenv("EP_WORKSPACE", str(tmp_path))
    return fake
