"""
ทดสอบ api/main.py ด้วยภาพปลอม (PIL + numpy) — ต้องมี checkpoint ของ config.APP_MODELS ใน checkpoints/

    pytest -q
"""
from __future__ import annotations

import io
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from src.config import API_MAX_UPLOAD_MB, APP_MODELS, CLASSES

client = TestClient(app)


def fake_jpeg(size: tuple[int, int] = (320, 240), seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def post(url: str, data: bytes, name: str = "plant.jpg", **params):
    return client.post(url, params=params, files={"file": (name, data, "image/jpeg")})


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_models_lists_app_models():
    body = client.get("/api/models").json()
    assert [m["key"] for m in body] == APP_MODELS
    assert all(m["run"].startswith("flat_") for m in body)


@pytest.mark.parametrize("model", APP_MODELS)
def test_predict_schema(model: str):
    r = post("/api/predict", fake_jpeg(), model=model)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == model and body["gradcam_png"] is None
    assert body["genus"]["key"] in CLASSES and body["top_genera"][0] == body["genus"]
    probs = [g["prob"] for g in body["top_genera"]]
    assert probs == sorted(probs, reverse=True) and sum(probs) <= 1 + 1e-6
    if body["subclasses"]:                      # prob ภายในสกุลรวมกันได้ 1
        assert math.isclose(sum(s["prob"] for s in body["subclasses"]), 1, abs_tol=1e-4)
    if not body["confident"]:                   # ไม่มั่นใจ → มีข้อความ ไม่มีคำแนะนำ
        assert body["message"] and body["care"] is None


def test_predict_gradcam_png():
    body = post("/api/predict", fake_jpeg(), gradcam="true").json()
    assert body["gradcam_png"].startswith("data:image/png;base64,")


def test_compare_runs_every_model():
    body = post("/api/compare", fake_jpeg(seed=1)).json()
    assert [b["model"] for b in body] == APP_MODELS


def test_rejects_non_image():
    r = post("/api/predict", b"not an image at all", name="note.txt")
    assert r.status_code == 400


def test_rejects_unknown_model():
    assert post("/api/predict", fake_jpeg(), model="vgg16").status_code == 422


def test_rejects_oversized_upload():
    big = b"\xff" * (API_MAX_UPLOAD_MB * 1024 * 1024 + 10)
    assert post("/api/predict", big).status_code == 413


def test_rejects_empty_upload():
    assert post("/api/predict", b"").status_code == 400
