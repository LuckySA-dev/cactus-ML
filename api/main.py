"""
api/main.py — REST API หลังบ้านสำหรับหน้าเว็บ (React ฯลฯ): ภาพ → สกุล + ชนิด + ความมั่นใจ + Grad-CAM + วิธีดูแล

    uvicorn api.main:app --reload --port 8000          # docs อัตโนมัติที่ http://localhost:8000/docs

    GET  /api/health                                   เซิร์ฟเวอร์พร้อมไหม
    GET  /api/models                                   โมเดลที่ใช้ได้ (APP_MODELS) + คะแนนจาก model_compare.csv
    POST /api/predict?model=effnetv2b0&gradcam=false   multipart field "file" = ภาพ 1 ภาพ
    POST /api/compare?gradcam=false                    ภาพเดียว ทายด้วยทุกโมเดลใน APP_MODELS

ใช้ predict.Predictor ตัวเดียวกับแอป Streamlit / CLI → ผลตรงกับตัวเลขในรายงาน (fp32, eval transform เดียวกัน)
endpoint ทายเป็น def (ไม่ใช่ async) → FastAPI รันใน threadpool ไม่บล็อก event loop ระหว่างโมเดลคำนวณ
"""
from __future__ import annotations

import base64
import csv
import io
import sys
import threading
from functools import cache
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pillow_heif
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # ให้ import src ได้เมื่อรันจากโฟลเดอร์ไหนก็ได้

from src.config import (  # noqa: E402
    API_CORS_ORIGINS,
    API_MAX_UPLOAD_MB,
    APP_MODELS,
    CARE_TOPICS,
    CLASS_TH,
    CONFIDENCE_THRESHOLD,
    MODEL_NAMES,
    REPORT_DIR,
    SUBCLASS_TH,
)
from src.datasets import load_image  # noqa: E402
from src.gradcam import spotlight  # noqa: E402
from src.predict import NOT_FOUND_MESSAGE, UNSURE_MESSAGE, Prediction, Predictor, best_run  # noqa: E402

pillow_heif.register_heif_opener()          # ภาพจาก iPhone (.HEIC)
ModelKey = Literal[tuple(APP_MODELS)]       # ชื่อโมเดลอื่น → 422 อัตโนมัติ
# ponytail: lock เดียวทั้งเซิร์ฟเวอร์ — Grad-CAM ติด hook บนโมเดล ถ้ารันพร้อมกันจะชนกัน · ใช้ lock ต่อโมเดลถ้าคนใช้เยอะ
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# schemas — สัญญากับหน้าเว็บ (ดูใน /docs)
# ---------------------------------------------------------------------------
class ClassProb(BaseModel):
    key: str                 # ชื่อในระบบ เช่น astrophytum / asterias
    name_th: str
    prob: float


class CareTopic(BaseModel):
    summary: str
    detail: str


class Care(BaseModel):
    level: Literal["species", "genus"]      # species = คำแนะนำเฉพาะชนิด · genus = ทั่วไปของสกุล
    name_th: str
    water: CareTopic
    light: CareTopic
    soil: CareTopic
    common_mistake: str
    references: list[str]


class PredictionOut(BaseModel):
    model: str
    model_name: str
    run: str
    confident: bool                         # False = ความมั่นใจสกุล < threshold → ให้ผู้ใช้ถ่ายใหม่
    status: Literal["confident", "unsure", "not_found"]   # not_found = ไม่น่าใช่กระบองเพชร 7 สกุล (ไม่ควรแสดงสกุล)
    message: str | None                     # ข้อความสำหรับผู้ใช้เมื่อไม่มั่นใจ
    genus: ClassProb                        # สกุลอันดับ 1 (แม้ไม่มั่นใจก็ส่งมา ให้หน้าเว็บแสดงเป็น "ใกล้เคียงสุด")
    top_genera: list[ClassProb]
    subclass: ClassProb | None              # ชนิดอันดับ 1 ถ้ามั่นใจพอ (สกุลที่ไม่มีชั้น 2 = None)
    subclasses: list[ClassProb]             # ทุกชนิดในสกุลอันดับ 1 (prob ภายในสกุล)
    care: Care | None
    gradcam_png: str | None                 # data URL (image/png) ถ้าขอ gradcam=true


class ModelInfo(BaseModel):
    key: str
    name: str
    run: str
    genus_acc: float | None                 # เฉลี่ยทุก seed บน test (reports/model_compare.csv)
    species_acc: float | None
    realworld_genus_acc: float | None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@cache
def get_predictor(model: str) -> Predictor:
    return Predictor(best_run(model))


def read_image(file: UploadFile) -> Image.Image:
    """ตรวจไฟล์ที่ผู้ใช้อัปโหลด — ไม่ใช่ภาพ / ใหญ่เกิน / ภาพเสีย → 4xx พร้อมข้อความอ่านเข้าใจ"""
    data = file.file.read(API_MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > API_MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"ไฟล์ใหญ่เกิน {API_MAX_UPLOAD_MB} MB")
    if not data:
        raise HTTPException(400, "ไม่มีไฟล์ภาพ")
    try:
        return load_image(io.BytesIO(data))           # หมุนตาม EXIF เหมือนตอนเทรน
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise HTTPException(400, "อ่านไฟล์ภาพไม่ได้ (รองรับ JPG / PNG / WEBP / HEIC)") from e


def to_data_url(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((image * 255).clip(0, 255).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def care_out(predictor: Predictor, pred: Prediction) -> Care | None:
    if pred.care is None:
        return None
    level = "species" if pred.subclass and predictor.care.get((pred.genus, pred.subclass)) is pred.care else "genus"
    c = pred.care
    return Care(level=level, name_th=c["name_th"], **{t: CareTopic(**c[t]) for t in CARE_TOPICS},
                common_mistake=c["common_mistake"],
                references=[r.strip() for r in c["reference"].split(";") if r.strip()])


def run_model(model: str, image: Image.Image, gradcam: bool) -> PredictionOut:
    predictor = get_predictor(model)
    with _LOCK:
        pred = predictor.predict(image)
        heat_png = to_data_url(spotlight(*predictor.explain(image, pred.class_index))) if gradcam else None
    genus, prob = pred.topk[0]
    sub = pred.subclass
    return PredictionOut(
        model=model, model_name=MODEL_NAMES[model], run=predictor.run,
        confident=pred.confident, status=pred.status,
        message={"confident": None, "not_found": NOT_FOUND_MESSAGE}.get(
            pred.status, f"{UNSURE_MESSAGE} (ความมั่นใจ {prob:.0%} < {CONFIDENCE_THRESHOLD:.0%})"),
        genus=ClassProb(key=genus, name_th=CLASS_TH[genus], prob=prob),
        top_genera=[ClassProb(key=g, name_th=CLASS_TH[g], prob=p) for g, p in pred.topk],
        subclass=next((ClassProb(key=s, name_th=SUBCLASS_TH[s], prob=p) for s, p in pred.subclasses if s == sub),
                      None),
        subclasses=[ClassProb(key=s, name_th=SUBCLASS_TH[s], prob=p) for s, p in pred.subclasses],
        care=care_out(predictor, pred),
        gradcam_png=heat_png,
    )


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
app = FastAPI(title="Cactus Classifier API", version="1.0.0",
              description="จำแนกสกุลและชนิดกระบองเพชรแคระจากภาพ + คำแนะนำการดูแล (rule-based)")
app.add_middleware(CORSMiddleware, allow_origins=API_CORS_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"])


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    path = REPORT_DIR / "model_compare.csv"
    scores = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            scores = {r["model"]: r for r in csv.DictReader(f)}

    def num(key: str, col: str) -> float | None:
        v = scores.get(key, {}).get(col)
        return float(v) if v not in (None, "") else None

    return [ModelInfo(key=k, name=MODEL_NAMES[k], run=best_run(k), genus_acc=num(k, "genus_acc_mean"),
                      species_acc=num(k, "leaf_acc_mean"), realworld_genus_acc=num(k, "rw_genus_acc_mean"))
            for k in APP_MODELS]


@app.post("/api/predict", response_model=PredictionOut)
def predict(file: UploadFile, model: Annotated[ModelKey, Query()] = APP_MODELS[0],
            gradcam: bool = False) -> PredictionOut:
    return run_model(model, read_image(file), gradcam)


@app.post("/api/compare", response_model=list[PredictionOut])
def compare(file: UploadFile, gradcam: bool = False) -> list[PredictionOut]:
    image = read_image(file)
    return [run_model(m, image, gradcam) for m in APP_MODELS]
