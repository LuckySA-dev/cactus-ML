"""
predict.py — ภาพ → สกุล + คลาสย่อย + ความมั่นใจ → ถ้ามั่นใจพอ ดึงคำแนะนำการดูแลจาก care_rules.json

    python -m src.predict photo.jpg                     # 1 ภาพ (โมเดล config.DEPLOY_MODEL, seed ที่ val ดีสุด)
    python -m src.predict data/raw/handon/astrophytum   # ทั้งโฟลเดอร์
    python -m src.predict photo.jpg --model resnet50 --topk 3 --cpu
    python -m src.predict photo.jpg --run flat_resnet50_seed43

ใช้ในแอป:
    predictor = Predictor(best_run("effnetv2b0"))         # โหลดครั้งเดียว (app ใช้ st.cache_resource)
    result = predictor.predict(load_image(uploaded))      # load_image หมุนภาพตาม EXIF ให้

ใช้ได้ทั้ง checkpoint งาน flat (ทาย leaf → สกุล = ผลรวม prob ของ leaf ในสกุล) และงาน genus (ชั้นเดียว)
    ความมั่นใจสกุล < CONFIDENCE_THRESHOLD : "ไม่แน่ใจ กรุณาถ่ายใหม่" — ไม่ให้คำแนะนำ (แนะนำผิดสกุลแล้วต้นเน่าได้)
    ไม่งั้น lookup care_rules.json ด้วยสกุลที่ทาย (rule-based ไม่ใช่ ML)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import (
    CARE_RULES_JSON,
    CARE_TOPICS,
    CHECKPOINT_DIR,
    CLASS_TH,
    CLASSES,
    CONFIDENCE_THRESHOLD,
    DEPLOY_MODEL,
    GRADCAM_LAYERS,
    RUNS_CSV,
    SUBCLASS_TH,
)
from src.datasets import build_transforms, load_image
from src.models import load_checkpoint
from src.tasks import LEAF_SEP, genus_of

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UNSURE_MESSAGE = "ไม่แน่ใจ กรุณาถ่ายใหม่ (ให้เห็นทั้งต้น ชัด แสงพอ)"


@dataclass
class Prediction:
    genus: str | None                    # None = มั่นใจไม่ถึง threshold
    confidence: float                    # ความน่าจะเป็นของสกุลอันดับ 1
    topk: list[tuple[str, float]]        # [(สกุล, prob)] เรียงมากไปน้อย
    subclasses: list[tuple[str, float]]  # [(คลาสย่อย, prob ภายในสกุลอันดับ 1)] — ว่าง = สกุลนี้ไม่มีชั้น 2
    care: dict | None                    # คำแนะนำ (None = ไม่มั่นใจ หรือ care_rules ของสกุลนี้ยังไม่ครบ)
    class_index: int                     # index คลาสของโมเดลที่ทาย (ใช้กับ Grad-CAM)

    @property
    def confident(self) -> bool:
        return self.genus is not None

    @property
    def subclass(self) -> str | None:
        """คลาสย่อยอันดับ 1 ถ้ามั่นใจพอ"""
        if self.confident and self.subclasses and self.subclasses[0][1] >= CONFIDENCE_THRESHOLD:
            return self.subclasses[0][0]
        return None


def best_run(model: str = DEPLOY_MODEL, task: str = "flat") -> str:
    """seed ที่ val macro F1 สูงสุดของโมเดลนี้ (เลือกด้วย val — ไม่ดู test) จาก reports/runs.csv"""
    with RUNS_CSV.open(encoding="utf-8-sig", newline="") as f:
        runs = [r for r in csv.DictReader(f)
                if r["model"] == model and r.get("task") == task and r["phase"] == "final"
                and (CHECKPOINT_DIR / f"{r['run']}_best.pt").exists()]
    if not runs:
        raise SystemExit(f"ไม่มี checkpoint งาน {task} ของ {model} — เทรนก่อน: "
                         f"python -m src.train --model {model} --task {task}")
    return max(runs, key=lambda r: float(r["best_val_f1"]))["run"]


def missing_care_fields(entry: dict) -> list[str]:
    """หัวข้อที่ยังว่างของสกุลหนึ่ง — ถ้ามี จะไม่แสดงคำแนะนำของสกุลนั้น (กันแสดงครึ่ง ๆ / ไม่มีอ้างอิง)"""
    missing = [f"{topic}.summary" for topic in CARE_TOPICS if not entry.get(topic, {}).get("summary", "").strip()]
    return missing + [key for key in ("common_mistake", "reference") if not entry.get(key, "").strip()]


def load_care_rules(path: Path = CARE_RULES_JSON) -> tuple[dict[str, dict], list[str]]:
    """(สกุลที่เนื้อหาครบ, ข้อความเตือนสกุลที่ยังไม่ครบ)"""
    if not path.exists():
        return {}, [f"ไม่พบ {path.name} — ทำนายได้ แต่ไม่มีคำแนะนำการดูแล"]
    rules = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(rules) - set(CLASSES))
    if unknown:
        raise SystemExit(f"{path.name} มีสกุลที่ไม่อยู่ใน config.CLASSES: {unknown}")
    complete, warnings = {}, []
    for genus in CLASSES:
        missing = missing_care_fields(rules.get(genus, {}))
        if missing:
            warnings.append(f"{genus}: ยังไม่ครบ ({', '.join(missing)})")
        else:
            complete[genus] = rules[genus]
    return complete, warnings


class Predictor:
    """โหลดโมเดล + care rules ครั้งเดียว แล้วทำนายได้หลายภาพ"""

    def __init__(self, run: str | None = None, device: str | None = None):
        self.run = run or best_run()
        path = CHECKPOINT_DIR / f"{self.run}_best.pt"
        if not path.exists():
            raise SystemExit(f"ไม่พบ {path} — เทรนก่อน")
        # เครื่องที่ไม่มี GPU (เช่นเครื่องที่เปิด demo) ใช้ CPU ได้ — ~25 ms/ภาพ สำหรับ effnetv2b0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.ckpt = load_checkpoint(path, self.device)
        _, self.transform = build_transforms(self.ckpt["mean"], self.ckpt["std"])
        self.classes: list[str] = self.ckpt["classes"]
        # [สกุล, คลาสของโมเดล] — งาน genus ได้ identity, งาน flat รวม leaf เป็นสกุล
        self.member = np.array([[genus_of(c) == g for c in self.classes] for g in CLASSES], dtype=float)
        self.care, self.care_warnings = load_care_rules()

    def _input(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image, topk: int = 3) -> Prediction:
        """image ควรเปิดด้วย load_image() (หมุนตาม EXIF แล้ว)"""
        probs = self.model(self._input(image)).float().softmax(1)[0].cpu().numpy()
        genus_probs = self.member @ probs
        order = genus_probs.argsort()[::-1][:topk]
        top = [(CLASSES[i], float(genus_probs[i])) for i in order]
        genus, confidence = top[0]

        in_genus = [i for i, c in enumerate(self.classes) if genus_of(c) == genus]
        subs = sorted(((self.classes[i].split(LEAF_SEP)[1], float(probs[i] / confidence))
                       for i in in_genus if LEAF_SEP in self.classes[i]), key=lambda s: -s[1])
        best_class = max(in_genus, key=lambda i: probs[i])
        if confidence < CONFIDENCE_THRESHOLD:
            return Prediction(None, confidence, top, subs, None, best_class)
        return Prediction(genus, confidence, top, subs, self.care.get(genus), best_class)

    def explain(self, image: Image.Image, class_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Grad-CAM → (ภาพที่โมเดลเห็น [224,224,3] 0–1, ความสำคัญ [224,224] 0–1) ต่อคลาสที่ทาย"""
        from pytorch_grad_cam import GradCAM  # import เฉพาะตอนใช้ — CLI ทำนายอย่างเดียวไม่ต้องโหลด
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        x = self._input(image)
        layer = self.model.get_submodule(GRADCAM_LAYERS[self.ckpt["model_key"]])
        with GradCAM(model=self.model, target_layers=[layer]) as cam:
            heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_index)])[0]
        mean = torch.tensor(self.ckpt["mean"]).view(3, 1, 1)
        std = torch.tensor(self.ckpt["std"]).view(3, 1, 1)
        return (x[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy(), heat


def iter_images(paths: list[str]) -> Iterator[Path]:
    """ไฟล์ภาพ หรือทุกภาพในโฟลเดอร์ (PowerShell ไม่ขยาย *.jpg ให้ จึงรับโฟลเดอร์ได้)"""
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTS)
        else:
            yield p


def format_prediction(pred: Prediction) -> list[str]:
    if pred.confident:
        lines = [f"→ {CLASS_TH[pred.genus]} ({pred.genus})  {pred.confidence:.1%}"]
        if pred.subclasses:
            sub, prob = pred.subclasses[0]
            note = "" if pred.subclass else f"  (ไม่แน่ใจ < {CONFIDENCE_THRESHOLD:.0%})"
            lines.append(f"   ชนิด: {SUBCLASS_TH[sub]}  {prob:.1%}{note}")
    else:
        best, prob = pred.topk[0]
        lines = [f"→ {UNSURE_MESSAGE}", f"   (สูงสุด {best} {prob:.1%} < {CONFIDENCE_THRESHOLD:.0%})"]
    lines += [f"   อันดับ {rank}: {genus} {prob:.1%}" for rank, (genus, prob) in enumerate(pred.topk[1:], 2)]
    if pred.care:
        lines += [f"   {label}: {pred.care[topic]['summary']}" for topic, label in CARE_TOPICS.items()]
        lines += [f"   ⚠️ ข้อผิดพลาดที่พบบ่อย: {pred.care['common_mistake']}",
                  f"   อ้างอิง: {pred.care['reference']}"]
    elif pred.confident:
        lines.append("   (ยังไม่มีคำแนะนำการดูแลของสกุลนี้ — เติม care_rules/care_rules.json)")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="ทำนายสกุล + ชนิดกระบองเพชร + คำแนะนำการดูแล")
    p.add_argument("paths", nargs="+", help="ไฟล์ภาพ หรือโฟลเดอร์")
    p.add_argument("--model", default=DEPLOY_MODEL, help="ใช้ seed ที่ val ดีสุดของโมเดลนี้")
    p.add_argument("--run", help="ระบุ checkpoint ตรง ๆ เช่น flat_resnet50_seed43 (แทน --model)")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--cpu", action="store_true", help="บังคับใช้ CPU (จำลองเครื่องที่ไม่มี GPU)")
    args = p.parse_args()

    predictor = Predictor(args.run or best_run(args.model), "cpu" if args.cpu else None)
    for warning in predictor.care_warnings:
        print(f"⚠️  care_rules — {warning}")
    print(f"model: {predictor.run} ({predictor.device})\n")
    for path in iter_images(args.paths):
        print(path)
        for line in format_prediction(predictor.predict(load_image(path), args.topk)):
            print(f"  {line}")


if __name__ == "__main__":
    main()
