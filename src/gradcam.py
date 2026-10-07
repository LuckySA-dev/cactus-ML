"""
gradcam.py — ดูว่าโมเดล "มองตรงไหน" ตอนตัดสินใจ (Grad-CAM) → reports/figures/gradcam_*.png

    python -m src.gradcam                            # run หลัก seed 42 ของ config.APP_MODELS + รูปเทียบ
    python -m src.gradcam --run flat_resnet50_seed43 --n 2

ใช้ภาพจาก test (โมเดลไม่เคยเห็น) + ผลทำนายจาก reports/predictions/ → ต้องรัน evaluate ก่อน
ต่อ 1 run ได้
    gradcam_<run>_correct.png   ทายถูก n ภาพต่อคลาส (1 คอลัมน์ = 1 คลาส)
    gradcam_<run>_wrong.png     ทายผิดที่มั่นใจที่สุด — ดูว่าผิดเพราะอะไร
และ gradcam_compare.png         ภาพเดียวกันให้ 3 โมเดลอธิบาย (1 ภาพต่อคลาส)

วิธีอ่าน: สว่าง = ส่วนที่ผลักให้โมเดลเลือกคลาสที่ทาย · มืด = แทบไม่มีผล · เส้นขาว = 50% ของค่าสูงสุด
ถ้าสว่างที่กระถาง / ป้าย / พื้นหลัง แทนตัวต้น = โมเดลจำ shortcut → ต้องแก้ที่ข้อมูล
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from src.config import (
    APP_MODELS,
    CHECKPOINT_DIR,
    CLASSES,
    DEVICE,
    GRADCAM_LAYERS,
    MODEL_NAMES,
    PRED_DIR,
    PROJECT_ROOT,
    SEED,
)
from src.datasets import build_transforms, load_image
from src.models import load_checkpoint
from src.tasks import run_id
from src.visualize import INK2, SURFACE, new_figure, save

N_WRONG = 12


def explain(model: nn.Module, ckpt: dict, paths: list[str], targets: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    (ภาพที่ crop แล้ว [N,224,224,3] ค่า 0–1, cam [N,224,224] ค่า 0–1)
    ใช้ eval transform เดียวกับตอนประเมิน → cam ตรงกับภาพที่โมเดลเห็นจริง
    target = คลาสที่โมเดลทาย (อธิบายการตัดสินใจของโมเดล ไม่ใช่เฉลย)
    """
    _, eval_tf = build_transforms(ckpt["mean"], ckpt["std"])
    x = torch.stack([eval_tf(load_image(PROJECT_ROOT / p)) for p in paths])
    layer = model.get_submodule(GRADCAM_LAYERS[ckpt["model_key"]])
    with GradCAM(model=model, target_layers=[layer]) as cam:
        cams = cam(input_tensor=x.to(DEVICE), targets=[ClassifierOutputTarget(t) for t in targets])
    mean = torch.tensor(ckpt["mean"]).view(3, 1, 1)
    std = torch.tensor(ckpt["std"]).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy(), cams


def spotlight(image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """หรี่ส่วนที่โมเดลไม่ได้ใช้ — แทน colormap สีรุ้ง (jet): ไม่บังภาพ และอ่านได้แม้ตาบอดสี"""
    return image * (0.15 + 0.85 * cam[..., None])


def draw_grid(images, cams, captions: list[str], ncols: int, title: str, name: str) -> None:
    nrows = math.ceil(len(images) / ncols)
    caption_lines = max(c.count("\n") + 1 for c in captions)
    row_height = 1.75 + 0.17 * caption_lines                     # เผื่อที่ให้คำอธิบาย ไม่ให้ทับภาพแถวบน
    # constrained layout จัดที่ให้ title แต่ละช่อง + suptitle เอง (tight_layout ยังทับกันเมื่อ caption 2 บรรทัด)
    fig, axes = new_figure(nrows, ncols, figsize=(1.75 * ncols, row_height * nrows + 0.4), squeeze=False,
                           layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, cam, cap in zip(axes.flat, images, cams, captions, strict=False):
        if img is None:                                           # ช่องว่าง (คลาสนี้มีภาพไม่พอ)
            continue
        ax.imshow(spotlight(img, cam) if cam is not None else img)
        if cam is not None:
            ax.contour(cam, levels=[0.5], colors=SURFACE, linewidths=0.8)
        ax.set_title(cap, fontsize=7, color=INK2, loc="center", fontweight="normal")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    save(fig, name)


def load_run(run: str) -> tuple[nn.Module, dict, pd.DataFrame]:
    ckpt_path, pred_path = CHECKPOINT_DIR / f"{run}_best.pt", PRED_DIR / f"{run}_test.csv"
    if not ckpt_path.exists() or not pred_path.exists():
        raise SystemExit(f"ไม่พบ {ckpt_path.name} หรือ {pred_path.name} — เทรน + evaluate ก่อน")
    model, ckpt = load_checkpoint(ckpt_path)
    return model, ckpt, pd.read_csv(pred_path, encoding="utf-8-sig")


def correct_grid(run: str, n: int) -> None:
    """ทายถูก n ภาพต่อคลาส เรียงเป็นคอลัมน์ละคลาส"""
    model, ckpt, preds = load_run(run)
    picked = {c: preds[(preds.y_true == c) & (preds.correct == 1)].sample(frac=1, random_state=SEED).head(n)
              for c in CLASSES}
    cells = [picked[c].iloc[r] if r < len(picked[c]) else None for r in range(n) for c in CLASSES]   # row-major
    rows = [cell for cell in cells if cell is not None]
    images, cams = explain(model, ckpt, [r.filepath for r in rows], [ckpt["classes"].index(r.pred_class) for r in rows])
    it = iter(zip(images, cams, strict=True))
    images, cams = zip(*[next(it) if cell is not None else (None, None) for cell in cells], strict=True)
    captions = [(f"{cell.y_true}\n" if i < len(CLASSES) else "") + f"conf {cell.confidence:.2f}"
                if cell is not None else "" for i, cell in enumerate(cells)]
    draw_grid(images, cams, captions, len(CLASSES),
              f"Grad-CAM · {run} · correct test predictions (bright = evidence for the predicted class)",
              f"gradcam_{run}_correct")


def wrong_grid(run: str) -> None:
    """ทายผิดที่มั่นใจที่สุด — ผิดเพราะภาพกำกวม, label ผิด หรือโมเดลดูผิดที่"""
    model, ckpt, preds = load_run(run)
    rows = preds[preds.correct == 0].sort_values("confidence", ascending=False).head(N_WRONG)
    if rows.empty:
        return print(f"  {run}: ไม่มีภาพที่ทายผิด")
    images, cams = explain(model, ckpt, list(rows.filepath), [ckpt["classes"].index(c) for c in rows.pred_class])
    captions = [f"true {r.y_true}\npred {r.y_pred} ({r.confidence:.2f})" for r in rows.itertuples()]
    draw_grid(images, cams, captions, 6, f"Grad-CAM · {run} · most confident mistakes on test",
              f"gradcam_{run}_wrong")


def compare_models() -> None:
    """ภาพเดียวกัน 1 ภาพต่อคลาส × (ต้นฉบับ + 3 โมเดล seed 42) — แต่ละโมเดลอธิบายคำทายของตัวเอง"""
    runs = [(m, run_id(m, SEED)) for m in APP_MODELS if (PRED_DIR / f"{run_id(m, SEED)}_test.csv").exists()]
    if not runs:
        return print("  ข้าม compare: ไม่มี run seed 42")
    base = pd.read_csv(PRED_DIR / f"{runs[0][1]}_test.csv", encoding="utf-8-sig")
    paths = [base[base.y_true == c].sample(1, random_state=SEED).filepath.iloc[0] for c in CLASSES]

    columns = []                                                  # [(ภาพ, cam, caption)] ต่อคอลัมน์
    for m, run in runs:
        model, ckpt, preds = load_run(run)
        pred = preds.set_index("filepath").loc[paths]
        images, cams = explain(model, ckpt, paths, [ckpt["classes"].index(c) for c in pred.pred_class])
        # ข้อความธรรมดา: ฟอนต์ Segoe UI ไม่มี ✓ ✗ (ออกมาเป็นกล่องสี่เหลี่ยม)
        marks = ["" if t == p else "WRONG · " for t, p in zip(pred.y_true, pred.y_pred, strict=True)]
        columns.append([(img, cam, f"{MODEL_NAMES[m]}\n{mark}{p} ({c:.2f})")
                        for img, cam, mark, p, c in zip(images, cams, marks, pred.y_pred, pred.confidence,
                                                        strict=True)])
        del model
    originals = [(img, None, f"original\n{c}") for img, c in zip(images, CLASSES, strict=True)]
    grid = [cell for row in zip(originals, *columns, strict=True) for cell in row]   # row-major
    draw_grid(*zip(*grid, strict=True), ncols=len(runs) + 1,
              title="Grad-CAM · same test photo, three models (seed 42)", name="gradcam_compare")


def main() -> None:
    p = argparse.ArgumentParser(description="Grad-CAM บนภาพ test")
    p.add_argument("--run", help="เช่น flat_resnet50_seed43 (ไม่ใส่ = run หลัก seed 42 ของ APP_MODELS + รูปเทียบ)")
    p.add_argument("--n", type=int, default=3, help="จำนวนภาพที่ทายถูกต่อคลาส")
    args = p.parse_args()

    runs = [args.run] if args.run else [run_id(m, SEED) for m in APP_MODELS]
    for run in runs:
        print(run)
        correct_grid(run, args.n)
        wrong_grid(run)
    if not args.run:
        print("compare")
        compare_models()


if __name__ == "__main__":
    main()
