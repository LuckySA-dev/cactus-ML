"""
flag_suspects.py — ให้โมเดลช่วยคัดภาพน่าสงสัย (label ผิด / ภาพใช้ไม่ได้) แทนการไล่ดูทุกภาพ

    python -m src.data.flag_suspects            # k-fold → reports/qc/suspects.{csv,html} + oof_predictions.csv
    python -m src.data.flag_suspects --smoke    # 2 fold × 1+1 epoch เช็คว่าไม่พัง

วิธี (out-of-fold prediction — แนวคิดเดียวกับ confident learning)
- แบ่งภาพทั้งหมด k ส่วน (stratified + ต้นเดียวกันอยู่ส่วนเดียวกัน) → เทรนด้วย k-1 ส่วน → ทำนายส่วนที่เหลือ
- ทุกภาพถูกทำนายโดยโมเดลที่ไม่เคยเห็นภาพนั้น
- flag ถ้าทายไม่ตรง label หรือ p(label) < SUSPECT_PROB → คนดูเฉพาะกลุ่มนี้ แล้ว
      python -m src.data.qc_report --apply reports/qc/suspects.csv
- schedule ตายตัว ไม่มี early stopping: ถ้าใช้ fold ที่ทำนายไปเลือก epoch = โกงตัวเอง
"""
from __future__ import annotations

import argparse
import csv
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import (
    CLASSES,
    DEVICE,
    LABEL_SMOOTHING,
    QC_DIR,
    QC_USABLE,
    SEED,
    STAGE1,
    STAGE2,
    SUSPECT_CV,
    SUSPECT_PROB,
    WEIGHT_DECAY,
)
from src.data.metadata import MetadataStore
from src.data.qc_report import SHEET_COLS, read_sheet, write_html
from src.data.split import materialize, plant_of
from src.datasets import build_loaders
from src.engine import evaluate, set_seed, train_one_epoch
from src.models import build_model, data_config, head_params, param_groups, set_backbone_trainable

OOF_COLS = ["image_id", "label", "pred", "p_label", "p_pred", "fold", "qc_status", "flagged"]


def assign_folds(rows: list[dict], k: int, seed: int) -> np.ndarray:
    """fold ของแต่ละแถว — stratified ตาม label และต้นเดียวกันอยู่ fold เดียวกัน"""
    y = [CLASSES.index(r["label"]) for r in rows]
    groups = [plant_of(r) for r in rows]
    folds = np.empty(len(rows), dtype=int)
    for f, (_, test_idx) in enumerate(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(rows, y, groups)):
        folds[test_idx] = f
    return folds


def train_fixed(root: Path, epochs1: int, epochs2: int, seed: int) -> tuple[nn.Module, dict]:
    """เทรน 2 stage แบบ schedule ตายตัว (ไม่มี val) บน <root>/train"""
    model = build_model(SUSPECT_CV["model"]).to(DEVICE, memory_format=torch.channels_last)
    cfg = data_config(model)
    loader = build_loaders(cfg["mean"], cfg["std"], splits=["train"], seed=seed, root=root)["train"]
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    set_backbone_trainable(model, False)
    opt = AdamW(head_params(model), lr=STAGE1["lr"], weight_decay=WEIGHT_DECAY)
    for _ in range(epochs1):
        train_one_epoch(model, loader, criterion, opt, frozen_backbone=True)

    set_backbone_trainable(model, True)
    opt = AdamW(param_groups(model, STAGE2["lr_backbone"], STAGE2["lr_head"]), weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=epochs2)
    for _ in range(epochs2):
        train_one_epoch(model, loader, criterion, opt, frozen_backbone=False)
        sched.step()
    return model, cfg


def out_of_fold(rows: list[dict], folds: np.ndarray, epochs1: int, epochs2: int, seed: int) -> list[dict]:
    """ทำนายทุกภาพด้วยโมเดลที่ไม่เคยเห็นมัน → แถวละ 1 ผล"""
    by_id = {r["image_id"]: r for r in rows}
    oof = []
    for f in range(folds.max() + 1):
        with tempfile.TemporaryDirectory(dir=QC_DIR) as tmp:   # อยู่ไดรฟ์เดียวกับ data → hardlink ได้
            root = Path(tmp)
            materialize([r | {"split": "test" if fold == f else "train"} for r, fold in zip(rows, folds, strict=True)],
                        root)
            model, cfg = train_fixed(root, epochs1, epochs2, seed)
            loader = build_loaders(cfg["mean"], cfg["std"], splits=["test"], num_workers=0, root=root)["test"]
            r = evaluate(model, loader, nn.CrossEntropyLoss())
            for (path, _), t, probs in zip(loader.dataset.samples, r.y_true, r.probs, strict=True):
                row = by_id[Path(path).stem]
                pred = int(probs.argmax())
                oof.append({"image_id": row["image_id"], "label": CLASSES[t], "pred": CLASSES[pred],
                            "p_label": round(float(probs[t]), 4), "p_pred": round(float(probs[pred]), 4),
                            "fold": f, "qc_status": row["qc_status"],
                            "flagged": int(pred != t or probs[t] < SUSPECT_PROB)})
            print(f"  fold {f}: acc {r.acc:.4f}  flagged {sum(o['flagged'] for o in oof if o['fold'] == f)}")
            del model, loader
    return oof


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def report(oof: list[dict]) -> None:
    """ความแม่นยำ OOF + อัตรา flag ต่อคลาส + วิธีนี้จับภาพเสียที่ QC สุ่มเจอได้กี่ %"""
    n, acc = len(oof), sum(o["label"] == o["pred"] for o in oof) / len(oof)
    print(f"\nOOF acc {acc:.4f}  |  flagged {sum(o['flagged'] for o in oof)}/{n}")
    for label in CLASSES:
        sub = [o for o in oof if o["label"] == label]
        print(f"  {label:<15} flagged {sum(o['flagged'] for o in sub):>4}/{len(sub)}")
    known_bad = [o for o in oof if o["qc_status"] not in QC_USABLE]
    if known_bad:
        caught = sum(o["flagged"] for o in known_bad)
        print(f"ภาพเสียที่ QC สุ่มเจอแล้ว: flag ได้ {caught}/{len(known_bad)} "
              f"(recall {caught / len(known_bad):.0%}) — ใช้วัดว่าวิธีนี้ได้ผลแค่ไหน")


def main() -> None:
    p = argparse.ArgumentParser(description="คัดภาพน่าสงสัยด้วย out-of-fold prediction")
    p.add_argument("--smoke", action="store_true", help="2 fold × 1+1 epoch")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    suspects_csv = QC_DIR / "suspects.csv"
    if suspects_csv.exists() and any(r["qc_status"].strip() for r in read_sheet(suspects_csv)):
        raise SystemExit(f"{suspects_csv} มีผลตรวจอยู่แล้ว — --apply ก่อน แล้วย้ายไฟล์ออกค่อยรันใหม่")

    set_seed(args.seed)
    store = MetadataStore()
    # ทำนายทุกภาพ รวมภาพเสียที่ QC เจอแล้ว → วัดได้ว่าวิธีนี้จับภาพเสียได้กี่ % (ส่งให้คนดูเฉพาะที่ยังไม่ตรวจ)
    rows = [r for r in store.rows if r["label"] in CLASSES]
    k, e1, e2 = (2, 1, 1) if args.smoke else (SUSPECT_CV["folds"], SUSPECT_CV["epochs1"], SUSPECT_CV["epochs2"])
    print(f"{len(rows)} ภาพ, {k} folds, {SUSPECT_CV['model']} {e1}+{e2} epochs")

    QC_DIR.mkdir(parents=True, exist_ok=True)
    oof = out_of_fold(rows, assign_folds(rows, k, args.seed), e1, e2, args.seed)
    write_csv(QC_DIR / "oof_predictions.csv", oof, OOF_COLS)
    report(oof)

    # ภาพที่ต้องให้คนดู = ถูก flag และยังไม่เคยตรวจ — เรียงจากโมเดลไม่เชื่อ label มากสุดก่อน
    by_id = {r["image_id"]: r for r in rows}
    todo = sorted((o for o in oof if o["flagged"] and o["qc_status"] in ("", "pending")),
                  key=lambda o: (o["label"], o["p_label"]))
    sheet = [by_id[o["image_id"]] | {"qc_status": "", "notes": f"model: {o['pred']} {o['p_pred']:.2f}"}
             for o in todo]
    write_csv(suspects_csv, sheet, SHEET_COLS)
    write_html(sheet, suspects_csv.with_suffix(".html"))
    print(f"\nให้คนตรวจ {len(sheet)} ภาพ ({Counter(o['label'] for o in todo)}) → {suspects_csv}")


if __name__ == "__main__":
    main()
