"""
evaluate.py — ประเมิน checkpoint บน test + realworld_test → reports/results.csv + ผลทำนายรายภาพ

    python -m src.evaluate                               # ทุก checkpoint ใน checkpoints/ (เขียน results.csv ใหม่)
    python -m src.evaluate --run flat_resnet50_seed42    # run เดียว (พิมพ์ผลอย่างเดียว)

ใช้ได้ทั้ง checkpoint งาน flat (ทายคลาสย่อย → สกุล = ผลรวม prob ของคลาสย่อยในสกุล) และงาน genus
ทุก split ใช้ภาพทั้งหมดของ split นั้น (รวมภาพที่ไม่รู้คลาสย่อย) → ตัวเลขชั้นสกุลเทียบกันได้ทุกงาน

ต่อ 1 run ได้
    reports/results.csv                    แถวละ run: ชั้นสกุล acc / macro F1 / MCC / coverage, ชั้นย่อย leaf acc / F1,
                                           realworld, gap, ขนาด, ความเร็ว
    reports/predictions/<run>_<split>.csv  ผลรายภาพ: y_true / y_pred (สกุล), confidence, leaf_true, pred_class
                                           (pred_class = คลาสของโมเดลที่ทาย — ใช้กับ Grad-CAM และ McNemar)
    reports/figures/cm_<run>_<split>.png   confusion matrix ชั้นสกุล (normalize ตามแถว = recall)

⚠️ ใช้ val เลือก checkpoint ไปแล้ว → ตัวเลขที่รายงานต้องมาจาก test / realworld_test เท่านั้น
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, precision_recall_fscore_support
from torch import nn

from src.config import (
    CHECKPOINT_DIR,
    CLASSES,
    CONFIDENCE_THRESHOLD,
    IMG_SIZE,
    PRED_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RESULTS_CSV,
)
from src.hierarchy import class_probs
from src.models import count_params, load_checkpoint
from src.tasks import LEAVES, genus_of
from src.visualize import plot_confusion

EVAL_SPLITS = {"test": "test", "realworld_test": "realworld"}      # split → prefix ในตาราง


def predict_split(model: nn.Module, ckpt: dict, split: str) -> pd.DataFrame:
    """ทายทุกภาพของ split → ตารางรายภาพ (ว่าง = split นี้ไม่มีภาพ)"""
    files = sorted((PROCESSED_DIR / split).glob("*/*.jpg"))
    if not files:
        return pd.DataFrame()
    leaf_true = {f.name: f.parent.name for f in (PROCESSED_DIR / "flat" / split).glob("*/*.jpg")}
    classes = ckpt["classes"]
    probs = class_probs(model, ckpt, files)
    genus_probs = probs @ np.array([[genus_of(c) == g for c in classes] for g in CLASSES], dtype=float).T
    df = pd.DataFrame({
        "filepath": [f.relative_to(PROJECT_ROOT).as_posix() for f in files],
        "y_true": [f.parent.name for f in files],
        "y_pred": [CLASSES[i] for i in genus_probs.argmax(1)],
        "confidence": genus_probs.max(1).round(4),
        "leaf_true": [leaf_true.get(f.name, "") for f in files],      # ว่าง = ไม่รู้คลาสย่อย
        "pred_class": [classes[i] for i in probs.argmax(1)],
    })
    df["correct"] = (df.y_true == df.y_pred).astype(int)
    return df


def metrics(df: pd.DataFrame, flat: bool) -> dict:
    p, rc, f1, _ = precision_recall_fscore_support(df.y_true, df.y_pred, labels=CLASSES, average="macro",
                                                   zero_division=0)
    kept = df.confidence >= CONFIDENCE_THRESHOLD          # ภาพที่แอปจะ "ตอบ" (ที่เหลือตอบว่าไม่แน่ใจ)
    row = {
        "acc": df.correct.mean(), "macro_p": p, "macro_r": rc, "macro_f1": f1,
        "mcc": matthews_corrcoef(df.y_true, df.y_pred),
        "coverage": kept.mean(),                                          # สัดส่วนที่กล้าตอบ
        "acc_confident": df.correct[kept].mean() if kept.any() else float("nan"),
    }
    if flat:                                               # ชั้นย่อย: เฉพาะภาพที่รู้คลาสย่อย
        known = df[df.leaf_true != ""]
        row["leaf_acc"] = (known.pred_class == known.leaf_true).mean()
        row["leaf_f1"] = f1_score(known.leaf_true, known.pred_class, labels=LEAVES, average="macro", zero_division=0)
    return row


def save_predictions(df: pd.DataFrame, stem: str) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRED_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    report = classification_report(df.y_true, df.y_pred, labels=CLASSES, digits=4, zero_division=0)
    (PRED_DIR / f"{stem}_report.txt").write_text(report, encoding="utf-8")


@torch.no_grad()                        # ตรงกับการใช้งานจริง (predict/app) — ไม่เก็บ graph
def inference_ms(model: nn.Module, device: str, n: int = 50) -> float:
    """เวลาทำนาย 1 ภาพ (batch 1, fp32) — ค่า CPU ใกล้เคียงการใช้งานบนเครื่องทั่วไป/มือถือที่สุด"""
    model = model.to(device)
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    for _ in range(5):                 # warm-up
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def evaluate_run(ckpt_path: Path) -> dict:
    run = ckpt_path.stem.removesuffix("_best")
    model, ckpt = load_checkpoint(ckpt_path)
    flat = ckpt["task"] == "flat"
    row = {"run": run, "task": ckpt["task"], "model": ckpt["model_key"], "seed": ckpt["seed"],
           "pretrained": ckpt["pretrained"], "best_stage": ckpt["stage"], "best_epoch": ckpt["epoch"],
           "val_f1": ckpt["val_f1"]}
    for split, prefix in EVAL_SPLITS.items():
        df = predict_split(model, ckpt, split)
        if df.empty:
            continue
        save_predictions(df, f"{run}_{split}")
        plot_confusion(df.y_true.map(CLASSES.index).to_numpy(), df.y_pred.map(CLASSES.index).to_numpy(),
                       f"{run} · {split}", f"cm_{run}_{split}")
        row |= {f"{prefix}_{k}": v for k, v in metrics(df, flat).items()}
    if "realworld_acc" in row:
        row["gap"] = row["test_acc"] - row["realworld_acc"]   # จุดขายของรายงาน: ยิ่งน้อยยิ่ง generalize ดี

    row["params_m"] = count_params(model)
    row["size_mb"] = sum(t.numel() * t.element_size() for t in ckpt["model"].values()) / 1e6
    row["infer_ms_gpu"] = inference_ms(model, "cuda") if torch.cuda.is_available() else float("nan")
    row["infer_ms_cpu"] = inference_ms(model, "cpu")
    del model
    torch.cuda.empty_cache()
    return row


def write_results(rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))    # รวมคอลัมน์ทุก run ตามลำดับที่เจอ
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)


def main() -> None:
    p = argparse.ArgumentParser(description="ประเมินโมเดลบน test / realworld_test")
    p.add_argument("--run", help="ชื่อ run เช่น flat_resnet50_seed42 (ไม่ใส่ = ทุก checkpoint)")
    args = p.parse_args()

    paths = [CHECKPOINT_DIR / f"{args.run}_best.pt"] if args.run else sorted(CHECKPOINT_DIR.glob("*_best.pt"))
    if not paths or not paths[0].exists():
        raise SystemExit(f"ไม่พบ checkpoint ใน {CHECKPOINT_DIR}")

    rows = []
    for path in paths:
        row = evaluate_run(path)
        rows.append(row)
        leaf = f"  leaf {row['test_leaf_acc']:.4f}" if "test_leaf_acc" in row else ""
        rw = f"  realworld {row['realworld_acc']:.4f}  gap {row['gap']:+.4f}" if "gap" in row else ""
        print(f"{row['run']:<30} test acc {row['test_acc']:.4f}  F1 {row['test_macro_f1']:.4f}{leaf}{rw}  "
              f"| {row['params_m']:.1f}M  {row['infer_ms_cpu']:.1f}ms CPU")
    if not args.run:                   # ครบทุก run → เขียนตารางใหม่ทั้งไฟล์
        write_results(rows)
        print(f"\n→ {RESULTS_CSV}")


if __name__ == "__main__":
    main()
