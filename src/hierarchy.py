"""
hierarchy.py — ทาย 2 ชั้น (สกุล → คลาสย่อย) + เปรียบเทียบ 2 โครงสร้างบน test ชุดเดียวกัน

    python -m src.hierarchy --model effnetv2b0 --seed 42    → reports/hierarchy_<model>_seed<N>.csv

    A · แยกชั้น   โมเดลสกุล → ถ้าสกุลนั้นมีชั้น 2 ส่งต่อโมเดลของสกุลนั้น
                  (ทายสกุลผิด = คลาสย่อยผิดแน่นอน — error ส่งต่อ)
    B · แบนราบ    โมเดลเดียวทายทุกคลาสย่อย (leaf) แล้วอนุมานสกุล = ผลรวม prob ของ leaf ในสกุลนั้น
                  (ลังเลระหว่าง 2 สปีชีส์ในสกุลเดียวกัน สกุลยังถูก)

วัดบน test ชุดเดียวกัน (split เหมือนกันทุกงาน):
    genus_acc            ภาพ test ทั้งหมด (รวมภาพที่ไม่รู้คลาสย่อย)
    leaf_acc             ภาพที่รู้คลาสย่อย — ถูกทั้งสกุลและคลาสย่อย
    sub_acc_given_genus  ความแม่นชั้น 2 เฉพาะภาพที่สกุลถูกแล้ว (แยก error ของชั้น 2 ออกจากชั้น 1)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.config import CHECKPOINT_DIR, CLASSES, DEVICE, PROCESSED_DIR, REPORT_DIR, SUBCLASSES
from src.datasets import build_transforms, load_image
from src.models import load_checkpoint
from src.tasks import LEAF_SEP, LEAVES, genus_of


@torch.no_grad()
def class_probs(model: torch.nn.Module, ckpt: dict, files: list[Path]) -> np.ndarray:
    """softmax fp32 [N, คลาสของ checkpoint] บนไฟล์ที่กำหนด — eval transform เดียวกับ predict/แอป
    ทายจากไฟล์ตรง ๆ เพราะชั้น 2 ต้องทายภาพที่ชั้น 1 ส่งมา ซึ่งอาจไม่อยู่ในโฟลเดอร์งานของตัวเอง"""
    _, tf = build_transforms(ckpt["mean"], ckpt["std"])
    device = next(model.parameters()).device
    out = [model(torch.stack([tf(load_image(f)) for f in files[i:i + 64]]).to(device)).float().softmax(1).cpu()
           for i in range(0, len(files), 64)]
    return torch.cat(out).numpy()


def predict_probs(run: str, files: list[Path]) -> np.ndarray:
    """class_probs ของ checkpoint ชื่อ run (โหลดแล้วคืน GPU ทันที)"""
    model, ckpt = load_checkpoint(CHECKPOINT_DIR / f"{run}_best.pt", DEVICE)
    probs = class_probs(model, ckpt, files)
    del model
    torch.cuda.empty_cache()
    return probs


def load_test(split: str = "test") -> tuple[list[Path], np.ndarray, np.ndarray]:
    """(ไฟล์ของ split, index สกุล, index leaf — -1 = ไม่รู้คลาสย่อย) จากโฟลเดอร์ของงาน genus / flat"""
    files = sorted((PROCESSED_DIR / split).glob("*/*.jpg"))
    leaf_of_name = {f.name: LEAVES.index(f.parent.name) for f in (PROCESSED_DIR / "flat" / split).glob("*/*.jpg")}
    genus = np.array([CLASSES.index(f.parent.name) for f in files])
    leaf = np.array([leaf_of_name.get(f.name, -1) for f in files])
    return files, genus, leaf


def predict_a(files: list[Path], model: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(genus_pred, leaf_pred) แบบแยกชั้น"""
    genus_pred = predict_probs(f"{model}_seed{seed}", files).argmax(1)
    leaf_pred = np.array([LEAVES.index(CLASSES[g]) if CLASSES[g] in LEAVES else -1 for g in genus_pred])
    for genus, subs in SUBCLASSES.items():
        routed = np.flatnonzero(genus_pred == CLASSES.index(genus))       # ภาพที่ชั้น 1 ส่งมาสกุลนี้
        if len(routed):
            sub = predict_probs(f"species-{genus}_{model}_seed{seed}", [files[i] for i in routed]).argmax(1)
            leaf_pred[routed] = [LEAVES.index(f"{genus}{LEAF_SEP}{subs[k]}") for k in sub]
    return genus_pred, leaf_pred


def predict_b(files: list[Path], model: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(genus_pred, leaf_pred) แบบแบนราบ"""
    leaf_probs = predict_probs(f"flat_{model}_seed{seed}", files)
    member = np.array([[genus_of(leaf) == g for leaf in LEAVES] for g in CLASSES], dtype=float)   # [G, L]
    return (leaf_probs @ member.T).argmax(1), leaf_probs.argmax(1)


def score(genus: np.ndarray, leaf: np.ndarray, genus_pred: np.ndarray, leaf_pred: np.ndarray) -> dict:
    known = leaf >= 0
    genus_ok = genus_pred == genus
    leaf_ok = leaf_pred[known] == leaf[known]
    row = {
        "genus_acc": genus_ok.mean(),
        "genus_macro_f1": f1_score(genus, genus_pred, average="macro"),
        "leaf_acc": leaf_ok.mean(),
        "leaf_macro_f1": f1_score(leaf[known], leaf_pred[known], average="macro",
                                  labels=range(len(LEAVES)), zero_division=0),
        "sub_acc_given_genus": leaf_ok[genus_ok[known]].mean(),
    }
    for g in SUBCLASSES:                         # ความแม่นชั้น 2 รายสกุล
        in_genus = np.array([genus_of(LEAVES[i]) == g for i in leaf[known]])
        row[f"leaf_acc_{g}"] = leaf_ok[in_genus].mean()
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="เทียบ A (แยกชั้น) vs B (แบนราบ) บน test ชุดเดียวกัน")
    p.add_argument("--model", default="effnetv2b0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    files, genus, leaf = load_test()
    print(f"test {len(files)} ภาพ (รู้คลาสย่อย {(leaf >= 0).sum()})")
    rows = []
    for name, fn in (("A_hierarchical", predict_a), ("B_flat", predict_b)):
        row = {"approach": name, "model": args.model, "seed": args.seed} | score(genus, leaf, *fn(files, args.model,
                                                                                              args.seed))
        rows.append(row)
        print(f"{name:<15} " + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if isinstance(v, float)))
    out = REPORT_DIR / f"hierarchy_{args.model}_seed{args.seed}.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
