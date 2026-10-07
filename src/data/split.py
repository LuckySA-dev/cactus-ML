"""
split.py — แบ่ง train / val / test (70/15/15) + realworld_test แบบ stratified และแยกตาม plant_id

    python -m src.data.split --dry-run   # ดูตัวเลขอย่างเดียว ไม่เขียนอะไร
    python -m src.data.split             # เขียนคอลัมน์ split ลง metadata.csv + สร้าง data/processed/

กฎที่ห้ามผิด
- ภาพของต้นเดียวกัน (plant_id เดียวกัน) อยู่ split เดียวกันเสมอ → กัน leakage
- แบ่งทีละคลาส → ทุก split มีสัดส่วนคลาสเท่ากัน (stratified)
- ภาพถ่ายเอง (handon) ครึ่งหนึ่ง (นับเป็นต้น) → realworld_test ที่โมเดลห้ามเห็นเด็ดขาด
- ใช้เฉพาะ label ใน config.CLASSES และ qc_status ใน config.QC_USABLE

ไม่มีขั้น data/interim/ — ภาพซ้ำถูกตัดตั้งแต่ตอนลงทะเบียน และผล QC อยู่ใน metadata.csv แล้ว
จึงสร้าง data/processed/ จาก data/raw/ โดยตรง (hardlink ไม่กินที่ดิสก์เพิ่ม)
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from src.config import CLASSES, PROCESSED_DIR, PROJECT_ROOT, QC_USABLE, REALWORLD_FRAC, SEED, SPLIT_RATIO
from src.data.metadata import MetadataStore
from src.tasks import LEAF_SEP, TASKS, get_task, leaf_of

SPLITS = ["train", "val", "test", "realworld_test"]


def plant_of(row: dict) -> str:
    # ภาพที่ไม่มี plant_id ถือว่าเป็นต้นของตัวเอง
    return row["plant_id"] or row["image_id"]


def assign_plants(plants: dict[str, int], rng: random.Random) -> dict[str, str]:
    """
    plants = {plant_id: จำนวนภาพ} → {plant_id: split}
    สุ่มลำดับต้น แล้วแจกทั้งต้นให้ val/test จนครบเป้า (นับเป็นภาพ) ที่เหลือเป็น train
    """
    ids = sorted(plants)          # sort ก่อน shuffle → seed เดิมได้ผลเดิมเสมอ
    rng.shuffle(ids)
    total = sum(plants.values())
    filled: Counter = Counter()
    out = {}
    for pid in ids:
        split = next((s for s in ("val", "test") if filled[s] < SPLIT_RATIO[s] * total), "train")
        out[pid] = split
        filled[split] += plants[pid]
    return out


def make_splits(store: MetadataStore, seed: int) -> None:
    rng = random.Random(seed)
    for row in store.rows:
        row["split"] = ""         # คลาสที่ถูกตัด / ภาพที่ QC ไม่ผ่าน → ไม่อยู่ split ไหนเลย

    # stratify ทีละ "กลุ่ม" = คลาสย่อยสุด (leaf) ไม่ใช่แค่สกุล → ทุกคลาสย่อยมีภาพใน val/test ด้วย
    # ภาพที่ไม่รู้คลาสย่อย (iNat ระบุแค่สกุล) เป็นกลุ่มของตัวเองต่อสกุล — ใช้ได้เฉพาะงาน genus
    usable = [r for r in store.rows if r["label"] in CLASSES and r["qc_status"] in QC_USABLE]
    groups = sorted({leaf_of(r) or f"{r['label']}{LEAF_SEP}?" for r in usable})
    for group in groups:
        rows = [r for r in usable if (leaf_of(r) or f"{r['label']}{LEAF_SEP}?") == group]
        handon = sorted({plant_of(r) for r in rows if r["source"] == "handon"})
        rng.shuffle(handon)
        assign = {p: "realworld_test" for p in handon[: round(len(handon) * REALWORLD_FRAC)]}
        pool = Counter(plant_of(r) for r in rows if plant_of(r) not in assign)
        assign |= assign_plants(pool, rng)
        for r in rows:
            r["split"] = assign[plant_of(r)]

    # guard: ต้นเดียวต้องไม่อยู่หลาย split (ถ้าพัง = leakage → accuracy สูงเกินจริง)
    seen: dict[str, str] = {}
    for r in store.rows:
        if r["split"] and seen.setdefault(plant_of(r), r["split"]) != r["split"]:
            raise AssertionError(f"leakage: plant {plant_of(r)} อยู่ทั้ง {seen[plant_of(r)]} และ {r['split']}")


def build_processed(store: MetadataStore) -> None:
    """
    สร้างโฟลเดอร์ของทุกงาน (src/tasks.py) ใหม่ทั้งหมด — เป็นข้อมูลที่ derive ได้
    split ของภาพเดียวกันเหมือนกันทุกงาน → test ของชั้น 1 และชั้น 2 เป็นภาพชุดเดียวกัน เทียบกันได้
    """
    for s in SPLITS:
        shutil.rmtree(PROCESSED_DIR / s, ignore_errors=True)
    for sub in ("species", "flat"):
        shutil.rmtree(PROCESSED_DIR / sub, ignore_errors=True)
    for name in TASKS:
        task = get_task(name)
        materialize(store.rows, task.root, task.class_of)


def materialize(rows: list[dict], root: Path, class_of=lambda r: r["label"]) -> None:
    """<root>/<split>/<class>/<image_id>.jpg จากแถวที่มี split — hardlink ไม่กินที่ดิสก์เพิ่ม"""
    for r in rows:
        cls = class_of(r)
        if not r["split"] or not cls:
            continue
        src = PROJECT_ROOT / r["filepath"]
        dst = root / r["split"] / cls / f"{r['image_id']}{src.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.hardlink_to(src)
        except OSError:           # คนละไดรฟ์ / filesystem ไม่รองรับ
            shutil.copy2(src, dst)


def summary(store: MetadataStore) -> str:
    c = Counter((r["label"], r["split"]) for r in store.rows if r["split"])
    lines = [f"{'label':<15}" + "".join(f"{s:>16}" for s in SPLITS)]
    for label in CLASSES:
        lines.append(f"{label:<15}" + "".join(f"{c[(label, s)]:>16}" for s in SPLITS))
    lines.append(f"{'TOTAL':<15}" + "".join(f"{sum(c[(label, s)] for label in CLASSES):>16}" for s in SPLITS))
    excluded = sum(1 for r in store.rows if r["label"] in CLASSES and not r["split"])
    lines.append(f"\nไม่ถูกใช้ (QC ไม่ผ่าน): {excluded} ภาพ")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="แบ่ง train/val/test/realworld_test ตาม plant_id")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--dry-run", action="store_true", help="ดูตัวเลขอย่างเดียว")
    args = p.parse_args()

    store = MetadataStore()
    make_splits(store, args.seed)
    print(summary(store))
    if args.dry_run:
        return
    store.save()
    build_processed(store)
    print(f"\nสร้าง {PROCESSED_DIR} แล้ว")


if __name__ == "__main__":
    main()
