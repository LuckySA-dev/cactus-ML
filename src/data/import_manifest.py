"""
import_manifest.py — นำภาพจากโฟลเดอร์ใด ๆ เข้าโปรเจคตาม manifest ที่ตรวจแล้ว (label / subclass / ต้น / QC รายภาพ)

    python -m src.data.import_manifest data/manifests/field_2026-10.csv
    python -m src.data.import_manifest data/manifests/field_2026-10.csv --dry-run

ใช้กับภาพที่ทีมรวบรวมมาเป็นโฟลเดอร์ตามชื่อไทย/ชื่อสปีชีส์ ซึ่งชื่อโฟลเดอร์ไม่ตรงกับ label เสมอไป
(เช่นในโฟลเดอร์ asterias มีหมวกแก๊ปปน) — จึงให้คนตรวจทีละภาพลง manifest ก่อน แล้วสคริปต์นี้ทำตามนั้น

คอลัมน์ manifest
    src        path ภาพต้นฉบับ (สัมพัทธ์จาก project root) — ไม่แก้ไฟล์ต้นฉบับ
    source     handon (ถ่ายเอง → ครึ่งหนึ่งเข้า realworld_test) / web (ภาพจากเว็บ → เหมือน iNat)
    label      สกุล (config.CLASSES)          subclass   คลาสชั้น 2 (ว่าง = ระบุไม่ได้)
    plant      รหัสต้น (handon: หลายมุมของต้นเดียวกันต้องรหัสเดียวกัน · ว่าง = ภาพละต้น)
    qc_status  ok / not_potted / wrong_label / not_plant / multi_plant / blurry

ทุกภาพถูกแปลงเป็น JPEG (หมุนตาม EXIF แล้ว) ไว้ที่ data/raw/<source>/<label>/ → HEIC/WEBP/AVIF ไม่ต้องมี
decoder ตอนเทรน · ภาพซ้ำ (phash) กับภาพเดิมในโปรเจคถูกข้าม · รันซ้ำได้ (ภาพที่นำเข้าแล้วถูกข้าม)
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import imagehash
import pillow_heif
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from src.config import CLASSES, DUP_HAMMING_DISTANCE, MIN_IMAGE_SIZE, PROJECT_ROOT, RAW_DIR, SUBCLASSES
from src.data.metadata import MetadataStore, make_row

pillow_heif.register_heif_opener()      # ภาพจาก iPhone (.HEIC)
QC_CHOICES = {"ok", "not_potted", "wrong_label", "not_plant", "multi_plant", "blurry"}
SOURCE_CHOICES = {"handon", "web"}


def validate(rows: list[dict]) -> None:
    """ตรวจ manifest ทั้งไฟล์ก่อนเริ่ม — พังกลางทางแล้ว metadata ครึ่ง ๆ แก้ยากกว่า"""
    errors = []
    for n, r in enumerate(rows, 2):                     # บรรทัดในไฟล์ (บรรทัด 1 = header)
        if r["source"] not in SOURCE_CHOICES:
            errors.append(f"บรรทัด {n}: source '{r['source']}'")
        if r["label"] not in CLASSES:
            errors.append(f"บรรทัด {n}: label '{r['label']}'")
        elif r["subclass"] and r["subclass"] not in SUBCLASSES.get(r["label"], []):
            errors.append(f"บรรทัด {n}: subclass '{r['subclass']}' ไม่มีใน {r['label']}")
        if r["qc_status"] not in QC_CHOICES:
            errors.append(f"บรรทัด {n}: qc_status '{r['qc_status']}'")
        if not (PROJECT_ROOT / r["src"]).exists():
            errors.append(f"บรรทัด {n}: ไม่พบไฟล์ {r['src']}")
    if errors:
        raise SystemExit("manifest ไม่ถูกต้อง:\n  " + "\n  ".join(errors[:30]))


def image_id_of(r: dict) -> str:
    """ไม่ซ้ำข้ามโฟลเดอร์ต้นทาง (IMG_0001 มีได้หลายโฟลเดอร์) และคงที่ทุกครั้งที่รัน"""
    src = Path(r["src"])
    tag = "_".join(src.parent.parts[1:3])          # Cactus_1/Astrophytum
    return f"{r['source']}_{tag}_{src.stem}".replace(" ", "")


def import_row(store: MetadataStore, r: dict, dry_run: bool) -> str:
    """นำเข้า 1 ภาพ → สถานะสำหรับสรุป"""
    image_id = image_id_of(r)
    if store.has_id(image_id):
        return "already"
    try:
        img = ImageOps.exif_transpose(Image.open(PROJECT_ROOT / r["src"])).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return "broken"
    if min(img.size) < MIN_IMAGE_SIZE:
        return "too_small"
    ph = imagehash.phash(img)
    if store.find_duplicate(ph, DUP_HAMMING_DISTANCE):
        return "duplicate"

    dst = RAW_DIR / r["source"] / r["label"] / f"{image_id}.jpg"
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, quality=95)
    store.add(make_row(
        image_id=image_id, filepath=dst.relative_to(PROJECT_ROOT).as_posix(),
        label=r["label"], subclass=r["subclass"], source=r["source"],
        plant_id=f"{r['source']}_{r['plant']}" if r["plant"] else "",
        phash=str(ph), width=img.width, height=img.height,
        qc_status=r["qc_status"], notes=f"from {r['src']}",
    ))
    return "added"


def main() -> None:
    p = argparse.ArgumentParser(description="นำภาพเข้าโปรเจคตาม manifest ที่ตรวจแล้ว")
    p.add_argument("manifest", type=Path)
    p.add_argument("--dry-run", action="store_true", help="ตรวจ + นับอย่างเดียว ไม่เขียนไฟล์")
    args = p.parse_args()

    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    validate(rows)

    store = MetadataStore()
    result = Counter()
    for r in tqdm(rows, desc="import", unit="img", ncols=90):
        result[(r["source"], import_row(store, r, args.dry_run))] += 1
    if not args.dry_run:
        store.save()
    print("\n".join(f"  {src:<7} {status:<10} {n:>5}" for (src, status), n in sorted(result.items())))


if __name__ == "__main__":
    main()
