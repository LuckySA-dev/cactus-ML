"""
register_local.py — ลงทะเบียนภาพจาก Roboflow และภาพที่ถ่ายเอง เข้า metadata.csv

ภาพจาก iNaturalist ถูกลงทะเบียนอัตโนมัติโดย download_inat.py แล้ว
ส่วนภาพที่เราเอามาเองต้อง copy เข้าโฟลเดอร์ก่อน แล้วรันสคริปต์นี้

การใช้งาน
---------
    python -m src.data.register_local --source roboflow
    python -m src.data.register_local --source handon
    python -m src.data.register_local --source all

⚠️ Roboflow หลาย dataset license ต่างกัน: copy เข้าทีละ dataset แล้วรันทีละรอบ
   (ภาพที่ลงทะเบียนแล้วจะถูกข้าม license ของแต่ละ dataset จึงไม่ปนกัน)

    python -m src.data.register_local --source roboflow --license "CC BY 4.0" \\
        --source-url https://universe.roboflow.com/... --attribution "<workspace>"


📸 วิธีจัดไฟล์ภาพที่ถ่ายเอง (สำคัญมาก — เกี่ยวกับ plant_id)
------------------------------------------------------------
เพราะเราถ่าย "หลายมุมต่อ 1 ต้น" ภาพของต้นเดียวกันจึงคล้ายกันมาก
ถ้าปล่อยให้ภาพต้นเดียวกันกระจายไปทั้ง train และ test → โมเดลเหมือนได้ดูข้อสอบ
ล่วงหน้า (data leakage) ค่า accuracy จะสูงเกินจริง

ทางแก้คือ **แยกด้วย plant_id** แล้วตอน split ใช้ GroupShuffleSplit
ให้ทุกภาพของต้นเดียวกันอยู่ชุดเดียวกันเสมอ

จัดโฟลเดอร์แบบนี้ (แนะนำ — ง่ายที่สุดเวลาอยู่หน้างาน):

    data/raw/handon/astrophytum/
        ├── plant001/          ← 1 โฟลเดอร์ = 1 ต้น
        │   ├── IMG_0001.jpg   ← มุมบน
        │   ├── IMG_0002.jpg   ← มุมข้าง
        │   └── IMG_0003.jpg   ← มุมเฉียง
        ├── plant002/
        └── plant003/

หรือถ้าไม่อยากสร้างโฟลเดอร์ย่อย ให้ตั้งชื่อไฟล์แบบนี้:

    data/raw/handon/astrophytum/plant001__01.jpg
    data/raw/handon/astrophytum/plant001__02.jpg

สคริปต์อ่าน plant_id จากชื่อโฟลเดอร์ย่อยก่อน ถ้าไม่มีจึงดูจากชื่อไฟล์
ถ้าไม่เข้าทั้งสองแบบ จะถือว่าภาพนั้นเป็นคนละต้นกันทั้งหมด (ปลอดภัยไว้ก่อน)


🌵 เคล็ดลับตอนถ่าย (โจทย์คือกระบองเพชรแคระ)
--------------------------------------------
- ถ่ายต้นที่ยังเล็ก/แคระจริง ๆ อย่าถ่ายต้นใหญ่โตเต็มวัย
- ต่อ 1 ต้นถ่าย 5-8 มุม: บนตรง ๆ, เฉียง 45°, ด้านข้างระดับสายตา, ใกล้เห็นหนาม
- เปลี่ยนพื้นหลังบ้าง (โต๊ะไม้ / พื้นปูน / ผ้าขาว) ไม่งั้นโมเดลจะจำพื้นหลังแทนต้นไม้
- ถ่ายทั้งในร่มและกลางแจ้ง
- ใส่กระถางทั้งใบในเฟรมบ้าง ครอปเฉพาะต้นบ้าง
- จดชื่อสกุลตอนถ่ายทันที อย่าไว้มาเดาทีหลัง
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.config import (
    CLASSES,
    DUP_HAMMING_DISTANCE,
    METADATA_CSV,
    MIN_IMAGE_SIZE,
    PROJECT_ROOT,
    RAW_DIR,
    ensure_dirs,
)
from src.data.metadata import MetadataStore, make_row

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def derive_plant_id(image_path: Path, genus_dir: Path, source: str) -> str:
    """
    หา plant_id ของภาพ

    1) อยู่ในโฟลเดอร์ย่อย  -> ใช้ชื่อโฟลเดอร์ย่อย  (astrophytum/plant001/x.jpg)
    2) ชื่อไฟล์มี '__'     -> ใช้ส่วนหน้า          (plant001__02.jpg)
    3) ไม่เข้าทั้งสองแบบ   -> ถือว่าเป็นคนละต้น     (ปลอดภัยไว้ก่อน)

    Roboflow export ภาพ augment (หมุน/พลิก) เป็น IMG_1_jpg.rf.<hash>.jpg
    phash จับไม่ได้ว่าซ้ำ → ตัด .rf.<hash> ออก ให้ทุกเวอร์ชันได้ plant_id เดียวกัน
    ไม่งั้นภาพเดียวกันจะกระจายไปทั้ง train และ test (leakage)
    """
    rel = image_path.relative_to(genus_dir)
    if len(rel.parts) > 1:
        return f"{source}_{genus_dir.name}_{rel.parts[0]}"
    stem = image_path.stem.split(".rf.")[0]
    if "__" in stem:
        return f"{source}_{genus_dir.name}_{stem.split('__')[0]}"
    return f"{source}_{genus_dir.name}_{stem}"


def register_source(store: MetadataStore, source: str, args: argparse.Namespace) -> int:
    root = RAW_DIR / source
    if not root.exists():
        print(f"ไม่พบโฟลเดอร์ {root} — ข้าม")
        return 0

    print(f"\n{'=' * 70}\n{source}\n{'=' * 70}")
    added = 0
    skipped = {"already": 0, "too_small": 0, "duplicate": 0, "broken": 0}

    for genus in CLASSES:
        genus_dir = root / genus
        if not genus_dir.exists():
            continue

        files = sorted(
            p for p in genus_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not files:
            continue

        for path in tqdm(files, desc=f"  {genus}", unit="img", ncols=90):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            if store.has_path(rel):
                skipped["already"] += 1
                continue

            try:
                img = Image.open(path)
                img.load()
                img = img.convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                skipped["broken"] += 1
                tqdm.write(f"    [เปิดไม่ได้] {rel}: {e}")
                continue

            w, h = img.size
            if min(w, h) < args.min_size:
                skipped["too_small"] += 1
                tqdm.write(f"    [เล็กเกินไป {w}x{h}] {rel}")
                continue

            ph = imagehash.phash(img)
            dup_of = store.find_duplicate(ph, args.dup_distance)
            if dup_of:
                skipped["duplicate"] += 1
                tqdm.write(f"    [ซ้ำกับ {dup_of}] {rel}")
                continue

            plant_id = derive_plant_id(path, genus_dir, source)
            # image_id ต้องมาจาก path เต็ม ไม่ใช่แค่ชื่อไฟล์
            # ไม่งั้น plant001/IMG_0001.jpg กับ plant002/IMG_0001.jpg จะได้ id ชนกัน
            stem = path.relative_to(genus_dir).with_suffix("").as_posix().replace("/", "_")
            store.add(make_row(
                image_id=f"{source}_{genus}_{stem}",
                filepath=rel,
                label=genus,
                source=source,
                source_url=args.source_url,
                license=args.license,
                attribution=args.attribution,
                plant_id=plant_id,
                phash=str(ph),
                width=w,
                height=h,
                qc_status="pending",
            ))
            added += 1

    store.save()
    print(f"  เพิ่ม {added} ภาพ   |   ข้าม: " +
          (", ".join(f"{k}={v}" for k, v in skipped.items() if v) or "ไม่มี"))
    return added


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ลงทะเบียนภาพ Roboflow / ภาพถ่ายเอง เข้า metadata.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", required=True, choices=["roboflow", "handon", "all"])
    p.add_argument("--source-url", default="",
                   help="ลิงก์ dataset ต้นทาง (สำคัญสำหรับ Roboflow — ใช้อ้างอิงในรายงาน)")
    p.add_argument("--license", default="",
                   help="license ของ dataset เช่น CC BY 4.0")
    p.add_argument("--attribution", default="",
                   help="เครดิตเจ้าของ เช่น ชื่อ workspace บน Roboflow")
    p.add_argument("--min-size", type=int, default=MIN_IMAGE_SIZE)
    p.add_argument("--dup-distance", type=int, default=DUP_HAMMING_DISTANCE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    store = MetadataStore(METADATA_CSV)
    print(f"metadata.csv มีอยู่แล้ว {len(store):,} แถว")

    sources = ["roboflow", "handon"] if args.source == "all" else [args.source]
    total = sum(register_source(store, s, args) for s in sources)

    print(f"\n{'=' * 70}")
    print(f"เพิ่มภาพใหม่ทั้งหมด {total} ภาพ   (metadata.csv รวม {len(store):,} แถว)")
    print(f"{'=' * 70}\n")
    print(store.summary())


if __name__ == "__main__":
    main()
