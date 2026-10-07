"""
subclass.py — เติมคอลัมน์ subclass (คลาสชั้น 2) ใน metadata.csv

    python -m src.data.subclass                 # สกุลที่แยกตามสปีชีส์: เติมจาก species อัตโนมัติ + สรุปจำนวน
    python -m src.data.subclass --gymno-sheet   # ยิมโน: สร้าง reports/qc/gymno_phenotype.{csv,html} ให้ติด label
    python -m src.data.subclass --apply reports/qc/gymno_phenotype.csv

กติกา (สกุลใน config.SUBCLASS_TAXA)
    species อยู่ในรายชื่อ       → ชื่อคลาสย่อย (เช่น Lobivia silvestrii → chamaecereus)
    species อื่นที่ระบุได้      → other
    ระบุได้แค่สกุล (species ว่าง) → ""  ไม่ใช้เทรนชั้น 2 (อาจเป็นสปีชีส์เป้าหมายก็ได้ ใส่ other = label ผิด)
ยิมโน: หน้าตา colored (หัวสี) / variegated (ด่าง) / normal (ปกติ) — ติดเอง ไม่ใช่สปีชีส์ จึงไม่ถูกเขียนทับ
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src.config import GYMNO_PHENOTYPES, QC_DIR, QC_USABLE, SUBCLASS_OTHER, SUBCLASS_TAXA, SUBCLASSES
from src.data.metadata import MetadataStore
from src.data.qc_report import read_sheet, write_html

GYMNO = "gymnocalycium"
SHEET_COLS = ["image_id", "label", "species", "subclass", "filepath"]


def subclass_from_species(genus: str, species: str) -> str:
    if not species:
        return ""
    return SUBCLASS_TAXA[genus].get(species, SUBCLASS_OTHER)


def fill_from_species(store: MetadataStore) -> int:
    changed = 0
    for row in store.rows:
        if row["label"] in SUBCLASS_TAXA:
            new = subclass_from_species(row["label"], row["species"])
            changed += row["subclass"] != new
            row["subclass"] = new
    return changed


def gymno_sheet(store: MetadataStore, out: Path) -> list[dict]:
    """ภาพยิมโนที่ใช้ได้และยังไม่มี phenotype → CSV ให้กรอก subclass + HTML ดูภาพ (เลข # = แถวใน CSV)"""
    rows = [r for r in store.rows if r["label"] == GYMNO and r["qc_status"] in QC_USABLE and not r["subclass"]]
    if out.exists() and any(r["subclass"] for r in read_sheet(out)):
        raise SystemExit(f"{out} มีผลที่กรอกแล้ว — --apply ก่อน แล้วย้ายไฟล์ออกค่อยสร้างใหม่")
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    hint = f"กรอกคอลัมน์ subclass: {', '.join(f'{k} = {v}' for k, v in GYMNO_PHENOTYPES.items())}"
    write_html([r | {"qc_status": "", "notes": hint} for r in rows], out.with_suffix(".html"))
    return rows


def apply_sheet(store: MetadataStore, sheet: Path) -> Counter:
    by_id = {r["image_id"]: r for r in store.rows}
    filled = [r for r in read_sheet(sheet) if r["subclass"].strip()]
    bad = sorted({r["subclass"].strip() for r in filled} - set(GYMNO_PHENOTYPES))
    if bad:
        raise SystemExit(f"subclass ไม่รู้จัก: {bad} — ใช้ได้: {list(GYMNO_PHENOTYPES)}")
    for r in filled:
        if by_id.get(r["image_id"], {}).get("label") != GYMNO:
            raise SystemExit(f"{r['image_id']} ไม่ใช่ภาพยิมโนใน metadata.csv")
        by_id[r["image_id"]]["subclass"] = r["subclass"].strip()
    return Counter(r["subclass"].strip() for r in filled)


def summary(store: MetadataStore) -> None:
    usable = [r for r in store.rows if r["label"] in SUBCLASSES and r["qc_status"] in QC_USABLE]
    for genus, subs in SUBCLASSES.items():
        c = Counter(r["subclass"] for r in usable if r["label"] == genus)
        print(f"{genus:<14} " + "  ".join(f"{s}={c[s]}" for s in subs) + f"  | ไม่ใช้ในชั้น 2={c['']}")


def main() -> None:
    p = argparse.ArgumentParser(description="เติมคลาสชั้น 2 (subclass) ใน metadata.csv")
    p.add_argument("--gymno-sheet", action="store_true", help="สร้างแผ่นติด label หน้าตายิมโน")
    p.add_argument("--apply", type=Path, help="เขียน subclass จากแผ่นที่กรอกแล้วกลับ metadata.csv")
    args = p.parse_args()

    store = MetadataStore()
    if args.gymno_sheet:
        rows = gymno_sheet(store, QC_DIR / "gymno_phenotype.csv")
        print(f"ยิมโน {len(rows)} ภาพ → {QC_DIR / 'gymno_phenotype.html'}")
        return
    if args.apply:
        print("เขียนกลับ:", dict(apply_sheet(store, args.apply)))
    else:
        print(f"เติมจาก species: เปลี่ยน {fill_from_species(store)} แถว")
    store.save()
    summary(store)


if __name__ == "__main__":
    main()
