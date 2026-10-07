"""
qc_report.py — สุ่มตรวจ label 10% ต่อคลาส แล้วเขียนผลกลับ metadata.csv

ขั้นตอน
-------
    # 1) สร้างชุดตรวจ → reports/qc/qc_sheet.html (ดูภาพ) + qc_sheet.csv (กรอกผล)
    python -m src.data.qc_report --sample 0.10

    # 2) เปิด qc_sheet.html ดูภาพ แล้วกรอกคอลัมน์ qc_status ใน qc_sheet.csv ด้วย Excel
    #    ค่าที่ใช้ได้: ok / wrong_label / not_potted / blurry / multi_plant / not_plant
    #    แถวที่เว้นว่าง = ยังไม่ได้ตรวจ (ไม่ถูกเขียนกลับ)

    # 3) เขียนผลกลับ metadata.csv + สรุปอัตรา error ต่อคลาส (ใช้ในรายงาน)
    python -m src.data.qc_report --apply reports/qc/qc_sheet.csv

    # 4) สร้างหน้าตรวจซ้ำ (มีผล QC + กรอบแดงภาพที่ไม่ผ่าน) → reports/qc/qc_sheet.html
    python -m src.data.qc_report --render reports/qc/qc_sheet.csv

HTML ฝังรูปย่อไว้ในไฟล์ → ส่งต่อทาง LINE/Drive เปิดเครื่องไหนก็เห็นรูป (data/raw ไม่ได้ถูก commit)

อยากตรวจทุกภาพ (เช่นคัดภาพที่ไม่ใช่กระถาง) ใช้ --sample 1.0
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.config import CLASSES, PROJECT_ROOT, QC_DIR, QC_SAMPLE_FRAC, QC_STATUSES, SEED
from src.data.metadata import MetadataStore

SHEET_COLS = ["image_id", "label", "species", "qc_status", "notes", "source_url", "filepath"]


def sample_rows(store: MetadataStore, frac: float, seed: int) -> list[dict]:
    """สุ่มแบบ stratified: ทุกคลาสได้ ceil(frac × จำนวนภาพที่ยังไม่ตรวจ)"""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in store.rows:
        if row["label"] in CLASSES and row["qc_status"] in ("", "pending"):
            by_label[row["label"]].append(row)
    rng = random.Random(seed)
    picked = []
    for label in CLASSES:
        rows = sorted(by_label[label], key=lambda r: r["image_id"])   # sort ก่อน → seed เดิมได้ชุดเดิม
        picked += rng.sample(rows, min(len(rows), math.ceil(frac * len(rows))))
    return picked


def write_sheet(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """ชุดตรวจใหม่: CSV (qc_status ว่าง ให้ทีมกรอก) + HTML ดูภาพ"""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, html_path = out_dir / "qc_sheet.csv", out_dir / "qc_sheet.html"
    blank = [{c: ("" if c in ("qc_status", "notes") else r[c]) for c in SHEET_COLS} for r in rows]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLS)
        w.writeheader()
        w.writerows(blank)
    write_html(blank, html_path)
    return csv_path, html_path


def thumbnail_uri(filepath: str, size: int = 320) -> str:
    """ฝังรูปย่อใน HTML เลย — data/raw ไม่ถูก commit ถ้าอ้างไฟล์ เปิดเครื่องอื่นจะ not found"""
    im = Image.open(PROJECT_ROOT / filepath).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def write_html(rows: list[dict], html_path: Path) -> None:
    """หน้าดูภาพไฟล์เดียว ส่งต่อทาง LINE/Drive ได้ — ภาพที่ qc_status ไม่ใช่ ok มีกรอบแดง"""
    cards = defaultdict(list)
    for i, r in enumerate(rows):
        status = r.get("qc_status", "")
        bad = status not in ("", "ok")
        tag = f'<br><b class="{"bad" if bad else "ok"}">{html.escape(status)}</b>' if status else ""
        if r.get("notes"):
            tag += f"<br><small>{html.escape(r['notes'])}</small>"
        cards[r["label"]].append(
            f'<figure class="{"bad" if bad else ""}">'
            f'<img loading="lazy" src="{thumbnail_uri(r["filepath"])}">'
            f'<figcaption>#{i} {html.escape(r["image_id"])}<br><i>{html.escape(r["species"] or "-")}</i>{tag}'
            f'</figcaption></figure>'
        )
    body = "".join(f"<h2>{label} ({len(c)})</h2><div class=g>{''.join(c)}</div>" for label, c in cards.items())
    html_path.write_text(
        "<!doctype html><meta charset=utf-8><title>QC sheet</title><style>"
        "body{font-family:sans-serif;margin:16px;background:#fff;color:#111}.g{display:flex;flex-wrap:wrap;gap:8px}"
        "figure{margin:0;width:200px;font-size:12px;border:3px solid transparent}figure.bad{border-color:#d33}"
        "img{width:200px;height:200px;object-fit:cover}b.bad{color:#d33}b.ok{color:#383}"
        f"</style><p>สถานะที่ใช้ได้: {', '.join(QC_STATUSES)} — แก้ใน qc_sheet.csv (เลข # = ลำดับแถวใน CSV)</p>"
        f"{body}",
        encoding="utf-8",
    )


def read_sheet(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def apply_sheet(store: MetadataStore, sheet: Path) -> dict[str, Counter]:
    """เขียน qc_status/notes กลับ metadata — คืน Counter ของสถานะต่อคลาส"""
    by_id = {r["image_id"]: r for r in store.rows}
    filled = [r for r in read_sheet(sheet) if r["qc_status"].strip()]

    bad = sorted({r["qc_status"].strip() for r in filled} - set(QC_STATUSES))
    if bad:
        raise SystemExit(f"qc_status ไม่รู้จัก: {bad} — ใช้ได้: {QC_STATUSES}")
    unknown = [r["image_id"] for r in filled if r["image_id"] not in by_id]
    if unknown:
        raise SystemExit(f"image_id ไม่อยู่ใน metadata.csv: {unknown[:5]}")

    stats: dict[str, Counter] = defaultdict(Counter)
    for r in filled:
        row = by_id[r["image_id"]]
        row["qc_status"] = r["qc_status"].strip()
        row["notes"] = r["notes"].strip() or row["notes"]
        stats[row["label"]][row["qc_status"]] += 1
    store.save()
    return stats


def print_stats(stats: dict[str, Counter]) -> None:
    print(f"{'label':<15}{'ตรวจ':>6}{'ok':>6}{'error %':>9}   ปัญหาที่พบ")
    total = Counter()
    for label in CLASSES:
        c = stats.get(label, Counter())
        n = sum(c.values())
        total += c
        errs = ", ".join(f"{k}={v}" for k, v in c.items() if k != "ok")
        print(f"{label:<15}{n:>6}{c['ok']:>6}{(1 - c['ok'] / n) * 100 if n else 0:>8.1f}%   {errs}")
    n = sum(total.values())
    print(f"{'TOTAL':<15}{n:>6}{total['ok']:>6}{(1 - total['ok'] / n) * 100 if n else 0:>8.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="สุ่มตรวจ label ต่อคลาส")
    p.add_argument("--sample", type=float, default=QC_SAMPLE_FRAC, help="สัดส่วนที่สุ่มตรวจต่อคลาส")
    p.add_argument("--apply", type=Path, help="qc_sheet.csv ที่กรอกผลแล้ว → เขียนกลับ metadata.csv")
    p.add_argument("--render", type=Path, help="สร้าง HTML (พร้อมผล QC) จาก qc_sheet.csv ที่กรอกแล้ว ไว้ตรวจซ้ำ")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    if args.render:
        html_path = args.render.with_suffix(".html")
        write_html(read_sheet(args.render), html_path)
        print(f"→ {html_path}")
        return

    store = MetadataStore()
    if args.apply:
        print_stats(apply_sheet(store, args.apply))
        return

    existing = QC_DIR / "qc_sheet.csv"
    if existing.exists() and any(r["qc_status"].strip() for r in read_sheet(existing)):
        raise SystemExit(f"{existing} มีผล QC อยู่แล้ว — ย้าย/เปลี่ยนชื่อไฟล์ก่อนสุ่มชุดใหม่ (กันผลตรวจหาย)")

    rows = sample_rows(store, args.sample, args.seed)
    csv_path, html_path = write_sheet(rows, QC_DIR)
    print(f"สุ่มได้ {len(rows)} ภาพ: " + ", ".join(f"{k}={v}" for k, v in Counter(r['label'] for r in rows).items()))
    print(f"  ดูภาพ : {html_path}\n  กรอกผล: {csv_path}")


if __name__ == "__main__":
    main()
