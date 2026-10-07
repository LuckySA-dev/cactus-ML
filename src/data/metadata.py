"""
metadata.py — ทะเบียนภาพกลางของโปรเจค (data/metadata.csv)

ทุกภาพในโปรเจคต้องมี 1 แถวในไฟล์นี้ ไม่ว่าจะมาจากแหล่งไหน
เป็นไฟล์ที่ทำให้ตอบคำถามอาจารย์ได้ทันทีว่า "แต่ละแหล่งเอาภาพมาเท่าไหร่"

รันดูสรุปได้ด้วย:
    python -m src.data.metadata
"""
from __future__ import annotations

import contextlib
import csv
from collections.abc import Iterable
from pathlib import Path

import imagehash

from src.config import METADATA_CSV

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
COLUMNS = [
    "image_id",       # รหัสเฉพาะของภาพ เช่น inat_12345_67890
    "filepath",       # path สัมพัทธ์จาก project root
    "label",          # ชื่อสกุล (= ชื่อโฟลเดอร์)
    "species",        # ชื่อสปีชีส์ 2 คำ เช่น "Astrophytum asterias" / ว่าง = ระบุได้แค่สกุล
                      # ใช้กับโมเดลชั้นที่ 2 (สกุล → สปีชีส์)
    "subclass",       # คลาสชั้น 2 ใน config.SUBCLASSES (asterias / other / colored ...) ว่าง = ไม่ใช้ในชั้น 2
                      # สกุลทั่วไปเติมอัตโนมัติจาก species (src/data/subclass.py) · ยิมโนติด label เอง
    "source",         # inaturalist / roboflow / handon
    "source_url",     # ลิงก์ต้นทาง (สำหรับอ้างอิงในรายงาน)
    "license",        # cc0 / cc-by / cc-by-nc / ...
    "attribution",    # ชื่อเจ้าของภาพ
    "plant_id",       # ⭐ ต้นเดียวกัน = plant_id เดียวกัน → ใช้ split by group
    "phash",          # perceptual hash (ตรวจภาพซ้ำ)
    "width",
    "height",
    "quality_grade",  # iNat: research / needs_id / casual
    "captive",        # iNat: True = ต้นปลูก (ในกระถาง) / False = ต้นในธรรมชาติ
    "n_agree",        # iNat: จำนวนคนที่ยืนยัน identification ตรงกัน
    "split",          # train / val / test / realworld_test  (เติมทีหลังโดย split.py)
    "qc_status",      # ok / wrong_label / blurry / multi_plant / ...
    "notes",
]


def make_row(**kwargs) -> dict:
    """สร้างแถวที่มีครบทุกคอลัมน์ คอลัมน์ที่ไม่ได้ส่งมาจะเป็นค่าว่าง"""
    unknown = set(kwargs) - set(COLUMNS)
    if unknown:
        raise KeyError(f"คอลัมน์ที่ไม่รู้จัก: {sorted(unknown)}")
    row = {c: "" for c in COLUMNS}
    row.update({k: ("" if v is None else str(v)) for k, v in kwargs.items()})
    return row


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class MetadataStore:
    """
    อ่าน/เขียน metadata.csv แบบ resume ได้

    - โหลดของเดิมขึ้นมาตอนสร้าง object → รันสคริปต์ซ้ำจะไม่ดาวน์โหลดภาพเดิม
    - เช็คภาพซ้ำด้วย perceptual hash ข้ามทุกแหล่งที่มา
    """

    def __init__(self, path: Path | str = METADATA_CSV):
        self.path = Path(path)
        self.rows: list[dict] = []
        self._ids: set[str] = set()
        self._paths: set[str] = set()
        self._hashes: list[tuple[imagehash.ImageHash, str]] = []
        if self.path.exists():
            self.load()

    # -- io -----------------------------------------------------------------
    def load(self) -> None:
        with self.path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                full = {c: row.get(c, "") or "" for c in COLUMNS}
                self._index(full)
                self.rows.append(full)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".csv.tmp")
        # utf-8-sig เพื่อให้ Excel บน Windows เปิดภาษาไทยไม่เป็นตัวต่างดาว
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)
        tmp.replace(self.path)          # เขียนทับแบบ atomic กันไฟล์พังตอน Ctrl+C

    # -- lookup -------------------------------------------------------------
    def _index(self, row: dict) -> None:
        self._ids.add(row["image_id"])
        if row["filepath"]:
            self._paths.add(row["filepath"].replace("\\", "/"))
        if row["phash"]:
            with contextlib.suppress(ValueError):    # phash เสีย → ข้ามการเช็คซ้ำของแถวนี้
                self._hashes.append((imagehash.hex_to_hash(row["phash"]), row["image_id"]))

    def has_id(self, image_id: str) -> bool:
        return image_id in self._ids

    def has_path(self, filepath: str) -> bool:
        return str(filepath).replace("\\", "/") in self._paths

    def find_duplicate(self, phash: imagehash.ImageHash, max_distance: int = 5) -> str | None:
        """คืน image_id ของภาพที่ซ้ำ ถ้าไม่ซ้ำคืน None"""
        for existing, image_id in self._hashes:
            if (phash - existing) <= max_distance:
                return image_id
        return None

    # -- mutate -------------------------------------------------------------
    def add(self, row: dict) -> None:
        missing = set(COLUMNS) - set(row)
        if missing:
            raise KeyError(f"แถวขาดคอลัมน์: {sorted(missing)}")
        # กัน image_id ชนกันเงียบ ๆ — ถ้าชนแปลว่า logic ตั้งชื่อมีปัญหา
        if row["image_id"] in self._ids:
            raise ValueError(
                f"image_id ซ้ำ: {row['image_id']!r} (ไฟล์ {row['filepath']}) — "
                "แปลว่าวิธีตั้งชื่อ image_id ไม่ unique พอ"
            )
        self._index(row)
        self.rows.append(row)

    def extend(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.add(row)

    def __len__(self) -> int:
        return len(self.rows)

    # -- report -------------------------------------------------------------
    def summary(self) -> str:
        """ตารางสรุป จำนวนภาพ แยกตาม label × source"""
        from collections import defaultdict

        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        sources: list[str] = []
        for row in self.rows:
            counts[row["label"]][row["source"]] += 1
            if row["source"] not in sources:
                sources.append(row["source"])
        sources.sort()

        label_w = max([len("label")] + [len(x) for x in counts]) + 2
        head = "label".ljust(label_w) + "".join(s.rjust(14) for s in sources) + "TOTAL".rjust(9)
        lines = [head, "-" * len(head)]

        totals = defaultdict(int)
        for label in sorted(counts):
            row_total = sum(counts[label].values())
            line = label.ljust(label_w)
            for s in sources:
                n = counts[label][s]
                totals[s] += n
                line += str(n).rjust(14)
            lines.append(line + str(row_total).rjust(9))

        lines.append("-" * len(head))
        line = "TOTAL".ljust(label_w)
        for s in sources:
            line += str(totals[s]).rjust(14)
        lines.append(line + str(sum(totals.values())).rjust(9))
        return "\n".join(lines)


def main() -> None:
    store = MetadataStore()
    if not len(store):
        print(f"ยังไม่มีข้อมูลใน {METADATA_CSV}")
        return
    print(store.summary())
    print()

    # สรุปเพิ่มเติมที่ใช้ตอบในรายงาน
    from collections import Counter

    for col in ("license", "quality_grade", "captive", "split", "qc_status"):
        counter = Counter(r[col] for r in store.rows if r[col])
        if counter:
            pretty = ", ".join(f"{k}={v}" for k, v in counter.most_common())
            print(f"{col:<14}: {pretty}")

    plants = {r["plant_id"] for r in store.rows if r["plant_id"]}
    print(f"{'unique plants':<14}: {len(plants)}")


if __name__ == "__main__":
    main()
