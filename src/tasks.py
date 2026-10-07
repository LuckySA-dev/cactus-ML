"""
tasks.py — "งาน" ที่เทรน/ประเมินได้ ใช้ร่วมกันใน split / train / evaluate

    genus              ชั้น 1: 7 สกุล (ใช้ทุกภาพ)                      data/processed/<split>/<สกุล>/
    species-<สกุล>     ชั้น 2 แบบ A: คลาสย่อยของสกุลนั้น (ต่อจาก genus)  data/processed/species/<สกุล>/<split>/<คลาสย่อย>/
    flat               แบบ B: ทุกคลาสย่อยในโมเดลเดียว แล้วอนุมานสกุล     data/processed/flat/<split>/<leaf>/

leaf = คลาสย่อยสุด เช่น astrophytum__asterias, echinocactus (สกุลที่ไม่มีชั้น 2)
ภาพที่ไม่รู้คลาสย่อย (iNat ระบุแค่สกุล) ใช้ได้เฉพาะงาน genus
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.config import CLASSES, PROCESSED_DIR, SUBCLASSES

LEAF_SEP = "__"
LEAVES = [genus if genus not in SUBCLASSES else f"{genus}{LEAF_SEP}{sub}"
          for genus in CLASSES for sub in SUBCLASSES.get(genus, [None])]


def leaf_of(row: dict) -> str | None:
    genus = row["label"]
    if genus not in SUBCLASSES:
        return genus
    return f"{genus}{LEAF_SEP}{row['subclass']}" if row["subclass"] in SUBCLASSES[genus] else None


def genus_of(leaf: str) -> str:
    return leaf.split(LEAF_SEP)[0]


@dataclass(frozen=True)
class Task:
    name: str
    classes: list[str]                          # ลำดับ = index ของโมเดล
    root: Path                                  # มี <split>/<class>/ อยู่ข้างใน
    class_of: Callable[[dict], str | None]      # แถว metadata → ชื่อคลาส (None = ไม่ใช้ในงานนี้)


def get_task(name: str) -> Task:
    if name == "genus":
        return Task(name, list(CLASSES), PROCESSED_DIR, lambda r: r["label"] if r["label"] in CLASSES else None)
    if name == "flat":
        return Task(name, LEAVES, PROCESSED_DIR / "flat", leaf_of)
    genus = name.removeprefix("species-")
    if name.startswith("species-") and genus in SUBCLASSES:
        subs = SUBCLASSES[genus]
        return Task(name, subs, PROCESSED_DIR / "species" / genus,
                    lambda r: r["subclass"] if r["label"] == genus and r["subclass"] in subs else None)
    raise SystemExit(f"ไม่รู้จักงาน '{name}' — เลือกจาก: {', '.join(TASKS)}")


TASKS = ["genus", *[f"species-{genus}" for genus in SUBCLASSES], "flat"]
# งานหลักที่รายงาน/ใช้ในแอป — แบบ B ชนะแบบ A ทั้งชั้นสกุลและชั้นย่อย (reports/hierarchy_*.csv)
MAIN_TASK = "flat"


def run_id(model: str, seed: int, task: str = MAIN_TASK) -> str:
    """ชื่อ run หลัก (= train.run_name เมื่อไม่มี --tag) — งาน genus ใช้ชื่อเดิมไม่มี prefix"""
    return f"{model}_seed{seed}" if task == "genus" else f"{task}_{model}_seed{seed}"
