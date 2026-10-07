"""
download_inat.py — ดึงภาพกระบองเพชรจาก iNaturalist API เข้า data/raw/inaturalist/

การใช้งาน
---------
    # ทดสอบก่อนว่า API ตอบอะไร (ไม่ดาวน์โหลดจริง)
    python -m src.data.download_inat --genus astrophytum --limit 10 --dry-run

    # ดึงทีละสกุล
    python -m src.data.download_inat --genus astrophytum --limit 200

    # ดึงครบทั้ง 7 สกุล
    python -m src.data.download_inat --genus all --limit 200

    # ดูสรุปว่าได้อะไรมาบ้าง
    python -m src.data.metadata


⚠️ เรื่องสำคัญที่ต้องเข้าใจก่อนรัน: quality_grade กับ captive
--------------------------------------------------------------
iNaturalist ออกแบบมาเพื่อบันทึก "สิ่งมีชีวิตในธรรมชาติ"
ภาพต้นไม้ที่ปลูกในกระถางจะถูกติดธง captive=True และถูกลดชั้นเป็น "casual"
ทำให้ **ไม่มีทางเป็น research grade ได้เลย**

แปลว่า:
  --quality-grade research  → ได้ภาพกระบองเพชรในทะเลทราย พื้นหลังหิน/ทราย
                               label เชื่อถือได้สูง แต่คนละโลกกับภาพที่ user จะถ่าย
  --quality-grade any       → ได้ภาพต้นในกระถางด้วย ตรงกับการใช้งานจริง
                               แต่ label อาจผิดได้ ต้อง QC หนักขึ้น

โปรเจคนี้ทำนายภาพกระบองเพชร *แคระในกระถาง* → ต้องมีภาพ captive ด้วย
ค่าเริ่มต้นจึงเป็น --quality-grade any + --min-agreements 1
(รับเฉพาะภาพที่มีคนอื่นยืนยัน ID อย่างน้อย 1 คน เพื่อกรอง label มั่ว)

สคริปต์บันทึก quality_grade / captive / n_agree ลง metadata.csv ทุกภาพ
→ ภายหลังจะกรองยังไงก็ได้ และเอาไปเขียนในรายงานได้ว่าข้อมูลมีสัดส่วนเท่าไหร่
"""
from __future__ import annotations

import argparse
import io
import sys
import time

import imagehash
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.config import (
    ALLOWED_PHOTO_LICENSES,
    CLASSES,
    DUP_HAMMING_DISTANCE,
    INAT_CONTACT,
    INAT_TAXON,
    INAT_USER_AGENT,
    METADATA_CSV,
    MIN_IMAGE_SIZE,
    PROJECT_ROOT,
    RAW_DIR,
    SUBCLASS_TAXA,
    ensure_dirs,
)
from src.data.metadata import MetadataStore, make_row

API = "https://api.inaturalist.org/v1"
SOURCE = "inaturalist"

# iNat ขอไม่เกิน 60 request/นาที และแนะนำให้ต่ำกว่านั้นมากสำหรับงาน bulk
DEFAULT_SLEEP = 1.2


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": INAT_USER_AGENT, "Accept": "application/json"})
    return s


def api_get(session: requests.Session, path: str, params: dict, retries: int = 4) -> dict:
    """เรียก API พร้อม retry แบบ exponential backoff (กัน 429 / 5xx)"""
    url = f"{API}/{path.lstrip('/')}"
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  [rate limit] รอ {wait}s ...", file=sys.stderr)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{retries}] {e} — รอ {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def resolve_taxon_id(session: requests.Session, scientific_name: str) -> int:
    """
    แปลงชื่อสกุลหรือสปีชีส์ (config.INAT_TAXON) -> taxon_id

    ใช้ taxon_id แทน taxon_name เพราะแม่นยำกว่า
    (taxon_name อาจไปชนกับชื่อพ้องของสิ่งมีชีวิตอื่น)
    """
    data = api_get(session, "taxa", {"q": scientific_name, "per_page": 10})
    for result in data.get("results", []):
        if result.get("name", "").lower() == scientific_name.lower():
            n = result.get("observations_count", 0)
            print(f"  taxon_id={result['id']}  ({result['name']}, rank={result.get('rank')}, "
                  f"มี {n:,} observations บน iNat)")
            return result["id"]
    raise LookupError(f"หา taxon_id ของ '{scientific_name}' ไม่เจอ")


def iter_observations(
    session: requests.Session,
    taxon_id: int,
    quality_grade: str,
    licenses: list[str],
    per_page: int,
    sleep: float,
    only_captive: bool = False,
):
    """ไล่อ่าน observations ทีละหน้า (iNat จำกัด page * per_page <= 10,000)"""
    page = 1
    while page * per_page <= 10_000:
        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "photo_license": ",".join(licenses),
            "per_page": per_page,
            "page": page,
            "order_by": "id",
            "order": "desc",
        }
        if quality_grade != "any":
            params["quality_grade"] = quality_grade
        # กรองฝั่ง API — ถ้ากรองฝั่งเราจะชนเพดาน 10,000 obs ก่อนเจอภาพกระถางครบ
        # (เช่น Ferocactus มีภาพธรรมชาติ 90%)
        if only_captive:
            params["captive"] = "true"

        data = api_get(session, "observations", params)
        results = data.get("results", [])
        if not results:
            return
        yield from results
        if page * per_page >= data.get("total_results", 0):
            return
        page += 1
        time.sleep(sleep)


def species_name(taxon: dict | None) -> str:
    """
    taxon ของ observation -> ชื่อสปีชีส์ 2 คำ
    variety/subspecies ตัดเหลือ 2 คำ ("Astrophytum myriostigma var. nudum" -> "Astrophytum myriostigma")
    ระบุได้แค่สกุล หรือเป็นลูกผสม (hybrid) -> "" เพราะใช้เป็น label สปีชีส์ไม่ได้
    """
    if not taxon or taxon.get("rank") not in ("species", "subspecies", "variety", "form"):
        return ""
    return " ".join(taxon.get("name", "").split()[:2])


def backfill_species(session: requests.Session, store: MetadataStore, sleep: float) -> int:
    """เติม species ให้แถว iNat ที่ดาวน์โหลดไปก่อนมีคอลัมน์นี้ (ไม่ต้องโหลดภาพใหม่)"""
    todo: dict[str, list[dict]] = {}
    for row in store.rows:
        if row["source"] == SOURCE and not row["species"] and row["plant_id"].startswith("inat_obs_"):
            todo.setdefault(row["plant_id"].removeprefix("inat_obs_"), []).append(row)

    ids = list(todo)
    filled = 0
    for i in tqdm(range(0, len(ids), 200), desc="  backfill", unit="req", ncols=90):
        data = api_get(session, "observations", {"id": ",".join(ids[i:i + 200]), "per_page": 200})
        for obs in data.get("results", []):
            name = species_name(obs.get("taxon"))
            for row in todo.get(str(obs["id"]), []):
                row["species"] = name
                filled += bool(name)
        store.save()                    # เซฟทุก batch → หลุดกลางทางรันต่อได้
        time.sleep(sleep)
    return filled


def to_large_url(url: str) -> str:
    """
    iNat ส่ง url ขนาด square (75px) มาให้ ต้องเปลี่ยนเป็น large (1024px)
    เช่น  .../photos/123/square.jpg  ->  .../photos/123/large.jpg
    """
    # เส้นทาง IPv4 ไป S3 ช้ามาก (~8 KB/s เมื่อ 2026-09-19) แต่ endpoint dualstack เร็วปกติ
    base = url.split("?")[0].replace(
        "inaturalist-open-data.s3.amazonaws.com",
        "inaturalist-open-data.s3.dualstack.us-east-1.amazonaws.com",
    )
    for size in ("square", "thumb", "small", "medium"):
        marker = f"/{size}."
        if marker in base:
            return base.replace(marker, "/large.")
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def download_taxon(
    session: requests.Session,
    store: MetadataStore,
    genus: str,
    scientific: str,
    args: argparse.Namespace,
    other_than: set[str] | None = None,
) -> int:
    """
    ดึงภาพของ taxon หนึ่งเข้าคลาส genus
    scientific เป็นสกุล (Astrophytum) หรือสปีชีส์ (Lobivia silvestrii → เข้าคลาส echinopsis) ก็ได้
    other_than: เก็บเฉพาะภาพที่ระบุสปีชีส์ได้ และไม่ใช่สปีชีส์ในชุดนี้ → ใช้เป็นคลาส "other" ของชั้น 2
    """
    print(f"\n{'=' * 70}\n{genus}  ({scientific}{' · other species' if other_than else ''})\n{'=' * 70}")

    taxon_id = resolve_taxon_id(session, scientific)
    time.sleep(args.sleep)

    out_dir = RAW_DIR / SOURCE / genus
    out_dir.mkdir(parents=True, exist_ok=True)

    # --limit = เป้าหมายรวม (ต่อสกุล หรือต่อสปีชีส์ถ้า scientific เป็นชื่อ 2 คำ) ไม่ใช่จำนวนที่เพิ่มต่อรอบ
    # ไม่งั้นรันซ้ำหลังสคริปต์หลุดกลางทางจะได้ภาพเกินเป้า
    is_species = " " in scientific

    def wanted(species: str) -> bool:
        if other_than is not None:
            return species != "" and species not in other_than
        return not is_species or species == scientific

    existing = sum(1 for r in store.rows if r["source"] == SOURCE and r["label"] == genus and wanted(r["species"]))
    need = args.limit - existing
    if need <= 0:
        print(f"  มีครบแล้ว {existing}/{args.limit} ภาพ — ข้าม")
        return 0
    if existing:
        print(f"  มีอยู่แล้ว {existing} ภาพ → ดึงเพิ่มอีก {need}")

    kept = 0
    skipped = {"already": 0, "no_license": 0, "low_agree": 0, "not_wanted": 0, "too_small": 0,
               "duplicate": 0, "broken": 0}

    bar = tqdm(total=need, desc=f"  {genus}", unit="img", ncols=90)
    try:
        for obs in iter_observations(
            session, taxon_id, args.quality_grade, args.licenses, args.per_page, args.sleep,
            args.only_captive,
        ):
            if kept >= need:
                break

            n_agree = obs.get("num_identification_agreements", 0) or 0
            grade = obs.get("quality_grade", "")
            # research grade ผ่านชัวร์ ส่วน casual/needs_id ต้องมีคนยืนยันตามเกณฑ์
            if grade != "research" and n_agree < args.min_agreements:
                skipped["low_agree"] += 1
                continue

            obs_id = obs["id"]
            captive = obs.get("captive", False)
            if args.only_captive and not captive:
                continue
            species = species_name(obs.get("taxon"))
            if not wanted(species):
                skipped["not_wanted"] += 1
                continue

            photos = obs.get("photos", []) or []
            for photo in photos[: args.max_photos_per_obs]:
                if kept >= need:
                    break

                license_code = (photo.get("license_code") or "").lower()
                if license_code not in args.licenses:
                    skipped["no_license"] += 1
                    continue

                image_id = f"inat_{obs_id}_{photo['id']}"
                if store.has_id(image_id):
                    skipped["already"] += 1
                    continue

                if args.dry_run:
                    tqdm.write(
                        f"    [dry-run] {image_id}  grade={grade:<9} captive={str(captive):<5} "
                        f"agree={n_agree}  {license_code}"
                    )
                    kept += 1
                    bar.update(1)
                    continue

                # ---- download -------------------------------------------------
                try:
                    r = session.get(to_large_url(photo["url"]), timeout=45)
                    r.raise_for_status()
                    img = Image.open(io.BytesIO(r.content))
                    img.load()
                    img = img.convert("RGB")
                except (requests.RequestException, UnidentifiedImageError, OSError) as e:
                    skipped["broken"] += 1
                    tqdm.write(f"    [เสีย] {image_id}: {e}")
                    continue
                finally:
                    time.sleep(args.photo_sleep)

                w, h = img.size
                if min(w, h) < args.min_size:
                    skipped["too_small"] += 1
                    continue

                ph = imagehash.phash(img)
                dup_of = store.find_duplicate(ph, args.dup_distance)
                if dup_of:
                    skipped["duplicate"] += 1
                    tqdm.write(f"    [ซ้ำกับ {dup_of}] {image_id}")
                    continue

                out_path = out_dir / f"{image_id}.jpg"
                img.save(out_path, "JPEG", quality=92)

                store.add(make_row(
                    image_id=image_id,
                    filepath=out_path.relative_to(PROJECT_ROOT).as_posix(),
                    label=genus,
                    species=species,
                    source=SOURCE,
                    source_url=f"https://www.inaturalist.org/observations/{obs_id}",
                    license=license_code,
                    attribution=photo.get("attribution", ""),
                    # ⭐ ภาพจาก observation เดียวกัน = ต้นเดียวกัน
                    #    ใช้คู่กับ GroupShuffleSplit เพื่อกัน data leakage
                    plant_id=f"inat_obs_{obs_id}",
                    phash=str(ph),
                    width=w,
                    height=h,
                    quality_grade=grade,
                    captive=captive,
                    n_agree=n_agree,
                    qc_status="pending",
                ))
                kept += 1
                bar.update(1)

                if kept % args.save_every == 0:
                    store.save()
    finally:
        bar.close()
        if not args.dry_run:
            store.save()

    print(f"  ได้เพิ่ม {kept} ภาพ   |   ข้าม: " +
          ", ".join(f"{k}={v}" for k, v in skipped.items() if v))
    if kept < need:
        print(f"  ⚠️  ได้ไม่ถึงเป้า {args.limit} ภาพ (รวม {existing + kept}) — ลองลด --min-agreements "
              f"หรือเพิ่ม --max-photos-per-obs")
    return kept


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ดึงภาพกระบองเพชรจาก iNaturalist",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--genus",
                   help=f"ชื่อสกุล หรือ 'all'  ({', '.join(CLASSES)})")
    p.add_argument("--species",
                   help="ชื่อ taxon ใน config.SUBCLASS_TAXA (เช่น 'Mammillaria vetula') หรือ 'all' — --limit = ต่อสปีชีส์")
    p.add_argument("--other", action="store_true",
                   help="ภาพสปีชีส์นอกรายชื่อ SUBCLASS_TAXA ของแต่ละสกุล (คลาส other) — --limit = ต่อสกุล")
    p.add_argument("--backfill-species", action="store_true",
                   help="เติมคอลัมน์ species ให้ภาพ iNat ที่ดาวน์โหลดไปแล้ว (ไม่โหลดภาพใหม่)")
    p.add_argument("--limit", type=int, default=200,
                   help="จำนวนภาพที่ต้องการต่อสกุล")
    p.add_argument("--quality-grade", default="any",
                   choices=["any", "research", "casual", "needs_id"],
                   help="any = รวมภาพต้นปลูกในกระถางด้วย (แนะนำ อ่าน docstring)")
    p.add_argument("--min-agreements", type=int, default=1,
                   help="ภาพที่ไม่ใช่ research grade ต้องมีคนยืนยัน ID อย่างน้อยเท่านี้")
    p.add_argument("--only-captive", action="store_true",
                   help="เอาเฉพาะต้นปลูก (ทดลองดูว่าได้กี่ภาพ)")
    p.add_argument("--max-photos-per-obs", type=int, default=1,
                   help="เอากี่ภาพต่อ 1 observation (1 = หลากหลายสุด)")
    p.add_argument("--per-page", type=int, default=200, help="ขนาดหน้าของ API")
    p.add_argument("--min-size", type=int, default=MIN_IMAGE_SIZE,
                   help="ด้านสั้นของภาพต้องไม่น้อยกว่านี้ (px)")
    p.add_argument("--dup-distance", type=int, default=DUP_HAMMING_DISTANCE,
                   help="ระยะ perceptual hash ที่ถือว่าเป็นภาพซ้ำ")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                   help="หน่วงระหว่างเรียก API (วินาที)")
    p.add_argument("--photo-sleep", type=float, default=0.3,
                   help="หน่วงระหว่างโหลดภาพ (วินาที)")
    p.add_argument("--save-every", type=int, default=25,
                   help="เซฟ metadata.csv ทุกกี่ภาพ")
    p.add_argument("--dry-run", action="store_true",
                   help="ดูว่าจะได้อะไรบ้าง โดยไม่ดาวน์โหลดจริง")
    return p.parse_args()


def resolve_targets(args: argparse.Namespace) -> list[tuple[str, str, set[str] | None]]:
    """--genus / --species / --other → [(คลาสสกุล, taxon, other_than)]"""
    species = {taxon: genus for genus, taxa in SUBCLASS_TAXA.items() for taxon in taxa}
    if args.other:
        return [(genus, INAT_TAXON[genus], set(taxa)) for genus, taxa in SUBCLASS_TAXA.items()]
    if args.species:
        if args.species != "all" and args.species not in species:
            raise SystemExit(f"ไม่รู้จัก '{args.species}' — เลือกจาก config.SUBCLASS_TAXA: {', '.join(species)}")
        names = list(species) if args.species == "all" else [args.species]
        return [(species[name], name, None) for name in names]
    if not args.genus:
        raise SystemExit("ต้องระบุ --genus, --species, --other หรือ --backfill-species")
    genera = CLASSES if args.genus == "all" else [args.genus]
    unknown = [g for g in genera if g not in INAT_TAXON]
    if unknown:
        raise SystemExit(f"ไม่รู้จักสกุล {unknown} — เลือกจาก: {', '.join(CLASSES)}")
    return [(g, INAT_TAXON[g], None) for g in genera]


def main() -> None:
    args = parse_args()
    args.licenses = [x.lower() for x in ALLOWED_PHOTO_LICENSES]

    if not INAT_CONTACT:
        print("⚠️  ยังไม่ได้ตั้งอีเมลติดต่อ — set INAT_CONTACT=you@example.com ก่อนรัน\n"
              "   (iNaturalist ขอให้ทุก client ระบุช่องทางติดต่อใน User-Agent)\n")

    if args.backfill_species:
        store = MetadataStore(METADATA_CSV)
        n = backfill_species(make_session(), store, args.sleep)
        print(f"เติม species ได้ {n} ภาพ (ที่เหลือว่าง = iNat ระบุได้แค่ระดับสกุล)")
        return
    targets = resolve_targets(args)       # [(คลาสสกุล, ชื่อ taxon บน iNat)]

    ensure_dirs()
    store = MetadataStore(METADATA_CSV)
    print(f"metadata.csv มีอยู่แล้ว {len(store):,} แถว")

    session = make_session()
    total = 0
    for genus, scientific, other_than in targets:
        total += download_taxon(session, store, genus, scientific, args, other_than)

    if not args.dry_run:
        store.save()
    print(f"\n{'=' * 70}")
    print(f"เพิ่มภาพใหม่ทั้งหมด {total} ภาพ   (metadata.csv รวม {len(store):,} แถว)")
    print(f"{'=' * 70}\n")
    print(store.summary())


if __name__ == "__main__":
    main()
