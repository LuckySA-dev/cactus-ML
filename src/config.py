"""
ค่าคงที่ทั้งหมดของโปรเจค — ห้าม hard-code path หรือชื่อคลาสกระจายตามไฟล์อื่น

หมายเหตุ: ไฟล์นี้ตั้งใจไม่ import torch เพื่อให้รันสคริปต์เก็บข้อมูล (Phase 1)
ได้ก่อนที่จะติดตั้ง PyTorch
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"          # ภาพดิบ แยกตามแหล่งที่มา (ห้ามแก้ไข)
PROCESSED_DIR = DATA_DIR / "processed"    # หลัง split (ตัวที่เอาไปเทรน)
METADATA_CSV  = DATA_DIR / "metadata.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORT_DIR     = PROJECT_ROOT / "reports"
FIGURE_DIR     = REPORT_DIR / "figures"

SOURCES = ["inaturalist", "roboflow", "handon"]

# ---------------------------------------------------------------------------
# Classes — ลำดับนี้ต้องตรงกับ ImageFolder.class_to_idx (เรียงตามตัวอักษร)
# เหตุผลที่เลือก 7 คลาสนี้ + แผนสำรอง: docs/class_selection.md
# (ariocarpus / lophophora / turbinicarpus ถูกตัดออก 2026-09-19 ภาพเดิมยังอยู่ใน data/raw)
# ---------------------------------------------------------------------------
CLASSES = [
    "astrophytum",
    "echinocactus",
    "echinopsis",
    "gymnocalycium",
    "mammillaria",
    "opuntia",
    "parodia",
]
NUM_CLASSES = len(CLASSES)

CLASS_TH = {
    "astrophytum":   "แอสโตรไฟตัม",
    "echinocactus":  "ถังทอง",
    "echinopsis":    "เอคคินอปซิส",
    "gymnocalycium": "ยิมโนคาไลเซียม",
    "mammillaria":   "แมมมิลาเรีย",
    "opuntia":       "หูกระต่าย",
    "parodia":       "พาโรเดีย",
}

# taxon ที่ใช้ค้นบน iNaturalist (เป็นสกุลหรือสปีชีส์ก็ได้)
# ถังทอง: iNat ย้ายไปสกุล Kroenleinia แล้ว / หูกระต่าย: เอาแค่ microdasys ไม่เอา Opuntia ใบพายต้นใหญ่
INAT_TAXON = {
    "astrophytum":   "Astrophytum",
    "echinocactus":  "Kroenleinia grusonii",
    "echinopsis":    "Echinopsis",
    "gymnocalycium": "Gymnocalycium",
    "mammillaria":   "Mammillaria",
    "opuntia":       "Opuntia microdasys",
    "parodia":       "Parodia",
}

# ---------------------------------------------------------------------------
# ชั้นที่ 2: คลาสย่อยต่อสกุล (ตัดสินใจ 2026-09-20 — เหตุผลใน docs/class_selection.md ข้อ 9)
# key = ชื่อ taxon บน iNat (= ค่าในคอลัมน์ species) → ชื่อคลาสย่อย
# เลือกเฉพาะสปีชีส์ที่นิยมในไทย + มีภาพกระถางบน iNat ≥ ~140
# ---------------------------------------------------------------------------
SUBCLASS_TAXA = {
    "astrophytum": {
        "Astrophytum asterias":    "asterias",
        "Astrophytum myriostigma": "myriostigma",
        "Astrophytum ornatum":     "ornatum",
    },
    "echinopsis": {
        "Echinopsis oxygona": "oxygona",
        "Lobivia silvestrii": "chamaecereus",     # ถั่วลิสง — iNat ย้ายไป Lobivia, วงการไทยเรียก Echinopsis
    },
    "mammillaria": {
        "Mammillaria elongata":  "elongata",
        "Mammillaria plumosa":   "plumosa",
        "Mammillaria prolifera": "prolifera",
        "Mammillaria vetula":    "vetula",
    },
    "parodia": {
        "Parodia lenninghausii": "lenninghausii",
        "Parodia magnifica":     "magnifica",
    },
}
# ระบุสปีชีส์ได้ แต่ไม่อยู่ในรายชื่อ → "other" (ภาพที่ iNat ระบุได้แค่สกุลไม่ใช้เทรนชั้น 2: อาจเป็นสปีชีส์เป้าหมายก็ได้)
SUBCLASS_OTHER = "other"
# ยิมโน: แยกตามหน้าตาที่ซื้อขายในไทย ไม่ใช่สปีชีส์ — ติด label เอง (หัวสี = ไร้คลอโรฟิลล์ ต้องเสียบตอ)
GYMNO_PHENOTYPES = {"colored": "หัวสี", "variegated": "ด่าง", "normal": "ปกติ"}
SUBCLASSES = {genus: sorted([*taxa.values(), SUBCLASS_OTHER]) for genus, taxa in SUBCLASS_TAXA.items()}
SUBCLASSES["gymnocalycium"] = sorted(GYMNO_PHENOTYPES)
# ชื่อที่แสดงในแอป (ชื่อเรียกในวงการไทย)
SUBCLASS_TH = {
    "asterias": "แอสโตรดาว (asterias)", "myriostigma": "หมวกแก๊ป (myriostigma)",
    "ornatum": "ออร์นาตัม (ornatum)", "oxygona": "oxygona", "chamaecereus": "ถั่วลิสง (chamaecereus)",
    "elongata": "ก้างปลา (elongata)", "plumosa": "ขนนก (plumosa)", "prolifera": "prolifera",
    "vetula": "vetula", "lenninghausii": "Yellow tower (leninghausii)", "magnifica": "magnifica",
    **GYMNO_PHENOTYPES, SUBCLASS_OTHER: "สปีชีส์อื่น",
}
# echinocactus / opuntia: สปีชีส์เดียว → ไม่มีชั้น 2

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
TARGET_PER_CLASS = 400

# ⚠️ iNaturalist ขอให้ทุก client ระบุตัวตนและช่องทางติดต่อใน User-Agent
#    ตั้งอีเมลผ่าน environment variable (ไม่เก็บในโค้ดที่ขึ้น git):  set INAT_CONTACT=you@example.com
INAT_CONTACT = os.environ.get("INAT_CONTACT", "")
INAT_USER_AGENT = f"CactusClassifier/0.1 (student project, KMUTNB; contact: <{INAT_CONTACT}>)"

# license ที่นำมาใช้ได้ (ตัด all-rights-reserved ออก)
ALLOWED_PHOTO_LICENSES = ["cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"]

# ระยะ Hamming ของ perceptual hash ที่ถือว่าเป็นภาพเดียวกัน
DUP_HAMMING_DISTANCE = 5

# ภาพที่ด้านสั้นเล็กกว่านี้ตัดทิ้ง (เล็กเกินไป upscale แล้วเบลอ)
MIN_IMAGE_SIZE = 256

# QC label — สุ่มตรวจ 10%/คลาส (วิธีเดียวกับงานวิจัยอ้างอิง [1])
QC_DIR = REPORT_DIR / "qc"
QC_SAMPLE_FRAC = 0.10
# not_potted: iNat captive=True แปลว่า "ปลูกโดยคน" ไม่ใช่ "ในกระถาง" เสมอไป
QC_STATUSES = ["ok", "wrong_label", "not_potted", "blurry", "multi_plant", "not_plant"]

# ---------------------------------------------------------------------------
# Training (ใช้ใน Phase 2 เป็นต้นไป)
# ---------------------------------------------------------------------------
IMG_SIZE    = 224
EVAL_RESIZE = 256         # eval: Resize(256) → CenterCrop(224) เท่ากันทุกโมเดล
BATCH_SIZE  = 64          # RTX 5060 Ti 16GB รับได้สบาย
NUM_WORKERS = 8           # คอขวดคือ decode ภาพ ~1024px: 4 workers 215 img/s → 8 workers 380 img/s (20 CPU)
                          # Windows: ต้องสร้าง DataLoader ใต้ if __name__ == "__main__"
SEED        = 42
DEVICE      = "cuda"

# hyperparameter ชุดเดียวใช้กับทุกโมเดล (plan ข้อ 7)
DROP_RATE       = 0.3
LABEL_SMOOTHING = 0.1     # ทำให้ confidence น่าเชื่อถือขึ้น (ใช้กับ CONFIDENCE_THRESHOLD)
WEIGHT_DECAY    = 1e-4
STAGE1 = {"epochs": 10, "lr": 1e-3}                                    # freeze backbone
# fine-tune ทั้งตัว — lr จูนแล้ว (plan เดิม 1e-5/1e-4 ต่ำเกิน): val F1 0.890 → 0.930±0.005 (3 seeds)
# ที่มา: docs/tuning.md รอบ 1–3
STAGE2 = {"epochs": 40, "lr_backbone": 3e-4, "lr_head": 3e-3, "patience": 8}

LOG_DIR     = REPORT_DIR / "logs"
PRED_DIR    = REPORT_DIR / "predictions"   # ผลทำนายรายภาพ → McNemar / วิเคราะห์ภาพที่ผิด โดยไม่ต้องรันใหม่
RESULTS_CSV = REPORT_DIR / "results.csv"   # ตารางเปรียบเทียบโมเดล (สร้างใหม่ทั้งไฟล์ทุกครั้งที่ evaluate)
RUNS_CSV    = REPORT_DIR / "runs.csv"      # ทุก run ที่เคยเทรน: hyperparameter + best val F1 (ตารางจูนในรายงาน)

MODELS = {
    "mobilenetv3": "mobilenetv3_large_100",
    "effnetv2b0":  "tf_efficientnetv2_b0",
    "resnet50":    "resnet50",
    # เพิ่ม 2026-09-26 จากงานวิจัยอ้างอิง: [2] DenseNet-121 (ชนะ) / EfficientNet-B3, [1] ResNet34 (งานกระบองเพชร)
    "densenet121": "densenet121",
    "resnet34":    "resnet34",
    "effnetb3":    "efficientnet_b3",       # ออกแบบมาที่ 288px แต่ใช้ 224 เท่าตัวอื่น เพื่อเทียบยุติธรรม
}
# Grad-CAM ต้องใช้ชั้นสุดท้ายที่ยังเป็น feature map (7×7) ก่อน global pool — ชื่อไม่เหมือนกันแต่ละโมเดล
# ⚠️ mobilenetv3 ห้ามใช้ conv_head: ใน timm อยู่หลัง global pool แล้ว (1×1) → CAM ไม่มีความหมาย
GRADCAM_LAYERS = {
    "mobilenetv3": "blocks.6",
    "effnetv2b0":  "bn2",
    "resnet50":    "layer4",
    "densenet121": "features.norm5",
    "resnet34":    "layer4",
    "effnetb3":    "conv_head",     # bn2 ให้ cam สว่างที่มุมซ้ายบนแทบทุกภาพ (artifact) — conv_head อยู่กลางต้น
}

# ชื่อที่ใช้ในกราฟ/รายงาน (deck เขียน ResNet50V2 แต่ของเราคือ ResNet50)
MODEL_NAMES = {
    "mobilenetv3": "MobileNetV3-L",
    "effnetv2b0":  "EfficientNetV2-B0",
    "resnet50":    "ResNet50",
    "densenet121": "DenseNet-121",
    "resnet34":    "ResNet34",
    "effnetb3":    "EfficientNet-B3",
}

SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
# ภาพถ่ายเอง (handon) ครึ่งหนึ่ง (แบ่งตามต้น) → realworld_test ที่โมเดลห้ามเห็น อีกครึ่งเข้า pool 70/15/15
REALWORLD_FRAC = 0.5
# qc_status ที่ยังใช้เทรนได้ (pending = ยังไม่ถูกตรวจ ถือว่าใช้ได้)
# not_potted ใช้ได้ (ตัดสินใจ 2026-09-19): label ถูก แค่ปลูกลงดิน — ถ้าตัดทิ้งถังทองจะเหลือ ~25%
# ผลกระทบวัดจาก realworld_test (ภาพกระถางไทย) + ทดลองเทียบ "กระถางล้วน vs รวม" ได้ภายหลัง
QC_USABLE = {"", "pending", "ok", "not_potted"}

# คัดภาพน่าสงสัยด้วยโมเดล (flag_suspects.py): k-fold → ทำนายภาพที่โมเดลไม่เคยเห็น
# flag ถ้าทายไม่ตรง label หรือความน่าจะเป็นของ label < SUSPECT_PROB → ให้คนดูเฉพาะกลุ่มนี้
SUSPECT_CV = {"folds": 5, "model": "effnetv2b0", "epochs1": 5, "epochs2": 15}
SUSPECT_PROB = 0.5

# ค่าความมั่นใจต่ำกว่านี้ → ตอบ "ไม่แน่ใจ" แทนการเดา
# (test: effnetv2b0 ตอบ 91% ของภาพ ถูก 97.7% ของที่ตอบ — ดู reports/figures/confidence.png)
CONFIDENCE_THRESHOLD = 0.60

# ---------------------------------------------------------------------------
# ใช้งานจริง (predict.py / app)
# ---------------------------------------------------------------------------
# โมเดลค่าเริ่มต้นของ predict.py — ใช้ seed ที่ val macro F1 สูงสุด (predict.best_run) ไม่ได้ดู test
DEPLOY_MODEL = "effnetv2b0"
# โมเดลที่ให้เลือกในแอป Streamlit (3 อันดับแรกจาก python -m src.compare --summary)
APP_MODELS = ["effnetv2b0", "effnetb3", "resnet50"]   # 2026-10-07: leaf acc 0.883 / 0.873 / 0.868

# คำแนะนำการดูแล (rule-based ไม่ใช่ ML) — ทีมเขียนเนื้อหา ทุกสกุลต้องมี reference
CARE_RULES_JSON = PROJECT_ROOT / "care_rules" / "care_rules.json"
CARE_TOPICS = {"water": "💧 น้ำ", "light": "☀️ แสง", "soil": "🪴 ดิน"}   # แต่ละหัวข้อมี summary + detail


def ensure_dirs() -> None:
    """สร้างโฟลเดอร์ที่จำเป็นทั้งหมด (เรียกซ้ำได้ ไม่ error)"""
    for source in SOURCES:
        for genus in CLASSES:
            (RAW_DIR / source / genus).mkdir(parents=True, exist_ok=True)
    for d in (PROCESSED_DIR, CHECKPOINT_DIR, REPORT_DIR, FIGURE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
