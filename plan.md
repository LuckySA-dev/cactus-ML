# plan.md — Cactus Classifier

**ระบบจำแนกสายพันธุ์กระบองเพชรแคระ + ระบบแนะนำวิธีดูแล**

| | |
|---|---|
| **งาน** | Multi-class Image Classification (7 คลาส ระดับสกุล/Genus) |
| **สมาชิก** | กลุ่มนักศึกษา 4 คน (KMUTNB) |
| **Framework** | **PyTorch + timm** (Transfer Learning) |
| **Hardware** | RTX 5060 Ti 16GB (Blackwell, sm_120) |
| **โมเดลที่เปรียบเทียบ** | MobileNetV3-Large / EfficientNetV2-B0 / ResNet50 |
| **Target** | Test Accuracy ≥ 91% (baseline จากงานวิจัยที่ 1) |
| **สถานะปัจจุบัน** | 🔴 ยังไม่มีภาพเลย — Phase 1 คืองานเร่งด่วนที่สุด |

---

## 0. สรุปสิ่งที่ต้องสร้าง

ระบบมี **2 ส่วน** ที่ต้องแยกให้ชัด:

1. **ส่วน ML** — จำแนกภาพ → ทำนายสกุลกระบองเพชร (7 คลาส) + ค่าความมั่นใจ
2. **ส่วน Rule-based** — รับสกุลที่ทำนายได้ → lookup คำแนะนำการดูแล (น้ำ/แสง/ดิน) จากไฟล์ JSON

> ⚠️ ส่วนที่ 2 **ไม่ใช่งาน Machine Learning** เป็นแค่การอ่านไฟล์ JSON ที่เราเขียนเอง คะแนนของวิชานี้อยู่ที่ส่วนที่ 1 อย่าไปเสียเวลากับส่วนที่ 2 มาก

### 🔴 อ่านก่อนเริ่ม — Bottleneck จริงของโปรเจคนี้

ด้วย RTX 5060 Ti 16GB กับ dataset 2,800 ภาพ **การเทรนใช้เวลาแค่ไม่กี่นาทีต่อโมเดล** (2,800 ภาพ ÷ batch 64 = 44 steps/epoch — ระดับ 5-15 วินาที/epoch)

แปลว่า **การเทรนไม่ใช่ปัญหาเลย** — 90% ของความสำเร็จอยู่ที่ **คุณภาพและความสะอาดของข้อมูล**
อย่าเพิ่งไปนั่งจูน hyperparameter ถ้าข้อมูลยังไม่สะอาด และเพราะเทรนเร็ว เราจึงทำสิ่งที่งานวิจัยทั้ง 3 ชิ้นไม่ได้ทำได้ เช่น **5-fold Cross Validation** และ **รันซ้ำ 3 seeds** (ดูข้อ 7.5) ซึ่งจะทำให้ผลของเรา**น่าเชื่อถือกว่าทั้ง 3 งานวิจัย**

---

## 1. คลาสทั้ง 7 (Genus level)

> 🔄 **เปลี่ยนชุดคลาสเมื่อ 2026-09-19** — เหตุผล หลักฐาน (จำนวนภาพ, ความนิยมในไทย, pilot experiment)
> และแผนสำรอง อยู่ที่ **`docs/class_selection.md`** · ชุดเดิม (ariocarpus / lophophora / turbinicarpus) ถูกตัดออก

| # | Label (folder name) | ชื่อไทย | ลักษณะเด่น | iNat taxon |
|---|---|---|---|---|
| 0 | `astrophytum` | แอสโตรไฟตัม | ลำต้นกลม มีจุดขาว (speckle) สันชัด | *Astrophytum* |
| 1 | `echinocactus` | ถังทอง | ทรงถังกลมใหญ่ หนามเหลืองทอง | *Kroenleinia grusonii* |
| 2 | `echinopsis` | เอคคินอปซิส | ทรงกระบอก/กลม สันลึก หนามยาว | *Echinopsis* |
| 3 | `gymnocalycium` | ยิมโนคาไลเซียม | กลมแบน สันโค้ง หนามโค้ง | *Gymnocalycium* |
| 4 | `mammillaria` | แมมมิลาเรีย | ตุ่ม (tubercle) เรียงเป็นเกลียว ขนขาว | *Mammillaria* |
| 5 | `opuntia` | หูกระต่าย | ลำต้นเป็นแผ่นแบน จุดขนเป็นกระจุก | *Opuntia microdasys* |
| 6 | `parodia` | พาโรเดีย / Yellow tower | กลม-ทรงกระบอก หนามทองละเอียด | *Parodia* |

### ความเสี่ยงที่ต้องรู้

- **echinopsis ↔ mammillaria / gymnocalycium** สับสนกันมากที่สุดใน pilot (echinopsis มีลูกผสมหลากหลาย) → **อธิบายในรายงานเป็นผลการทดลอง**
- **mammillaria** recall ต่ำสุดใน pilot เพราะมี 144 สปีชีส์หน้าตาต่างกันมาก → เก็บภาพให้ครอบคลุมหลายสปีชีส์
- iNat `captive=True` = "ปลูกโดยคน" ไม่ใช่ "ในกระถาง" เสมอไป (ถังทอง/หูกระต่ายบางภาพปลูกลงดิน) → QC ตัดออก
- ทุกคลาสมีภาพกระถาง ≥ 1,000 บน iNat จึง **ไม่ควรเกิดปัญหาภาพไม่พอ** ถ้าเกิด → ดูแผนสำรองใน `docs/class_selection.md` ข้อ 8

---

## 2. โครงสร้างโปรเจค

```
cactus-classifier/
├── data/
│   ├── raw/                      # ภาพดิบ แยกตามแหล่งที่มา (ห้ามแก้ไข)
│   │   ├── inaturalist/<genus>/
│   │   ├── roboflow/<genus>/
│   │   └── handon/<genus>/
│   ├── interim/<genus>/          # หลัง dedup + QC
│   ├── processed/                # หลัง split (ตัวที่เอาไปเทรน)
│   │   ├── train/<genus>/
│   │   ├── val/<genus>/
│   │   ├── test/<genus>/
│   │   └── realworld_test/<genus>/   # ⭐ ภาพถ่ายเอง 200 ใบ โมเดลห้ามเห็น
│   └── metadata.csv              # ⭐ ทะเบียนภาพทุกใบ (อ่านข้อ 4.3)
├── src/
│   ├── config.py                 # ค่าคงที่ทั้งหมดอยู่ที่เดียว
│   ├── data/
│   │   ├── download_inat.py      # ดึงภาพจาก iNaturalist API
│   │   ├── dedup.py              # ตรวจภาพซ้ำด้วย perceptual hash
│   │   ├── qc_report.py          # สุ่มตรวจ label 10% ต่อคลาส
│   │   └── split.py              # แบ่ง train/val/test แบบ stratified
│   ├── datasets.py               # Dataset + transforms + DataLoader
│   ├── models.py                 # สร้างโมเดล 3 ตัวผ่าน timm
│   ├── engine.py                 # train_one_epoch / validate
│   ├── train.py                  # loop หลัก (2 stage)
│   ├── evaluate.py               # metrics + confusion matrix
│   ├── gradcam.py                # ภาพอธิบายว่าโมเดลมองตรงไหน
│   ├── export.py                 # → ONNX
│   └── predict.py                # ทำนาย 1 ภาพ + ดึงคำแนะนำการดูแล
├── care_rules/care_rules.json
├── checkpoints/                  # .pt ที่เทรนเสร็จ
├── reports/
│   ├── figures/                  # กราฟ, confusion matrix, gradcam
│   └── results.csv               # ผลเปรียบเทียบ 3 โมเดล
├── app/app.py                    # Streamlit demo
├── requirements.txt
└── plan.md
```

---

## 3. Environment

### 🔴 ข้อควรระวังอันดับ 1: RTX 5060 Ti เป็น Blackwell (sm_120)

**`pip install torch` เฉย ๆ จะใช้ไม่ได้** — จะขึ้น error
```
CUDA error: no kernel image is available for execution on the device
```
เพราะ PyTorch build มาตรฐาน (CUDA 12.6) ยังไม่ได้ compile kernel สำหรับ sm_120

**ต้องลง build ที่รองรับ CUDA 12.8 ขึ้นไป:**
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**ตรวจสอบก่อนทำอย่างอื่นทั้งหมด:**
```python
import torch
print(torch.__version__)                      # ควร >= 2.7
print(torch.cuda.is_available())              # ต้องเป็น True
print(torch.cuda.get_device_name(0))          # ควรเห็น "RTX 5060 Ti"
print(torch.cuda.get_device_capability(0))    # ต้องได้ (12, 0)
x = torch.randn(8, 3, 224, 224, device="cuda")
print((x @ x.transpose(-1, -2)).sum())        # ถ้าไม่ error = ใช้ได้จริง
```
> บรรทัดสุดท้ายสำคัญ — `cuda.is_available()` เป็น True ได้ทั้งที่ kernel ใช้ไม่ได้ ต้องลองคำนวณจริง

### `requirements.txt`
```
timm>=1.0.0
numpy pandas
matplotlib seaborn
scikit-learn
pillow imagehash
requests tqdm
grad-cam
onnx onnxruntime
streamlit
tensorboard
```

### `src/config.py` — ค่าคงที่อยู่ไฟล์เดียว ห้าม hard-code กระจาย
```python
import torch

IMG_SIZE    = 224
BATCH_SIZE  = 64          # 16GB VRAM รับได้สบาย
NUM_WORKERS = 8
SEED        = 42
DEVICE      = "cuda"
NUM_CLASSES = 7

CLASSES = ["astrophytum", "echinocactus", "echinopsis", "gymnocalycium",
           "mammillaria", "opuntia", "parodia"]   # เรียงตามตัวอักษร = ImageFolder (ของจริงดู src/config.py)

MODELS = {
    "mobilenetv3": "mobilenetv3_large_100",
    "effnetv2b0":  "tf_efficientnetv2_b0",
    "resnet50":    "resnet50",
}

SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
```

### ตั้ง seed ให้ครบ (ไม่งั้นเทรนซ้ำได้ผลไม่เหมือนเดิม → เขียนรายงานไม่ได้)
```python
def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True    # เร็วขึ้นเพราะ input size คงที่
```

---

## 4. Phase 1 — เก็บและเตรียมข้อมูล ⬅️ **เริ่มตรงนี้ (~40% ของโปรเจค)**

### 4.1 เป้าหมายจำนวนภาพ

| แหล่ง | จำนวน | หมายเหตุ |
|---|---|---|
| iNaturalist | 2,800 | **400/คลาส ภาพกระถางล้วน** (`--only-captive`, research หรือ agree ≥ 1) — อัปเดต 2026-09-19 |
| Roboflow Universe | เสริม | ถ้าหลัง QC คลาสไหนไม่พอ — ต้องเช็ค license ของแต่ละ dataset |
| เก็บเอง (Hand-On) | 400 | ~57/คลาส — ถ่ายจากร้านต้นไม้ด้วยมือถือ (ครึ่งหนึ่งเป็น realworld_test) |

> ⚠️ ตัวเลขนี้ **ไม่ตรงกับหน้า reference ใน deck** ที่เขียนว่า Kaggle + Mendeley Data — ต้องแก้ให้ตรงกันก่อนส่งรายงาน (ดูข้อ 12)
> Kaggle ประเมินแล้วไม่ใช้ (มีแค่ ~60 ภาพ/สกุล และไม่ระบุ license) — `docs/class_selection.md` ข้อ 4.2

### 4.2 iNaturalist API

```
GET https://api.inaturalist.org/v1/observations
    ?taxon_id=<จาก config.INAT_TAXON>
    &captive=true                      # ต้นปลูก (ในกระถาง) — ไม่มีวันเป็น research grade
    &photo_license=cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa
    &per_page=200&page=1
```
- casual ต้องมีคนอื่นยืนยัน ID ≥ 1 (`num_identification_agreements`) — ไม่นับ ID ของเจ้าของภาพเอง (ตรวจแล้ว)
- ดึง `photos[].url` แล้วเปลี่ยน `square` → `large` เพื่อได้ภาพความละเอียดสูง
- **เก็บ `license` + `attribution` + `observation_id` ลง metadata.csv ทุกภาพ** — จำเป็นสำหรับรายงาน ไม่งั้นตอบไม่ได้ว่าเอาภาพมาจากไหน
- ใส่ `time.sleep(1)` ระหว่างการเรียก ไม่งั้นโดน rate limit

### 4.3 ⭐ metadata.csv — หัวใจของโปรเจค

สร้างตั้งแต่ภาพแรก ทุกภาพต้องมี 1 แถว:

| column | ตัวอย่าง | ใช้ทำอะไร |
|---|---|---|
| `image_id` | `inat_astro_0001` | ชื่ออ้างอิงเฉพาะ |
| `filepath` | `data/raw/inaturalist/astrophytum/...jpg` | path |
| `label` | `astrophytum` | คลาส |
| `source` | `inaturalist` / `roboflow` / `handon` | **ตอบข้อ 6 ในรายงาน** |
| `source_url` | `https://inaturalist.org/observations/12345` | อ้างอิง |
| `license` | `CC-BY-NC` | ป้องกันปัญหาลิขสิทธิ์ |
| `phash` | `f8e2c1a...` | ตรวจภาพซ้ำ |
| `split` | `train` / `val` / `test` / `realworld_test` | บอกว่าอยู่ชุดไหน |
| `qc_status` | `ok` / `wrong_label` / `blurry` / `multi_plant` | ผลตรวจคุณภาพ |

→ ตอบอาจารย์ได้ทันทีว่า *"แต่ละแหล่งเอามาเท่าไหร่"* ด้วย `df.groupby(['source','label']).size()`

### 4.4 ทำความสะอาดข้อมูล (รันตามลำดับ)

1. **ลบภาพซ้ำ (`dedup.py`)** — `imagehash.phash()` ถ้า Hamming distance ≤ 5 ถือว่าซ้ำ เก็บไว้ใบเดียว
   > 🔴 **สำคัญที่สุด** — iNaturalist กับ Roboflow มีโอกาสสูงมากที่จะมีภาพเดียวกัน (Roboflow หลาย dataset ก็ scrape มาจาก iNaturalist) ถ้าภาพซ้ำไปอยู่คนละ split จะเกิด **data leakage** ทำให้ accuracy สูงเกินจริง แล้วโดนอาจารย์จับได้
2. **คัดภาพเสีย** — เบลอ, ภาพวาด/การ์ตูน, หลายต้นคละสกุล, ดอกอย่างเดียวไม่เห็นลำต้น, ขนาด < 200px
3. **QC label 10% ต่อคลาส (`qc_report.py`)** — สุ่ม 40 ภาพ/คลาส ให้สมาชิกช่วยกันดูว่า label ถูกไหม บันทึกลง `qc_status`
   > วิธีนี้ลอกมาจากงานวิจัยที่ 1 → เขียนในรายงานได้ว่า *"ใช้วิธีเดียวกับ [1]"*

### 4.5 แบ่งชุดข้อมูล (`split.py`) — ตามที่ตกลงไว้

```
ภาพ iNaturalist + Roboflow (2,400) + Hand-On ครึ่งแรก (200)
        │
        └─► Stratified split 70/15/15 ──► train / val / test

ภาพ Hand-On อีกครึ่ง (200)
        │
        └─► realworld_test/     ⬅️ โมเดลห้ามเห็นเด็ดขาด
```

- ใช้ `sklearn.model_selection.train_test_split(stratify=y, random_state=42)`
- **แบ่งก่อน augment เสมอ** — augment เฉพาะ train
- **แบ่ง Hand-On ให้เป็น stratified ด้วย** (แต่ละสกุลได้ ~28 ใบใน train / ~28 ใบใน realworld_test)
- ⚠️ ระวัง: ถ้าถ่ายต้นเดียวกันหลายมุม **ต้องให้ทุกมุมของต้นนั้นอยู่ split เดียวกัน** ไม่งั้น leakage → เก็บ `plant_id` ตอนถ่ายด้วย แล้ว split by group (`GroupShuffleSplit`)

> 💡 **ทำไมต้องมี realworld_test:** งานวิจัยที่ 1 ได้ val accuracy 91% แต่พอทดสอบกับภาพใหม่จากโลกจริงเหลือ **81.5%** ตกไป ~10%
> การมี realworld_test แยก ทำให้เรารายงาน accuracy ได้ 2 ตัว และ**อธิบาย gap ได้อย่างมีหลักฐาน** — นี่คือจุดขายของงานเรา

---

## 5. Phase 2 — Data Pipeline

### 5.1 Dataset

```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

train_ds = ImageFolder("data/processed/train", transform=train_tf)
val_ds   = ImageFolder("data/processed/val",   transform=eval_tf)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,
                      num_workers=8, pin_memory=True,
                      persistent_workers=True, drop_last=True)
```
> ⚠️ ถ้ารันบน **Windows** ต้องครอบโค้ดด้วย `if __name__ == "__main__":` ไม่งั้น `num_workers > 0` จะ crash

**ตรวจสอบเสมอว่า `train_ds.class_to_idx` ตรงกับลำดับใน `CLASSES`** — `ImageFolder` เรียงตามตัวอักษร ถ้าไม่ตรง confusion matrix จะ label ผิดทั้งหมด

### 5.2 Transforms

```python
from torchvision.transforms import v2

train_tf = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(20),
    v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
    v2.RandomErasing(p=0.25),
])

eval_tf = v2.Compose([
    v2.Resize(256), v2.CenterCrop(224),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])
```

**ห้ามใส่:** `RandomVerticalFlip` (กระบองเพชรมีบน-ล่างชัดเจน), `hue` แรง ๆ (สีเป็น feature สำคัญในการแยกสกุล)

### 5.3 🔴 กับดัก: ค่า Normalize ต้องตรงกับโมเดล

แต่ละ pretrained model ใช้ mean/std คนละค่า **ถ้าใส่ผิด accuracy ตกแบบหาสาเหตุไม่เจอ**

| Model | mean / std |
|---|---|
| ResNet50, MobileNetV3 | ImageNet: `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` |
| tf_efficientnetv2_b0 | ~~Inception-style `[0.5,0.5,0.5]`~~ → **timm 1.0.29 (`.in1k`) ใช้ ImageNet** (ตรวจ 2026-09-19) |

> บทเรียน: ค่าในตารางนี้เปลี่ยนได้ตาม pretrained tag ของ timm — นี่คือเหตุผลที่ต้องถาม timm ไม่ใช่จำเอง

**อย่าจำเอง — ให้ timm บอก:**
```python
import timm
cfg  = timm.data.resolve_model_data_config(model)
MEAN, STD = cfg["mean"], cfg["std"]
```
วิธีนี้ทำให้เปลี่ยนโมเดลแล้วค่า normalize เปลี่ยนตามอัตโนมัติ ไม่มีทางพลาด

---

## 6. Phase 3 — Model

```python
import timm

def build_model(name: str, num_classes: int = 7, pretrained: bool = True):
    return timm.create_model(MODELS[name], pretrained=pretrained,
                             num_classes=num_classes, drop_rate=0.3)
```

| key | timm model name | Params | เหตุผลที่เลือก |
|---|---|---|---|
| `mobilenetv3` | `mobilenetv3_large_100` | ~5.5M | เบา เหมาะกับมือถือ — ตรงเป้าหมายทำแอป, งานวิจัยที่ 1 เลือกตัวนี้ |
| `effnetv2b0` | `tf_efficientnetv2_b0` | ~7.1M | สมดุลความแม่นยำ/ขนาดดีที่สุดในกลุ่มเล็ก |
| `resnet50` | `resnet50` | ~25.6M | Baseline มาตรฐาน, residual แยกของหน้าตาคล้ายกันได้ดี (ตามงานวิจัยที่ 3) |

> **หมายเหตุเรื่องชื่อ:** ใน deck เขียน "ResNet50**V2**" ซึ่งเป็นชื่อฝั่ง Keras — ในโลก PyTorch ตัวมาตรฐานคือ `resnet50` (v1.5) ที่ผลดีกว่าและใช้กันทั่วไป **ในรายงานให้เขียนว่า "ResNet50" เฉย ๆ** (ถ้าอยากได้ V2 จริง ๆ คือ `resnetv2_50` แต่ไม่จำเป็นและ pretrained weight มีตัวเลือกน้อยกว่า)

---

## 7. Phase 4 — Training

### 7.1 เทรน 2 Stage (อย่าข้าม)

| | **Stage 1: Feature Extraction** | **Stage 2: Fine-tuning** |
|---|---|---|
| Backbone | Freeze ทั้งหมด | Unfreeze ทั้งหมด |
| Learning rate | `1e-3` | ~~`1e-5` / `1e-4`~~ → **`3e-4` (backbone) / `3e-3` (head)** จูนแล้ว (`docs/tuning.md`) |
| Epochs | 10 | 40 (early stopping patience 8) |
| Optimizer | AdamW (`weight_decay=1e-4`) | AdamW |
| Scheduler | – | CosineAnnealingLR |

**ทำไมต้อง 2 stage:** ถ้า unfreeze ตั้งแต่แรกด้วย lr สูง gradient จาก head ที่ยัง random จะพังน้ำหนัก pretrained ทั้งหมด (catastrophic forgetting) — ผลจะแย่กว่าไม่ fine-tune เลย

```python
# Stage 1
for p in model.parameters():
    p.requires_grad = False
for p in model.get_classifier().parameters():
    p.requires_grad = True

# Stage 2 — discriminative learning rate
optimizer = torch.optim.AdamW([
    {"params": backbone_params,   "lr": 3e-4},   # จูนแล้ว (เดิม 1e-5 ต่ำเกิน)
    {"params": classifier_params, "lr": 3e-3},
], weight_decay=1e-4)
```

> **BatchNorm ต้องระวังตอน Stage 1** — `model.train()` จะอัปเดต running stats ของ BN แม้ `requires_grad=False` ทำให้ผล Stage 1 เพี้ยน
> วิธีแก้: หลัง `model.train()` ให้วนปิด BN ที่ freeze ไว้ → `if isinstance(m, nn.BatchNorm2d): m.eval()`

### 7.2 Loss

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```
`label_smoothing` ช่วยไม่ให้โมเดลมั่นใจเกินจริง → **ค่า confidence ที่ได้จะน่าเชื่อถือขึ้น** ซึ่งสำคัญกับเรา เพราะเราจะใช้ threshold 0.60 ตัดสินว่า "ไม่แน่ใจ"

ถ้าข้อมูลไม่สมดุล ใช้ `WeightedRandomSampler` แทนการใส่ `weight` ใน loss (ได้ผลดีกว่าสำหรับ image classification)

### 7.3 AMP + Performance (ใช้ GPU ให้คุ้ม)

```python
model = model.to(DEVICE, memory_format=torch.channels_last)

with torch.autocast("cuda", dtype=torch.bfloat16):
    out  = model(x)
    loss = criterion(out, y)
loss.backward()
```
- **ใช้ `bfloat16`** — Blackwell รองรับเต็มที่ และ**ไม่ต้องใช้ `GradScaler`** ต่างจาก `float16` (โค้ดสั้นลง ไม่มีปัญหา NaN)
- `channels_last` เร็วขึ้น ~20-30% กับ conv net
- `torch.compile(model)` เร็วขึ้นอีกได้ แต่**ลองทีหลัง** — compile บน GPU สถาปัตยกรรมใหม่บางทีมีปัญหา ถ้า error ให้ข้ามไป ไม่ใช่สาระสำคัญของงาน

### 7.4 Checkpoint & Early Stopping

PyTorch ไม่มี callback ให้ ต้องเขียนเอง:
```python
if val_f1 > best_f1:
    best_f1, patience_counter = val_f1, 0
    torch.save({"model": model.state_dict(),
                "epoch": epoch, "val_f1": val_f1,
                "class_to_idx": train_ds.class_to_idx},   # ⬅️ เก็บด้วยเสมอ
               f"checkpoints/{name}_best.pt")
else:
    patience_counter += 1
    if patience_counter >= 8:
        break
```
> เลือก best จาก **macro F1** ไม่ใช่ accuracy — เพราะถ้าบางคลาสภาพน้อย accuracy จะหลอกตา

### 7.5 ⭐ สิ่งที่ทำให้งานเราเหนือกว่างานวิจัยทั้ง 3 ชิ้น

เทรนเร็วมาก (ไม่กี่นาที/โมเดล) เลยทำสิ่งเหล่านี้ได้ฟรี ๆ:

1. **รันซ้ำ 3 seeds (42, 43, 44)** แล้วรายงาน `mean ± std` → งานวิจัยที่ 1 ทำ (3 รอบ) แต่ที่ 2 และ 3 ไม่ได้ทำ
2. **5-fold Cross Validation** → ไม่มีงานวิจัยไหนในสามชิ้นทำเลย **นี่คือจุดที่จะได้คะแนนเพิ่ม**
3. **เทรน baseline แบบ `pretrained=False`** 1 รอบ → พิสูจน์เชิงตัวเลขว่า Transfer Learning ช่วยจริงแค่ไหน

---

## 8. Phase 5 — Evaluation

วัดบน **test set ที่โมเดลไม่เคยเห็น** เท่านั้น (ห้ามรายงาน val accuracy เป็นผลสุดท้าย)

### Metrics ที่ต้องมี
- Accuracy, **Macro** Precision / Recall / F1 (macro เพราะทุกคลาสสำคัญเท่ากัน)
- **Per-class report** — `classification_report(y_true, y_pred, target_names=CLASSES, digits=4)`
- **Confusion Matrix** (normalized `'true'`) — คาดว่าจะเห็นความสับสน Echinopsis ↔ Mammillaria / Gymnocalycium (จาก pilot)
- **Inference time / model size / params** — ตอบว่าตัวไหนเหมาะกับมือถือ (ตามงานวิจัยที่ 2)

### ⭐ ตารางผลลัพธ์ที่เป็นจุดขายของรายงาน

| Model | Test Acc | Macro F1 | **Real-world Acc** | **Gap** | Params | Inference (ms) |
|---|---|---|---|---|---|---|
| MobileNetV3-L | | | | | 5.5M | |
| EfficientNetV2-B0 | | | | | 7.1M | |
| ResNet50 | | | | | 25.6M | |

**คอลัมน์ Gap คือหัวใจ** — งานวิจัยที่ 1 มี gap 9.86% ถ้าเราทำ gap ให้น้อยกว่าได้ = เราทำได้ดีกว่างานที่อ้างอิง

### สิ่งที่ควรเพิ่ม
- **Grad-CAM** (`pip install grad-cam`) — ดูว่าโมเดลมองตรงไหน ถ้าไปดูกระถางหรือพื้นหลังแทนต้นไม้ = ข้อมูลมีปัญหา
  ```python
  from pytorch_grad_cam import GradCAM
  # ชั้นที่ใช้จริงอยู่ใน config.GRADCAM_LAYERS: resnet50 → layer4, effnetv2b0 → bn2, mobilenetv3 → blocks.6
  # ⚠️ ห้ามใช้ conv_head กับ MobileNetV3 — ใน timm อยู่หลัง global pool (1×1) CAM จะไม่มีความหมาย
  ```
- **Confidence threshold** — ถ้า max softmax < 0.60 ตอบ *"ไม่แน่ใจ กรุณาถ่ายใหม่"* แทนการเดามั่วแล้วให้คำแนะนำการดูแลผิด

### `reports/results.csv`
```
model,seed,test_acc,macro_f1,realworld_acc,gap,params_m,infer_ms,size_mb
```

---

## 9. Phase 6 — ระบบแนะนำวิธีดูแล (Rule-based)

**`care_rules/care_rules.json`**
```json
{
  "astrophytum": {
    "name_th": "แอสโตรไฟตัม",
    "water":  { "summary": "รดเมื่อดินแห้งสนิท ~7-10 วัน/ครั้ง", "detail": "..." },
    "light":  { "summary": "แดดรำไร 4-6 ชม./วัน", "detail": "..." },
    "soil":   { "summary": "ดินโปร่ง ระบายน้ำดี ผสมหินภูเขาไฟ 50%", "detail": "..." },
    "common_mistake": "รดน้ำบ่อยเกินไปทำให้โคนเน่า",
    "reference": "<แหล่งอ้างอิงที่เชื่อถือได้>"
  }
}
```

> ⚠️ **ทุกสกุลต้องมี `reference`** — อย่าเขียนจากความเข้าใจส่วนตัว อาจารย์ถามได้ว่า *"เอาข้อมูลนี้มาจากไหน"*

**`predict.py` ทำแค่นี้:**
```
ภาพ → eval_tf → model → softmax → max prob
     → ถ้า prob < 0.60 : "ไม่แน่ใจ กรุณาถ่ายใหม่"
     → ไม่งั้น : lookup care_rules.json → แสดงผล
```

---

## 10. Phase 7 — Demo App + Export

**`app/app.py` (Streamlit)**
```
1. st.file_uploader / st.camera_input
2. แสดงภาพที่อัปโหลด
3. ทำนาย → top-3 พร้อม % ความมั่นใจ (bar chart)
4. การ์ดคำแนะนำ: 💧 น้ำ | ☀️ แสง | 🪴 ดิน | ⚠️ ข้อผิดพลาดที่พบบ่อย
5. (ถ้ามีเวลา) Grad-CAM ข้างภาพต้นฉบับ
```
> โหลดโมเดลด้วย `@st.cache_resource` ไม่งั้นจะโหลดใหม่ทุกครั้งที่กดปุ่ม

**Export → ONNX** (แทน TFLite เพราะเราใช้ PyTorch)
```python
torch.onnx.export(model, dummy, "cactus.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}}, opset_version=17)
```
เขียนในรายงานว่า *"ONNX ต่อยอดเป็น ONNX Runtime Mobile / แปลงเป็น TFLite ได้"*

---

## 11. ลำดับการทำงาน

| # | งาน | ผลลัพธ์ที่ต้องได้ | เวลา |
|---|---|---|---|
| 0 | **ติดตั้ง PyTorch cu128 + ทดสอบ GPU** | `get_device_capability()` = (12,0) | 0.5 วัน |
| 1 | Setup repo + config.py | รันได้ไม่ error | 0.5 วัน |
| 2 | `download_inat.py` | ~1,400 ภาพ + metadata.csv | 2 วัน |
| 3 | หา Roboflow dataset + เช็ค license | ~1,000 ภาพ | 1 วัน |
| 4 | **ถ่ายภาพเอง** (แบ่งกันไปตามร้าน, เก็บ `plant_id`) | 400 ภาพ | 2-3 วัน |
| 5 | Dedup + คัดภาพเสีย + QC 10% | `data/interim/` สะอาด | 2 วัน |
| 6 | Split 70/15/15 + แยก realworld_test | `data/processed/` | 0.5 วัน |
| 7 | `datasets.py` + `models.py` + `engine.py` | forward pass ผ่าน | 1 วัน |
| 8 | **Smoke test: 3 epochs กับ 50 ภาพ** | pipeline ไม่พัง | 0.5 วัน |
| 9 | เทรน 3 โมเดล × 2 stage × 3 seeds | 9 checkpoints | 1 วัน |
| 10 | Evaluate + Confusion Matrix + Grad-CAM | `results.csv` + กราฟ | 1 วัน |
| 11 | (ถ้ามีเวลา) 5-fold CV | ตาราง mean ± std | 0.5 วัน |
| 12 | `care_rules.json` (แบ่งกันเขียน) | JSON ครบ 7 สกุล | 1 วัน |
| 13 | Streamlit app + ONNX export | demo ใช้งานได้ | 1 วัน |
| 14 | เขียนรายงาน + slide | ส่งงาน | 2 วัน |

> **ขั้นที่ 0 และ 8 ห้ามข้าม** — ขั้น 0 กันไม่ให้เจอปัญหา GPU ตอนใกล้ส่ง, ขั้น 8 กันไม่ให้รอเทรนเสร็จแล้วค่อยรู้ว่า pipeline พัง

---

## 12. ความเสี่ยง & ทางแก้

| ความเสี่ยง | สัญญาณเตือน | ทางแก้ |
|---|---|---|
| **PyTorch ไม่รองรับ Blackwell** | `no kernel image is available` | ลง `--index-url .../cu128` (ข้อ 3) |
| **Data leakage จากภาพซ้ำ** | test acc สูงผิดปกติ (>98%) | รัน `dedup.py` ก่อน split เสมอ |
| **Leakage จากภาพต้นเดียวกันหลายมุม** | realworld acc ต่ำกว่า test มาก | split by `plant_id` ด้วย `GroupShuffleSplit` |
| **class_to_idx ไม่ตรงกับ CLASSES** | confusion matrix label สลับ | `assert train_ds.class_to_idx == expected` |
| **เก็บภาพไม่ครบ 400/คลาส** | หลัง QC คลาสไหนเหลือ < 300 | เติม Roboflow/ถ่ายเอง หรือสลับคลาสสำรอง (`docs/class_selection.md` ข้อ 8) |
| **โมเดลจำพื้นหลังแทนต้นไม้** | Grad-CAM ชี้ไปที่กระถาง/โต๊ะ | เพิ่มความหลากหลายพื้นหลัง, เพิ่ม `RandomResizedCrop` scale |
| **Overfit** | train 99% / val 75% (gap > 15%) | เพิ่ม `drop_rate`, augmentation, `RandomErasing` |
| **echinopsis สับสนกับแมม/ยิมโน** | confusion นอก diagonal เข้ม | ปกติ — อธิบายในรายงาน ถ้าหนักมากดูแผนสำรอง |

---

## 13. 🔧 สิ่งที่ต้องแก้ใน Pitch Deck ก่อนส่งรายงานฉบับเต็ม

1. **Framework ไม่ตรง** — หน้า Key Takeaways เขียน *"ใช้ Transfer Learning บน Keras"* แต่เราจะใช้ **PyTorch** → ต้องแก้
2. **แหล่งที่มาของภาพไม่ตรงกัน** — หน้า 6 เขียน `iNaturalist / Roboflow / เก็บเอง` แต่ reference [4] เขียน `iNaturalist, Kaggle, Mendeley Data และภาพที่ถ่ายเอง` → เลือกให้ตรงกัน
3. **งานวิจัยที่ 2 ลิงก์ผิด** — ระบุว่าตีพิมพ์ใน *BITS* แต่ URL ชี้ไปที่ `ph01.tci-thaijo.org/index.php/**ecticit**/...` ซึ่งเป็นวารสาร **ECTI-CIT** คนละฉบับ → ตรวจสอบและแก้
4. **หัวสไลด์หน้า 9-11, 13-14, 16-18 ยังเป็นงานวิจัยที่ 1** ทั้งที่เนื้อหาเป็นงานวิจัยที่ 2 และ 3 (ลืมแก้ตอน duplicate slide)
5. **เพิ่มหัวข้อ Real-world Test Set** — เป็นจุดต่างที่ทำให้งานเราเหนือกว่างานวิจัยที่อ้างอิง ควรใส่ในสไลด์

---

## 14. คำสั่งที่จะใช้บ่อย

```bash
# ตรวจ GPU ก่อนทำอะไรทั้งหมด
python -c "import torch; print(torch.cuda.get_device_capability(0))"

# เตรียมข้อมูล (dedup ทำอัตโนมัติตอนดาวน์โหลด/ลงทะเบียน)
python -m src.data.download_inat --genus all --limit 400 --only-captive
python -m src.data.register_local --source handon
python -m src.data.qc_report --sample 0.10          # → กรอก reports/qc/qc_sheet.csv
python -m src.data.qc_report --apply reports/qc/qc_sheet.csv
python -m src.data.flag_suspects                    # โมเดลช่วยคัดภาพน่าสงสัย → reports/qc/suspects.csv
python -m src.data.split

# เทรน (3 โมเดล × 3 seeds, hparam จาก config — จูนแล้ว ดู docs/tuning.md)
python -m src.train --model effnetv2b0 --seed 42
python -m src.train --model effnetv2b0 --seed 42 --no-pretrained   # baseline ไม่ใช้ transfer learning

# ประเมิน + กราฟ
python -m src.evaluate                              # ทุก checkpoint → reports/results.csv
python -m src.visualize                             # → reports/figures/*.png
python -m src.gradcam  --model resnet50 --n 20

# ส่งออก + demo
python -m src.export --model resnet50
streamlit run app/app.py
```
