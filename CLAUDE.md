# CLAUDE.md

บริบทโปรเจคสำหรับ Claude Code — อ่านไฟล์นี้ก่อนเริ่มทำงานทุกครั้ง
แผนงานฉบับเต็มอยู่ที่ `plan.md` (อ่านก่อนเขียนโค้ดใหม่ทุกครั้ง)

---

## โปรเจคนี้คืออะไร

**Cactus Classifier** — งานวิชา Machine Learning (นักศึกษา KMUTNB)
จำแนกสกุลกระบองเพชรแคระจากภาพถ่าย 7 คลาส แล้วต่อยอดเป็นระบบแนะนำวิธีดูแล

ระบบมี 2 ส่วน แยกให้ชัด:
1. **ML** — image classification 7 คลาส + ค่าความมั่นใจ ← คะแนนอยู่ตรงนี้
2. **Rule-based** — lookup `care_rules/care_rules.json` ด้วยสกุลที่ทำนายได้ **ไม่ใช้ ML**

**เป้าหมาย:** Test Accuracy ≥ 91% (baseline จากงานวิจัยอ้างอิง)

---

## Stack & สภาพแวดล้อม

| | |
|---|---|
| OS | **Windows** |
| GPU | **RTX 5060 Ti 16GB (Blackwell, sm_120)** |
| Framework | **PyTorch + timm** (ไม่ใช่ Keras) |
| Image size | 224×224 |
| Batch size | 64 |

### 🔴 กฎที่ห้ามผิด

1. **PyTorch ต้องเป็น build CUDA 12.8+** — Blackwell (sm_120) ใช้ build มาตรฐานไม่ได้
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   อาการถ้าผิด: `CUDA error: no kernel image is available for execution on the device`
   อย่าใส่ `torch` ลง `requirements.txt`

2. **Windows + DataLoader** — โค้ดที่สร้าง DataLoader ต้องอยู่ใต้ `if __name__ == "__main__":`
   ไม่งั้น `num_workers > 0` จะ crash

3. **ค่า normalize ต้องถามจาก timm เสมอ** — อย่า hard-code mean/std
   ค่าเปลี่ยนตาม pretrained tag (timm 1.0.29: ทั้ง 3 ตัวใช้ ImageNet — `tf_efficientnetv2_b0.in1k`
   ไม่ได้ใช้ 0.5 อย่างที่เอกสารเก่าเขียน · ตรวจแล้ว 2026-09-19) → ใช้ `models.data_config(model)`
   ```python
   cfg = timm.data.resolve_model_data_config(model)
   mean, std = cfg["mean"], cfg["std"]
   ```

4. **ใช้ bfloat16 ไม่ใช่ float16** สำหรับ AMP — Blackwell รองรับเต็มที่ และไม่ต้องใช้ `GradScaler`

5. **ทุกอย่างที่เป็นค่าคงที่อยู่ใน `src/config.py`** — ห้าม hard-code path / ชื่อคลาส / hyperparameter กระจายตามไฟล์

---

## 7 คลาส (เรียงตามตัวอักษร = ตรงกับ `ImageFolder.class_to_idx`)

```
astrophytum, echinocactus, echinopsis, gymnocalycium,
mammillaria, opuntia, parodia
```

**เหตุผลที่เลือก + หลักฐาน + แผนสำรอง อยู่ที่ `docs/class_selection.md` — อ่านก่อนเปลี่ยนคลาส**
(เปลี่ยนจากชุดเดิม 2026-09-19: ตัด ariocarpus / lophophora / turbinicarpus ภาพเดิมยังอยู่ใน `data/raw/`)

- `echinocactus` = ถังทอง ดึงจาก iNat ด้วย *Kroenleinia grusonii* (iNat ย้ายสกุลแล้ว) — ดู `config.INAT_TAXON`
- `opuntia` = หูกระต่าย *O. microdasys* เท่านั้น
- ทุกคลาสมีภาพกระถาง ≥ 1,000 บน iNat → เป้า **400 ภาพ/คลาส ภาพกระถางล้วน สมดุล**
- คาดว่าจะสับสน: echinopsis ↔ mammillaria / gymnocalycium (pilot) — **เป็นผลการทดลอง ไม่ใช่บั๊ก**
- สำรองถ้าคลาสไหนพัง: cephalocereus → melocactus

---

## Data Pipeline

```
data/raw/{inaturalist,roboflow,handon}/<genus>/   ภาพดิบ ห้ามแก้
        ↓ dedup (ตอนลงทะเบียน) + QC label 10% (qc_status ใน metadata.csv)
        ↓ split.py: stratified ทีละคลาส + ทั้งต้นอยู่ split เดียว (hardlink จาก raw)
data/processed/{train,val,test,realworld_test}/<genus>/
```

**`data/metadata.csv` คือหัวใจของโปรเจค** — ทุกภาพมี 1 แถว
คอลัมน์: `image_id, filepath, label, source, source_url, license, attribution,
plant_id, phash, width, height, quality_grade, captive, n_agree, split, qc_status, notes`

### กฎเรื่องข้อมูลที่ห้ามผิด

1. **dedup ก่อน split เสมอ** — iNaturalist กับ Roboflow มีภาพซ้ำกันสูงมาก
   (Roboflow หลาย dataset scrape มาจาก iNat) ถ้าภาพซ้ำอยู่คนละ split = **data leakage**

2. **split ต้องแยกตาม `plant_id`** (ทำใน `split.py` มี guard ตรวจ leakage) — ทีมถ่ายภาพหลายมุมต่อ 1 ต้น
   ถ้าภาพต้นเดียวกันกระจายทั้ง train และ test = leakage, accuracy สูงเกินจริง
   - iNat: `plant_id = inat_obs_<observation_id>`
   - handon: มาจากชื่อโฟลเดอร์ย่อย (`astrophytum/plant001/`) หรือชื่อไฟล์ (`plant001__02.jpg`)

3. **`realworld_test/` โมเดลห้ามเห็นเด็ดขาด** — ภาพถ่ายเอง 200 ใบ (อีก 200 เข้า train)
   ใช้วัด generalization เทียบกับ test set ปกติ **คอลัมน์ Gap คือจุดขายของรายงาน**
   (งานวิจัยอ้างอิงมี gap 9.86% — ถ้าเราต่ำกว่าได้ = ทำได้ดีกว่า)

4. **iNaturalist: ต้นปลูกในกระถางไม่มีวันเป็น research grade** — จะโดนติดธง `captive=True`
   และลดชั้นเป็น `casual` เราต้องการภาพในกระถาง จึงใช้ `--quality-grade any --min-agreements 1`
   อย่าเปลี่ยนกลับไปกรอง research อย่างเดียวโดยไม่ถามก่อน

---

## โมเดล (เปรียบเทียบ 3 ตัว ใช้ hyperparameter เดียวกันทั้งหมด)

| key ใน `config.MODELS` | timm name | หมายเหตุ |
|---|---|---|
| `mobilenetv3` | `mobilenetv3_large_100` | เบา เหมาะมือถือ |
| `effnetv2b0` | `tf_efficientnetv2_b0` | สมดุลความแม่นยำ/ขนาด |
| `resnet50` | `resnet50` | baseline |

> deck เขียน "ResNet50V2" ซึ่งเป็นชื่อฝั่ง Keras — ในรายงานให้เขียน **"ResNet50"**

**เทรน 2 stage เสมอ:**
- Stage 1: freeze backbone, lr 1e-3, 10 epochs — ต้อง set `BatchNorm2d` เป็น `.eval()` ด้วย
- Stage 2: unfreeze, lr **3e-4 (backbone) / 3e-3 (head)**, 40 epochs, early stopping patience 8
  (จูนแล้ว — plan เดิม 1e-5/1e-4 ต่ำเกิน val F1 0.890 → 0.930 · ดู `docs/tuning.md` · จูนต้องดู val ห้ามดู test)

**เลือก best checkpoint จาก macro F1 ไม่ใช่ accuracy**
Loss ใช้ `CrossEntropyLoss(label_smoothing=0.1)` (ทำให้ค่า confidence น่าเชื่อถือขึ้น
ซึ่งจำเป็น เพราะเราใช้ threshold 0.60 ตัดสินว่า "ไม่แน่ใจ")

---

## สไตล์โค้ด

- Python 3.10+, type hints, `from __future__ import annotations`
- **คอมเมนต์เป็นภาษาไทย** โค้ด/ชื่อตัวแปรเป็นอังกฤษ — เจ้าของโปรเจคเป็นนักศึกษาไทย
- ใช้ `pathlib.Path` ไม่ใช่ `os.path` (โปรเจครันบน Windows)
- CSV เขียนด้วย `encoding="utf-8-sig"` (ให้ Excel บน Windows เปิดภาษาไทยได้)
- ทุกสคริปต์เป็น `python -m src.xxx` (absolute import จาก project root)
- สคริปต์ที่กินเวลานานต้อง **resume ได้** และเซฟ progress เป็นระยะ
- ตั้ง seed ทุกที่ (`config.SEED = 42`) — ต้องเทรนซ้ำแล้วได้ผลเดิม ไม่งั้นเขียนรายงานไม่ได้

---

## สถานะปัจจุบัน

**เสร็จแล้ว**
- `src/config.py` — ค่าคงที่ทั้งหมด
- `src/data/metadata.py` — MetadataStore (resume, ตรวจภาพซ้ำด้วย phash, สรุปสถิติ)
- `src/data/download_inat.py` — ดึงภาพจาก iNaturalist API
- `src/data/register_local.py` — ลงทะเบียนภาพ Roboflow / ถ่ายเอง
- `src/data/qc_report.py` — สุ่มตรวจ 10%/คลาส → `reports/qc/qc_sheet.{html,csv}` → `--apply` เขียนกลับ
  HTML ฝังรูปในไฟล์ (ส่ง LINE/Drive ได้ — data/raw ไม่ถูก commit) · `--render` สร้างหน้าตรวจซ้ำจาก CSV ที่กรอกแล้ว
  มี guard กันสุ่มชุดใหม่ทับ qc_sheet.csv ที่มีผลแล้ว
- `src/data/flag_suspects.py` — 5-fold out-of-fold prediction → flag ภาพที่โมเดลไม่เชื่อ label
  → `reports/qc/suspects.{csv,html}` ให้คนดูเฉพาะกลุ่มนี้ → `qc_report --apply reports/qc/suspects.csv`
- `src/data/split.py` — 70/15/15 แบ่งทีละคลาส + ทั้งต้นอยู่ split เดียว + handon ครึ่งหนึ่ง → realworld_test
  สร้าง `data/processed/` จาก raw โดยตรงด้วย hardlink (**ไม่มีขั้น `data/interim/`** — dedup ทำตอนลงทะเบียน, QC อยู่ใน metadata)
- `docs/class_selection.md` — เหตุผลเลือก 7 คลาส + pilot experiment + แผนสำรอง

- `src/datasets.py` — `CactusFolder` ล็อก class_to_idx ตาม config (split ไหนขาดคลาส index ไม่เลื่อน)
- `src/models.py` — build_model (**head zero-init** เพื่อให้ 3 โมเดลเริ่ม loss เท่ากัน), freeze/BN/param_groups
- `src/engine.py` — train/eval 1 epoch (bf16, channels_last), `set_seed` deterministic (ผลซ้ำได้เป๊ะ)
- `src/train.py` — 2 stage + best macro F1 + early stop → `checkpoints/<run>_best.pt`, `reports/logs/<run>.csv`
- `src/evaluate.py` — test/realworld → `reports/results.csv` + confusion matrix + predictions รายภาพ
- `src/visualize.py` — กราฟรายงาน (matplotlib + seaborn) → `reports/figures/`: dataset, tuning, learning_curves,
  model_comparison, per_class_f1, cm_<model>_all_seeds, confidence · สีตาม dataviz palette (ตรวจ CVD แล้ว)
  สีผูกกับโมเดล (`MODEL_COLOR`) — โมเดลเดียวกันสีเดียวกันทุกกราฟ · รันหลัง evaluate ทุกครั้ง
- `src/gradcam.py` — Grad-CAM บน test → `gradcam_<run>_{correct,wrong}.png` + `gradcam_compare.png` (3 โมเดล)
  ชั้นอยู่ใน `config.GRADCAM_LAYERS` · แสดงแบบ spotlight (หรี่ส่วนที่ไม่ใช้) ไม่ใช้ colormap jet
  ผล (seed 42): โมเดลดูที่ตัวต้น ไม่ใช่กระถาง/พื้นหลัง · แต่ **ดอกเป็น cue แรง** (ดอกเหลือง → astrophytum/parodia,
  ภาพดอกล้วนยังหลุด QC อยู่ใน test) · ภาพ "parodia" แถวเทียบ ทั้ง 3 โมเดลทาย echinopsis → น่าตรวจ label
- `src/predict.py` — `Predictor(best_run(model)).predict(load_image(path))` → สกุล + ชนิด (ชั้น 2) / "ไม่แน่ใจ" (< 0.60)
  + care rules + `explain()` (Grad-CAM) · ใช้ได้ทั้ง checkpoint flat/genus · `best_run` = seed ที่ val F1 ดีสุด (runs.csv)
- `app/app.py` — Streamlit: อัปโหลด (รับ HEIC) / ถ่ายภาพ (เปิดกล้องเมื่อกดเท่านั้น) · เลือกโมเดลทีละตัว หรือเทียบ
  `config.APP_MODELS` ทั้งหมด · Grad-CAM · ทดสอบในเบราว์เซอร์แล้ว · รัน `streamlit run app/app.py`
- `evaluate.py` (เขียนใหม่ 2026-10-07) — ทุก checkpoint ทุกงาน: ชั้นสกุล (อนุมานจาก leaf) + leaf acc/F1 + realworld + gap
  predictions มีคอลัมน์ `leaf_true`, `pred_class` · `compare.py` อ่านจาก results.csv (ไม่ทายซ้ำ) + McNemar + แนะนำ APP_MODELS
- run หลัก = `tasks.run_id(model, seed)` (งาน `tasks.MAIN_TASK = "flat"`) — visualize / gradcam / compare ใช้ร่วมกัน
- **เปิดภาพทุกที่ด้วย `datasets.load_image`** (หมุนตาม EXIF — รูปมือถือแนวตั้ง) · `models.load_checkpoint(path, device)`
- `engine.evaluate` ใช้ **fp32** (เทรนยัง bf16) → ความมั่นใจในรายงานตรงกับแอป
- `reports/runs.csv` มีคอลัมน์ `phase` (tuning = มี --tag, final = ไม่มี) — กราฟ tuning อ่านเฉพาะ tuning
- ตรวจโค้ด: `ruff check src/` (config ใน `pyproject.toml`) — ต้องผ่านก่อน commit

**ภาพที่ทีมรวบรวม (2026-10-07)** — `data/Cactus_1` = ถ่ายเองที่ฟาร์ม (iPhone HEIC) · `data/Cactus_2` = **ภาพจากเว็บ**
(มีลายน้ำ shutterstock/cactus-art ฯลฯ ไม่ใช่ภาพลงพื้นที่) → source `web` ไม่เข้า realworld_test
- Claude ตรวจทีละภาพ 965 ภาพ → `data/manifests/field_2026-10.csv` (label/subclass/ต้น/QC) → `python -m src.data.import_manifest`
- แก้ label: ในโฟลเดอร์ asterias มีหมวกแก๊ป 13 ภาพ · ขนนกทอง/ขนแมว = แมม other · Opuntia จากเว็บหลายภาพไม่ใช่หูกระต่าย
  · ไฟล์เดียวกันอยู่ 2 โฟลเดอร์สปีชีส์ (prolifera+vetula) → เก็บแค่ชั้นสกุล · ตัดภาพสต็อกติดลายน้ำ 8 + ภาพวาด 1
- นำเข้า handon 260 + web 504 (ซ้ำ/เล็ก/เสียถูกข้าม) · realworld_test = 116 ภาพ (ยังไม่มี echinopsis — ต้นน้อยเกิน)
- ต้น (plant) ของภาพถ่ายเองจัดกลุ่มจาก contact sheet แบบกว้างไว้ก่อน (กัน leakage) · checkpoint ก่อนนำเข้าอยู่ `checkpoints/pre_field/`

**ยังไม่มี**
- `src/export.py`
- **เนื้อหา** `care_rules/care_rules.json` — มีโครงแล้ว (name_th เท่านั้น) ทีมต้องเขียน + ใส่ reference ทุกสกุล
  สกุลที่ยังไม่ครบ predict/app จะไม่แสดงคำแนะนำ (กันแสดงครึ่ง ๆ / ไม่มีอ้างอิง)

**จูนเสร็จแล้ว (ก่อน QC, 2026-09-19)** — `docs/tuning.md`, ตัวเลขดิบ `reports/runs.csv`
- val macro F1 (seed 42, lr ใหม่): effnetv2b0 0.928 · mobilenetv3 0.909 · resnet50 0.907
- ต่อจากนี้จูนไม่คุ้ม: error ~ครึ่งหนึ่งเป็นภาพไม่ดี/label ผิด → **QC คือคันโยกถัดไป**
- checkpoint ช่วงจูนอยู่ `checkpoints/tuning/` (evaluate.py ไม่นับ) · 1 run ≈ 5 นาที
**ผลจริงหลัง QC (2026-09-19, 3 seeds, test 412 ภาพ):** mean test acc
EffNetV2-B0 **0.944** · MobileNetV3 0.923 · ResNet50 0.921 — ผ่านเป้า 91% ทุกโมเดล
(ยังไม่มี realworld_test — รอภาพถ่ายเอง) · อ่อนสุด: echinopsis / mammillaria · opuntia ~0.99

**ภาพ (2026-09-19):** iNat ภาพกระถาง 400/คลาส × 7 คลาส ครบแล้ว (2,800 ภาพ, captive 100%)
ยังไม่มี Roboflow / handon

**QC (2026-09-19, Claude ทำก่อน ทีมจะตรวจซ้ำ):** สุ่ม 10% = 280 ภาพ → label ผิด **0**,
ปลูกลงดิน 81 (29% — ถังทอง 73%), ใช้ไม่ได้ 20 (7%: ดอกล้วน/หลายสกุล/ต้นเล็ก)
- **ตัดสินใจ: `not_potted` ใช้เทรนได้** (อยู่ใน `QC_USABLE`) — ตัดเฉพาะ wrong_label/not_plant/multi_plant/blurry
- รอบ 2: `flag_suspects.py` (OOF acc 0.934) flag 286 ภาพ → Claude ตรวจ 250 ที่ยังไม่เคยตรวจ →
  wrong_label 3, ใช้ไม่ได้ 44, ลงดิน 48 · วิธีนี้จับภาพเสียจากรอบสุ่มได้ 11/20 (55%)
- รวม: ตรวจแล้ว 530/2,800 · ตัดทิ้ง 67 · split ใหม่ = 1,909 / 412 / 412 (266–278 train ต่อคลาส)
- รอบ 3 (2026-09-26, หลังเพิ่มภาพสปีชีส์): OOF acc 0.940 flag 398/4,797 → ตรวจ 228 ที่ยังไม่เคยตรวจ →
  wrong_label 2, ใช้ไม่ได้ 42, ลงดิน 66 · รวมตรวจแล้ว 758 · ตัดทิ้ง 111 · **ยังไม่ได้ split ใหม่**
- หน้าตรวจซ้ำสำหรับทีม: `reports/qc/qc_sheet.html`, `reports/qc/suspects_round1.html`, `reports/qc/suspects.html` (รอบ 3)
  (กรอบแดง = ไม่ผ่าน)
- `--only-captive` กรองฝั่ง API แล้ว (กันชนเพดาน 10,000 obs)
- ภาพโหลดผ่าน S3 dualstack endpoint (IPv4 ไป S3 ช้ามากจากเครื่องนี้) — ดู `to_large_url`

**ชั้น 2 (2026-09-20)** — คลาสย่อยใน `config.SUBCLASS_TAXA` / `SUBCLASSES` (20 leaf รวมสกุลที่ไม่มีชั้น 2)
- งานใน `src/tasks.py`: `genus` / `species-<สกุล>` / `flat` · `train.py --task` · เทียบด้วย `python -m src.hierarchy`
- `metadata.subclass` เติมด้วย `python -m src.data.subclass` · ยิมโน colored/variegated/normal Claude ติดเอง
  (`reports/qc/gymno_phenotype.html` รอทีมตรวจ) · ถั่วลิสง = iNat *Lobivia silvestrii* เข้าคลาส echinopsis
- ดึงเพิ่ม `download_inat --species all` / `--other` → รวม 5,068+ ภาพ · **ภาพใหม่ ~1,700 ยังไม่ QC**
- ผล seed 42 (test 722): A แยกชั้น genus 0.938 / leaf 0.862 · **B แบนราบ genus 0.942 / leaf 0.878**
  อ่อน: `*__other` (0.41–0.73), gymno variegated (26 ภาพ), echinopsis oxygona↔other
- checkpoint ชั้นสกุลเดิม (ก่อนเพิ่มภาพ) อยู่ `checkpoints/pre_species/` — genus ตอนนี้ไม่สมดุล (mamm 816 vs 275)

**เทียบ 6 โมเดล งาน flat (2026-09-27, หลัง QC รอบ 3, 3 seeds, test 716)** — `python -m src.compare --summary`
→ `reports/model_compare_mean.csv` · EffNetV2-B0 ดีสุด genus 0.934 / leaf 0.860 · ห่างตัวสุดท้ายแค่ ~2 จุด
ส่วนใหญ่ไม่ต่างอย่างมีนัยสำคัญ · `--balance weight/undersample` ไม่ช่วย (docs/tuning.md รอบ 6)
checkpoint ก่อน QC รอบ 3 อยู่ `checkpoints/pre_qc3/` · DataLoader val/test ใช้ 2 workers (กันชนเพดาน commit ของ Windows)

**ผลหลังนำเข้าภาพทีม (2026-10-07, flat, 3 seeds, test 794 / realworld 116)** — `reports/model_compare.csv`
EffNetV2-B0 ดีสุด: สกุล 0.939 · ชนิด 0.883 · ภาพจริง 0.888 · gap 5.1% · ทุกโมเดลผ่าน 91% · gap 5–9% (อ้างอิง 9.86%)
ภาพจริงชั้นชนิด 0.73 (ยาก) · `APP_MODELS` = effnetv2b0 / effnetb3 / resnet50 · Grad-CAM effnetb3 ใช้ `conv_head` (bn2 มี artifact มุมภาพ)
Grad-CAM: ภาพยิมโนที่มีดอกใหญ่ B3/ResNet50 ดูดอกแล้วทายผิดเป็น echinopsis — ดอกยังเป็น cue แรง

**ขอบเขต: ทาย 2 ชั้น สกุล → สปีชีส์** (เน้นนิยมในไทย แจ้งอาจารย์แล้ว)
- `metadata.csv` มีคอลัมน์ `species` แล้ว (ว่าง = ระบุได้แค่สกุล), เติมย้อนหลังด้วย `--backfill-species`
- ตัวเลือกสปีชีส์ + เรื่องที่ยังไม่ตัดสิน: `docs/class_selection.md` ข้อ 9

**Environment:** ใช้ `.venv` เท่านั้น (torch 2.11+cu128) — python global มี torch cu124 ใช้กับ GPU ไม่ได้

---

## วิธีทำงานกับโปรเจคนี้

1. อ่าน `plan.md` ก่อนเขียนโค้ดใหม่ ดูว่าไฟล์นั้นควรทำอะไร
2. เขียนทีละไฟล์ อย่าสร้างรวดเดียวหลายไฟล์แล้วส่งให้รัน
3. **ทดสอบทุกสคริปต์ที่เขียน** — สร้างภาพปลอมด้วย PIL + numpy ทดสอบ แล้วลบทิ้ง
   (เคยเจอบั๊ก `image_id` ชนกันด้วยวิธีนี้มาแล้ว)
4. dataset เล็ก (2,800 ภาพ) + GPU แรง → **เทรนใช้เวลาไม่กี่นาที**
   Bottleneck คือคุณภาพข้อมูล ไม่ใช่การเทรน อย่าเสียเวลาจูน hyperparameter ก่อนข้อมูลสะอาด
5. ถ้าเจอจุดที่ deck/plan ขัดแย้งกันเอง หรือเจอ trade-off ที่ไม่มีคำตอบชัด → **ถามก่อน อย่าเดา**
