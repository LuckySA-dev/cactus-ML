# 🌵 Cactus Classifier

จำแนก **สกุลและชนิดกระบองเพชรแคระ** ที่นิยมในไทยจากภาพถ่าย แล้วแนะนำวิธีดูแล
งานวิชา Machine Learning · มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (KMUTNB)

- **ชั้น 1 — สกุล (7):** Astrophytum · Echinocactus (ถังทอง) · Echinopsis · Gymnocalycium · Mammillaria · Opuntia (หูกระต่าย) · Parodia
- **ชั้น 2 — ชนิด/ลักษณะ:** เช่น แอสโตรดาว / หมวกแก๊ป / ออร์นาตัม, ถั่วลิสง, แมมขนนก / ก้างปลา, ยิมโนหัวสี / ด่าง / ปกติ (รวม 20 คลาส)
- **วิธีดูแล:** rule-based จาก [`care_rules/care_rules.json`](care_rules/care_rules.json) — ไม่ใช่ ML และแสดงเฉพาะเมื่อโมเดลมั่นใจ ≥ 60%

## ผลลัพธ์

6 สถาปัตยกรรม × 3 seeds · สูตรเทรนเดียวกันทั้งหมด (2 stage + EMA + augment แรง) · ข้อมูลผ่าน QC ด้วยตาครบทุกภาพ ·
test 750 ภาพ · realworld_test 107 ภาพ (ภาพถ่ายเองที่โมเดลไม่เคยเห็น)

| โมเดล | สกุล · test | ชนิด · test | สกุล · ภาพจริง | Gap | ขนาด | CPU/ภาพ |
|---|---|---|---|---|---|---|
| **DenseNet-121** | 95.8 ± 0.3% | **91.0 ± 0.5%** | 84.4 ± 1.9% | 11.3% | 7.0M | 70 ms |
| **EfficientNet-B3** | **95.9 ± 0.3%** | 90.4 ± 0.4% | **86.6 ± 2.2%** | 9.3% | 10.7M | 49 ms |
| **EfficientNetV2-B0** | 95.1 ± 0.2% | 89.8 ± 0.5% | 84.7 ± 0.5% | 10.4% | 5.9M | 34 ms |
| MobileNetV3-L | 94.6 ± 0.1% | 89.0 ± 0.9% | 83.8 ± 2.4% | 10.8% | 4.2M | 18 ms |
| ResNet34 | 93.4 ± 0.7% | 88.0 ± 0.7% | 84.7 ± 1.4% | 8.7% | 21.3M | 30 ms |
| ResNet50 | 93.5 ± 0.5% | 87.7 ± 0.4% | 82.2 ± 1.6% | 11.3% | 23.5M | 59 ms |

- ผ่านเป้า **test accuracy ≥ 91%** ทุกโมเดล · Gap (test − ภาพจริง) 8.7–11.3% — EfficientNet-B3 และ ResNet34
  ต่ำกว่างานวิจัยอ้างอิง (9.86%) ส่วนอีก 4 ตัวสูงกว่าเล็กน้อย
- QC ครบทุกภาพ (2026-10-24, ตัดภาพดอกบังต้น/หลายสกุล/เบลอ/label ผิด 297 ภาพ) ทำให้ test สะอาดขึ้น → test ขึ้น
  ภาพจริงลด 1–3 จุด (realworld สุ่มชุดใหม่ 107 ภาพ แกว่งได้ ~±3 จุด) → gap กว้างขึ้น · ผลก่อน QC อยู่ `reports/pre_qc5/`
- ภาพจริงชั้นชนิดยังยากกว่ามาก (69–72%)
- แอป/API ใช้ 3 อันดับแรก (ชนิด · test): DenseNet-121, EfficientNet-B3, EfficientNetV2-B0 (seed ที่ val ดีสุด)
- ความเร็ว CPU วัดตอนเครื่องมีงานอื่นรันอยู่ ใช้เทียบกันเองได้ ไม่ใช่ค่าสัมบูรณ์
- ตารางเต็ม `reports/model_compare.csv` · กราฟ `reports/figures/` (สร้างด้วย `python -m src.visualize`)

## โครงสร้าง

```
src/
  config.py            ค่าคงที่ทั้งหมด (คลาส, path, hyperparameter)
  data/                เก็บ + ทำความสะอาดข้อมูล
    download_inat.py     ดึงภาพจาก iNaturalist (ภาพกระถาง, captive)
    import_manifest.py   นำภาพที่ทีมรวบรวมเข้าตาม manifest ที่ตรวจแล้ว (label / ชนิด / ต้น / QC รายภาพ)
    register_local.py    ลงทะเบียนภาพ Roboflow / ภาพถ่ายเองที่จัดโฟลเดอร์แล้ว
    qc_report.py         สุ่มตรวจคุณภาพ → หน้า HTML ให้ทีมตรวจซ้ำ
    flag_suspects.py     ให้โมเดล (5-fold out-of-fold) คัดภาพน่าสงสัยมาให้คนดู
    subclass.py          เติมคลาสชั้น 2
    split.py             แบ่ง train/val/test/realworld_test ตามต้น (กัน leakage)
  tasks.py             งานที่เทรนได้: genus / species-<สกุล> / flat (งานหลัก)
  models.py · datasets.py · engine.py · train.py     เทรน 2 stage (freeze → fine-tune)
  evaluate.py          test + realworld_test → reports/results.csv
  compare.py           เทียบโมเดล mean ± SD ข้าม seed + McNemar
  visualize.py · gradcam.py                          กราฟรายงาน + Grad-CAM
  predict.py           ภาพ → สกุล + ชนิด + คำแนะนำ (CLI / ใช้ในแอป)
app/                   เว็บแอป Streamlit: app.py (เมนู) · predict_page.py · eval_page.py · ธีมใน .streamlit/
api/main.py            REST API (FastAPI) สำหรับหน้าเว็บ frontend
tests/                 ทดสอบ API ด้วยภาพปลอม (pytest)
docs/                  เหตุผลการเลือกคลาส, บันทึกการจูน
data/metadata.csv      ทุกภาพ 1 แถว: แหล่งที่มา, license, ต้น, QC, split
data/manifests/        ผลตรวจภาพที่ทีมรวบรวมเอง (รายภาพ)
```

## ติดตั้ง

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> GPU รุ่น Blackwell (RTX 50xx) ต้องใช้ PyTorch build CUDA 12.8 ขึ้นไป · เครื่องที่ไม่มี GPU ใช้แอป/ทำนายบน CPU ได้

## ใช้งาน

```bash
streamlit run app/app.py                       # เว็บแอป 2 หน้า: ทายชื่อ + วิธีดูแล · ประเมินผลโมเดล (รันจาก project root)
uvicorn api.main:app --port 8000               # REST API สำหรับหน้าเว็บ — เอกสาร http://localhost:8000/docs
python -m src.predict photo.jpg                # ทำนายจาก command line
pytest -q                                      # ทดสอบ API
```

| API | ทำอะไร |
|---|---|
| `GET /api/health` | เช็กว่าเซิร์ฟเวอร์พร้อม |
| `GET /api/models` | 3 โมเดลที่ใช้ได้ + คะแนนเฉลี่ย |
| `POST /api/predict?model=effnetv2b0&gradcam=false` | อัปโหลดภาพ (field `file`) → สกุล + ชนิด + ความมั่นใจ + วิธีดูแล (+ Grad-CAM) |
| `POST /api/compare?gradcam=false` | ภาพเดียว ทายด้วยทั้ง 3 โมเดล |

frontend ที่รันคนละ origin ต้องตั้ง `CORS_ORIGINS` (ค่าเริ่มต้น `http://localhost:5173,http://localhost:3000`)

### Deploy บน Streamlit Community Cloud

1. push repo ขึ้น GitHub (มีโมเดล 3 ตัวผ่าน Git LFS แล้ว)
2. [share.streamlit.io](https://share.streamlit.io) → **Create app** → เลือก repo / branch `main`
3. **Main file path:** `app/app.py` · **Advanced settings → Python:** 3.12
4. Deploy — Cloud ติดตั้งจาก [`app/requirements.txt`](app/requirements.txt) (torch build CPU, เวอร์ชันล็อกไว้แล้ว)
   ส่วน [`requirements.txt`](requirements.txt) ที่ root ใช้กับเครื่องเทรนเท่านั้น

ครั้งแรกใช้เวลาติดตั้ง ~5 นาที · ธีมอ่านจาก `.streamlit/config.toml` · ภาพที่ผู้ใช้อัปโหลดไม่ถูกเก็บ
หน้า "ประเมินผลโมเดล" ทำงานได้โดยไม่มีภาพในเครื่อง (ใช้ `reports/` + `data/metadata.csv`) ยกเว้นแกลเลอรีภาพที่ทายผิด

repo มีโมเดล 3 ตัวที่แอปใช้อยู่แล้วใน `checkpoints/` (เก็บด้วย [Git LFS](https://git-lfs.com) — ติดตั้ง LFS ก่อน clone
ไม่งั้นได้แค่ไฟล์ pointer) · checkpoint ของ seed อื่น / โมเดลอื่นเทรนเองได้ตามขั้นตอนด้านล่าง

## ทำซ้ำผลการทดลอง

```bash
set INAT_CONTACT=you@example.com                        # iNaturalist ขอให้ระบุอีเมลติดต่อใน User-Agent
python -m src.data.download_inat --species all          # 1. ภาพ iNaturalist
python -m src.data.import_manifest data/manifests/field_2026-10.csv   # 2. ภาพที่ทีมรวบรวม
python -m src.data.split                                # 3. แบ่งชุดข้อมูล
python -m src.train --model effnetv2b0 --task flat --seed 42          # 4. เทรน (ทำทุกโมเดล × seed 42/43/44)
python -m src.evaluate                                  # 5. ประเมิน test + realworld_test
python -m src.compare                                   # 6. สรุปเทียบโมเดล
python -m src.visualize && python -m src.gradcam        # 7. กราฟ + Grad-CAM
```

ทุกขั้นตั้ง seed (ผลซ้ำได้) · เลือก checkpoint และจูนด้วย val เท่านั้น ไม่ได้ดู test

## ข้อมูล

| แหล่ง | ใช้ทำอะไร |
|---|---|
| iNaturalist (ภาพต้นปลูก, CC license ตามแต่ละภาพใน `metadata.csv`) | train / val / test |
| ภาพจากเว็บที่ทีมรวบรวม (ตรวจรายภาพ, ตัดภาพติดลายน้ำสต็อก) | train / val / test |
| ภาพถ่ายเองที่ฟาร์ม | ครึ่งหนึ่ง (นับเป็นต้น) → **realworld_test** ที่โมเดลไม่เคยเห็น · อีกครึ่งเข้าชุดปกติ |

ภาพไม่ได้อยู่ใน git (ขนาด + ลิขสิทธิ์ของเจ้าของภาพ) — `metadata.csv` เก็บที่มาและ license ของทุกภาพ
