# 🌵 Cactus Classifier

จำแนก **สกุลและชนิดกระบองเพชรแคระ** ที่นิยมในไทยจากภาพถ่าย แล้วแนะนำวิธีดูแล
งานวิชา Machine Learning · มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (KMUTNB)

- **ชั้น 1 — สกุล (7):** Astrophytum · Echinocactus (ถังทอง) · Echinopsis · Gymnocalycium · Mammillaria · Opuntia (หูกระต่าย) · Parodia
- **ชั้น 2 — ชนิด/ลักษณะ:** เช่น แอสโตรดาว / หมวกแก๊ป / ออร์นาตัม, ถั่วลิสง, แมมขนนก / ก้างปลา, ยิมโนหัวสี / ด่าง / ปกติ (รวม 20 คลาส)
- **วิธีดูแล:** rule-based จาก [`care_rules/care_rules.json`](care_rules/care_rules.json) — ไม่ใช่ ML และแสดงเฉพาะเมื่อโมเดลมั่นใจ ≥ 60%

## ผลลัพธ์

6 สถาปัตยกรรม × 3 seeds · hyperparameter เดียวกันทั้งหมด · test 794 ภาพ · realworld_test 116 ภาพ (ภาพถ่ายเองที่โมเดลไม่เคยเห็น)

| โมเดล | สกุล · test | ชนิด · test | สกุล · ภาพจริง | Gap | ขนาด | CPU/ภาพ |
|---|---|---|---|---|---|---|
| **EfficientNetV2-B0** | **93.9 ± 0.3%** | **88.3 ± 0.4%** | **88.8 ± 0.9%** | 5.1% | 5.9M | 15 ms |
| EfficientNet-B3 | 93.7 ± 0.4% | 87.3 ± 0.4% | 86.5 ± 3.3% | 7.2% | 10.7M | 23 ms |
| ResNet50 | 93.2 ± 0.6% | 86.8 ± 0.6% | 87.1 ± 2.6% | 6.2% | 23.5M | 30 ms |
| MobileNetV3-L | 93.2 ± 0.4% | 86.7 ± 1.1% | 84.2 ± 1.8% | 9.0% | 4.2M | 8 ms |
| DenseNet-121 | 93.1 ± 0.6% | 86.2 ± 0.9% | 87.4 ± 1.0% | 5.8% | 7.0M | 34 ms |
| ResNet34 | 93.0 ± 0.5% | 86.0 ± 0.6% | 88.2 ± 1.3% | 4.8% | 21.3M | 23 ms |

- ผ่านเป้า **test accuracy ≥ 91%** ทุกโมเดล · Gap (test − ภาพจริง) 5–9% ต่ำกว่างานวิจัยอ้างอิง (9.86%)
- ภาพจริงชั้นชนิดยังยากกว่ามาก (EfficientNetV2-B0 73%) — realworld_test มีแค่ 116 ภาพ ตัวเลขแกว่งตาม seed
- แอปใช้ 3 อันดับแรก (ชนิด · test): EfficientNetV2-B0, EfficientNet-B3, ResNet50
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
app/app.py             เว็บแอป Streamlit
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
streamlit run app/app.py                       # เว็บแอป: อัปโหลด/ถ่ายภาพ เลือกโมเดล หรือเทียบทุกโมเดล
python -m src.predict photo.jpg                # ทำนายจาก command line
```

แอปต้องมี checkpoint ใน `checkpoints/` (ไม่ได้อยู่ใน git เพราะไฟล์ใหญ่) — เทรนเองตามขั้นตอนด้านล่าง

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
