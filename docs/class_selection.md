# การเลือก 7 คลาส — บันทึกการตัดสินใจ (Decision Record)

> วันที่ตัดสินใจ: 2026-09-19 · ใช้อ้างอิงในรายงาน และเป็น **แผนสำรอง** ถ้าคลาสไหนมีปัญหาภายหลัง
> ตัวเลขทั้งหมดดึงจาก iNaturalist API / Roboflow / Wikimedia Commons ณ วันที่ 2026-09-18–19

---

## 1. สรุปผล

| # | label (โฟลเดอร์) | ชื่อไทย | taxon ที่ใช้ดึงจาก iNat | หมายเหตุ |
|---|---|---|---|---|
| 1 | `astrophytum` | แอสโตร | *Astrophytum* (genus) | |
| 2 | `echinocactus` | ถังทอง | *Kroenleinia grusonii* (species) | iNat ย้ายถังทองไปสกุล *Kroenleinia* แล้ว (2014) ในวงการไทยยังเรียก *Echinocactus grusonii* |
| 3 | `echinopsis` | อิชินอปซิส | *Echinopsis* (genus) | iNat แยก *Lobivia* ออก → ถั่วลิสง (*E. chamaecereus* = *Lobivia silvestrii*) ไม่อยู่ในคลาสนี้ |
| 4 | `gymnocalycium` | ยิมโน | *Gymnocalycium* (genus) | |
| 5 | `mammillaria` | แมม | *Mammillaria* (genus) | |
| 6 | `opuntia` | หูกระต่าย | *Opuntia microdasys* (species) | ใช้แค่หูกระต่าย ไม่เอา Opuntia ใบพายต้นใหญ่ |
| 7 | `parodia` | โนโต / Yellow tower | *Parodia* (genus) | |

**คุณสมบัติสำคัญของชุดนี้:** ทุกคลาสมีภาพ **ต้นในกระถาง** ที่ผ่านเกณฑ์ ≥ 1,000 ภาพบน iNat
→ ทำ **400 ภาพ/คลาส เป็นภาพกระถางล้วน และสมดุลทุกคลาส** ได้จาก iNat อย่างเดียว เหลือเผื่อ QC
→ ภาพธรรมชาติ (ไม่ใช่กระถาง) ไม่จำเป็นต้องใช้ เก็บไว้เป็นการทดลองเสริมได้ (ข้อ 8)

---

## 2. ลำดับการตัดสินใจ

1. **แผนเดิม** 7 คลาส: ariocarpus, astrophytum, echinopsis, gymnocalycium, lophophora, mammillaria, turbinicarpus
2. ดาวน์โหลด iNat 200/สกุลได้ครบ แต่พบว่า **สัดส่วนภาพกระถางต่างกันมาก** (Ariocarpus 8% vs Astrophytum 58%)
   → เสี่ยงโมเดลจำฉากหลังแทนต้นไม้ (เห็นหิน = Ariocarpus)
3. ขอบเขตเปลี่ยนเป็น **ทาย 2 ชั้น สกุล → สปีชีส์ เน้นสปีชีส์ที่นิยมในไทย** (แจ้งอาจารย์แล้ว)
4. ทีมเสนอรายชื่อไม้ยอดนิยมในไทย → คัดเหลือ **13 สกุลตัวเลือก**
5. ทีมตัดสินใจ: **คงไว้ 7 คลาส ตัดตัวที่หาข้อมูลยาก เรื่องสปีชีส์ค่อยทำทีหลัง**
6. ทำการวิจัยเชิงลึก + pilot experiment → ได้ 7 คลาสสุดท้าย (ข้อ 1)

---

## 3. เกณฑ์ที่ใช้เลือก

1. **ข้อมูลพอจริง** — ภาพกระถางที่ผ่านเกณฑ์ `download_inat.py` (license CC + research grade หรือมีคนยืนยัน ID ≥ 1) ต้อง ≥ 400
2. **นิยมในไทย** — observation ในไทยบน iNat, บทความไทย 5 แหล่ง, งานวิจัยอ้างอิง [1] (ร้านในโคราช)
3. **แยกออกจากกันได้** — ผล pilot experiment (ข้อ 5)
4. **หาถ่ายเองได้** — มีขายทั่วไปตามร้านไทย ไม่ผิดกฎหมาย

---

## 4. หลักฐาน

### 4.1 จำนวนภาพ + ความนิยม

| สกุล | ภาพกระถางใช้ได้ (iNat) | captive obs ทั้งหมด | obs ในไทย | บทความไทย (5) | งานวิจัย [1] | Roboflow | Commons |
|---|---|---|---|---|---|---|---|
| Astrophytum | ≥ 1,000 | 1,294 | 64 | 5/5 | ✅ | ✅✅ (ไทยหลายชุด) | 614 |
| Mammillaria | ≥ 1,000 | 10,468 | 126 | 5/5 | ✅ 7 ใน 10 คลาส | ✅✅ | 2,936 |
| Gymnocalycium | ≥ 1,000 | 2,332 | 138 | 4/5 | ✅ | ✅✅ | 1,122 |
| Echinocactus (ถังทอง) | ≥ 1,000 | 7,874 | 76 | 4/5 | – | ⚪ | 352 |
| Opuntia microdasys | ≥ 1,000 | 4,261 | 30 | 2/5 | – | ⚪ | 98 |
| Echinopsis | ≥ 1,000 | 4,522 | 28 | 1/5 | – | ✅ | 1,927 |
| Parodia | ≥ 1,000 | 1,683 | 10 | 1/5 | ✅ "Yellow tower" | ❌ | 544 |
| Cephalocereus senilis | **262** | 286 | 0 | 2/5 | – | ⚪ | 90 |
| Melocactus | **316** | 634 | 13 | 0/5 | – | ❌ | 764 |
| Lophophora | **391** | 499 | 7 | 1/5 | – | ✅ | 172 |
| Ferocactus | ≥ 1,000 | 4,609 | 6 | 0/5 | – | ⚪ | 1,752 |
| Ariocarpus | **170** | 207 | 1 | 1/5 | – | ❌ | 179 |
| Turbinicarpus | **104** | 146 | 1 | 2/5 | – | ❌ | 177 |

- "ภาพกระถางใช้ได้" = นับทุก observation จริง (captive=true กรองฝั่ง API) หยุดนับที่ 1,000
- ⚠️ `captive=True` บน iNat = "ปลูกโดยคน" ไม่ใช่ "อยู่ในกระถาง" เสมอไป (บางภาพเป็นต้นลงดินในสวน) → QC จะตัดออก คาดว่าหาย 10–30%
- บทความไทยที่ใช้นับ: dtac, HOMEDAY, Kapook, Sale Here, Time Out Bangkok (ลิงก์ในข้อ 10)
- Roboflow: ✅✅ = หลาย dataset หลักร้อย–พัน (ส่วนใหญ่คนไทยสร้าง), ✅ = มีชุดขนาดกลาง, ⚪ = เป็นคลาสย่อยไม่กี่สิบภาพ, ❌ = ไม่มี
  ⚠️ จำนวนภาพบน Roboflow อาจนับรวมภาพ augment แล้ว (มัก ×3) และหลายชุดใช้ภาพซ้ำกัน
- Commons = จำนวนไฟล์ภาพที่มีชื่อสกุลในชื่อไฟล์ (ปนภาพ herbarium / ภาพวาด)

### 4.2 แหล่งข้อมูลที่ประเมินแล้ว

| แหล่ง | ผล | เหตุผล |
|---|---|---|
| **iNaturalist** | ✅ แหล่งหลัก | license ชัด, มี `captive`, มี observation id ใช้เป็น `plant_id` ได้ |
| **Roboflow Universe** | ✅ แหล่งเสริม | มี dataset ไทย CC BY 4.0 เช่น [s Workspace 6k](https://universe.roboflow.com/s-workspace-slhvc/cactus-z5mbu), [Cactus Classification 10 สกุล](https://universe.roboflow.com/cactus-iobh1/cactus-g94d9), [มหิดลวิทยานุสรณ์](https://universe.roboflow.com/mahidol-wittayanusorn-school-ogdkd/cactus-detection-2fm9p) · ส่วนใหญ่เป็น object detection → ต้อง crop ตาม bbox · ต้องใช้บัญชี Roboflow ของทีมดาวน์โหลดเอง |
| Wikimedia Commons | ⚪ สำรอง | license ชัด แต่ปนภาพ herbarium/ภาพวาด ต้องคัดมือ |
| Kaggle | ❌ | มีชุดเดียวที่เกี่ยว ([emredo77/cactus](https://www.kaggle.com/datasets/emredo77/cactus) ~60 ภาพ/สกุล 3 สกุล) และ license ไม่ระบุ · อีกชุดเป็นไฟล์เดียวกันอัปซ้ำ |
| GBIF | ❌ | ภาพเกือบทั้งหมดคือ iNat ชุดเดิม ที่เหลือเป็น herbarium (ต้นแห้ง) |
| ถ่ายเอง (handon) | ✅ จำเป็น | ใช้ทำ `realworld_test` + เพิ่ม train |

---

## 5. Pilot experiment (ยืนยันว่าคลาสไหนแยกยาก)

**วิธี:** ภาพกระถาง 100 ภาพ/สกุล × 13 สกุล (จาก iNat, 1 ภาพ/observation) → ดึง feature ด้วยโมเดล pretrained
ที่แช่แข็งไว้ (ResNet50 และ EfficientNetV2-B0, normalize ตาม timm) → Logistic Regression + StandardScaler
→ 5-fold Stratified CV (seed 42)

> ⚠️ นี่คือ linear probe (ไม่ fine-tune) ตัวเลขจึงต่ำกว่าผลจริงมาก ใช้ **เทียบกัน** เท่านั้น
> n = 100/คลาส → ความคลาดเคลื่อนราว ±3% · label ยังไม่ผ่าน QC

### ชุด 7 คลาสแบบต่าง ๆ (accuracy: ResNet50 / EffNetV2-B0)

| ชุด | acc |
|---|---|
| เดิม (มี ario/lopho/turb) | 0.701 / 0.673 ← แย่สุด |
| **ที่เลือก (+parodia)** | **0.729 / 0.754** |
| แทนด้วย melocactus | 0.733 / 0.747 |
| แทนด้วย lophophora | 0.746 / 0.754 |
| แทนด้วย ferocactus | 0.697 / 0.744 |
| แทนด้วย cephalocereus | 0.766 / 0.779 ← แยกง่ายสุด |

### คู่ที่สับสนมากที่สุด (13 คลาส, อัตราผิดไป-กลับรวม)

| ResNet50 | EffNetV2-B0 |
|---|---|
| ariocarpus ↔ turbinicarpus **0.24** | ariocarpus ↔ turbinicarpus **0.24** |
| echinopsis ↔ mammillaria 0.19 | echinopsis ↔ gymnocalycium 0.17 |
| echinocactus ↔ ferocactus **0.19** | mammillaria ↔ turbinicarpus 0.16 |
| gymnocalycium ↔ turbinicarpus 0.13 | ariocarpus ↔ astrophytum 0.15 |
| echinopsis ↔ ferocactus 0.12 | echinocactus ↔ ferocactus **0.15** |

**recall รายคลาส (EffNetV2-B0):** mammillaria 0.50, astrophytum 0.54, gymnocalycium 0.60, ariocarpus 0.65,
echinopsis 0.65, turbinicarpus 0.68, ferocactus 0.71, lophophora 0.74, melocactus 0.74, parodia 0.80,
echinocactus 0.86, opuntia 0.89, cephalocereus 0.91

**สิ่งที่ pilot ยืนยันชัด:** (1) ชุดเดิมแย่ที่สุด (2) ariocarpus↔turbinicarpus สับสนกันมากสุด (3) ถังทอง↔ferocactus สับสนกันจริง (4) parodia ไม่ได้สับสนกับ mammillaria รุนแรง
**สิ่งที่ pilot แยกไม่ออก:** parodia / melocactus / lophophora (ต่างกันไม่เกิน noise)

---

## 6. เหตุผลที่เลือกแต่ละคลาส

| คลาส | เหตุผล |
|---|---|
| **astrophytum** | บทความไทย 5/5, อยู่ใน [1], dataset ไทยบน Roboflow หลายชุด, ทำชั้นสปีชีส์ได้ (asterias, myriostigma, capricorne, ornatum) |
| **echinocactus** | obs ในไทยอันดับ 3 ของไม้แคระ, บทความ 4/5, ภาพกระถางมากสุด (7,874), pilot recall 0.85–0.86 |
| **echinopsis** | อยู่ในรายชื่อหลักของทีม, ภาพกระถางมาก, ดาวน์โหลดไว้แล้ว · ⚠️ จุดอ่อน: บทความไทยแค่ 1/5 และสับสนกับ mammillaria/gymnocalycium (ลูกผสมหลากหลาย) |
| **gymnocalycium** | obs ในไทยมากสุดในกลุ่มไม้แคระ, อยู่ใน [1], dataset ไทยบน Roboflow แยกสปีชีส์ไว้แล้ว |
| **mammillaria** | บทความไทย 5/5, 7 ใน 10 คลาสของ [1] เป็นแมม · ⚠️ recall ต่ำสุดใน pilot เพราะมี 144 สปีชีส์ หน้าตาหลากหลาย → ต้องเก็บภาพให้ครอบคลุมหลายสปีชีส์ |
| **opuntia** | ไม้มือใหม่ยอดนิยม, แยกง่ายสุด (recall ~0.9) ช่วยดึง accuracy รวม |
| **parodia** | [1] สำรวจร้านไทยแล้วพบว่า "Yellow tower" (*Parodia*) ขายดี, ภาพกระถาง ≥ 1,000, ไม่มีคู่สับสนรุนแรง, ทำให้ทุกคลาสมีภาพกระถาง ≥ 1,000 |

## 7. เหตุผลที่ไม่เลือก (6 จาก 13)

| สกุล | เหตุผลหลัก | สถานะ |
|---|---|---|
| **Ariocarpus** | ภาพกระถางทั้งโลกแค่ 170, obs ในไทย 1, Roboflow ไม่มี, สับสนกับ turbinicarpus มากสุด (0.24) | ❌ ตัด |
| **Turbinicarpus** | ภาพกระถางน้อยสุด 104, obs ในไทย 1, สับสนกับ ariocarpus 0.24 และ mammillaria 0.16 | ❌ ตัด |
| **Ferocactus** | สับสนกับถังทองจริง (0.15–0.19), บทความไทย 0/5 (Time Out เขียนว่า Ferocactus แต่ภาพคือถังทอง), บน iNat 90% เป็นภาพธรรมชาติ | ❌ ตัด |
| **Lophophora** | สาร mescaline เป็น **วัตถุออกฤทธิ์ประเภท 1** ตามกฎหมายไทย → ซื้อ/ถ่ายตามร้านเสี่ยง และไม่เหมาะกับแอปแนะนำการปลูก · ภาพกระถาง 391 < 400 | ❌ ตัด |
| **Melocactus** | ภาพกระถางทั้งหมด 316 (สแกนครบแล้ว) หลัง QC เหลือ ~250, บทความไทย 0/5, Roboflow ไม่มี, ชั้นสปีชีส์ทำไม่ได้ | 🔁 **สำรองอันดับ 2** |
| **Cephalocereus** | ภาพกระถางทั้งหมด 262 หลัง QC เหลือ ~200 → ไม่สมดุลกับคลาสอื่น, สปีชีส์เดียว, obs ในไทย 0 · แต่ **แยกง่ายสุดใน pilot** | 🔁 **สำรองอันดับ 1** |

ข้อมูลของ ariocarpus / lophophora / turbinicarpus ที่ดาวน์โหลดไปแล้ว (600 ภาพ) **ยังอยู่ใน `data/raw/` และ `metadata.csv`**
ไม่ลบ (กฎ raw ห้ามแก้) — ขั้น split จะกรองเฉพาะ label ที่อยู่ใน `config.CLASSES`

---

## 8. แผนสำรอง

| ถ้าเกิด... | ทำแบบนี้ |
|---|---|
| คลาสไหนหลัง QC เหลือ < 300 ภาพกระถาง | เติมด้วยภาพธรรมชาติ / Roboflow / ถ่ายเอง ก่อน ถ้ายังไม่พอ → สลับเป็น **Cephalocereus** แล้ว **Melocactus** |
| echinopsis สับสนกับแมม/ยิมโนมากเกินไปในผลจริง | สลับเป็น **Cephalocereus** (ยอมคลาสนี้ ~200 ภาพ + `WeightedRandomSampler`) |
| mammillaria recall ต่ำ | เก็บภาพให้ครอบคลุมหลายสปีชีส์ (ดูข้อ 9) ก่อนคิดเรื่องโมเดล |
| iNat โหลดภาพช้ามาก | ใช้ host `inaturalist-open-data.s3.dualstack.us-east-1.amazonaws.com` (เส้นทาง IPv4 ไป S3 ช้ามาก ~8 KB/s เมื่อ 2026-09-19) |
| อยากเพิ่มคลาสทีหลัง | เรียงตามความพร้อม: Cephalocereus → Melocactus → Ferocactus (ระวังสับสนกับถังทอง) |

**การทดลองเสริมที่แนะนำ (ไม่มีในงานวิจัยอ้างอิงทั้ง 3):** เทรน A = ภาพกระถางล้วน vs B = กระถาง + ธรรมชาติ
แล้ววัดบน `realworld_test` → ตอบได้ว่าภาพที่ไม่ใช่กระถางช่วยหรือทำให้แย่ลง

---

## 9. เรื่องที่ยังไม่ตัดสิน (ทำหลังโมเดลชั้นสกุลเสร็จ)

- **ชั้นสปีชีส์** — ตัวเลือกที่ข้อมูลพอ (ภาพกระถางบน iNat):
  - astrophytum: asterias, myriostigma, capricorne, ornatum
  - mammillaria: elongata, plumosa, prolifera, vetula (+ bocasana, hahniana ต้องถ่ายเสริม) — [1] ใช้ carmenae, bocasana, hahniana ("oldman"), elongata ("lady finger") ด้วย
  - gymnocalycium: mihanovichii (หัวสี/ด่าง/LB2178 ไม่ใช่สปีชีส์ ต้องติด label เองจากหน้าตา) — Roboflow มี baldianum, damsii, ragonesei ของคนไทย
  - parodia: magnifica, lenninghausii
  - echinocactus / opuntia: สปีชีส์เดียว → ไม่มีชั้นสปีชีส์
- คลาส "อื่นๆ" ต่อสกุล (ภาพที่ระบุได้แค่สกุล / สปีชีส์นอกรายชื่อ)
- วิธีจดสปีชีส์ของภาพถ่ายเอง (เสนอ: `astrophytum/asterias__plant001/`)

---

## 10. อ้างอิง

**งานวิจัย**
1. จักรินทร์ สันติรัตนภักดี, ณิชาภัทร ตุลาธาร (2567). การจำแนกสายพันธุ์กระบองเพชรด้วยโมเดลการเรียนรู้เชิงลึก. *วารสารศรีปทุมปริทัศน์ ฉบับวิทยาศาสตร์และเทคโนโลยี*, 16, 73–94. — 10 สายพันธุ์จากร้านในโคราช 4,512 ภาพ, MobileNetV3 91.36% (val), ภาพใหม่ 400 ภาพตรวจโดยผู้เชี่ยวชาญ 81.50% (gap 9.86%) · แบ่งข้อมูลแบบสุ่ม 80/20 ไม่ได้แยกตามต้น
2. Valuvanathorn, S. & Supaartagorn, C. (2026). Comparison of CNN Architectures for Thai Medicinal Plant Classification. *ECTI-CIT*, 20(2), 247–257. — 10 ชนิด × 500 ภาพ, ตั้งใจถ่ายหลายแสง/มุม/ฉากหลัง "เพื่อไม่ให้โมเดลเรียนรู้ความสัมพันธ์ลวงกับฉากหลัง", ใช้ McNemar's test + MCC + model size/inference time
3. Ashiddiq, M. I. et al. (2026). Evaluasi Komparatif Model Transfer learning untuk Klasifikasi Tanaman Aquascape. *Telcomatics*, 11(1), 38–45. — PyTorch, MobileNetV3-L / ResNet18 / EfficientNet-B0, 1,998 ภาพ 6 คลาส ได้ 88.7–92.7%, AdamW + early stopping + class weighting + gaussian blur

**ความนิยมในไทย**
- [dtac](https://www.dtac.co.th/liv/lifestyle/cactus.html) ·
  [HOMEDAY](https://homeday.co.th/articles/blogs/10popular-cactus) ·
  [Kapook](https://home.kapook.com/view211495.html) ·
  [Sale Here](https://salehere.co.th/articles/cactus-species) ·
  [Time Out Bangkok](https://www.timeout.com/bangkok/th/things-to-do/cactus-plants)

**กฎหมาย**
- [กองควบคุมวัตถุเสพติด อย. — วัตถุออกฤทธิ์ประเภท 1](https://narcotic.fda.moph.go.th/detail-psycho-new1/category/detail-psychrotrophic1) ·
  [Thai PBS — วัตถุออกฤทธิ์ทางจิตและประสาท](https://www.thaipbs.or.th/now/content/2780)
