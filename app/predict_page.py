"""
predict_page.py — หน้า "ทายชื่อ + วิธีดูแล" (ผู้ใช้ทั่วไป): ถ่าย/อัปโหลดภาพ → ป้ายชื่อต้นไม้ + วิธีดูแล

- โมเดลค่าเริ่มต้น = config.APP_MODELS[0] (seed ที่ val ดีสุด) · เปลี่ยน/เทียบ 3 โมเดลได้ใน sidebar
- รองรับภาพจาก iPhone (.HEIC) และหมุนตาม EXIF เหมือนตอนเทรน
"""
from __future__ import annotations

import csv
import html
from urllib.parse import unquote, urlparse

import pillow_heif
import streamlit as st
from PIL import Image

from src.config import (
    APP_MODELS,
    CARE_TOPICS,
    CLASS_TH,
    CLASSES,
    CONFIDENCE_THRESHOLD,
    MODEL_NAMES,
    REPORT_DIR,
    SUBCLASS_TH,
)
from src.datasets import load_image
from src.gradcam import spotlight
from src.predict import Prediction, Predictor, best_run

pillow_heif.register_heif_opener()
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "heic", "heif"]
RETAKE_TIPS = ["ถ่ายให้เห็นทั้งต้น ไม่ตัดขอบ", "ต้นอยู่กลางภาพ ใหญ่พอ (ไม่ไกลเกิน)", "แสงพอ ไม่ย้อนแสง ภาพไม่เบลอ",
               "ถ้ามีดอกใหญ่บังต้น ลองถ่ายมุมที่เห็นลำต้นชัด"]
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Mali:wght@600;700&display=swap');
/* ป้ายปักกระถาง: ป้ายขาว มีรูร้อยเชือก ชื่อต้นไม้เขียนด้วยลายมือ — จุดเด่นเดียวของหน้า */
.tag {position:relative; background:#fff; border:1px solid #d3d8c6; border-radius:6px 18px 18px 6px;
      padding:16px 20px 16px 46px; margin-bottom:12px; box-shadow:0 2px 0 #d3d8c6;}
.tag::before {content:""; position:absolute; left:16px; top:50%; width:14px; height:14px; margin-top:-7px;
              border-radius:50%; background:#f4f6ef; border:2px solid #2f6b3a;}
.tag.unsure::before {border-color:#b7791f;}
.tag .k {font-size:.9rem; color:#5b6a5c;}
.tag .th {font-family:'Mali', cursive; font-size:2.3rem; font-weight:700; line-height:1.2; margin:2px 0;
          color:#2f6b3a;}
.tag.unsure .th {color:#7a5414;}
.tag .la {font-style:italic; color:#5b6a5c;}
.tag .sp {display:inline-block; margin-top:8px; padding:3px 12px; border-radius:999px; background:#e8e4d6;
          color:#1d2a1f; font-weight:600;}
.care-head {display:flex; flex-wrap:wrap; align-items:baseline; gap:8px 12px; margin:8px 0 4px;}
.care-head h3 {margin:0; font-size:1.25rem; line-height:1.35;}
.lvl {padding:2px 10px; border-radius:999px; font-size:.85rem; font-weight:600; background:#e6efe3; color:#2f6b3a;}
.lvl.genus {background:#f3ead8; color:#7a5414;}
.season {color:#5b6a5c; font-size:.9rem; margin-bottom:14px;}
.care-grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px;}
.care {background:#fff; border:1px solid #dcd8c8; border-radius:12px; padding:16px 18px;}
.care .t {display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.05rem; margin-bottom:10px;}
.care .ic {width:36px; height:36px; border-radius:50%; display:grid; place-items:center; font-size:1.2rem;}
.care .s {font-size:1.05rem; font-weight:600; line-height:1.5; color:#1d2a1f;}
.care .d {font-size:.95rem; line-height:1.6; color:#3d4a3e; margin-top:8px; padding-top:8px;
          border-top:1px dashed #e3dfd0;}
.watch {display:flex; gap:12px; margin-top:14px; padding:14px 18px; border-radius:12px; background:#fdf3e1;
        border:1px solid #f0d9a8; color:#5c3d06; line-height:1.6;}
.watch b {white-space:nowrap;}
.refs {margin-top:12px; font-size:.88rem; color:#5b6a5c;}
.refs a {color:#2f6b3a; margin-right:12px;}
/* วิธีใช้: ลำดับขั้นจริง → เลขในวงกลมต่อด้วยเส้นแนวตั้ง */
.steps {margin:4px 0 18px;}
.step {display:flex; align-items:center; gap:14px; padding:8px 0; position:relative; font-size:1.02rem;}
.step b {flex:none; width:34px; height:34px; border-radius:50%; display:grid; place-items:center;
         background:#2f6b3a; color:#fff; font-weight:700; position:relative; z-index:1;}
.step:not(:last-child)::after {content:""; position:absolute; left:16px; top:42px; height:calc(100% - 34px);
                               border-left:2px solid #d3d8c6;}
</style>
"""
CARE_ICON_BG = {"water": "#e3effb", "light": "#fdf0d3", "soil": "#f3e6dc"}     # สีอ่อนของแต่ละหัวข้อ
SEASON_NOTE = "ช่วงร้อนและฝน (มี.ค.–ต.ค.) คือช่วงที่ต้นโต ส่วนช่วงหนาว (พ.ย.–ก.พ.) คือช่วงพัก"
REF_NAMES = {"llifle.com": "LLIFLE", "ncsu.edu": "NC State Extension", "uada.edu": "UA Extension",
             "succulentsandsunshine.com": "Succulents and Sunshine"}
STEPS = [("1", "ถ่ายหรืออัปโหลดรูป ให้เห็นทั้งต้น"), ("2", "รู้ชื่อสกุล + ชนิด พร้อมความมั่นใจ"),
         ("3", "ได้วิธีดูแลเรื่องน้ำ แสง และดิน")]


@st.cache_data
def model_scores() -> dict[str, float]:
    """ความแม่นยำระดับชนิดบนชุดทดสอบ (เฉลี่ย 3 seeds) ของแต่ละโมเดล — ไม่มีไฟล์ = ไม่แสดงตัวเลข"""
    path = REPORT_DIR / "model_compare.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["model"]: float(r["leaf_acc_mean"]) for r in csv.DictReader(f)}


def model_label(key: str) -> str:
    """'1. DenseNet-121 (91.0%)' — APP_MODELS เรียงจากแม่นสุดอยู่แล้ว (compare.py)"""
    score = model_scores().get(key)
    return f"{APP_MODELS.index(key) + 1}. {MODEL_NAMES[key]}" + (f" ({score:.1%})" if score else "")


@st.cache_resource(show_spinner="กำลังโหลดโมเดล…")
def get_predictor(model_key: str) -> Predictor:
    return Predictor(best_run(model_key))


def confidence_word(p: float) -> str:
    return "มั่นใจสูง" if p >= 0.85 else "ค่อนข้างมั่นใจ" if p >= CONFIDENCE_THRESHOLD else "ไม่แน่ใจ"


def plant_tag(pred: Prediction) -> None:
    """ป้ายชื่อแบบป้ายปักกระถาง — ชื่อไทยใหญ่ · ชื่อวิทยาศาสตร์ · ชนิด (ข้อความทั้งหมดมาจาก config ไม่ใช่ผู้ใช้)"""
    genus, prob = pred.topk[0]
    sub = pred.subclass
    sub_html = f'<div class="sp">ชนิด: {html.escape(SUBCLASS_TH[sub])}</div>' if sub else ""
    head = "ผลที่ได้" if pred.confident else "ใกล้เคียงที่สุด (ยังไม่แน่ใจ)"
    st.markdown(f'<div class="tag{"" if pred.confident else " unsure"}"><div class="k">{head}</div>'
                f'<div class="th">{html.escape(CLASS_TH[genus])}</div><div class="la">{genus.capitalize()}</div>'
                f"{sub_html}</div>", unsafe_allow_html=True)
    st.progress(prob, text=f"{confidence_word(prob)} ({prob:.0%})")


def unsure_help() -> None:
    st.warning(f"ความมั่นใจต่ำกว่า {CONFIDENCE_THRESHOLD:.0%} จึงยังไม่แนะนำวิธีดูแล "
               "(แนะนำผิดสกุลอาจทำให้ต้นเสียหาย) — ลองถ่ายใหม่:\n\n" + "\n".join(f"- {t}" for t in RETAKE_TIPS))


def ref_link(url: str) -> str:
    """ลิงก์อ้างอิงแสดงเป็นชื่อแหล่ง (LLIFLE: Astrophytum asterias) แทนเลข — รับเฉพาะ http(s)"""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        return ""
    name = next((n for host, n in REF_NAMES.items() if parts.netloc.endswith(host)), parts.netloc)
    if name == "LLIFLE":                                   # ชื่อพืชอยู่ท้าย path ของ LLIFLE
        name += ": " + unquote(parts.path.rstrip("/").rsplit("/", 1)[-1]).replace("_", " ")
    return f'<a href="{html.escape(url)}" target="_blank" rel="noopener">{html.escape(name)}</a>'


def care_section(pred: Prediction) -> None:
    care = pred.care
    if care is None:
        st.info("ยังไม่มีคำแนะนำการดูแลของสกุลนี้")
        return
    # สกุลที่มีชนิดเดียวในระบบ (ถังทอง/หูกระต่าย) คำแนะนำของสกุล = ของชนิดนั้นอยู่แล้ว
    species_level = not pred.subclasses or (bool(pred.subclass) and care["name_th"] != CLASS_TH.get(pred.genus))
    level = ('<span class="lvl">เฉพาะชนิดนี้</span>' if species_level
             else '<span class="lvl genus">ทั่วไปของสกุล</span>')
    cards = "".join(
        f'<div class="care"><div class="t"><span class="ic" style="background:{CARE_ICON_BG[topic]}">'
        f"{label.split()[0]}</span>{html.escape(label.split(maxsplit=1)[1])}</div>"
        f'<div class="s">{html.escape(care[topic]["summary"])}</div>'
        + (f'<div class="d">{html.escape(care[topic]["detail"])}</div>' if care[topic].get("detail") else "")
        + "</div>" for topic, label in CARE_TOPICS.items())
    refs = " ".join(filter(None, (ref_link(r.strip()) for r in care["reference"].split(";") if r.strip())))
    st.markdown(
        f'<div class="care-head"><h3>วิธีดูแล{html.escape(care["name_th"])}</h3>{level}</div>'
        f'<div class="season">{SEASON_NOTE}</div><div class="care-grid">{cards}</div>'
        f'<div class="watch"><b>⚠️ ระวัง</b><span>{html.escape(care["common_mistake"])}</span></div>'
        f'<div class="refs">แหล่งอ้างอิง: {refs}</div>', unsafe_allow_html=True)


def why_section(predictor: Predictor, image: Image.Image, pred: Prediction) -> None:
    with st.expander("ทำไมโมเดลถึงตอบแบบนี้"):
        left, right = st.columns(2)
        with left:
            st.caption("สกุลที่เป็นไปได้")
            for genus, prob in pred.topk:
                st.progress(prob, text=f"{CLASS_TH[genus]} · {prob:.1%}")
            if len(pred.subclasses) > 1:
                st.caption("ชนิดภายในสกุลอันดับ 1")
                for sub, prob in pred.subclasses:
                    st.progress(prob, text=f"{SUBCLASS_TH[sub]} · {prob:.1%}")
        with right:
            if st.toggle("แสดงจุดที่โมเดลมอง (Grad-CAM)"):
                seen, heat = predictor.explain(image, pred.class_index)
                st.image(spotlight(seen, heat), width="stretch", caption="ส่วนที่สว่าง = ส่วนที่โมเดลใช้ตัดสินใจ")
        st.caption(f"โมเดล {MODEL_NAMES[predictor.ckpt['model_key']]} · {predictor.run} · {predictor.device}")


def compare_models(image: Image.Image) -> None:
    rows = []
    for key in APP_MODELS:
        p = get_predictor(key).predict(image)
        genus, prob = p.topk[0]
        rows.append({"โมเดล": model_label(key), "สกุล": CLASS_TH[genus],
                     "ชนิด": SUBCLASS_TH[p.subclass] if p.subclass else "–", "ความมั่นใจ": prob,
                     "ผล": confidence_word(prob)})
    st.dataframe(rows, hide_index=True, width="stretch", column_config={
        "ความมั่นใจ": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent")})


def reset() -> None:
    """เริ่มใหม่ — เปลี่ยน key = file_uploader ตัวใหม่ (ล้างไฟล์ที่อัปโหลดไว้)"""
    st.session_state.upload_key += 1


def how_to() -> None:
    """หน้าว่าง: วิธีใช้ 3 ขั้น + เคล็ดลับถ่ายรูปให้ทายแม่น"""
    st.markdown("#### วิธีใช้")
    st.markdown('<div class="steps">' + "".join(f'<div class="step"><b>{n}</b><span>{t}</span></div>'
                                                for n, t in STEPS) + "</div>", unsafe_allow_html=True)
    st.markdown("#### ถ่ายรูปอย่างไรให้ทายแม่น")
    st.markdown("\n".join(f"- {t}" for t in RETAKE_TIPS))
    st.caption("รู้จัก 7 สกุล: " + " ".join(CLASS_TH[g] for g in CLASSES))


def pick_image() -> tuple[Image.Image | None, str]:
    upload, camera = st.tabs(["📁 อัปโหลดรูป", "📷 ถ่ายรูป"])
    with upload:
        file = st.file_uploader("รูปกระบองเพชร (ให้เห็นทั้งต้น)", type=UPLOAD_TYPES, label_visibility="collapsed",
                                key=f"up_{st.session_state.upload_key}")
    with camera:          # เปิดกล้องเมื่อผู้ใช้กดเท่านั้น — ไม่งั้นเบราว์เซอร์ขอสิทธิ์กล้องทันทีที่เปิดหน้า
        shot = st.camera_input("ถ่ายรูป", label_visibility="collapsed") if st.toggle("เปิดกล้อง") else None
    chosen = shot or file
    try:
        if chosen:
            return load_image(chosen), chosen.name
    except OSError:
        st.error("เปิดไฟล์นี้ไม่ได้ — ใช้ JPG / PNG / WEBP / HEIC")
    return None, ""


st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("upload_key", 0)
with st.sidebar:
    st.subheader("ตั้งค่าโมเดล")
    model = st.radio("โมเดลที่ใช้ทาย", APP_MODELS, format_func=model_label,
                     help="เรียงจากแม่นที่สุด ตัวเลขคือความแม่นยำในการทายชนิดบนชุดทดสอบ (เฉลี่ย 3 seeds) "
                          "ทั้ง 3 ตัวทายสกุลผ่านเป้า 91%")
    st.caption("อันดับ 1 แม่นที่สุด ตัวเลขในวงเล็บคือความแม่นยำการทายชนิด")
    compare_all = st.toggle("เทียบผลทั้ง 3 โมเดล")
    st.caption(f"ความมั่นใจต่ำกว่า {CONFIDENCE_THRESHOLD:.0%} → ตอบว่าไม่แน่ใจ แทนการเดา")

st.title("กระบองเพชรต้นนี้ชื่ออะไร")
st.caption("ถ่ายรูปหรืออัปโหลดรูปต้นกระบองเพชร แล้วแอปจะบอกชื่อและวิธีดูแล รู้จัก 7 สกุลที่นิยมเลี้ยงในไทย")
left, right = st.columns([5, 7], gap="large")
with left:
    image, name = pick_image()
    if image is not None:
        st.image(image, caption=name, width="stretch")
        st.button("↺ เลือกรูปอื่น", on_click=reset, width="stretch")
with right:
    if image is None:
        how_to()
    else:
        try:
            predictor = get_predictor(model)
        except (Exception, SystemExit):    # ไม่มี checkpoint / LFS ยังไม่ดึง (ได้แค่ไฟล์ pointer)
            st.error("โหลดโมเดลไม่ได้ — ตรวจว่ามีไฟล์ใน checkpoints/ (repo นี้เก็บโมเดลด้วย Git LFS)")
            st.stop()
        with st.spinner("กำลังดู…"):
            result = predictor.predict(image)
        plant_tag(result)
        if compare_all:
            compare_models(image)
        # วิธีดูแลอยู่ใต้ผลทันที (เห็นพร้อมกันโดยไม่ต้องเลื่อนผ่านรูป) · คำอธิบายโมเดลไว้ท้ายสุด
        if result.confident:
            care_section(result)
        else:
            unsure_help()
        why_section(predictor, image, result)
st.divider()
st.caption("ผลจากโมเดลอาจผิดพลาดได้ ใช้ประกอบการตัดสินใจ ไม่ใช่แทนผู้เชี่ยวชาญ "
           "รูปที่อัปโหลดใช้ทายผลเท่านั้นและไม่ถูกเก็บไว้ — โปรเจควิชา Machine Learning, KMUTNB")
