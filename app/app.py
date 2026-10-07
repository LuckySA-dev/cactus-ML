"""
app.py — เว็บแอปทดสอบโมเดล: อัปโหลด/ถ่ายภาพกระบองเพชร → สกุล + ชนิด + ความมั่นใจ + จุดที่โมเดลมอง + วิธีดูแล

    streamlit run app/app.py

- เลือกโมเดลได้ทีละตัว (config.APP_MODELS) หรือเทียบทุกตัวกับภาพเดียวกัน
- แต่ละโมเดลใช้ seed ที่ val macro F1 สูงสุด (predict.best_run — เลือกด้วย val ไม่ได้ดู test)
- รองรับภาพจาก iPhone (.HEIC) และหมุนตาม EXIF เหมือนตอนเทรน
"""
from __future__ import annotations

import sys
from pathlib import Path

import pillow_heif
import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # ให้ import src ได้เมื่อรันจากโฟลเดอร์ไหนก็ได้

from src.config import (  # noqa: E402
    APP_MODELS,
    CARE_TOPICS,
    CLASS_TH,
    CONFIDENCE_THRESHOLD,
    MODEL_NAMES,
    SUBCLASS_TH,
)
from src.gradcam import spotlight  # noqa: E402
from src.predict import UNSURE_MESSAGE, Prediction, Predictor, best_run  # noqa: E402

pillow_heif.register_heif_opener()
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "heic", "heif"]


@st.cache_resource(show_spinner="กำลังโหลดโมเดล…")
def get_predictor(model_key: str) -> Predictor:
    return Predictor(best_run(model_key))


def open_image(file) -> Image.Image:
    """หมุนตาม EXIF เหมือน datasets.load_image — รูปมือถือแนวตั้งไม่นอนตะแคง"""
    with Image.open(file) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def genus_label(genus: str) -> str:
    return f"{CLASS_TH[genus]} ({genus.capitalize()})"


def show_prediction(pred: Prediction) -> None:
    if not pred.confident:
        st.warning(f"**{UNSURE_MESSAGE}**  \nสูงสุดคือ {genus_label(pred.topk[0][0])} "
                   f"{pred.topk[0][1]:.0%} ซึ่งต่ำกว่าเกณฑ์ {CONFIDENCE_THRESHOLD:.0%}")
    else:
        st.success(f"### {genus_label(pred.genus)}\nความมั่นใจ **{pred.confidence:.0%}**")
        if pred.subclasses:
            sub, prob = pred.subclasses[0]
            if pred.subclass:
                st.markdown(f"**ชนิด:** {SUBCLASS_TH[sub]} — {prob:.0%}")
            else:
                st.markdown(f"**ชนิด:** ไม่แน่ใจ (ใกล้เคียงสุด {SUBCLASS_TH[sub]} {prob:.0%})")

    st.caption("สกุลที่เป็นไปได้")
    for genus, prob in pred.topk:
        st.progress(prob, text=f"{genus_label(genus)} · {prob:.1%}")
    if len(pred.subclasses) > 1:
        with st.expander("ชนิดภายในสกุลนี้"):
            for sub, prob in pred.subclasses:
                st.progress(prob, text=f"{SUBCLASS_TH[sub]} · {prob:.1%}")


def show_gradcam(predictor: Predictor, image: Image.Image, pred: Prediction) -> None:
    seen, heat = predictor.explain(image, pred.class_index)
    st.image(spotlight(seen, heat), caption="ส่วนที่สว่าง = ส่วนที่โมเดลใช้ตัดสินใจ (Grad-CAM)", width=300)


def show_care(pred: Prediction) -> None:
    if not pred.confident:
        return
    if pred.care is None:
        st.info("ยังไม่มีคำแนะนำการดูแลของสกุลนี้ (ทีมกำลังเขียน care_rules.json พร้อมแหล่งอ้างอิง)")
        return
    st.subheader("วิธีดูแล")
    for topic, label in CARE_TOPICS.items():
        st.markdown(f"**{label}** {pred.care[topic]['summary']}")
        if pred.care[topic].get("detail"):
            st.caption(pred.care[topic]["detail"])
    st.warning(f"ข้อผิดพลาดที่พบบ่อย: {pred.care['common_mistake']}")
    st.caption(f"อ้างอิง: {pred.care['reference']}")


def main() -> None:
    st.set_page_config(page_title="Cactus Classifier", page_icon="🌵", layout="wide")
    st.title("🌵 Cactus Classifier")
    st.caption("จำแนกสกุลและชนิดกระบองเพชรแคระ 7 สกุลจากภาพถ่าย · งานวิชา Machine Learning KMUTNB")

    with st.sidebar:
        st.header("ตั้งค่า")
        compare_all = st.radio("โหมด", ["โมเดลเดียว", "เทียบทุกโมเดล"]) == "เทียบทุกโมเดล"
        models = APP_MODELS if compare_all else [st.radio("โมเดล", APP_MODELS, format_func=MODEL_NAMES.get)]
        with_cam = st.toggle("แสดงจุดที่โมเดลมอง (Grad-CAM)", value=True)
        st.divider()
        st.caption(f"ความมั่นใจต่ำกว่า {CONFIDENCE_THRESHOLD:.0%} → ตอบว่า \"ไม่แน่ใจ\" แทนการเดา "
                   "(แนะนำการดูแลผิดสกุลทำให้ต้นเสียหายได้)")

    upload_tab, camera_tab = st.tabs(["📁 อัปโหลดภาพ", "📷 ถ่ายภาพ"])
    with upload_tab:
        files = st.file_uploader("เลือกภาพกระบองเพชร (ให้เห็นทั้งต้น)", type=UPLOAD_TYPES,
                                 accept_multiple_files=True)
    with camera_tab:      # เปิดกล้องเมื่อผู้ใช้กดเท่านั้น — ไม่งั้นเบราว์เซอร์ขอสิทธิ์กล้องทันทีที่เปิดหน้า
        shot = st.camera_input("ถ่ายภาพ") if st.toggle("เปิดกล้อง") else None
    files = [*(files or []), *([shot] if shot else [])]
    if not files:
        st.info("อัปโหลดหรือถ่ายภาพเพื่อเริ่มทำนาย")
        return

    for file in files:
        image = open_image(file)
        st.divider()
        cols = st.columns([1, *[2] * len(models)])
        cols[0].image(image, caption=file.name, width="stretch")
        for col, key in zip(cols[1:], models, strict=True):
            predictor = get_predictor(key)
            pred = predictor.predict(image)
            with col:
                st.markdown(f"#### {MODEL_NAMES[key]}")
                st.caption(f"checkpoint: {predictor.run} · {predictor.device}")
                show_prediction(pred)
                if with_cam:
                    show_gradcam(predictor, image, pred)
                if len(models) == 1:
                    show_care(pred)


if __name__ == "__main__":
    main()
