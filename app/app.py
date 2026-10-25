"""
app.py — เว็บแอป Streamlit 2 หน้า

    streamlit run app/app.py        (รันจาก project root เพื่อให้ใช้ธีมใน .streamlit/config.toml)

    predict_page.py   ทายชื่อ + วิธีดูแล (ผู้ใช้ทั่วไป)
    eval_page.py      ประเมินผลโมเดล (นำเสนอ / ทีม) — อ่านผลจาก reports/ ไม่ทายใหม่
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # ให้ทุกหน้า import src ได้ เมื่อรันจากโฟลเดอร์ไหนก็ได้

st.set_page_config(page_title="Cactus Classifier", page_icon="🌵", layout="wide")
st.logo(str(Path(__file__).parent / "logo.svg"), size="large")      # ชื่อกลุ่มมุมซ้ายบน (config.TEAM_NAME)
st.navigation([
    st.Page("predict_page.py", title="ทายชื่อ + วิธีดูแล", icon="🌵", default=True),
    st.Page("eval_page.py", title="ประเมินผลโมเดล", icon="📊"),
]).run()
