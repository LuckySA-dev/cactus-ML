"""
eval_page.py — หน้า "ประเมินผลโมเดล" (สำหรับนำเสนอ): เทียบ 6 โมเดล + เจาะรายโมเดล

อ่านผลที่ evaluate.py / compare.py เขียนไว้ใน reports/ อย่างเดียว — หน้านี้ไม่ทายใหม่
    reports/model_compare.csv        ค่าเฉลี่ย ± SD ข้าม seed ต่อโมเดล
    reports/predictions/<run>_*.csv  ผลรายภาพ (confusion / รายคลาส / threshold / ภาพที่ผิด)
    reports/v1_base/                 ผลสูตรเทรนเดิม (เทียบก่อน/หลัง)
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import timm
import torch
from sklearn.metrics import precision_recall_fscore_support

from src.config import (
    APP_MODELS,
    BATCH_SIZE,
    CLASS_TH,
    CLASSES,
    CONFIDENCE_THRESHOLD,
    DROP_RATE,
    EMA_DECAY,
    EVAL_RESIZE,
    IMG_SIZE,
    LABEL_SMOOTHING,
    METADATA_CSV,
    MODEL_NAMES,
    MODELS,
    PRED_DIR,
    PROJECT_ROOT,
    QC_USABLE,
    REALWORLD_FRAC,
    REFERENCE_GAP,
    REPORT_DIR,
    RUNS_CSV,
    SEED,
    SPLIT_RATIO,
    STAGE1,
    STAGE2,
    SUBCLASS_TH,
    TARGET_ACC,
    WEIGHT_DECAY,
)
from src.predict import load_care_rules
from src.tasks import LEAF_SEP, LEAVES, run_id
from src.visualize import BLUES_HEX, INK2, MODEL_COLOR

SEEDS = [SEED, SEED + 1, SEED + 2]
METRICS = {"genus_acc": "ทายสกุล (test)", "leaf_acc": "ทายชนิด (test)", "rw_genus_acc": "ทายสกุล (ภาพจริง)"}
LINE_COLORS = {"ตอบได้ (coverage)": INK2, "ถูกต้อง (ของที่ตอบ)": BLUES_HEX[4]}


@st.cache_data
def read_csv(path: str, keep_blank: bool = False) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=not keep_blank)


# จำนวนภาพนับจากไฟล์ผลรายภาพ (ไม่ hard-code — split ใหม่แล้วเลขไม่ค้าง)
N_IMAGES = {s: len(read_csv(str(PRED_DIR / f"{run_id(APP_MODELS[0], SEED)}_{s}.csv")))
            for s in ("test", "realworld_test")}
SPLITS = {"test": f"ชุดทดสอบ ({N_IMAGES['test']})", "realworld_test": f"ภาพจริง ({N_IMAGES['realworld_test']})"}


def best_seed(model: str) -> int:
    """seed ที่ val macro F1 สูงสุด จาก runs.csv — ไม่ต้องมี checkpoint (ตอน deploy มีแค่ 3 โมเดลของแอป
    predict.best_run จะหาไฟล์ .pt ไม่เจอสำหรับโมเดลอื่น) · run ชื่อซ้ำใช้แถวล่าสุดเหมือน best_run"""
    runs = read_csv(str(RUNS_CSV), keep_blank=True)
    runs = runs[(runs.phase == "final") & (runs.task == "flat") & (runs.model == model)].drop_duplicates(
        "run", keep="last")
    return int(runs.loc[runs.best_val_f1.astype(float).idxmax(), "seed"]) if len(runs) else SEED


def order_of(key: str) -> int:
    """ลำดับเดียวกับรายงาน (config.CLASSES / tasks.LEAVES)"""
    return LEAVES.index(key) if key in LEAVES else CLASSES.index(key) if key in CLASSES else len(LEAVES)


def label_of(key: str) -> str:
    """ชื่อไทยของสกุล / คลาสย่อย (leaf = สกุล__ชนิด)"""
    if LEAF_SEP in key:
        genus, sub = key.split(LEAF_SEP)
        return f"{CLASS_TH[genus]} · {SUBCLASS_TH[sub]}"
    return CLASS_TH.get(key, key)


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ---------------------------------------------------------------------------
# ภาพรวม + เทียบ 6 โมเดล
# ---------------------------------------------------------------------------
def overview(cmp: pd.DataFrame) -> None:
    top = cmp.set_index("model").loc[APP_MODELS[0]]
    st.subheader(f"ผลของ {MODEL_NAMES[APP_MODELS[0]]} โมเดลหลักของแอป")
    c = st.columns(4)
    # delta เฉพาะตัวเลขที่เทียบกับเกณฑ์จริง — ข้อความอธิบายอยู่ใน help (delta ข้อความจะได้ลูกศรขึ้นที่ชวนเข้าใจผิด)
    c[0].metric(METRICS["genus_acc"], pct(top.genus_acc_mean),
                f"{(top.genus_acc_mean - TARGET_ACC) * 100:+.1f} จุด vs เป้า",
                help=f"เป้าหมาย {TARGET_ACC:.0%} (baseline งานวิจัยอ้างอิง)", border=True)
    c[1].metric(METRICS["leaf_acc"], pct(top.leaf_acc_mean), help=f"± {top.leaf_acc_std * 100:.1f} จุด ข้าม 3 seeds",
                border=True)
    c[2].metric(METRICS["rw_genus_acc"], pct(top.rw_genus_acc_mean),
                help=f"ภาพถ่ายเองที่ฟาร์ม โมเดลไม่เคยเห็น ({N_IMAGES['realworld_test']} ภาพ)", border=True)
    c[3].metric("Gap (test − ภาพจริง)", pct(top.gap_mean), f"{(top.gap_mean - REFERENCE_GAP) * 100:+.1f} จุด vs งานวิจัย",
                delta_color="inverse", help=f"งานวิจัยอ้างอิงมี gap {REFERENCE_GAP:.2%} — ต่ำกว่า = ใช้กับภาพจริงได้ดีกว่า",
                border=True)


def comparison(cmp: pd.DataFrame) -> None:
    st.subheader("เทียบ 6 โมเดล")
    st.caption("จุดคือค่าเฉลี่ยจาก 3 seeds เส้นคือ ± SD")
    long = pd.concat([
        pd.DataFrame({"โมเดล": cmp.name, "ตัวชี้วัด": label, "ค่า": cmp[f"{m}_mean"], "sd": cmp[f"{m}_std"]})
        for m, label in METRICS.items()])
    long["ต่ำ"], long["สูง"] = long["ค่า"] - long["sd"], long["ค่า"] + long["sd"]
    names = [MODEL_NAMES[m] for m in MODELS]
    color = alt.Color("โมเดล:N", scale=alt.Scale(domain=names, range=[MODEL_COLOR[m] for m in MODELS]), legend=None)
    y = alt.Y("โมเดล:N", sort=list(cmp.name), title=None)
    x = alt.X("ค่า:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="%"), title=None)
    tip = [alt.Tooltip("โมเดล:N"), alt.Tooltip("ตัวชี้วัด:N"), alt.Tooltip("ค่า:Q", format=".1%"),
           alt.Tooltip("sd:Q", format=".1%", title="± SD")]
    # 3 กราฟแยกคอลัมน์ (ไม่ใช่ facet) → ยืดตามจอ และเรียงลงเป็นแนวตั้งบนมือถือ
    for col, label in zip(st.columns(len(METRICS)), METRICS.values(), strict=True):
        base = alt.Chart(long[long["ตัวชี้วัด"] == label])
        chart = alt.layer(
            base.mark_rule(strokeWidth=2).encode(x="ต่ำ:Q", x2="สูง:Q", y=y, color=color),
            base.mark_circle(size=110, opacity=1, stroke="#f4f6ef", strokeWidth=2).encode(x=x, y=y, color=color,
                                                                                           tooltip=tip),
        ).properties(title=label, height=230)
        col.altair_chart(chart, width="stretch")
    st.caption("แกนแต่ละกราฟซูมคนละช่วง (ไม่เริ่มที่ 0) เพื่อให้เห็นความต่าง "
               f"ภาพจริงมีแค่ {N_IMAGES['realworld_test']} ภาพ ตัวเลขจึงแกว่งได้ราว ±3 จุด")

    table = pd.DataFrame({
        "โมเดล": cmp.name,
        **{label: [f"{a * 100:.1f} ± {b * 100:.1f}" for a, b in zip(cmp[f"{m}_mean"], cmp[f"{m}_std"], strict=True)]
           for m, label in METRICS.items()},
        "Gap": [pct(v) for v in cmp.gap_mean],
        "พารามิเตอร์ (M)": cmp.params_m.round(1),
        "CPU ms/ภาพ": cmp.cpu_ms.round(0).astype(int),
        "แพ้ตัวดีสุดอย่างมีนัยสำคัญ": [f"{int(n)}/3 seeds" for n in cmp.seeds_sig_worse],
    })
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("เรียงตามความแม่นยำระดับชนิดบน test คอลัมน์สุดท้ายใช้ McNemar test (p < 0.05) "
               "เทียบกับโมเดลที่ดีที่สุดใน seed เดียวกัน")


# ---------------------------------------------------------------------------
# เจาะรายโมเดล
# ---------------------------------------------------------------------------
def confusion_chart(y_true: pd.Series, y_pred: pd.Series, many: bool) -> alt.Chart:
    labels = sorted(set(y_true) | set(y_pred), key=order_of)
    ct = pd.crosstab(y_true, y_pred).reindex(index=labels, columns=labels, fill_value=0)
    share = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
    # ชื่อคอลัมน์ภาษาอังกฤษ: นิพจน์ vega (datum.xxx) อ่านชื่อภาษาไทยไม่ได้ → สีหาย ตัวเลขเป็น NaN
    df = (share.stack().rename("share").reset_index().set_axis(["true", "pred", "share"], axis=1)
          .assign(n=ct.stack().to_numpy()))
    df["true"], df["pred"] = df["true"].map(label_of), df["pred"].map(label_of)
    order = [label_of(k) for k in labels]
    base = alt.Chart(df).encode(
        x=alt.X("pred:N", sort=order, title="โมเดลทายว่า",
                axis=alt.Axis(labelAngle=-90 if many else -35, labelLimit=200, labelOverlap=False)),
        y=alt.Y("true:N", sort=order, title="คลาสจริง", axis=alt.Axis(labelLimit=220, labelOverlap=False)))
    cells = base.mark_rect(stroke="#f4f6ef", strokeWidth=2).encode(
        color=alt.condition("datum.n > 0",
                            alt.Color("share:Q", scale=alt.Scale(domain=[0, 1], range=BLUES_HEX),
                                      legend=alt.Legend(format="%", title="สัดส่วนของแถว")),
                            alt.value("#e8e4d6")),
        tooltip=[alt.Tooltip("true:N", title="จริง"), alt.Tooltip("pred:N", title="ทาย"),
                 alt.Tooltip("share:Q", format=".1%", title="สัดส่วน"), alt.Tooltip("n:Q", title="จำนวนภาพ")])
    # ความสูงตามจำนวนคลาส ความกว้างยืดเต็มคอลัมน์ (ตั้ง width ตายตัว = ช่องเหลือไม่กี่ px แล้ว label หาย)
    height = 30 * len(order) if many else 58 * len(order)
    text = base.mark_text(fontSize=10 if many else 13).encode(
        text=alt.condition("datum.n > 0", alt.Text("n:Q"), alt.value("")),
        color=alt.condition("datum.share > 0.5", alt.value("white"), alt.value("#1d2a1f")))
    return (cells + text).properties(height=height).configure_axis(labelFontSize=12, titleFontSize=13,
                                                                     titleColor=INK2, labelColor="#1d2a1f")


def per_class_table(y_true: pd.Series, y_pred: pd.Series) -> None:
    labels = sorted(set(y_true), key=order_of)
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    df = pd.DataFrame({"คลาส": [label_of(k) for k in labels], "Precision": p, "Recall": r, "F1": f,
                       "จำนวนภาพ": n}).sort_values("F1")
    bar = {c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="percent")
           for c in ("Precision", "Recall", "F1")}
    st.dataframe(df, hide_index=True, width="stretch", column_config=bar)


def threshold_section(pred: pd.DataFrame) -> None:
    st.markdown("#### เกณฑ์ความมั่นใจ")
    st.caption("ถ้าความมั่นใจต่ำกว่าเกณฑ์ แอปจะตอบว่า \"ไม่แน่ใจ\" แทนการเดา ลองเลื่อนดูว่าตอบได้กี่ภาพและถูกกี่ %")
    t = st.slider("เกณฑ์", 0.0, 0.95, CONFIDENCE_THRESHOLD, 0.05, label_visibility="collapsed")
    conf, ok = pred.confidence.to_numpy(), pred.correct.to_numpy().astype(bool)
    grid = np.round(np.arange(0, 0.96, 0.01), 2)
    curve = pd.DataFrame([{"เกณฑ์": g, "เส้น": name, "ค่า": v} for g in grid for name, v in (
        ("ตอบได้ (coverage)", (conf >= g).mean()),
        ("ถูกต้อง (ของที่ตอบ)", ok[conf >= g].mean() if (conf >= g).any() else np.nan))])
    kept = conf >= t
    c = st.columns(3)
    c[0].metric("ตอบได้", pct(kept.mean()), border=True)
    c[1].metric("ถูกต้องในส่วนที่ตอบ", pct(ok[kept].mean()) if kept.any() else "–", border=True)
    c[2].metric("ส่งให้ถ่ายใหม่", f"{(~kept).sum()} ภาพ", border=True)
    color = alt.Color("เส้น:N", scale=alt.Scale(domain=list(LINE_COLORS), range=list(LINE_COLORS.values())),
                      legend=alt.Legend(orient="bottom", title=None))
    lines = alt.Chart(curve).mark_line(strokeWidth=2).encode(
        x=alt.X("เกณฑ์:Q", axis=alt.Axis(format=".0%")), y=alt.Y("ค่า:Q", axis=alt.Axis(format="%"),
                                                              scale=alt.Scale(zero=False), title=None),
        color=color, tooltip=["เส้น:N", alt.Tooltip("เกณฑ์:Q", format=".0%"), alt.Tooltip("ค่า:Q", format=".1%")])
    rule = alt.Chart(pd.DataFrame({"เกณฑ์": [t]})).mark_rule(strokeDash=[4, 4], color=INK2).encode(x="เกณฑ์:Q")
    st.altair_chart((lines + rule).properties(height=240), width="stretch")


def mistakes_gallery(wrong: pd.DataFrame, true_col: str, pred_col: str) -> None:
    wrong = wrong.sort_values("confidence", ascending=False).head(12)
    if wrong.empty:
        st.success("ไม่มีภาพที่ทายผิดใน split นี้")
        return
    paths = [PROJECT_ROOT / f for f in wrong.filepath]
    if not any(p.exists() for p in paths):
        st.info("ไม่มีไฟล์ภาพในเครื่องนี้ (ภาพไม่ได้อยู่ใน git) — ดูได้เฉพาะเครื่องที่มี data/processed/")
        return
    cols = st.columns(6)
    for i, (row, path) in enumerate(zip(wrong.itertuples(), paths, strict=True)):
        if path.exists():
            true, guess = label_of(getattr(row, true_col)), label_of(getattr(row, pred_col))
            cols[i % 6].image(str(path), width="stretch", caption=f"จริง {true}\n→ ทาย {guess} ({row.confidence:.0%})")


def drilldown() -> None:
    st.subheader("ดูผลรายโมเดล")
    c = st.columns([2, 1.2, 1.6, 1.2])
    model = c[0].selectbox("โมเดล", list(MODELS), index=list(MODELS).index(APP_MODELS[0]), format_func=MODEL_NAMES.get)
    app_seed = best_seed(model)                                     # ค่าเริ่มต้น = seed ที่ val ดีสุด
    seed = c[1].segmented_control("seed", SEEDS, default=app_seed, key=f"seed_{model}",
                                  help="ค่าเริ่มต้น = seed ที่แอปใช้ (val macro F1 สูงสุด)")
    split = c[2].segmented_control("ชุดข้อมูล", list(SPLITS), default="test", format_func=SPLITS.get)
    level = c[3].segmented_control("ระดับ", ["สกุล", "ชนิด"], default="สกุล")
    path = PRED_DIR / f"{run_id(model, seed or SEED)}_{split or 'test'}.csv"
    if not path.exists():
        st.warning(f"ยังไม่มีผล {path.name} — รัน python -m src.evaluate")
        return
    pred = read_csv(str(path), keep_blank=True)
    species = level == "ชนิด"
    rows = pred[pred.leaf_true != ""] if species else pred
    true_col, pred_col = ("leaf_true", "pred_class") if species else ("y_true", "y_pred")
    y_true, y_pred = rows[true_col].rename("จริง"), rows[pred_col].rename("ทาย")
    correct = y_true.to_numpy() == y_pred.to_numpy()
    present = sorted(set(y_true))
    f1 = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro", zero_division=0)[2]

    m = st.columns(3)
    m[0].metric("Accuracy", pct(correct.mean()), border=True)
    m[1].metric("Macro F1", pct(f1), border=True)
    m[2].metric("จำนวนภาพ", f"{len(rows):,}", border=True)

    st.markdown("#### Confusion matrix")
    st.caption("แต่ละแถวคือคลาสจริง ตัวเลขในช่องคือจำนวนภาพ สีเข้มคือสัดส่วนของแถวนั้น "
               "ช่องแนวทแยงคือทายถูก ช่องนอกแนวทแยงคือทายผิดไปเป็นคลาสอื่น")
    st.altair_chart(confusion_chart(y_true, y_pred, many=species), width="stretch")
    st.markdown("#### ความแม่นยำรายคลาส")
    st.caption("เรียงจากคลาสที่โมเดลอ่อนที่สุดขึ้นก่อน")
    per_class_table(y_true, y_pred)
    threshold_section(pred)
    st.markdown("#### ภาพที่ทายผิดทั้งที่มั่นใจสูง")
    st.caption("ใช้ดูว่าผิดเพราะภาพกำกวม label ผิด หรือโมเดลพลาดจริง")
    mistakes_gallery(rows[~correct], true_col, pred_col)


# ---------------------------------------------------------------------------
# รายละเอียดโปรเจกต์ — ตัวเลขอ่านจาก metadata.csv / runs.csv / config (ไม่พิมพ์เอง ข้อมูลเปลี่ยนแล้วตามทัน)
# ---------------------------------------------------------------------------
SOURCE_TH = {"inaturalist": "iNaturalist (ภาพต้นปลูก CC license)", "web": "ภาพจากเว็บที่ทีมรวบรวม (ตรวจรายภาพ)",
             "handon": "ภาพถ่ายเองที่ฟาร์ม (iPhone)"}
SPLIT_TH = {"train": "train", "val": "validation", "test": "test", "realworld_test": "ภาพจริง"}
QC_TH = {"wrong_label": "label ผิด", "not_plant": "ดอกบังต้น / ไม่เห็นต้น", "multi_plant": "หลายสกุลในภาพเดียว",
         "blurry": "เบลอ / มืด / ต้นเล็กเกิน"}
AUGMENT = [
    ("RandomResizedCrop", f"สุ่มครอปแล้วย่อเป็น {IMG_SIZE}×{IMG_SIZE} พื้นที่ 35–100% ของภาพ"),
    ("RandomHorizontalFlip", "กลับซ้าย–ขวา (p = 0.5)"),
    ("RandomRotation", "หมุน ±20°"),
    ("RandomPerspective", "บิดมุมมอง distortion 0.2 (p = 0.3)"),
    ("GaussianBlur", "เบลอ kernel 5, σ 0.1–2 (p = 0.2) จำลองภาพมือถือไม่ชัด"),
    ("ColorJitter", "ความสว่าง ±25% · คอนทราสต์ ±25% · ความอิ่มสี ±20% · hue ±0.02"),
    ("RandomErasing", "ลบสี่เหลี่ยมสุ่มบนภาพ (p = 0.25) ให้ไม่พึ่งจุดเดียว เช่น ดอก"),
    ("Normalize", "mean / std ของ pretrained weights จาก timm (ไม่ hard-code)"),
]
# คำอธิบายยาวรายขั้น (dropdown) — ลำดับเดียวกับ AUGMENT และแถวในภาพ reports/figures/augment.png
AUGMENT_DETAIL = {
    "RandomResizedCrop": (
        f"สุ่มเลือกกรอบสี่เหลี่ยมขนาด 35–100% ของพื้นที่ภาพ อัตราส่วน 3:4 ถึง 4:3 แล้วย่อ/ขยายเป็น {IMG_SIZE}×{IMG_SIZE} px "
        "ทุก epoch ภาพเดิมจึงถูกครอปไม่ซ้ำกัน\n\n"
        "**ทำไมใช้:** ภาพจริงที่ฟาร์มมักถ่ายไกล ต้นเล็กในเฟรม หรือถ่ายใกล้จนเห็นแค่บางส่วน การครอปกว้างถึง 35% "
        "(สูตร strong, สูตรเดิม 70%) ทำให้โมเดลเจอทั้งต้นเต็มเฟรมและต้นเล็ก ๆ ช่วยลด gap กับภาพจริง"),
    "RandomHorizontalFlip": (
        "กลับภาพซ้าย–ขวาแบบสุ่ม ครึ่งหนึ่งของภาพ (p = 0.5)\n\n"
        "**ทำไมใช้:** กระบองเพชรสมมาตรซ้าย–ขวา ต้นที่กลับด้านยังเป็นสกุลเดิม ได้ข้อมูลเพิ่มเป็นสองเท่าฟรี ๆ\n\n"
        "**ทำไมไม่กลับบน–ล่าง:** ต้นมีบน–ล่างชัด (ยอด ดอก กระถางอยู่ล่าง) ภาพกลับหัวไม่มีในโลกจริง"),
    "RandomRotation": (
        "หมุนภาพแบบสุ่มระหว่าง −20° ถึง +20° มุมที่ว่างเติมด้วยสีดำ\n\n"
        "**ทำไมใช้:** คนถ่ายด้วยมือถือมักถือเอียงเล็กน้อย จำกัดไว้ที่ ±20° เพื่อไม่ให้ต้นกลับหัว"),
    "RandomPerspective": (
        "บิดภาพเหมือนมองจากมุมกล้องที่ต่างไป (distortion 0.2) ใช้กับ 30% ของภาพ\n\n"
        "**ทำไมใช้:** ภาพ iNaturalist ส่วนใหญ่ถ่ายตรง ๆ แต่ผู้ใช้จริงมักก้มถ่ายจากด้านบนหรือเฉียง "
        "เพิ่มในสูตร strong เพื่อให้โมเดลทนต่อมุมกล้อง"),
    "GaussianBlur": (
        "เบลอด้วย Gaussian kernel 5 px ความแรง σ สุ่ม 0.1–2 ใช้กับ 20% ของภาพ\n\n"
        "**ทำไมใช้:** จำลองภาพมือถือที่โฟกัสพลาด มือสั่น หรือแสงน้อย ทำให้โมเดลไม่พึ่งรายละเอียดหนามเล็ก ๆ อย่างเดียว"),
    "ColorJitter": (
        "ปรับความสว่าง ±25% คอนทราสต์ ±25% ความอิ่มสี ±20% และเฉดสี (hue) ±0.02 แบบสุ่ม\n\n"
        "**ทำไมใช้:** แสงในโรงเรือน แดดจัด และกล้องแต่ละรุ่นให้สีต่างกัน\n\n"
        "**ทำไม hue น้อยมาก:** สีเป็นลักษณะสำคัญของบางชนิด เช่น ยิมโนหัวสีแดง/เหลือง ถ้าเปลี่ยนเฉดแรง label จะผิด"),
    "RandomErasing": (
        "ลบพื้นที่สี่เหลี่ยมสุ่ม (2–33% ของภาพ) ใช้กับ 25% ของภาพ ทำหลังแปลงเป็น tensor\n\n"
        "**ทำไมใช้:** บังคับให้โมเดลดูหลายส่วนของต้น ไม่พึ่งจุดเดียว เช่น ดอกสีเหลือง ซึ่ง Grad-CAM "
        "พบว่าเป็น cue แรงจนทายผิดได้ และจำลองกรณีมีของบังบางส่วน"),
    "Normalize": (
        "ลบค่าเฉลี่ยและหารส่วนเบี่ยงเบนมาตรฐานรายช่องสี ใช้ค่าจาก pretrained weights ของแต่ละโมเดลใน timm "
        "(ไม่ hard-code)\n\n"
        "**ทำไมใช้:** ทำให้ input อยู่ช่วงเดียวกับตอน pretrain บน ImageNet ทำทั้งตอนเทรน ทดสอบ และในแอป "
        "ไม่ใช่การสุ่ม จึงไม่มีแถวในแผนที่ภาพ"),
}


def fact_table(rows: list[tuple[str, str]], cols: tuple[str, str] = ("หัวข้อ", "รายละเอียด")) -> None:
    """st.table ตัดบรรทัดข้อความยาวให้ (st.dataframe ตัดทิ้งเป็น …)"""
    st.table(pd.DataFrame(rows, columns=list(cols)).set_index(cols[0]))


def project_details(cmp: pd.DataFrame) -> None:
    meta = read_csv(str(METADATA_CSV), keep_blank=True)
    meta = meta[meta.label.isin(CLASSES)]
    used = meta[meta.split != ""]
    runs = read_csv(str(RUNS_CSV), keep_blank=True)
    final = runs[(runs.phase == "final") & (runs.task == "flat")].drop_duplicates("run", keep="last")
    n_tuning = int((runs.phase == "tuning").sum())
    rules, _ = load_care_rules()

    st.subheader("โจทย์")
    st.markdown(
        f"จำแนก**กระบองเพชรแคระ 7 สกุล**ที่นิยมเลี้ยงในไทยจากภาพถ่าย และทายต่อถึง**ชนิด/ลักษณะ** รวม "
        f"{len(LEAVES)} คลาส (งาน flat: ทายคลาสย่อยตรง ๆ แล้วรวมความน่าจะเป็นขึ้นเป็นสกุล) "
        f"จากนั้นแนะนำวิธีดูแลด้วย **rule-based** (ไม่ใช่ ML) เป้าหมายคือ test accuracy ระดับสกุล ≥ {TARGET_ACC:.0%} "
        f"ตามงานวิจัยอ้างอิง และ gap ระหว่าง test กับภาพจริงไม่เกิน {REFERENCE_GAP:.2%}")
    st.markdown("สกุล: " + ", ".join(f"{CLASS_TH[g]} (*{g.capitalize()}*)" for g in CLASSES))

    st.subheader("ข้อมูล")
    c = st.columns(4)
    c[0].metric("ภาพที่ใช้", f"{len(used):,}", border=True)
    c[1].metric("ต้น (plant) ไม่ซ้ำ", f"{used.plant_id.nunique():,}", border=True)
    c[2].metric("ตรวจด้วยตาแล้ว", f"{(used.qc_status != 'pending').mean():.0%}", border=True)
    c[3].metric("ตัดทิ้งจาก QC", f"{(~meta.qc_status.isin(QC_USABLE)).sum():,}", border=True)
    table = pd.crosstab(used.label, used.split.map(SPLIT_TH), margins=True, margins_name="รวม")
    table = table.reindex([*CLASSES, "รวม"])[[*SPLIT_TH.values(), "รวม"]]
    table.index = [CLASS_TH.get(k, k) for k in table.index]
    st.dataframe(table.rename_axis("สกุล"), width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("**แหล่งที่มา**")
        src = used.source.map(SOURCE_TH).value_counts().rename_axis("แหล่ง").reset_index(name="ภาพ")
        st.dataframe(src, hide_index=True, width="stretch")
    with right:
        st.markdown("**ภาพที่ตัดทิ้งจากการ QC**")
        bad = meta[~meta.qc_status.isin(QC_USABLE)].qc_status.map(QC_TH).value_counts()
        st.dataframe(bad.rename_axis("เหตุผล").reset_index(name="ภาพ"), hide_index=True, width="stretch")
    fact_table([
        ("การแบ่งข้อมูล", f"train / val / test = {SPLIT_RATIO['train']:.0%} / {SPLIT_RATIO['val']:.0%} / "
                          f"{SPLIT_RATIO['test']:.0%} แบ่งทีละคลาสย่อย (stratified)"),
        ("กัน data leakage", "ภาพของต้นเดียวกันอยู่ split เดียวกันเสมอ (แบ่งตาม plant_id) และตัดภาพซ้ำด้วย "
                             "perceptual hash ก่อนแบ่ง ตรวจซ้ำอัตโนมัติแล้วไม่มีต้นหรือภาพซ้ำข้าม split"),
        ("ภาพจริง (realworld_test)", f"ภาพถ่ายเองที่ฟาร์ม {REALWORLD_FRAC:.0%} ของต้นแยกไว้ทดสอบ โมเดลไม่เคยเห็น "
                                     "ส่วนที่เหลือเข้า train/val/test ภาพจากเว็บไม่นับเป็นภาพจริง"),
        ("iNaturalist", "ภาพต้นปลูกในกระถาง (captive) ซึ่งไม่มีวันเป็น research grade จึงรับทุก quality grade "
                        "ที่มีคนยืนยันชื่อ ≥ 1 คน"),
        ("QC", "สุ่มตรวจ 10% → ให้โมเดล 5-fold คัดภาพน่าสงสัย 4 รอบ → ตรวจด้วยตาครบทุกภาพ (2026-10-24) "
               "ภาพปลูกลงดินยังใช้เทรนได้ ตัดเฉพาะ label ผิด / ดอกบังต้น / หลายสกุล / เบลอ"),
    ])

    st.subheader("โมเดล")
    params = cmp.set_index("model").params_m
    st.dataframe(pd.DataFrame({
        "โมเดล": [MODEL_NAMES[m] for m in MODELS], "timm": list(MODELS.values()),
        "พารามิเตอร์ (ล้าน)": [round(params.get(m, float("nan")), 1) for m in MODELS],
        "ใช้ในแอป": ["อันดับ " + str(APP_MODELS.index(m) + 1) if m in APP_MODELS else "" for m in MODELS],
    }), hide_index=True, width="stretch")
    st.caption("ทุกโมเดลเริ่มจาก ImageNet pretrained ใช้ hyperparameter ชุดเดียวกันทั้งหมดเพื่อเทียบอย่างยุติธรรม "
               "หัว classifier เริ่มที่ศูนย์ (zero-init) ทุกโมเดลเริ่ม loss เท่ากัน")

    st.subheader("การเทรน")
    fact_table([
        ("จำนวน run", f"{len(MODELS)} โมเดล × {len(SEEDS)} seeds ({', '.join(map(str, SEEDS))}) = {len(final)} run "
                      f"+ ช่วงจูน {n_tuning} run · รวมเวลาเทรน run จริง ≈ {final.minutes.sum() / 60:.1f} ชั่วโมง"),
        ("ขนาดภาพ / batch", f"{IMG_SIZE}×{IMG_SIZE} px · batch {BATCH_SIZE}"),
        ("Stage 1", f"freeze backbone เทรนเฉพาะหัว {STAGE1['epochs']} epochs · AdamW lr {STAGE1['lr']:g} "
                    "(BatchNorm ของ backbone อยู่ใน eval mode)"),
        ("Stage 2", f"fine-tune ทั้งตัว สูงสุด {STAGE2['epochs']} epochs · AdamW lr backbone "
                    f"{STAGE2['lr_backbone']:g} / หัว {STAGE2['lr_head']:g} · Cosine annealing · "
                    f"early stopping patience {STAGE2['patience']}"),
        ("epoch ที่ใช้จริง", f"stage 2 เฉลี่ย {final.stage2_epochs_run.mean():.0f} epochs "
                             f"(checkpoint ที่ดีสุดเฉลี่ย epoch {final.best_epoch.mean():.0f})"),
        ("Optimizer", f"AdamW · weight decay {WEIGHT_DECAY:g}"),
        ("Loss", f"CrossEntropy + label smoothing {LABEL_SMOOTHING} (ทำให้ค่าความมั่นใจเชื่อถือได้ขึ้น)"),
        ("Regularization", f"dropout {DROP_RATE} ก่อนหัว · EMA ของน้ำหนัก decay {EMA_DECAY} ใน stage 2"),
        ("Mixed precision", "เทรนด้วย bfloat16 autocast · ประเมินผลด้วย fp32 (ค่าความมั่นใจตรงกับแอป)"),
        ("เลือก checkpoint", "macro F1 บน validation สูงสุด (ไม่ใช้ accuracy และไม่ดู test ตอนจูน)"),
        ("ทำซ้ำได้", "ตั้ง seed ทุกที่ + deterministic algorithms → เทรนซ้ำได้ผลเดิมเป๊ะ"),
        ("ฮาร์ดแวร์", f"NVIDIA RTX 5060 Ti 16GB · PyTorch {torch.__version__} · timm {timm.__version__}"),
    ])

    st.subheader("Data augmentation")
    st.caption("ใช้กับชุด train เท่านั้น ตามลำดับนี้ · ไม่กลับบน–ล่าง และไม่เปลี่ยนสีแรง (สีเป็นลักษณะของสกุล)")
    fact_table(AUGMENT, ("ขั้น", "ทำอะไร"))
    st.caption(f"validation / test / แอป: Resize {EVAL_RESIZE} → CenterCrop {IMG_SIZE} → Normalize อย่างเดียว")
    st.markdown("**รายละเอียดแต่ละขั้น** (กดเพื่อดูคำอธิบายและตัวอย่างภาพ)")
    st.caption("ตัวอย่างใช้ภาพถ่ายจริงจากฟาร์ม 1 ภาพ ทำเฉพาะขั้นนั้นขั้นเดียว (บังคับให้ทำทุกครั้ง) สุ่ม 4 แบบ")
    fig_dir = REPORT_DIR / "figures"
    for i, (name, short) in enumerate(AUGMENT, start=1):
        with st.expander(f"{i}. {name}: {short}"):
            st.markdown(AUGMENT_DETAIL[name])
            strip = fig_dir / f"augment_{name}.png"
            if strip.exists():                                  # Normalize ไม่สุ่ม จึงไม่มีภาพ
                st.image(str(strip), width="stretch")
    if (fig_dir / "augment_all.png").exists():
        with st.expander("ทุกขั้นรวมกัน: ภาพที่โมเดลเห็นตอนเทรนจริง"):
            st.markdown("ทุกขั้นต่อกันตามลำดับด้วยความน่าจะเป็นจริงของแต่ละขั้น แต่ละ epoch ภาพเดิมจึงออกมาไม่ซ้ำกัน")
            st.image(str(fig_dir / "augment_all.png"), width="stretch")
        with st.expander("แผนที่ augmentation: ทุกขั้นใน 1 รูป"):
            st.caption("แถวคือขั้น คอลัมน์คือการสุ่ม 4 ครั้ง แถวล่างสุดคือทุกขั้นรวมกัน")
            st.image(str(fig_dir / "augment.png"), width=560)    # จำกัดความกว้าง (ภาพสูง) · จอเล็กย่อตามเอง
    else:
        st.info("ยังไม่มีภาพตัวอย่าง: รัน python -m src.visualize --only augment")

    st.subheader("การประเมินผล")
    fact_table([
        ("ชุดทดสอบ", f"test {N_IMAGES['test']} ภาพ + ภาพจริงจากฟาร์ม {N_IMAGES['realworld_test']} ภาพ"),
        ("ตัวชี้วัด", "accuracy และ macro F1 ทั้งระดับสกุลและชนิด · Gap = test − ภาพจริง"),
        ("ความน่าเชื่อถือ", "รายงานค่าเฉลี่ย ± SD จาก 3 seeds · McNemar test เทียบโมเดลรายภาพ"),
        ("ไม่แน่ใจ", f"ความมั่นใจต่ำกว่า {CONFIDENCE_THRESHOLD:.0%} แอปตอบว่าไม่แน่ใจและยังไม่แนะนำวิธีดูแล"),
        ("อธิบายผล", "Grad-CAM แสดงส่วนของภาพที่โมเดลใช้ตัดสินใจ"),
        ("ที่ลองแล้วไม่ช่วย", "test-time augmentation, ensemble หลายโมเดล, ถ่วงน้ำหนัก / undersample คลาส"),
    ])

    st.subheader("ระบบที่ใช้งาน")
    n_genus = sum(1 for _, sub in rules if sub is None)
    fact_table([
        ("วิธีดูแล (rule-based)", f"care_rules.json {n_genus} สกุล + {len(rules) - n_genus} ชนิด เรื่องน้ำ แสง ดิน "
                                  "และข้อควรระวัง พร้อมแหล่งอ้างอิง ใช้ของชนิดเมื่อมั่นใจชนิด ≥ 60% ไม่งั้นใช้ของสกุล"),
        ("แอป", "Streamlit: อัปโหลด / ถ่ายรูป (รองรับ HEIC) · เลือกหรือเทียบ 3 โมเดล · Grad-CAM"),
        ("API", "FastAPI: /api/predict และ /api/compare จำกัดไฟล์ 10 MB · มีชุดทดสอบอัตโนมัติ (pytest)"),
    ])


# ตาราง "สูตรเดิม → สูตรใหม่" ถูกเอาออก (2026-10-25): หลัง QC ครบทุกภาพ split ใหม่ test คนละชุดกัน เทียบตรง ๆ ไม่ได้
# ผลเดิมยังอยู่ reports/v1_base/ และ reports/pre_qc5/ · วิธีเลือกสูตรอยู่ docs/tuning.md รอบ 7

st.title("ประเมินผลโมเดล")
st.caption(f"ตัวเลขทั้งหมดวัดจากภาพที่โมเดลไม่เคยเห็น: ชุดทดสอบ {N_IMAGES['test']} ภาพ "
           f"และภาพถ่ายจริงจากฟาร์ม {N_IMAGES['realworld_test']} ภาพ เลือก checkpoint ด้วยชุด validation เท่านั้น")
cmp_path = REPORT_DIR / "model_compare.csv"
if not cmp_path.exists():
    st.error("ยังไม่มี reports/model_compare.csv — รัน python -m src.evaluate แล้ว python -m src.compare")
    st.stop()
compare_df = read_csv(str(cmp_path))
results_tab, about_tab = st.tabs(["ผลลัพธ์", "รายละเอียดโปรเจกต์"])
with results_tab:
    overview(compare_df)
    st.divider()
    comparison(compare_df)
    st.divider()
    drilldown()
with about_tab:
    project_details(compare_df)
