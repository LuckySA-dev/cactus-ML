"""
visualize.py — กราฟสำหรับรายงาน (matplotlib + seaborn) → reports/figures/*.png

    python -m src.visualize                      # ทุกกราฟที่มีข้อมูลพอ
    python -m src.visualize --only models curves # เฉพาะบางกราฟ

| ชื่อ       | ไฟล์                        | อ่านจาก                          |
|-----------|-----------------------------|----------------------------------|
| dataset   | dataset.png                 | data/metadata.csv                |
| tuning    | tuning.png                  | reports/runs.csv (phase=tuning)  |
| curves    | learning_curves.png         | reports/logs/<run หลัก seed42>.csv |
| models    | model_comparison.png        | reports/results.csv              |
| per_class | per_class_f1.png            | reports/predictions/*_test.csv   |
| confusion | cm_<model>_all_seeds.png    | reports/predictions/*_test.csv   |
| confidence| confidence.png              | reports/predictions/*_test.csv   |

run หลัก = งาน tasks.MAIN_TASK (flat) · กราฟชั้นสกุลใช้สกุลที่อนุมานจากคลาสย่อย (evaluate.py)
ตัวเลขทุกกราฟมีเป็นตาราง CSV อยู่แล้ว — กราฟเป็นตัวช่วยอ่าน ไม่ใช่ที่เดียวที่มีข้อมูล
สี: dataviz reference palette — 6 สีโมเดลผ่าน validate_palette.js (adjacent, light)
    สีตามโมเดล (entity) ไม่ใช่ตามลำดับ → โมเดลเดียวกันสีเดียวกันทุกกราฟ
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, f1_score

from src.config import (
    CLASSES,
    CONFIDENCE_THRESHOLD,
    FIGURE_DIR,
    LOG_DIR,
    METADATA_CSV,
    MODEL_NAMES,
    MODELS,
    PRED_DIR,
    QC_USABLE,
    RESULTS_CSV,
    RUNS_CSV,
    SEED,
)
from src.tasks import run_id

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# 6 สีตามลำดับ palette (ผ่าน validate_palette.js แบบ adjacent) — มีป้ายชื่อทุกจุด จึงไม่พึ่งสีอย่างเดียว
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MODEL_COLOR = dict(zip(MODELS, SERIES, strict=True))
BLUE_ACCENT, BLUE_LIGHT = "#256abf", "#86b6ef"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


# ---------------------------------------------------------------------------
# style + helpers
# ---------------------------------------------------------------------------
def new_figure(*args, **kwargs):
    """plt.subplots ที่ตั้ง style ก่อนทุกครั้ง — เส้น grid/แกนจาง ให้ข้อมูลเด่น"""
    plt.switch_backend("Agg")                    # เซฟไฟล์อย่างเดียว ไม่เปิดหน้าต่าง
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2, "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
    })
    return plt.subplots(*args, **kwargs)


def save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path.relative_to(FIGURE_DIR.parents[1])}")


def label_models(ax, models: list[str]) -> None:
    ax.set_xticks(range(len(models)), [MODEL_NAMES[m] for m in models], rotation=30, ha="right",
                  rotation_mode="anchor")
    ax.grid(axis="x", visible=False)


def value_note(ax, title: str, values: dict[str, str], x: float, ha: str) -> None:
    """ค่าของแต่ละโมเดล ณ จุดเดียว เป็นกล่องข้อความ — แทนการแปะเลขบนเส้นที่ซ้อนกันจนอ่านไม่ออก"""
    lines = [title] + [f"{MODEL_NAMES[m]}  {v}" for m, v in values.items()]
    ax.text(x, 0.06, "\n".join(lines), transform=ax.transAxes, ha=ha, va="bottom", fontsize=8, color=INK2,
            linespacing=1.5)


def final_runs() -> pd.DataFrame:
    """run หลักของงาน MAIN_TASK (ไม่นับ scratch / tag จูน) จาก results.csv"""
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig")
    return df[[run == run_id(m, s) for run, m, s in zip(df.run, df.model, df.seed, strict=True)]]


def load_predictions(split: str = "test") -> pd.DataFrame:
    """ผลทำนายรายภาพของทุก run หลัก ต่อกันเป็นตารางเดียว (+ model, seed)"""
    runs = final_runs()
    frames = [pd.read_csv(PRED_DIR / f"{r.run}_{split}.csv", encoding="utf-8-sig").assign(model=r.model, seed=r.seed)
              for r in runs.itertuples() if (PRED_DIR / f"{r.run}_{split}.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def to_index(labels: pd.Series) -> np.ndarray:
    return labels.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, name: str) -> None:
    """confusion matrix normalize ตามแถว (เส้นทแยง = recall) — ใช้ทั้งจาก evaluate.py และรวมหลาย seed"""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)), normalize="true")
    empty = cm < 0.005                                               # ช่อง 0 เว้นว่าง ไม่ลงสี ไม่ใส่เลข
    fig, ax = new_figure(figsize=(6.6, 5.6))
    sns.heatmap(cm, ax=ax, mask=empty, cmap=BLUES, vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=2, linecolor=SURFACE, square=True, xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={"label": "share of true class", "shrink": 0.8})
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    save(fig, name)


def fig_dataset() -> None:
    """ภาพต่อคลาสหลัง QC: ใช้เทรน (กระถาง / ปลูกลงดิน) vs ถูกตัด"""
    meta = pd.read_csv(METADATA_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    meta = meta[meta.label.isin(CLASSES)]
    usable = meta.qc_status.isin(QC_USABLE)
    parts = pd.DataFrame({
        "used · potted": (usable & (meta.qc_status != "not_potted")).groupby(meta.label).sum(),
        "used · planted in ground": (meta.qc_status == "not_potted").groupby(meta.label).sum(),
        "removed by QC": (~usable).groupby(meta.label).sum(),
    }).reindex(CLASSES[::-1])
    fig, ax = new_figure(figsize=(7.5, 3.6))
    left = np.zeros(len(parts))
    for col, color in zip(parts, [BLUE_ACCENT, BLUE_LIGHT, AXIS], strict=True):
        ax.barh(parts.index, parts[col], left=left, height=0.62, color=color, label=col,
                edgecolor=SURFACE, linewidth=1.5)               # ช่องว่าง 2px ระหว่างส่วน
        left += parts[col].to_numpy()
    for y, (used, total) in enumerate(zip(parts.iloc[:, :2].sum(axis=1), left, strict=True)):
        ax.annotate(f"{used} used", (total, y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
    ax.set(title=f"Images per class after QC ({int(parts.iloc[:, :2].to_numpy().sum()):,} used)",
           xlabel="images")
    ax.grid(axis="y", visible=False)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "dataset")


def fig_tuning() -> None:
    """(A) lr ของ stage 2 vs val F1 — EffNetV2-B0  (B) lr เดิม vs lr ใหม่ ทุกโมเดล"""
    if not RUNS_CSV.exists():
        return print("  ข้าม tuning: ไม่มี reports/runs.csv")
    runs = pd.read_csv(RUNS_CSV, encoding="utf-8-sig")
    tune = runs[(runs.phase == "tuning") & (runs.model == "effnetv2b0") & (runs.epochs2 == 40)
                & (runs.weight_decay == 1e-4)]
    fig, (a, b) = new_figure(1, 2, figsize=(9.5, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})

    sweep = tune[tune.seed == SEED].groupby("lr_backbone").best_val_f1.max()
    a.plot(sweep.index, sweep.values, "-o", color=MODEL_COLOR["effnetv2b0"], markersize=7, mec=SURFACE, mew=1.5,
           label="seed 42")
    others = tune[tune.seed != SEED]
    a.scatter(others.lr_backbone, others.best_val_f1, s=26, color=MODEL_COLOR["effnetv2b0"], alpha=0.45,
              edgecolors=SURFACE, label="seeds 43, 44")
    for lr, f1 in sweep.items():
        a.annotate(f"{f1:.3f}", (lr, f1), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8,
                   color=INK2)
    a.set(xscale="log", xlim=(6e-6, 1.7e-3), xlabel="stage-2 backbone lr (head lr = 10×)",
          ylabel="best val macro F1", title="A · Learning rate sweep (EfficientNetV2-B0)")
    a.legend(loc="lower right")

    old = runs[(runs.phase == "tuning") & (runs.lr_backbone == 1e-5) & (runs.epochs2 == 40) & (runs.seed == SEED)]
    new = runs[(runs.phase == "tuning") & (runs.lr_backbone == 3e-4) & (runs.epochs2 == 40) & (runs.seed == SEED)
               & (runs.weight_decay == 1e-4)]
    models = [m for m in MODELS if m in set(old.model) and m in set(new.model)]
    for i, m in enumerate(models):
        o, n = old[old.model == m].best_val_f1.iloc[0], new[new.model == m].best_val_f1.iloc[0]
        b.plot([i, i], [o, n], color=GRID, linewidth=3, zorder=1)
        b.scatter(i, o, s=48, color=MUTED, edgecolors=SURFACE, zorder=2)
        b.scatter(i, n, s=48, color=MODEL_COLOR[m], edgecolors=SURFACE, zorder=2)
        b.annotate(f"+{(n - o) * 100:.1f}", (i, (o + n) / 2), xytext=(8, 0), textcoords="offset points",
                   va="center", fontsize=8, color=INK2)
    label_models(b, models)
    b.set(ylabel="best val macro F1", title="B · Plan lr vs tuned lr (seed 42)")
    # legend ทำเอง: จุด tuned มีสีตามโมเดล — ถ้าใช้ label ของจุดแรกจะได้สี MobileNet ซึ่งชวนเข้าใจผิด
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label="plan lr 1e-5"),
                      Line2D([], [], marker="o", ls="", color=INK2, label="tuned lr 3e-4 (model colour)")],
             loc="lower center")
    save(fig, "tuning")


def fig_learning_curves() -> None:
    """loss / accuracy ต่อ epoch (seed 42) — train สีเทา, val สีของโมเดล, เส้นแบ่ง stage"""
    models = [m for m in MODELS if (LOG_DIR / f"{run_id(m, SEED)}.csv").exists()]
    if not models:
        return print("  ข้าม curves: ไม่มี log")
    fig, axes = new_figure(len(models), 2, figsize=(9.5, 2.6 * len(models)), squeeze=False)
    for row, m in zip(axes, models, strict=True):
        log = pd.read_csv(LOG_DIR / f"{run_id(m, SEED)}.csv", encoding="utf-8-sig")
        log["x"] = np.arange(1, len(log) + 1)                     # นับ epoch ต่อกันทั้ง 2 stage
        stage2 = log.loc[log.stage == 2, "x"].min()
        best = log[log.is_best == 1].iloc[-1]                    # checkpoint ที่ถูกเซฟจริง
        for ax, metric in zip(row, ["loss", "acc"], strict=True):
            ax.plot(log.x, log[f"train_{metric}"], color=AXIS, label="train")
            ax.plot(log.x, log[f"val_{metric}"], color=MODEL_COLOR[m], label="val")
            ax.axvline(stage2 - 0.5, color=AXIS, linewidth=0.8)
            ax.annotate("fine-tune →", (stage2 - 0.5, 1), xycoords=("data", "axes fraction"), xytext=(4, -4),
                        textcoords="offset points", va="top", fontsize=7, color=MUTED)
            ax.set(ylabel="loss" if metric == "loss" else "accuracy")
        row[1].plot(best.x, best.val_acc, "o", markersize=8, color=MODEL_COLOR[m], mec=SURFACE, mew=2)
        row[1].text(0.98, 0.05, f"● saved checkpoint: epoch {int(best.x)}, val F1 {best.val_f1:.3f}",
                    transform=row[1].transAxes, ha="right", fontsize=8, color=INK2)
        row[0].set_title(f"{MODEL_NAMES[m]} · loss")
        row[1].set_title(f"{MODEL_NAMES[m]} · accuracy")
    axes[0][0].legend(loc="upper right")                          # loss ลดลง → มุมบนขวาว่าง
    for ax in axes[-1]:
        ax.set_xlabel("epoch (stage 1 → stage 2)")
    fig.tight_layout()
    save(fig, "learning_curves")


def fig_model_comparison() -> None:
    """ความแม่นยำ (จุด = แต่ละ seed, วงใหญ่ = mean ± SD) + ขนาด/ความเร็ว — 1 แผง 1 หน่วย ไม่มีแกนคู่"""
    runs = final_runs()
    if runs.empty:
        return print("  ข้าม models: ไม่มี results.csv")
    models = [m for m in MODELS if m in set(runs.model)]
    metrics = [("test_acc", "Genus accuracy · test"), ("test_leaf_acc", "Species accuracy · test")]
    if "realworld_acc" in runs and runs.realworld_acc.notna().any():
        metrics.append(("realworld_acc", "Genus accuracy · real-world"))
    costs = [("params_m", "Parameters (M)"), ("infer_ms_cpu", "CPU ms / image")]
    fig, axes = new_figure(1, len(metrics) + len(costs), figsize=(2.9 * (len(metrics) + len(costs)), 3.6))
    for ax, (col, title) in zip(axes, metrics, strict=False):
        for i, m in enumerate(models):
            v = runs.loc[runs.model == m, col]
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=22, color=MODEL_COLOR[m],
                       alpha=0.45, edgecolors=SURFACE)
            ax.errorbar(i, v.mean(), yerr=v.std() if len(v) > 1 else None, fmt="o", markersize=8,
                        color=MODEL_COLOR[m], mec=SURFACE, mew=2, elinewidth=2)
            ax.annotate(f"{v.mean():.3f}", (i, v.mean()), xytext=(12, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK2)
        ax.set_title(title)
        ax.set_xlim(-0.5, len(models) - 0.3)
        label_models(ax, models)
    for ax, (col, title) in zip(axes[len(metrics):], costs, strict=True):
        vals = [runs.loc[runs.model == m, col].mean() for m in models]
        ax.bar(range(len(models)), vals, width=0.55, color=[MODEL_COLOR[m] for m in models])
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.1f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                        color=INK2)
        ax.set_title(title)
        label_models(ax, models)
    n_seeds = runs.groupby("model").size().max()
    fig.suptitle(f"Model comparison · mean ± SD over {n_seeds} seeds (small dots = individual seeds)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "model_comparison")


def fig_per_class(preds: pd.DataFrame) -> None:
    """F1 รายคลาส × โมเดล (เฉลี่ยทุก seed) — หาคลาสที่ยากที่สุด"""
    models = [m for m in MODELS if m in set(preds.model)]
    table = pd.DataFrame({
        MODEL_NAMES[m]: np.mean([f1_score(g.y_true, g.y_pred, labels=CLASSES, average=None, zero_division=0)
                                 for _, g in preds[preds.model == m].groupby("seed")], axis=0)
        for m in models
    }, index=CLASSES)
    vmin = min(0.8, np.floor(table.to_numpy().min() * 20) / 20)     # สเกลเริ่มใกล้ค่าต่ำสุด → เห็นความต่าง
    fig, ax = new_figure(figsize=(1.6 * len(models) + 2.6, 4))
    sns.heatmap(table, ax=ax, cmap=BLUES, vmin=vmin, vmax=1, annot=True, fmt=".3f", annot_kws={"size": 8},
                linewidths=2, linecolor=SURFACE, cbar_kws={"label": f"F1 (scale starts at {vmin:.2f})", "shrink": 0.8})
    ax.set(title="Per-class test F1 · mean over seeds", xlabel="", ylabel="")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=0)
    save(fig, "per_class_f1")


def fig_confusion(preds: pd.DataFrame) -> None:
    """confusion matrix รวมทุก seed ต่อโมเดล — นิ่งกว่าดูทีละ run"""
    for m in [m for m in MODELS if m in set(preds.model)]:
        sub = preds[preds.model == m]
        plot_confusion(to_index(sub.y_true), to_index(sub.y_pred),
                       f"{MODEL_NAMES[m]} · test · {sub.seed.nunique()} seeds pooled", f"cm_{m}_all_seeds")


def fig_confidence(preds: pd.DataFrame) -> None:
    """threshold ความมั่นใจ: แอปตอบกี่ % ของภาพ และตอบถูกกี่ % — เลือก CONFIDENCE_THRESHOLD"""
    thresholds = np.round(np.arange(0, 0.96, 0.01), 2)
    at = int(np.searchsorted(thresholds, CONFIDENCE_THRESHOLD))
    fig, (a, b) = new_figure(1, 2, figsize=(9.5, 3.3))
    cov_at, acc_at = {}, {}
    for m in [m for m in MODELS if m in set(preds.model)]:
        sub = preds[preds.model == m]
        conf, correct = sub.confidence.to_numpy(), sub.correct.to_numpy().astype(bool)
        coverage = [(conf >= t).mean() for t in thresholds]
        accuracy = [correct[conf >= t].mean() if (conf >= t).any() else np.nan for t in thresholds]
        a.plot(thresholds, coverage, color=MODEL_COLOR[m], label=MODEL_NAMES[m])
        b.plot(thresholds, accuracy, color=MODEL_COLOR[m], label=MODEL_NAMES[m])
        cov_at[m], acc_at[m] = f"{coverage[at]:.0%}", f"{accuracy[at]:.1%}"
    for ax in (a, b):
        ax.axvline(CONFIDENCE_THRESHOLD, color=AXIS, linewidth=0.8)
        ax.set_xlabel("confidence threshold (max softmax)")
    value_note(a, f"at {CONFIDENCE_THRESHOLD}:", cov_at, 0.42, "left")
    value_note(b, f"at {CONFIDENCE_THRESHOLD}:", acc_at, 0.97, "right")
    a.set(title=f"A · Share of photos the app answers (threshold {CONFIDENCE_THRESHOLD})", ylabel="coverage")
    b.set(title="B · Accuracy on answered photos", ylabel="accuracy")
    a.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "confidence")


FIGURES = {
    "dataset": fig_dataset,
    "tuning": fig_tuning,
    "curves": fig_learning_curves,
    "models": fig_model_comparison,
    "per_class": fig_per_class,
    "confusion": fig_confusion,
    "confidence": fig_confidence,
}
NEEDS_PREDICTIONS = {"per_class", "confusion", "confidence"}


def main() -> None:
    p = argparse.ArgumentParser(description="สร้างกราฟสำหรับรายงาน → reports/figures/")
    p.add_argument("--only", nargs="+", choices=list(FIGURES), help="เลือกเฉพาะบางกราฟ")
    args = p.parse_args()

    names = args.only or list(FIGURES)
    preds = load_predictions() if NEEDS_PREDICTIONS & set(names) else pd.DataFrame()
    for name in names:
        print(name)
        if name in NEEDS_PREDICTIONS:
            if preds.empty:
                print("  ข้าม: ไม่มี predictions — รัน python -m src.evaluate ก่อน")
                continue
            FIGURES[name](preds)
        else:
            FIGURES[name]()


if __name__ == "__main__":
    main()
