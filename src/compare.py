"""
compare.py — เทียบทุกโมเดล (run หลัก = งาน flat) จากผลของ evaluate.py → reports/model_compare.csv

    python -m src.evaluate      # ก่อน: ทายทุก checkpoint บน test + realworld_test
    python -m src.compare       # สรุป mean ± SD ข้าม seed + McNemar + แนะนำ 3 โมเดลสำหรับแอป

ต่อโมเดล (เฉลี่ยทุก seed)
    genus_acc / genus_f1       ชั้นสกุล (สกุล = ผลรวม prob ของคลาสย่อยในสกุลนั้น) บน test
    leaf_acc / leaf_f1         ถูกถึงคลาสย่อย (เฉพาะภาพที่รู้คลาสย่อย) บน test
    rw_genus_acc / rw_leaf_acc ภาพถ่ายเอง (realworld_test) ที่โมเดลไม่เคยเห็น
    gap                        genus acc: test − realworld (ยิ่งต่ำ = ใช้กับภาพจริงได้ใกล้ตัวเลขบน test)
    params_m / cpu_ms          ความเหมาะกับมือถือ
    seeds_sig_worse            กี่ seed ที่แพ้ตัวดีที่สุดของ seed นั้นอย่างมีนัยสำคัญ (McNemar p < 0.05 บนชั้นย่อย)
                               (exact McNemar บนภาพที่ 2 โมเดลตอบไม่ตรงกัน — วิธีเดียวกับงานวิจัยอ้างอิง [2])
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.config import MODEL_NAMES, MODELS, PRED_DIR, REPORT_DIR, RESULTS_CSV
from src.tasks import MAIN_TASK, run_id

N_APP_MODELS = 3


def mcnemar_p(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    only_a, only_b = int((correct_a & ~correct_b).sum()), int((~correct_a & correct_b).sum())
    return 1.0 if only_a + only_b == 0 else binomtest(only_a, only_a + only_b).pvalue


def leaf_correct(run: str) -> np.ndarray:
    """ถูก/ผิดรายภาพชั้นย่อยบน test (เรียงตาม filepath เหมือนกันทุก run)"""
    df = pd.read_csv(PRED_DIR / f"{run}_test.csv", encoding="utf-8-sig", keep_default_na=False)
    known = df[df.leaf_true != ""].sort_values("filepath")
    return (known.pred_class == known.leaf_true).to_numpy()


def main_runs() -> pd.DataFrame:
    """run หลักของงาน MAIN_TASK (ไม่นับ run จูน / scratch) จาก results.csv"""
    if not RESULTS_CSV.exists():
        raise SystemExit("ไม่มี reports/results.csv — รัน python -m src.evaluate ก่อน")
    df = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig")
    df = df[df.apply(lambda r: r.run == run_id(r.model, r.seed), axis=1)]
    if df.empty:
        raise SystemExit(f"ไม่มี run หลักของงาน {MAIN_TASK} ใน results.csv")
    return df


def add_mcnemar(df: pd.DataFrame) -> pd.DataFrame:
    """ต่อ seed: เทียบทุกโมเดลกับตัวที่ leaf acc สูงสุดของ seed นั้น"""
    df = df.copy()
    for _, group in df.groupby("seed"):
        best = group.loc[group.test_leaf_acc.idxmax(), "run"]
        ref = leaf_correct(best)
        for i, r in group.iterrows():
            df.loc[i, "mcnemar_p"] = 1.0 if r.run == best else mcnemar_p(ref, leaf_correct(r.run))
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"test_acc": "genus_acc", "test_macro_f1": "genus_f1", "test_leaf_acc": "leaf_acc",
                            "test_leaf_f1": "leaf_f1", "realworld_acc": "rw_genus_acc",
                            "realworld_leaf_acc": "rw_leaf_acc", "infer_ms_cpu": "cpu_ms"})
    metrics = [m for m in ("genus_acc", "genus_f1", "leaf_acc", "leaf_f1", "rw_genus_acc", "rw_leaf_acc", "gap",
                           "val_f1") if m in df]
    g = df.groupby("model")
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out = out.join(g[["params_m", "cpu_ms"]].mean()).join(g.size().rename("n_seeds"))
    out["seeds_sig_worse"] = (df.mcnemar_p < 0.05).groupby(df.model).sum()
    out.insert(0, "name", out.index.map(MODEL_NAMES))
    return out.sort_values("leaf_acc_mean", ascending=False).reset_index()


def main() -> None:
    out = summarize(add_mcnemar(main_runs()))
    shown = [m for m in ("genus_acc", "leaf_acc", "leaf_f1", "rw_genus_acc", "rw_leaf_acc", "gap")
             if f"{m}_mean" in out]
    print(f"\nงาน {MAIN_TASK} · เฉลี่ย {out.n_seeds.max()} seeds · rw = realworld_test · gap = test − rw (genus acc)")
    print(f"{'model':<18}" + "".join(f"{m:>15}" for m in shown) + f"{'params':>8}{'CPU ms':>8}{'sig':>5}")
    for _, r in out.iterrows():
        print(f"{r['name']:<18}" + "".join(f"{r[m + '_mean']:>9.3f}±{r[m + '_std']:.3f}" for m in shown)
              + f"{r.params_m:>7.1f}M{r.cpu_ms:>8.1f}{r.seeds_sig_worse:>5}")
    path = REPORT_DIR / "model_compare.csv"
    out.round(4).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"→ {path}")

    missing = sorted(set(MODELS) - set(out.model))
    if missing:
        print(f"⚠️  ยังไม่มีผลของ: {', '.join(missing)}")
    top = list(out.model[:N_APP_MODELS])
    print(f"\n{N_APP_MODELS} อันดับแรก (leaf acc เฉลี่ย) → config.APP_MODELS = {top}")


if __name__ == "__main__":
    main()
