"""
运营画像基线：按船型对做归一化，并对比多种评分

两个目的
------------------------------------------------------------------
1. **纠正评估口径**：13 个真值 / 44,826 条边 = 基准率 0.029%。
   在这个基准率上，top-50 命中 4 个（8% 精确率）不是"差"，
   而是**相对随机提升了约 270 倍**。报告必须用 lift，不能只报精确率——
   否则会把自己最好的结果说成很糟。

2. **船型对归一化**：实测发现评分榜首全是"港作拖轮""观光客船"这类
   **基础设施式永久关系**。同一船型对内部的共现强度本来就有系统性差异
   （拖轮之间天天见面很正常），所以应该在**同船型对内部**比较。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = ROOT / "experiments/results"


def load(year: str = "2017") -> pd.DataFrame:
    e = pd.read_csv(OUT / f"cooccurrence_edges_{year}.csv")
    h = pd.read_csv(OUT / f"cooccurrence_truth_hits_{year}.csv")
    truth = {frozenset((str(a), str(b))) for a, b in zip(h["name_a"], h["name_b"])}
    e["is_truth"] = [frozenset((str(a), str(b))) in truth
                     for a, b in zip(e["name_a"], e["name_b"])]
    # 船型对：排序后拼接，保证 (拖轮,客船) 和 (客船,拖轮) 落进同一组
    e["class_pair"] = [tuple(sorted((str(a), str(b))))
                       for a, b in zip(e["class_a"], e["class_b"])]
    return e


def build_scores(e: pd.DataFrame) -> pd.DataFrame:
    """构造多种评分，供对比。"""
    e = e.copy()

    # A. 裸强度
    e["s_raw"] = e["n_both_moving"]

    # B. 强度 × √月数（之前的评分）
    e["s_pers_sqrt"] = e["n_both_moving"] * np.sqrt(1 + e["n_months"])

    # C. 单位月强度（持久性归一）
    e["s_per_month"] = e["n_both_moving"] / (1 + e["n_months"])

    # D. ★ 船型对内 z-score：这条边相对"同类型关系"有多异常
    grp = e.groupby("class_pair")["s_per_month"]
    mu, sd = grp.transform("mean"), grp.transform("std")
    e["s_class_z"] = ((e["s_per_month"] - mu) / sd.replace(0, np.nan)).fillna(0)

    # E. ★ 船型对内百分位 + 强度下限（避免"小组内唯一一条"被误判为最异常）
    e["s_class_pct"] = grp.rank(pct=True)
    e["s_class_pct_w"] = e["s_class_pct"] * np.log1p(e["n_both_moving"])

    return e


def report(e: pd.DataFrame, scores: list[str], n_truth: int) -> pd.DataFrame:
    base_rate = n_truth / len(e)

    print(f"\n{'=' * 92}")
    print(f"多评分对比　（总边数 {len(e):,}，真值 {n_truth} 条，"
          f"基准率 {base_rate * 100:.4f}%）")
    print(f"{'=' * 92}")
    print(f"  ★ 关键：精确率必须跟基准率比才有意义。lift = 精确率 / 基准率")
    print(f"     随机猜的话，报警 K 条期望命中 {base_rate * 100:.4f}% × K 条")

    rows = []
    for s in scores:
        print(f"\n  ── {s} ──")
        print(f"  {'K':>6} │ {'捞回':>5} {'召回率':>7} {'精确率':>7} {'随机期望':>9} {'lift':>8}")
        for k in [10, 20, 50, 100, 200, 500, 1000]:
            top = e.nlargest(k, s)
            hits = int(top["is_truth"].sum())
            prec = hits / k
            exp = base_rate * k
            lift = prec / base_rate if base_rate else 0
            print(f"  {k:>6} │ {hits:>5} {hits / n_truth * 100:>6.1f}% "
                  f"{prec * 100:>6.2f}% {exp:>9.2f} {lift:>7.0f}x")
            rows.append({"score": s, "K": k, "hits": hits,
                         "recall": hits / n_truth, "precision": prec, "lift": lift})

    return pd.DataFrame(rows)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2017")
    a = ap.parse_args()
    YEAR = a.year
    e = load(YEAR)
    n_truth = int(e["is_truth"].sum())
    e = build_scores(e)

    scores = ["s_raw", "s_pers_sqrt", "s_per_month", "s_class_z", "s_class_pct_w"]
    res = report(e, scores, n_truth)

    # --- 船型对基线速览 ---
    print(f"\n{'=' * 92}")
    print("船型对的共现强度基线（解释为什么必须做船型归一化）")
    print(f"{'=' * 92}")
    cp = (e.groupby("class_pair")
            .agg(n=("n_both_moving", "size"),
                 med=("n_both_moving", "median"),
                 mx=("n_both_moving", "max"),
                 n_truth=("is_truth", "sum"))
            .sort_values("n", ascending=False).head(12))
    print(f"  {'船型对':<44} {'边数':>7} {'强度中位':>9} {'强度最大':>9} {'真值':>5}")
    for idx, r in cp.iterrows():
        print(f"  {str(idx):<44} {int(r['n']):>7,} {r['med']:>9.0f} "
              f"{r['mx']:>9.0f} {int(r['n_truth']):>5}")

    # --- 最佳评分下的 top 12 ---
    best = res.groupby("score")["lift"].apply(lambda s: s.iloc[1:5].mean())  # K=20..200 平均 lift
    best_name = best.idxmax()
    print(f"\n{'=' * 92}")
    print(f"表现最好的评分：{best_name}（K=20~200 平均 lift {best.max():.0f}x）")
    print(f"{'=' * 92}")
    for i, (_, r) in enumerate(e.nlargest(12, best_name).iterrows(), 1):
        tag = "  ★真值事件" if r["is_truth"] else ""
        print(f"  {i:>3}. {str(r['name_a'])[:17]:<17}({str(r['class_a'])[:9]:<9})"
              f" — {str(r['name_b'])[:17]:<17}({str(r['class_b'])[:9]:<9})"
              f" 航行{int(r['n_both_moving']):>5} 月{int(r['n_months']):>3}"
              f" 船型内百分位{r['s_class_pct']:>5.2f}{tag}")

    res.to_csv(OUT / f"score_comparison_{YEAR}.csv", index=False, encoding="utf-8-sig")
    e.to_csv(OUT / f"edges_with_class_baseline_{YEAR}.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT / f'score_comparison_{YEAR}.csv'}")
    print(f"  → {OUT / f'edges_with_class_baseline_{YEAR}.csv'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
