"""
持久性分析：验证「共现越强越可疑」这个朴素假设是否成立

背景
------------------------------------------------------------------
全年评估得到一个反直觉的结果：评分 Top-12 的边**全部是 12 个月都出现、
窗口数 1000–3700 的"永久关系"**（港作拖轮、观光客船），
而真值事件（拖带/救援）的共同航行窗口只有 6–800。

这提示一个假设：

    **共现越强，越可能是常规运营，而不是异常。**

本脚本验证这个假设，并测试一个替代评分：
    用「单位月强度」= 共同航行窗口 / (1 + 出现月数)
把"永久基础设施式关系"压下去，把"短期高强度事件"顶上来。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    e = pd.read_csv(ROOT / "experiments/results/cooccurrence_edges_2017.csv")
    h = pd.read_csv(ROOT / "experiments/results/cooccurrence_truth_hits_2017.csv")

    # 真值命中表里存的是船名对（当初没存 MMSI，用名字匹配）
    truth_pairs = {frozenset((str(a), str(b)))
                   for a, b in zip(h["name_a"], h["name_b"])}
    e["is_truth"] = [frozenset((str(a), str(b))) in truth_pairs
                     for a, b in zip(e["name_a"], e["name_b"])]

    print("=" * 76)
    print("持久性分析")
    print("=" * 76)
    print(f"  总边数 {len(e):,}   真值命中 {int(e['is_truth'].sum())} 条")

    # ---- 假设检验：持久性 ----
    m = e[e["is_truth"]]
    print(f"\n【假设】共现越强越可疑  →  验证")
    print(f"  全体边    ：出现月数 中位 {e['n_months'].median():.0f}，"
          f"共同航行窗口 中位 {e['n_both_moving'].median():.0f}")
    print(f"  真值命中边：出现月数 中位 {m['n_months'].median():.0f}，"
          f"共同航行窗口 中位 {m['n_both_moving'].median():.0f}")
    print(f"  12/12 月的永久关系边占比：{(e['n_months'] == 12).mean() * 100:.1f}%"
          f"   其中真值只占 {int(e.loc[e['n_months'] == 12, 'is_truth'].sum())} 条")

    # ---- 替代评分：单位月强度 ----
    e["score_base"] = e["n_both_moving"] * (1 + e["n_months"]).apply(lambda x: x ** 0.5)
    e["score_pers"] = e["n_both_moving"] / (1 + e["n_months"])

    print(f"\n{'=' * 76}")
    print("两种评分对比：top-K 能捞回多少真值事件")
    print(f"{'=' * 76}")
    print(f"  {'K':>7} │ {'原评分(n_both_moving×√月数)':>28} │ {'新评分(÷(1+月数))':>22}")
    print(f"  {'':>7} │ {'捞回':>10} {'召回率':>8} {'精确率':>7} │ {'捞回':>8} {'召回率':>7} {'精确率':>6}")
    print(f"  {'-' * 74}")

    n_truth = int(e["is_truth"].sum())
    rows = []
    for k in [10, 50, 100, 200, 500, 1000, 2000]:
        b = e.nlargest(k, "score_base")
        p = e.nlargest(k, "score_pers")
        hb, hp = int(b["is_truth"].sum()), int(p["is_truth"].sum())
        print(f"  {k:>7,} │ {hb:>10} {hb / n_truth * 100:>7.1f}% {hb / k * 100:>6.1f}% │ "
              f"{hp:>8} {hp / n_truth * 100:>6.1f}% {hp / k * 100:>5.1f}%")
        rows.append({"K": k, "base_hits": hb, "pers_hits": hp})

    # ---- 新评分下的真值排名 ----
    e = e.sort_values("score_pers", ascending=False).reset_index(drop=True)
    e["rank_pers"] = e.index + 1
    t = e[e["is_truth"]]
    print(f"\n  新评分下真值事件的排名：{sorted(t['rank_pers'].tolist())}")
    print(f"    中位排名 {t['rank_pers'].median():.0f} / {len(e):,}"
          f"    最好 {t['rank_pers'].min()}    最差 {t['rank_pers'].max()}")

    # ---- 新 Top 10 ----
    print(f"\n{'=' * 76}")
    print("新评分 Top 10（看是否还是那些永久关系）")
    print(f"{'=' * 76}")
    for i, r in e.head(10).iterrows():
        tag = "  ★真值事件" if r["is_truth"] else ""
        print(f"  {i + 1:>3}. {str(r['name_a'])[:17]:<17}({str(r['class_a'])[:9]:<9})"
              f" — {str(r['name_b'])[:17]:<17}({str(r['class_b'])[:9]:<9})"
              f" 航行{int(r['n_both_moving']):>5} 月{int(r['n_months']):>3}"
              f" 分{r['score_pers']:>7.1f}{tag}")

    # ---- 落盘 ----
    out = ROOT / "experiments/results/persistence_analysis_2017.csv"
    e.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  → {out}")

    # ---- 结论 ----
    best = max(rows, key=lambda r: r["pers_hits"])
    print(f"\n{'=' * 76}")
    print("结论")
    print(f"{'=' * 76}")
    base_at = {r['K']: r['base_hits'] for r in rows}
    pers_at = {r['K']: r['pers_hits'] for r in rows}
    if pers_at.get(500, 0) > base_at.get(500, 0):
        print(f"  ✅ 「单位月强度」评分在 top-500 上捞回更多真值"
              f"（{pers_at[500]} vs {base_at[500]}），持久性归一化有效。")
    else:
        print(f"  ⚠️ 持久性归一化未带来 top-K 提升"
              f"（top-500: {pers_at.get(500, 0)} vs {base_at.get(500, 0)}）。")
        print(f"     说明真值事件与常规运营在「强度」维度上仍然重叠，")
        print(f"     需要更细的运营画像基线（按船型/区域分别建模）。")
    print(f"{'=' * 76}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
