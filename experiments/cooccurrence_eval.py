"""
主锚点核心实验：多月份共现检测 + 用真实标注事件评估效果

为什么需要这个实验
------------------------------------------------------------------
单个事件样本太小 —— 2017_01 只有 3 起事件、其中 1 起涉及两船。
用 n=1 算召回率没有意义。必须跨多个月累积，才能得到有统计意义的数字。

做法
------------------------------------------------------------------
  1. 逐月载入 AIS，构建船舶共现边（沿用 src/network_analysis.py 的同一套逻辑）
  2. 跨月累积：同一条边在不同月份的共现窗口数相加
  3. 载入全部 154 起真值事件，挑出**涉及两船及以上**的
  4. 检查这些真值船对在共现网络里的排名 —— 这就是**召回率 / 精确率**

为什么要看"排名"而不只是"是否命中"
------------------------------------------------------------------
  真实数据里共现边有上万条，如果只是"命中/未命中"，
  一个把所有边都标成异常的检测器也能拿 100% 召回。
  真正有意义的是：**在只报警 top-K 条边的前提下，能捞回多少真值事件**
  —— 这正是银行风控的真实场景（人力有限，只能复核 top-K）。

用法
------------------------------------------------------------------
    .venv/Scripts/python.exe experiments/cooccurrence_eval.py --year 2017
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from network_analysis import (  # noqa: E402
    INCIDENT_CSV, build_cooccurrence_edges, load_month,
)

OUT_DIR = ROOT / "experiments/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def accumulate_edges(year: str, config: dict | None = None) -> pd.DataFrame:
    """逐月构建共现边并跨月累积。

    ⚠️ 跨月累积要注意：同一条边在不同月份的 n_windows 直接相加是有意义的
    （每个月都是独立的时间窗计数），但 **lift 不能相加** —— lift 是比值，
    必须用累积后的总量重算。这里先累积原始计数，最后统一重算 lift。
    """
    frames = []
    for m in range(1, 13):
        month = f"{year}_{m:02d}"
        try:
            df = load_month(month)
        except FileNotFoundError:
            print(f"  ⚠️ 跳过 {month}（文件不存在）")
            continue
        print(f"\n[{month}] {len(df):,} 点 / {df['MMSI'].nunique()} 船")
        e = build_cooccurrence_edges(df, config)
        if not e.empty:
            e["month"] = month
            frames.append(e)

    if not frames:
        raise RuntimeError("没有任何月份的数据")

    all_e = pd.concat(frames, ignore_index=True)

    # --- 跨月合并 ---
    keys = ["mmsi_a", "mmsi_b"]
    agg = (all_e.groupby(keys, as_index=False)
                .agg(n_windows=("n_windows", "sum"),
                     n_both_stopped=("n_both_stopped", "sum"),
                     n_both_moving=("n_both_moving", "sum"),
                     n_months=("month", "nunique"),
                     name_a=("name_a", "first"), name_b=("name_b", "first"),
                     class_a=("class_a", "first"), class_b=("class_b", "first")))
    agg["mooring_ratio"] = agg["n_both_stopped"] / agg["n_windows"]
    agg["moving_ratio"] = agg["n_both_moving"] / agg["n_windows"]
    print(f"\n跨月合并：{len(all_e):,} 条月度边 → {len(agg):,} 条唯一船对")
    return agg


def score_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """给每条边打一个"可疑度"分。

    设计原则（来自实测教训）：
      · 不能用原始共现权重 —— 实测 Top 全是观光客船/港作拖轮（正常运营）
      · 不能用纯 lift —— 实测 Top 全是稀有船（附属艇/军舰），lift 虚高
      · **核心信号是「双方都在航行时仍长期同框」**：
        正常航运里两艘船不该反复同速同行，这才是海上过驳/拖带/结伴的物理证据
      · 再叠加"跨月复现"：偶发一次不算，反复出现才是稳定关系

    最终分数 = 共同航行窗口数 × 跨月稳定因子
    """
    e = edges.copy()
    # 共同航行窗口数已经是"小时"量级（1 窗口 = 1 小时）
    e["score"] = e["n_both_moving"] * np.log1p(e["n_months"])
    return e.sort_values("score", ascending=False).reset_index(drop=True)


def evaluate(edges: pd.DataFrame, months_covered: set[str]) -> dict:
    """用真值事件评估：top-K 报警能捞回多少真事件。"""
    inc = pd.read_csv(INCIDENT_CSV)
    inc["_t"] = pd.to_datetime(inc["ais_incident_start_bound_hst"], errors="coerce", utc=True)
    inc["_month"] = inc["_t"].dt.strftime("%Y_%m")
    inc = inc[inc["_month"].isin(months_covered)]

    size = inc.groupby("incident_num")["MMSI"].nunique()
    multi = size[size >= 2].index
    inc_m = inc[inc["incident_num"].isin(multi)]

    # 真值船对集合
    truth = {}
    for inum, g in inc_m.groupby("incident_num"):
        for a, b in combinations(sorted(g["MMSI"].unique()), 2):
            truth[frozenset((a, b))] = inum

    print(f"\n{'=' * 74}")
    print(f"真值评估：覆盖 {len(months_covered)} 个月")
    print(f"{'=' * 74}")
    print(f"  该时间范围内的两船事件：{len(truth)} 个船对"
          f"（涉及 {inc_m['incident_num'].nunique()} 起事故）")

    if not truth:
        return {"n_truth": 0}

    # 排名
    edges = edges.reset_index(drop=True)
    edges["rank"] = np.arange(1, len(edges) + 1)

    found = []
    for _, r in edges.iterrows():
        key = frozenset((r["mmsi_a"], r["mmsi_b"]))
        if key in truth:
            found.append({"incident_num": truth[key], "rank": r["rank"],
                          "score": r["score"], "n_both_moving": r["n_both_moving"],
                          "name_a": r["name_a"], "name_b": r["name_b"],
                          "class_a": r["class_a"], "class_b": r["class_b"]})

    f = pd.DataFrame(found)
    print(f"  在共现网络中被找到：{len(f)} / {len(truth)}"
          f"  = **召回 {len(f) / len(truth) * 100:.1f}%**")

    # 但"被找到"不等于"被排在前面" —— 这才是关键
    print(f"\n  ★ 关键：只报警 top-K 条边时的表现（银行风控的真实约束）")
    print(f"     {'K':>7} {'报警边数':>9} {'捞回真事件':>10} {'召回率':>8} {'精确率':>8}")
    rows = []
    for k in [10, 50, 100, 500, 1000, len(edges)]:
        k = min(k, len(edges))
        topk = edges.head(k)
        hits = sum(1 for _, r in topk.iterrows()
                   if frozenset((r["mmsi_a"], r["mmsi_b"])) in truth)
        rec = hits / len(truth)
        prec = hits / k
        print(f"     {k:>7,} {k:>9,} {hits:>10} {rec * 100:>7.1f}% {prec * 100:>7.1f}%")
        rows.append({"K": k, "hits": hits, "recall": rec, "precision": prec})

    if not f.empty:
        print(f"\n  命中事件的排名分布：")
        print(f"     中位排名 {f['rank'].median():.0f} / {len(edges):,}"
              f"   最好 {f['rank'].min()}   最差 {f['rank'].max()}")
        print(f"\n  命中的真值船对（按排名）：")
        for _, r in f.sort_values("rank").iterrows():
            print(f"     排名 {r['rank']:>5}  {str(r['name_a'])[:18]:<18}"
                  f"({str(r['class_a'])[:11]:<11}) ↔ {str(r['name_b'])[:18]:<18}"
                  f"({str(r['class_b'])[:11]:<11}) 共同航行 {int(r['n_both_moving']):>4}")

    return {"n_truth": len(truth), "n_found": len(f),
            "recall": len(f) / len(truth), "curve": rows, "found": f}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="主锚点核心实验")
    ap.add_argument("--year", default="2017")
    args = ap.parse_args(argv)

    print(f"\n{'=' * 74}")
    print(f"主锚点核心实验：{args.year} 年全年共现检测 + 真值评估")
    print(f"{'=' * 74}")

    edges = accumulate_edges(args.year)
    edges = score_edges(edges)

    ep = OUT_DIR / f"cooccurrence_edges_{args.year}.csv"
    edges.to_csv(ep, index=False, encoding="utf-8-sig")
    print(f"\n  → 全年共现边表：{ep}")

    months = {f"{args.year}_{m:02d}" for m in range(1, 13)}
    res = evaluate(edges, months)

    if res.get("found") is not None and not res["found"].empty:
        fp = OUT_DIR / f"cooccurrence_truth_hits_{args.year}.csv"
        res["found"].to_csv(fp, index=False, encoding="utf-8-sig")
        print(f"\n  → 真值命中明细：{fp}")

    print(f"\n{'=' * 74}")
    print("✅ 实验完成")
    print(f"{'=' * 74}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
