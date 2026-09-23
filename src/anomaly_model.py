"""
阶段四 · 机器学习异常检测【支撑模块】—— 在真实标注上做定量评估

这是整个项目里**唯一能算出真实性能指标**的地方
------------------------------------------------------------------
别处（贸易单据、循环贸易团伙）都没有真实标注，只能用合成数据验证。
但 HawaiiCoast_GT 提供了 **154 起真实世界事件的 AIS 真值标注**，
所以这里可以做出**真正的、可复现的性能数字**。

关键纪律：**无监督训练，只在评估时用标签**
------------------------------------------------------------------
Isolation Forest 是**无监督**方法 —— 训练时**完全不看标签**，
标签只用来算 AUC / 召回。这一点非常重要：
如果拿标签去调参或选特征，指标就失去意义了。
（踩过的坑见 experiments/class_baseline.py 里的"选择偏差自省"。）

评估口径
------------------------------------------------------------------
· 不用"准确率" —— 事件是极少数类，全判正常也能有高准确率
· 用 **ROC-AUC**（排序能力）+ **precision@K**（银行只能复核 top-K）
· 并给出 **lift**（相对基准率的提升倍数）

答辩价值
------------------------------------------------------------------
"我们的异常检测在真实标注数据上 AUC = X，top-50 报警的召回 = Y%"
—— 这是一个**能被追问、也能被复现**的数字。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
INCIDENT_CSV = ROOT / "data/raw/HawaiiCoast_GT/incident_data/hawaii_primary_trajectories_of_interest_2017_2020.csv"
OUT = ROOT / "experiments/results"
OUT.mkdir(parents=True, exist_ok=True)

# 用于建模的特征（全部来自阶段二的特征工程）
FEATURES = [
    "duration_hours", "n_points",
    "avg_sog", "std_sog", "max_sog",
    "stopped_ratio", "course_std",
    "n_signal_conflict", "max_gap_hours", "n_gaps",
]

# 「尺寸类」特征：航次时长与点数。
# ⚠️ 这两列有**标记偏差风险**：拆航次是按时间缺口切的，而事件标签用
#    "时间窗重叠"判定 —— **航次越长，越容易撞上任何时间窗**。
#    实测：事件航次的 duration 中位数是正常航次的 30 倍、n_points 是 22 倍。
#    这既可能是真实的（失去动力/被拖带确实会让航次变长），
#    也可能纯粹是标记偏差。**必须做消融测试才能区分。**
SIZE_FEATURES = ["duration_hours", "n_points"]

# 消融用的特征集
FEATURE_SETS = {
    "全部特征": FEATURES,
    "去掉尺寸特征": [f for f in FEATURES if f not in SIZE_FEATURES],
    "仅行为特征": ["avg_sog", "std_sog", "max_sog", "stopped_ratio", "course_std"],
    "仅信号特征": ["n_signal_conflict", "max_gap_hours", "n_gaps"],
}

CONFIG = {
    "contamination": 0.05,   # 预期异常比例（粗略先验，不来自标签）
    "n_estimators": 300,
    "random_state": 42,
}


# ==========================================================================
# 1. 打标签：找出与真实事件时间窗重叠的航次
# ==========================================================================

def label_voyages(feats: pd.DataFrame, pad_hours: float = 6.0) -> pd.DataFrame:
    """给航次打标签：该航次是否与某起真实事件的时间窗重叠。

    ⚠️ 标签**只用于评估**，绝不进入训练。

    做法：
      事件表给的是 (MMSI, 事件起止时间)。对每条船，凡是时间窗与之重叠的
      航次，标记为 incident。
    """
    inc = pd.read_csv(INCIDENT_CSV)
    for c in ("ais_incident_start_bound_hst", "ais_incident_end_bound_hst"):
        inc[c] = pd.to_datetime(inc[c], errors="coerce", utc=True)
    inc = inc.dropna(subset=["ais_incident_start_bound_hst"])

    f = feats.copy()
    f["t_start"] = pd.to_datetime(f["t_start"], errors="coerce", utc=True)
    f["t_end"] = pd.to_datetime(f["t_end"], errors="coerce", utc=True)
    f["is_incident"] = False

    pad = pd.Timedelta(hours=pad_hours)
    n_events = 0
    for mmsi, g in inc.groupby("MMSI"):
        sel = f["MMSI"] == mmsi
        if not sel.any():
            continue
        for _, ev in g.iterrows():
            t0 = ev["ais_incident_start_bound_hst"] - pad
            t1 = (ev["ais_incident_end_bound_hst"]
                  if pd.notna(ev["ais_incident_end_bound_hst"])
                  else ev["ais_incident_start_bound_hst"]) + pad
            # 时间窗重叠判定
            hit = sel & (f["t_start"] <= t1) & (f["t_end"] >= t0)
            if hit.any():
                f.loc[hit, "is_incident"] = True
                n_events += 1

    n_pos = int(f["is_incident"].sum())
    print(f"  打标签：{n_events} 个事件时间窗命中航次，"
          f"覆盖 {n_pos:,} / {len(f):,} 个航次（{n_pos / max(len(f), 1) * 100:.2f}%）")
    return f


# ==========================================================================
# 2. 无监督训练（★ 不看标签）
# ==========================================================================

def prepare_matrix(f: pd.DataFrame, features: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """构造特征矩阵：缺失值中位数填充 + 标准化。

    ⚠️ 填充和标准化的统计量必须只从**训练数据**算，不能碰测试数据。
    这里为了简洁用全量算，但在正式实验里应该按时间切分后分别算
    —— 这一点在文档里要如实说明。
    """
    cols = [c for c in (features or FEATURES) if c in f.columns]
    X = f[cols].copy()
    # 中位数填充：max_gap_hours 在无缺口的航次上是 NaN，填 0 更有业务含义
    for c in cols:
        if c == "max_gap_hours":
            X[c] = X[c].fillna(0.0)
        else:
            X[c] = X[c].fillna(X[c].median())
    X = X.replace([np.inf, -np.inf], 0.0)
    return StandardScaler().fit_transform(X.values), cols


def train_iforest(X: np.ndarray) -> IsolationForest:
    """训练 Isolation Forest —— 注意：**不使用任何标签**。"""
    model = IsolationForest(
        n_estimators=CONFIG["n_estimators"],
        contamination=CONFIG["contamination"],
        random_state=CONFIG["random_state"],
        n_jobs=-1,
    )
    model.fit(X)          # ← 只有 X，没有 y
    return model


# ==========================================================================
# 3. 评估
# ==========================================================================

def evaluate(f: pd.DataFrame, scores: np.ndarray) -> dict:
    """算 AUC / precision@K / lift。"""
    from sklearn.metrics import roc_auc_score

    y = f["is_incident"].to_numpy().astype(int)
    n_pos, n = int(y.sum()), len(y)
    base = n_pos / n

    print(f"\n{'=' * 78}")
    print(f"评估（真实标注）")
    print(f"{'=' * 78}")
    print(f"  样本 {n:,} 个航次，其中事件航次 {n_pos:,} 个（基准率 {base * 100:.2f}%）")

    if n_pos == 0 or n_pos == n:
        print("  ⚠️ 正负样本不足，无法评估")
        return {}

    auc = roc_auc_score(y, scores)
    print(f"\n  ★ ROC-AUC = {auc:.3f}")
    print(f"     （0.5 = 随机；这是**无监督**模型的排序能力，训练时没见过标签）")

    # precision@K
    order = np.argsort(-scores)
    print(f"\n  {'K':>7} │ {'捞回':>5} {'召回率':>7} {'精确率':>7} {'随机期望':>9} {'lift':>8}")
    rows = []
    for k in [10, 20, 50, 100, 200, 500]:
        k = min(k, n)
        hits = int(y[order[:k]].sum())
        prec = hits / k
        exp = base * k
        lift = prec / base if base else 0
        print(f"  {k:>7} │ {hits:>5} {hits / n_pos * 100:>6.1f}% "
              f"{prec * 100:>6.2f}% {exp:>9.2f} {lift:>7.1f}x")
        rows.append({"K": k, "hits": hits, "recall": hits / n_pos,
                     "precision": prec, "lift": lift})

    return {"auc": auc, "base_rate": base, "n": n, "n_pos": n_pos, "curve": rows}


def feature_contribution(f: pd.DataFrame, scores: np.ndarray) -> None:
    """看看事件航次在哪些特征上明显偏离正常航次。

    这不是"特征重要性"（Isolation Forest 没有直接的），
    而是**对比两组的中位数**，用来解释"模型抓到了什么"。
    """
    cols = [c for c in FEATURES if c in f.columns]
    y = f["is_incident"].to_numpy()
    print(f"\n{'=' * 78}")
    print("事件航次 vs 正常航次：特征中位数对比（用来说明模型抓到了什么）")
    print(f"{'=' * 78}")
    print(f"  {'特征':<22} {'事件航次':>12} {'正常航次':>12} {'差异':>10}")
    for c in cols:
        a = f.loc[y, c].median()
        b = f.loc[~y, c].median()
        if pd.isna(a) or pd.isna(b):
            continue
        ratio = (a / b) if b else np.nan
        flag = "  ←" if (not np.isnan(ratio) and (ratio > 1.5 or ratio < 0.67)) else ""
        print(f"  {c:<22} {a:>12.2f} {b:>12.2f} {ratio:>9.2f}x{flag}")


# ==========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阶段四：机器学习异常检测（真实标注评估）")
    ap.add_argument("--pattern", default="vessel_features_2017_*.csv",
                    help="特征文件通配符")
    args = ap.parse_args(argv)

    files = sorted(PROC.glob(args.pattern))
    if not files:
        print(f"❌ 找不到特征文件：{PROC / args.pattern}")
        print("   先跑：  src/feature_engineering.py --months 2017_01 2017_02 ...")
        return 1

    print(f"\n{'=' * 78}")
    print("阶段四：机器学习异常检测 —— 在真实标注上评估")
    print(f"{'=' * 78}")
    print(f"  载入 {len(files)} 个月的特征文件")

    feats = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"  合计 {len(feats):,} 个航次")

    print(f"\n【1/5】打标签（仅用于评估，不进入训练）")
    feats = label_voyages(feats)

    # ---------------------------------------------------------------
    # ★ 消融实验：先回答"模型到底学到了什么"
    #   事件航次的 duration/n_points 是正常航次的 30x/22x，必须区分
    #   这是真实信号还是标记偏差。
    # ---------------------------------------------------------------
    print(f"\n【2/5】★ 消融实验：换不同特征集，看 AUC 稳不稳")
    print(f"{'=' * 78}")
    print(f"  {'特征集':<18} {'维度':>5} {'ROC-AUC':>9} {'top-100召回':>11} {'top-100 lift':>13}")
    from sklearn.metrics import roc_auc_score
    y_all = feats["is_incident"].to_numpy().astype(int)
    base = y_all.mean()
    ablation = []
    for name, fset in FEATURE_SETS.items():
        if not fset:
            continue
        Xi, ci = prepare_matrix(feats, fset)
        mi = train_iforest(Xi)
        si = -mi.decision_function(Xi)
        auc_i = roc_auc_score(y_all, si) if 0 < y_all.sum() < len(y_all) else float("nan")
        order = np.argsort(-si)
        k = min(100, len(si))
        hits = int(y_all[order[:k]].sum())
        prec = hits / k
        lift = prec / base if base else 0
        print(f"  {name:<18} {len(ci):>5} {auc_i:>9.3f} {hits / y_all.sum() * 100:>10.1f}% {lift:>12.1f}x")
        ablation.append({"feature_set": name, "n_features": len(ci),
                         "auc": auc_i, "top100_hits": hits, "lift": lift})
    pd.DataFrame(ablation).to_csv(OUT / "anomaly_ablation.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"{'=' * 78}")

    print(f"\n【3/5】构造完整特征矩阵")
    X, cols = prepare_matrix(feats)
    print(f"  特征 {len(cols)} 个：{', '.join(cols)}")

    print(f"\n【4/5】无监督训练 Isolation Forest（★ 不看标签）")
    model = train_iforest(X)
    # decision_function 越大越正常 → 取负号变成"越大越异常"
    scores = -model.decision_function(X)
    print(f"  完成：{CONFIG['n_estimators']} 棵树，contamination={CONFIG['contamination']}")

    print(f"\n【5/5】评估")
    res = evaluate(feats, scores)
    if res:
        feature_contribution(feats, scores)

    if res:
        pd.DataFrame(res["curve"]).to_csv(
            OUT / "anomaly_model_curve.csv", index=False, encoding="utf-8-sig")
        out = feats.copy()
        out["anomaly_score"] = scores
        out.sort_values("anomaly_score", ascending=False).to_csv(
            OUT / "anomaly_scores.csv", index=False, encoding="utf-8-sig")
        print(f"\n  → {OUT / 'anomaly_model_curve.csv'}")
        print(f"  → {OUT / 'anomaly_scores.csv'}")

    print(f"\n{'=' * 78}")
    if res:
        print(f"✅ 完成。ROC-AUC = {res['auc']:.3f}（无监督，训练时未使用标签）")
    print(f"{'=' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
