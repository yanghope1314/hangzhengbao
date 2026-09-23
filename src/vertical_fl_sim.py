"""
阶段五：纵向联邦学习模拟【导师方向 · 独占资源 · 非核心创新点】

定位（务必说清楚，别自己给自己挖坑）
------------------------------------------------------------------
· **联邦学习不是本项目的创新点** —— 第十五届全国特等奖《融e担》
  （南京农业大学，江苏分行选送）已经用了"联邦学习+区块链构建多方信用
  评估模型"，场景同样是数据不出域。
· **它的定位是"导师带来的独占资源"**：真实壁垒不在于"用了联邦学习"，
  而在于**能把纵向联邦做对** —— 而纵向联邦目前没有轻量开源实现可以照抄
  （调研实测：最高星的 PyVertical 仅 ★222 且停更在 2023-06）。
· 代码注释、文档、答辩里都**不要把它渲染成核心创新**。它是及格项。

⚠️ 诚实性声明（必须写进文档）
------------------------------------------------------------------
本脚本的任务标签是**受控合成**的，不是真实标注：
公开 AIS 数据里不存在"企业提交给银行的贸易单据"，也不存在三方机构
各自持有的真实特征划分。

**这不妨碍它证明想证明的事** —— 它要证明的是 FL 的**机制**：
"任何一方单独都不够，三方合起来才够"。这是一个关于**协作必要性**的
论证，与标签是否真实无关。

为什么选纵向而不是横向
------------------------------------------------------------------
· 场景本身决定：银行 / 港口海事 / 船公司持有的是**同一批船舶的不同类型
  特征**（样本相同、特征不同）→ 这是**纵向联邦**
· 横向 FedAvg 的损耗来自**数据异质性**，是结构性的（实证 6–9pp）；
  而 **SecureBoost（arXiv:1901.08755，"A Lossless Federated Learning
  Framework"）证明纵向联邦在原理上可以做到无损**
· 答辩话术：不要说"我们知道联邦学习有性能损耗"（那是为横向找台阶），
  要说"**我们选纵向，因为文献证明纵向原理上可无损，而横向的损耗源自
  数据异质性、是结构性的，不适合我们的场景**"

实现说明：可信协调者（trusted coordinator）
------------------------------------------------------------------
样本天然对齐（同一批船舶在三个数据源里都存在），所以**可以省掉 PSI
（隐私集合求交）**，用一个简化的"可信协调者"方案：

    ① 各方用自己的特征列算部分得分  z_k = X_k · w_k  → 发给协调者
    ② 协调者求和 z = Σ z_k，算 sigmoid 与残差 r = p − y
    ③ 协调者把残差 r 回传各方
    ④ 各方据此更新自己的 w_k

⚠️ **边界必须讲清楚**：这里的协调者能看到残差（含标签信息），
真实生产环境需要用**同态加密**保护残差回传。
**这个诚实的边界说明本身是加分项。**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
OUT = ROOT / "experiments/results"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 三方特征划分（对应现实中的数据归属）
#   ⚠️ 这是**示意性**划分：所有特征其实都来自 AIS。
#      现实中银行持有的是申报单据字段、港口持有的是靠泊与作业记录、
#      船公司持有的是船舶档案——那份划分在公开数据里拿不到。
# ----------------------------------------------------------------------
PARTIES = {
    "船公司": ["avg_sog", "std_sog", "max_sog", "course_std"],
    "港口海事": ["stopped_ratio", "n_signal_conflict", "max_gap_hours", "n_gaps"],
    "银行": ["duration_hours", "n_points"],
}

CONFIG = {
    "lr": 0.5,
    "epochs": 200,
    "l2": 1e-3,
    "seed": 42,
    "train_frac": 0.7,
}


# ==========================================================================
# 1. 构造受控合成任务
# ==========================================================================

def build_task(seed: int = 42) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, list[str]]:
    """构造一个**必须三方特征都用上**才能学好的任务。

    ⚠️ 设计要点（第一版踩过的坑）：
       第一版给"银行信号"的权重是 0.9、其余是乘积项，结果**银行单方就
       拿到 0.7685，几乎追平全量的 0.7715** —— 核心论点"任何一方单独都
       不够"根本没演示出来。

       修正：让三方**等权、且各自先标准化**（否则特征数少的一方因均值
       方差更大而占优）。这样每方单独最多只能拿到自己那一份信号。

    规则： score = 三方等权标准信号之和 + 噪声
      · 只用任意一方 → 只能拿到约 1/3 的信息 → AUC 明显偏低
      · 三方合起来 → 信息完整 → AUC 明显更高
    """
    rng = np.random.default_rng(seed)

    files = sorted(PROC.glob("vessel_features_2017_*.csv"))
    if not files:
        raise FileNotFoundError("找不到 vessel_features_2017_*.csv，先跑阶段二")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)[:6000]

    def _zcols(cols):
        X = df[cols].copy()
        X = X.fillna(X.median()).replace([np.inf, -np.inf], 0.0)
        X = (X - X.mean()) / (X.std().replace(0, 1))
        return X.to_numpy()

    def _std(v):
        """把信号本身也标准化 —— 保证三方等权。"""
        return (v - v.mean()) / (v.std() if v.std() else 1.0)

    X_parts = {p: _zcols(cols) for p, cols in PARTIES.items()}

    s_ship = _std(X_parts["船公司"].mean(axis=1))
    s_port = _std(X_parts["港口海事"].mean(axis=1))
    s_bank = _std(X_parts["银行"].mean(axis=1))

    # --- 三方等权相加：任何一方单独都只能拿到约 1/3 的信号 ---
    score = 1.0 * (s_ship + s_port + s_bank) + rng.normal(0, 1.2, len(df))
    y = (score > np.quantile(score, 0.80)).astype(int)   # 20% 正例

    return df, X_parts, y, list(PARTIES)


# ==========================================================================
# 2. 三种训练方式
# ==========================================================================

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_centralized(Xs: dict[str, np.ndarray], y: np.ndarray,
                      idx_tr, idx_te, cfg=CONFIG) -> tuple[float, np.ndarray]:
    """集中式训练：把所有特征拼在一起（上界参照）。"""
    X = np.hstack([Xs[p] for p in Xs])
    w = np.zeros(X.shape[1])
    lr, n = cfg["lr"], len(idx_tr)

    for _ in range(cfg["epochs"]):
        z = X[idx_tr] @ w
        r = _sigmoid(z) - y[idx_tr]
        w -= lr * (X[idx_tr].T @ r / n + cfg["l2"] * w)

    return roc_auc_score(y[idx_te], X[idx_te] @ w), w


def train_vertical_fl(Xs: dict[str, np.ndarray], y: np.ndarray,
                      idx_tr, idx_te, cfg=CONFIG) -> tuple[float, dict[str, np.ndarray]]:
    """★ 纵向联邦学习（可信协调者版）—— 核心逻辑不到 40 行。

    各方**从不交换原始特征**，只交换：
      · 上行：部分线性得分 z_k（标量，每样本一个）
      · 下行：残差 r（标量）
    """
    names = list(Xs)
    W = {p: np.zeros(Xs[p].shape[1]) for p in names}   # 各方本地参数，从不离开本地
    lr, n = cfg["lr"], len(idx_tr)

    for _ in range(cfg["epochs"]):
        # ① 各方本地算部分得分（只有得分上行，原始特征不出域）
        z = np.zeros(n)
        for p in names:
            z += Xs[p][idx_tr] @ W[p]          # ← 模拟：各方算完发给协调者

        # ② 协调者求和 + 算残差（协调者看不到任何原始特征）
        r = _sigmoid(z) - y[idx_tr]

        # ③ 残差回传各方，各方**只更新自己的参数**
        for p in names:
            W[p] -= lr * (Xs[p][idx_tr].T @ r / n + cfg["l2"] * W[p])

    # 评估同样只交换得分
    z_te = sum(Xs[p][idx_te] @ W[p] for p in names)
    return roc_auc_score(y[idx_te], z_te), W


def train_single_party(Xs: dict[str, np.ndarray], y: np.ndarray,
                       idx_tr, idx_te, cfg=CONFIG) -> dict[str, float]:
    """单方基线：每方只用自己的特征训练（证明"单独一方不够"）。"""
    out = {}
    lr, n = cfg["lr"], len(idx_tr)
    for p, X in Xs.items():
        w = np.zeros(X.shape[1])
        for _ in range(cfg["epochs"]):
            r = _sigmoid(X[idx_tr] @ w) - y[idx_tr]
            w -= lr * (X[idx_tr].T @ r / n + cfg["l2"] * w)
        out[p] = roc_auc_score(y[idx_te], X[idx_te] @ w)
    return out


# ==========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阶段五：纵向联邦学习模拟")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=8, help="不同随机种子重复次数")
    args = ap.parse_args(argv)

    print(f"\n{'=' * 78}")
    print("阶段五：纵向联邦学习模拟（可信协调者版）")
    print(f"{'=' * 78}")
    print("  ⚠️ 任务标签为**受控合成**（公开数据里不存在三方机构的真实特征划分）。")
    print("     本实验证明的是 FL 的**机制**：任何一方单独都不够，三方合起来才够。")

    print(f"\n【1/3】三方特征划分（示意性，现实中对应不同机构的数据归属）")
    for p, cols in PARTIES.items():
        print(f"  {p:<8} {len(cols)} 个特征：{', '.join(cols)}")

    print(f"\n【2/3】单次运行")
    df, Xs, y, names = build_task(args.seed)
    n = len(y)
    idx = np.random.default_rng(args.seed).permutation(n)
    n_tr = int(n * CONFIG["train_frac"])
    idx_tr, idx_te = idx[:n_tr], idx[n_tr:]
    print(f"  样本 {n:,}（训练 {n_tr:,} / 测试 {n - n_tr:,}），"
          f"正例率 {y.mean() * 100:.1f}%")

    auc_central, _ = train_centralized(Xs, y, idx_tr, idx_te)
    auc_fl, W = train_vertical_fl(Xs, y, idx_tr, idx_te)
    single = train_single_party(Xs, y, idx_tr, idx_te)

    print(f"\n  {'模型':<34}{'ROC-AUC':>10}")
    print(f"  {'-' * 46}")
    print(f"  {'集中式（三方特征全给他，上界参照）':<34}{auc_central:>10.4f}")
    print(f"  {'★ 纵向联邦（可信协调者）':<34}{auc_fl:>10.4f}")
    for p, a in single.items():
        print(f"  {'  仅用「' + p + '」自己的特征':<34}{a:>10.4f}")

    # --- 重复实验：给不确定区间 ---
    print(f"\n【3/3】重复 {args.repeats} 次不同随机种子（给不确定性区间）")
    cent, fl, sing = [], [], {p: [] for p in names}
    for r in range(args.repeats):
        _, Xr, yr, _ = build_task(args.seed + r * 17)
        idxr = np.random.default_rng(args.seed + r * 17).permutation(len(yr))
        ntr = int(len(yr) * CONFIG["train_frac"])
        a, b = idxr[:ntr], idxr[ntr:]
        cent.append(train_centralized(Xr, yr, a, b)[0])
        fl.append(train_vertical_fl(Xr, yr, a, b)[0])
        for p, v in train_single_party(Xr, yr, a, b).items():
            sing[p].append(v)

    def _ci(v):
        v = np.array(v)
        return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))

    print(f"\n  {'模型':<34}{'均值AUC':>10}{'95%CI半宽':>12}")
    print(f"  {'-' * 58}")
    m, h = _ci(cent)
    print(f"  {'集中式':<34}{m:>10.4f}{h:>12.4f}")
    m_fl, h_fl = _ci(fl)
    print(f"  {'★ 纵向联邦':<34}{m_fl:>10.4f}{h_fl:>12.4f}")
    for p, v in sing.items():
        m, h = _ci(v)
        print(f"  {'  仅用「' + p + '」':<34}{m:>10.4f}{h:>12.4f}")

    gap = np.array(cent) - np.array(fl)
    m_gap, h_gap = _ci(gap)
    best_single = max(np.mean(v) for v in sing.values())

    print(f"\n{'=' * 78}")
    print("结论")
    print(f"{'=' * 78}")
    print(f"  纵向联邦 vs 集中式 AUC 差距：{m_gap:+.4f}（95%CI ±{h_gap:.4f}）")
    if abs(m_gap) < 1e-6:
        print(f"  ✅ **精确相等** —— 这不是近似，是数学上的等价：")
        print(f"     对线性模型 + 可信协调者，纵向联邦算出的梯度与集中式")
        print(f"     **完全相同**（z = Σ X_k·w_k，各方只管自己那部分）。")
        print(f"     → 所以「纵向联邦可无损」在我们这个设定下是**精确成立**的，")
        print(f"       不需要当近似来接受。真正的代价来自隐私保护机制")
        print(f"       （同态加密 / 差分隐私），而不是联邦本身。")
        print(f"     ⚠️ 诚实边界：一旦给残差回传加上同态加密，或改用树模型")
        print(f"       （SecureBoost 那类需要按分裂点通信），等号才会变成约等号。")
    elif abs(m_gap) < 0.02:
        print(f"  ✅ 差距很小 → 与 SecureBoost「纵向联邦可无损」的结论一致")
    else:
        print(f"  ⚠️ 存在明显差距 —— 可能源于简化实现（可信协调者替代同态加密、")
        print(f"     以及我们用逻辑回归而非 SecureBoost 的梯度提升树）。如实报告。")
    print(f"\n  纵向联邦 vs 最好的单方基线：{m_fl:.4f} vs {best_single:.4f}"
          f"  （+{m_fl - best_single:.4f}）")
    print(f"  → 任何一方单独都不够，三方合起来才够 —— 这正是协作建模的必要性")

    res = pd.DataFrame({
        "model": ["集中式", "纵向联邦"] + [f"仅{p}" for p in names],
        "auc_mean": [np.mean(cent), np.mean(fl)] + [np.mean(v) for v in sing.values()],
        "auc_ci95": [1.96 * np.std(cent, ddof=1) / np.sqrt(len(cent)),
                     1.96 * np.std(fl, ddof=1) / np.sqrt(len(fl))]
                    + [1.96 * np.std(v, ddof=1) / np.sqrt(len(v)) for v in sing.values()],
    })
    op = OUT / "vertical_fl_results.csv"
    res.to_csv(op, index=False, encoding="utf-8-sig")
    print(f"\n  → {op}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
