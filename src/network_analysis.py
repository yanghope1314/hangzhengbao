"""
阶段三：船舶物理共现网络分析 ★主锚点★

这是整个方案的核心，不是配套模块。其余（基础特征、规则引擎、机器学习、
联邦学习）都是为它服务的支撑能力。

----------------------------------------------------------------------
为什么主锚点是"共现网络"
----------------------------------------------------------------------
现有商业化产品（亿海蓝提单盾、中远海科银航宝）和已知的两条中国专利
（CN121882718A、CN121639228A）**全部停留在"单证 / 单船 / 单企业"级别**：
给一个提单号，查这条船的轨迹对不对。

**没有人做跨主体、跨运单的网络结构分析。**

而循环贸易（A→B→C→A）的货物如果真在空转，**必然在物理世界留下痕迹**：
同一组船舶反复承运同一批企业、反复往同一组港口跑、在同一泊位长时间并靠。
这是纯单据/资金流手段查不出来、但物理轨迹能查出来的东西。

----------------------------------------------------------------------
★ 诚实分层：哪一层是真的，哪一层是合成的
----------------------------------------------------------------------
本模块**必须**分两层，且在代码、文档、答辩里都讲清楚：

  【第一层·真实数据】船舶↔船舶 共现网络
      数据来源：HawaiiCoast_GT 的真实 AIS 轨迹
      验证方式：用 incident_data 里 154 起真实事件中**涉及两船互动**的
                事件（拖带/救援/碰撞）做真值验证 —— README 原文用的是
                "spatiotemporally concurrent data points"（时空共现数据点），
                即这些事件本身就是被标注过的共现事件。
      ✅ 这一层的每一个数字都是真的、可复现的。

  【第二层·合成注入】企业↔船舶↔港口 异构图 + 循环贸易团伙
      数据来源：**真实船舶、真实港口**为底，**注入合成的**企业节点与
                申报关系，并注入已知的循环贸易团伙结构。
      ⚠️ 这一层是**合成**的 —— 因为公开数据里根本不存在"企业提交给银行
         的贸易单据"。**必须在文档和答辩里明确标注为构造的验证用例**，
         绝不可暗示是真实数据。这和我们用合成异常轨迹做基准是同一套
         学术惯例（真实标注不可得时，用受控合成数据验证检测能力）。

用法
------------------------------------------------------------------
    .venv/Scripts/python.exe src/network_analysis.py --month 2017_01
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import networkx as nx
except ImportError:
    print("需要 networkx：uv pip install networkx")
    raise

ROOT = Path(__file__).resolve().parent.parent
AIS_DIR = ROOT / "data/raw/HawaiiCoast_GT/AIS_data"
INCIDENT_CSV = ROOT / "data/raw/HawaiiCoast_GT/incident_data/hawaii_primary_trajectories_of_interest_2017_2020.csv"
OUT_DIR = ROOT / "data/processed"
RESULT_DIR = ROOT / "experiments/results"

CONFIG = {
    # --- 共现的时空判定 ---
    "time_bin": "1h",        # 时间分箱粒度
    "space_deg": 0.02,       # 空间网格边长（度），0.02° 在夏威夷纬度约 2.2 km
    "min_windows": 3,        # 至少共现这么多个时间窗，才算一条稳定的边
    # 「结伴航行」的门槛：双方都在航行且同框的窗口数。
    # 定得比 min_windows 高，因为这才是高区分度信号，宁缺毋滥。
    "min_moving_windows": 5,
    # --- 领域过滤：排除"母船附属艇" ---
    # AIS 规范保留 98 开头的 MMSI 给"与母船相关联的艇"（tenders/救生艇）。
    # 它们定义上就跟母船绑在一起，共现是必然的，不构成任何信号。
    # 实测教训：不过滤的话，lift 排行榜前几名全是
    # "T/T STAR PRINCESS 19 ↔ T/T STAR PRINCESS 21" 这种，lift 高达 4930。
    "exclude_mmsi_prefix": ("98",),
    # --- "并靠"判定（循环贸易最直接的物理信号）---
    "stopped_sog": 0.5,      # 低于此航速算"停着"
    "moving_sog": 2.0,       # 高于此航速算"在航行"
    "mooring_ratio": 0.6,    # 一条边中"双方都停着"的窗口占比超过此值 → 判定为并靠关系
    # --- 社区发现 ---
    "min_community": 4,      # 小于这个规模的社区不视为可疑团伙
}


# ==========================================================================
# 第一层：真实数据 —— 船舶共现网络
# ==========================================================================

def load_month(month: str, cols: list[str] | None = None) -> pd.DataFrame:
    f = AIS_DIR / f"Hawaii_{month}.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    cols = cols or ["MMSI", "datetime_utc", "lat", "lon",
                    "speed_over_ground_knots", "vessel_name", "vessel_class"]
    df = pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime_utc", "lat", "lon", "MMSI"])
    return df


def build_cooccurrence_edges(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """从真实 AIS 构建船舶共现边。

    做法（避免 O(n²) 的两两比较）：
      1. 每个点落到一个 (时间窗, 空间格) 里
      2. 同一个格子里的所有船，两两算一次共现
      3. 按 (船A, 船B) 汇总：共现了多少个窗口、其中多少是"双方都停着"

    Args:
        df: 含 MMSI / datetime_utc / lat / lon / speed_over_ground_knots

    Returns:
        边表：mmsi_a, mmsi_b, n_windows, n_both_stopped, mooring_ratio,
              name_a, name_b, class_a, class_b
    """
    cfg = {**CONFIG, **(config or {})}

    d = df.copy()

    # --- 领域过滤：剔除母船附属艇（MMSI 98 开头）---
    if cfg.get("exclude_mmsi_prefix"):
        n0 = d["MMSI"].nunique()
        pref = tuple(cfg["exclude_mmsi_prefix"])
        d = d[~d["MMSI"].astype(str).str.startswith(pref)]
        print(f"  领域过滤：剔除母船附属艇（MMSI {pref} 开头）"
              f"{n0 - d['MMSI'].nunique()} 艘 —— 它们与母船共现是定义使然，无信号价值")
    d["_tbin"] = d["datetime_utc"].dt.floor(cfg["time_bin"])
    d["_clat"] = np.floor(d["lat"] / cfg["space_deg"]).astype(int)
    d["_clon"] = np.floor(d["lon"] / cfg["space_deg"]).astype(int)
    d["_stopped"] = d["speed_over_ground_knots"] < cfg["stopped_sog"]

    # 每个 (时间窗, 空间格) 里有哪些船；用 groupby 拿到去重后的 MMSI 集合
    cells = d.groupby(["_tbin", "_clat", "_clon"], sort=False)

    # 记录每条船在每个格子里的状态（是否停着），用于后面判"并靠"
    meta = (d.groupby(["MMSI"])
              .agg(name=("vessel_name", "first"),
                   cls=("vessel_class", "first")))
    name_of = meta["name"].to_dict()
    cls_of = meta["cls"].to_dict()

    d["_moving"] = d["speed_over_ground_knots"] > cfg["moving_sog"]

    pair_windows: dict[tuple[int, int], int] = {}
    pair_stopped: dict[tuple[int, int], int] = {}
    pair_moving: dict[tuple[int, int], int] = {}

    n_cells = 0
    for _, grp in cells:
        mmsis = grp["MMSI"].unique()
        if len(mmsis) < 2:
            continue
        n_cells += 1
        stopped_map = grp.groupby("MMSI")["_stopped"].max().to_dict()
        moving_map = grp.groupby("MMSI")["_moving"].max().to_dict()
        for a, b in combinations(sorted(mmsis), 2):
            key = (a, b)
            pair_windows[key] = pair_windows.get(key, 0) + 1
            if stopped_map.get(a, False) and stopped_map.get(b, False):
                pair_stopped[key] = pair_stopped.get(key, 0) + 1
            # ★ 双方都在航行中还待在一起 —— 正常航运里几乎不该发生。
            #   港口里一堆船"同框"是常态（所以并靠边占了九成），
            #   但两艘船在**航速都 >2 节**的情况下反复同框，是强信号
            #   （结伴航行 / 海上过驳 / 拖带）。
            if moving_map.get(a, False) and moving_map.get(b, False):
                pair_moving[key] = pair_moving.get(key, 0) + 1

    print(f"  共现扫描：{n_cells:,} 个非空 (时间窗×空间格) 单元，"
          f"得到 {len(pair_windows):,} 对候选共现")

    if not pair_windows:
        return pd.DataFrame(columns=["mmsi_a", "mmsi_b", "n_windows",
                                     "n_both_stopped", "mooring_ratio"])

    edges = pd.DataFrame(
        [(a, b, w, pair_stopped.get((a, b), 0), pair_moving.get((a, b), 0))
         for (a, b), w in pair_windows.items()],
        columns=["mmsi_a", "mmsi_b", "n_windows", "n_both_stopped", "n_both_moving"],
    )
    edges["mooring_ratio"] = edges["n_both_stopped"] / edges["n_windows"]
    edges["moving_ratio"] = edges["n_both_moving"] / edges["n_windows"]

    # ------------------------------------------------------------------
    # ★ lift 归一化：把"因为本来就在同一片海域"的部分除掉
    #
    # 为什么必须做（这是实测踩出来的坑）：
    #   原始权重最高的边，几乎全是**正常运营**——
    #     · 观光客船 ↔ 观光客船（檀香山同一航线天天跑）
    #     · 港作拖轮 ↔ 港作拖轮
    #     · 拖轮 ↔ 客船（港内协助作业）
    #   这些船本来就密集地在同一片水域活动，同框是必然的。
    #   直接拿原始权重当风险信号 → 满屏误报，一个能用的都没有。
    #
    # 做法：lift = 观测共现 / 独立假设下的期望共现
    #        P(a,b) / (P(a) · P(b))
    #   lift ≈ 1  两船相互独立（同框纯属都在那片海域）
    #   lift >> 1 远超随机 —— 才是"这两条船老是凑在一起"的信号
    # ------------------------------------------------------------------
    cell_count = cells.ngroups                      # 非空 (时间窗×空间格) 总数
    vessel_cells = d.groupby("MMSI")["_tbin"].size()  # 近似：各船出现过的点数
    # 更准确：各船出现在多少个不同的 (时间窗,空间格) 里
    vessel_cells = (d.groupby(["MMSI"])[["_tbin", "_clat", "_clon"]]
                      .apply(lambda g: len(g.drop_duplicates()))
                      .rename("n_cells"))

    ca = edges["mmsi_a"].map(vessel_cells).astype(float)
    cb = edges["mmsi_b"].map(vessel_cells).astype(float)
    # lift = (n_windows/N) / ((ca/N)*(cb/N)) = n_windows * N / (ca * cb)
    edges["lift"] = edges["n_windows"] * cell_count / (ca * cb)
    edges["lift"] = edges["lift"].replace([np.inf, -np.inf], np.nan)

    # 过滤：至少共现 min_windows 个窗口才算稳定关系
    before = len(edges)
    edges = edges[edges["n_windows"] >= cfg["min_windows"]].reset_index(drop=True)
    print(f"  稳定共现边（≥{cfg['min_windows']}个时间窗）：{len(edges):,}"
          f"（过滤掉 {before - len(edges):,} 条偶发共现）")

    edges["name_a"] = edges["mmsi_a"].map(name_of)
    edges["name_b"] = edges["mmsi_b"].map(name_of)
    edges["class_a"] = edges["mmsi_a"].map(cls_of)
    edges["class_b"] = edges["mmsi_b"].map(cls_of)
    return edges


def validate_against_incidents(edges: pd.DataFrame, month: str | None = None) -> pd.DataFrame:
    """★ 用真实标注事件验证共现检测 —— 这是主锚点的真值检验。

    逻辑：incident_data 里"两船互动"的事件（拖带/救援/碰撞）本身就是
    **被人工标注过的时空共现事件**（README 原文用的是
    "spatiotemporally concurrent data points"）。

    如果我们的共现检测是有效的，那么这些事件的船对，**应该出现在
    高权重的共现边里**。这是一个可量化的召回率检验。

    ⚠️ 关键：**必须把真值限制在同一个时间范围内比较。**
    事件数据跨 2017–2020 共 4 年，而我们通常只处理其中一个月。
    拿一个月的结果去比四年的真值，召回率会被严重低估
    （踩过这个坑：只用 2017_01 时召回率仅 19%，因为 89% 的真值事件
    根本不在这一个月里）。

    Args:
        edges: 共现边
        month: 形如 "2017_01"；给定时只验证该月内的事件

    Returns:
        每个两船事件的验证结果
    """
    inc = pd.read_csv(INCIDENT_CSV)

    # --- 时间范围对齐 ---
    if month:
        y, m = month.split("_")
        t0 = pd.Timestamp(f"{y}-{m}-01", tz="UTC")
        t1 = t0 + pd.offsets.MonthBegin(1)
        inc["_t"] = pd.to_datetime(inc["ais_incident_start_bound_hst"],
                                   errors="coerce", utc=True)
        before = inc["incident_num"].nunique()
        inc = inc[(inc["_t"] >= t0) & (inc["_t"] < t1)]
        print(f"    （真值时间对齐：{month} 内有 {inc['incident_num'].nunique()} 起"
              f"事件，全量共 {before} 起）")
        if inc.empty:
            print("    ⚠️ 该月没有两船互动的真值事件，无法验证")
            return pd.DataFrame()

    size = inc.groupby("incident_num")["MMSI"].nunique()
    multi = size[size >= 2].index

    rows = []
    for inum in multi:
        g = inc[inc["incident_num"] == inum]
        mmsis = sorted(g["MMSI"].unique())
        # 一个事件可能涉及 2 条以上船，取所有两两组合
        for a, b in combinations(mmsis, 2):
            hit = edges[((edges["mmsi_a"] == a) & (edges["mmsi_b"] == b))
                        | ((edges["mmsi_a"] == b) & (edges["mmsi_b"] == a))]
            rows.append({
                "incident_num": inum,
                "mmsi_a": a, "mmsi_b": b,
                "incident_type": g["ais_incident_type"].iloc[0],
                "evidence": g["ais_incident_evidence"].iloc[0],
                "detected": len(hit) > 0,
                "n_windows": int(hit["n_windows"].iloc[0]) if len(hit) else 0,
            })

    v = pd.DataFrame(rows)
    if v.empty:
        print("  ⚠️ 没有找到两船互动的事件")
        return v

    print(f"\n  ★ 真值验证：{len(v)} 个两船事件（涉及 "
          f"{v['incident_num'].nunique()} 起事故）")
    print(f"    共现检测命中：{int(v['detected'].sum())} / {len(v)}"
          f"  = {v['detected'].mean() * 100:.1f}%")
    if v["detected"].any():
        print(f"    命中事件的共现窗口数：中位 {v.loc[v['detected'], 'n_windows'].median():.0f}，"
              f"最大 {v.loc[v['detected'], 'n_windows'].max()}")
    return v


# ==========================================================================
# 第二层：合成注入 —— 企业-船舶-港口异构图
# ==========================================================================

def inject_synthetic_trade_layer(
    edges: pd.DataFrame,
    n_fraud_rings: int = 5,
    ring_size: int = 4,
    n_normal: int = 300,
    seed: int = 42,
) -> tuple[nx.Graph, pd.DataFrame]:
    """⚠️ 【合成数据层】注入企业节点与循环贸易团伙，构造异构图。

    **这一层是构造的，不是真实数据。** 理由：公开 AIS 数据里不存在
    "企业提交给银行的贸易单据"。真实标注的贸易欺诈数据在全球范围内都
    不可得（这也正是我们第一章说的研究空白）。

    做法（受控合成，学术上的标准做法）：
      · 正常企业：随机关联少量船舶与港口，结构稀疏
      · 循环贸易团伙：构造 A→B→C→…→A 的闭环，团伙内企业共享同一组船舶、
        反复往同一组港口跑 —— 这正是"物理共现"要抓的结构

    Args:
        edges:       真实共现边（提供真实的船舶节点与"哪些船会一起出现"）
        n_fraud_rings: 注入的循环贸易团伙数
        ring_size:     每个团伙的企业数
        n_normal:      正常企业数
        seed:          固定随机种子，保证可复现

    Returns:
        (异构图, 企业真值标签表)
    """
    rng = np.random.default_rng(seed)

    # 用真实共现边构建"船舶邻接"，团伙内部的船从这个真实结构里挑，
    # 保证注入的团伙在物理上是"可能一起出现"的
    real_vessels = sorted(set(edges["mmsi_a"]) | set(edges["mmsi_b"]))
    if len(real_vessels) < ring_size * 2:
        raise ValueError("真实船舶数太少，无法构造团伙")

    G = nx.Graph()
    G.add_nodes_from(real_vessels, node_type="vessel")

    # 港口节点：用真实经纬度格心当作"港口"占位（真实数据里没有港口表）
    ports = [f"PORT_{i:03d}" for i in range(30)]
    G.add_nodes_from(ports, node_type="port")

    labels = []
    ent_id = 0

    # --- 正常企业：随机连 1-2 条船、1-2 个港口，结构稀疏 ---
    for _ in range(n_normal):
        e = f"ENT_{ent_id:04d}"
        ent_id += 1
        G.add_node(e, node_type="enterprise", is_fraud=False)
        for v in rng.choice(real_vessels, size=rng.integers(1, 3), replace=False):
            G.add_edge(e, v, edge_type="declares", synthetic=True)
        for p in rng.choice(ports, size=rng.integers(1, 3), replace=False):
            G.add_edge(e, p, edge_type="ships_to", synthetic=True)
        labels.append({"enterprise": e, "is_fraud": False, "ring_id": -1})

    # --- 循环贸易团伙：闭环 + 共享船舶/港口 ---
    for r in range(n_fraud_rings):
        ring_ents = [f"ENT_{ent_id + i:04d}" for i in range(ring_size)]
        ent_id += ring_size
        # 团伙共用一小撮船和港口 —— 这是"物理共现"的根源
        shared_v = rng.choice(real_vessels, size=max(2, ring_size // 2), replace=False)
        shared_p = rng.choice(ports, size=2, replace=False)

        for e in ring_ents:
            G.add_node(e, node_type="enterprise", is_fraud=True, ring_id=r)
            for v in shared_v:
                G.add_edge(e, v, edge_type="declares", synthetic=True)
            for p in shared_p:
                G.add_edge(e, p, edge_type="ships_to", synthetic=True)
            labels.append({"enterprise": e, "is_fraud": True, "ring_id": r})

        # 闭环边 A→B→C→…→A（有向贸易关系，这里用无向图存结构，另记边类型）
        for i in range(ring_size):
            a, b = ring_ents[i], ring_ents[(i + 1) % ring_size]
            G.add_edge(a, b, edge_type="trades_with", synthetic=True)

    lab = pd.DataFrame(labels)
    n_f = int(lab["is_fraud"].sum())
    print(f"\n  ⚠️【合成层】注入 {n_normal} 家正常企业 + {n_f} 家团伙企业"
          f"（{n_fraud_rings} 个团伙 × {ring_size} 家）")
    print(f"     图规模：{G.number_of_nodes():,} 节点 / {G.number_of_edges():,} 边")
    print(f"     —— 这一层是构造的验证用例，文档中必须明确标注为合成数据")
    return G, lab


# ==========================================================================
# 可疑结构检测
# ==========================================================================

def detect_suspicious_communities(G: nx.Graph, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """在图上找"异常密集的子结构"。

    方法（刻意用可解释的传统图指标，不用 GNN）：
      1. Louvain 社区发现（networkx 内置）
      2. 对企业社区计算**边密度**：正常贸易网络应当稀疏，
         循环贸易团伙会在图上表现为异常紧密的小圈子
      3. 输出每个社区的规模、密度、成员

    ⚠️ **为什么不用图神经网络（GNN）——答辩必答**
       不是"算力不够"，而是：**这个场景不存在带标注的欺诈样本**。
       GNN 是监督/半监督方法，没有标注就没有训练信号。
       （学术界明确结论：海事异常检测领域不存在公开的标准标注数据集。）
       所以我们选择**无监督 + 可解释**的传统图指标：
       既能用，又能对银行解释"为什么判定这家企业可疑"。
    """
    # 只在企业子图上做社区发现（船和港口是"证据"，不是"主体"）
    ents = [n for n, d in G.nodes(data=True) if d.get("node_type") == "enterprise"]
    sub = G.subgraph(ents)

    comms = nx.community.louvain_communities(sub, seed=42)

    rows = []
    for i, c in enumerate(comms):
        members = sorted(c)
        if len(members) < CONFIG["min_community"]:
            continue
        # 社区内部边密度
        n = len(members)
        possible = n * (n - 1) / 2
        actual = sub.subgraph(members).number_of_edges()
        density = actual / possible if possible else 0.0

        # 这个社区共享了多少条船 / 多少个港口（团伙的物理共现特征）
        shared_v = {v for e in members for v in G.neighbors(e)
                    if G.nodes[v].get("node_type") == "vessel"}
        shared_p = {p for e in members for p in G.neighbors(e)
                    if G.nodes[p].get("node_type") == "port"}

        rows.append({
            "community_id": i,
            "size": n,
            "internal_density": round(density, 3),
            "n_shared_vessels": len(shared_v),
            "n_shared_ports": len(shared_p),
            "members": ",".join(map(str, members)),
            # MMSI 是 numpy int64，join 前必须转 str，否则 TypeError
            "vessels": ",".join(map(str, sorted(shared_v))),
        })

    res = pd.DataFrame(rows).sort_values("internal_density", ascending=False) \
        if rows else pd.DataFrame()

    if labels is not None and not res.empty:
        fraud = set(labels.loc[labels["is_fraud"], "enterprise"])
        res["n_fraud_true"] = res["members"].apply(
            lambda s: len(set(s.split(",")) & fraud))
        res["precision"] = res["n_fraud_true"] / res["size"]

    print(f"\n  社区发现：{len(comms)} 个社区，其中 {len(res)} 个规模 ≥{CONFIG['min_community']}")
    if not res.empty:
        print(f"    边密度最高的 5 个社区：")
        for _, r in res.head(5).iterrows():
            extra = (f"  真值命中 {r['n_fraud_true']}/{r['size']}"
                     f"（精确率 {r['precision']:.0%}）") if "precision" in r else ""
            print(f"      #{r['community_id']:<3} 规模 {r['size']:<3} "
                  f"密度 {r['internal_density']:.3f}  "
                  f"共享船 {r['n_shared_vessels']:<3} 共享港 {r['n_shared_ports']}{extra}")
    return res


def detect_cycles(G: nx.Graph, max_len: int = 6) -> pd.DataFrame:
    """检测企业之间的**闭环贸易路径**（A→B→…→A）。

    这是循环贸易最直接的结构特征 —— 真实的贸易链是树状的（上游→下游），
    只有循环贸易才需要构成闭环。
    """
    ents = [n for n, d in G.nodes(data=True) if d.get("node_type") == "enterprise"]
    sub = G.subgraph(ents)

    found = []
    seen = set()
    for cyc in nx.simple_cycles(sub, length_bound=max_len) if sub.is_directed() else []:
        key = frozenset(cyc)
        if key not in seen:
            seen.add(key)
            found.append({"length": len(cyc), "members": "→".join(cyc + [cyc[0]])})

    # networkx.Graph 是无向的，simple_cycles 不适用；改用环基（cycle basis）
    if not found:
        try:
            basis = nx.minimum_cycle_basis(sub)
            for cyc in basis:
                if 3 <= len(cyc) <= max_len:
                    found.append({"length": len(cyc),
                                  "members": "→".join(cyc + [cyc[0]])})
        except Exception as e:
            print(f"    （环检测跳过：{e}）")

    res = pd.DataFrame(found)
    print(f"\n  闭环检测：发现 {len(res)} 条企业间闭环路径"
          f"（长度 3–{max_len}）")
    return res


# ==========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阶段三：船舶物理共现网络分析")
    ap.add_argument("--month", default="2017_01")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 74}")
    print("阶段三：船舶物理共现网络分析 ★主锚点★")
    print(f"{'=' * 74}")

    print(f"\n【第一层·真实数据】{args.month}")
    df = load_month(args.month)
    print(f"  载入 {len(df):,} 个 AIS 点 / {df['MMSI'].nunique():,} 艘船")

    edges = build_cooccurrence_edges(df)
    if not edges.empty:
        ep = OUT_DIR / f"cooccurrence_edges_{args.month}.csv"
        edges.to_csv(ep, index=False, encoding="utf-8-sig")
        print(f"  → 共现边表：{ep}")

        print(f"\n  【并靠关系】(双方都停着的窗口占比 ≥{CONFIG['mooring_ratio']})")
        moor = edges[edges["mooring_ratio"] >= CONFIG["mooring_ratio"]]
        print(f"    {len(moor):,} 条并靠边 / {len(edges):,} 条共现边"
              f"（{len(moor) / max(len(edges), 1) * 100:.1f}%）")
        print(f"    ⚠️ 这个比例高是**正常的**——港口里一堆船同框是常态，")
        print(f"       所以「并靠」本身区分度弱，只能当辅助证据。")

        print(f"\n  ★【结伴航行】(双方**都在航行**时仍反复同框)")
        mv = edges[edges["n_both_moving"] >= CONFIG["min_moving_windows"]]
        print(f"    {len(mv):,} 条边 / {len(edges):,}"
              f"（{len(mv) / max(len(edges), 1) * 100:.2f}%）")
        print(f"    —— 正常航运里两艘船不该反复同速同行；")
        print(f"       这是**海上过驳 / 结伴航行 / 拖带**的强物理信号。")

        # ★ 用 lift 排序，而不是原始权重。
        #   实测教训：按原始权重的 Top 全是观光客船和港作拖轮（正常运营），
        #   按 lift 排序才能把"超出其运营画像的异常同框"顶上来。
        if not mv.empty:
            print(f"\n    (a) 按【原始权重】排序 —— ⚠️ 基本是正常运营，仅供参考")
            for _, r in mv.nlargest(3, "n_both_moving").iterrows():
                print(f"        {str(r['name_a'])[:18]:<18}({str(r['class_a'])[:10]:<10})"
                      f" ↔ {str(r['name_b'])[:18]:<18}({str(r['class_b'])[:10]:<10})"
                      f" {int(r['n_both_moving']):>4}   lift={r['lift']:.1f}")

            print(f"\n    (b) ★按【lift】排序 —— 超出运营画像的异常同框，这才是信号")
            for _, r in mv.nlargest(5, "lift").iterrows():
                print(f"        {str(r['name_a'])[:18]:<18}({str(r['class_a'])[:10]:<10})"
                      f" ↔ {str(r['name_b'])[:18]:<18}({str(r['class_b'])[:10]:<10})"
                      f" {int(r['n_both_moving']):>4}   lift={r['lift']:.1f}")

    # --- 边权分布诊断：用来校准 min_windows 阈值 ---
    if not edges.empty:
        print(f"\n  【边权分布】共现窗口数（决定「什么算稳定关系」的阈值该定在哪）")
        w = edges["n_windows"]
        for q in (0.5, 0.75, 0.9, 0.99):
            print(f"      {int(q * 100):>3d} 分位: {w.quantile(q):8.0f} 个窗口")
        print(f"      最大    : {w.max():.0f}")
        print(f"      —— 中位数 {w.median():.0f} 说明大量边只是「同在一个港口待过」，")
        print(f"         真正可疑的是**反复**共现的高权重边（如 >100 个窗口）")

    print(f"\n【真值验证】用真实标注事件检验共现检测")
    val = validate_against_incidents(edges, month=args.month)
    if not val.empty:
        vp = RESULT_DIR / f"cooccurrence_validation_{args.month}.csv"
        val.to_csv(vp, index=False, encoding="utf-8-sig")
        print(f"  → 验证结果：{vp}")

    print(f"\n【第二层·合成注入】企业-船舶-港口异构图")
    G, labels = inject_synthetic_trade_layer(edges, seed=args.seed)

    print(f"\n【可疑结构检测】")
    comm = detect_suspicious_communities(G, labels)
    if not comm.empty:
        cp = OUT_DIR / f"suspicious_communities_{args.month}.csv"
        comm.to_csv(cp, index=False, encoding="utf-8-sig")
        print(f"  → 社区表：{cp}")

    cyc = detect_cycles(G)
    if not cyc.empty:
        print(f"    示例：{cyc['members'].iloc[0]}")

    print(f"\n{'=' * 74}")
    print("✅ 阶段三核心链路跑通")
    print(f"{'=' * 74}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
