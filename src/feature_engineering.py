"""
阶段二：AIS 基础特征工程【支撑模块，不是主锚点】

定位
------------------------------------------------------------------
本模块产出的特征是**支撑**主锚点（船舶物理共现网络）和规则引擎的底座，
**不是**方案的核心创新点。代码要写扎实，但文档和 PPT 里不要把它当卖点渲染。

三组特征
------------------------------------------------------------------
  A. 基础运动特征   —— 平均航速、航速标准差、停留时长占比
  B. ★ AIS 信号交叉印证特征（三分类）
  C. ★ 吃水/装卸特征（对应"空转贸易"）

为什么 B 要做成三分类而不是"超阈值=异常"
------------------------------------------------------------------
海事监测领域有一条铁律：

    「没有 AIS 信号 ≠ 主动关闭了 AIS」

岸基 AIS 是视距传播（典型 30–50 海里），接收受**距岸距离、地形遮蔽、
船舶密集度（时隙冲突）**影响。正常航行也会出现信号缺失。

如果我们把"信号中断"直接判成"疑似关闭 AIS 欺诈"，就会把**信号差**
误判成**欺诈** —— 对银行来说这是不可接受的误伤（会冤枉正常企业）。

所以本模块输出**三分类**：

    证据支持   AIS 轨迹与预期吻合，且信号连续性正常
    证据缺失   信号中断发生在**已知接收覆盖较弱**的区域 → 不下结论，存疑
    证据矛盾   信号中断发生在**本应有良好覆盖**的区域 → 这才是高风险信号

**覆盖度怎么算？** 我们没有岸基站位置数据，所以用一个数据驱动的代理指标：
把海域切成网格，统计每格里**出现过的不同船舶数 × 不同天数**。
- 一格里有大量船舶长期稳定上报 → 说明这里接收覆盖好
- 一格里几乎没有船上报     → 说明这里覆盖弱（或者没船去）

这个代理指标的优点：不需要外部数据、可复现、答辩时讲得清楚。
**局限也要诚实说明**：如果某片海域覆盖好但恰好没船去，会被误判成"覆盖弱"，
结果是**偏保守**（把矛盾降级成缺失），这正是我们想要的方向 —— 宁可漏报，
不可误伤。

用法
------------------------------------------------------------------
    # 单月特征
    .venv/Scripts/python.exe src/feature_engineering.py --months 2017_01

    # 多月（内存够的话）
    .venv/Scripts/python.exe src/feature_engineering.py --months 2017_01 2017_02

产出
------------------------------------------------------------------
    data/processed/vessel_features_<period>.csv    每船一行的汇总特征
    data/processed/voyages_<period>.csv            航次切分结果（供阶段三用）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
AIS_DIR = ROOT / "data/raw/HawaiiCoast_GT/AIS_data"
OUT_DIR = ROOT / "data/processed"

# 只读需要的列，省内存
USECOLS = [
    "MMSI", "datetime_utc", "lat", "lon",
    "speed_over_ground_knots", "course_over_ground_deg", "heading_deg",
    "vessel_name", "vessel_class", "draft_depth_m", "status",
]

# ---- 可调参数（全部集中在这里，方便做敏感性分析）----
CONFIG = {
    # 航次切分：同一 MMSI 相邻两点时间差超过这个值，就认为是新航次
    "voyage_gap_hours": 6.0,
    # 停留判定：对地航速低于这个值算"停着"
    "stopped_sog_knots": 0.5,
    # 覆盖度网格边长（度）。0.05° 在夏威夷纬度约 5.5 km
    "grid_deg": 0.05,
    # 一个网格被认为"覆盖良好"所需的最少（不同船舶数 × 不同天数）
    "coverage_min_vessels": 5,
    "coverage_min_days": 3,
    # 信号缺口：超过这个时长才算一次"gap"
    "gap_min_minutes": 30.0,
    # heading=511 是 AIS 规范里"不可用"的约定值
    "heading_unavailable": 511,
}


# ==========================================================================
# 0. 载入与清洗
# ==========================================================================

def load_month(period: str) -> pd.DataFrame:
    """读一个月的数据并做基础清洗。

    Args:
        period: 形如 "2017_01"

    Returns:
        清洗后的 DataFrame
    """
    f = AIS_DIR / f"Hawaii_{period}.csv"
    if not f.exists():
        raise FileNotFoundError(f"找不到 {f}")

    df = pd.read_csv(f, usecols=lambda c: c in USECOLS, low_memory=False)
    n0 = len(df)

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime_utc", "lat", "lon", "MMSI"])

    # --- 坑 2：heading=511 表示"不可用"，必须转成 NaN ---
    # 不处理的话，511 会混进"平均船首向"之类的统计里，把结果彻底带偏。
    if "heading_deg" in df.columns:
        df.loc[df["heading_deg"] == CONFIG["heading_unavailable"], "heading_deg"] = np.nan

    # --- 经纬度合法性 ---
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]

    df = df.sort_values(["MMSI", "datetime_utc"]).reset_index(drop=True)
    print(f"  载入 {period}: {len(df):,} 行（原始 {n0:,}，清洗掉 {n0 - len(df):,}）")
    return df


# ==========================================================================
# 1. 航次切分（★ README 明确说数据集没做这件事）
# ==========================================================================

def split_voyages(df: pd.DataFrame, gap_hours: float | None = None) -> pd.DataFrame:
    """按时间缺口把每艘船的点序列切成若干航次。

    ⚠️ 为什么必须自己做：HawaiiCoast_GT 的 README 明确写了
       "we do not split the trajectory by time gap or lack of movement.
        All points with the same MMSI are treated as a single trajectory each month."
       即每条船每月只有一条"轨迹"。不切分的话，跨缺口的首点速度是错的
       （用两个相隔几小时的点算出来的"速度"），会污染所有速度类特征。

    Args:
        df:        已排序的 AIS 点
        gap_hours: 缺口阈值，默认取 CONFIG

    Returns:
        加了 voyage_id 列的 DataFrame
    """
    gap_hours = gap_hours or CONFIG["voyage_gap_hours"]
    thresh = pd.Timedelta(hours=gap_hours)

    dt = df.groupby("MMSI")["datetime_utc"].diff()
    # 每条船的第一个点 diff 是 NaT → 必定是新航次
    new_voyage = dt.isna() | (dt > thresh)

    # 用 cumsum 给每个航次编号，再拼上 MMSI 保证全局唯一
    df = df.copy()
    df["voyage_seq"] = new_voyage.groupby(df["MMSI"]).cumsum()
    df["voyage_id"] = df["MMSI"].astype(str) + "_" + df["voyage_seq"].astype(str)
    df["gap_hours"] = dt.dt.total_seconds() / 3600.0

    n_v = df["voyage_id"].nunique()
    print(f"  航次切分：{df['MMSI'].nunique():,} 艘船 → {n_v:,} 个航次"
          f"（阈值 {gap_hours}h）")
    return df


# ==========================================================================
# 2. A. 基础运动特征
# ==========================================================================

def kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
    """按航次算基础运动特征。

    业务含义：
      · avg_sog / std_sog —— 航速水平与稳定性。异常绕航、漂航会体现在这里
      · stopped_ratio   —— 停留时长占比。长期滞留（尤其非港口区域）是可疑信号
      · course_std      —— 航向变化率的离散度。航线是否稳定
    """
    g = df.groupby("voyage_id")

    out = pd.DataFrame({
        "MMSI": g["MMSI"].first(),
        "vessel_name": g["vessel_name"].first(),
        "vessel_class": g["vessel_class"].first(),
        "t_start": g["datetime_utc"].min(),
        "t_end": g["datetime_utc"].max(),
        "n_points": g.size(),
        "duration_hours": (g["datetime_utc"].max() - g["datetime_utc"].min())
                          .dt.total_seconds() / 3600.0,
        "avg_sog": g["speed_over_ground_knots"].mean(),
        "std_sog": g["speed_over_ground_knots"].std(),
        "max_sog": g["speed_over_ground_knots"].max(),
        "lat_start": g["lat"].first(), "lon_start": g["lon"].first(),
        "lat_end": g["lat"].last(), "lon_end": g["lon"].last(),
    })

    # 停留时长占比
    df = df.copy()
    df["_stopped"] = df["speed_over_ground_knots"] < CONFIG["stopped_sog_knots"]
    stopped = df.groupby("voyage_id")["_stopped"].mean()
    out["stopped_ratio"] = stopped

    # 航向变化率的标准差
    # 航向是角度，直接 diff 会在 0/360 处跳变（359→1 应该是 +2 而不是 -358），
    # 所以先取相邻差、再折到 [-180, 180]
    def _course_std(s: pd.Series) -> float:
        d = s.diff().dropna()
        if d.empty:
            return np.nan
        d = (d + 180) % 360 - 180
        return float(d.std())

    out["course_std"] = df.groupby("voyage_id")["course_over_ground_deg"].apply(_course_std)

    return out.reset_index()


# ==========================================================================
# 3. ★ B. AIS 信号交叉印证特征（三分类）
# ==========================================================================

def build_coverage_grid(df: pd.DataFrame, grid_deg: float | None = None) -> pd.DataFrame:
    """构建接收覆盖度网格（数据驱动的代理指标）。

    思路：把海域切格，统计每格里**出现过多少不同船舶、横跨多少天**。
    覆盖好的海域，会有大量船舶长期稳定上报；覆盖弱的海域则稀疏。

    Returns:
        DataFrame[cell_lat, cell_lon, n_vessels, n_days, coverage_ok]
    """
    grid_deg = grid_deg or CONFIG["grid_deg"]

    g = df.copy()
    g["cell_lat"] = np.floor(g["lat"] / grid_deg) * grid_deg
    g["cell_lon"] = np.floor(g["lon"] / grid_deg) * grid_deg
    g["_day"] = g["datetime_utc"].dt.date

    agg = (g.groupby(["cell_lat", "cell_lon"])
             .agg(n_points=("MMSI", "size"),
                  n_vessels=("MMSI", "nunique"),
                  n_days=("_day", "nunique"))
             .reset_index())

    agg["coverage_ok"] = (
        (agg["n_vessels"] >= CONFIG["coverage_min_vessels"])
        & (agg["n_days"] >= CONFIG["coverage_min_days"])
    )

    n_ok = int(agg["coverage_ok"].sum())
    print(f"  覆盖度网格：{len(agg):,} 格，其中 {n_ok:,} 格判定为「覆盖良好」"
          f"（≥{CONFIG['coverage_min_vessels']}船 且 ≥{CONFIG['coverage_min_days']}天）")
    return agg


def classify_signal_gaps(df: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """★ 核心：把 AIS 信号缺口做成三分类，而不是"超阈值=异常"。

    对每个缺口：
      1. 用缺口前后两点的位置插值出缺口期间的**大致位置**
      2. 查这个位置的覆盖度
      3. 分三类：
           证据矛盾   位置落在「覆盖良好」格 → 本应收得到，却没收到 → 高风险
           证据缺失   位置落在覆盖弱的格   → 可能就是信号问题 → 不下结论
           证据支持   没有显著缺口

    Args:
        df:       带 voyage_id 的 AIS 点
        coverage: build_coverage_grid 的输出

    Returns:
        DataFrame，每个显著缺口一行
    """
    grid_deg = CONFIG["grid_deg"]
    min_gap_h = CONFIG["gap_min_minutes"] / 60.0

    # 用 numpy 位置数组做配对。
    # 踩过的坑：早先版本用 `before.loc[idx]`（idx 是 d 的标签索引）去索引子表，
    # 标签对不上直接 KeyError。改成位置数组后既快又不会有标签对齐的坑。
    d = df[["MMSI", "voyage_id", "datetime_utc", "lat", "lon", "gap_hours"]].reset_index(drop=True)

    mmsi = d["MMSI"].to_numpy()
    gap_h = d["gap_hours"].fillna(0).to_numpy()

    i_after = np.flatnonzero(gap_h >= min_gap_h)      # 重新出现的那个点
    i_before = i_after - 1                            # 消失前的最后一个点

    # 两道防御：① 不能越界 ② 前后两点必须属于同一条船
    ok = (i_before >= 0) & (mmsi[i_before] == mmsi[i_after])
    i_after, i_before = i_after[ok], i_before[ok]

    if len(i_after) == 0:
        print("  信号缺口：本批次没有超过阈值的缺口")
        return pd.DataFrame(columns=["MMSI", "voyage_id_before", "voyage_id_after",
                                     "gap_start", "gap_end", "gap_hours",
                                     "lat_mid", "lon_mid", "coverage_ok", "signal_class"])

    v_id = d["voyage_id"].to_numpy()
    lat = d["lat"].to_numpy()
    lon = d["lon"].to_numpy()
    dt = d["datetime_utc"].to_numpy()

    gaps = pd.DataFrame({
        "MMSI": mmsi[i_before],
        # 缺口归属：记在两个航次上。聚合时按"消失前所属的那个航次"计，
        # 因为缺口是那个航次的终点行为。
        "voyage_id_before": v_id[i_before],
        "voyage_id_after": v_id[i_after],
        "gap_start": dt[i_before],
        "gap_end": dt[i_after],
        "gap_hours": gap_h[i_after],
        "lat_mid": (lat[i_before] + lat[i_after]) / 2,
        "lon_mid": (lon[i_before] + lon[i_after]) / 2,
    })

    # 查覆盖度
    gaps["cell_lat"] = np.floor(gaps["lat_mid"] / grid_deg) * grid_deg
    gaps["cell_lon"] = np.floor(gaps["lon_mid"] / grid_deg) * grid_deg
    gaps = gaps.merge(
        coverage[["cell_lat", "cell_lon", "coverage_ok", "n_vessels"]],
        on=["cell_lat", "cell_lon"], how="left",
    )
    # 网格表里没有的格子 = 从没有船在那里上报过 = 覆盖弱
    gaps["coverage_ok"] = gaps["coverage_ok"].fillna(False).astype(bool)
    gaps["n_vessels"] = gaps["n_vessels"].fillna(0).astype(int)

    # ★ 三分类
    # ------------------------------------------------------------------
    # ★ 关键：光看覆盖度不够，必须叠加【运动状态】
    #
    # 实测校准（Hawaii 2017-01）：
    #   · 原始岸基 AIS，未降采样：航行中报文间隔中位数 1.17 min，停泊 2.95 min
    #   · 99 分位的间隔才 6 分钟；>30 分钟的间隔只占 0.1%
    #   → 所以 30 分钟以上的缺口是真·信号消失，不是正常抖动
    #
    # 但"消失"本身分两种完全不同的情况：
    #   船在**航行中**突然消失 → 才值得追问（正常不该关）
    #   船**停泊时**信号断了   → 大概率是在港关机/靠泊遮挡，很常见
    # 只按覆盖度分类会得出 66.8% 都是"矛盾"这种没有区分度的结果。
    # ------------------------------------------------------------------
    sog_before = df["speed_over_ground_knots"].to_numpy()[i_before]
    gaps["sog_before"] = sog_before
    gaps["was_underway"] = sog_before > 2.0        # 消失前是否在航行
    gaps["was_stopped"] = sog_before < 0.5         # 消失前是否停着

    def _classify(r) -> str:
        # ① 覆盖弱 → 不下结论（可能是盲区）
        if not r["coverage_ok"]:
            return "证据缺失"
        # ② 覆盖良好 + 航行中消失 → 高风险，这才叫"矛盾"
        if r["was_underway"]:
            return "证据矛盾"
        # ③ 覆盖良好但消失前停着 → 大概率在港关机/靠泊遮挡，存疑但不判风险
        return "证据缺失_停泊期间"

    gaps["signal_class"] = gaps.apply(_classify, axis=1)

    n_conflict = int((gaps["signal_class"] == "证据矛盾").sum())
    print(f"  信号缺口：{len(gaps):,} 处  →  "
          f"**证据矛盾 {n_conflict:,} 处（{n_conflict / max(len(gaps), 1) * 100:.1f}%）**")
    print(f"    其中：航行中消失 {int(gaps['was_underway'].sum()):,} 处，"
          f"停泊时消失 {int(gaps['was_stopped'].sum()):,} 处")
    print(f"    ⚠️ 只有「证据矛盾」（覆盖良好 + 航行中消失）才进风险判断。")
    print(f"       停泊期间断信号、以及覆盖弱区域的缺口，都必须放过——")
    print(f"       否则会把正常关机/信号盲区误伤成欺诈。")
    return gaps

    return gaps


# ==========================================================================
# 4. ★ C. 吃水/装卸特征（对应"空转贸易"）
# ==========================================================================

def draft_features(df: pd.DataFrame) -> pd.DataFrame:
    """★ 吃水变化特征 —— 原本想用来识别"空转贸易"，**实测后被改造**。

    ⚠️⚠️ 重要发现（HawaiiCoast_GT 2017-01 实测）⚠️⚠️
    ---------------------------------------------------------------
    最初的设想很漂亮：
      贸易声称在某港装/卸了 N 吨货 → 装卸必然改变吃水 → 吃水纹丝不动 = 空转贸易。
      监管也背书：MAS 2025《商品融资信息文件》把「通过吃水判断货物是否已装卸」
      列为 good practice。

    **但实测把这个设想打掉了**：
      · 有吃水记录的 115 艘船里，**113 艘（98.3%）整月只报同一个吃水值**
      · 只有 2 艘（1.7%）的吃水发生过变化
    **因为 AIS 的 draft 是船员在设备上人工录入的静态值，绝大多数人装上就不管了。**
    所以"吃水无变化"是**默认状态**，不是异常信号 —— 按原设计写进方案，
    评委一句"正常船不也不变吗"就能问倒。

    ✅ 但这个发现本身有很高的价值，**必须转成方案的论据而不是删掉**：
      它用我们自己实测的数字，证明了**单一依赖 AIS 任一字段做验真都不可靠**，
      必须多源交叉印证 —— 这正好是方案的核心理念。
      答辩话术：
        "我们实测发现 AIS 吃水字段 98.3% 的船舶整月不更新，因为它是人工录入的。
         这说明任何单一 AIS 字段都不能当真值用，必须多源交叉印证——
         这正是我们方案的设计前提，我们是用数据验证过这个前提的。"

    因此本函数输出的是**吃水字段可用性**，而不是"装卸证据"：
      · 先判断这条船是否属于"会更新吃水"的那 1.7%
      · 只有对这部分船，吃水变化才具备解释力
    """
    d = df[["voyage_id", "MMSI", "draft_depth_m"]].copy()

    # --- 第一步：判断哪些船"会更新吃水"（在整批数据上判断）---
    per_vessel = (d[d["draft_depth_m"].notna()]
                  .groupby("MMSI")["draft_depth_m"].nunique()
                  .rename("draft_n_unique"))
    reliable_mmsi = set(per_vessel[per_vessel >= 2].index)
    n_with = per_vessel.shape[0]
    n_rel = len(reliable_mmsi)

    print(f"  吃水字段可用性：{n_with:,} 艘船有吃水记录，"
          f"其中仅 {n_rel:,} 艘（{n_rel / max(n_with, 1) * 100:.1f}%）的吃水值发生过变化")
    if n_with and n_rel / n_with < 0.1:
        print(f"    ⚠️ 绝大多数船舶的吃水是**人工录入的静态值、从不更新**。")
        print(f"    → 「吃水无变化」是默认状态，**不能作为空转贸易的判据**。")
        print(f"    → 这个实测发现本身要写进文档：它证明单一AIS字段不可当真值，")
        print(f"       必须多源交叉印证（这正是方案的核心理念）。")

    # --- 第二步：只有"会更新吃水"的船，才计算装卸证据 ---
    has = d["draft_depth_m"].notna()
    gb = d[has].groupby("voyage_id")
    out = pd.DataFrame({
        "MMSI": gb["MMSI"].first(),          # 后面要按 MMSI 判断"该船是否会更新吃水"
        "draft_n": gb["draft_depth_m"].size(),
        "draft_min": gb["draft_depth_m"].min(),
        "draft_max": gb["draft_depth_m"].max(),
        "draft_mean": gb["draft_depth_m"].mean(),
        "draft_std": gb["draft_depth_m"].std(),
    })
    out["draft_range"] = out["draft_max"] - out["draft_min"]
    out = out.reset_index()

    total = d["voyage_id"].nunique()
    out["draft_reliable_vessel"] = out["MMSI"].isin(reliable_mmsi)

    # 三分类（诚实版）
    def _cls(r) -> str:
        if pd.isna(r["draft_range"]):
            return "证据缺失_无吃水数据"
        if not r["draft_reliable_vessel"]:
            return "证据缺失_该船吃水字段不更新"
        return "证据支持_有装卸" if r["draft_range"] > 0.2 else "存疑_未观测到装卸"

    out["draft_class"] = out.apply(_cls, axis=1)

    # MMSI 只在上面判断"该船是否会更新吃水"时用，合并前丢掉，
    # 否则和 kin 表 merge 会生成 MMSI_x / MMSI_y 两个重名列。
    out = out.drop(columns=["MMSI"])

    print(f"    航次级：{len(out):,} / {total:,} 个航次有吃水数据"
          f"（{len(out) / max(total, 1) * 100:.1f}%）")
    return out


# ==========================================================================
# 主流程
# ==========================================================================

def build_features(period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """跑完一个月的全流程。"""
    print(f"\n{'=' * 74}\n特征工程：{period}\n{'=' * 74}")

    df = load_month(period)
    df = split_voyages(df)

    print("\n  [A] 基础运动特征")
    kin = kinematic_features(df)

    print("\n  [B] AIS 信号交叉印证（三分类）")
    coverage = build_coverage_grid(df)
    gaps = classify_signal_gaps(df, coverage)
    # 汇总到航次级
    if not gaps.empty:
        # 按"消失前所属的航次"聚合——缺口是那个航次的终点行为
        gb = gaps.groupby("voyage_id_before")
        gap_agg = (gb["signal_class"]
                   .apply(lambda s: (s == "证据矛盾").sum())
                   .rename("n_signal_conflict"))
        gap_agg.index.name = "voyage_id"
        gap_max = gb["gap_hours"].max().rename("max_gap_hours")
        gap_max.index.name = "voyage_id"
        n_gap = gb.size().rename("n_gaps")
        n_gap.index.name = "voyage_id"
    else:
        gap_agg = pd.Series(dtype=int, name="n_signal_conflict")
        gap_max = pd.Series(dtype=float, name="max_gap_hours")
        n_gap = pd.Series(dtype=int, name="n_gaps")

    print("\n  [C] 吃水/装卸特征")
    draft = draft_features(df)

    # 合并
    feats = (kin
             .merge(gap_agg, on="voyage_id", how="left")
             .merge(gap_max, on="voyage_id", how="left")
             .merge(n_gap, on="voyage_id", how="left")
             .merge(draft, on="voyage_id", how="left"))
    for c in ("n_signal_conflict", "n_gaps"):
        feats[c] = feats[c].fillna(0).astype(int)
    # 没有吃水数据的航次 → 证据缺失
    feats["draft_class"] = feats["draft_class"].fillna("证据缺失_无吃水数据")

    return feats, df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阶段二：AIS 基础特征工程")
    ap.add_argument("--months", nargs="+", default=["2017_01"],
                    help="要处理的月份，如 2017_01 2017_02")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_feats = []

    for period in args.months:
        feats, voyages = build_features(period)
        all_feats.append(feats)

        # 航次明细落盘，供阶段三（共现网络）使用
        vp = OUT_DIR / f"voyages_{period}.csv"
        keep = ["MMSI", "voyage_id", "datetime_utc", "lat", "lon",
                "speed_over_ground_knots", "vessel_name", "vessel_class", "draft_depth_m"]
        voyages[keep].to_csv(vp, index=False, encoding="utf-8-sig")
        print(f"\n  → 航次明细：{vp}  ({len(voyages):,} 行)")

    feats_all = pd.concat(all_feats, ignore_index=True)
    tag = "_".join(args.months) if len(args.months) <= 3 else f"{args.months[0]}_等{len(args.months)}月"
    fp = OUT_DIR / f"vessel_features_{tag}.csv"
    feats_all.to_csv(fp, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 74}")
    print(f"✅ 特征表：{fp}")
    print(f"   航次数：{len(feats_all):,}")
    print(f"   列：{', '.join(feats_all.columns)}")
    print(f"\n   信号矛盾分布：{feats_all['n_signal_conflict'].value_counts().sort_index().to_dict()}")
    print(f"   吃水分类分布：{feats_all['draft_class'].value_counts().to_dict()}")
    print(f"{'=' * 74}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
