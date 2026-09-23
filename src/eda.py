"""
阶段一 EDA：HawaiiCoast_GT 探索性分析

产出三张图（全部落盘到 experiments/results/，计划书"数据与合规说明"章节要用）：
    fig1_incident_map.png     真实标注事件的轨迹地图 ★ 最重要的一张
    fig2_incident_profile.png 事件类型 / 证据等级分布
    fig3_draft_availability.png  吃水字段可用性（诚实展示数据边界）

为什么用脚本而不是 Jupyter notebook
------------------------------------------------------------------
总纲阶段一要求的是 `notebooks/01_eda.ipynb`，这里改成脚本 `src/eda.py`，理由：
  1. 总纲同时要求 `run_all.py` 能"一键跑完全流程"——脚本才能被 run_all 调用，
     notebook 需要额外依赖（jupyter/nbconvert）才能批量执行
  2. 16GB 数据不适合反复在交互式环境里试错，脚本的"读多少、算什么"更容易控制
  3. 结论可复现、可 diff
如果后面确实需要交互式探索，再单独写 notebook 探索子集即可。

内存策略（16GB 数据 / 16GB 机器）
------------------------------------------------------------------
**绝不**把 AIS_data 全部读进内存。做法：
  1. 只读小的元数据文件（船名表 30KB、事件表 212 行）
  2. 需要轨迹时，**只读事件涉及的那几个月**，并在读入时就用 usecols 裁列
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无窗口后端，服务器/批处理环境必需
import matplotlib.pyplot as plt
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- 中文字体（Windows）---
for _f in ("Microsoft YaHei", "SimHei", "SimSun"):
    try:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/HawaiiCoast_GT"
AIS = DATA / "AIS_data"
OUT = ROOT / "experiments/results"
OUT.mkdir(parents=True, exist_ok=True)

# 我们真正需要的列。用 usecols 裁列，能显著省内存和省时间。
USECOLS = [
    "MMSI", "datetime_utc", "lat", "lon",
    "speed_over_ground_knots", "course_over_ground_deg", "heading_deg",
    "vessel_name", "vessel_class", "draft_depth_m", "incident_num",
]


# ==========================================================================
# 1. 元数据总览
# ==========================================================================

def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读两个小表：船舶清单 + 事件标注。"""
    vessels = pd.read_csv(AIS / "vessel_names_and_classes.csv")
    incidents = pd.read_csv(DATA / "incident_data/hawaii_primary_trajectories_of_interest_2017_2020.csv")
    return vessels, incidents


def print_overview(vessels: pd.DataFrame, incidents: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("HawaiiCoast_GT 总览")
    print("=" * 74)
    print(f"  唯一船舶数        : {vessels['MMSI'].nunique():,}")
    print(f"  事件标注条目      : {len(incidents):,} 行")
    print(f"  唯一事件数        : {incidents['incident_num'].nunique():,}")
    print(f"  涉及船舶数        : {incidents['MMSI'].nunique():,}")

    print("\n  【船舶类别分布】(Top 8)")
    for k, v in vessels["vessel_class"].value_counts().head(8).items():
        print(f"    {str(k):28s} {v:5,}")

    print("\n  【AIS事件类型分布】")
    for k, v in incidents["ais_incident_type"].value_counts().items():
        print(f"    {str(k):42s} {v:4,}")

    print("\n  【AIS证据等级分布】★ 这张分布本身就是我们的机会说明")
    ev = incidents["ais_incident_evidence"].value_counts()
    for k, v in ev.items():
        print(f"    {str(k):24s} {v:4,}  ({v / len(incidents) * 100:4.1f}%)")
    no_ev = ev.get("No obvious evidence", 0)
    if no_ev:
        print(f"\n    ⭐ 有 {no_ev} 起事件的 AIS 里【没有肉眼可见的证据】——")
        print("       README 原话：这是 'a rich area to develop more specific features")
        print("       that may reveal the ground truth'。这正是我们做特征工程的位置。")


# ==========================================================================
# 2. ★ 真实事件轨迹图（计划书要用的一张）
# ==========================================================================

def pick_showcase_incidents(incidents: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """挑几个适合放进 PPT 的事件。

    筛选标准（可解释、答辩时能讲）：
      1. 证据等级为 High evidence —— 图上肉眼能看出异常，讲故事最有说服力
      2. AIS 时间边界完整（起止时间都有）
      3. 优先挑"两船互动"的事件（拖带/救援）—— 这类事件天然涉及
         **船舶时空共现**，正好对应我们的主锚点
      4. 参与船舶记录较少的事件优先（图干净、好讲）
    """
    df = incidents.dropna(subset=["ais_incident_start_bound_hst", "ais_incident_end_bound_hst"]).copy()
    df = df[df["ais_incident_evidence"] == "High evidence"]

    size = df.groupby("incident_num").size()
    two = size[size == 2].index.tolist()      # 两船互动 → 直接展示"时空共现"
    one = size[size == 1].index.tolist()      # 单船事件 → 展示轨迹形态异常

    # 选法：一半给两船互动（对应主锚点），一半给不同类型的单船事件（体现多样性）
    picked: list[int] = []
    picked += two[: max(1, n // 2)]

    # 单船事件按类型去重，避免四张图全是同一种
    seen_types: set[str] = set()
    for inum in one:
        t = str(df[df["incident_num"] == inum]["ais_incident_type"].iloc[0])
        if t in seen_types:
            continue
        seen_types.add(t)
        picked.append(inum)
        if len(picked) >= n:
            break
    # 不够就从两船互动里补
    for inum in two:
        if len(picked) >= n:
            break
        if inum not in picked:
            picked.append(inum)

    out = df[df["incident_num"].isin(picked[:n])].copy()
    # 转成 datetime，方便按时间窗过滤
    for c in ("ais_incident_start_bound_hst", "ais_incident_end_bound_hst"):
        out[c] = pd.to_datetime(out[c], errors="coerce", utc=True)
    return out


def load_tracks_for(incidents: pd.DataFrame, pad_hours: int = 6) -> pd.DataFrame:
    """只读事件涉及的那些月份，并按事件时间窗裁剪轨迹。

    ⚠️ 内存红线：不要全量读。这里按月读、按 MMSI + 时间窗过滤后再留下。
    """
    frames = []
    for _, row in incidents.iterrows():
        mmsi = row["MMSI"]
        t0, t1 = row["ais_incident_start_bound_hst"], row["ais_incident_end_bound_hst"]
        if pd.isna(t0) or pd.isna(t1):
            continue
        t0 -= pd.Timedelta(hours=pad_hours)
        t1 += pd.Timedelta(hours=pad_hours)

        # 事件可能跨月，取涉及的月份
        months = pd.date_range(t0.tz_convert(None).replace(day=1), t1.tz_convert(None), freq="MS")
        for m in months:
            f = AIS / f"Hawaii_{m.year}_{m.month:02d}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, usecols=lambda c: c in USECOLS, low_memory=False)
            df = df[df["MMSI"] == mmsi]
            if df.empty:
                continue
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce", utc=True)
            df = df[(df["datetime_utc"] >= t0) & (df["datetime_utc"] <= t1)]
            if not df.empty:
                df["incident_num"] = row["incident_num"]
                frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _wrap(s: str, width: int = 46, max_lines: int = 3) -> str:
    """按宽度折行，超出 max_lines 就截断加省略号。"""
    import textwrap
    lines = textwrap.wrap(str(s), width=width)[:max_lines]
    if len(textwrap.wrap(str(s), width=width)) > max_lines:
        lines[-1] = lines[-1][: width - 1] + "…"
    return "\n".join(lines)


def fig_incident_map(incidents: pd.DataFrame, tracks: pd.DataFrame) -> None:
    """画真实事件的轨迹地图 —— 计划书里的核心配图。

    排版上踩过的坑（都已修）：
      · 事件描述太长会把标题挤到相邻面板上 → 折行 + 截断
      · 每个子图各配一个 colorbar 太占地方且重复 → 改用**全局共享**一个
      · 拖带类事件里两条船位置几乎重合，船名会叠在一起 → 标注错开
    """
    inums = sorted(incidents["incident_num"].unique())
    n = len(inums)
    if n == 0 or tracks.empty:
        print("  ⚠️ 没有可用于绘图的事件轨迹，跳过 fig1")
        return

    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.6 * ncol, 4.6 * nrow), squeeze=False)

    # 全局航速范围，保证各面板配色可比
    vmax = float(tracks["speed_over_ground_knots"].quantile(0.98)) or 20.0
    sc_last = None

    for idx, inum in enumerate(inums):
        ax = axes[idx // ncol][idx % ncol]
        grp = incidents[incidents["incident_num"] == inum]
        meta = grp.iloc[0]

        names = []
        for j, (mmsi, t) in enumerate(tracks[tracks["incident_num"] == inum].groupby("MMSI")):
            t = t.sort_values("datetime_utc")
            names.append(str(t["vessel_name"].iloc[0]))
            sc_last = ax.scatter(t["lon"], t["lat"], c=t["speed_over_ground_knots"],
                                 cmap="coolwarm", s=7, alpha=0.85, vmin=0, vmax=vmax,
                                 edgecolors="none")
            ax.plot(t["lon"], t["lat"], lw=0.5, alpha=0.30, color="dimgray", zorder=0)

        # 船名标注：放在轨迹两个不同端点，避免重叠
        for j, (mmsi, t) in enumerate(tracks[tracks["incident_num"] == inum].groupby("MMSI")):
            t = t.sort_values("datetime_utc")
            name = str(t["vessel_name"].iloc[0])
            k = 0 if j % 2 == 0 else -1          # 一条标在起点，一条标在终点
            ax.annotate(name, (t["lon"].iloc[k], t["lat"].iloc[k]),
                        fontsize=8.5, fontweight="bold", color="#1a1a1a",
                        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#888", lw=0.5, alpha=0.85),
                        xytext=(4, 4 if j % 2 == 0 else -14), textcoords="offset points")

        title = f"事件 #{inum}｜{meta['ais_incident_type']}"
        if len(names) >= 2:
            title += "（两船互动）"
        ax.set_title(title + "\n" + _wrap(meta["report_based_incident_description"]),
                     fontsize=9.5, loc="left")
        # ⚠️ 必须显式关掉科学计数法与 offset——否则经度会显示成 "-1.578e2"，
        #    地图坐标看起来完全不像经纬度。
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.set_xlabel("经度", fontsize=9)
        ax.set_ylabel("纬度", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.22)

    # 关掉多余的空面板
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("HawaiiCoast_GT：真实标注事件的 AIS 轨迹　"
                 "（Sandia National Laboratories ｜ CC BY 4.0 ｜ 2017–2020 夏威夷沿海）",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))

    # 全局共享一个色条
    if sc_last is not None:
        cax = fig.add_axes((0.30, 0.012, 0.40, 0.018))
        cb = fig.colorbar(sc_last, cax=cax, orientation="horizontal")
        cb.set_label("对地航速（节）", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    p = OUT / "fig1_incident_map.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {p}")


# ==========================================================================
# 3. 事件类型 / 证据等级分布
# ==========================================================================

def fig_incident_profile(incidents: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    vc = incidents["ais_incident_type"].value_counts().head(12)[::-1]
    ax1.barh(vc.index, vc.values, color="#3b6ea5")
    ax1.set_title("AIS 事件类型分布（Top 12）", fontsize=11, fontweight="bold")
    ax1.set_xlabel("事件数")
    ax1.grid(axis="x", alpha=0.25)

    order = ["High evidence", "Good evidence", "Some evidence", "No obvious evidence"]
    ev = incidents["ais_incident_evidence"].value_counts().reindex(order).fillna(0)
    colors = ["#2e7d32", "#7cb342", "#f9a825", "#c62828"]
    bars = ax2.bar(range(len(ev)), ev.values, color=colors)
    ax2.set_xticks(range(len(ev)))
    ax2.set_xticklabels([o.replace(" evidence", "\nevidence") for o in ev.index], fontsize=8)
    ax2.set_title("AIS 中可见的证据等级\n（来源：HawaiiCoast_GT README 定义）",
                  fontsize=11, fontweight="bold")
    ax2.set_ylabel("事件数")
    ax2.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, ev.values):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{int(v)}", ha="center", fontsize=9)
    ax2.annotate("无肉眼可见证据的部分，\n是用特征工程挖掘真值的机会",
                 xy=(3, ev.values[3]), xytext=(1.4, ev.values.max() * 0.75),
                 fontsize=8, color="#c62828",
                 arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.1))

    fig.tight_layout()
    p = OUT / "fig2_incident_profile.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {p}")


# ==========================================================================
# 4. 吃水字段可用性（诚实展示数据边界）
# ==========================================================================

def fig_draft_availability(months: int = 3) -> None:
    """扫前 N 个月的吃水字段可用性。

    为什么要专门画这张：吃水是我们最好讲的一条旗舰特征，但它有缺失。
    **主动把缺失率画出来**，比等评委问"你数据全不全"要强得多。
    """
    files = sorted(AIS.glob("Hawaii_*.csv"))[:months]
    if not files:
        print("  ⚠️ 找不到 AIS 文件，跳过 fig3")
        return

    stats = []
    for f in files:
        d = pd.read_csv(f, usecols=["draft_depth_m", "vessel_class"], low_memory=False)
        overall = d["draft_depth_m"].notna().mean()
        # 分船型看：不同船型的吃水上报率差异很大
        by_class = d.groupby("vessel_class")["draft_depth_m"].apply(lambda s: s.notna().mean())
        stats.append({"file": f.stem, "overall": overall, "by_class": by_class})
        print(f"     {f.stem}: 吃水非空 {overall * 100:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    names = [s["file"].replace("Hawaii_", "") for s in stats]
    ax1.bar(names, [s["overall"] * 100 for s in stats], color="#3b6ea5")
    ax1.set_ylim(0, 100)
    ax1.set_title("吃水字段非空率（按月）", fontsize=11, fontweight="bold")
    ax1.set_ylabel("非空占比 (%)")
    ax1.grid(axis="y", alpha=0.25)
    for i, s in enumerate(stats):
        ax1.text(i, s["overall"] * 100 + 1.5, f"{s['overall'] * 100:.1f}%", ha="center", fontsize=9)

    bc = stats[-1]["by_class"].dropna().sort_values()
    ax2.barh(bc.index.astype(str), bc.values * 100, color="#6a9fd8")
    ax2.set_xlim(0, 100)
    ax2.set_title(f"吃水非空率（按船型，{names[-1]}）", fontsize=11, fontweight="bold")
    ax2.set_xlabel("非空占比 (%)")
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("吃水字段可用性 —— 诚实展示数据边界（缺失 ≠ 没装货，须按“证据缺失”处理）",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = OUT / "fig3_draft_availability.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {p}")


# ==========================================================================

def main() -> int:
    if not DATA.exists():
        print(f"❌ 找不到数据集：{DATA}")
        print("   先跑：  .venv/Scripts/python.exe src/setup_data.py")
        return 1

    print("\n【1/4】读元数据")
    vessels, incidents = load_metadata()
    print_overview(vessels, incidents)

    print("\n【2/4】挑选展示事件 + 载入轨迹（只读需要的月份，避免 16GB 全量入内存）")
    showcase = pick_showcase_incidents(incidents)
    print(f"  选中事件：{showcase['incident_num'].unique().tolist()}")
    tracks = load_tracks_for(showcase)
    print(f"  轨迹点数：{len(tracks):,}")

    print("\n【3/4】出图")
    fig_incident_map(showcase, tracks)
    fig_incident_profile(incidents)

    print("\n【4/4】吃水字段可用性")
    fig_draft_availability()

    print(f"\n全部图表已落盘：{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
