"""
阶段四 · 规则引擎【支撑模块】—— 银行审单环节的可解释核验层

定位
------------------------------------------------------------------
规则引擎是**支撑主锚点的补充判断层**，不是核心卖点。
它的价值在于**可解释**：银行合规最看重"为什么拒绝"，
规则触发的结论能直接写进审单意见，ML 的分数不能。

⚠️ 关于数据真实性 —— 必须讲清楚
------------------------------------------------------------------
本模块的"申报单据"是**构造的验证用例**，不是真实银行单据。
理由：公开数据里不存在企业提交给银行的贸易单据（这是本项目第一章
指出的研究空白）。

构造方式是**受控合成**：
  · 正常单据：从**真实航次**反推（船真的走了这条航线 → 开一张相符的单据）
  · 异常单据：在真实航次上**注入已知的欺诈模式**（改航线/改靠泊港/改时间）
这样每个用例都有确定答案，可以算检出率。

**文档与答辩中必须明确标注为"构造的验证用例"。**

三条规则
------------------------------------------------------------------
  规则1  申报航线（起止港） vs AIS 实际轨迹
  规则2  申报靠泊港/时间   vs AIS 实际靠泊记录
  规则3  AIS 信号交叉印证（三分类：证据支持 / 证据缺失 / 证据矛盾）

规则3 的设计遵循阶段二的结论：**"没有AIS信号 ≠ 关闭了AIS"**，
只有在"覆盖良好 且 船舶正在航行"时段的缺口才判为矛盾。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
OUT = ROOT / "experiments/results"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "port_grid_deg": 0.03,     # 停留点聚成"港口"的网格粒度
    "port_min_vessels": 3,     # 一个格子至少这么多船停过才算港口
    "stopped_sog": 0.5,
    "arrival_radius_deg": 0.05,  # 判定"到达某港"的半径
    "gap_conflict_hours": 2.0,   # 超过这个时长的缺口才考虑
}

# 注入的欺诈模式（每条对应一种现实中的贸易融资欺诈手法）
FRAUD_TYPES = {
    "伪造航线": "申报的起止港与实际轨迹不符（虚构运输凭证）",
    "伪造靠泊": "申报的靠泊港船舶从未到过",
    "时间不符": "申报的运输时段与 AIS 记录不符（单据倒签）",
    "缺口期申报": "申报在 AIS 信号中断期间完成运输（疑似关闭AIS掩盖）",
}


# ==========================================================================
# 1. 从真实数据里识别"港口"
# ==========================================================================

def detect_ports(voyages: pd.DataFrame) -> pd.DataFrame:
    """从停留点聚类出"港口"。

    HawaiiCoast_GT 没有港口表，用数据驱动的方式提取：
    把停泊状态（SOG<0.5）的点按空间网格聚合，
    一个格子里有多艘船长期停留 → 认定为一个港口/锚地。
    """
    g = CONFIG["port_grid_deg"]
    d = voyages[voyages["speed_over_ground_knots"] < CONFIG["stopped_sog"]].copy()
    d["clat"] = np.floor(d["lat"] / g) * g
    d["clon"] = np.floor(d["lon"] / g) * g

    agg = (d.groupby(["clat", "clon"])
             .agg(n_vessels=("MMSI", "nunique"),
                  n_points=("MMSI", "size"),
                  lat=("lat", "mean"), lon=("lon", "mean"))
             .reset_index())
    ports = agg[agg["n_vessels"] >= CONFIG["port_min_vessels"]] \
        .sort_values("n_vessels", ascending=False).reset_index(drop=True)
    ports["port_id"] = [f"P{i:02d}" for i in range(len(ports))]

    print(f"  港口识别：{len(ports)} 个停泊热点"
          f"（≥{CONFIG['port_min_vessels']}艘船停过）")
    for _, p in ports.head(6).iterrows():
        print(f"    {p['port_id']}  ({p['lat']:.3f}, {p['lon']:.3f})  "
              f"{int(p['n_vessels'])} 艘船 / {int(p['n_points']):,} 个停留点")
    return ports


def nearest_port(lat: float, lon: float, ports: pd.DataFrame) -> tuple[str | None, float]:
    """返回最近的港口 ID 和距离（度）。"""
    if ports.empty:
        return None, np.inf
    d = np.hypot(ports["lat"] - lat, ports["lon"] - lon)
    i = int(d.idxmin())
    return ports.loc[i, "port_id"], float(d.loc[i])


# ==========================================================================
# 2. 构造申报单据（受控合成）
# ==========================================================================

@dataclass
class Declaration:
    decl_id: str
    voyage_id: str          # ★ 必须记录是哪条航次，否则核验会拿整月数据去比
    MMSI: int
    vessel_name: str
    declared_departure: str
    declared_arrival: str
    declared_start: pd.Timestamp
    declared_end: pd.Timestamp
    is_fraud: bool
    fraud_type: str = ""
    note: str = ""


def build_declarations(voyages: pd.DataFrame, ports: pd.DataFrame,
                       n_normal: int = 10, seed: int = 42) -> list[Declaration]:
    """构造申报单据：正常单据从真实航次反推，异常单据注入已知欺诈模式。

    Returns:
        Declaration 列表（含真值标签 is_fraud / fraud_type）
    """
    rng = np.random.default_rng(seed)
    decls: list[Declaration] = []

    # --- 挑一些"从港口出发、到港口结束"的干净航次做底 ---
    # ⚠️ 这里的距离阈值必须和 apply_rules 里核验用的阈值**完全一致**。
    #    踩过的坑：构造时用 0.1、核验时用 0.05，导致正常单据被大面积误报
    #    （船确实停在该港附近，但距离落在 0.05~0.1 之间，核验判为"不在附近"）。
    RADIUS = CONFIG["arrival_radius_deg"]
    grp = voyages.sort_values("datetime_utc").groupby("voyage_id")
    cand = []
    for vid, t in grp:
        if len(t) < 50:
            continue
        p0, d0 = nearest_port(t["lat"].iloc[0], t["lon"].iloc[0], ports)
        p1, d1 = nearest_port(t["lat"].iloc[-1], t["lon"].iloc[-1], ports)
        if p0 and p1 and p0 != p1 and d0 <= RADIUS and d1 <= RADIUS:
            cand.append((vid, t, p0, p1))
    if len(cand) < n_normal:
        print(f"  ⚠️ 只找到 {len(cand)} 条干净航次，少于目标 {n_normal}")

    picked = cand[:n_normal]
    port_ids = ports["port_id"].tolist()

    for i, (vid, t, p0, p1) in enumerate(picked):
        decls.append(Declaration(
            decl_id=f"D{i:03d}", voyage_id=vid, MMSI=int(t["MMSI"].iloc[0]),
            vessel_name=str(t["vessel_name"].iloc[0]),
            declared_departure=p0, declared_arrival=p1,
            declared_start=t["datetime_utc"].iloc[0],
            declared_end=t["datetime_utc"].iloc[-1],
            is_fraud=False, note="从真实航次反推，申报与轨迹相符",
        ))

        # --- 在同一条航次上注入 4 种欺诈各一例 ---
        for j, ftype in enumerate(FRAUD_TYPES):
            if i >= 3:
                break
            d = Declaration(
                decl_id=f"D{i:03d}F{j}", voyage_id=vid, MMSI=int(t["MMSI"].iloc[0]),
                vessel_name=str(t["vessel_name"].iloc[0]),
                declared_departure=p0, declared_arrival=p1,
                declared_start=t["datetime_utc"].iloc[0],
                declared_end=t["datetime_utc"].iloc[-1],
                is_fraud=True, fraud_type=ftype,
            )
            if ftype == "伪造航线":
                # 把目的港改成一个这条船没去过的
                other = [p for p in port_ids if p not in (p0, p1)]
                d.declared_arrival = str(rng.choice(other))
                d.note = "申报目的港与实际不符"
            elif ftype == "伪造靠泊":
                other = [p for p in port_ids if p not in (p0, p1)]
                d.declared_departure = str(rng.choice(other))
                d.note = "申报起运港船舶从未停靠"
            elif ftype == "时间不符":
                # 单据倒签：把申报时段整体前移
                d.declared_start = t["datetime_utc"].iloc[0] - pd.Timedelta(days=3)
                d.declared_end = t["datetime_utc"].iloc[0] - pd.Timedelta(days=1)
                d.note = "申报时段早于实际航行（倒签）"
            elif ftype == "缺口期申报":
                d.declared_start = t["datetime_utc"].iloc[0] + pd.Timedelta(hours=1)
                d.declared_end = t["datetime_utc"].iloc[0] + pd.Timedelta(hours=4)
                d.note = "申报时段落在AIS信号异常/停滞区间"
            decls.append(d)

    n_f = sum(1 for d in decls if d.is_fraud)
    print(f"  构造单据：{len(decls)} 张（正常 {len(decls) - n_f}，"
          f"注入欺诈 {n_f}）")
    return decls


# ==========================================================================
# 3. 规则引擎
# ==========================================================================

@dataclass
class RuleResult:
    decl_id: str
    triggered: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def risk(self) -> str:
        if not self.triggered:
            return "低"
        return "高" if len(self.triggered) >= 2 else "中"


def apply_rules(d: Declaration, voyages: pd.DataFrame, ports: pd.DataFrame) -> RuleResult:
    """对一张申报单据跑三条规则。"""
    r = RuleResult(decl_id=d.decl_id)
    # ⚠️ 必须按 voyage_id 过滤，不能用整月数据 ——
    #    用整月数据会导致"起点来自另一条航次"，正常单据被大面积误报。
    t = voyages[voyages["voyage_id"] == d.voyage_id].sort_values("datetime_utc")
    if t.empty:
        r.triggered.append("规则0_无轨迹")
        r.evidence.append(f"AIS 中找不到 MMSI {d.MMSI} 的任何轨迹")
        return r

    # ---- 规则1：申报航线 vs 实际轨迹 ----
    p_start, _ = nearest_port(t["lat"].iloc[0], t["lon"].iloc[0], ports)
    p_end, _ = nearest_port(t["lat"].iloc[-1], t["lon"].iloc[-1], ports)
    if d.declared_departure != p_start:
        r.triggered.append("规则1_航线不符")
        r.evidence.append(
            f"申报起运港 {d.declared_departure}，实际起点最近港 {p_start}")
    if d.declared_arrival != p_end:
        r.triggered.append("规则1_航线不符")
        r.evidence.append(
            f"申报目的港 {d.declared_arrival}，实际终点最近港 {p_end}")

    # ---- 规则2：申报靠泊 vs 实际位置 ----
    # ⚠️ 语义要正确：申报时段是**航行时段**，船当然在动，
    #    所以不能要求"时段内停靠过"（那会把正常单据全部误报 —— 踩过这个坑）。
    #    正确做法：检查**申报的起运/到达时刻，船是否分别在起运港/目的港附近**。
    win = t[(t["datetime_utc"] >= d.declared_start) & (t["datetime_utc"] <= d.declared_end)]
    if win.empty:
        r.triggered.append("规则2_时段无记录")
        r.evidence.append(f"申报时段 {d.declared_start:%Y-%m-%d %H:%M}–"
                          f"{d.declared_end:%Y-%m-%d %H:%M} 内无任何 AIS 记录")
    else:
        r_port = CONFIG["arrival_radius_deg"]

        def _near(pt_lat, pt_lon, port_id: str) -> bool:
            """某位置是否在指定港口附近。"""
            pid, dist = nearest_port(pt_lat, pt_lon, ports)
            return pid == port_id and dist <= r_port

        # 起运时刻：取申报时段起点附近（±1小时）的轨迹位置
        near_start = win.iloc[: max(1, len(win) // 20)]
        ok_dep = any(_near(p["lat"], p["lon"], d.declared_departure)
                     for _, p in near_start.iterrows())
        # 到达时刻：取申报时段末尾附近的位置
        near_end = win.iloc[-max(1, len(win) // 20):]
        ok_arr = any(_near(p["lat"], p["lon"], d.declared_arrival)
                     for _, p in near_end.iterrows())

        if not ok_dep:
            r.triggered.append("规则2_靠泊不符")
            r.evidence.append(
                f"申报起运时刻，船舶实际不在申报起运港 {d.declared_departure} 附近")
        if not ok_arr:
            r.triggered.append("规则2_靠泊不符")
            r.evidence.append(
                f"申报到达时刻，船舶实际不在申报目的港 {d.declared_arrival} 附近")

    # ---- 规则3：AIS 信号交叉印证（三分类，来自阶段二的设计）----
    if len(win) >= 2:
        dt = win["datetime_utc"].diff().dt.total_seconds() / 3600
        idx = dt.idxmax()
        max_gap = float(dt.max()) if pd.notna(dt.max()) else 0.0
        if max_gap >= CONFIG["gap_conflict_hours"]:
            # 缺口前的航速 —— 决定这是"证据矛盾"还是"证据缺失"
            pos = win.index.get_loc(idx)
            prev_sog = float(win["speed_over_ground_knots"].iloc[max(pos - 1, 0)])
            if prev_sog > 2.0:
                r.triggered.append("规则3_信号矛盾")
                r.evidence.append(
                    f"申报时段内出现 {max_gap:.1f} 小时信号缺口，"
                    f"且缺口前船舶正在航行（航速 {prev_sog:.1f} 节）"
                    f"—— 本应有良好覆盖")
            else:
                r.evidence.append(
                    f"申报时段内有 {max_gap:.1f} 小时信号缺口，"
                    f"但缺口前船舶处于停泊状态，判为「证据缺失」不予采信")
    return r


# ==========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阶段四：规则引擎（构造用例定性演示）")
    ap.add_argument("--month", default="2017_01")
    ap.add_argument("--n-normal", type=int, default=10)
    args = ap.parse_args(argv)

    fp = PROC / f"voyages_{args.month}.csv"
    if not fp.exists():
        print(f"❌ 找不到 {fp}\n   先跑：src/feature_engineering.py --months {args.month}")
        return 1

    print(f"\n{'=' * 78}")
    print("阶段四：规则引擎 —— 申报单据核验（构造用例定性演示）")
    print(f"{'=' * 78}")
    print("  ⚠️ 申报单据为**构造的验证用例**（正常单据从真实航次反推，")
    print("     异常单据注入已知欺诈模式）。不是真实银行单据。")

    print(f"\n【1/3】载入航次 + 识别港口")
    voy = pd.read_csv(fp)
    voy["datetime_utc"] = pd.to_datetime(voy["datetime_utc"], errors="coerce", utc=True)
    print(f"  {len(voy):,} 个点 / {voy['MMSI'].nunique()} 艘船")
    ports = detect_ports(voy)

    print(f"\n【2/3】构造申报单据")
    decls = build_declarations(voy, ports, n_normal=args.n_normal)

    print(f"\n【3/3】跑规则引擎")
    print(f"{'=' * 78}")
    rows = []
    for d in decls:
        r = apply_rules(d, voy, ports)
        # 预测量：触发了任一规则 → 判为可疑
        pred_fraud = len(r.triggered) > 0
        rows.append({
            # MMSI 要带上：Demo 界面要用它去查轨迹（踩过的坑：漏了这一列，
            # Streamlit 里 d['MMSI'] 直接 KeyError 把页面打崩）
            "decl_id": d.decl_id, "MMSI": d.MMSI,
            "voyage_id": d.voyage_id, "vessel": d.vessel_name,
            "declared": f"{d.declared_departure}→{d.declared_arrival}",
            "truth_fraud": d.is_fraud, "fraud_type": d.fraud_type,
            "pred_fraud": pred_fraud, "risk": r.risk,
            "n_rules": len(r.triggered),
            "triggered": ";".join(r.triggered),
            "evidence": " | ".join(r.evidence)[:220],
        })

    res = pd.DataFrame(rows)

    # --- 打印 ---
    print(f"  {'单据':<9}{'真值':<7}{'判定':<6}{'风险':<5}{'触发规则'}")
    print(f"  {'-' * 70}")
    for _, r in res.iterrows():
        truth = "欺诈" if r["truth_fraud"] else "正常"
        pred = "可疑" if r["pred_fraud"] else "通过"
        mark = "✅" if r["truth_fraud"] == r["pred_fraud"] else "❌"
        print(f"  {r['decl_id']:<9}{truth:<7}{pred:<6}{r['risk']:<5}"
              f"{r['triggered'][:44]:<46}{mark}")

    # --- 定性结论（不做统计显著性主张，样本太小）---
    tp = int(((res.truth_fraud) & (res.pred_fraud)).sum())
    fn = int(((res.truth_fraud) & (~res.pred_fraud)).sum())
    fp = int(((~res.truth_fraud) & (res.pred_fraud)).sum())
    tn = int(((~res.truth_fraud) & (~res.pred_fraud)).sum())

    print(f"\n{'=' * 78}")
    print("定性小结（⚠️ 样本量小，此处只作定性演示，不做统计显著性主张）")
    print(f"{'=' * 78}")
    print(f"  注入的欺诈单据 {tp + fn} 张 → 检出 {tp} 张，漏报 {fn} 张")
    print(f"  正常单据     {tn + fp} 张 → 误报 {fp} 张，正确放行 {tn} 张")
    if fp:
        print(f"\n  误报明细：")
        for _, r in res[(~res.truth_fraud) & (res.pred_fraud)].iterrows():
            print(f"    {r['decl_id']}  {r['triggered']}  ← {r['evidence'][:90]}")
    if fn:
        print(f"\n  漏报明细：")
        for _, r in res[(res.truth_fraud) & (~res.pred_fraud)].iterrows():
            print(f"    {r['decl_id']}  ({r['fraud_type']})  ← 该模式未触发任何规则")

    op = OUT / f"rule_engine_results_{args.month}.csv"
    res.to_csv(op, index=False, encoding="utf-8-sig")
    print(f"\n  → {op}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
