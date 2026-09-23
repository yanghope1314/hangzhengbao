"""
工e航鉴 · Streamlit 演示界面（阶段七）

设计原则
------------------------------------------------------------------
1. **主锚点优先**：关联方共现网络图放在**最显眼的位置**，不与其它图表平铺。
   路演讲解的重点就在这一张图上。
2. **产品形态提示**：这不是"我们的系统"，而是**演示界面**。
   真实形态是嵌入银行审单流程的验真微服务/API（见 docs/竞品分析与产品定位.md）。
   界面上专门有一个"模拟 API 调用"的说明块。
3. **诚实标注**：凡是构造的数据（申报单据、合成团伙）都明确标出，
   不能让人误以为是真实银行数据。

启动
------------------------------------------------------------------
    .venv/Scripts/streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
RES = ROOT / "experiments/results"

DEFAULT_MONTH = "2017_01"

# ==========================================================================
# ★ 方案名：全项目只在这里定义一次
#   现行推荐：《工e航鉴——基于船舶物理共现网络的国际结算贸易背景核验系统》
#   命名依据（往届特等奖铁律）：工X/eX + —— + 基于[具体算法]的[银行流程步骤]
#     · 工e    = 工行 + 数字化（16届5个特等奖中3个带工/e前缀）
#     · 航鉴   = 航运 + 鉴真
#     · 共现网络 = 算法锚点；国际结算 = 银行真实流程步骤
#   如需换名，**只改下面这一行即可**。
# ==========================================================================
PRODUCT_NAME = "工e航鉴"
PRODUCT_SUB = "基于船舶物理共现网络的国际结算贸易背景核验系统"

st.set_page_config(page_title=f"{PRODUCT_NAME} · 演示", layout="wide",
                   initial_sidebar_state="expanded")


# ==========================================================================
# 数据加载（带缓存，避免每次交互都读大文件）
# ==========================================================================

SNAP = ROOT / "demo_data"


@st.cache_data(show_spinner="载入数据…")
def load_all(month: str):
    """载入演示数据。

    ★ 优先读 `demo_data/`（**演示快照**，约 1.8 MB，随代码一起提交）。
      完整数据集 16 GB，不适合随作品提交，也不应要求评委下载。
      只有在快照缺失时才回退到完整数据目录。
    """
    out = {}
    use_snapshot = (SNAP / "rule_engine_results.csv").exists()
    out["source"] = "演示快照（1.8 MB）" if use_snapshot else "完整数据集"

    if use_snapshot:
        out["decls"] = pd.read_csv(SNAP / "rule_engine_results.csv")
        # ⚠️ 快照里的边表**已经是精选过的**（高权重边 ∪ 单据涉及船舶的边），
        #    这里**绝对不能再做 nlargest** —— 踩过的坑：再筛一次会把刚补进去的
        #    单据边又筛掉，导致选中任何单据都显示"该船不在图中"。
        out["edges"] = pd.read_csv(SNAP / "cooccurrence_edges.csv")
        fp = SNAP / "tracks.csv"
        v = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
        if not v.empty:
            v["datetime_utc"] = pd.to_datetime(v["datetime_utc"], errors="coerce", utc=True)
        out["voy"] = v
        return out

    # ---- 回退：完整数据目录 ----
    fp = RES / f"rule_engine_results_{month}.csv"
    out["decls"] = pd.read_csv(fp) if fp.exists() else pd.DataFrame()

    fp = PROC / f"cooccurrence_edges_{month}.csv"
    out["edges"] = pd.read_csv(fp).nlargest(400, "n_windows") if fp.exists() else pd.DataFrame()

    fp = PROC / f"voyages_{month}.csv"
    if fp.exists():
        v = pd.read_csv(fp, usecols=lambda c: c in
                        ["MMSI", "voyage_id", "datetime_utc", "lat", "lon",
                         "speed_over_ground_knots", "vessel_name", "vessel_class"])
        v["datetime_utc"] = pd.to_datetime(v["datetime_utc"], errors="coerce", utc=True)
        out["voy"] = v
    else:
        out["voy"] = pd.DataFrame()

    return out


# ==========================================================================
# 图：关联方共现网络（★ 主锚点，放在最显眼位置）
# ==========================================================================

# 船型配色（与 src/network_viz.py 保持一致）
CLASS_COLOR = {
    "Passenger": "#3b6ea5", "Tug Tow": "#e08a3c", "Fishing": "#4f9d69",
    "Cargo": "#a54f6e", "Tanker": "#7c5cbf", "Pleasure Craft/Sailing": "#8fb8d8",
    "Pilot Vessel": "#c9a227", "Military": "#6b6b6b",
}
DEFAULT_COLOR = "#b0b0b0"


def fig_network(edges: pd.DataFrame, highlight_mmsi: int | None = None):
    """画共现网络。返回 (图, 选中的船是否在密集核心里)。

    ⚠️ 踩过的坑：不要把**船型字符串数组**直接传给 `marker.color` ——
    plotly 会把它当成颜色值去校验，直接抛 ValueError 把整个页面打崩。
    正确做法是**按船型分层**，每类一个 trace（顺带还能自动生成图例）。
    """
    import networkx as nx

    G = nx.Graph()
    cls_of: dict[str, str] = {}
    # ★ 记录 船名 → MMSI，用于"当前选中单据的船"高亮
    mmsi_of: dict[str, int] = {}
    for _, r in edges.iterrows():
        a, b = str(r["name_a"]), str(r["name_b"])
        G.add_edge(a, b, w=float(r["n_windows"]))
        cls_of.setdefault(a, str(r["class_a"]))
        cls_of.setdefault(b, str(r["class_b"]))
        mmsi_of.setdefault(a, int(r["mmsi_a"]))
        mmsi_of.setdefault(b, int(r["mmsi_b"]))

    # 当前选中单据的船在网络中的节点名（可能不在图里）
    sel_node = None
    if highlight_mmsi is not None:
        for nm, mm in mmsi_of.items():
            if mm == int(highlight_mmsi):
                sel_node = nm
                break

    if G.number_of_nodes() == 0:
        return go.Figure(), False

    # --- 只保留最大连通分量 ---
    # ⚠️ 踩过的坑：不取最大连通分量的话，spring_layout 会把几个孤立小团体
    #    甩到很远的地方，主图被挤到角落、中间一大片空白。
    if G.number_of_nodes() > 0:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        if comps and len(comps[0]) >= 3:
            G = G.subgraph(comps[0]).copy()

    # --- 压掉"链状尾巴" + 限制节点数，否则密集区会糊成一团 ---
    # ⚠️ 踩过的坑（三个，都试过）：
    #   ① 只取 2-core **没用** —— k-core 只剥度为 1 的端点，
    #      而链 A−B−C−D 的内部节点度恰好是 2，整条链会被完整保留 → 图变成一条长虫。
    #   ② 为保住"选中的船"把它强行塞回去，会把链子一起带回来。
    #   ③ 只按度≥3 筛还不够 —— 密集核心里仍有 60+ 个节点，
    #      标签全叠在一起，看起来"挤成一团"。
    #
    # 最终做法：度≥3 筛掉链子 → **再按度数取前 N 个节点**（保证可读性）
    #          → **无论如何保住选中的船**（否则用户会觉得图跟单据无关）。
    # ⚠️ 第四个坑：**度≥3 过滤会把图切断**。
    #    删掉低度节点后，剩下的小碎片会与主体分离，力导向布局
    #    把碎片甩到很远，独立归一化后被碎片撑开、主体挤成一条竖线。
    #    → 过滤完必须**再取一次最大连通分量**。
    MAX_NODES = 34
    try:
        dense = [n for n in G.nodes() if G.degree(n) >= 3]
        if len(dense) >= 5:
            G = G.subgraph(set(dense)).copy()
            comps = sorted(nx.connected_components(G), key=len, reverse=True)
            if comps and len(comps[0]) >= 4:
                G = G.subgraph(comps[0]).copy()
            if G.number_of_nodes() > MAX_NODES:
                top = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:MAX_NODES]
                G = G.subgraph(top).copy()
    except Exception:
        pass

    # ⚠️ 关键设计决策：**不要把选中的船强行塞进图里**。
    #    试过，结果是它被力导向布局推到很远，整张图被拉散、又变回"长虫"。
    #    正确做法是把这件事**变成一个结论**：
    #      · 在密集核心里   → 高亮（说明它嵌入了紧密的共现结构）
    #      · 不在密集核心里 → 明确说明"该船未嵌入密集共现结构"
    #        —— 这本身就是**低风险信号**，比硬画一条线过去更有意义。
    sel_in_core = bool(sel_node and sel_node in G)

    # --- 布局 ---
    # k 控制节点斥力：调大 = 铺得更开。节点少了以后可以给更大的 k，
    # 否则 30 个点也会挤在中间一小块。
    pos = nx.spring_layout(G, seed=42, k=2.2 / max(G.number_of_nodes() ** 0.5, 1),
                           iterations=400)
    # --- 归一化到 [0.05, 0.95]，填满画布 ---
    # ⚠️ 踩过的坑：早先用"等比例缩放 + scaleanchor 锁定纵横比"，
    #    结果图只占了画布中间一小块（因为数据包围盒通常不是正方形，
    #    锁比例后会大量留白；再叠加 range 与 autorange 冲突，直接缩成一团）。
    #    力导向布局本身没有物理量纲，**x/y 各自独立缩放不损失任何信息**，
    #    所以这里直接独立归一化，保证图铺满。
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    sx = (xs.max() - xs.min()) or 1.0
    sy = (ys.max() - ys.min()) or 1.0
    pos = {n: (0.05 + 0.90 * (p[0] - xs.min()) / sx,
               0.05 + 0.90 * (p[1] - ys.min()) / sy) for n, p in pos.items()}

    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.6, color="#cfcfcf"),
                             hoverinfo="skip", showlegend=False, name=""))

    # ⚠️ 不要把**所有**节点都标名字 —— 密集区域会糊成一团看不清。
    #    只标注度最高的几个（节点总数已经压到 34，标 8 个足够），
    #    其余仅显示圆点（hover 仍可看名字）。
    deg = dict(G.degree())
    top_label = set(sorted(deg, key=deg.get, reverse=True)[:8])

    by_cls: dict[str, list[str]] = {}
    for n in G.nodes():
        by_cls.setdefault(cls_of.get(n, "nan"), []).append(n)

    for cls, nodes in sorted(by_cls.items(), key=lambda kv: -len(kv[1])):
        labels = [n[:13] if n in top_label else "" for n in nodes]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers+text",
            text=labels, textposition="top center",
            textfont=dict(size=10, color="#222"),
            marker=dict(size=12, color=CLASS_COLOR.get(cls, DEFAULT_COLOR),
                        line=dict(width=0.8, color="white")),
            hovertemplate="%{text}<extra></extra>",
            name=f"{cls}（{len(nodes)}）",
        ))

    # --- ★ 高亮当前选中单据对应的船舶（仅当它确实在密集核心内）---
    if sel_node and sel_in_core and sel_node in pos:
        # ① 先把它与邻居的边描红加粗
        nb = list(G.neighbors(sel_node))
        ex, ey = [], []
        for v in nb:
            ex += [pos[sel_node][0], pos[v][0], None]
            ey += [pos[sel_node][1], pos[v][1], None]
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(width=2.4, color="#d9480f"),
            hoverinfo="skip", showlegend=False, name="",
        ))
        # ② 再把它自己画大、描红、强制标注
        deg = G.degree(sel_node)
        fig.add_trace(go.Scatter(
            x=[pos[sel_node][0]], y=[pos[sel_node][1]],
            mode="markers+text",
            text=[f"▼ 当前单据：{sel_node[:13]}"],
            textposition="bottom center",
            textfont=dict(size=11, color="#d9480f"),
            marker=dict(size=24, color="#d9480f",
                        line=dict(width=2.5, color="white"), symbol="circle"),
            hovertemplate=f"{sel_node}<br>共现邻居 {deg} 个<extra></extra>",
            name="当前单据的船舶",
        ))
    # ⚠️⚠️ 这里**绝对不能提前 return**！
    #    踩过的坑：把提示文字移到图外时顺手在这加了个 `return fig, sel_in_core`，
    #    结果它下面的 `update_layout` / `update_xaxes` 全成了**死代码** ——
    #    刻度关不掉、图例方向不对、背景色不生效，全都是这一个 return 造成的。
    #    排查了很久才发现：函数里存在两个 return，靠前的那个把后面的全挡了。

    # --- 布局（必须在本函数**最后**执行）---
    fig.update_layout(
        height=660, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.02,
                    xanchor="left", x=0, font=dict(size=10)),
    )
    # 关掉坐标轴：网络图是抽象布局，坐标没有业务含义，
    # 留着 0 / 0.2 / … / 1 一串数字会让人误以为有量纲。
    _HIDE = dict(visible=False, showticklabels=False, showgrid=False,
                 zeroline=False, showline=False, ticks="",
                 range=[0, 1], fixedrange=True)
    fig.update_xaxes(**_HIDE)
    fig.update_yaxes(**_HIDE)
    return fig, sel_in_core


# ==========================================================================
# 图：AIS 轨迹地图
# ==========================================================================

def fig_ego(edges: pd.DataFrame, mmsi: int | None = None) -> go.Figure:
    """画当前单据船舶的「自我网络」（ego network）。

    ⚠️ 为什么单独做这张图：
       上面的全网图只画"密集核心"（度数最高的 34 个节点），而那些节点
       是拖轮、渡轮这类高频共现的船 —— **单据里的普通货船/渔船根本挤不进去**。
       实测：7 艘单据船舶 **0 艘**出现在核心图里，导致"红色高亮"永远不出现。
       强行塞进去又会把图拉成长条（试过）。

       所以拆成两张：**上面是全网概览，下面是以当前船舶为中心的自我网络** ——
       红点永远在正中央，切换单据时一定跟着变。
    """
    fig = go.Figure()
    if mmsi is None or edges.empty:
        fig.update_layout(height=320, title="（无数据）")
        return fig

    mmsi = int(mmsi)
    sub = edges[(edges["mmsi_a"] == mmsi) | (edges["mmsi_b"] == mmsi)].copy()
    if sub.empty:
        fig.update_layout(
            height=300,
            title=dict(text="该船无稳定共现关系 —— 未嵌入任何反复共现的结构（低风险信号）",
                       font=dict(size=12, color="#2e7d32")),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor="white",
        )
        return fig

    # 本船在网络里的名字
    row0 = sub.iloc[0]
    me = str(row0["name_a"]) if int(row0["mmsi_a"]) == mmsi else str(row0["name_b"])

    # 对端（邻居）及共现强度
    nb = []
    for _, r in sub.iterrows():
        if int(r["mmsi_a"]) == mmsi:
            other, cls, w = str(r["name_b"]), str(r["class_b"]), float(r["n_windows"])
        else:
            other, cls, w = str(r["name_a"]), str(r["class_a"]), float(r["n_windows"])
        nb.append((other, cls, w))
    nb.sort(key=lambda x: -x[2])
    nb = nb[:14]                      # 最多画 14 个邻居，保证可读

    # 环形布局：本船在圆心，邻居均匀分布
    n = len(nb)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    px, py = 0.5 + 0.38 * np.cos(ang), 0.5 + 0.38 * np.sin(ang)

    wmax = max(w for _, _, w in nb) or 1.0

    # 边：从圆心到各邻居，粗细表示共现强度
    ex, ey = [], []
    for x, y in zip(px, py):
        ex += [0.5, x, None]
        ey += [0.5, y, None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(width=1.2, color="#d9480f"),
                             hoverinfo="skip", showlegend=False, name=""))

    # 邻居节点（按船型着色）
    by_cls: dict[str, list[int]] = {}
    for i, (_, cls, _) in enumerate(nb):
        by_cls.setdefault(cls, []).append(i)
    for cls, idxs in sorted(by_cls.items(), key=lambda kv: -len(kv[1])):
        fig.add_trace(go.Scatter(
            x=[px[i] for i in idxs], y=[py[i] for i in idxs],
            mode="markers+text",
            text=[nb[i][0][:12] for i in idxs], textposition="top center",
            textfont=dict(size=9, color="#333"),
            marker=dict(size=12, color=CLASS_COLOR.get(cls, DEFAULT_COLOR),
                        line=dict(width=0.8, color="white")),
            hovertemplate="%{text}<extra></extra>",
            name=f"{cls}（{len(idxs)}）",
        ))

    # ★ 本船：永远在圆心，永远高亮
    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5], mode="markers+text",
        text=[f"当前单据：{me[:12]}"], textposition="bottom center",
        textfont=dict(size=11, color="#d9480f"),
        marker=dict(size=26, color="#d9480f",
                    line=dict(width=2.5, color="white")),
        hovertemplate=f"{me}<br>共现邻居 {n} 个<extra></extra>",
        name="当前单据的船舶",
    ))

    fig.update_layout(
        height=460, margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor="white",
        title=dict(text=f"当前单据船舶的共现邻居（共 {len(sub)} 条稳定共现关系）",
                   font=dict(size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.02,
                    xanchor="left", x=0, font=dict(size=9)),
    )
    _HIDE = dict(visible=False, showticklabels=False, showgrid=False,
                 zeroline=False, showline=False, ticks="",
                 range=[0, 1], fixedrange=True)
    fig.update_xaxes(**_HIDE)
    fig.update_yaxes(**_HIDE)
    return fig


def fig_track(voy: pd.DataFrame, mmsi, title: str, use_basemap: bool = False) -> go.Figure:
    """画 AIS 轨迹。

    use_basemap=True  → 夏威夷地图底图（好看，但**依赖 OpenStreetMap 的瓦片 CDN**）
    use_basemap=False → 经纬度坐标轴（**零外部请求，永远能出图**）

    ⚠️ 关于底图的实测经验（别再来回改）：
      · 瓦片来自 OpenStreetMap 的 CDN —— **时好时坏**：同一台机器不同次打开
        都可能有/没有；**Edge 的跟踪防护会直接拦掉**（Chrome 正常）。
      · 曾经出现过"图塌成 400×400"的现象，那是**布局写法**的问题，
        不是地图本身不可用 —— 现在用 `map=dict(...)` + 显式 height 已修复。
      · 底图加载失败时整个绘图区会全白，**连轨迹都看不见** —— 这是最大风险。
      · 所以默认关闭；演示时若网络好就打开，网络差就关掉，**两者都不影响核验结论**。
    """
    fig = go.Figure()
    # 防御：数据未就绪时 voy 可能是空表（连列都没有）
    if voy.empty or "MMSI" not in voy.columns:
        fig.update_layout(height=400, title="暂无轨迹数据")
        return fig

    t = voy[voy["MMSI"] == mmsi].sort_values("datetime_utc")
    if t.empty:
        fig.update_layout(height=400, title="该船本月无轨迹记录")
        return fig

    colorbar = dict(title="航速<br>(节)", thickness=12)
    hover = dict(
        text=[f"{x:%m-%d %H:%M}" for x in t["datetime_utc"]],
        hovertemplate="%{text}<br>%{lon:.4f}, %{lat:.4f}<extra></extra>",
    )

    if use_basemap:
        # ---- 地图底图模式 ----
        fig.add_trace(go.Scattermap(
            lon=t["lon"], lat=t["lat"], mode="lines",
            line=dict(width=1.6, color="#777777"),
            hoverinfo="skip", showlegend=False, name=""))
        fig.add_trace(go.Scattermap(
            lon=t["lon"], lat=t["lat"], mode="markers",
            marker=dict(size=7, color=t["speed_over_ground_knots"],
                        colorscale="YlOrRd", showscale=True, colorbar=colorbar),
            name="", **hover))
        fig.update_layout(
            height=430, margin=dict(l=0, r=0, t=30, b=0),
            title=dict(text=title or "", font=dict(size=12)),
            # ⚠️ 必须显式给 center/zoom —— 否则地图可能缩成默认小尺寸
            map=dict(style="open-street-map",
                     center=dict(lat=float(t["lat"].mean()),
                                 lon=float(t["lon"].mean())),
                     zoom=9),
        )
        return fig

    # ---- 坐标轴模式（默认，零外部依赖）----
    fig.add_trace(go.Scatter(
        x=t["lon"], y=t["lat"], mode="lines",
        line=dict(width=1.4, color="#9a9a9a"),
        hoverinfo="skip", showlegend=False, name=""))
    fig.add_trace(go.Scatter(
        x=t["lon"], y=t["lat"], mode="markers",
        marker=dict(size=6, color=t["speed_over_ground_knots"],
                    colorscale="YlOrRd", showscale=True, colorbar=colorbar),
        name="", **hover))
    fig.update_layout(
        height=430, margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=title or "", font=dict(size=12)),
        xaxis=dict(title="经度", showgrid=True, gridcolor="#eee", zeroline=False),
        yaxis=dict(title="纬度", showgrid=True, gridcolor="#eee", zeroline=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
    )
    return fig


# ==========================================================================

def main() -> None:
    st.title(f"{PRODUCT_NAME} · 国际结算贸易背景核验")
    st.caption(
        f"{PRODUCT_SUB}　|　"
        "⚠️ 演示界面中的**申报单据为构造的验证用例**，非真实银行单据"
    )

    # ---------------- 侧栏 ----------------
    with st.sidebar:
        st.header("申报单据")
        data = load_all(DEFAULT_MONTH)
        decls = data["decls"]
        if decls.empty:
            st.error(f"找不到单据数据。请先跑：\n\n"
                     f"`python src/rule_engine.py --month {DEFAULT_MONTH}`")
            return

        labels = [f"{r.decl_id}　{r.vessel}" for r in decls.itertuples()]
        idx = st.selectbox("选择一张申报单据", range(len(labels)),
                           format_func=lambda i: labels[i])
        d = decls.iloc[idx]
        st.caption(
            "**编号规则**：`D000`…`D009` 是**正常单据**（从真实航次反推）；"
            "带 **F** 的（如 `D000F0`）是**注入欺诈的用例**。\n\n"
            "→ **想看差异，请选带 F 的单据与不带 F 的对比。**"
            "正常单据之间看起来相似是正常的——它们都不触发规则。"
        )

        st.divider()
        st.caption("**产品形态说明**")
        st.info(
            "这不是一个独立系统。\n\n"
            "真实形态是**嵌入银行审单流程的验真微服务 / API**：\n"
            "审单员在现有系统里点一下 → 调用本模块 → "
            "返回一致性评分、异常类型、证据链快照。\n\n"
            "本页面是**演示界面**。"
        )
        show_truth = st.toggle("显示真值标签（演示用）", value=True)
        # ★ 夏威夷地图底图开关
        #   实测经验：瓦片来自 OpenStreetMap 的 CDN，**时好时坏** ——
        #   同一台机器不同次打开都可能有/没有；**Edge 的跟踪防护会直接拦掉**（Chrome 正常）。
        #   底图加载失败时绘图区会全白、连轨迹都看不见，所以**默认关闭**。
        #   演示时网络好就打开（"我的夏威夷"回来了），网络差就关掉 —— 两种都不影响核验结论。
        use_basemap = st.toggle(
            "夏威夷地图底图（需联网）", value=False,
            help="开启后显示 OpenStreetMap 地图底图。若底图空白，说明网络/浏览器"
                 "拦掉了地图瓦片 —— 关掉即可，轨迹照常显示。")
        # ⚠️ 现场保命开关：地图底图需要联网，断网时会全白。
        #    关掉后走"降级模式"（纯经纬度坐标轴），永远能出图。

        # ------------------------------------------------------------
        # 数据来源说明 + 自愿下载完整数据集
        # ★ 设计原则：**默认零下载**。16 GB 不该成为看演示的门槛。
        # ------------------------------------------------------------
        st.divider()
        st.caption("**数据来源**")
        st.caption(
            f"当前使用：**{data.get('source', '—')}**\n\n"
            "HawaiiCoast_GT · Sandia National Laboratories\n\n"
            "Zenodo DOI `10.5281/zenodo.8253611` · CC BY 4.0"
        )

        with st.expander("想复现完整实验？"):
            st.caption(
                "演示只需演示快照（约 1.8 MB，已随代码提交）。\n\n"
                "**完整数据集为 16 GB，仅在做全量复现时才需要下载。**\n\n"
                "下载后会自动校验 md5，并跑通全部实验阶段。"
            )
            st.code("python src/setup_data.py", language="bash")
            st.caption("或在项目根目录执行：`python run_all.py`")
            if st.button("⬇️ 我确认要下载完整数据集（16 GB）",
                         use_container_width=True):
                st.warning(
                    "16 GB 下载在浏览器里会阻塞界面且耗时较长。\n\n"
                    "**建议在终端里执行**，可以看到进度、支持断点续传：\n\n"
                    "```\npython src/setup_data.py\n```"
                )

    # ---------------- 主区 ----------------
    # ★ 主锚点放最上面、最显眼
    st.subheader("① 关联方物理共现网络")
    st.caption(
        "**主锚点**：识别循环贸易团伙时，看这几家企业名下的船是否在物理世界里"
        "反复往同一组港口跑、在同一泊位反复并靠。"
        "这是纯单据/资金流手段查不出来、但物理轨迹能查出来的结构。"
    )
    if not data["edges"].empty:
        _mmsi = d.get("MMSI")
        _mmsi = int(_mmsi) if (_mmsi is not None and not pd.isna(_mmsi)) else None
        _fig, _in_core = fig_network(data["edges"], highlight_mmsi=_mmsi)
        st.plotly_chart(_fig, use_container_width=True, key="net")
        # ⚠️ 提示放在图**外面** —— 放进图里（y=1.03）会被绘图区裁掉一半。
        if _mmsi is not None:
            if _in_core:
                st.success(
                    f"当前单据的船舶（MMSI {_mmsi}）**出现在密集共现核心中**"
                    f"（图中红色高亮）—— 它与其他船舶形成了反复共现的紧密结构，"
                    f"这是循环贸易团伙的结构特征。")
            else:
                st.info(
                    f"当前单据的船舶（MMSI {_mmsi}）**未出现在密集共现核心中**。"
                    "**这本身即为低风险信号** —— 它没有嵌入任何反复共现的紧密结构。")
        st.caption(
            "**全网概览**：度 ≥ 3 的密集共现核心（取度数最高的 34 个节点）。\n\n"
            "⚠️ 图上大部分连接是正常运营模式（同航线客船、港作拖轮），"
            "**朴素共现权重本身不构成风险信号**——详见 `docs/主锚点实验记录.md`。"
        )

        # ---- ★ 当前船舶的自我网络（红点永远在圆心，切换单据必然跟着变）----
        st.markdown("##### 当前船舶的共现邻居")
        st.caption(
            "全网图只画'度数最高的核心节点'，而单据里的普通货船/渔船通常挤不进去"
            "（实测 7 艘单据船舶 0 艘在核心图中）。所以这里单独画出**当前船舶自己**"
            "的共现关系——**红点即当前单据的船**，射线指向与它反复同框过的船，"
            "**线越粗表示共现越多**。"
        )
        st.plotly_chart(fig_ego(data["edges"], _mmsi),
                        use_container_width=True, key="ego")
    else:
        st.warning("缺少共现边数据")

    st.divider()

    # ---------------- 核验结果 + 轨迹 ----------------
    left, right = st.columns([1, 1.6])

    with left:
        st.subheader("② 核验结果")
        risk = str(d.get("risk", "—"))
        color = {"高": "🔴", "中": "🟠", "低": "🟢"}.get(risk, "⚪")
        st.metric("风险等级", f"{color} {risk}")
        st.metric("触发规则数", int(d.get("n_rules", 0)))

        trig = str(d.get("triggered", ""))
        if trig and trig != "nan":
            st.write("**触发的规则**")
            for t in trig.split(";"):
                if t.strip():
                    st.write(f"- `{t.strip()}`")
        else:
            st.success("未触发任何规则")

        ev = str(d.get("evidence", ""))
        if ev and ev != "nan":
            st.write("**证据链**")
            st.caption(ev)

        if show_truth:
            truth = "欺诈" if d.get("truth_fraud") else "正常"
            ft = d.get("fraud_type", "")
            st.divider()
            st.caption(f"真值标签（演示）：**{truth}**"
                       + (f"　注入模式：{ft}" if ft and str(ft) != "nan" else ""))

    with right:
        st.subheader("③ AIS 实际轨迹")
        # 用 .get() 防御：字段缺失时给出提示，而不是把整个页面打崩
        mmsi = d.get("MMSI")
        if mmsi is None or pd.isna(mmsi):
            st.warning("当前结果表缺 MMSI 列 —— 请重跑 "
                       "`python src/rule_engine.py --month 2017_01`")
        else:
            st.caption(f"MMSI {int(mmsi)}　申报 {d.get('declared', '—')}"
                       f"　船舶 {d.get('vessel', '—')}")
            if use_basemap:
                st.caption("🗺️ 已开启夏威夷地图底图（底图空白说明网络访问不到瓦片服务器）")
            st.plotly_chart(fig_track(data["voy"], int(mmsi), "",
                                      use_basemap=use_basemap),
                            use_container_width=True, key="trk")

    st.divider()

    # ---------------- 架构图 ----------------
    st.subheader("④ 方案架构")
    st.code(
        "数据层（用现成的）\n"
        "  AIS 轨迹清洗与航次切分 · 事件真值数据集 · 企业关联关系 · 制裁名单\n"
        "        ↓\n"
        "翻译层 ★（我们的核心）\n"
        "  ① 物理证据抽取：航次 / 靠泊 / 吃水变化 / STS 会遇 / AIS 时间缺口\n"
        "  ② 逐票核验：单据 vs 轨迹（可解释的规则引擎）\n"
        "  ③ 跨票网络：关联关系 × 船舶物理共现 → 循环贸易团伙识别  ★ 主锚点\n"
        "  ④ 输出结构化决策变量：一致性评分 + 异常类型 + 证据链快照 + 置信度\n"
        "        ↓\n"
        "决策层（银行侧）\n"
        "  贷中审单环节的可插拔验真微服务 / API\n"
        "  纵向联邦学习：数据不出域的协作建模（导师方向）",
        language=None,
    )

    st.caption(
        "⚠️ 本页所有『申报单据』均为**构造的验证用例**：正常单据从真实航次反推，"
        "异常单据注入已知欺诈模式。**不是真实银行数据。**"
    )


if __name__ == "__main__":
    main()
