"""
阶段三可视化：共现网络图 + "朴素方法为什么失败"图

产出两张图（roadshow 用）：

  fig4_cooccurrence_network.png
      船舶共现网络全图。节点=船舶，边=共现，按船型着色、按强度定粗细。
      这张图放 PPT 时要配一句话说明：**图上大部分连接是正常运营**，
         不能让人误以为"连着的就是可疑的"。

  fig5_why_naive_fails.png  ★ 这张是 FAQ 的杀手锏
      把"原始权重 vs lift"画成散点图，并标注两类误报的来源。
      一图讲清楚：为什么朴素共现网络不能当欺诈检测器、
      以及我们因此做了什么改进。
      **诚实展示方法论的局限，比藏起来强得多** —— 评委问到时，
      这张图就是答案。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    raise SystemExit("需要 networkx")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for _f in ("Microsoft YaHei", "SimHei", "SimSun"):
    try:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/results"
OUT.mkdir(parents=True, exist_ok=True)

# 船型配色：专业克制，不用花哨色
CLASS_COLOR = {
    "Passenger": "#3b6ea5", "Tug Tow": "#e08a3c", "Fishing": "#4f9d69",
    "Cargo": "#a54f6e", "Tanker": "#7c5cbf", "Pleasure Craft/Sailing": "#8fb8d8",
    "Pilot Vessel": "#c9a227", "Military": "#6b6b6b", "Other": "#b0b0b0",
}
DEFAULT_COLOR = "#c0c0c0"


def _color(cls) -> str:
    return CLASS_COLOR.get(str(cls), DEFAULT_COLOR)


# ==========================================================================
# fig4：共现网络
# ==========================================================================

def fig_network(edges: pd.DataFrame, top_n: int = 400) -> None:
    """画共现网络。只画权重最高的 top_n 条边，否则图会糊成一团。"""
    e = edges.nlargest(top_n, "n_windows")

    G = nx.Graph()
    for _, r in e.iterrows():
        G.add_edge(r["mmsi_a"], r["mmsi_b"],
                   weight=r["n_windows"], moving=r["n_both_moving"],
                   cls_a=r["class_a"], cls_b=r["class_b"])

    print(f"  网络图：{G.number_of_nodes()} 节点 / {G.number_of_edges()} 边"
          f"（取权重最高的 {top_n} 条）")

    # 只保留最大连通分量，避免一地碎点
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if comps and len(comps[0]) > 3:
        G = G.subgraph(comps[0]).copy()
        print(f"    取最大连通分量：{G.number_of_nodes()} 节点")

    pos = nx.spring_layout(G, seed=42, k=1.2 / np.sqrt(max(G.number_of_nodes(), 1)))

    node_cls = {}
    for u, v, d in G.edges(data=True):
        node_cls[u] = d["cls_a"]
        node_cls[v] = d["cls_b"]
    node_colors = [_color(node_cls.get(n)) for n in G.nodes()]

    # 边的"结伴航行"占比决定颜色深浅：越高越可疑
    e_moving = [d["moving"] / max(d["weight"], 1) for _, _, d in G.edges(data=True)]
    e_width = [0.3 + 2.2 * (d["weight"] / e["n_windows"].max()) for _, _, d in G.edges(data=True)]

    fig, ax = plt.subplots(figsize=(13, 10))
    nx.draw_networkx_edges(G, pos, ax=ax, width=e_width,
                           edge_color=e_moving, edge_cmap=plt.cm.YlOrRd,
                           alpha=0.55, edge_vmin=0, edge_vmax=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=55, linewidths=0.4, edgecolors="white")

    # 标注度最高的几个节点
    deg = dict(G.degree(weight="weight"))
    for n in sorted(deg, key=deg.get, reverse=True)[:8]:
        ax.annotate(str(n), pos[n], fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#999",
                              lw=0.4, alpha=0.85))

    # 图例
    present = {c for c in node_cls.values() if pd.notna(c)}
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=8,
                          markerfacecolor=_color(c), markeredgecolor="white", label=str(c))
               for c in sorted(present, key=str)]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9,
              title="船型", title_fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                               norm=plt.Normalize(vmin=0, vmax=0.5))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("「结伴航行」窗口占比\n（双方都在航行时同框的比例）", fontsize=9)

    # ⚠️ 图上文字不能用 markdown 标记（matplotlib 不解析，`**` 会原样印出来），
    #    也不要写 "见 fig5" 这类内部文件名 —— 评委看不懂（踩过）。
    ax.set_title(f"HawaiiCoast_GT 船舶共现网络（2017-01，权重最高的 {top_n} 条边）\n"
                 f"注意：图上大部分连接属正常运营模式（同航线客船、港作拖轮），\n"
                 f"朴素共现权重本身不构成风险信号",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    p = OUT / "fig4_cooccurrence_network.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {p}")


# ==========================================================================
# fig5：为什么朴素方法失败 ★
# ==========================================================================

def fig_why_naive_fails(edges: pd.DataFrame) -> None:
    """散点图：原始权重 vs lift，标注两类系统性误报。"""
    e = edges.dropna(subset=["lift"]).copy()
    e = e[e["lift"] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.6))

    # ---- 左图：原始权重 vs lift ----
    ax1.scatter(e["n_windows"], e["lift"], s=9, alpha=0.35,
                c=[_color(c) for c in e["class_a"]], edgecolors="none")

    # 标注两类误报
    mis1 = e.nlargest(2, "n_windows")
    mis2 = e.nlargest(3, "lift")
    for _, r in mis1.iterrows():
        ax1.annotate(f"{str(r['name_a'])[:14]}\n— {str(r['name_b'])[:14]}",
                     (r["n_windows"], r["lift"]), fontsize=7.5, ha="center",
                     xytext=(0, 14), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.2", fc="#ffe0e0", ec="#c62828", lw=0.6),
                     arrowprops=dict(arrowstyle="->", color="#c62828", lw=0.8))
    for _, r in mis2.iterrows():
        ax1.annotate(f"{str(r['name_a'])[:16]}\n— {str(r['name_b'])[:16]}",
                     (r["n_windows"], r["lift"]), fontsize=7.5, ha="center",
                     xytext=(0, -26), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.2", fc="#e0ecff", ec="#3b6ea5", lw=0.6),
                     arrowprops=dict(arrowstyle="->", color="#3b6ea5", lw=0.8))

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xlabel("原始共现权重（共现窗口数）", fontsize=10)
    ax1.set_ylabel("lift（观测共现 / 随机期望，对数轴）", fontsize=10)
    ax1.set_title("① 两种排序各有一类系统性误报\n"
                  "红框=高权重误报（同航线客船/港作拖轮）　蓝框=高 lift 误报（附属艇/军舰）",
                  fontsize=10, fontweight="bold")
    ax1.grid(alpha=0.2, which="both")

    # ---- 右图：结伴航行 —— 唯一有区分度的维度 ----
    mv = e[e["n_both_moving"] > 0]
    other = e[e["n_both_moving"] == 0]
    ax2.scatter(other["n_windows"], other["moving_ratio"], s=7, alpha=0.25,
                c="#c0c0c0", edgecolors="none", label=f"无共同航行 (n={len(other):,})")
    sc = ax2.scatter(mv["n_windows"], mv["moving_ratio"], s=13, alpha=0.6,
                     c=mv["n_both_moving"], cmap="YlOrRd",
                     edgecolors="none", label=f"有共同航行 (n={len(mv):,})")
    cb = fig.colorbar(sc, ax=ax2, fraction=0.04, pad=0.01)
    cb.set_label("共同航行窗口数", fontsize=9)

    ax2.axhline(0.5, color="#c62828", ls="--", lw=1.2)
    # 放在坐标区内部偏左，避免压到右侧色条
    ax2.text(0.03, 0.62, "「结伴航行」阈值参考线（半数以上时间都在同行）",
             transform=ax2.transAxes, fontsize=8.5, color="#c62828")
    ax2.set_xscale("log")
    ax2.set_xlabel("原始共现权重（共现窗口数，对数轴）", fontsize=10)
    ax2.set_ylabel("结伴航行占比（双方都在航行时同框的比例）", fontsize=10)
    ax2.set_title(f"② 真正有区分度的是「结伴航行」\n"
                  f"正常航运里两艘船不该反复同速同行（{len(mv):,} / {len(e):,} 条边）",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.2, which="both")

    fig.suptitle("为什么朴素共现网络不能当欺诈检测器 —— 以及我们因此改了什么",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "fig5_why_naive_fails.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {p}")


# ==========================================================================

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2017_01")
    args = ap.parse_args()

    fp = ROOT / f"data/processed/cooccurrence_edges_{args.month}.csv"
    if not fp.exists():
        print(f"❌ 找不到 {fp}\n   先跑：src/network_analysis.py --month {args.month}")
        return 1

    edges = pd.read_csv(fp)
    print(f"载入共现边：{len(edges):,} 条")

    fig_network(edges)
    fig_why_naive_fails(edges)
    return 0


if __name__ == "__main__":
    sys.exit(main())
