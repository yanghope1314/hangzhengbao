"""
生成「演示快照」—— 让评委零下载就能看到 Demo

为什么需要这个
------------------------------------------------------------------
原始数据集 16 GB，**任何评委都不会为了看一个演示去下载它**。
但 Demo 实际需要的数据量很小：

    · 22 张申报单据的核验结果        几 KB
    · 9,816 条共现边（演示只用前 400）  约 1 MB
    · 演示涉及的少数几条船的实际轨迹    几 MB
    · 航次特征表                      约 200 KB
    ------------------------------------------------
    合计：**几十 MB**，可以随代码一起提交

因此本脚本把 Demo 需要的部分**预计算并抽取出来**，落到 `demo_data/`：

    demo_data/
      ├── README.txt                说明这是什么、从哪来、怎么重建
      ├── rule_engine_results.csv   单据核验结果
      ├── cooccurrence_edges.csv    共现网络（Demo 用前 400 条）
      ├── vessel_features.csv       航次特征
      └── tracks.parquet            演示所需船舶的轨迹（只含需要的 MMSI）

Demo 默认读这个快照 → **评委双击就能看，不需要下载 16 GB**。
想看完整复现的人，界面上有一个按钮，点了才去下载全量数据。

用法
------------------------------------------------------------------
    .venv/Scripts/python.exe src/make_snapshot.py --month 2017_01
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ⚠️ 文件相对路径：本脚本要在别人电脑上也能跑
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
RES = ROOT / "experiments/results"
SNAP = ROOT / "demo_data"

# 演示涉及的船舶，只保留这几条船的轨迹 —— 这是快照体积的大头，必须裁剪
TRACK_COLS = ["MMSI", "voyage_id", "datetime_utc", "lat", "lon",
              "speed_over_ground_knots", "vessel_name", "vessel_class"]


def build(month: str, track_pad_hours: int = 12) -> int:
    SNAP.mkdir(exist_ok=True)
    print(f"\n{'=' * 74}")
    print(f"生成演示快照：{month}")
    print(f"{'=' * 74}")

    # ---- 1. 单据核验结果 ----
    f = RES / f"rule_engine_results_{month}.csv"
    if not f.exists():
        print(f"❌ 找不到 {f}\n   先跑：src/rule_engine.py --month {month}")
        return 1
    decls = pd.read_csv(f)
    decls.to_csv(SNAP / "rule_engine_results.csv", index=False, encoding="utf-8-sig")
    print(f"  ① 单据核验结果   {len(decls):>5} 行   {_size(SNAP / 'rule_engine_results.csv')}")

    # ---- 2. 共现网络 ----
    # ⚠️ 关键：不能只取"权重最高的 N 条"！
    #    Demo 里选中一张单据时，要在网络上**高亮这艘船**；如果这艘船恰好
    #    不在高权重子图里，界面就会显示"该船不在图中"——第一印象很差。
    #    所以边集 = 高权重边 ∪ **所有单据涉及船舶的边**。
    f = PROC / f"cooccurrence_edges_{month}.csv"
    if f.exists():
        e = pd.read_csv(f)
        decl_mmsi = set(decls["MMSI"].dropna().astype(int))
        top = e.nlargest(500, "n_windows")
        involving = e[e["mmsi_a"].isin(decl_mmsi) | e["mmsi_b"].isin(decl_mmsi)]
        merged = (pd.concat([top, involving])
                    .drop_duplicates(subset=["mmsi_a", "mmsi_b"])
                    .sort_values("n_windows", ascending=False))
        merged.to_csv(SNAP / "cooccurrence_edges.csv",
                      index=False, encoding="utf-8-sig")
        hit = merged[merged["mmsi_a"].isin(decl_mmsi)
                     | merged["mmsi_b"].isin(decl_mmsi)]["mmsi_a"].nunique()
        print(f"  ② 共现网络       {len(merged):>5} 条   "
              f"{_size(SNAP / 'cooccurrence_edges.csv')}")
        print(f"      （高权重 500 条 ∪ 单据涉及船舶的边；"
              f"覆盖 {len(decl_mmsi)} 艘单据船舶中的 "
              f"{len(decl_mmsi & set(merged['mmsi_a']) | decl_mmsi & set(merged['mmsi_b']))} 艘）")
    else:
        print("  ② 共现网络       ⚠️ 缺，跳过")

    # ---- 3. 航次特征 ----
    f = PROC / f"vessel_features_{month}.csv"
    if f.exists():
        shutil.copy2(f, SNAP / "vessel_features.csv")
        print(f"  ③ 航次特征       {pd.read_csv(f).shape[0]:>5} 行   "
              f"{_size(SNAP / 'vessel_features.csv')}")

    # ---- 4. ★ 只截取演示需要的船舶轨迹（体积大头）----
    f = PROC / f"voyages_{month}.csv"
    if f.exists():
        need = set(decls["MMSI"].dropna().astype(int))
        print(f"  ④ 轨迹裁剪：目标 {len(need)} 艘船，从 "
              f"{_size(f)} 的全量文件中筛选…")
        # 分块读，只留需要的 MMSI —— 避免把 1.9M 行全灌进内存
        keep = []
        for chunk in pd.read_csv(f, usecols=lambda c: c in TRACK_COLS,
                                 chunksize=500_000, low_memory=False):
            keep.append(chunk[chunk["MMSI"].isin(need)])
        tr = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
        tr["datetime_utc"] = pd.to_datetime(tr["datetime_utc"], errors="coerce", utc=True)
        out = SNAP / "tracks.csv"
        tr.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"      → {len(tr):,} 行 / {tr['MMSI'].nunique()} 艘船   {_size(out)}")

    # ---- 5. 说明文件 ----
    total = sum(f.stat().st_size for f in SNAP.iterdir() if f.is_file())
    (SNAP / "README.txt").write_text(
        f"""演示快照说明
================================================================================
本目录是「工e航鉴」演示界面所需的**最小数据子集**，可随代码一起提交。

为什么需要它
--------------------------------------------------------------------------------
完整数据集 HawaiiCoast_GT 为 16 GB，不适合随作品提交，也不应要求评委下载。
演示界面实际只需要以下预计算结果：

  rule_engine_results.csv   申报单据核验结果（构造的验证用例）
  cooccurrence_edges.csv    船舶物理共现网络（演示取权重最高的 500 条）
  vessel_features.csv       航次级特征表
  tracks.csv                演示涉及船舶的 AIS 轨迹（已按期裁剪）

本目录合计约 {total / 1024**2:.1f} MB。

数据来源
--------------------------------------------------------------------------------
HawaiiCoast_GT — Sandia National Laboratories
  Zenodo DOI : 10.5281/zenodo.8253611
  许可证     : CC BY 4.0
  原始来源   : MarineCadastre (NOAA / BOEM)
  规模       : 88,749,176 个 AIS 点 / 2,622 艘船 / 154 起真实标注事件
  时间与区域 : 2017-2020 年，夏威夷沿海

⚠️ 本快照中的「申报单据」为项目自建的**构造验证用例**
   （正常单据由真实航次反推、异常单据注入已知欺诈模式），
   不是真实银行单据。

如何重建完整数据
--------------------------------------------------------------------------------
在项目根目录执行：

    python src/setup_data.py              # 自动下载 16 GB 数据集并校验 md5

完整流程复现：

    python run_all.py                     # 一键跑完全部阶段

生成日期：由 src/make_snapshot.py 自动生成
================================================================================
""", encoding="utf-8")

    print(f"\n{'=' * 74}")
    print(f"✅ 快照已生成：{SNAP}")
    print(f"   总计 {total / 1024**2:.1f} MB —— 可随代码提交，评委零下载即可运行 Demo")
    print(f"{'=' * 74}\n")
    return 0


def _size(p: Path) -> str:
    if not p.exists():
        return "—"
    n = p.stat().st_size
    return f"{n / 1024**2:.2f} MB" if n > 1024**2 else f"{n / 1024:.1f} KB"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="生成演示快照")
    ap.add_argument("--month", default="2017_01")
    sys.exit(build(ap.parse_args().month))
