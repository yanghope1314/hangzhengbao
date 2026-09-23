"""
一键跑完全流程（总纲"零点五"硬性要求）

国赛不要求交源码，但**评审若要求复现，必须能跑通**。
本脚本是那个"一键复现"的入口。

用法
------------------------------------------------------------------
    # 全流程（数据已下载时自动跳过下载校验的重复工作）
    .venv/Scripts/python.exe run_all.py

    # 只跑数据之后的步骤（数据已就绪，跳过解压）
    .venv/Scripts/python.exe run_all.py --skip-data

    # 换月份 / 换年份
    .venv/Scripts/python.exe run_all.py --month 2017_02
    .venv/Scripts/python.exe run_all.py --year 2018

    # 查看会跑哪些步骤，不实际执行
    .venv/Scripts/python.exe run_all.py --dry-run

设计说明
------------------------------------------------------------------
· 每一步都是独立脚本，run_all 只负责按顺序调用、计时、汇总结果
· 任何一步失败会**立即停止并报出是哪一步**，而不是静默继续
  （静默继续会产出一套"看起来跑完了但数据是坏的"结果，比直接失败更危险）
· 已存在的产物默认跳过，除非加 --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
PY = sys.executable


@dataclass
class Step:
    name: str
    cmd: list[str]
    produces: list[Path] = field(default_factory=list)
    note: str = ""


def build_steps(args) -> list[Step]:
    """按依赖顺序列出所有步骤。"""
    month = args.month
    year = args.year

    steps = [
        Step(
            "阶段一 · 数据校验与解压",
            [PY, "src/setup_data.py"],
            [ROOT / "data/raw/HawaiiCoast_GT/AIS_data"],
            "校验 md5 → 探查结构 → 核实吃水字段 → 解压",
        ),
        Step(
            "阶段一 · EDA 出图",
            [PY, "src/eda.py"],
            [ROOT / "experiments/results/fig1_incident_map.png"],
            "产出计划书用的真实事件轨迹图",
        ),
        Step(
            "阶段二 · AIS 基础特征工程",
            [PY, "src/feature_engineering.py", "--months", month],
            [ROOT / f"data/processed/vessel_features_{month}.csv"],
            "运动特征 + 信号交叉印证三分类 + 吃水可用性",
        ),
        Step(
            "阶段三 · 船舶物理共现网络（★主锚点）",
            [PY, "src/network_analysis.py", "--month", month],
            [ROOT / f"data/processed/cooccurrence_edges_{month}.csv"],
            "真实共现网络 + 真值验证 + 社区/闭环检测",
        ),
        Step(
            "阶段三 · 共现网络可视化",
            [PY, "src/network_viz.py", "--month", month],
            [ROOT / "experiments/results/fig5_why_naive_fails.png"],
            "共现网络图 + 「朴素方法为什么失败」图",
        ),
        Step(
            "阶段三 · 全年共现评估实验",
            [PY, "experiments/cooccurrence_eval.py", "--year", year],
            [ROOT / f"experiments/results/cooccurrence_edges_{year}.csv"],
            "跨月累积 + top-K 召回评估（耗时较长，约 20 分钟）",
        ),
        Step(
            "阶段三 · 船型基线评分对比",
            [PY, "experiments/class_baseline.py", "--year", year],
            [ROOT / f"experiments/results/score_comparison_{year}.csv"],
            "五种评分的 top-K 对比 + lift 口径修正",
        ),
        Step(
            "阶段四 · 规则引擎",
            [PY, "src/rule_engine.py", "--month", month],
            [ROOT / f"experiments/results/rule_engine_results_{month}.csv"],
            "三条规则 + 构造申報单据（定性演示）",
        ),
        Step(
            "阶段四 · 异常检测（消融对照）",
            [PY, "src/anomaly_model.py",
             "--pattern", f"vessel_features_{year}_*.csv"],
            [ROOT / "experiments/results/anomaly_ablation.csv"],
            "Isolation Forest + 特征集消融（证明通用航次异常检测无效）",
        ),
        Step(
            "阶段五 · 纵向联邦学习模拟",
            [PY, "src/vertical_fl_sim.py"],
            [ROOT / "experiments/results/vertical_fl_results.csv"],
            "可信协调者版纵向 FL vs 集中式 vs 单方基线",
        ),
        Step(
            "阶段八 · 生成演示快照",
            [PY, "src/make_snapshot.py", "--month", month],
            [ROOT / "demo_data/rule_engine_results.csv"],
            "抽取 Demo 所需的最小数据子集（约 1.8 MB），随代码提交",
        ),
        Step(
            "阶段七 · Streamlit Demo（需手动启动）",
            [], [],
            "运行： python -m streamlit run app/streamlit_app.py",
        ),
    ]
    return steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="一键跑完全流程")
    ap.add_argument("--month", default="2017_01", help="单月处理用，如 2017_01")
    ap.add_argument("--year", default="2017", help="全年实验用")
    ap.add_argument("--skip-data", action="store_true",
                    help="跳过数据校验/解压（数据已就绪时用）")
    ap.add_argument("--skip-eval", action="store_true",
                    help="跳过耗时的全年评估实验")
    ap.add_argument("--force", action="store_true", help="已有产物也重跑")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    args = ap.parse_args(argv)

    steps = build_steps(args)
    if args.skip_data:
        steps = [s for s in steps if "数据校验" not in s.name]
    if args.skip_eval:
        steps = [s for s in steps if "全年共现评估" not in s.name]

    print(f"\n{'=' * 74}")
    print("工e航鉴 · 一键复现全流程")
    print(f"{'=' * 74}")
    print(f"  工作目录：{ROOT}")
    print(f"  Python  ：{PY}")
    print(f"  计划执行 {len(steps)} 个步骤\n")

    results: list[tuple[str, str, float]] = []
    t_all = time.time()

    for i, s in enumerate(steps, 1):
        # --- 待实现的步骤：明确报出，不假装成功 ---
        if not s.cmd:
            print(f"[{i}/{len(steps)}] {s.name}")
            print(f"          ⏭  跳过 —— {s.note}")
            results.append((s.name, "未实现", 0.0))
            continue

        # --- 依赖检查：产物已存在则跳过 ---
        if s.produces and not args.force and all(p.exists() for p in s.produces):
            print(f"[{i}/{len(steps)}] {s.name}")
            print(f"          ⏭  产物已存在，跳过（--force 可强制重跑）")
            results.append((s.name, "已存在", 0.0))
            continue

        print(f"[{i}/{len(steps)}] {s.name}")
        print(f"          $ {' '.join(str(c) for c in s.cmd[1:])}")
        if args.dry_run:
            results.append((s.name, "dry-run", 0.0))
            continue

        t0 = time.time()
        proc = subprocess.run(s.cmd, cwd=ROOT)
        dt = time.time() - t0

        if proc.returncode != 0:
            print(f"\n{'!' * 74}")
            print(f"❌ 步骤失败：「{s.name}」（退出码 {proc.returncode}）")
            print(f"   已用时 {dt:.1f}s。**后续步骤已停止** —— 不静默继续，")
            print(f"   否则会产出一套「看起来跑完了但数据是坏的」结果。")
            print(f"{'!' * 74}\n")
            results.append((s.name, f"失败({proc.returncode})", dt))
            break

        results.append((s.name, "成功", dt))
        print(f"          ✅ 完成（{dt:.1f}s）\n")

    # --- 汇总 ---
    total = time.time() - t_all
    print(f"\n{'=' * 74}")
    print("执行汇总")
    print(f"{'=' * 74}")
    for name, status, dt in results:
        mark = {"成功": "✅", "已存在": "⏭ ", "未实现": "⏭ ", "dry-run": "·"}.get(status, "❌")
        print(f"  {mark}  {name:<42} {status:<12} {dt:>7.1f}s")
    print(f"\n  总耗时：{total:.1f}s")
    n_fail = sum(1 for _, s, _ in results if s.startswith("失败"))
    print(f"  结果：{'❌ 有步骤失败' if n_fail else '✅ 全部通过'}")
    print(f"{'=' * 74}\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
