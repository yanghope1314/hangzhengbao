"""
数据加载与 schema 核实。

设计说明（为什么这个模块的第一职责是"查字段"而不是"读数据"）
------------------------------------------------------------------
阶段一的核心任务**不是**把数据读进来跑个模型，而是回答一个决定方案的问题：

    数据里到底有没有 Draft（吃水）字段？

早期设想用"船舶吃水在申报交易期间无变化 = 根本没装货"识别空转贸易
（MAS 2025 文件把"通过吃水判断货物是否已装卸"列为 good practice）。

⚠️ **该设想已由本项目实测证伪**：吃水字段虽然存在，但实测 **98.3% 的船舶
整月不更新**（船员人工录入的静态值），"无变化"是默认状态而非异常信号。

保留本检查的用途：确认字段可用性，并**记录该字段被排除的原因**，
避免后续重复走这条弯路。见 docs/计划书_定稿.md 第 4.3 节。

用法：
    python src/data_loader.py --inspect data/raw/HawaiiCoast_GT.zip
    python src/data_loader.py --inspect            # 自动找 data/raw 下的 zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Windows 控制台默认 GBK，输出 emoji 会抛 UnicodeEncodeError。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 吃水字段的匹配关键词。⚠️ 必须用**子串**匹配，不能用精确等值——
# 实际踩过的坑：HawaiiCoast_GT 的列名是 `draft_depth_m`，
# 用精确匹配 ("draft", "draught", "draft_m") 会全部落空，误报"没有吃水字段"。
# 各数据源命名差异极大（Draft / draft / draught / draft_depth_m / Draft_m …），
# 只有子串匹配才稳。
DRAFT_KEYWORDS: tuple[str, ...] = ("draft", "draught")

# 我们需要关注的 AIS「概念」及各自的可能列名（子串匹配）。
# ⚠️ 不要用 MarineCadastre 的原始列名（SOG/LAT/COG…）去核对——
# 不同数据集会重命名，例如 HawaiiCoast_GT 用的是
# speed_over_ground_knots / lat / course_over_ground_deg，直接比对会全部误报"缺失"。
FIELD_CONCEPTS: dict[str, tuple[str, ...]] = {
    "船舶标识": ("mmsi",),
    "时间": ("datetime", "basedatetime", "timestamp"),
    "纬度": ("lat",),
    "经度": ("lon",),
    "对地航速": ("speed_over_ground", "sog"),
    "对地航向": ("course_over_ground", "cog"),
    "船首向": ("heading",),
    "船名": ("vessel_name", "shipname", "name"),
    "IMO": ("imo",),
    "呼号": ("call_sign", "callsign"),
    "船型": ("vessel_type", "ship_type"),
    "状态": ("status", "navstat"),
    "船长": ("length",),
    "船宽": ("width", "beam"),
    "吃水": ("draft", "draught"),
    "货物类型": ("cargo",),
    "应答器类别": ("transceiver",),
}


# --------------------------------------------------------------------------
# 1. 压缩包结构探查（不解压，秒级返回）
# --------------------------------------------------------------------------

def inspect_zip(zip_path: str | Path) -> list[dict]:
    """列出一个 zip 里有什么文件，不解压。

    为什么先做这一步：3GB 的包解压出来可能是几十个 CSV，我们先要知道
    文件怎么组织的（按月？按年？还是一个总表），才能决定怎么读。

    Args:
        zip_path: zip 文件路径

    Returns:
        每个成员一个 dict：{name, size_mb, is_dir}
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"找不到文件：{zip_path}")

    members: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            members.append({
                "name": info.filename,
                "size_mb": round(info.file_size / 1024 / 1024, 2),
                "is_dir": info.is_dir(),
            })
    return members


def print_zip_contents(zip_path: str | Path, max_show: int = 60) -> None:
    """把 zip 内容打印成人能看的样子。"""
    members = [m for m in inspect_zip(zip_path) if not m["is_dir"]]
    total_mb = sum(m["size_mb"] for m in members)

    print(f"\n{'=' * 72}")
    print(f"压缩包：{Path(zip_path).name}")
    print(f"文件数：{len(members)}    解压后总计：{total_mb:,.1f} MB")
    print(f"{'=' * 72}")

    for m in members[:max_show]:
        print(f"  {m['size_mb']:>10,.2f} MB   {m['name']}")
    if len(members) > max_show:
        print(f"  ... 另有 {len(members) - max_show} 个文件未显示")
    print()


# --------------------------------------------------------------------------
# 2. ★ 核心：从压缩包里直接读表头，核实字段
# --------------------------------------------------------------------------

def peek_columns_in_zip(
    zip_path: str | Path,
    member: Optional[str] = None,
    nrows: int = 5,
) -> pd.DataFrame:
    """从 zip 里**不解压**直接读某个 CSV 的前几行，用来核实真实列名。

    这是本模块最重要的函数。它让我们不必解压 3GB 就能知道字段清单。

    Args:
        zip_path: zip 路径
        member:   zip 内的 CSV 文件名；None 则自动选第一个 .csv
        nrows:    读多少行做样本

    Returns:
        样本 DataFrame

    Raises:
        ValueError: 压缩包里没有 CSV
    """
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [
            n for n in zf.namelist()
            if n.lower().endswith(".csv") and not n.endswith("/")
        ]
        if not csv_names:
            raise ValueError(f"{zip_path.name} 里没有找到 CSV 文件")

        target = member or csv_names[0]
        print(f"读取样本：{target}  （共 {len(csv_names)} 个 CSV）")

        with zf.open(target) as fh:
            df = pd.read_csv(fh, nrows=nrows, low_memory=False)

    return df


def find_draft_columns(columns: Iterable[str]) -> list[str]:
    """在列名里找吃水字段。

    用**子串**匹配（大小写不敏感），覆盖 draft / draught / draft_depth_m 等各种命名。
    不要改回精确等值匹配——那样会漏掉 `draft_depth_m` 这类带后缀的列名。
    """
    hits: list[str] = []
    for c in columns:
        cl = str(c).lower()
        if any(kw in cl for kw in DRAFT_KEYWORDS):
            hits.append(str(c))
    return hits


def report_schema(df: pd.DataFrame, label: str = "") -> dict:
    """报告一份 AIS 数据的字段情况——这是写进计划书"数据与合规说明"的素材。

    Returns:
        dict，含 has_draft / draft_columns / missing_expected / columns
    """
    cols = [str(c) for c in df.columns]
    lowered = [c.lower() for c in cols]
    draft_cols = find_draft_columns(cols)

    # 概念级核对：每个概念只要有一列的子串命中就算"存在"
    concept_hits: dict[str, list[str]] = {}
    for concept, aliases in FIELD_CONCEPTS.items():
        hit = [c for c, cl in zip(cols, lowered) if any(a in cl for a in aliases)]
        concept_hits[concept] = hit
    missing = [k for k, v in concept_hits.items() if not v]

    print(f"\n{'=' * 72}")
    print(f"Schema 核实报告{'：' + label if label else ''}")
    print(f"{'=' * 72}")
    print(f"列数：{len(cols)}")
    print(f"\n全部列名：\n  {', '.join(cols)}\n")

    # 吃水字段可用性（该识别思路已被实测证伪，见计划书 4.3 节）
    if draft_cols:
        print(f"  ✅ 吃水字段：存在 → {draft_cols}")
        print("     ⚠️  但字段存在 ≠ 可用：实测 98.3% 的船舶整月不更新，")
        print("        「吃水无变化」是默认状态。该识别思路已证伪并降级，")
        print("        转为一条独立发现：任何单一 AIS 字段都不能作为真值使用。")
    else:
        print("  ❌ 吃水字段：**不存在**")
        print("     → 「吃水判断空转贸易」连数据基础都不具备，直接排除。")

    if missing:
        print(f"\n  未出现的常见AIS字段：{', '.join(missing)}")
    print(f"{'=' * 72}\n")

    return {
        "has_draft": bool(draft_cols),
        "draft_columns": draft_cols,
        "missing_expected": missing,
        "columns": cols,
    }


# --------------------------------------------------------------------------
# 3. 数据加载（schema 确认后再用）
# --------------------------------------------------------------------------

def load_ais_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """读 AIS CSV 并做基础清洗。

    清洗规则（都是踩过的坑）：
      1. 经纬度超出合理范围 → 丢弃（0,0 是典型的"设备没定位到"）
      2. 同一 MMSI 同一时间戳的重复记录 → 保留第一条
      3. 按 MMSI + 时间排序（后续所有按船分组的操作都依赖这个顺序）

    Args:
        path:   CSV 路径
        kwargs: 透传给 pd.read_csv

    Returns:
        清洗后的 DataFrame
    """
    df = pd.read_csv(path, low_memory=False, **kwargs)

    # --- 列名标准化：统一大写，方便后续处理 ---
    df.columns = [str(c).strip() for c in df.columns]

    # --- 时间列解析 ---
    time_col = next((c for c in df.columns if c.lower() in ("basedatetime", "timestamp", "time")), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # --- 经纬度合法性 ---
    lat_col = next((c for c in df.columns if c.upper() == "LAT"), None)
    lon_col = next((c for c in df.columns if c.upper() == "LON"), None)
    if lat_col and lon_col:
        before = len(df)
        df = df[
            df[lat_col].between(-90, 90) & df[lon_col].between(-180, 180)
            & ~((df[lat_col] == 0) & (df[lon_col] == 0))
        ]
        dropped = before - len(df)
        if dropped:
            print(f"  清洗：丢弃 {dropped:,} 条非法经纬度记录")

    # --- 去重 + 排序 ---
    mmsi_col = next((c for c in df.columns if c.upper() == "MMSI"), None)
    if mmsi_col and time_col:
        before = len(df)
        df = df.drop_duplicates(subset=[mmsi_col, time_col], keep="first")
        if before - len(df):
            print(f"  清洗：丢弃 {before - len(df):,} 条重复记录")
        df = df.sort_values([mmsi_col, time_col]).reset_index(drop=True)

    print(f"  载入完成：{len(df):,} 行 × {len(df.columns)} 列")
    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _auto_find_zip() -> Path:
    """在 data/raw 下自动找一个 zip。

    ⚠️ 用**文件相对路径**（基于本文件位置推算项目根），不要写成
    `Path("data/raw")` —— 那是相对**当前工作目录**的，换个目录运行就会
    找不到文件。本项目要打包提交、在别人电脑上复现，路径必须与 CWD 无关。
    """
    root = Path(__file__).resolve().parent.parent
    candidates = sorted((root / "data/raw").glob("*.zip"))
    if not candidates:
        raise FileNotFoundError(f"{root / 'data/raw'} 下没有 zip 文件")
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="数据加载与 schema 核实")
    parser.add_argument("--inspect", action="store_true",
                        help="探查压缩包结构并核实字段（含吃水字段）")
    parser.add_argument("--zip", type=str, default=None, help="指定 zip 路径")
    parser.add_argument("--member", type=str, default=None, help="指定 zip 内的 CSV")
    args = parser.parse_args(argv)

    zip_path = Path(args.zip) if args.zip else _auto_find_zip()

    if args.inspect:
        print_zip_contents(zip_path)
        sample = peek_columns_in_zip(zip_path, member=args.member)
        report_schema(sample, label=zip_path.name)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
