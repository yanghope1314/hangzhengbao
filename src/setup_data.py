"""
阶段一：数据落地的完整链路 —— 校验 → 探查 → 核实字段 → 解压

下载完 HawaiiCoast_GT.zip 之后，跑这一条命令即可走完阶段一的前半程：

    .venv/Scripts/python.exe src/setup_data.py

它会依次做四件事：
    1. 校验 md5（对比 Zenodo 公布的 e7dd0951d1512357af7ca50cad710ebe）
    2. 列出压缩包内容（不解压，看清楚文件怎么组织的）
    3. ★ 从压缩包里直接读表头，核实有没有 Draft（吃水）字段
    4. 按需解压到 data/raw/HawaiiCoast_GT/

第 3 步核实吃水字段是否存在。

⚠️ 字段"存在"不等于"可用"：本项目实测该字段 **98.3% 的船舶整月不更新**
（船员人工录入的静态值），"吃水无变化"是默认状态而非异常信号。
「吃水判断空转贸易」这条思路已由实测证伪并降级，
转为一条独立发现：任何单一 AIS 字段都不能作为真值使用。
详见 docs/计划书_定稿.md 第 4.3 节 / 第 7.2 节。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

# Windows 控制台默认 GBK，输出 emoji/中文进度会抛 UnicodeEncodeError。
# 强制 stdout 走 UTF-8，errors="replace" 兜底避免再次崩掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 把 src 加入路径，方便直接 import 同目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (  # noqa: E402
    inspect_zip,
    peek_columns_in_zip,
    print_zip_contents,
    report_schema,
)

# Zenodo 公布的校验值（核实来源：https://zenodo.org/records/8253611）
EXPECTED_MD5 = "e7dd0951d1512357af7ca50cad710ebe"

# 数据集直链。⚠️ 16 GB 数据集不可能随作品提交，必须让使用者能自行下载。
ZENODO_URL = ("https://zenodo.org/records/8253611/files/"
              "HawaiiCoast_GT.zip?download=1")

# ⚠️ 全部用**文件相对路径**：基于本文件位置推算项目根。
#    写成 `Path("data/raw/...")` 是相对当前工作目录的，
#    本项目要打包提交、在别人电脑上复现，必须与 CWD 无关。
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ZIP = ROOT / "data/raw/HawaiiCoast_GT.zip"
EXTRACT_DIR = ROOT / "data/raw/HawaiiCoast_GT"


def md5_of(path: Path, chunk_mb: int = 32) -> str:
    """分块算 md5（3GB 文件不要一次读进内存）。"""
    h = hashlib.md5()
    chunk = chunk_mb * 1024 * 1024
    total = path.stat().st_size
    read = 0
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
            read += len(block)
            pct = read / total * 100
            print(f"\r  校验中… {pct:5.1f}%  ({read / 1024**3:.2f} / {total / 1024**3:.2f} GB)",
                  end="", flush=True)
    print()
    return h.hexdigest()


def download(zip_path: Path, url: str = ZENODO_URL) -> bool:
    """从 Zenodo 下载数据集（3.0 GB）。

    为什么必须内置这一步
    ------------------------------------------------------------------
    **16 GB 的数据集体积远超作品提交限制，不可能随代码一起上传。**
    因此项目必须能让任何人在任何一台电脑上自行把数据拉下来。
    这也是"可复现"的硬性要求。

    实现说明：
      · 用标准库 urllib，**不依赖 curl / wget**（Windows 自带 curl 但版本不一）
      · 支持**断点续传**：若目标文件已存在且小于远端，从断点继续
      · 下载后会自动走 md5 校验，坏了能立刻发现
    """
    import urllib.request

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 查询远端大小，用于续传判断 ---
    remote_size = None
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            remote_size = int(resp.headers.get("Content-Length", 0)) or None
    except Exception as e:
        print(f"  （无法获取远端大小：{e}，将直接开始下载）")

    done = zip_path.stat().st_size if zip_path.exists() else 0

    if remote_size and done >= remote_size:
        print(f"  文件已完整（{done / 1024**3:.2f} GB），跳过下载")
        return True

    headers = {}
    mode = "wb"
    if done > 0 and remote_size and done < remote_size:
        headers["Range"] = f"bytes={done}-"
        mode = "ab"
        print(f"  检测到未完成下载（已 {done / 1024**3:.2f} GB），从断点续传…")
    else:
        print(f"  开始下载数据集（约 3.0 GB）…")
        done = 0

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, zip_path.open(mode) as fh:
            total = remote_size or int(resp.headers.get("Content-Length", 0)) or 0
            got = done
            while True:
                chunk = resp.read(1024 * 1024)      # 1 MB/次
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if total:
                    pct = got / total * 100
                    print(f"\r    下载中… {pct:5.1f}%  "
                          f"({got / 1024**3:.2f} / {total / 1024**3:.2f} GB)",
                          end="", flush=True)
        print()
        print(f"  ✅ 下载完成：{zip_path.stat().st_size / 1024**3:.2f} GB")
        return True
    except Exception as e:
        print(f"\n  ❌ 下载失败：{e}")
        print(f"     可手动下载后放到：{zip_path}")
        print(f"     地址：{url}")
        return False


def verify_md5(zip_path: Path, expected: str = EXPECTED_MD5) -> bool:
    """校验下载完整性。不通过就不要往下走——数据坏了后面全白做。"""
    print(f"\n{'=' * 72}")
    print("步骤 1/4  校验下载完整性")
    print(f"{'=' * 72}")
    print(f"文件：{zip_path}  ({zip_path.stat().st_size / 1024**3:.2f} GB)")

    actual = md5_of(zip_path)
    ok = actual.lower() == expected.lower()

    print(f"  期望 md5：{expected}")
    print(f"  实际 md5：{actual}")
    print(f"  {'✅ 通过' if ok else '❌ 不匹配 —— 文件可能损坏或不完整，建议重新下载'}")

    return ok


def extract(zip_path: Path, out_dir: Path) -> None:
    """解压。已经解压过就跳过（幂等）。"""
    print(f"\n{'=' * 72}")
    print("步骤 4/4  解压")
    print(f"{'=' * 72}")

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  {out_dir} 已存在且非空，跳过解压。")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        print(f"  解压 {len(members)} 个文件到 {out_dir} …")
        for i, m in enumerate(members, 1):
            zf.extract(m, out_dir)
            print(f"\r  [{i}/{len(members)}] {m.filename[:60]:<60}", end="", flush=True)
    print("\n  ✅ 解压完成")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段一数据落地链路")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out", type=Path, default=EXTRACT_DIR)
    parser.add_argument("--skip-md5", action="store_true", help="跳过校验（调试用）")
    parser.add_argument("--no-extract", action="store_true", help="只核实不解压")
    parser.add_argument("--download", action="store_true",
                        help="强制重新下载数据集")
    args = parser.parse_args(argv)

    # --- 数据缺失时自动下载 ---
    # ⚠️ 这一步是"可复现"的关键：16 GB 数据集不可能随作品上传，
    #    别人拿到代码后跑这一条命令就能自行拉取。
    if args.download or not args.zip.exists():
        print(f"\n{'=' * 72}")
        print("步骤 0/4  获取数据集")
        print(f"{'=' * 72}")
        if not download(args.zip):
            return 1

    # --- 1. 校验 ---
    if not args.skip_md5:
        if not verify_md5(args.zip):
            print("\n⚠️  校验不通过，已停止。请重新下载后再跑。")
            return 2
    else:
        print("\n（已跳过 md5 校验）")

    # --- 2. 结构 ---
    print(f"\n{'=' * 72}")
    print("步骤 2/4  压缩包结构")
    print(f"{'=' * 72}")
    print_zip_contents(args.zip)

    # --- 3. ★ 字段核实 ---
    print(f"{'=' * 72}")
    print("步骤 3/4  ★ 字段核实（决定「吃水判断空转贸易」能否成立）")
    print(f"{'=' * 72}")

    sample = peek_columns_in_zip(args.zip)
    result = report_schema(sample, label=args.zip.name)

    # 落盘，供后续引用（计划书"数据与合规说明"章节要用）
    import json
    out_json = ROOT / "docs/schema_hawaii_coast_gt.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "source": str(args.zip),
                "md5_expected": EXPECTED_MD5,
                **{k: v for k, v in result.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"字段核实结果已写入：{out_json}")

    # --- 4. 解压 ---
    if args.no_extract:
        print("\n（--no-extract：跳过解压）")
    else:
        extract(args.zip, args.out)

    # --- 收尾结论 ---
    print(f"\n{'=' * 72}")
    if result["has_draft"]:
        print("✅ 结论：吃水字段【存在】")
        print("   ⚠️  但字段存在 ≠ 可用：实测 98.3% 的船舶整月不更新该字段")
        print("       （船员人工录入的静态值），「吃水无变化」是默认状态而非异常信号。")
        print("   → 该识别思路已由本项目实测证伪并降级，转为一条独立发现：")
        print("     任何单一 AIS 字段都不能作为真值使用，必须多源交叉印证。")
        print("   → 见 docs/计划书_定稿.md 第 4.3 节 / 第 7.2 节。")
    else:
        print("⚠️  结论：吃水字段【不存在】")
        print("   → 「吃水判断空转贸易」连数据基础都不具备，直接排除。")
    print(f"{'=' * 72}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
