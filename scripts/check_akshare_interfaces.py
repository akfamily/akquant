"""检查示例、脚本与文档里调用的 AKShare 接口在已安装的 AKShare 中是否还存在.

AKShare 会给接口改名并直接删除旧名(1.20.45、1.20.64 各改过一批)。示例取数失败时
会回退到合成数据而不报错, 所以改名不会让示例或测试失败; 这个脚本专门盯接口名,
由 ``.github/workflows/akshare-compat.yml`` 每天用最新版 AKShare 跑一遍。

只检查接口名是否存在, 不联网取数。本地运行::

    pip install --upgrade akshare
    python scripts/check_akshare_interfaces.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
#: 扫描范围: 示例、脚本、用户文档与 README(设计文档与计划是历史记录, 不扫)
SCAN_GLOBS = (
    "README.md",
    "examples/**/*.py",
    "scripts/*.py",
    "docs/zh/**/*.md",
    "docs/en/**/*.md",
    "python/akquant/**/*.py",
)
_IMPORT = re.compile(r"^\s*import akshare as ak\b", re.MULTILINE)
_CALL = re.compile(r"\bak\.([a-z][a-z0-9_]*)\(")

Usage = Tuple[str, int]


def find_interfaces(root: Path = ROOT) -> Dict[str, List[Usage]]:
    """返回 {接口名: [(相对路径, 行号), ...]}, 只看写了 import akshare as ak 的文件."""
    found: Dict[str, List[Usage]] = {}
    for pattern in SCAN_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            if not _IMPORT.search(text):
                continue
            rel = path.relative_to(root).as_posix()
            for lineno, line in enumerate(text.splitlines(), start=1):
                for name in _CALL.findall(line):
                    found.setdefault(name, []).append((rel, lineno))
    return found


def missing_interfaces(names: Iterable[str], module: object) -> List[str]:
    """返回 module 里不存在(或不可调用)的接口名."""
    return sorted(n for n in names if not callable(getattr(module, n, None)))


def main() -> int:
    """扫描并核对, 有接口不存在时返回 1."""
    try:
        import akshare as ak
    except ImportError:
        print("未安装 AKShare: pip install --upgrade akshare", file=sys.stderr)
        return 2

    found = find_interfaces()
    version = getattr(ak, "__version__", "未知")
    missing = missing_interfaces(found, ak)
    print(f"AKShare {version}: 共用到 {len(found)} 个接口, {len(missing)} 个不存在")
    for name in missing:
        print(f"\n{name} 在 AKShare {version} 中不存在, 用到它的位置:")
        for rel, lineno in found[name]:
            print(f"  {rel}:{lineno}")
    if missing:
        print(
            "\n接口很可能已改名: 到 https://akshare.akfamily.xyz/changelog.html "
            "的“接口更名一览表”查新名字, 然后改掉上面这些位置。"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
