"""scripts/check_akshare_interfaces.py: 接口名扫描与缺失判定(不联网)."""

import importlib.util
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> Any:
    path = ROOT / "scripts" / "check_akshare_interfaces.py"
    spec = importlib.util.spec_from_file_location("check_akshare_interfaces", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scan_finds_calls_only_in_files_importing_akshare(tmp_path: Path) -> None:
    """只统计写了 import akshare as ak 的文件, 并记下位置."""
    script = _load_script()
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "a.py").write_text(
        "import akshare as ak\n\ndf = ak.stock_zh_a_hist_sina(symbol='sh600000')\n",
        encoding="utf-8",
    )
    # 没有导入 akshare: ak 是别的东西(如 backtest/merge.py 里的局部变量)
    (tmp_path / "examples" / "b.py").write_text(
        "ak = object()\nak.where(1)\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "```python\nimport akshare as ak\n"
        "ak.fund_etf_hist_sina(symbol='sh510300')\n```\n",
        encoding="utf-8",
    )
    found = script.find_interfaces(tmp_path)
    assert found == {
        "stock_zh_a_hist_sina": [("examples/a.py", 3)],
        "fund_etf_hist_sina": [("README.md", 3)],
    }


def test_missing_interfaces_reports_removed_names() -> None:
    """模块里没有的接口名被报出来, 存在的不报."""
    script = _load_script()
    fake = types.ModuleType("akshare")
    fake.stock_zh_a_hist_sina = lambda **kwargs: None  # type: ignore[attr-defined]
    fake.not_callable = 1  # type: ignore[attr-defined]
    names = ["stock_zh_a_hist_sina", "stock_zh_a_daily", "not_callable"]
    assert script.missing_interfaces(names, fake) == [
        "not_callable",
        "stock_zh_a_daily",
    ]


def test_repository_uses_at_least_one_interface() -> None:
    """扫描范围没有写错: 仓库里确实能扫到在用的接口."""
    script = _load_script()
    found = script.find_interfaces()
    assert "stock_zh_a_hist_sina" in found
