"""akquant-mcp 入口: 缺 SDK 时给出安装提示."""

import sys
from pathlib import Path

import pytest
from akquant.mcp.__main__ import INSTALL_HINT, main


def test_missing_sdk_prints_install_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """缺 mcp 包时返回 1, stderr 提示 pip install "akquant[mcp]"."""
    for name in [m for m in sys.modules if m == "mcp" or m.startswith("mcp.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.delitem(sys.modules, "akquant.mcp.server", raising=False)
    assert main([]) == 1
    assert INSTALL_HINT in capsys.readouterr().err
    assert "akquant[mcp]" in INSTALL_HINT


def test_bad_root_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--root 不存在时返回 1 并说明原因."""
    pytest.importorskip("mcp")
    assert main(["--root", str(tmp_path / "missing")]) == 1
    assert "不存在" in capsys.readouterr().err
