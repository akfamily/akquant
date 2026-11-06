"""``akquant`` 命令: ``akquant mcp`` 与 ``akquant-mcp`` 等价.

MCP 官方注册表的客户端按包名启动服务(``uvx ... akquant mcp --root=...``),
所以 ``akquant mcp`` 必须把参数原样交给 ``akquant-mcp`` 的入口.
"""

from pathlib import Path
from typing import List, Optional

import akquant
import pytest
from akquant import cli


def test_mcp_subcommand_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """子命令 akquant mcp 之后的参数(含 --flag=value 写法)原样交给 akquant-mcp."""
    seen: List[Optional[List[str]]] = []

    def fake_main(argv: Optional[List[str]] = None, prog: str = "") -> int:
        seen.append(argv)
        assert prog == "akquant mcp"
        return 7

    monkeypatch.setattr("akquant.mcp.__main__.main", fake_main)
    assert cli.main(["mcp", "--root=/tmp/x", "--max-runs", "3"]) == 7
    assert seen == [["--root=/tmp/x", "--max-runs", "3"]]


def test_mcp_subcommand_runs_real_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """不 mock 时走到真实入口: --root 不存在返回 1, 报错写 stderr 且 stdout 干净."""
    pytest.importorskip("mcp")
    assert cli.main(["mcp", f"--root={tmp_path / 'missing'}"]) == 1
    captured = capsys.readouterr()
    assert "不存在" in captured.err
    # stdio 方式下 stdout 只能输出协议消息
    assert captured.out == ""


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """选项 akquant --version 打印版本号."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert akquant.__version__ in capsys.readouterr().out


@pytest.mark.parametrize("argv", [[], ["unknown"]])
def test_missing_or_unknown_command_fails(
    argv: List[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """没有子命令或子命令不存在时返回非 0, 并在 stderr 提示用法."""
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code != 0
    assert "mcp" in capsys.readouterr().err
