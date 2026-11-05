"""python_plain 加载器: 无 strategy_attr 时优先选本文件定义的 Strategy 子类."""

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from akquant import Bar, run_backtest
from akquant.strategy_loader import resolve_strategy_input

_HEAD = "from akquant.strategy import Strategy\n"


def _load(path: Path) -> Any:
    return resolve_strategy_input(
        strategy_source=str(path), strategy_loader="python_plain"
    )


def _bars(symbol: str) -> list[Bar]:
    idx = pd.date_range("2023-01-01", periods=3, freq="D") + pd.Timedelta(hours=2)
    return [
        Bar(
            timestamp=ts.value,
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1000.0,
            symbol=symbol,
        )
        for i, ts in enumerate(idx)
    ]


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_module_name_matches_class_module(tmp_path: Path) -> None:
    """模块名是编码后的稳定名, 与本文件类的 __module__ 一致."""
    path = _write(
        tmp_path / "own_mod.py",
        _HEAD + "class Own(Strategy):\n    def on_bar(self, bar):\n        pass\n",
    )
    cls = _load(path)
    assert cls.__module__ in sys.modules
    assert sys.modules[cls.__module__].__dict__["Own"] is cls


def test_cross_file_base_selects_local_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From base import MyBase + class S(MyBase): 选中 S, 回测可跑通."""
    _write(
        tmp_path / "cand_base_a.py",
        _HEAD + "class MyBase(Strategy):\n    def on_bar(self, bar):\n        pass\n",
    )
    s_path = _write(
        tmp_path / "cand_s_a.py",
        "from cand_base_a import MyBase\n\nclass S(MyBase):\n    pass\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert _load(s_path).__name__ == "S"
    result = run_backtest(
        data=_bars("CAND"),
        strategy_source=str(s_path),
        strategy_loader="python_plain",
        symbols="CAND",
        show_progress=False,
    )
    assert result is not None


def test_two_local_classes_still_multiple(tmp_path: Path) -> None:
    """本文件定义两个策略类: 仍报 multiple."""
    path = _write(
        tmp_path / "cand_two.py",
        _HEAD + "class A(Strategy):\n    pass\n\nclass B(Strategy):\n    pass\n",
    )
    with pytest.raises(ValueError, match="multiple Strategy subclasses found"):
        _load(path)


def test_single_reexport_still_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """本文件不定义策略类、只 re-export 一个: 仍能选中."""
    _write(
        tmp_path / "cand_base_c.py",
        _HEAD + "class Only(Strategy):\n    pass\n",
    )
    path = _write(tmp_path / "cand_re_c.py", "from cand_base_c import Only\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert _load(path).__name__ == "Only"


def test_two_imported_still_multiple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """本文件没有策略类、只 import 了两个: 仍报 multiple."""
    _write(
        tmp_path / "cand_base_d.py",
        _HEAD + "class P(Strategy):\n    pass\n\nclass Q(Strategy):\n    pass\n",
    )
    path = _write(tmp_path / "cand_re_d.py", "from cand_base_d import P, Q\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="multiple Strategy subclasses found"):
        _load(path)


def test_no_strategy_message_unchanged(tmp_path: Path) -> None:
    """没有任何策略类: 报错文字不变."""
    path = _write(tmp_path / "cand_none.py", "X = 1\n")
    with pytest.raises(ValueError, match="no Strategy subclass found in module"):
        _load(path)
