"""symbols 语义: 传了就只跑这些标的, 不传则沿用「数据即订阅」.

设计见 docs/superpowers/specs/2026-08-14-symbols-filtering-design.md
"""

from typing import Any

import akquant as aq
import pytest


class _RecordingStrategy(aq.Strategy):
    """记录每个标的的 on_bar 触发次数."""

    def on_start(self) -> None:
        """初始化命中表."""
        self.hits: dict[str, int] = {}

    def on_bar(self, bar: aq.Bar) -> None:
        """累计该标的的触发次数."""
        self.hits[bar.symbol] = self.hits.get(bar.symbol, 0) + 1


def _bars() -> list[aq.Bar]:
    """X 与 Y 各 3 根 bar."""
    out = []
    for i in range(3):
        for symbol, base in (("X", 10.0), ("Y", 100.0)):
            out.append(
                aq.Bar(
                    timestamp=f"2025-01-{2 + i:02d}",  # type: ignore[arg-type]
                    symbol=symbol,
                    open=base + i,
                    high=base + i,
                    low=base + i,
                    close=base + i,
                    volume=100.0,
                )
            )
    return out


def _feed() -> Any:
    """同样的数据装进 DataFeed —— 走 Rust 层过滤这条路径."""
    feed = aq.DataFeed()
    for bar in _bars():
        feed.add_bar(bar)
    feed.sort()
    return feed


def _run(data: Any, **kwargs: Any) -> dict[str, int]:
    strategy = _RecordingStrategy()
    aq.run_backtest(strategy=strategy, data=data, initial_cash=100000, **kwargs)
    return strategy.hits


def test_explicit_symbols_filters_list_of_bars() -> None:
    """传了 symbols=['X'] 时 Y 不得参与(List[Bar] 形态)."""
    assert _run(_bars(), symbols=["X"]) == {"X": 3}


def test_explicit_symbols_filters_data_feed() -> None:
    """DataFeed 形态同样被过滤 —— 这条只能靠 Rust 层白名单."""
    assert _run(_feed(), symbols=["X"]) == {"X": 3}


def test_omitting_symbols_keeps_data_as_subscription() -> None:
    """不传 symbols 时沿用「数据即订阅」, 两个标的都跑(回归底线)."""
    assert _run(_bars()) == {"X": 3, "Y": 3}


def test_empty_symbols_list_is_rejected() -> None:
    """显式传空列表是参数错误, 必须报错而非静默跑出空回测."""
    with pytest.raises(ValueError, match="symbols"):
        _run(_bars(), symbols=[])


def _bars_with_benchmark_symbol() -> list[aq.Bar]:
    """标的代码分别为 BENCHMARK 与 OTHER, 各 3 根 bar.

    BENCHMARK 同时是本文件内部代表"未指定标的"的哨兵字面量, 也是这里用作
    真实标的代码的测试数据 —— 二者刚好撞了同一个字符串, 这正是要验证的场景。
    """
    out = []
    for i in range(3):
        for symbol, base in (("BENCHMARK", 10.0), ("OTHER", 100.0)):
            out.append(
                aq.Bar(
                    timestamp=f"2025-01-{2 + i:02d}",  # type: ignore[arg-type]
                    symbol=symbol,
                    open=base + i,
                    high=base + i,
                    low=base + i,
                    close=base + i,
                    volume=100.0,
                )
            )
    return out


def test_explicit_symbols_literal_benchmark_still_filters() -> None:
    """symbols=["BENCHMARK"] 显式传入时必须真的过滤掉 "OTHER".

    回归护栏: 曾经的实现把"显式传入的 BENCHMARK"错误折算成"未显式传入",
    导致这种情况下过滤永远不生效、"OTHER" 会被静默一并跑出。
    """
    assert _run(_bars_with_benchmark_symbol(), symbols=["BENCHMARK"]) == {
        "BENCHMARK": 3
    }
