"""Rust 侧标的白名单过滤: 只有它能覆盖 DataFeed 对象输入.

``DataFeed`` 的公开方法只有 add_arrays / add_bar / add_bars / add_tick /
create_live / from_csv / from_parquet / sort —— **只写不读**, Python 层无从
枚举它内含哪些标的(``backtest/engine.py:3128-3134`` 的注释亦坦承此点)。
故 DataFeed 形态的过滤只能在 Rust 层做, 本文件直接调 FFI 验证这一层。
"""

from typing import Any

import akquant


def _feed_with_two_symbols() -> Any:
    """构造含 X 与 Y 各 3 根 bar 的 DataFeed(时间戳交错递增)."""
    feed = akquant.DataFeed()
    for i in range(3):
        for symbol, base in (("X", 10.0), ("Y", 100.0)):
            feed.add_bar(
                akquant.Bar(
                    timestamp=f"2025-01-{2 + i:02d}",  # type: ignore[arg-type]
                    symbol=symbol,
                    open=base + i,
                    high=base + i,
                    low=base + i,
                    close=base + i,
                    volume=100.0,
                )
            )
    feed.sort()
    return feed


class _RecordingStrategy(akquant.Strategy):
    """记录每个标的的 on_bar 触发次数."""

    def on_start(self) -> None:
        """初始化命中表."""
        self.hits: dict[str, int] = {}

    def on_bar(self, bar: akquant.Bar) -> None:
        """累计该标的的触发次数."""
        self.hits[bar.symbol] = self.hits.get(bar.symbol, 0) + 1


def _run(whitelist: list[str] | None) -> dict[str, int]:
    engine = akquant.Engine()
    engine.use_simple_market(0.0)
    engine.set_force_session_continuous(True)
    engine.set_cash(100000.0)
    if whitelist is not None:
        engine.set_symbol_whitelist(whitelist)
    engine.add_data(_feed_with_two_symbols())
    strategy = _RecordingStrategy()
    engine.run(strategy, False)
    return strategy.hits


def test_whitelist_filters_out_other_symbols() -> None:
    """设了白名单 ['X'] 后, Y 的 bar 事件不得到达策略."""
    assert _run(["X"]) == {"X": 3}


def test_no_whitelist_lets_everything_through() -> None:
    """未设白名单时放行全部 —— 这是不传 symbols 的回归底线."""
    assert _run(None) == {"X": 3, "Y": 3}


def test_empty_whitelist_is_treated_as_unset() -> None:
    """空列表按「未设置」处理, 不得跑出一个什么都不放行的空回测.

    Python 侧会在参数解析阶段就拒绝空 symbols(见 Task 2), 这里是 Rust 侧的
    防御: 万一空列表传到底层, 也不能静默产出一个零事件的回测。
    """
    assert _run([]) == {"X": 3, "Y": 3}
