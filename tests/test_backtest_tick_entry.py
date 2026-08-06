"""run_backtest 的 tick 入口.

此前 ``BacktestDataInput`` 只列 ``List[Bar]``, 传 Tick 会在 Rust 层抛
``TypeError: argument 'bars': 'Tick' object is not an instance of 'Bar'``。
"""

from typing import Any, List

import pytest
from akquant import Strategy, run_backtest
from akquant.akquant import Bar, Tick
from akquant.backtest.fill_mode import CurrentClose

SYMBOL = "TKENTRY"
_BASE_NS = 1_672_707_000_000_000_000
_MINUTE_NS = 60_000_000_000


def _ns(minutes: int) -> int:
    """构造纳秒级时间戳."""
    return _BASE_NS + minutes * _MINUTE_NS


def _bar(minutes: int, close: float = 10.0) -> Bar:
    """构造一根 bar."""
    return Bar(
        timestamp=_ns(minutes),
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=1000.0,
        symbol=SYMBOL,
    )


def _tick(minutes: int, price: float = 10.0) -> Tick:
    """构造一个 tick."""
    return Tick(timestamp=_ns(minutes), price=price, volume=100.0, symbol=SYMBOL)


class _Recorder(Strategy):
    """记录收到的 bar 与 tick 时间戳."""

    def __init__(self) -> None:
        """初始化记录容器."""
        self.bars: List[int] = []
        self.ticks: List[int] = []

    def on_start(self) -> None:
        """订阅标的."""
        self.subscribe(SYMBOL)

    def on_bar(self, bar: Any) -> None:
        """记录 bar."""
        self.bars.append(int(bar.timestamp))

    def on_tick(self, tick: Any) -> None:
        """记录 tick."""
        self.ticks.append(int(tick.timestamp))


def _run(data: List[Any]) -> _Recorder:
    """跑一次回测并返回策略实例."""
    strategy = _Recorder()
    run_backtest(
        data=data,
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )
    return strategy


def test_tick_list_reaches_on_tick() -> None:
    """纯 tick 列表: on_tick 收到全部, on_bar 零次."""
    strategy = _run([_tick(1), _tick(2), _tick(3)])

    assert strategy.ticks == [_ns(1), _ns(2), _ns(3)]
    assert strategy.bars == []


def test_mixed_list_delivers_both_in_timestamp_order() -> None:
    """混合列表: bar 与 tick 都到达, 各自按时间戳有序."""
    strategy = _run([_bar(4), _tick(1), _bar(2), _tick(3)])

    assert strategy.bars == [_ns(2), _ns(4)]
    assert strategy.ticks == [_ns(1), _ns(3)]


def test_bar_only_list_still_works() -> None:
    """回归防护: 纯 bar 输入的既有行为不得改变."""
    strategy = _run([_bar(1), _bar(2)])

    assert strategy.bars == [_ns(1), _ns(2)]
    assert strategy.ticks == []


def test_empty_list_raises() -> None:
    """空列表早失败."""
    with pytest.raises(ValueError, match="空"):
        _run([])


def test_foreign_element_raises_type_error() -> None:
    """混合列表中的非法元素在 Python 层就报错, 并指名位置."""
    with pytest.raises(TypeError) as exc_info:
        _run([_bar(1), 42])

    assert "1" in str(exc_info.value)
