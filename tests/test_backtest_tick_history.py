"""纯 tick 回测下的历史缓冲区.

此前 ``src/pipeline/stages/data.rs`` 的 ``Event::Tick`` 分支只写
``current_symbol_events``, 不调 ``buffer.update``, 导致 tick 回测下
``get_history`` 恒为空——且是**静默**的, 既不报错也不工作。
"""

from typing import Any, List, Optional

import pytest
from akquant import Strategy, run_backtest
from akquant.akquant import DataFeed, Tick
from akquant.backtest.fill_mode import CurrentClose

SYMBOL = "TKHIST"
_BASE_NS = 1_672_707_000_000_000_000
_MINUTE_NS = 60_000_000_000


def _ns(minutes: int) -> int:
    """构造纳秒级时间戳: 基准时刻 + ``minutes`` 分钟."""
    return _BASE_NS + minutes * _MINUTE_NS


def _tick(minutes: int, price: float) -> Tick:
    """构造一个 tick."""
    return Tick(timestamp=_ns(minutes), price=price, volume=100.0, symbol=SYMBOL)


class _HistoryProbe(Strategy):
    """在第 6 个 tick 上读取历史深度与内容."""

    def __init__(self) -> None:
        """初始化探针状态."""
        self.tick_count = 0
        self.observed: Optional[List[float]] = None

    def on_start(self) -> None:
        """订阅并启用历史."""
        self.subscribe(SYMBOL)
        self.set_history_depth(10)

    def on_tick(self, tick: Any) -> None:
        """累计 tick 并在第 6 个上抓取历史."""
        self.tick_count += 1
        if self.tick_count == 6:
            # 注意参数顺序: get_history(count, symbol=None, field="close")
            # 见 python/akquant/strategy_history.py:31-33
            values = self.get_history(5, SYMBOL, "close")
            self.observed = None if values is None else [float(v) for v in values]


def test_tick_backtest_populates_history() -> None:
    """纯 tick 回测下 get_history 必须返回价格序列, 而非空.

    tick 以退化 bar 写入(open=high=low=close=price), 故 ``"close"`` 字段
    取到的就是成交价序列。
    """
    feed = DataFeed()
    for minute in range(1, 9):
        feed.add_tick(_tick(minute, 10.0 + minute * 0.1))
    feed.sort()

    strategy = _HistoryProbe()
    run_backtest(
        data=feed,
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    assert strategy.tick_count == 8
    assert strategy.observed is not None, "get_history 返回 None: tick 未进入历史缓冲区"
    assert len(strategy.observed) == 5
    # 第 6 个 tick 时, 最近 5 个价格是第 2..6 个 tick 的价格。
    #
    # 必须用 pytest.approx: 引擎的 f64 -> Decimal -> f64 往返
    # (extract_decimal, src/model/market_data.rs, 走 Decimal::from_f64_retain)
    # 对部分十进制字面量不精确——实测 10.1 与 10.6 会变成 10.100000000000001 /
    # 10.600000000000001, 而 10.2/10.3/10.4/10.5 精确。这是**先于本计划存在**的
    # 引擎性质, 对 Bar 与 Tick 完全一样(Bar(close=10.6).close == 10.6 也是 False),
    # 与 tick 历史无关。本仓库既有测试用 pytest.approx 处理浮点断言(159 处)。
    assert strategy.observed == pytest.approx([10.2, 10.3, 10.4, 10.5, 10.6])
