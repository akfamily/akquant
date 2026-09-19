"""实盘路径: replay 行情源 + subscribe_bars, 基础周期已知时窗口即时闭合."""

from typing import List, Optional

import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar

SYMBOL = "REPLAY_W"


def _instrument(symbol: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bars(n: int) -> List[Bar]:
    idx = pd.date_range(
        "2024-01-02 09:31:00", periods=n, freq="1min", tz="Asia/Shanghai"
    )
    return [
        Bar(int(ts.value), 10.0 + i, 10.5 + i, 9.5 + i, 10.1 + i, 100.0, SYMBOL)
        for i, ts in enumerate(idx)
    ]


class _LiveWindows(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars("5min")
        self.base_ts: List[int] = []
        # 元素为 (窗口标签, 触发时已收基础 bar 数)
        self.window_events: List[tuple[int, int]] = []
        self.freq_seen: Optional[str] = None

    def on_start(self) -> None:
        self.set_history_depth(20)
        self.freq_seen = self.freq

    def on_bar(self, bar: Bar) -> None:
        self.base_ts.append(int(bar.timestamp))

    def on_window_bar(self, bar: Bar) -> None:
        self.window_events.append((int(bar.timestamp), len(self.base_ts)))


def test_live_replay_window_closes_immediately_with_declared_freq() -> None:
    """声明 freq 时窗口即时闭合, 会话结束 flush 尾部未满窗口."""
    strategy = _LiveWindows()
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument(SYMBOL)],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars(7), "freq": "1min"},
        duration="60s",
    )
    assert strategy.freq_seen == "1min"
    # 09:35 窗口在第 5 根基础 bar 那一步闭合(即时), 尾部 09:36-09:37 在会话结束 flush,
    # 标签 = ceil(09:37, 5min) = 09:40
    assert len(strategy.window_events) == 2
    assert strategy.window_events[0] == (strategy.base_ts[4], 5)
    assert strategy.window_events[1] == (
        int(pd.Timestamp("2024-01-02 09:40", tz="Asia/Shanghai").value),
        7,
    )


def test_live_replay_window_defers_without_declared_freq() -> None:
    """未声明 freq 时窗口延迟闭合(需多等一根基础 bar 才能推断周期)."""
    strategy = _LiveWindows()
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument(SYMBOL)],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars(7)},
        duration="60s",
    )
    assert strategy.freq_seen is None
    # 延迟闭合: 09:35 窗口在第 6 根(09:36)之后才派发
    assert strategy.window_events[0] == (strategy.base_ts[4], 6)


def test_live_replay_dispatches_window_closed_on_terminal_event() -> None:
    """有界 replay 会话的最后一根基础 bar 恰好即时闭合一个窗口时不能丢派发.

    `broker='replay'` 靠 `bounded_event_total`(等于喂入的事件总数)驱动会话结束:
    处理完最后一个事件后从 Python 侧抛 `KeyboardInterrupt`, 由
    `Engine::run()` 的收尾分支调用 `flush_window_tail()`。若第 5 根 bar
    (09:35)本身就是即时闭合边界, 该窗口 bar 会在 `DataProcessor` 阶段写进
    `pending_window_bars`, 但 `StrategyProcessor` 派发那一步还没跑到就被
    `KeyboardInterrupt` 中断——`flush_window_tail` 必须把 `pending_window_bars`
    一并排空派发, 否则这根窗口 bar 会静默丢失(不回调、不出现在尾部 flush 里)。
    """
    strategy = _LiveWindows()
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument(SYMBOL)],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars(5), "freq": "1min"},
        duration="60s",
    )
    assert strategy.freq_seen == "1min"
    assert strategy.window_events == [(strategy.base_ts[4], 5)]
