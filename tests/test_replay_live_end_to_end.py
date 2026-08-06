"""端到端: 数据真正流经 live feed → Rust 引擎 → 策略回调.

这是此前完全没有测试覆盖的通路(见 spec 背景), 也是用户反馈中
"订阅无数据 / current_tick 为 None / 无 bar-tick 推送 / 多品种部分无数据"
四条问题所在的位置。
"""

from typing import Any, List

import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar, Tick

SYMBOL = "REPLAY_A"


def _instrument(symbol: str) -> Instrument:
    """构造一个股票标的."""
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


def _ts(text: str) -> int:
    """把本地时间字符串转成纳秒时间戳."""
    return int(pd.Timestamp(text, tz="Asia/Shanghai").value)


def _bar(ts: int, symbol: str, close: float) -> Bar:
    """构造一根 bar."""
    return Bar(
        timestamp=ts,
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=1000.0,
        symbol=symbol,
    )


class _Recorder(Strategy):
    """记录收到的 bar/tick 与 current_tick 状态."""

    def __init__(self) -> None:
        """初始化记录容器."""
        self.bars: List[Any] = []
        self.ticks: List[Any] = []
        self.tick_had_current: List[bool] = []

    def on_bar(self, bar: Bar) -> None:
        """记录 bar."""
        self.bars.append((int(bar.timestamp), str(bar.symbol)))

    def on_tick(self, tick: Tick) -> None:
        """记录 tick, 并检查 current_tick 是否可见."""
        self.ticks.append((int(tick.timestamp), str(tick.symbol)))
        self.tick_had_current.append(self.current_tick is not None)


def test_bars_reach_strategy_in_order() -> None:
    """全部 bar 按推送顺序到达 on_bar, 且会话自行结束."""
    strategy = _Recorder()
    stamps = [
        _ts("2023-01-03 09:30:00"),
        _ts("2023-01-03 10:00:00"),
        _ts("2023-01-03 14:00:00"),
    ]

    run_live(
        strategy_cls=strategy,
        instruments=[_instrument(SYMBOL)],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": [_bar(ts, SYMBOL, 10.0) for ts in stamps]},
        cash=100_000.0,
        show_progress=False,
        duration="60s",
    )

    assert [ts for ts, _ in strategy.bars] == stamps


def test_current_tick_is_visible_inside_on_tick() -> None:
    """on_tick 内 current_tick 必须非 None（直接对应用户反馈）."""
    strategy = _Recorder()
    ticks = [
        Tick(
            timestamp=_ts("2023-01-03 09:30:00"),
            price=10.0,
            volume=100.0,
            symbol=SYMBOL,
        ),
        Tick(
            timestamp=_ts("2023-01-03 09:30:01"),
            price=10.1,
            volume=100.0,
            symbol=SYMBOL,
        ),
    ]

    run_live(
        strategy_cls=strategy,
        instruments=[_instrument(SYMBOL)],
        broker="replay",
        trading_mode="paper",
        gateway_options={"ticks": ticks},
        cash=100_000.0,
        show_progress=False,
        duration="60s",
    )

    assert len(strategy.ticks) == 2, f"tick 未全部到达: {strategy.ticks}"
    assert all(strategy.tick_had_current), "on_tick 内 current_tick 为 None"


def test_multi_symbol_events_arrive_globally_ordered() -> None:
    """三个品种的事件按时间戳全局有序到达（对应"多品种部分无数据"反馈）."""
    strategy = _Recorder()
    plan = [
        (_ts("2023-01-03 09:30:00"), "REPLAY_A"),
        (_ts("2023-01-03 09:31:00"), "REPLAY_B"),
        (_ts("2023-01-03 09:32:00"), "REPLAY_C"),
        (_ts("2023-01-03 09:33:00"), "REPLAY_A"),
    ]
    symbols = ["REPLAY_A", "REPLAY_B", "REPLAY_C"]

    run_live(
        strategy_cls=strategy,
        instruments=[_instrument(s) for s in symbols],
        broker="replay",
        trading_mode="paper",
        # 故意乱序传入, 验证网关自己排序
        gateway_options={"bars": [_bar(ts, sym, 10.0) for ts, sym in reversed(plan)]},
        cash=100_000.0,
        show_progress=False,
        duration="60s",
    )

    assert strategy.bars == plan


def test_dataframe_input_works_end_to_end() -> None:
    """DataFrame 入参同样能走通实盘通路."""
    strategy = _Recorder()
    stamps = [_ts("2023-01-03 09:30:00"), _ts("2023-01-03 09:31:00")]
    df = pd.DataFrame(
        [
            {
                # 列名必须是 date: dataframe_to_bars 的必需字段映射为
                # {"date": "timestamp", ...}，传 timestamp 会抛 ValueError。
                "date": pd.Timestamp(ts, unit="ns", tz="UTC"),
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
                "volume": 1000.0,
                "股票代码": SYMBOL,
            }
            for ts in stamps
        ]
    )

    run_live(
        strategy_cls=strategy,
        instruments=[_instrument(SYMBOL)],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": df},
        cash=100_000.0,
        show_progress=False,
        duration="60s",
    )

    assert [ts for ts, _ in strategy.bars] == stamps
