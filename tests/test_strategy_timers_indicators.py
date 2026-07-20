from typing import Any, cast
from unittest.mock import MagicMock

import akquant
import pandas as pd
import pytest
from akquant.akquant import StrategyContext
from akquant.indicator import Indicator
from akquant.strategy import Strategy


# Mock Indicator
class SMA(Indicator):
    """Mock SMA indicator."""

    def __init__(self, period: int) -> None:
        """Initialize Mock SMA."""
        super().__init__(f"sma{period}", lambda df: df["close"].rolling(period).mean())
        self.period = period


class MyTimerStrategy(Strategy):
    """Mock strategy for timer testing."""

    inc_sma: Any

    def __init__(self) -> None:
        """Initialize."""
        self.timer_triggered = False

    def on_start(self) -> None:
        """On start."""
        # Test manual timer
        self.schedule("2023-01-01 10:00:00", "manual_timer")
        # Test daily timer
        self.schedule_daily("14:55:00", "daily_timer")

    def on_timer(self, payload: str) -> None:
        """On timer."""
        if payload == "manual_timer" or payload == "daily_timer":
            self.timer_triggered = True


def test_timer_registration() -> None:
    """Test timer registration."""
    strategy = MyTimerStrategy()

    # Mock context and trading days
    ctx = MagicMock(spec=StrategyContext)
    strategy.ctx = ctx
    strategy._trading_days = [
        pd.Timestamp("2023-01-01").tz_localize("Asia/Shanghai"),
        pd.Timestamp("2023-01-02").tz_localize("Asia/Shanghai"),
    ]

    # Run on_start manually (usually called by backtest engine)
    strategy.on_start()

    # Verify schedule calls
    # 1. Manual timer: 2023-01-01 10:00:00 Asia/Shanghai
    manual_ts = pd.Timestamp("2023-01-01 10:00:00").tz_localize("Asia/Shanghai").value
    # ctx.schedule.assert_any_call(manual_ts, "manual_timer")

    # 2. Daily timer: 2023-01-01 14:55:00 and 2023-01-02 14:55:00
    daily_ts_1 = pd.Timestamp("2023-01-01 14:55:00").tz_localize("Asia/Shanghai").value
    daily_ts_2 = pd.Timestamp("2023-01-02 14:55:00").tz_localize("Asia/Shanghai").value

    # Verify calls exist in call_args_list
    # Note: assert_any_call can be tricky with exact matches if types differ
    # slightly (e.g. numpy int vs python int)
    # So we iterate and check values

    calls = strategy.ctx.schedule.call_args_list
    call_args = []
    for c in calls:
        ts_arg = c.args[0]
        # Convert timestamp to int value if it's not already
        if hasattr(ts_arg, "value"):
            ts_arg = ts_arg.value
        call_args.append((ts_arg, c.args[1]))

    # Debug print
    # print(f"Expected: {(manual_ts, 'manual_timer')}")
    # print(f"Actual: {call_args}")

    # Check if manual_ts is in call_args
    # (might need to handle type mismatch int vs numpy.int64)
    # Let's convert everything to python int for safety
    manual_ts = int(manual_ts)
    daily_ts_1 = int(daily_ts_1)
    daily_ts_2 = int(daily_ts_2)

    clean_call_args = []
    for ts, payload in call_args:
        # Handle string timestamp if mocking didn't convert it or logic failed
        if isinstance(ts, str):
            # Try to parse or just skip
            try:
                ts = int(pd.Timestamp(ts).tz_localize("Asia/Shanghai").value)
            except Exception:
                pass

        if isinstance(ts, (int, float)):
            clean_call_args.append((int(ts), payload))
        elif hasattr(ts, "value"):
            clean_call_args.append((int(ts.value), payload))

    assert (manual_ts, "manual_timer") in clean_call_args
    # Updated to match new wrapped payload format: __daily__|14:55:00|daily_timer
    assert (daily_ts_1, "__daily__|14:55:00|daily_timer") in clean_call_args
    assert (daily_ts_2, "__daily__|14:55:00|daily_timer") in clean_call_args


def test_register_precomputed_indicator() -> None:
    """Registers precomputed indicator under precompute mode."""
    strategy = MyTimerStrategy()
    strategy.indicator_mode = "precompute"
    indicator = SMA(5)

    strategy.register_precomputed_indicator("sma5", indicator)

    assert getattr(strategy, "sma5") is indicator
    assert indicator in strategy._precomputed_indicators


def test_register_incremental_indicator_requires_incremental_mode() -> None:
    """Rejects incremental registration when mode is precompute."""
    strategy = MyTimerStrategy()
    strategy.indicator_mode = "precompute"

    with pytest.raises(ValueError, match="indicator_mode='incremental'"):
        strategy.register_incremental_indicator("sma5", MagicMock())


def test_register_precomputed_indicator_requires_precompute_mode() -> None:
    """Rejects precomputed registration when mode is incremental."""
    strategy = MyTimerStrategy()
    strategy.indicator_mode = "incremental"

    with pytest.raises(ValueError, match="indicator_mode='precompute'"):
        strategy.register_precomputed_indicator("sma5", SMA(5))


def test_incremental_indicator_updates_from_bar_close() -> None:
    """Updates incremental indicator from close price source."""

    class IncrementalIndicator:
        def __init__(self) -> None:
            self.values: list[float] = []

        def update(self, value: float) -> None:
            self.values.append(value)

    class FakeBar:
        def __init__(self, close: float) -> None:
            self.open = 0.0
            self.high = 0.0
            self.low = 0.0
            self.close = close
            self.volume = 0.0

    strategy = MyTimerStrategy()
    strategy.indicator_mode = "incremental"
    indicator = IncrementalIndicator()
    strategy.register_incremental_indicator("inc_sma", indicator, source="close")

    strategy._update_incremental_indicators(FakeBar(close=12.5))  # type: ignore[arg-type]

    assert indicator.values == [12.5]


def test_incremental_indicator_symbol_filter() -> None:
    """Updates only when bar symbol is in configured filter."""

    class IncrementalIndicator:
        def __init__(self) -> None:
            self.values: list[float] = []

        def update(self, value: float) -> None:
            self.values.append(value)

    class FakeBar:
        def __init__(self, symbol: str, close: float) -> None:
            self.symbol = symbol
            self.open = 0.0
            self.high = 0.0
            self.low = 0.0
            self.close = close
            self.volume = 0.0

    strategy = MyTimerStrategy()
    strategy.indicator_mode = "incremental"
    indicator = IncrementalIndicator()
    strategy.register_incremental_indicator(
        "inc_sma",
        indicator,
        source="close",
        symbols=["000001.SZ_1D"],
    )

    strategy._update_incremental_indicators(
        cast(Any, FakeBar(symbol="000001.SZ", close=12.5))
    )
    strategy._update_incremental_indicators(
        cast(Any, FakeBar(symbol="000001.SZ_1D", close=13.0))
    )

    assert indicator.values == [13.0]


def test_incremental_indicator_uses_symbol_scoped_instances() -> None:
    """不同 symbol 的增量指标应维护独立状态."""

    class IncrementalIndicator:
        def __init__(self) -> None:
            self.values: list[float] = []

        def update(self, value: float) -> None:
            self.values.append(value)

    class FakeBar:
        def __init__(self, symbol: str, close: float) -> None:
            self.symbol = symbol
            self.open = close
            self.high = close
            self.low = close
            self.close = close
            self.volume = 0.0

    strategy = MyTimerStrategy()
    strategy.indicator_mode = "incremental"
    strategy.register_incremental_indicator(
        "inc_sma",
        indicator_factory=IncrementalIndicator,
        source="close",
    )

    strategy._update_incremental_indicators(cast(Any, FakeBar("000001.SZ", 10.0)))
    strategy._update_incremental_indicators(cast(Any, FakeBar("000002.SZ", 20.0)))
    strategy.current_bar = cast(Any, FakeBar("000001.SZ", 10.0))
    first_symbol_values = strategy.inc_sma.values
    strategy.current_bar = cast(Any, FakeBar("000002.SZ", 20.0))
    second_symbol_values = strategy.inc_sma.values

    assert first_symbol_values == [10.0]
    assert second_symbol_values == [20.0]


def test_incremental_indicator_bootstrap_respects_active_start_boundary() -> None:
    """历史预热只使用 start_time 之前的 bars，不提前消费首根有效 bar."""

    class IncrementalIndicator:
        def __init__(self) -> None:
            self.values: list[float] = []

        def update(self, value: float) -> None:
            self.values.append(value)

    class BootstrapStrategy(Strategy):
        inc_sma: Any

        def __init__(self) -> None:
            super().__init__()
            self.runtime_config = {"indicator_mode": "incremental"}

        def on_start(self) -> None:
            self.register_incremental_indicator(
                "inc_sma",
                indicator_factory=IncrementalIndicator,
                source="close",
                warmup_bars=2,
            )

        def on_bar(self, bar: Any) -> None:
            pass

    symbol = "BOOTSTRAP_TEST"
    timestamps = pd.date_range("2024-01-01 09:30:00", periods=4, freq="min", tz="UTC")
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [10.0, 11.0, 12.0, 13.0],
            "low": [10.0, 11.0, 12.0, 13.0],
            "close": [10.0, 11.0, 12.0, 13.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
            "symbol": [symbol, symbol, symbol, symbol],
        }
    )

    result = akquant.run_backtest(
        strategy=BootstrapStrategy,
        data=data,
        symbols=[symbol],
        start_time=timestamps[2],
        end_time=timestamps[-1],
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
    )

    strategy = result.strategy
    assert strategy is not None
    indicator = strategy.inc_sma.get_instance(symbol)
    assert indicator.values == [10.0, 11.0, 12.0, 13.0]
