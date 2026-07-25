import csv
import json
import logging
import time
import warnings
from datetime import date, datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, cast

import akquant
import numpy as np
import pandas as pd
import pytest
from akquant.backtest import engine as backtest_engine
from akquant.backtest.fill_mode import (
    CurrentClose,
    NextAverage,
    NextClose,
    NextHighLowMid,
    NextOpen,
)
from akquant.data import ParquetDataCatalog


def test_engine_initialization() -> None:
    """Test Engine initialization defaults."""
    engine = akquant.Engine()
    assert engine.portfolio.cash == 100000.0
    assert len(engine.trades) == 0
    assert len(engine.orders) == 0


class DummyStrategy(akquant.Strategy):
    """A dummy strategy for testing purposes."""


class RegressionStrategy(akquant.Strategy):
    """Regression strategy for baseline checks."""

    def __init__(self) -> None:
        """Initialize the regression strategy."""
        super().__init__()
        self.bar_index = 0

    def on_bar(self, bar: akquant.Bar) -> None:
        """Handle bar events for deterministic trades."""
        if self.bar_index == 0:
            self.buy(symbol=bar.symbol, quantity=10)
        elif self.bar_index == 2:
            self.sell(symbol=bar.symbol, quantity=10)
        self.bar_index += 1


class NoopStrategy(akquant.Strategy):
    """No-op strategy used for performance baselines."""

    dummy = akquant.IntParam(0)

    def on_bar(self, bar: akquant.Bar) -> None:
        """Handle bar events without generating orders."""
        return


class FailingOnStopStrategy(akquant.Strategy):
    """Strategy that raises during on_stop for logging assertions."""

    def on_bar(self, bar: akquant.Bar) -> None:
        """Keep the strategy inactive during bar processing."""
        _ = bar

    def on_stop(self) -> None:
        """Raise a deterministic failure for log context checks."""
        raise RuntimeError("boom")


class WorkerLogStrategy(akquant.Strategy):
    """Strategy used to verify worker log forwarding in parallel optimization."""

    dummy = akquant.IntParam(0)

    def on_start(self) -> None:
        """Reset the one-shot logging guard."""
        self._logged = False

    def on_bar(self, bar: akquant.Bar) -> None:
        """Log once per task to test cross-process log forwarding."""
        if self._logged:
            return
        self.log(f"worker-log-{self.params.dummy}")
        self._logged = True


class ProfileCaptureStrategy(akquant.Strategy):
    """Capture resolved market profile fields from strategy runtime."""

    def __init__(self) -> None:
        """Initialize captured snapshot container."""
        super().__init__()
        self.snapshot: dict[str, float | int | str] = {}

    def on_start(self) -> None:
        """Capture strategy runtime fields after backtest startup."""
        self.snapshot = {
            "commission_rate": float(self.commission_rate),
            "commission_policy_type": str(
                getattr(self, "commission_policy", {}).get("type", "")
            ),
            "commission_policy_value": float(
                getattr(self, "commission_policy", {}).get("value", 0.0)
            ),
            "stamp_tax_rate": float(self.stamp_tax_rate),
            "transfer_fee_rate": float(self.transfer_fee_rate),
            "min_commission": float(self.min_commission),
            "lot_size": int(self.lot_size),
        }


class SingleBuyStrategy(akquant.Strategy):
    """Place a single buy order on first bar only."""

    def __init__(self) -> None:
        """Initialize one-shot state."""
        super().__init__()
        self._submitted = False

    def on_bar(self, bar: akquant.Bar) -> None:
        """Submit first-bar buy only once."""
        if self._submitted:
            return
        self.buy(symbol=bar.symbol, quantity=10)
        self._submitted = True


class DualBuyStrategy(akquant.Strategy):
    """Place two buy orders on first two bars."""

    def __init__(self) -> None:
        """Initialize counter state."""
        super().__init__()
        self._step = 0

    def on_bar(self, bar: akquant.Bar) -> None:
        """Submit a buy order on first two bars."""
        if self._step < 2:
            self.buy(symbol=bar.symbol, quantity=10)
        self._step += 1


class BuyBuySellBuyStrategy(akquant.Strategy):
    """Buy, buy, sell, then buy for reduce-only transition checks."""

    def __init__(self) -> None:
        """Initialize step counter."""
        super().__init__()
        self._step = 0

    def on_bar(self, bar: akquant.Bar) -> None:
        """Submit deterministic sequence for reduce-only verification."""
        if self._step == 0:
            self.buy(symbol=bar.symbol, quantity=10)
        elif self._step == 1:
            self.buy(symbol=bar.symbol, quantity=10)
        elif self._step == 2:
            self.sell(symbol=bar.symbol, quantity=5)
        elif self._step == 3:
            self.buy(symbol=bar.symbol, quantity=5)
        self._step += 1


class PositionEntryPriceCaptureStrategy(akquant.Strategy):
    """Capture runtime position helper state across buy/sell transitions."""

    def __init__(self) -> None:
        """Initialize step counter and capture buffer."""
        super().__init__()
        self._step = 0
        self.snapshots: list[dict[str, float | str]] = []

    def on_bar(self, bar: akquant.Bar) -> None:
        """Record helper values before submitting deterministic orders."""
        pos = self.position
        ctx_entry_price = 0.0
        if self.ctx is not None:
            ctx_entry_price = float(self.ctx.get_position_entry_price(bar.symbol))
        self.snapshots.append(
            {
                "size": float(pos.size),
                "available": float(pos.available),
                "entry_price": float(pos.entry_price),
                "avg_price": float(pos.avg_price),
                "ctx_entry_price": ctx_entry_price,
                "repr": repr(pos),
            }
        )
        if self._step == 0:
            self.buy(symbol=bar.symbol, quantity=10)
        elif self._step == 1:
            self.buy(symbol=bar.symbol, quantity=10)
        elif self._step == 2:
            self.sell(symbol=bar.symbol, quantity=5)
        elif self._step == 3:
            self.sell(symbol=bar.symbol, quantity=15)
        self._step += 1


class ContinuousBuyStrategy(akquant.Strategy):
    """Submit a buy order on every bar."""

    def on_bar(self, bar: akquant.Bar) -> None:
        """Submit deterministic repeated buy orders."""
        self.buy(symbol=bar.symbol, quantity=10)


class ContinuousSmallBuyStrategy(akquant.Strategy):
    """Submit a small buy order on every bar."""

    def on_bar(self, bar: akquant.Bar) -> None:
        """Submit deterministic repeated small buy orders."""
        self.buy(symbol=bar.symbol, quantity=5)


class TimerCurrentCloseStrategy(akquant.Strategy):
    """Submit order from timer and capture timer/trade timestamps."""

    def __init__(self) -> None:
        """Initialize capture state."""
        super().__init__()
        self.timer_timestamp: int | None = None
        self.trade_timestamp: int | None = None
        self.trade_price: float | None = None
        self.symbol_ref: str = "TIMER_BUG"
        self.timer_trigger: pd.Timestamp = pd.Timestamp(
            "2023-01-02 10:00:01", tz="Asia/Shanghai"
        )

    def on_start(self) -> None:
        """Register a timer between first and second bar."""
        self.schedule(self.timer_trigger, "timer_buy")

    def on_timer(self, payload: str) -> None:
        """Submit market buy on timer event."""
        if payload != "timer_buy":
            return
        if self.ctx is None:
            return
        self.timer_timestamp = int(self.ctx.current_time)
        self.buy(symbol=self.symbol_ref, quantity=1)

    def on_trade(self, trade: akquant.Trade) -> None:
        """Capture trade timestamp and price."""
        self.trade_timestamp = int(trade.timestamp)
        self.trade_price = float(trade.price)


class BarOnlyCaptureStrategy(akquant.Strategy):
    """Capture on_bar order fill timestamp and price."""

    def __init__(self) -> None:
        """Initialize capture state."""
        super().__init__()
        self.submitted = False
        self.trade_timestamp: int | None = None
        self.trade_price: float | None = None

    def on_bar(self, bar: akquant.Bar) -> None:
        """Submit one market order on first bar."""
        if self.submitted:
            return
        self.buy(symbol=bar.symbol, quantity=1)
        self.submitted = True

    def on_trade(self, trade: akquant.Trade) -> None:
        """Capture first trade timestamp and price."""
        self.trade_timestamp = int(trade.timestamp)
        self.trade_price = float(trade.price)


class MixedBarTimerCaptureStrategy(akquant.Strategy):
    """Submit one order on bar and one order on timer, then capture fills."""

    def __init__(self) -> None:
        """Initialize capture state."""
        super().__init__()
        self.bar_submitted = False
        self.timer_submitted = False
        self.trade_timestamps: list[int] = []
        self.trade_prices: list[float] = []
        self.symbol_ref: str = "TIMER_BUG"
        self.timer_timestamp: int | None = None
        self.timer_trigger: pd.Timestamp = pd.Timestamp(
            "2023-01-02 10:00:01", tz="Asia/Shanghai"
        )

    def on_start(self) -> None:
        """Register timer trigger between two bars."""
        self.schedule(self.timer_trigger, "timer_buy")

    def on_bar(self, bar: akquant.Bar) -> None:
        """Submit bar-side order once."""
        if self.bar_submitted:
            return
        self.buy(symbol=bar.symbol, quantity=1)
        self.bar_submitted = True

    def on_timer(self, payload: str) -> None:
        """Submit timer-side order once."""
        if payload != "timer_buy" or self.timer_submitted:
            return
        if self.ctx is not None:
            self.timer_timestamp = int(self.ctx.current_time)
        self.buy(symbol=self.symbol_ref, quantity=1)
        self.timer_submitted = True

    def on_trade(self, trade: akquant.Trade) -> None:
        """Capture all trade timestamps and prices."""
        self.trade_timestamps.append(int(trade.timestamp))
        self.trade_prices.append(float(trade.price))


class DailyTimerBuyStrategy(akquant.Strategy):
    """Submit one order from daily timer for trading-day alignment checks."""

    def __init__(self, symbol: str) -> None:
        """Initialize symbol and one-shot state."""
        super().__init__()
        self.symbol_ref = symbol
        self.submitted = False
        self.exited = False

    def on_start(self) -> None:
        """Register a daily timer at session close."""
        self.schedule_daily("15:00:00", "daily_buy")

    def on_bar(self, bar: akquant.Bar) -> None:
        """Close the position on the first bar after timer entry."""
        if self.exited or not self.submitted:
            return
        if self.position.size <= 0:
            return
        self.sell(symbol=bar.symbol, quantity=1)
        self.exited = True

    def on_timer(self, payload: str) -> None:
        """Submit one buy on the first matching daily timer."""
        if payload != "daily_buy" or self.submitted:
            return
        self.buy(symbol=self.symbol_ref, quantity=1)
        self.submitted = True


class DailyTimerOrderLevelCurrentCloseStrategy(akquant.Strategy):
    """Use order-level current-close fill policy on daily timers."""

    def __init__(self, symbol: str) -> None:
        """Initialize symbol and order-level fill policy."""
        super().__init__()
        self.symbol_ref = symbol
        self.fill_mode = CurrentClose()

    def on_start(self) -> None:
        """Register opening and closing daily timers."""
        self.schedule_daily("09:25:00", "daily_buy")
        self.schedule_daily("14:56:00", "daily_sell")

    def on_timer(self, payload: str) -> None:
        """Buy at the first timer and sell available shares at the second."""
        if payload == "daily_buy":
            self.buy(
                symbol=self.symbol_ref,
                quantity=1,
                tag="timer-buy",
                fill_mode=self.fill_mode,
            )
            return
        if payload != "daily_sell":
            return
        available = self.get_available_position(self.symbol_ref)
        if available <= 0:
            return
        self.sell(
            symbol=self.symbol_ref,
            quantity=available,
            tag="timer-sell",
            fill_mode=self.fill_mode,
        )


def _ns(dt: datetime) -> int:
    """
    Convert a datetime to nanoseconds since epoch.

    :param dt: Datetime object.
    :return: Nanoseconds since epoch.
    """
    return int(dt.timestamp() * 1e9)


def _build_regression_bars(symbol: str) -> list[akquant.Bar]:
    """
    Build a deterministic 3-bar series for regression verification.

    :param symbol: Symbol for bars.
    :return: List of Bar objects.
    """
    day1 = _ns(datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc))
    day2 = _ns(datetime(2023, 1, 3, 15, 0, tzinfo=timezone.utc))
    day3 = _ns(datetime(2023, 1, 4, 15, 0, tzinfo=timezone.utc))
    return [
        akquant.Bar(day1, 10.0, 10.0, 10.0, 10.0, 1000.0, symbol),
        akquant.Bar(day2, 12.0, 12.0, 12.0, 12.0, 1000.0, symbol),
        akquant.Bar(day3, 11.0, 11.0, 11.0, 11.0, 1000.0, symbol),
    ]


def _build_daily_loss_bars(symbol: str) -> list[akquant.Bar]:
    """Build bars where the second bar marks down unrealized PnL."""
    day1 = _ns(datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc))
    day2 = _ns(datetime(2023, 1, 3, 15, 0, tzinfo=timezone.utc))
    day3 = _ns(datetime(2023, 1, 4, 15, 0, tzinfo=timezone.utc))
    return [
        akquant.Bar(day1, 10.0, 10.0, 10.0, 10.0, 1000.0, symbol),
        akquant.Bar(day2, 8.0, 8.0, 8.0, 8.0, 1000.0, symbol),
        akquant.Bar(day3, 8.0, 8.0, 8.0, 8.0, 1000.0, symbol),
    ]


def _build_position_entry_price_bars(symbol: str) -> list[akquant.Bar]:
    """Build bars for weighted-average position entry price checks."""
    day1 = _ns(datetime(2023, 2, 1, 15, 0, tzinfo=timezone.utc))
    day2 = _ns(datetime(2023, 2, 2, 15, 0, tzinfo=timezone.utc))
    day3 = _ns(datetime(2023, 2, 3, 15, 0, tzinfo=timezone.utc))
    day4 = _ns(datetime(2023, 2, 4, 15, 0, tzinfo=timezone.utc))
    day5 = _ns(datetime(2023, 2, 5, 15, 0, tzinfo=timezone.utc))
    return [
        akquant.Bar(day1, 10.0, 10.0, 10.0, 10.0, 1000.0, symbol),
        akquant.Bar(day2, 12.0, 12.0, 12.0, 12.0, 1000.0, symbol),
        akquant.Bar(day3, 11.0, 11.0, 11.0, 11.0, 1000.0, symbol),
        akquant.Bar(day4, 11.0, 11.0, 11.0, 11.0, 1000.0, symbol),
        akquant.Bar(day5, 11.0, 11.0, 11.0, 11.0, 1000.0, symbol),
    ]


def _build_reduce_only_bars(symbol: str) -> list[akquant.Bar]:
    """Build 4 bars used to validate reduce-only fallback behavior."""
    day1 = _ns(datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc))
    day2 = _ns(datetime(2023, 1, 3, 15, 0, tzinfo=timezone.utc))
    day3 = _ns(datetime(2023, 1, 4, 15, 0, tzinfo=timezone.utc))
    day4 = _ns(datetime(2023, 1, 5, 15, 0, tzinfo=timezone.utc))
    return [
        akquant.Bar(day1, 10.0, 10.0, 10.0, 10.0, 1000.0, symbol),
        akquant.Bar(day2, 8.0, 8.0, 8.0, 8.0, 1000.0, symbol),
        akquant.Bar(day3, 8.0, 8.0, 8.0, 8.0, 1000.0, symbol),
        akquant.Bar(day4, 8.0, 8.0, 8.0, 8.0, 1000.0, symbol),
    ]


def _build_benchmark_data(n: int, symbol: str) -> pd.DataFrame:
    """
    Build a synthetic minute-level dataset for throughput tests.

    :param n: Number of rows.
    :param symbol: Symbol name.
    :return: DataFrame with OHLCV and symbol columns.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    returns = rng.normal(0, 0.001, n)
    price = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": np.full(n, 1000.0),
            "symbol": symbol,
        }
    )


def _build_multisymbol_benchmark_data(
    n_timestamps: int, symbols: list[str]
) -> pd.DataFrame:
    """
    Build synthetic minute-level data for multiple symbols sharing timestamps.

    :param n_timestamps: Number of distinct timestamps.
    :param symbols: Symbol list.
    :return: DataFrame sorted by timestamp then symbol.
    """
    rng = np.random.default_rng(17)
    dates = pd.date_range("2020-01-01", periods=n_timestamps, freq="min", tz="UTC")
    all_frames: list[pd.DataFrame] = []
    for index, symbol in enumerate(symbols):
        returns = rng.normal(0, 0.001, n_timestamps)
        price = (100 + index) * np.exp(np.cumsum(returns))
        all_frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": np.full(n_timestamps, 1000.0),
                    "symbol": symbol,
                }
            )
        )
    data = pd.concat(all_frames, ignore_index=True)
    return cast(
        pd.DataFrame,
        data.sort_values(["timestamp", "symbol"]).reset_index(drop=True),
    )


def test_current_close_timer_order_should_fill_at_timer_timestamp() -> None:
    """CurrentClose should fill timer order at timer timestamp, not next bar."""
    symbol = "TIMER_BUG"
    bars = [
        akquant.Bar(
            pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value,
            10.0,
            10.0,
            10.0,
            10.0,
            1000.0,
            symbol,
        ),
        akquant.Bar(
            pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value,
            11.0,
            11.0,
            11.0,
            11.0,
            1000.0,
            symbol,
        ),
    ]
    strategy = TimerCurrentCloseStrategy()
    strategy.symbol_ref = symbol

    _ = akquant.run_backtest(
        data=bars,
        strategy=strategy,
        symbols=symbol,
        fill_policy=CurrentClose(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    assert strategy.timer_timestamp is not None
    assert strategy.trade_timestamp is not None
    assert strategy.trade_timestamp == strategy.timer_timestamp
    assert strategy.trade_price == pytest.approx(10.0)


def test_daily_timer_trading_day_alignment_uses_local_calendar_day() -> None:
    """Daily timer should align with local trading days for date-only input."""
    symbol = "DAILY_TIMER_ALIGN"
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [10.0, 11.0],
            "low": [10.0, 11.0],
            "close": [10.0, 11.0],
            "volume": [1000.0, 1000.0],
            "symbol": [symbol, symbol],
        },
        index=pd.to_datetime(["2025-01-24", "2025-01-27"]),
    )
    strategy = DailyTimerBuyStrategy(symbol=symbol)

    result = akquant.run_backtest(
        data=data,
        strategy=strategy,
        symbols=symbol,
        fill_policy=CurrentClose(),
        t_plus_one=True,
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    executions_df = result.executions_df
    assert not executions_df.empty

    first_trade_time = pd.Timestamp(executions_df.iloc[0]["timestamp"]).tz_convert(
        "Asia/Shanghai"
    )
    assert first_trade_time == pd.Timestamp("2025-01-24 15:00:00", tz="Asia/Shanghai")
    assert first_trade_time.weekday() < 5

    trades_df = result.trades_df
    assert not trades_df.empty
    first_entry_time = pd.Timestamp(trades_df.iloc[0]["entry_time"]).tz_convert(
        "Asia/Shanghai"
    )
    assert first_entry_time == pd.Timestamp("2025-01-24 15:00:00", tz="Asia/Shanghai")
    assert first_entry_time.weekday() < 5


def test_order_level_current_close_daily_timer_sell_fills_same_day() -> None:
    """Order-level current-close timer sell should fill on the same trading day."""
    symbol = "DAILY_TIMER_OVERRIDE"
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [10.0, 11.0, 12.0],
            "low": [10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [1000.0, 1000.0, 1000.0],
            "symbol": [symbol, symbol, symbol],
        },
        index=pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
    )
    strategy = DailyTimerOrderLevelCurrentCloseStrategy(symbol=symbol)

    result = akquant.run_backtest(
        data=data,
        strategy=strategy,
        symbols=symbol,
        t_plus_one=True,
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    filled_sell_orders = result.orders_df[
        (result.orders_df["side"].astype(str).str.lower() == "sell")
        & (result.orders_df["status"].astype(str).str.lower() == "filled")
        & (result.orders_df["tag"].astype(str) == "timer-sell")
    ]
    assert not filled_sell_orders.empty

    sell_order = filled_sell_orders.iloc[0]
    sell_updated_at = pd.Timestamp(sell_order["updated_at"]).tz_convert("Asia/Shanghai")
    assert sell_updated_at == pd.Timestamp("2026-04-02 14:56:00", tz="Asia/Shanghai")
    assert float(sell_order["avg_price"]) == pytest.approx(11.0)

    executions_df = result.executions_df
    filled_sell_executions = executions_df[
        executions_df["side"].astype(str).str.lower() == "sell"
    ]
    assert not filled_sell_executions.empty
    sell_execution_time = pd.Timestamp(
        filled_sell_executions.iloc[0]["timestamp"]
    ).tz_convert("Asia/Shanghai")
    assert sell_execution_time == pd.Timestamp(
        "2026-04-02 14:56:00", tz="Asia/Shanghai"
    )


def test_current_close_timer_order_next_event_policy_fills_on_next_bar() -> None:
    """Timer orders should not fill at timer timestamp when policy is next_event."""
    symbol = "TIMER_BUG"
    first_ts = pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value
    second_ts = pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value
    third_ts = pd.Timestamp("2023-01-02 10:02:00", tz="Asia/Shanghai").value
    bars = [
        akquant.Bar(first_ts, 10.0, 10.0, 10.0, 10.0, 1000.0, symbol),
        akquant.Bar(second_ts, 11.0, 11.0, 11.0, 11.0, 1000.0, symbol),
        akquant.Bar(third_ts, 12.0, 12.0, 12.0, 12.0, 1000.0, symbol),
    ]
    strategy = TimerCurrentCloseStrategy()
    strategy.symbol_ref = symbol
    strategy.timer_trigger = pd.Timestamp("2023-01-02 10:01:30", tz="Asia/Shanghai")

    _ = akquant.run_backtest(
        data=bars,
        strategy=strategy,
        symbols=symbol,
        fill_policy=CurrentClose(timer_fill_timing="deferred"),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    assert strategy.timer_timestamp is not None
    if strategy.trade_timestamp is not None:
        assert strategy.trade_timestamp > strategy.timer_timestamp
    assert strategy.trade_timestamp != strategy.timer_timestamp


def test_current_close_bar_fill_unchanged_with_next_event_timer_policy() -> None:
    """Bar orders should still fill on current bar under current_close."""
    symbol = "TIMER_BUG"
    first_ts = pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value
    bars = [
        akquant.Bar(first_ts, 10.0, 10.0, 10.0, 10.0, 1000.0, symbol),
        akquant.Bar(
            pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value,
            11.0,
            11.0,
            11.0,
            11.0,
            1000.0,
            symbol,
        ),
    ]
    strategy = BarOnlyCaptureStrategy()

    _ = akquant.run_backtest(
        data=bars,
        strategy=strategy,
        symbols=symbol,
        fill_policy=CurrentClose(timer_fill_timing="deferred"),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    assert strategy.trade_timestamp == first_ts
    assert strategy.trade_price == pytest.approx(10.0)


def test_current_close_mixed_bar_timer_next_event_policy() -> None:
    """Mixed bar/timer orders should respect policy boundaries."""
    symbol = "TIMER_BUG"
    first_ts = pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value
    second_ts = pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value
    third_ts = pd.Timestamp("2023-01-02 10:02:00", tz="Asia/Shanghai").value
    bars = [
        akquant.Bar(first_ts, 10.0, 10.0, 10.0, 10.0, 1000.0, symbol),
        akquant.Bar(second_ts, 11.0, 11.0, 11.0, 11.0, 1000.0, symbol),
        akquant.Bar(third_ts, 12.0, 12.0, 12.0, 12.0, 1000.0, symbol),
    ]
    strategy = MixedBarTimerCaptureStrategy()
    strategy.symbol_ref = symbol
    strategy.timer_trigger = pd.Timestamp("2023-01-02 10:01:30", tz="Asia/Shanghai")

    _ = akquant.run_backtest(
        data=bars,
        strategy=strategy,
        symbols=symbol,
        fill_policy=CurrentClose(timer_fill_timing="deferred"),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    assert strategy.timer_submitted
    assert strategy.timer_timestamp is not None
    assert strategy.trade_timestamps
    assert strategy.trade_timestamps[0] == first_ts
    assert strategy.trade_timestamps[0] < strategy.timer_timestamp
    for ts in strategy.trade_timestamps:
        assert ts != strategy.timer_timestamp


def test_fill_policy_same_cycle_matches_legacy_parameters() -> None:
    """Fill policy should align with legacy current_close+same_cycle behavior."""
    symbol = "TIMER_BUG"
    bars = [
        akquant.Bar(
            pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value,
            10.0,
            10.0,
            10.0,
            10.0,
            1000.0,
            symbol,
        ),
        akquant.Bar(
            pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value,
            11.0,
            11.0,
            11.0,
            11.0,
            1000.0,
            symbol,
        ),
    ]
    strategy = TimerCurrentCloseStrategy()
    strategy.symbol_ref = symbol

    _ = akquant.run_backtest(
        data=bars,
        strategy=strategy,
        symbols=symbol,
        fill_policy=CurrentClose(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    assert strategy.timer_timestamp is not None
    assert strategy.trade_timestamp is not None
    assert strategy.trade_timestamp == strategy.timer_timestamp
    assert strategy.trade_price == pytest.approx(10.0)


def test_fill_policy_next_event_matches_legacy_parameters() -> None:
    """Fill policy next_event should match legacy next_event timer behavior."""
    symbol = "TIMER_BUG"
    first_ts = pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value
    second_ts = pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value
    third_ts = pd.Timestamp("2023-01-02 10:02:00", tz="Asia/Shanghai").value
    bars = [
        akquant.Bar(first_ts, 10.0, 10.0, 10.0, 10.0, 1000.0, symbol),
        akquant.Bar(second_ts, 11.0, 11.0, 11.0, 11.0, 1000.0, symbol),
        akquant.Bar(third_ts, 12.0, 12.0, 12.0, 12.0, 1000.0, symbol),
    ]
    strategy = TimerCurrentCloseStrategy()
    strategy.symbol_ref = symbol
    strategy.timer_trigger = pd.Timestamp("2023-01-02 10:01:30", tz="Asia/Shanghai")

    _ = akquant.run_backtest(
        data=bars,
        strategy=strategy,
        symbols=symbol,
        fill_policy=CurrentClose(timer_fill_timing="deferred"),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    assert strategy.timer_timestamp is not None
    assert strategy.trade_timestamp != strategy.timer_timestamp


def test_run_backtest_catalog_path_loads_data(tmp_path: Path) -> None:
    """run_backtest should load bars from explicit catalog_path when data is None."""
    symbol = "CATALOG_PATH_OK"
    catalog_root = tmp_path / "catalog"
    catalog = ParquetDataCatalog(root_path=str(catalog_root))
    data = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-02", periods=3, freq="D"),
            "open": [10.0, 11.0, 12.0],
            "high": [10.0, 11.0, 12.0],
            "low": [10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [1000.0, 1000.0, 1000.0],
            "symbol": [symbol, symbol, symbol],
        }
    ).set_index("date")
    _ = catalog.write(symbol, data)

    result = akquant.run_backtest(
        strategy=SingleBuyStrategy,
        symbols=symbol,
        catalog_path=str(catalog_root),
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
    )

    assert not result.orders_df.empty
    assert str(result.orders_df["symbol"].iloc[0]) == symbol


def test_backtest_result_top_reject_reasons_and_lot_size_category() -> None:
    """Lot-size rejects should be included in top reasons and order-size bucket."""
    symbol = "LOT_SIZE_REASON"
    bars = _build_regression_bars(symbol)

    result = akquant.run_backtest(
        data=bars,
        strategy=SingleBuyStrategy,
        symbols=symbol,
        fill_policy=CurrentClose(),
        lot_size=100,
        show_progress=False,
    )

    top_reasons = result.top_reject_reasons(top_n=5)
    assert not top_reasons.empty
    assert "reject_reason" in top_reasons.columns
    assert "count" in top_reasons.columns
    assert "ratio" in top_reasons.columns
    assert any(
        "lot size" in reason
        for reason in top_reasons["reject_reason"].fillna("").astype(str).tolist()
    )

    risk_df = result.risk_rejections_by_strategy()
    assert not risk_df.empty
    assert int(risk_df["order_size_limit_reject_count"].sum()) >= 1


def test_backtest_result_top_reject_reason_types_normalizes_dynamic_details() -> None:
    """Reject type view should merge dynamic messages into stable categories."""
    result = akquant.run_backtest(
        data=_build_regression_bars("REJECT_REASON_TYPE"),
        strategy=SingleBuyStrategy,
        symbols="REJECT_REASON_TYPE",
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
    )
    result.orders_df = pd.DataFrame(
        {
            "reject_reason": [
                (
                    "Risk: Insufficient margin at execution. "
                    "Required: 35, Available: -330445.517600"
                ),
                (
                    "Risk: Insufficient margin at execution. "
                    "Required: 35, Available: -310706.591600"
                ),
                "Order error: Risk: Daily loss 5.46% exceeds limit 5.00%",
                "Order error: Risk: Daily loss 8.79% exceeds limit 5.00%",
                "",
            ],
            "status": ["Rejected", "Rejected", "Rejected", "Rejected", "Rejected"],
        }
    )

    top_reason_types = result.top_reject_reason_types(top_n=5)

    assert not top_reason_types.empty
    assert "reject_reason_type" in top_reason_types.columns
    assert "sample_reject_reason" in top_reason_types.columns
    assert "count" in top_reason_types.columns
    assert "ratio" in top_reason_types.columns
    counts = dict(
        zip(
            top_reason_types["reject_reason_type"].astype(str),
            top_reason_types["count"].astype(int),
        )
    )
    assert counts["Insufficient Margin"] == 2
    assert counts["Daily Loss Limit"] == 2
    sample_detail = top_reason_types.loc[
        top_reason_types["reject_reason_type"].astype(str) == "Insufficient Margin",
        "sample_reject_reason",
    ].iloc[0]
    assert "Required: 35" in str(sample_detail)


def test_policy_resolver_next_close_same_cycle_sets_timer_same_cycle() -> None:
    """Close + bar_offset=1 + same_cycle should resolve timer policy."""
    resolved = backtest_engine._resolve_execution_policy(
        execution_mode="next_open",
        timer_execution_policy="next_event",
        fill_policy={"price_basis": "close", "bar_offset": 1, "temporal": "same_cycle"},
        logger=backtest_engine.get_logger(),
    )
    assert resolved.bar_offset == 1
    assert resolved.temporal == "same_cycle"
    assert resolved.price_basis == "close"
    assert resolved.source == "fill_policy"


def test_policy_resolver_next_close_next_event_sets_timer_next_event() -> None:
    """Close + bar_offset=1 + next_event should resolve timer policy."""
    resolved = backtest_engine._resolve_execution_policy(
        execution_mode="next_open",
        timer_execution_policy="same_cycle",
        fill_policy={"price_basis": "close", "bar_offset": 1, "temporal": "next_event"},
        logger=backtest_engine.get_logger(),
    )
    assert resolved.bar_offset == 1
    assert resolved.temporal == "next_event"
    assert resolved.price_basis == "close"
    assert resolved.source == "fill_policy"


@pytest.mark.parametrize(
    (
        "execution_mode",
        "timer_execution_policy",
        "fill_policy",
        "expected_basis",
        "expected_temporal",
        "expected_source",
    ),
    [
        (
            "next_open",
            "same_cycle",
            None,
            "open",
            "same_cycle",
            "legacy",
        ),
        (
            "current_close",
            "next_event",
            None,
            "close",
            "next_event",
            "legacy",
        ),
        (
            "next_close",
            "same_cycle",
            None,
            "close",
            "same_cycle",
            "legacy",
        ),
        (
            "next_open",
            "same_cycle",
            {"price_basis": "ohlc4", "bar_offset": 1, "temporal": "next_event"},
            "ohlc4",
            "next_event",
            "fill_policy",
        ),
        (
            "current_close",
            "next_event",
            {"price_basis": "hl2", "bar_offset": 1, "temporal": "same_cycle"},
            "hl2",
            "same_cycle",
            "fill_policy",
        ),
    ],
)
def test_policy_resolver_matrix(
    execution_mode: str,
    timer_execution_policy: str,
    fill_policy: Any,
    expected_basis: str,
    expected_temporal: str,
    expected_source: str,
) -> None:
    """Resolver matrix should map basis/temporal and source deterministically."""
    resolved = backtest_engine._resolve_execution_policy(
        execution_mode=execution_mode,
        timer_execution_policy=timer_execution_policy,
        fill_policy=fill_policy,
        logger=backtest_engine.get_logger(),
    )
    assert resolved.price_basis == expected_basis
    assert resolved.temporal == expected_temporal
    assert resolved.source == expected_source


def test_run_backtest_rejects_legacy_execution_mode_without_fill_policy() -> None:
    """run_backtest should reject legacy execution_mode."""
    symbol = "LEGACY_EXEC_MODE"
    bars = _build_benchmark_data(4, symbol)
    with pytest.raises(
        ValueError,
        match="run_backtest no longer accepts execution_mode/timer_execution_policy",
    ):
        legacy_kwargs: dict[str, Any] = {"execution_mode": "current_close"}
        _ = akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols=symbol,
            show_progress=False,
            **legacy_kwargs,
        )


def test_run_backtest_rejects_legacy_timer_policy_without_fill_policy() -> None:
    """run_backtest should reject legacy timer policy."""
    symbol = "LEGACY_TIMER_POLICY"
    bars = _build_benchmark_data(4, symbol)
    with pytest.raises(
        ValueError,
        match="run_backtest no longer accepts execution_mode/timer_execution_policy",
    ):
        legacy_kwargs: dict[str, Any] = {"timer_execution_policy": "next_event"}
        _ = akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols=symbol,
            show_progress=False,
            **legacy_kwargs,
        )


def test_run_backtest_rejects_legacy_execution_params() -> None:
    """run_backtest should reject legacy execution params."""
    symbol = "LEGACY_COMPAT_OFF"
    bars = _build_benchmark_data(4, symbol)
    with pytest.raises(
        ValueError,
        match="run_backtest no longer accepts execution_mode/timer_execution_policy",
    ):
        legacy_kwargs: dict[str, Any] = {"execution_mode": "current_close"}
        _ = akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols=symbol,
            show_progress=False,
            **legacy_kwargs,
        )


def test_run_backtest_rejects_non_bool_legacy_execution_policy_compat() -> None:
    """legacy_execution_policy_compat should be removed in run_backtest."""
    symbol = "LEGACY_COMPAT_TYPE"
    bars = _build_benchmark_data(4, symbol)
    with pytest.raises(
        TypeError, match="legacy_execution_policy_compat is no longer supported"
    ):
        compat_kwargs: dict[str, Any] = {"legacy_execution_policy_compat": "false"}
        _ = akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols=symbol,
            show_progress=False,
            **compat_kwargs,
        )


def test_run_backtest_rejects_legacy_by_default() -> None:
    """Default behavior should reject legacy execution params."""
    symbol = "LEGACY_DEFAULT_OFF"
    bars = _build_benchmark_data(4, symbol)
    with pytest.raises(
        ValueError,
        match="run_backtest no longer accepts execution_mode/timer_execution_policy",
    ):
        legacy_kwargs: dict[str, Any] = {"execution_mode": "current_close"}
        _ = akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols=symbol,
            show_progress=False,
            **legacy_kwargs,
        )


def test_run_backtest_rejects_invalid_legacy_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy env var should no longer affect fill_policy execution."""
    monkeypatch.setenv("AKQ_LEGACY_EXECUTION_POLICY_COMPAT", "not_bool")
    symbol = "LEGACY_ENV_BAD"
    bars = _build_benchmark_data(2, symbol)
    result = akquant.run_backtest(
        data=bars,
        strategy=NoopStrategy,
        symbols=symbol,
        fill_policy=CurrentClose(),
        show_progress=False,
    )
    assert result.resolved_execution_policy is not None
    assert result.resolved_execution_policy["source"] == "fill_policy"


def test_run_backtest_explicit_compat_overrides_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removed compat flag should raise even when env is set."""
    monkeypatch.setenv("AKQ_LEGACY_EXECUTION_POLICY_COMPAT", "false")
    symbol = "LEGACY_ENV_OVERRIDE"
    bars = _build_benchmark_data(4, symbol)
    with pytest.raises(
        TypeError, match="legacy_execution_policy_compat is no longer supported"
    ):
        compat_kwargs: dict[str, Any] = {"legacy_execution_policy_compat": True}
        _ = akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols=symbol,
            show_progress=False,
            **compat_kwargs,
        )


def test_fill_policy_next_close_maps_to_next_bar_close() -> None:
    """fill_policy close + bar_offset=1 should map to next bar close."""
    symbol = "NEXT_CLOSE_BASIS"
    first_ts = pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value
    second_ts = pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value
    bars = [
        akquant.Bar(first_ts, 9.0, 9.5, 8.5, 9.2, 1000.0, symbol),
        akquant.Bar(second_ts, 10.0, 15.0, 9.0, 12.0, 1000.0, symbol),
    ]
    strategy = BarOnlyCaptureStrategy()

    result = akquant.run_backtest(
        data=bars,
        strategy=strategy,
        symbols=symbol,
        fill_policy=NextClose(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    assert strategy.trade_timestamp == second_ts
    assert strategy.trade_price == pytest.approx(12.0)
    assert result.resolved_execution_policy is not None
    assert result.resolved_execution_policy["price_basis"] == "close"
    assert int(result.resolved_execution_policy["bar_offset"]) == 1
    assert result.resolved_execution_policy["temporal"] == "same_cycle"
    assert result.resolved_execution_policy["source"] == "fill_policy"


def test_execution_mode_next_close_string_maps_to_next_bar_close() -> None:
    """execution_mode should be removed and require fill_policy."""
    symbol = "NEXT_CLOSE_MODE"
    first_ts = pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value
    second_ts = pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value
    bars = [
        akquant.Bar(first_ts, 9.0, 9.5, 8.5, 9.2, 1000.0, symbol),
        akquant.Bar(second_ts, 10.0, 15.0, 9.0, 12.0, 1000.0, symbol),
    ]
    strategy = BarOnlyCaptureStrategy()

    with pytest.raises(
        ValueError,
        match="run_backtest no longer accepts execution_mode/timer_execution_policy",
    ):
        legacy_kwargs: dict[str, Any] = {"execution_mode": "next_close"}
        _ = akquant.run_backtest(
            data=bars,
            strategy=strategy,
            symbols=symbol,
            initial_cash=100000.0,
            commission_rate=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            min_commission=0.0,
            lot_size=1,
            show_progress=False,
            **legacy_kwargs,
        )


def test_fill_policy_ohlc4_maps_to_next_average() -> None:
    """fill_policy price_basis=ohlc4 should map to NextAverage pricing."""
    symbol = "TIMER_BUG"
    first_ts = pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value
    second_ts = pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value
    bars = [
        akquant.Bar(first_ts, 9.0, 9.5, 8.5, 9.2, 1000.0, symbol),
        akquant.Bar(second_ts, 10.0, 15.0, 9.0, 12.0, 1000.0, symbol),
    ]
    strategy = BarOnlyCaptureStrategy()

    _ = akquant.run_backtest(
        data=bars,
        strategy=strategy,
        symbols=symbol,
        fill_policy=NextAverage(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    assert strategy.trade_timestamp == second_ts
    assert strategy.trade_price == pytest.approx(11.5)


def test_fill_policy_hl2_maps_to_next_high_low_mid() -> None:
    """fill_policy price_basis=hl2 should map to NextHighLowMid pricing."""
    symbol = "TIMER_BUG"
    first_ts = pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value
    second_ts = pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value
    bars = [
        akquant.Bar(first_ts, 9.0, 9.5, 8.5, 9.2, 1000.0, symbol),
        akquant.Bar(second_ts, 10.0, 15.0, 9.0, 12.0, 1000.0, symbol),
    ]
    strategy = BarOnlyCaptureStrategy()

    _ = akquant.run_backtest(
        data=bars,
        strategy=strategy,
        symbols=symbol,
        fill_policy=NextHighLowMid(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    assert strategy.trade_timestamp == second_ts
    assert strategy.trade_price == pytest.approx(12.0)


def test_run_backtest_accepts_data_feed_adapter() -> None:
    """run_backtest should accept objects implementing DataFeedAdapter.load."""

    class InMemoryAdapter:
        """Simple in-memory adapter for testing."""

        name = "memory"

        def __init__(self, frame: pd.DataFrame) -> None:
            """Store source frame."""
            self.frame = frame
            self.requested_symbols: list[str] = []

        def load(self, request: Any) -> pd.DataFrame:
            """Return filtered frame for requested symbol."""
            self.requested_symbols.append(str(request.symbol))
            data = self.frame[self.frame["symbol"] == str(request.symbol)].copy()
            if request.start_time is not None:
                data = data[data["timestamp"] >= request.start_time]
            if request.end_time is not None:
                data = data[data["timestamp"] <= request.end_time]
            return cast(pd.DataFrame, data)

    symbol = "ADAPTER"
    data = _build_benchmark_data(10, symbol)
    adapter = InMemoryAdapter(data)

    result = akquant.run_backtest(
        data=adapter,
        strategy=SingleBuyStrategy,
        symbols=symbol,
        fill_policy=CurrentClose(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    assert adapter.requested_symbols == [symbol]
    assert not result.orders_df.empty
    assert set(result.orders_df["symbol"].astype(str)) == {symbol}


def test_run_backtest_accepts_symbols_alias_for_single_symbol() -> None:
    """run_backtest should accept symbols as the primary symbol argument."""
    data = _build_benchmark_data(6, "ALIAS_SYMBOL")
    result = akquant.run_backtest(
        data=data,
        strategy=SingleBuyStrategy,
        symbols="ALIAS_SYMBOL",
        fill_policy=CurrentClose(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )
    assert not result.orders_df.empty
    assert set(result.orders_df["symbol"].astype(str)) == {"ALIAS_SYMBOL"}


def test_run_backtest_rejects_legacy_symbol_keyword_alias() -> None:
    """run_backtest should reject removed symbol keyword alias."""
    data = _build_benchmark_data(6, "DEPREC_SYMBOL")
    with pytest.raises(ValueError, match="no longer accepts `symbol`"):
        akquant.run_backtest(
            data=data,
            strategy=SingleBuyStrategy,
            symbol="DEPREC_SYMBOL",
            fill_policy=CurrentClose(),
            initial_cash=100000.0,
            commission_rate=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            min_commission=0.0,
            lot_size=1,
            show_progress=False,
        )


def test_run_backtest_uses_symbols_without_deprecation_warnings() -> None:
    """run_backtest should not emit deprecation warning for symbols argument."""
    data = _build_benchmark_data(6, "NO_WARN_SYMBOLS")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = akquant.run_backtest(
            data=data,
            strategy=SingleBuyStrategy,
            symbols="NO_WARN_SYMBOLS",
            fill_policy=CurrentClose(),
            initial_cash=100000.0,
            commission_rate=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            min_commission=0.0,
            lot_size=1,
            show_progress=False,
        )
    assert not result.orders_df.empty
    assert [
        warning
        for warning in record
        if issubclass(warning.category, DeprecationWarning)
    ] == []


def test_run_backtest_rejects_conflicting_symbol_and_symbols() -> None:
    """run_backtest should reject conflicting symbol and symbols inputs."""
    data = _build_benchmark_data(4, "AAA")
    with pytest.raises(ValueError, match="no longer accepts `symbol`"):
        akquant.run_backtest(
            data=data,
            strategy=SingleBuyStrategy,
            symbol="AAA",
            symbols=["BBB"],
            show_progress=False,
        )


def test_run_backtest_dataframe_multisymbol_preserves_bar_symbol() -> None:
    """Keep bar.symbol aligned with per-row symbol values in DataFrame mode."""

    class CollectSymbolsStrategy(akquant.Strategy):
        """Collect symbols observed in on_bar callbacks."""

        def __init__(self) -> None:
            """Initialize the collected symbol container."""
            super().__init__()
            self.seen_symbols: list[str] = []

        def on_bar(self, bar: akquant.Bar) -> None:
            """Record callback symbol for later assertions."""
            self.seen_symbols.append(str(bar.symbol))

    rows = [
        {
            "timestamp": "2024-01-02",
            "symbol": "IF2401.CFX",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1000.0,
        },
        {
            "timestamp": "2024-01-02",
            "symbol": "IF2402.CFX",
            "open": 20.0,
            "high": 21.0,
            "low": 19.0,
            "close": 20.5,
            "volume": 1000.0,
        },
        {
            "timestamp": "2024-01-03",
            "symbol": "IF2401.CFX",
            "open": 11.0,
            "high": 12.0,
            "low": 10.0,
            "close": 11.5,
            "volume": 1000.0,
        },
        {
            "timestamp": "2024-01-03",
            "symbol": "IF2402.CFX",
            "open": 21.0,
            "high": 22.0,
            "low": 20.0,
            "close": 21.5,
            "volume": 1000.0,
        },
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    result = akquant.run_backtest(
        data=df,
        strategy=CollectSymbolsStrategy,
        symbols=["IF2401.CFX", "IF2402.CFX"],
        start_time="2024-01-02",
        end_time="2024-01-03",
        fill_policy=CurrentClose(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    strategy = cast(CollectSymbolsStrategy, result.strategy)
    assert len(strategy.seen_symbols) == len(df)
    assert set(strategy.seen_symbols) == {"IF2401.CFX", "IF2402.CFX"}
    assert strategy.seen_symbols.count("IF2401.CFX") == 2
    assert strategy.seen_symbols.count("IF2402.CFX") == 2


def test_run_backtest_naive_dataframe_boundaries_follow_configured_timezone() -> None:
    """Naive DataFrame windows should interpret start/end in configured timezone."""

    class CaptureBarTimestampsStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.seen_timestamps: list[str] = []

        def on_bar(self, bar: akquant.Bar) -> None:
            self.seen_timestamps.append(str(bar.timestamp_iso))

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02 20:00:00", "2024-01-02 21:00:00", "2024-01-02 22:00:00"]
            ),
            "symbol": ["BOUNDARY_DF"] * 3,
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.2, 11.2, 12.2],
            "volume": [1000.0, 1000.0, 1000.0],
        }
    ).set_index("timestamp")

    result = akquant.run_backtest(
        data=frame,
        strategy=CaptureBarTimestampsStrategy,
        symbols=["BOUNDARY_DF"],
        start_time="2024-01-02 13:00:00",
        end_time="2024-01-02 14:00:00",
        timezone="UTC",
        fill_policy=CurrentClose(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    strategy = cast(CaptureBarTimestampsStrategy, result.strategy)
    assert strategy.seen_timestamps == [
        "2024-01-02T13:00:00Z",
        "2024-01-02T14:00:00Z",
    ]


def test_run_backtest_dataframe_multisymbol_row_order_keeps_metrics_stable() -> None:
    """Metrics should not drift when same-timestamp symbol row order changes."""

    class BuyOncePerSymbolStrategy(akquant.Strategy):
        """Buy each symbol once and hold so equity depends on multi-symbol marking."""

        def __init__(self) -> None:
            super().__init__()
            self.bought_symbols: set[str] = set()

        def on_bar(self, bar: akquant.Bar) -> None:
            symbol = str(bar.symbol)
            if symbol in self.bought_symbols:
                return
            self.buy(symbol=symbol, quantity=1)
            self.bought_symbols.add(symbol)

    rows = [
        ("2024-01-02", "AAA", 10.0),
        ("2024-01-02", "BBB", 20.0),
        ("2024-01-03", "AAA", 11.0),
        ("2024-01-03", "BBB", 19.0),
        ("2024-01-04", "AAA", 12.0),
        ("2024-01-04", "BBB", 21.0),
        ("2024-01-05", "AAA", 13.0),
        ("2024-01-05", "BBB", 23.0),
    ]
    frame = pd.DataFrame(rows, columns=["timestamp", "symbol", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["open"] = frame["close"]
    frame["high"] = frame["close"]
    frame["low"] = frame["close"]
    frame["volume"] = 1000.0

    ascending = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    descending = frame.sort_values(
        ["timestamp", "symbol"], ascending=[True, False]
    ).reset_index(drop=True)

    common_args: dict[str, Any] = dict(
        strategy=BuyOncePerSymbolStrategy,
        symbols=["AAA", "BBB"],
        start_time="2024-01-02",
        end_time="2024-01-05",
        fill_policy=CurrentClose(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    result_ascending = akquant.run_backtest(data=ascending, **common_args)
    result_descending = akquant.run_backtest(data=descending, **common_args)

    assert len(result_ascending.orders_df) == len(result_descending.orders_df) == 2
    assert result_ascending.metrics.end_market_value == pytest.approx(
        result_descending.metrics.end_market_value, rel=1e-12
    )
    assert result_ascending.metrics.total_return == pytest.approx(
        result_descending.metrics.total_return, rel=1e-12
    )
    assert result_ascending.metrics.sharpe_ratio == pytest.approx(
        result_descending.metrics.sharpe_ratio, rel=1e-12
    )
    assert result_ascending.metrics.volatility == pytest.approx(
        result_descending.metrics.volatility, rel=1e-12
    )
    assert float(result_ascending.equity_curve.iloc[-1]) == pytest.approx(
        float(result_descending.equity_curve.iloc[-1]), rel=1e-12
    )


def test_run_backtest_list_boundaries_follow_configured_timezone() -> None:
    """Naive boundary strings should filter list[Bar] inputs in configured timezone."""

    class CaptureBarTimestampsStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.seen_timestamps: list[str] = []

        def on_bar(self, bar: akquant.Bar) -> None:
            self.seen_timestamps.append(str(bar.timestamp_iso))

    bars = [
        akquant.Bar(
            timestamp=pd.Timestamp("2024-01-02 20:00:00", tz="Asia/Shanghai").value,
            symbol="BOUNDARY_LIST",
            open=10.0,
            high=10.5,
            low=9.5,
            close=10.2,
            volume=1000.0,
        ),
        akquant.Bar(
            timestamp=pd.Timestamp("2024-01-02 21:00:00", tz="Asia/Shanghai").value,
            symbol="BOUNDARY_LIST",
            open=11.0,
            high=11.5,
            low=10.5,
            close=11.2,
            volume=1000.0,
        ),
        akquant.Bar(
            timestamp=pd.Timestamp("2024-01-02 22:00:00", tz="Asia/Shanghai").value,
            symbol="BOUNDARY_LIST",
            open=12.0,
            high=12.5,
            low=11.5,
            close=12.2,
            volume=1000.0,
        ),
    ]

    result = akquant.run_backtest(
        data=bars,
        strategy=CaptureBarTimestampsStrategy,
        symbols=["BOUNDARY_LIST"],
        start_time="2024-01-02 13:00:00",
        end_time="2024-01-02 14:00:00",
        timezone="UTC",
        fill_policy=CurrentClose(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    strategy = cast(CaptureBarTimestampsStrategy, result.strategy)
    assert strategy.seen_timestamps == [
        "2024-01-02T13:00:00Z",
        "2024-01-02T14:00:00Z",
    ]


def test_engine_run_empty() -> None:
    """Test running engine with no data."""
    engine = akquant.Engine()
    strategy = DummyStrategy()
    engine.run(strategy, show_progress=False)
    result = engine.get_results()

    # Result should indicate no trades, 0 return
    # result.metrics.total_return ? Or result.total_return?
    # BacktestResult has 'metrics' and 'trade_metrics' fields.
    assert result.trade_metrics.total_closed_trades == 0
    assert abs(result.metrics.total_return - 0.0) < 1e-9


def test_engine_set_cash() -> None:
    """Test setting initial cash."""
    engine = akquant.Engine()
    engine.set_cash(50000.0)
    assert engine.portfolio.cash == 50000.0


def test_engine_single_strategy_slot_defaults_and_update() -> None:
    """Engine should keep single-slot metadata consistent in phase 1."""
    engine = akquant.Engine()
    if not hasattr(engine, "get_strategy_slot_ids"):
        pytest.skip("Engine binary does not expose slot metadata methods")
    slot_ids = cast(list[str], cast(Any, engine).get_strategy_slot_ids())
    assert slot_ids == ["_default"]
    assert cast(int, cast(Any, engine).get_active_strategy_slot()) == 0

    cast(Any, engine).set_default_strategy_id("alpha_slot")
    updated_slot_ids = cast(list[str], cast(Any, engine).get_strategy_slot_ids())
    assert updated_slot_ids == ["alpha_slot"]
    assert cast(str, cast(Any, engine).get_default_strategy_id()) == "alpha_slot"


def test_engine_strategy_slot_configuration_api() -> None:
    """Engine should support configuring multi-slot metadata."""
    engine = akquant.Engine()
    if not hasattr(engine, "set_strategy_slots"):
        pytest.skip("Engine binary does not expose slot configuration methods")

    cast(Any, engine).set_strategy_slots(["alpha", "beta"])
    slot_ids = cast(list[str], cast(Any, engine).get_strategy_slot_ids())
    assert slot_ids == ["alpha", "beta"]
    assert cast(str, cast(Any, engine).get_default_strategy_id()) == "alpha"


def test_engine_run_with_configured_slot_strategy() -> None:
    """Engine should run when secondary slot strategy is configured."""
    engine = akquant.Engine()
    if not hasattr(engine, "set_strategy_for_slot"):
        pytest.skip("Engine binary does not expose slot strategy methods")

    symbol = "SLOT_RUN"
    engine.use_simple_market(0.0)
    engine.set_force_session_continuous(True)
    cast(Any, engine).set_fill_mode(akquant.ExecutionMode.CurrentClose, "same_cycle")
    engine.set_cash(100000.0)
    engine.set_stock_fee_rules(0.0, 0.0, 0.0, 0.0)

    instr = akquant.Instrument(
        symbol=symbol,
        asset_type=akquant.AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1.0,
    )
    engine.add_instrument(instr)
    engine.add_bars(_build_regression_bars(symbol))

    cast(Any, engine).set_strategy_slots(["slot_0", "slot_1"])
    cast(Any, engine).set_strategy_for_slot(1, NoopStrategy())
    engine.run(NoopStrategy(), show_progress=False)
    result = engine.get_results()
    assert result.metrics.initial_market_value == pytest.approx(100000.0, rel=1e-9)


def test_backtest_regression_baseline() -> None:
    """Verify baseline equity curve and trade sequence."""
    symbol = "REGRESS"
    engine = akquant.Engine()
    engine.use_simple_market(0.0)
    engine.set_force_session_continuous(True)
    cast(Any, engine).set_fill_mode(akquant.ExecutionMode.CurrentClose, "same_cycle")
    engine.set_cash(100000.0)
    engine.set_stock_fee_rules(0.0, 0.0, 0.0, 0.0)
    engine.set_t_plus_one(False)

    instr = akquant.Instrument(
        symbol=symbol,
        asset_type=akquant.AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        option_type=None,
        strike_price=None,
        expiry_date=None,
        lot_size=1.0,
    )
    engine.add_instrument(instr)

    bars = _build_regression_bars(symbol)
    engine.add_bars(bars)

    strategy = RegressionStrategy()
    engine.run(strategy, show_progress=False)
    result = engine.get_results()

    day1 = bars[0].timestamp
    day2 = bars[1].timestamp
    day3 = bars[2].timestamp
    expected_equity = [
        (day1, 100000.0),
        (day2, 100020.0),
        (day3, 100010.0),
    ]
    assert len(result.equity_curve) == len(expected_equity)
    for (ts, val), (exp_ts, exp_val) in zip(result.equity_curve, expected_equity):
        assert ts == exp_ts
        assert val == pytest.approx(exp_val, rel=1e-9)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.symbol == symbol
    assert trade.entry_time == day1
    assert trade.exit_time == day3
    assert trade.entry_price == pytest.approx(10.0, rel=1e-9)
    assert trade.exit_price == pytest.approx(11.0, rel=1e-9)
    assert trade.quantity == pytest.approx(10.0, rel=1e-9)
    assert trade.side == "Long"
    assert trade.pnl == pytest.approx(10.0, rel=1e-9)
    assert trade.net_pnl == pytest.approx(10.0, rel=1e-9)
    assert trade.return_pct == pytest.approx(10.0, rel=1e-9)
    assert trade.commission == pytest.approx(0.0, rel=1e-9)
    assert trade.duration_bars == 2


def test_metrics_df_exposes_display_friendly_trade_counts() -> None:
    """metrics_df should separate closed trades, executions, and open positions."""
    symbol = "DISPLAY_COUNTS"
    engine = akquant.Engine()
    engine.use_simple_market(0.0)
    engine.set_force_session_continuous(True)
    cast(Any, engine).set_fill_mode(akquant.ExecutionMode.CurrentClose, "same_cycle")
    engine.set_cash(100000.0)
    engine.set_stock_fee_rules(0.0, 0.0, 0.0, 0.0)
    engine.set_t_plus_one(False)

    instr = akquant.Instrument(
        symbol=symbol,
        asset_type=akquant.AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        option_type=None,
        strike_price=None,
        expiry_date=None,
        lot_size=1.0,
    )
    engine.add_instrument(instr)

    bars = [
        akquant.Bar(
            _ns(datetime(2023, 2, 1, 15, 0, tzinfo=timezone.utc)),
            10.0,
            10.0,
            10.0,
            10.0,
            1000.0,
            symbol,
        ),
        akquant.Bar(
            _ns(datetime(2023, 2, 2, 15, 0, tzinfo=timezone.utc)),
            11.0,
            11.0,
            11.0,
            11.0,
            1000.0,
            symbol,
        ),
        akquant.Bar(
            _ns(datetime(2023, 2, 3, 15, 0, tzinfo=timezone.utc)),
            12.0,
            12.0,
            12.0,
            12.0,
            1000.0,
            symbol,
        ),
        akquant.Bar(
            _ns(datetime(2023, 2, 4, 15, 0, tzinfo=timezone.utc)),
            13.0,
            13.0,
            13.0,
            13.0,
            1000.0,
            symbol,
        ),
    ]
    engine.add_bars(bars)

    engine.run(BuyBuySellBuyStrategy(), show_progress=False)
    result = engine.get_results()
    metrics_df = result.metrics_df

    assert float(metrics_df.loc["closed_trade_count", "value"]) == pytest.approx(1.0)
    assert float(metrics_df.loc["execution_count", "value"]) == pytest.approx(4.0)
    assert float(metrics_df.loc["open_position_count", "value"]) == pytest.approx(1.0)
    assert len(result.trades) == 1
    assert len(result.executions) == 4


def test_engine_set_fill_policy_roundtrip() -> None:
    """Engine fill policy API should expose three-axis tuple."""
    engine = akquant.Engine()
    if not hasattr(engine, "set_fill_mode"):
        pytest.skip("Engine binary does not expose fill policy methods")
    cast(Any, engine).set_fill_mode(akquant.ExecutionMode.NextClose, "next_event")
    basis, bar_offset, temporal = cast(
        tuple[str, int, str], cast(Any, engine).get_fill_policy()
    )
    assert basis == "close"
    assert int(bar_offset) == 1
    assert temporal == "next_event"


def test_position_helper_exposes_runtime_entry_price() -> None:
    """Position helper should expose weighted-average runtime entry price."""
    symbol = "POS_HELPER"
    engine = akquant.Engine()
    engine.use_simple_market(0.0)
    engine.set_force_session_continuous(True)
    cast(Any, engine).set_fill_mode(akquant.ExecutionMode.CurrentClose, "same_cycle")
    engine.set_cash(100000.0)
    engine.set_stock_fee_rules(0.0, 0.0, 0.0, 0.0)
    engine.set_t_plus_one(False)

    instr = akquant.Instrument(
        symbol=symbol,
        asset_type=akquant.AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        option_type=None,
        strike_price=None,
        expiry_date=None,
        lot_size=1.0,
    )
    engine.add_instrument(instr)
    engine.add_bars(_build_position_entry_price_bars(symbol))

    strategy = PositionEntryPriceCaptureStrategy()
    engine.run(strategy, show_progress=False)

    assert len(strategy.snapshots) == 5
    assert strategy.snapshots[0]["size"] == pytest.approx(0.0, rel=1e-9)
    assert strategy.snapshots[0]["entry_price"] == pytest.approx(0.0, rel=1e-9)
    assert strategy.snapshots[1]["size"] == pytest.approx(10.0, rel=1e-9)
    assert strategy.snapshots[1]["available"] == pytest.approx(10.0, rel=1e-9)
    assert strategy.snapshots[1]["entry_price"] == pytest.approx(10.0, rel=1e-9)
    assert strategy.snapshots[1]["avg_price"] == pytest.approx(10.0, rel=1e-9)
    assert strategy.snapshots[1]["ctx_entry_price"] == pytest.approx(10.0, rel=1e-9)
    assert strategy.snapshots[2]["size"] == pytest.approx(20.0, rel=1e-9)
    assert strategy.snapshots[2]["entry_price"] == pytest.approx(11.0, rel=1e-9)
    assert strategy.snapshots[3]["size"] == pytest.approx(15.0, rel=1e-9)
    assert strategy.snapshots[3]["entry_price"] == pytest.approx(34.0 / 3.0, rel=1e-9)
    assert strategy.snapshots[3]["avg_price"] == pytest.approx(34.0 / 3.0, rel=1e-9)
    assert strategy.snapshots[3]["ctx_entry_price"] == pytest.approx(
        34.0 / 3.0, rel=1e-9
    )
    assert strategy.snapshots[4]["size"] == pytest.approx(0.0, rel=1e-9)
    assert strategy.snapshots[4]["available"] == pytest.approx(0.0, rel=1e-9)
    assert strategy.snapshots[4]["entry_price"] == pytest.approx(0.0, rel=1e-9)
    assert "entry_price=0.0" in cast(str, strategy.snapshots[4]["repr"])


def test_backtest_performance_baseline() -> None:
    """Verify minimum throughput for a no-op strategy."""
    data = _build_benchmark_data(n=3000, symbol="PERF")
    t0 = time.perf_counter()
    result = akquant.run_backtest(
        data=data,
        strategy=NoopStrategy,
        symbols="PERF",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
    )
    duration = time.perf_counter() - t0
    throughput = len(data) / duration if duration > 0 else 0.0
    assert throughput >= 200.0
    assert result.metrics.initial_market_value == pytest.approx(100000.0, rel=1e-9)


def test_run_backtest_engine_oco_avoids_same_batch_double_fill() -> None:
    """Engine OCO should avoid double fill when both legs are matchable in one bar."""

    class OcoSameBarStrategy(akquant.Strategy):
        """Submit two same-bar matchable orders and bind them as OCO."""

        def __init__(self) -> None:
            """Initialize submit-once state."""
            super().__init__()
            self.submitted = False

        def on_bar(self, bar: akquant.Bar) -> None:
            """Submit OCO legs on first bar."""
            if self.submitted:
                return
            first = self.buy(symbol=bar.symbol, quantity=1, price=bar.close)
            second = self.buy(symbol=bar.symbol, quantity=1, price=bar.close)
            self.place_oco(first, second)
            self.submitted = True

    symbol = "OCO_SAME_BAR"
    bars = [
        akquant.Bar(
            _ns(datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)),
            10.0,
            10.0,
            10.0,
            10.0,
            1000.0,
            symbol,
        )
    ]
    result = akquant.run_backtest(
        data=bars,
        strategy=OcoSameBarStrategy,
        symbols=symbol,
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
    )

    assert len(result.orders_df) == 2
    total_filled_qty = float(result.orders_df["filled_quantity"].sum())
    assert total_filled_qty == pytest.approx(1.0, rel=1e-9)
    filled_quantities = sorted(
        float(v) for v in result.orders_df["filled_quantity"].tolist()
    )
    assert filled_quantities == [0.0, 1.0]


def test_run_backtest_engine_bracket_activates_exit_orders() -> None:
    """Engine bracket plan should activate exit orders after entry fill."""

    class BracketEngineStrategy(akquant.Strategy):
        """Submit one bracket and rely on engine-side activation."""

        def __init__(self) -> None:
            """Initialize one-shot state."""
            super().__init__()
            self.submitted = False

        def on_bar(self, bar: akquant.Bar) -> None:
            """Submit bracket on first bar only."""
            if self.submitted:
                return
            self.place_bracket(
                symbol=bar.symbol,
                quantity=1.0,
                entry_price=100.0,
                stop_trigger_price=95.0,
                take_profit_price=110.0,
                entry_tag="entry",
                stop_tag="stop",
                take_profit_tag="take",
            )
            self.submitted = True

    symbol = "BRACKET_ENGINE"
    bars = [
        akquant.Bar(
            _ns(datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)),
            100.0,
            100.0,
            100.0,
            100.0,
            1000.0,
            symbol,
        ),
        akquant.Bar(
            _ns(datetime(2023, 1, 3, 15, 0, tzinfo=timezone.utc)),
            110.0,
            111.0,
            100.0,
            110.0,
            1000.0,
            symbol,
        ),
    ]
    result = akquant.run_backtest(
        data=bars,
        strategy=BracketEngineStrategy,
        symbols=symbol,
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
    )

    tags = set(result.orders_df["tag"].astype(str))
    assert {"entry", "stop", "take"}.issubset(tags)


def test_run_backtest_on_event_emits_ordered_events() -> None:
    """Stream API should emit ordered lifecycle events."""
    data = _build_benchmark_data(n=20, symbol="STREAM")
    events: list[akquant.BacktestStreamEvent] = []

    def on_event(event: akquant.BacktestStreamEvent) -> None:
        events.append(event)

    result = akquant.run_backtest(
        data=data,
        strategy=NoopStrategy,
        symbols="STREAM",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        on_event=on_event,
        stream_progress_interval=5,
        stream_equity_interval=7,
        stream_batch_size=8,
        stream_max_buffer=64,
    )

    assert events
    assert events[0]["event_type"] == "started"
    assert events[-1]["event_type"] == "finished"
    seq_values = [event["seq"] for event in events]
    assert seq_values == sorted(seq_values)
    progress_count = sum(1 for event in events if event.get("event_type") == "progress")
    equity_count = sum(1 for event in events if event.get("event_type") == "equity")
    assert 0 < progress_count < 10
    assert 0 < equity_count < 10
    assert result.metrics.initial_market_value == pytest.approx(100000.0, rel=1e-9)


def test_run_backtest_progress_total_uses_unique_timestamps_for_multisymbol() -> None:
    """Progress events should report total as unique timestamps in multi-symbol runs."""
    symbols = ["STREAM_A", "STREAM_B", "STREAM_C"]
    data = _build_multisymbol_benchmark_data(n_timestamps=30, symbols=symbols)
    events: list[akquant.BacktestStreamEvent] = []

    _ = akquant.run_backtest(
        data=data,
        strategy=NoopStrategy,
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        on_event=events.append,
        stream_progress_interval=3,
        stream_equity_interval=7,
        stream_batch_size=16,
        stream_max_buffer=128,
    )

    progress_events = [
        event for event in events if event.get("event_type") == "progress"
    ]
    assert progress_events
    expected_total = int(data["timestamp"].nunique())
    expected_rows = int(len(data))
    assert expected_total < expected_rows
    totals = {
        int(cast(dict[str, Any], event.get("payload", {})).get("total", "0"))
        for event in progress_events
    }
    assert totals == {expected_total}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stream_progress_interval": 0},
        {"stream_equity_interval": 0},
        {"stream_batch_size": 0},
        {"stream_max_buffer": 0},
    ],
)
def test_run_backtest_on_event_rejects_non_positive_stream_options(
    kwargs: dict[str, Any],
) -> None:
    """Stream API should reject non-positive option values."""
    data = _build_benchmark_data(n=5, symbol="STREAM_OPT")

    with pytest.raises(ValueError):
        akquant.run_backtest(
            data=data,
            strategy=NoopStrategy,
            symbols="STREAM_OPT",
            show_progress=False,
            on_event=lambda _event: None,
            **kwargs,
        )


def test_run_backtest_on_event_matches_run_backtest_result() -> None:
    """Stream run should keep the same backtest result semantics."""
    data = _build_benchmark_data(n=120, symbol="CONSIST")
    common_args: dict[str, Any] = dict(
        data=data,
        strategy=NoopStrategy,
        symbols="CONSIST",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
    )

    normal = akquant.run_backtest(**common_args)
    stream_events: list[akquant.BacktestStreamEvent] = []
    stream = akquant.run_backtest(
        **common_args,
        on_event=stream_events.append,
        stream_progress_interval=8,
        stream_equity_interval=8,
        stream_batch_size=16,
        stream_max_buffer=128,
    )

    assert len(stream.trades) == len(normal.trades)
    assert len(stream.equity_curve) == len(normal.equity_curve)
    assert stream.metrics.total_return == pytest.approx(
        normal.metrics.total_return, rel=1e-9
    )
    assert stream.metrics.max_drawdown == pytest.approx(
        normal.metrics.max_drawdown, rel=1e-9
    )
    assert stream_events[0]["event_type"] == "started"
    assert stream_events[-1]["event_type"] == "finished"


def test_run_backtest_on_event_emits_owner_strategy_id_for_trade_events() -> None:
    """Stream trade events should include owner strategy id in payload."""
    bars = _build_regression_bars("STREAM_OWNER")
    events: list[akquant.BacktestStreamEvent] = []
    result = akquant.run_backtest(
        data=bars,
        strategy=RegressionStrategy,
        symbols="STREAM_OWNER",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        on_event=events.append,
        strategy_id="stream_alpha",
    )
    trade_events = [event for event in events if event.get("event_type") == "trade"]
    assert trade_events
    owner_ids = {
        str(event.get("payload", {}).get("owner_strategy_id", ""))
        for event in trade_events
    }
    assert owner_ids == {"stream_alpha"}
    assert result.metrics.initial_market_value == pytest.approx(100000.0, rel=1e-9)


def test_run_backtest_without_on_event_keeps_legacy_semantics() -> None:
    """run_backtest without on_event should keep non-stream semantics."""
    data = _build_benchmark_data(n=80, symbol="NO_EVENT")
    result = akquant.run_backtest(
        data=data,
        strategy=NoopStrategy,
        symbols="NO_EVENT",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
    )
    assert result.metrics.initial_market_value == pytest.approx(100000.0, rel=1e-9)
    assert len(result.equity_curve) == len(data)
    assert len(result.cash_curve) == len(data)
    assert len(result.margin_curve) == len(data)


def test_run_backtest_strategy_id_propagates_to_orders() -> None:
    """run_backtest should tag generated orders with owner strategy id."""
    bars = _build_regression_bars("OWNER")
    result = akquant.run_backtest(
        data=bars,
        strategy=RegressionStrategy,
        symbols="OWNER",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
    )
    orders_df = result.orders_df
    assert not orders_df.empty
    assert "owner_strategy_id" in orders_df.columns
    assert set(orders_df["owner_strategy_id"].dropna().astype(str)) == {"alpha"}


def test_run_backtest_accepts_strategies_by_slot() -> None:
    """run_backtest should accept optional strategies_by_slot mapping."""
    bars = _build_regression_bars("SLOT_MAP")
    result = akquant.run_backtest(
        data=bars,
        strategy=NoopStrategy,
        symbols="SLOT_MAP",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategies_by_slot={"beta": NoopStrategy},
    )
    assert result.trade_metrics.total_closed_trades == 0


def test_run_backtest_multi_slot_owner_strategy_ids_mixed() -> None:
    """run_backtest should expose mixed owner strategy ids across slots."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_slots") or not hasattr(
        probe, "set_strategy_for_slot"
    ):
        pytest.skip("Engine binary does not expose multi-slot strategy methods")

    bars = _build_regression_bars("SLOT_OWNER_MIX")
    result = akquant.run_backtest(
        data=bars,
        strategy=RegressionStrategy,
        symbols="SLOT_OWNER_MIX",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": RegressionStrategy},
    )
    orders_df = result.orders_df
    executions_df = result.executions_df
    assert not orders_df.empty
    assert not executions_df.empty
    assert "owner_strategy_id" in orders_df.columns
    assert "owner_strategy_id" in executions_df.columns
    order_owner_ids = set(orders_df["owner_strategy_id"].dropna().astype(str))
    exec_owner_ids = set(executions_df["owner_strategy_id"].dropna().astype(str))
    assert order_owner_ids == {"alpha", "beta"}
    assert exec_owner_ids == {"alpha", "beta"}


def test_run_backtest_functional_on_start_on_stop_callbacks() -> None:
    """Function-style strategy should support on_start/on_stop lifecycle callbacks."""
    events: list[str] = []

    def initialize(ctx: Any) -> None:
        _ = ctx
        events.append("initialize")

    def on_start(ctx: Any) -> None:
        _ = ctx
        events.append("on_start")

    def on_bar(ctx: Any, bar: akquant.Bar) -> None:
        _ = ctx
        _ = bar
        events.append("on_bar")

    def on_stop(ctx: Any) -> None:
        _ = ctx
        events.append("on_stop")

    _ = akquant.run_backtest(
        data=_build_regression_bars("FUNC_LIFECYCLE"),
        strategy=on_bar,
        symbols="FUNC_LIFECYCLE",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        initialize=initialize,
        on_start=on_start,
        on_stop=on_stop,
    )

    assert events[0] == "initialize"
    assert events.count("on_start") == 1
    assert events.count("on_stop") == 1
    assert events[-1] == "on_stop"
    assert events.count("on_bar") == 3


@pytest.mark.parametrize(
    ("limit_key", "limit_value", "rejection_marker", "required_method"),
    [
        (
            "strategy_max_order_value",
            {"alpha": 50.0, "beta": 200.0},
            "exceeds strategy limit",
            "set_strategy_max_order_value_limits",
        ),
        (
            "strategy_max_order_size",
            {"alpha": 5.0, "beta": 20.0},
            "order quantity",
            "set_strategy_max_order_size_limits",
        ),
    ],
)
def test_run_backtest_functional_multi_slot_risk_matrix(
    limit_key: str,
    limit_value: dict[str, float],
    rejection_marker: str,
    required_method: str,
) -> None:
    """Function-style multi-slot strategies should honor per-slot risk limits."""
    probe = akquant.Engine()
    if not hasattr(probe, required_method):
        pytest.skip("Engine binary does not expose required strategy risk methods")

    def alpha_on_bar(ctx: Any, bar: akquant.Bar) -> None:
        if getattr(ctx, "_submitted_once", False):
            return
        ctx.buy(symbol=bar.symbol, quantity=10)
        ctx._submitted_once = True

    def beta_on_bar(ctx: Any, bar: akquant.Bar) -> None:
        if getattr(ctx, "_submitted_once", False):
            return
        ctx.buy(symbol=bar.symbol, quantity=10)
        ctx._submitted_once = True

    events: list[akquant.BacktestStreamEvent] = []
    extra_limits: dict[str, Any] = {limit_key: limit_value}
    result = akquant.run_backtest(
        data=_build_regression_bars(f"FUNC_SLOT_{limit_key.upper()}"),
        strategy=alpha_on_bar,
        symbols=f"FUNC_SLOT_{limit_key.upper()}",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": beta_on_bar},
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=1,
        stream_max_buffer=256,
        **extra_limits,
    )

    orders_df = result.orders_df
    assert not orders_df.empty
    alpha_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "beta"]
    assert not alpha_rows.empty
    assert not beta_rows.empty
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    beta_reject_reasons = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    assert any(rejection_marker in reason for reason in alpha_reject_reasons)
    assert not any(rejection_marker in reason for reason in beta_reject_reasons)

    risk_owner_ids = {
        str(event["payload"].get("owner_strategy_id"))
        for event in events
        if event.get("event_type") == "risk"
    }
    assert risk_owner_ids == {"alpha"}


def test_run_backtest_strategy_max_order_value_by_slot() -> None:
    """Per-strategy order value limit should reject only limited slot orders."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_order_value_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_regression_bars("SLOT_RISK_LIMIT")
    result = akquant.run_backtest(
        data=bars,
        strategy=SingleBuyStrategy,
        symbols="SLOT_RISK_LIMIT",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": SingleBuyStrategy},
        strategy_max_order_value={"alpha": 50.0, "beta": 200.0},
    )
    orders_df = result.orders_df
    assert not orders_df.empty
    alpha_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "beta"]
    assert not alpha_rows.empty
    assert not beta_rows.empty
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    assert any("exceeds strategy limit" in reason for reason in alpha_reject_reasons)
    beta_reject_reasons = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    assert not any("exceeds strategy limit" in reason for reason in beta_reject_reasons)


def test_run_backtest_strategy_max_order_size_by_slot() -> None:
    """Per-strategy order size limit should reject only limited slot orders."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_order_size_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_regression_bars("SLOT_RISK_SIZE")
    result = akquant.run_backtest(
        data=bars,
        strategy=SingleBuyStrategy,
        symbols="SLOT_RISK_SIZE",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": SingleBuyStrategy},
        strategy_max_order_size={"alpha": 5.0, "beta": 20.0},
    )
    orders_df = result.orders_df
    assert not orders_df.empty
    alpha_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "beta"]
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    beta_reject_reasons = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    assert any("order quantity" in reason for reason in alpha_reject_reasons)
    assert not any("order quantity" in reason for reason in beta_reject_reasons)


def test_run_backtest_strategy_slot_risk_from_config() -> None:
    """Config strategy settings should drive slot topology and risk limits."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_order_size_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(
            initial_cash=100000.0,
            strategy_id="alpha",
            strategies_by_slot={"beta": SingleBuyStrategy},
            strategy_max_order_size={"alpha": 5.0, "beta": 20.0},
        )
    )
    bars = _build_regression_bars("SLOT_RISK_SIZE_CFG")
    result = akquant.run_backtest(
        data=bars,
        strategy=SingleBuyStrategy,
        symbols="SLOT_RISK_SIZE_CFG",
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        config=config,
    )
    orders_df = result.orders_df
    assert not orders_df.empty
    alpha_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "beta"]
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    beta_reject_reasons = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    assert any("order quantity" in reason for reason in alpha_reject_reasons)
    assert not any("order quantity" in reason for reason in beta_reject_reasons)


def test_run_backtest_unknown_risk_config_key_includes_structured_context(
    caplog: Any,
) -> None:
    """Unknown risk config warnings should expose structured backtest context."""
    with caplog.at_level(logging.WARNING, logger="akquant"):
        _ = akquant.run_backtest(
            data=_build_regression_bars("RISK_CONTEXT"),
            strategy=SingleBuyStrategy,
            symbols="RISK_CONTEXT",
            initial_cash=100000.0,
            commission_rate=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            min_commission=0.0,
            fill_policy=CurrentClose(),
            lot_size=1,
            show_progress=False,
            strategy_id="alpha",
            risk_config={"unknown_key": 1},
        )

    warning_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.backtest"
        and "Unknown risk config key" in record.getMessage()
    )
    assert warning_record.phase == "risk"
    assert warning_record.strategy_id == "alpha"
    assert warning_record.slot == "alpha"
    assert warning_record.symbol is None


def test_run_backtest_slot_on_stop_error_includes_structured_context(
    caplog: Any,
) -> None:
    """Slot lifecycle errors should expose slot identity in log records."""
    with caplog.at_level(logging.ERROR, logger="akquant"):
        _ = akquant.run_backtest(
            data=_build_regression_bars("STOP_SLOT"),
            strategy=NoopStrategy,
            symbols="STOP_SLOT",
            initial_cash=100000.0,
            commission_rate=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            min_commission=0.0,
            fill_policy=CurrentClose(),
            lot_size=1,
            show_progress=False,
            strategy_id="alpha",
            strategies_by_slot={"beta": FailingOnStopStrategy},
        )

    error_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.backtest"
        and "Error in slot on_stop" in record.getMessage()
    )
    assert error_record.phase == "strategy"
    assert error_record.strategy_id == "beta"
    assert error_record.slot == "beta"
    assert error_record.symbol == "STOP_SLOT"


def test_run_backtest_explicit_strategy_slot_risk_overrides_config() -> None:
    """Explicit strategy slot risk args should override config values."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_order_size_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(
            initial_cash=100000.0,
            strategy_id="alpha",
            strategies_by_slot={"beta": SingleBuyStrategy},
            strategy_max_order_size={"alpha": 5.0, "beta": 20.0},
        )
    )
    bars = _build_regression_bars("SLOT_RISK_SIZE_CFG_OVERRIDE")
    result = akquant.run_backtest(
        data=bars,
        strategy=SingleBuyStrategy,
        symbols="SLOT_RISK_SIZE_CFG_OVERRIDE",
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        config=config,
        strategy_max_order_size={"alpha": 20.0, "beta": 5.0},
    )
    orders_df = result.orders_df
    assert not orders_df.empty
    alpha_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "beta"]
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    beta_reject_reasons = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    assert not any("order quantity" in reason for reason in alpha_reject_reasons)
    assert any("order quantity" in reason for reason in beta_reject_reasons)


def test_backtest_result_strategy_level_views() -> None:
    """BacktestResult should provide strategy-level orders/executions views."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_slots") or not hasattr(
        probe, "set_strategy_for_slot"
    ):
        pytest.skip("Engine binary does not expose multi-slot strategy methods")

    bars = _build_regression_bars("SLOT_VIEW")
    result = akquant.run_backtest(
        data=bars,
        strategy=RegressionStrategy,
        symbols="SLOT_VIEW",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": RegressionStrategy},
    )
    orders_view = result.orders_by_strategy()
    executions_view = result.executions_by_strategy()

    assert set(orders_view["owner_strategy_id"].astype(str)) == {"alpha", "beta"}
    assert set(executions_view["owner_strategy_id"].astype(str)) == {"alpha", "beta"}
    assert int(orders_view["order_count"].sum()) == len(result.orders_df)
    assert int(executions_view["execution_count"].sum()) == len(result.executions_df)


def test_backtest_result_risk_rejections_by_strategy_view() -> None:
    """BacktestResult should aggregate strategy-level risk rejections."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_reduce_only_after_risk"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_reduce_only_bars("SLOT_RISK_VIEW")
    result = akquant.run_backtest(
        data=bars,
        strategy=BuyBuySellBuyStrategy,
        symbols="SLOT_RISK_VIEW",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": BuyBuySellBuyStrategy},
        strategy_max_daily_loss={"alpha": 5.0, "beta": 50.0},
        strategy_reduce_only_after_risk={"alpha": True, "beta": False},
    )
    risk_view = result.risk_rejections_by_strategy()
    assert not risk_view.empty
    assert "owner_strategy_id" in risk_view.columns
    alpha = risk_view[risk_view["owner_strategy_id"].astype(str) == "alpha"]
    assert not alpha.empty
    alpha_row = alpha.iloc[0]
    assert int(alpha_row["risk_reject_count"]) >= 2
    assert int(alpha_row["daily_loss_reject_count"]) >= 1
    assert int(alpha_row["reduce_only_reject_count"]) >= 1
    assert "strategy_risk_budget_reject_count" in risk_view.columns
    assert "portfolio_risk_budget_reject_count" in risk_view.columns


def test_backtest_result_risk_rejections_trend_view() -> None:
    """BacktestResult should provide daily trend for risk rejections."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_reduce_only_after_risk"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_reduce_only_bars("SLOT_RISK_TREND")
    result = akquant.run_backtest(
        data=bars,
        strategy=BuyBuySellBuyStrategy,
        symbols="SLOT_RISK_TREND",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": BuyBuySellBuyStrategy},
        strategy_max_daily_loss={"alpha": 5.0, "beta": 50.0},
        strategy_reduce_only_after_risk={"alpha": True, "beta": False},
    )
    trend_view = result.risk_rejections_trend(freq="D")
    assert not trend_view.empty
    assert "date" in trend_view.columns
    assert "risk_reject_count" in trend_view.columns
    assert int(trend_view["risk_reject_count"].sum()) >= 2
    assert int(trend_view["reduce_only_reject_count"].sum()) >= 1
    assert "strategy_risk_budget_reject_count" in trend_view.columns
    assert "portfolio_risk_budget_reject_count" in trend_view.columns


def test_backtest_result_risk_rejections_trend_by_strategy_view() -> None:
    """BacktestResult should provide strategy-split risk rejection trend."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_reduce_only_after_risk"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_reduce_only_bars("SLOT_RISK_TREND_BY_STRATEGY")
    result = akquant.run_backtest(
        data=bars,
        strategy=BuyBuySellBuyStrategy,
        symbols="SLOT_RISK_TREND_BY_STRATEGY",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": BuyBuySellBuyStrategy},
        strategy_max_daily_loss={"alpha": 5.0, "beta": 50.0},
        strategy_reduce_only_after_risk={"alpha": True, "beta": False},
    )
    trend_by_strategy = result.risk_rejections_trend_by_strategy(freq="D")
    assert not trend_by_strategy.empty
    assert "date" in trend_by_strategy.columns
    assert "owner_strategy_id" in trend_by_strategy.columns
    assert "risk_reject_count" in trend_by_strategy.columns
    assert "strategy_risk_budget_reject_count" in trend_by_strategy.columns
    assert "portfolio_risk_budget_reject_count" in trend_by_strategy.columns
    alpha = trend_by_strategy[
        trend_by_strategy["owner_strategy_id"].astype(str) == "alpha"
    ]
    assert not alpha.empty
    assert int(alpha["risk_reject_count"].sum()) >= 2


def test_run_backtest_rejects_invalid_strategies_by_slot_type() -> None:
    """run_backtest should validate strategies_by_slot type."""
    bars = _build_regression_bars("SLOT_BAD")
    with pytest.raises(TypeError, match="strategies_by_slot"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="SLOT_BAD",
            show_progress=False,
            strategies_by_slot=cast(Any, ["bad"]),
        )


def test_run_backtest_rejects_unknown_strategy_max_order_value_key() -> None:
    """Unknown strategy id in strategy_max_order_value should fail fast."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_order_value_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")
    bars = _build_regression_bars("SLOT_RISK_BAD")
    with pytest.raises(ValueError, match="unknown strategy id"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="SLOT_RISK_BAD",
            show_progress=False,
            strategy_id="alpha",
            strategy_max_order_value={"beta": 100.0},
        )


def test_run_backtest_rejects_unknown_strategy_max_order_size_key() -> None:
    """Unknown strategy id in strategy_max_order_size should fail fast."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_order_size_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")
    bars = _build_regression_bars("SLOT_RISK_SIZE_BAD")
    with pytest.raises(ValueError, match="unknown strategy id"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="SLOT_RISK_SIZE_BAD",
            show_progress=False,
            strategy_id="alpha",
            strategy_max_order_size={"beta": 10.0},
        )


def test_run_backtest_strategy_max_position_size_by_slot() -> None:
    """Per-strategy position limit should reject only limited slot orders."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_position_size_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_regression_bars("SLOT_RISK_POSITION")
    result = akquant.run_backtest(
        data=bars,
        strategy=DualBuyStrategy,
        symbols="SLOT_RISK_POSITION",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": DualBuyStrategy},
        strategy_max_position_size={"alpha": 15.0, "beta": 30.0},
    )
    orders_df = result.orders_df
    alpha_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "beta"]
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    beta_reject_reasons = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    assert any("projected position" in reason for reason in alpha_reject_reasons)
    assert not any("projected position" in reason for reason in beta_reject_reasons)


def test_run_backtest_rejects_unknown_strategy_max_position_size_key() -> None:
    """Unknown strategy id in strategy_max_position_size should fail fast."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_position_size_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")
    bars = _build_regression_bars("SLOT_RISK_POSITION_BAD")
    with pytest.raises(ValueError, match="unknown strategy id"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="SLOT_RISK_POSITION_BAD",
            show_progress=False,
            strategy_id="alpha",
            strategy_max_position_size={"beta": 10.0},
        )


def test_run_backtest_strategy_max_daily_loss_by_slot() -> None:
    """Per-strategy daily loss limit should reject only limited slot orders."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_daily_loss_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_daily_loss_bars("SLOT_RISK_DAILY_LOSS")
    result = akquant.run_backtest(
        data=bars,
        strategy=DualBuyStrategy,
        symbols="SLOT_RISK_DAILY_LOSS",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": DualBuyStrategy},
        strategy_max_daily_loss={"alpha": 5.0, "beta": 50.0},
    )
    orders_df = result.orders_df
    alpha_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "beta"]
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    beta_reject_reasons = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    assert any("daily loss" in reason for reason in alpha_reject_reasons)
    assert not any("daily loss" in reason for reason in beta_reject_reasons)


def test_run_backtest_rejects_unknown_strategy_max_daily_loss_key() -> None:
    """Unknown strategy id in strategy_max_daily_loss should fail fast."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_daily_loss_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")
    bars = _build_regression_bars("SLOT_RISK_DAILY_LOSS_BAD")
    with pytest.raises(ValueError, match="unknown strategy id"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="SLOT_RISK_DAILY_LOSS_BAD",
            show_progress=False,
            strategy_id="alpha",
            strategy_max_daily_loss={"beta": 10.0},
        )


def test_run_backtest_strategy_max_drawdown_by_slot() -> None:
    """Per-strategy drawdown limit should reject only limited slot orders."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_drawdown_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_daily_loss_bars("SLOT_RISK_DRAWDOWN")
    result = akquant.run_backtest(
        data=bars,
        strategy=DualBuyStrategy,
        symbols="SLOT_RISK_DRAWDOWN",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": DualBuyStrategy},
        strategy_max_drawdown={"alpha": 5.0, "beta": 50.0},
    )
    orders_df = result.orders_df
    alpha_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "beta"]
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    beta_reject_reasons = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    assert any("drawdown" in reason for reason in alpha_reject_reasons)
    assert not any("drawdown" in reason for reason in beta_reject_reasons)


def test_run_backtest_rejects_unknown_strategy_max_drawdown_key() -> None:
    """Unknown strategy id in strategy_max_drawdown should fail fast."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_drawdown_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")
    bars = _build_regression_bars("SLOT_RISK_DRAWDOWN_BAD")
    with pytest.raises(ValueError, match="unknown strategy id"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="SLOT_RISK_DRAWDOWN_BAD",
            show_progress=False,
            strategy_id="alpha",
            strategy_max_drawdown={"beta": 10.0},
        )


def test_run_backtest_reduce_only_after_risk_allows_only_closing_orders() -> None:
    """Reduce-only mode after risk should reject reopen orders."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_reduce_only_after_risk"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_reduce_only_bars("SLOT_RISK_REDUCE_ONLY")
    result = akquant.run_backtest(
        data=bars,
        strategy=BuyBuySellBuyStrategy,
        symbols="SLOT_RISK_REDUCE_ONLY",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": BuyBuySellBuyStrategy},
        strategy_max_daily_loss={"alpha": 5.0, "beta": 50.0},
        strategy_reduce_only_after_risk={"alpha": True, "beta": False},
    )
    orders_df = result.orders_df
    alpha_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "alpha"]
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    assert any("daily loss" in reason for reason in alpha_reject_reasons)
    assert any("reduce_only mode" in reason for reason in alpha_reject_reasons)


def test_run_backtest_strategy_risk_cooldown_blocks_orders() -> None:
    """Risk-triggered cooldown should reject subsequent orders for configured bars."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_risk_cooldown_bars"):
        pytest.skip("Engine binary does not expose strategy cooldown methods")

    bars = _build_reduce_only_bars("SLOT_RISK_COOLDOWN")
    result = akquant.run_backtest(
        data=bars,
        strategy=ContinuousBuyStrategy,
        symbols="SLOT_RISK_COOLDOWN",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategy_max_order_size={"alpha": 5.0},
        strategy_risk_cooldown_bars={"alpha": 2},
    )
    orders_df = result.orders_df
    assert not orders_df.empty
    reject_reasons = orders_df["reject_reason"].fillna("").astype(str).tolist()
    assert any("order quantity" in reason for reason in reject_reasons)
    assert any("cooldown" in reason for reason in reject_reasons)


def test_run_backtest_rejects_unknown_strategy_reduce_only_after_risk_key() -> None:
    """Unknown strategy id in reduce-only map should fail fast."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_reduce_only_after_risk"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")
    bars = _build_regression_bars("SLOT_RISK_REDUCE_ONLY_BAD")
    with pytest.raises(ValueError, match="unknown strategy id"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="SLOT_RISK_REDUCE_ONLY_BAD",
            show_progress=False,
            strategy_id="alpha",
            strategy_reduce_only_after_risk={"beta": True},
        )


def test_run_backtest_rejects_unknown_strategy_risk_cooldown_bars_key() -> None:
    """Unknown strategy id in strategy_risk_cooldown_bars should fail fast."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_risk_cooldown_bars"):
        pytest.skip("Engine binary does not expose strategy cooldown methods")
    bars = _build_regression_bars("SLOT_RISK_COOLDOWN_BAD")
    with pytest.raises(ValueError, match="unknown strategy id"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="SLOT_RISK_COOLDOWN_BAD",
            show_progress=False,
            strategy_id="alpha",
            strategy_risk_cooldown_bars={"beta": 2},
        )


def test_run_backtest_rejects_unknown_strategy_priority_key() -> None:
    """Unknown strategy id in strategy_priority should fail fast."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_priorities"):
        pytest.skip("Engine binary does not expose strategy priority methods")
    bars = _build_regression_bars("SLOT_PRIORITY_BAD")
    with pytest.raises(ValueError, match="unknown strategy id"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="SLOT_PRIORITY_BAD",
            show_progress=False,
            strategy_id="alpha",
            strategy_priority={"beta": 100},
        )


def test_run_backtest_rejects_unknown_strategy_risk_budget_key() -> None:
    """Unknown strategy id in strategy_risk_budget should fail fast."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_risk_budget_limits"):
        pytest.skip("Engine binary does not expose strategy risk budget methods")
    bars = _build_regression_bars("SLOT_RISK_BUDGET_BAD")
    with pytest.raises(ValueError, match="unknown strategy id"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="SLOT_RISK_BUDGET_BAD",
            show_progress=False,
            strategy_id="alpha",
            strategy_risk_budget={"beta": 100.0},
        )


def test_run_backtest_rejects_invalid_risk_budget_mode() -> None:
    """Invalid risk_budget_mode should fail fast."""
    bars = _build_regression_bars("SLOT_RISK_BUDGET_MODE_BAD")
    with pytest.raises(ValueError, match="risk_budget_mode"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="SLOT_RISK_BUDGET_MODE_BAD",
            show_progress=False,
            risk_budget_mode=cast(Any, "bad_mode"),
        )


def test_run_backtest_strategy_id_propagates_to_executions_df() -> None:
    """run_backtest should expose owner strategy id in executions dataframe."""
    bars = _build_regression_bars("OWNER_EXEC")
    result = akquant.run_backtest(
        data=bars,
        strategy=RegressionStrategy,
        symbols="OWNER_EXEC",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha_exec",
    )
    executions_df = result.executions_df
    assert not executions_df.empty
    assert "owner_strategy_id" in executions_df.columns
    owner_ids = set(executions_df["owner_strategy_id"].dropna().astype(str))
    assert owner_ids == {"alpha_exec"}


def test_run_backtest_with_on_event_matches_stream_entry() -> None:
    """run_backtest with on_event should match unified stream semantics."""
    data = _build_benchmark_data(n=120, symbol="EVENT_EQ")
    common_args: dict[str, Any] = dict(
        data=data,
        strategy=NoopStrategy,
        symbols="EVENT_EQ",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
    )
    direct_events: list[akquant.BacktestStreamEvent] = []
    via_run_backtest = akquant.run_backtest(
        **common_args,
        on_event=direct_events.append,
        stream_progress_interval=8,
        stream_equity_interval=8,
        stream_batch_size=16,
        stream_max_buffer=128,
    )
    stream_events: list[akquant.BacktestStreamEvent] = []
    via_stream_entry = akquant.run_backtest(
        **common_args,
        on_event=stream_events.append,
        stream_progress_interval=8,
        stream_equity_interval=8,
        stream_batch_size=16,
        stream_max_buffer=128,
    )

    assert direct_events
    assert direct_events[0]["event_type"] == "started"
    assert direct_events[-1]["event_type"] == "finished"
    direct_seq_values = [event["seq"] for event in direct_events]
    assert direct_seq_values == sorted(direct_seq_values)
    assert len(via_run_backtest.trades) == len(via_stream_entry.trades)
    assert len(via_run_backtest.equity_curve) == len(via_stream_entry.equity_curve)
    assert via_run_backtest.metrics.total_return == pytest.approx(
        via_stream_entry.metrics.total_return, rel=1e-9
    )
    assert via_run_backtest.metrics.max_drawdown == pytest.approx(
        via_stream_entry.metrics.max_drawdown, rel=1e-9
    )


def test_run_backtest_on_event_multi_slot_owner_strategy_ids_mixed() -> None:
    """run_backtest with on_event should emit mixed owner strategy ids across slots."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_slots") or not hasattr(
        probe, "set_strategy_for_slot"
    ):
        pytest.skip("Engine binary does not expose multi-slot strategy methods")

    bars = _build_regression_bars("STREAM_SLOT_OWNER")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=bars,
        strategy=RegressionStrategy,
        symbols="STREAM_SLOT_OWNER",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": RegressionStrategy},
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=1,
        stream_max_buffer=256,
    )

    owner_ids = {
        str(event["payload"]["owner_strategy_id"])
        for event in events
        if event.get("event_type") in {"order", "trade", "risk"}
        and "owner_strategy_id" in event.get("payload", {})
    }
    assert owner_ids == {"alpha", "beta"}


def test_run_backtest_on_event_strategy_priority_orders_requests_by_priority() -> None:
    """run_backtest with on_event should process higher-priority orders first."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_priorities"):
        pytest.skip("Engine binary does not expose strategy priority methods")

    bars = _build_regression_bars("STREAM_SLOT_PRIORITY")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=bars,
        strategy=SingleBuyStrategy,
        symbols="STREAM_SLOT_PRIORITY",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": SingleBuyStrategy},
        strategy_priority={"alpha": 1, "beta": 10},
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=1,
        stream_max_buffer=256,
    )
    submitted_owner_ids = [
        str(event["payload"].get("owner_strategy_id"))
        for event in events
        if event.get("event_type") == "order"
        and str(event.get("payload", {}).get("status")) == "New"
    ]
    assert len(submitted_owner_ids) >= 2
    assert submitted_owner_ids[0] == "beta"
    assert submitted_owner_ids[1] == "alpha"


def test_run_backtest_on_event_portfolio_risk_budget_respects_priority_order() -> None:
    """Portfolio risk budget should reject lower-priority strategy."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_portfolio_risk_budget_limit"):
        pytest.skip("Engine binary does not expose portfolio risk budget methods")

    bars = _build_regression_bars("STREAM_SLOT_PORTFOLIO_BUDGET")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=bars,
        strategy=SingleBuyStrategy,
        symbols="STREAM_SLOT_PORTFOLIO_BUDGET",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": SingleBuyStrategy},
        strategy_priority={"alpha": 1, "beta": 10},
        portfolio_risk_budget=100.0,
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=1,
        stream_max_buffer=256,
    )
    accepted = [
        str(event["payload"].get("owner_strategy_id"))
        for event in events
        if event.get("event_type") == "order"
        and str(event.get("payload", {}).get("status")) == "New"
    ]
    assert accepted == ["beta"]
    rejected = [
        (
            str(event["payload"].get("owner_strategy_id")),
            str(event["payload"].get("reason", "")),
        )
        for event in events
        if event.get("event_type") == "risk"
    ]
    assert any(
        owner_id == "alpha" and "portfolio risk budget" in reason.lower()
        for owner_id, reason in rejected
    )


def test_run_backtest_trade_notional_budget_blocks_later_orders() -> None:
    """Trade-notional budget mode should block later bars after accumulated fills."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_risk_budget_mode"):
        pytest.skip("Engine binary does not expose risk budget mode methods")
    bars = _build_regression_bars("SLOT_TRADE_NOTIONAL_BUDGET")
    result = akquant.run_backtest(
        data=bars,
        strategy=ContinuousBuyStrategy,
        symbols="SLOT_TRADE_NOTIONAL_BUDGET",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategy_risk_budget={"alpha": 100.0},
        risk_budget_mode="trade_notional",
    )
    reasons = result.orders_df["reject_reason"].fillna("").astype(str).tolist()
    assert any("risk budget" in reason.lower() for reason in reasons)


def test_run_backtest_trade_notional_budget_resets_daily() -> None:
    """Daily reset should allow next-day orders under trade-notional budget mode."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_risk_budget_reset_daily"):
        pytest.skip("Engine binary does not expose risk budget reset methods")
    bars = _build_regression_bars("SLOT_TRADE_NOTIONAL_RESET")
    result = akquant.run_backtest(
        data=bars,
        strategy=ContinuousSmallBuyStrategy,
        symbols="SLOT_TRADE_NOTIONAL_RESET",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategy_risk_budget={"alpha": 100.0},
        risk_budget_mode="trade_notional",
        risk_budget_reset_daily=True,
    )
    reasons = result.orders_df["reject_reason"].fillna("").astype(str).tolist()
    assert not any("risk budget" in reason.lower() for reason in reasons)


def test_run_backtest_on_event_strategy_max_order_value_by_slot() -> None:
    """Per-strategy order value limit should reflect in stream risk events."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_order_value_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_regression_bars("STREAM_SLOT_RISK")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=bars,
        strategy=SingleBuyStrategy,
        symbols="STREAM_SLOT_RISK",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": SingleBuyStrategy},
        strategy_max_order_value={"alpha": 50.0, "beta": 200.0},
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=1,
        stream_max_buffer=256,
    )
    risk_owner_ids = {
        str(event["payload"].get("owner_strategy_id"))
        for event in events
        if event.get("event_type") == "risk"
    }
    assert risk_owner_ids == {"alpha"}


def test_run_backtest_on_event_strategy_max_order_size_by_slot() -> None:
    """Per-strategy order size limit should reflect in stream risk events."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_order_size_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_regression_bars("STREAM_SLOT_SIZE")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=bars,
        strategy=SingleBuyStrategy,
        symbols="STREAM_SLOT_SIZE",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": SingleBuyStrategy},
        strategy_max_order_size={"alpha": 5.0, "beta": 20.0},
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=1,
        stream_max_buffer=256,
    )
    risk_owner_ids = {
        str(event["payload"].get("owner_strategy_id"))
        for event in events
        if event.get("event_type") == "risk"
    }
    assert risk_owner_ids == {"alpha"}


def test_run_backtest_on_event_strategy_max_position_size_by_slot() -> None:
    """Per-strategy position limit should reflect in stream risk events."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_position_size_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_regression_bars("STREAM_SLOT_POSITION")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=bars,
        strategy=DualBuyStrategy,
        symbols="STREAM_SLOT_POSITION",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": DualBuyStrategy},
        strategy_max_position_size={"alpha": 15.0, "beta": 30.0},
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=1,
        stream_max_buffer=256,
    )
    risk_owner_ids = {
        str(event["payload"].get("owner_strategy_id"))
        for event in events
        if event.get("event_type") == "risk"
    }
    assert risk_owner_ids == {"alpha"}


def test_run_backtest_on_event_strategy_max_daily_loss_by_slot() -> None:
    """Per-strategy daily loss limit should reflect in stream risk events."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_daily_loss_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_daily_loss_bars("STREAM_SLOT_DAILY_LOSS")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=bars,
        strategy=DualBuyStrategy,
        symbols="STREAM_SLOT_DAILY_LOSS",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": DualBuyStrategy},
        strategy_max_daily_loss={"alpha": 5.0, "beta": 50.0},
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=1,
        stream_max_buffer=256,
    )
    risk_owner_ids = {
        str(event["payload"].get("owner_strategy_id"))
        for event in events
        if event.get("event_type") == "risk"
    }
    assert risk_owner_ids == {"alpha"}


def test_run_backtest_on_event_strategy_max_drawdown_by_slot() -> None:
    """Per-strategy drawdown limit should reflect in stream risk events."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_max_drawdown_limits"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_daily_loss_bars("STREAM_SLOT_DRAWDOWN")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=bars,
        strategy=DualBuyStrategy,
        symbols="STREAM_SLOT_DRAWDOWN",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": DualBuyStrategy},
        strategy_max_drawdown={"alpha": 5.0, "beta": 50.0},
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=1,
        stream_max_buffer=256,
    )
    risk_owner_ids = {
        str(event["payload"].get("owner_strategy_id"))
        for event in events
        if event.get("event_type") == "risk"
    }
    assert risk_owner_ids == {"alpha"}


def test_run_backtest_on_event_reduce_only_after_risk_by_slot() -> None:
    """Stream risk events should include reduce-only rejections."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_strategy_reduce_only_after_risk"):
        pytest.skip("Engine binary does not expose strategy-level risk limit methods")

    bars = _build_reduce_only_bars("STREAM_SLOT_REDUCE_ONLY")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=bars,
        strategy=BuyBuySellBuyStrategy,
        symbols="STREAM_SLOT_REDUCE_ONLY",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": BuyBuySellBuyStrategy},
        strategy_max_daily_loss={"alpha": 5.0, "beta": 50.0},
        strategy_reduce_only_after_risk={"alpha": True, "beta": False},
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=1,
        stream_max_buffer=256,
    )
    alpha_reduce_only_events = [
        event
        for event in events
        if event.get("event_type") == "risk"
        and str(event.get("payload", {}).get("owner_strategy_id")) == "alpha"
        and "reduce_only mode" in str(event.get("payload", {}).get("reason", ""))
    ]
    assert alpha_reduce_only_events


def test_run_backtest_rejects_removed_engine_mode_option() -> None:
    """Removed internal _engine_mode option should raise fast."""
    data = _build_benchmark_data(n=10, symbol="BAD_MODE")
    with pytest.raises(TypeError, match="_engine_mode is no longer supported"):
        akquant.run_backtest(
            data=data,
            strategy=NoopStrategy,
            symbols="BAD_MODE",
            show_progress=False,
            _engine_mode="legacy_blocking",
        )


def test_run_backtest_on_event_high_frequency_keeps_critical_events() -> None:
    """High-frequency stream should keep critical events and sampled updates."""
    data = _build_benchmark_data(n=2000, symbol="HF")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=data,
        strategy=NoopStrategy,
        symbols="HF",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        on_event=events.append,
        stream_progress_interval=50,
        stream_equity_interval=40,
        stream_batch_size=64,
        stream_max_buffer=256,
    )

    assert events
    assert events[0]["event_type"] == "started"
    assert events[-1]["event_type"] == "finished"
    assert sum(1 for e in events if e.get("event_type") == "started") == 1
    assert sum(1 for e in events if e.get("event_type") == "finished") == 1
    progress_count = sum(1 for e in events if e.get("event_type") == "progress")
    equity_count = sum(1 for e in events if e.get("event_type") == "equity")
    assert progress_count < 80
    assert equity_count < 100
    seq_values = [event["seq"] for event in events]
    assert seq_values == sorted(seq_values)
    finished_payload = events[-1]["payload"]
    assert "dropped_event_count" in finished_payload
    assert "dropped_event_count_by_type" in finished_payload
    assert int(str(finished_payload["dropped_event_count"])) >= 0


def test_run_backtest_on_event_callback_error_continue_mode() -> None:
    """Continue mode should survive callback exceptions."""
    data = _build_benchmark_data(n=40, symbol="CALLBACK_CONT")
    events: list[akquant.BacktestStreamEvent] = []
    counter = {"n": 0}

    def on_event(event: akquant.BacktestStreamEvent) -> None:
        counter["n"] += 1
        if counter["n"] <= 3:
            raise RuntimeError("callback boom")
        events.append(event)

    result = akquant.run_backtest(
        data=data,
        strategy=NoopStrategy,
        symbols="CALLBACK_CONT",
        show_progress=False,
        on_event=on_event,
        stream_error_mode="continue",
    )

    assert events
    assert events[-1]["event_type"] == "finished"
    assert "callback_error_count" in events[-1]["payload"]
    assert int(str(events[-1]["payload"]["callback_error_count"])) >= 3
    assert result.metrics.initial_market_value == pytest.approx(100000.0, rel=1e-9)


def test_run_backtest_on_event_reports_dropped_events_under_backpressure() -> None:
    """Finished payload should report dropped events when buffer is constrained."""
    data = _build_benchmark_data(n=300, symbol="DROP")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=data,
        strategy=NoopStrategy,
        symbols="DROP",
        show_progress=False,
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=32,
        stream_max_buffer=2,
    )

    assert events
    assert events[-1]["event_type"] == "finished"
    payload = events[-1]["payload"]
    dropped_count = int(str(payload.get("dropped_event_count", "0")))
    dropped_by_type = str(payload.get("dropped_event_count_by_type", ""))
    assert dropped_count > 0
    assert dropped_by_type


def test_run_backtest_on_event_audit_mode_enforces_full_delivery() -> None:
    """Audit mode should disable sampling and avoid dropping non-critical events."""
    data = _build_benchmark_data(n=300, symbol="AUDIT")
    events: list[akquant.BacktestStreamEvent] = []
    akquant.run_backtest(
        data=data,
        strategy=NoopStrategy,
        symbols="AUDIT",
        show_progress=False,
        on_event=events.append,
        stream_progress_interval=50,
        stream_equity_interval=40,
        stream_batch_size=32,
        stream_max_buffer=2,
        stream_mode="audit",
    )

    assert events
    assert events[-1]["event_type"] == "finished"
    progress_count = sum(1 for e in events if e.get("event_type") == "progress")
    equity_count = sum(1 for e in events if e.get("event_type") == "equity")
    assert progress_count > 100
    assert equity_count > 100
    payload = events[-1]["payload"]
    assert str(payload.get("stream_mode")) == "audit"
    assert str(payload.get("sampling_enabled")) == "false"
    assert str(payload.get("backpressure_policy")) == "block"
    assert int(str(payload.get("dropped_event_count", "0"))) == 0
    assert str(payload.get("dropped_event_count_by_type", "")) == ""


def test_run_backtest_on_event_callback_error_fail_fast_mode() -> None:
    """Fail-fast mode should raise once callback throws."""
    data = _build_benchmark_data(n=40, symbol="CALLBACK_FAIL")

    def on_event(_event: akquant.BacktestStreamEvent) -> None:
        raise RuntimeError("callback boom")

    with pytest.raises(RuntimeError, match="stream callback failed in fail_fast mode"):
        akquant.run_backtest(
            data=data,
            strategy=NoopStrategy,
            symbols="CALLBACK_FAIL",
            show_progress=False,
            on_event=on_event,
            stream_error_mode="fail_fast",
        )


def test_run_backtest_on_event_rejects_invalid_error_mode() -> None:
    """Invalid stream error mode should be rejected."""
    data = _build_benchmark_data(n=5, symbol="CALLBACK_MODE")
    with pytest.raises(ValueError):
        akquant.run_backtest(
            data=data,
            strategy=NoopStrategy,
            symbols="CALLBACK_MODE",
            show_progress=False,
            on_event=lambda _event: None,
            stream_error_mode=cast(Any, "bad_mode"),
        )


def test_run_backtest_on_event_rejects_invalid_stream_mode() -> None:
    """Invalid stream mode should be rejected."""
    data = _build_benchmark_data(n=5, symbol="MODE_BAD")
    with pytest.raises(ValueError):
        akquant.run_backtest(
            data=data,
            strategy=NoopStrategy,
            symbols="MODE_BAD",
            show_progress=False,
            on_event=lambda _event: None,
            stream_mode=cast(Any, "bad_mode"),
        )


def test_run_backtest_on_event_audit_mode_latency_budget_benchmark() -> None:
    """Audit mode benchmark with fixed callback delays for budget baselining."""
    data = _build_benchmark_data(n=240, symbol="AUDIT_BUDGET")
    delay_ms_options = [0, 1, 5]
    durations: dict[int, float] = {}
    event_counts: dict[int, int] = {}

    for delay_ms in delay_ms_options:
        counter = {"n": 0}

        def on_event(_event: akquant.BacktestStreamEvent) -> None:
            counter["n"] += 1
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

        start = time.perf_counter()
        akquant.run_backtest(
            data=data,
            strategy=NoopStrategy,
            symbols="AUDIT_BUDGET",
            show_progress=False,
            on_event=on_event,
            stream_progress_interval=50,
            stream_equity_interval=50,
            stream_batch_size=32,
            stream_max_buffer=64,
            stream_mode="audit",
        )
        durations[delay_ms] = time.perf_counter() - start
        event_counts[delay_ms] = counter["n"]

    assert event_counts[0] > 100
    assert event_counts[1] == event_counts[0]
    assert event_counts[5] == event_counts[0]
    assert durations[1] > durations[0]
    assert durations[5] > durations[1]


def test_run_backtest_broker_profile_applies_defaults() -> None:
    """broker_profile should inject template defaults when explicit args are omitted."""
    result = akquant.run_backtest(
        data=_build_regression_bars("PROFILE"),
        strategy=ProfileCaptureStrategy,
        symbols="PROFILE",
        fill_policy=CurrentClose(),
        broker_profile="cn_stock_miniqmt",
        show_progress=False,
    )

    strategy = cast(ProfileCaptureStrategy, result.strategy)
    assert strategy.snapshot["commission_rate"] == pytest.approx(0.0003, rel=1e-12)
    assert strategy.snapshot["stamp_tax_rate"] == pytest.approx(0.001, rel=1e-12)
    assert strategy.snapshot["transfer_fee_rate"] == pytest.approx(0.00001, rel=1e-12)
    assert strategy.snapshot["min_commission"] == pytest.approx(5.0, rel=1e-12)
    assert strategy.snapshot["lot_size"] == 100


def test_run_backtest_broker_profile_explicit_args_override_profile() -> None:
    """Explicit parameters should keep highest precedence over broker_profile values."""
    result = akquant.run_backtest(
        data=_build_regression_bars("PROFILE_OVERRIDE"),
        strategy=ProfileCaptureStrategy,
        symbols="PROFILE_OVERRIDE",
        fill_policy=CurrentClose(),
        broker_profile="cn_stock_miniqmt",
        commission_rate=0.0011,
        stamp_tax_rate=0.0022,
        min_commission=1.5,
        lot_size=10,
        show_progress=False,
    )

    strategy = cast(ProfileCaptureStrategy, result.strategy)
    assert strategy.snapshot["commission_rate"] == pytest.approx(0.0011, rel=1e-12)
    assert strategy.snapshot["stamp_tax_rate"] == pytest.approx(0.0022, rel=1e-12)
    assert strategy.snapshot["min_commission"] == pytest.approx(1.5, rel=1e-12)
    assert strategy.snapshot["lot_size"] == 10


def test_run_backtest_commission_policy_per_unit_overrides_rate_defaults() -> None:
    """`commission_policy` should activate per-unit mode.

    Legacy rate fields remain backward compatible.
    """
    result = akquant.run_backtest(
        data=_build_regression_bars("PROFILE_POLICY"),
        strategy=ProfileCaptureStrategy,
        symbols="PROFILE_POLICY",
        fill_policy=CurrentClose(),
        commission_policy={"type": "per_unit", "value": 0.5},
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        show_progress=False,
    )

    strategy = cast(ProfileCaptureStrategy, result.strategy)
    assert strategy.snapshot["commission_policy_type"] == "per_unit"
    assert strategy.snapshot["commission_policy_value"] == pytest.approx(0.5, rel=1e-12)
    assert strategy.snapshot["commission_rate"] == pytest.approx(0.0, rel=1e-12)


def test_run_backtest_broker_profile_explicit_zero_values_are_preserved() -> None:
    """Explicit 0.0 fee values should not be treated as omitted values."""
    result = akquant.run_backtest(
        data=_build_regression_bars("PROFILE_ZERO"),
        strategy=ProfileCaptureStrategy,
        symbols="PROFILE_ZERO",
        fill_policy=CurrentClose(),
        broker_profile="cn_stock_miniqmt",
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        show_progress=False,
    )

    strategy = cast(ProfileCaptureStrategy, result.strategy)
    assert strategy.snapshot["stamp_tax_rate"] == pytest.approx(0.0, rel=1e-12)
    assert strategy.snapshot["transfer_fee_rate"] == pytest.approx(0.0, rel=1e-12)
    assert strategy.snapshot["min_commission"] == pytest.approx(0.0, rel=1e-12)


def test_run_backtest_broker_profile_rejects_unknown_profile() -> None:
    """Unknown broker_profile should raise a validation error."""
    with pytest.raises(ValueError, match="Unknown broker_profile"):
        akquant.run_backtest(
            data=_build_regression_bars("PROFILE_BAD"),
            strategy=NoopStrategy,
            symbols="PROFILE_BAD",
            broker_profile="does_not_exist",
            show_progress=False,
        )


@pytest.mark.parametrize(
    ("profile", "expected_commission", "expected_min_commission"),
    [
        ("cn_stock_t1_low_fee", 0.0002, 3.0),
        ("cn_stock_sim_high_slippage", 0.0003, 5.0),
    ],
)
def test_run_backtest_broker_profile_additional_templates(
    profile: str, expected_commission: float, expected_min_commission: float
) -> None:
    """Additional built-in broker profiles should be available and injectable."""
    result = akquant.run_backtest(
        data=_build_regression_bars("PROFILE_EXTRA"),
        strategy=ProfileCaptureStrategy,
        symbols="PROFILE_EXTRA",
        fill_policy=CurrentClose(),
        broker_profile=profile,
        show_progress=False,
    )

    strategy = cast(ProfileCaptureStrategy, result.strategy)
    assert strategy.snapshot["commission_rate"] == pytest.approx(
        expected_commission, rel=1e-12
    )
    assert strategy.snapshot["min_commission"] == pytest.approx(
        expected_min_commission, rel=1e-12
    )


def test_backtest_result_get_event_stats_from_finished_payload() -> None:
    """Result wrapper should expose stream event summary stats in a unified dict."""
    result = akquant.run_backtest(
        data=_build_benchmark_data(n=120, symbol="EVENT_STATS"),
        strategy=NoopStrategy,
        symbols="EVENT_STATS",
        show_progress=False,
        on_event=lambda _event: None,
        stream_mode="audit",
    )

    stats = result.get_event_stats()
    assert isinstance(stats, dict)
    assert int(stats.get("processed_events", 0)) > 0
    assert str(stats.get("stream_mode")) == "audit"
    assert int(stats.get("callback_error_count", 0)) == 0


def test_run_backtest_analyzer_plugins_lifecycle_and_output() -> None:
    """Analyzer plugins should receive lifecycle events and write result output."""

    class CountingAnalyzer:
        name = "counting"

        def __init__(self) -> None:
            self.starts = 0
            self.bars = 0
            self.trades = 0

        def on_start(self, context: dict[str, Any]) -> None:
            _ = context
            self.starts += 1

        def on_bar(self, context: dict[str, Any]) -> None:
            _ = context
            self.bars += 1

        def on_trade(self, context: dict[str, Any]) -> None:
            _ = context
            self.trades += 1

        def on_finish(self, context: dict[str, Any]) -> dict[str, Any]:
            _ = context
            return {
                "starts": self.starts,
                "bars": self.bars,
                "trades": self.trades,
            }

    analyzer = CountingAnalyzer()
    result = akquant.run_backtest(
        data=_build_regression_bars("ANALYZER"),
        strategy=RegressionStrategy,
        symbols="ANALYZER",
        fill_policy=CurrentClose(),
        show_progress=False,
        analyzer_plugins=[analyzer],
    )

    assert hasattr(result, "analyzer_outputs")
    outputs = cast(dict[str, dict[str, Any]], result.analyzer_outputs)
    assert "counting" in outputs
    assert outputs["counting"]["starts"] == 1
    assert outputs["counting"]["bars"] == 3
    assert outputs["counting"]["trades"] >= 1


def test_run_backtest_analyzer_plugins_multi_slot_owner_context() -> None:
    """Analyzer contexts should include owner strategy ids across slots."""

    class OwnerAwareAnalyzer:
        name = "owner_aware"

        def __init__(self) -> None:
            self.bar_owner_ids: set[str] = set()
            self.trade_owner_ids: set[str] = set()

        def on_start(self, context: dict[str, Any]) -> None:
            _ = context

        def on_bar(self, context: dict[str, Any]) -> None:
            owner_strategy_id = str(context.get("owner_strategy_id", "")).strip()
            if owner_strategy_id:
                self.bar_owner_ids.add(owner_strategy_id)

        def on_trade(self, context: dict[str, Any]) -> None:
            owner_strategy_id = str(context.get("owner_strategy_id", "")).strip()
            if owner_strategy_id:
                self.trade_owner_ids.add(owner_strategy_id)

        def on_finish(self, context: dict[str, Any]) -> dict[str, Any]:
            _ = context
            return {
                "bar_owner_ids": sorted(self.bar_owner_ids),
                "trade_owner_ids": sorted(self.trade_owner_ids),
            }

    analyzer = OwnerAwareAnalyzer()
    result = akquant.run_backtest(
        data=_build_regression_bars("ANALYZER_SLOT"),
        strategy=RegressionStrategy,
        symbols="ANALYZER_SLOT",
        fill_policy=CurrentClose(),
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": RegressionStrategy},
        analyzer_plugins=[analyzer],
    )

    outputs = cast(dict[str, dict[str, Any]], result.analyzer_outputs)
    assert "owner_aware" in outputs
    assert outputs["owner_aware"]["bar_owner_ids"] == ["alpha", "beta"]
    assert outputs["owner_aware"]["trade_owner_ids"] == ["alpha", "beta"]


def test_run_backtest_china_futures_validation_prefix_override() -> None:
    """Prefix validation config should override default futures lot validation."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_futures_validation_options_by_prefix"):
        pytest.skip("Engine binary does not expose futures prefix validation methods")

    class FractionalFuturesBuyStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self._submitted = False

        def on_bar(self, bar: akquant.Bar) -> None:
            if self._submitted:
                return
            self.buy(symbol=bar.symbol, quantity=1.5)
            self._submitted = True

    symbol = "RB2310"
    bars = _build_regression_bars(symbol)
    config_reject = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
                tick_size=0.2,
            )
        ],
        china_futures=akquant.ChinaFuturesConfig(
            enforce_lot_size=True,
            enforce_tick_size=True,
            enforce_sessions=False,
        ),
    )
    result_reject = akquant.run_backtest(
        data=bars,
        strategy=FractionalFuturesBuyStrategy,
        symbols=symbol,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=config_reject,
    )
    reject_reasons = (
        result_reject.orders_df["reject_reason"].fillna("").astype(str).tolist()
    )
    assert any("lot size" in reason for reason in reject_reasons)

    config_accept = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
                tick_size=0.2,
            )
        ],
        china_futures=akquant.ChinaFuturesConfig(
            enforce_lot_size=True,
            enforce_tick_size=True,
            enforce_sessions=False,
            validation_by_symbol_prefix=[
                akquant.ChinaFuturesValidationConfig(
                    symbol_prefix="RB",
                    enforce_lot_size=False,
                )
            ],
        ),
    )
    result_accept = akquant.run_backtest(
        data=bars,
        strategy=FractionalFuturesBuyStrategy,
        symbols=symbol,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=config_accept,
    )
    accept_reject_reasons = (
        result_accept.orders_df["reject_reason"].fillna("").astype(str).tolist()
    )
    assert not any("lot size" in reason for reason in accept_reject_reasons)


def test_run_backtest_china_futures_instrument_template_multiplier() -> None:
    """Instrument template should inject futures multiplier by symbol prefix."""

    class BuyAndHoldOnceStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self._submitted = False

        def on_bar(self, bar: akquant.Bar) -> None:
            if self._submitted:
                return
            self.buy(symbol=bar.symbol, quantity=1.0)
            self._submitted = True

    symbol = "RB_TMPL_01"
    bars = _build_regression_bars(symbol)
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(
            initial_cash=1_000_000.0,
            commission_rate=0.0,
            slippage=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
        ),
        china_futures=akquant.ChinaFuturesConfig(
            enforce_sessions=False,
            instrument_templates_by_symbol_prefix=[
                akquant.ChinaFuturesInstrumentTemplateConfig(
                    symbol_prefix="RB",
                    multiplier=10.0,
                    margin_ratio=0.1,
                    tick_size=0.2,
                    lot_size=1.0,
                )
            ],
        ),
    )
    result = akquant.run_backtest(
        data=bars,
        strategy=BuyAndHoldOnceStrategy,
        symbols=symbol,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=config,
    )
    final_equity = float(result.equity_curve.iloc[-1])
    assert final_equity == pytest.approx(1_000_009.9977, rel=0.0, abs=1e-6)


def test_run_backtest_instrument_lot_size_explicit_one_overrides_template() -> None:
    """Explicit instrument lot_size=1 should override template lot_size."""

    class LotProbeStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.observed_lot_size: float = -1.0

        def on_start(self) -> None:
            snapshot = self.get_instrument("RB_TMPL_LOT_01")
            self.observed_lot_size = float(snapshot.lot_size)

    symbol = "RB_TMPL_LOT_01"
    bars = _build_regression_bars(symbol)
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(
            initial_cash=1_000_000.0,
            commission_rate=0.0,
            slippage=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
        ),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="FUTURES",
                lot_size=1,
            )
        ],
        china_futures=akquant.ChinaFuturesConfig(
            enforce_sessions=False,
            instrument_templates_by_symbol_prefix=[
                akquant.ChinaFuturesInstrumentTemplateConfig(
                    symbol_prefix="RB",
                    multiplier=10.0,
                    margin_ratio=0.1,
                    tick_size=0.2,
                    lot_size=5.0,
                )
            ],
        ),
    )
    result = akquant.run_backtest(
        data=bars,
        strategy=LotProbeStrategy,
        symbols=symbol,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=config,
    )
    strategy = cast(LotProbeStrategy, result.strategy)
    assert strategy.observed_lot_size == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_run_backtest_china_futures_template_commission_prefix() -> None:
    """Template commission should be merged into prefix fee rules."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_futures_fee_rules_by_prefix"):
        pytest.skip("Engine binary does not expose futures prefix fee methods")

    class BuyAndHoldOnceStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self._submitted = False

        def on_bar(self, bar: akquant.Bar) -> None:
            if self._submitted:
                return
            self.buy(symbol=bar.symbol, quantity=1.0)
            self._submitted = True

    symbol = "RB_TMPL_FEE_01"
    bars = _build_regression_bars(symbol)
    base_config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(
            initial_cash=1_000_000.0,
            commission_rate=0.0,
            slippage=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
        ),
        china_futures=akquant.ChinaFuturesConfig(
            enforce_sessions=False,
            instrument_templates_by_symbol_prefix=[
                akquant.ChinaFuturesInstrumentTemplateConfig(
                    symbol_prefix="RB",
                    multiplier=10.0,
                    margin_ratio=0.1,
                    tick_size=0.2,
                    lot_size=1.0,
                )
            ],
        ),
    )
    high_fee_config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(
            initial_cash=1_000_000.0,
            commission_rate=0.0,
            slippage=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
        ),
        china_futures=akquant.ChinaFuturesConfig(
            enforce_sessions=False,
            instrument_templates_by_symbol_prefix=[
                akquant.ChinaFuturesInstrumentTemplateConfig(
                    symbol_prefix="RB",
                    multiplier=10.0,
                    margin_ratio=0.1,
                    tick_size=0.2,
                    lot_size=1.0,
                    commission_rate=0.001,
                )
            ],
        ),
    )
    result_base = akquant.run_backtest(
        data=bars,
        strategy=BuyAndHoldOnceStrategy,
        symbols=symbol,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=base_config,
    )
    result_high_fee = akquant.run_backtest(
        data=bars,
        strategy=BuyAndHoldOnceStrategy,
        symbols=symbol,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=high_fee_config,
    )
    assert float(result_high_fee.equity_curve.iloc[-1]) < float(
        result_base.equity_curve.iloc[-1]
    )


def test_run_backtest_china_futures_rejects_duplicate_template_prefix() -> None:
    """Duplicate template prefixes should fail fast."""
    bars = _build_regression_bars("RB_DUP_TPL")
    with pytest.raises(
        ValueError,
        match=(
            "instrument_templates_by_symbol_prefix\\[1\\] duplicates symbol_prefix 'RB'"
        ),
    ):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="RB_DUP_TPL",
            show_progress=False,
            config=akquant.BacktestConfig(
                strategy_config=akquant.StrategyConfig(),
                china_futures=akquant.ChinaFuturesConfig(
                    instrument_templates_by_symbol_prefix=[
                        akquant.ChinaFuturesInstrumentTemplateConfig(
                            symbol_prefix="RB",
                            multiplier=10.0,
                        ),
                        akquant.ChinaFuturesInstrumentTemplateConfig(
                            symbol_prefix="rb",
                            multiplier=20.0,
                        ),
                    ]
                ),
            ),
        )


def test_run_backtest_china_futures_rejects_negative_template_multiplier() -> None:
    """Negative template multiplier should fail fast."""
    bars = _build_regression_bars("RB_BAD_MULT")
    with pytest.raises(ValueError, match="multiplier must be > 0"):
        akquant.run_backtest(
            data=bars,
            strategy=NoopStrategy,
            symbols="RB_BAD_MULT",
            show_progress=False,
            config=akquant.BacktestConfig(
                strategy_config=akquant.StrategyConfig(),
                china_futures=akquant.ChinaFuturesConfig(
                    instrument_templates_by_symbol_prefix=[
                        akquant.ChinaFuturesInstrumentTemplateConfig(
                            symbol_prefix="RB",
                            multiplier=-1.0,
                        )
                    ]
                ),
            ),
        )


def test_run_backtest_china_options_fee_prefix() -> None:
    """China options prefix fee should override global option commission."""
    probe = akquant.Engine()
    if not hasattr(probe, "set_options_fee_rules_by_prefix"):
        pytest.skip("Engine binary does not expose options prefix fee methods")

    class BuyAndHoldOnceStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self._submitted = False

        def on_bar(self, bar: akquant.Bar) -> None:
            if self._submitted:
                return
            self.buy(symbol=bar.symbol, quantity=1.0)
            self._submitted = True

    symbol = "OPT_TMPL_FEE_01"
    bars = _build_regression_bars(symbol)
    base_config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(
            initial_cash=1_000_000.0,
            commission_rate=0.0,
            slippage=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
        ),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="OPTION",
                option_type="CALL",
                strike_price=2.0,
                underlying_symbol="510050.SH",
                multiplier=1.0,
                tick_size=0.0001,
                margin_ratio=1.0,
                lot_size=1,
            )
        ],
        china_options=akquant.ChinaOptionsConfig(
            fee_per_contract=0.0,
            fee_by_symbol_prefix=[
                akquant.ChinaOptionsFeeConfig(
                    symbol_prefix="OPT",
                    commission_per_contract=0.0,
                )
            ],
        ),
    )
    high_fee_config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(
            initial_cash=1_000_000.0,
            commission_rate=0.0,
            slippage=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
        ),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="OPTION",
                option_type="CALL",
                strike_price=2.0,
                underlying_symbol="510050.SH",
                multiplier=1.0,
                tick_size=0.0001,
                margin_ratio=1.0,
                lot_size=1,
            )
        ],
        china_options=akquant.ChinaOptionsConfig(
            fee_per_contract=0.0,
            fee_by_symbol_prefix=[
                akquant.ChinaOptionsFeeConfig(
                    symbol_prefix="OPT",
                    commission_per_contract=12.0,
                )
            ],
        ),
    )
    result_base = akquant.run_backtest(
        data=bars,
        strategy=BuyAndHoldOnceStrategy,
        symbols=symbol,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=base_config,
    )
    result_high_fee = akquant.run_backtest(
        data=bars,
        strategy=BuyAndHoldOnceStrategy,
        symbols=symbol,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=high_fee_config,
    )
    assert float(result_high_fee.equity_curve.iloc[-1]) < float(
        result_base.equity_curve.iloc[-1]
    )


def test_china_options_config_rejects_duplicate_prefix() -> None:
    """Duplicate china options fee prefixes should fail fast."""
    with pytest.raises(
        ValueError,
        match="fee_by_symbol_prefix\\[1\\] duplicates symbol_prefix 'OPT'",
    ):
        akquant.ChinaOptionsConfig(
            fee_by_symbol_prefix=[
                akquant.ChinaOptionsFeeConfig(
                    symbol_prefix="OPT",
                    commission_per_contract=2.0,
                ),
                akquant.ChinaOptionsFeeConfig(
                    symbol_prefix="opt",
                    commission_per_contract=3.0,
                ),
            ]
        )


def test_china_futures_validation_config_requires_at_least_one_switch() -> None:
    """Validation config should fail if both switches are omitted."""
    with pytest.raises(
        ValueError, match="must set enforce_tick_size or enforce_lot_size"
    ):
        akquant.ChinaFuturesValidationConfig(symbol_prefix="RB")


def test_china_futures_session_profile_rejects_invalid_value() -> None:
    """China futures session profile should validate allowed presets."""
    with pytest.raises(ValueError, match="session_profile must be one of"):
        akquant.ChinaFuturesConfig(session_profile="CN_FUTURES_UNKNOWN")


def test_china_futures_session_profile_accepts_cffex_presets() -> None:
    """China futures session profile should accept CFFEX day presets."""
    config_stock = akquant.ChinaFuturesConfig(
        session_profile="CN_FUTURES_CFFEX_STOCK_INDEX_DAY"
    )
    config_bond = akquant.ChinaFuturesConfig(
        session_profile="CN_FUTURES_CFFEX_BOND_DAY"
    )
    assert config_stock.session_profile == "CN_FUTURES_CFFEX_STOCK_INDEX_DAY"
    assert config_bond.session_profile == "CN_FUTURES_CFFEX_BOND_DAY"


def test_run_grid_search_parallel_accepts_fill_policy() -> None:
    """Parallel grid search should accept fill_policy in kwargs."""
    data = _build_benchmark_data(n=40, symbol="OPT_EXEC_MODE_ENUM")

    results = akquant.run_grid_search(
        strategy=NoopStrategy,
        param_grid={"dummy": [1, 2]},
        data=data,
        symbol="OPT_EXEC_MODE_ENUM",
        fill_policy=CurrentClose(),
        max_workers=2,
        return_df=True,
        show_progress=False,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2


def test_run_grid_search_parallel_fail_fast_for_unpickleable_callback() -> None:
    """Parallel grid search should fail fast with clear error for lambda callback."""
    data = _build_benchmark_data(n=40, symbol="OPT_PICKLE_FAILFAST")

    with pytest.raises(
        TypeError,
        match="kwargs\\['on_event'\\] failed",
    ):
        akquant.run_grid_search(
            strategy=NoopStrategy,
            param_grid={"dummy": [1, 2]},
            data=data,
            symbol="OPT_PICKLE_FAILFAST",
            max_workers=2,
            return_df=True,
            show_progress=False,
            on_event=lambda _event: None,
        )


def test_run_grid_search_strict_params_raises_on_constructor_mismatch() -> None:
    """Grid search should fail fast when strategy constructor params mismatch."""

    class StrictParamStrategy(akquant.Strategy):
        def __init__(self, threshold: float = 1.0) -> None:
            super().__init__()
            self.threshold = float(threshold)

        def on_bar(self, bar: akquant.Bar) -> None:
            return

    data = _build_benchmark_data(n=20, symbol="OPT_STRICT_PARAMS")
    with pytest.raises(TypeError, match="Unknown strategy param\\(s\\) in param_grid"):
        akquant.run_grid_search(
            strategy=StrictParamStrategy,
            param_grid={"not_exist": [1, 2]},
            data=data,
            symbol="OPT_STRICT_PARAMS",
            max_workers=1,
            return_df=True,
            show_progress=False,
        )


def test_configure_logging_supports_mixed_console_and_file_levels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Console/file handler levels should work together without losing records."""
    log_file = tmp_path / "mixed_levels.log"
    akquant.configure_logging(
        akquant.LogConfig(
            level="INFO",
            console=True,
            filename=str(log_file),
            file_level="DEBUG",
        )
    )
    logger = akquant.get_logger()

    logger.debug("debug-file-only")
    logger.info("info-both")

    captured = capsys.readouterr()
    file_text = log_file.read_text(encoding="utf-8")

    assert logger.level == logging.DEBUG
    assert "debug-file-only" not in captured.out
    assert "info-both" in captured.out
    assert "debug-file-only" in file_text
    assert "info-both" in file_text

    akquant.register_logger(console=False, level="INFO")


def test_configure_logging_optimize_profile_renders_process_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Optimize profile should include process name and logger name in output."""
    akquant.configure_logging(akquant.LogConfig(profile="optimize", level="WARNING"))
    akquant.get_logger("optimize").warning("optimize-profile-check")

    captured = capsys.readouterr()
    assert "MainProcess" in captured.out
    assert "akquant.optimize" in captured.out
    assert "optimize-profile-check" in captured.out

    akquant.register_logger(console=False, level="INFO")


def test_configure_logging_supports_rotating_file_handler(tmp_path: Path) -> None:
    """File rotation should preserve records across rollover files."""
    log_file = tmp_path / "rotating.log"
    akquant.configure_logging(
        akquant.LogConfig(
            console=False,
            filename=str(log_file),
            file_level="INFO",
            file_max_bytes=120,
            file_backup_count=1,
        )
    )
    logger = akquant.get_logger("strategy")

    logger.info("first-message-%s", "A" * 80)
    logger.info("second-message-%s", "B" * 80)

    rotated_file = log_file.with_name(f"{log_file.name}.1")
    assert rotated_file.exists()

    combined_text = rotated_file.read_text(encoding="utf-8") + log_file.read_text(
        encoding="utf-8"
    )
    assert "first-message-" in combined_text
    assert "second-message-" in combined_text

    akquant.register_logger(console=False, level="INFO")


def test_configure_logging_supports_json_file_output(tmp_path: Path) -> None:
    """JSON file logging should emit structured JSON records."""
    import akquant.log as aklog

    log_file = tmp_path / "json.log"
    akquant.configure_logging(
        akquant.LogConfig(
            console=False,
            filename=str(log_file),
            file_json=True,
            file_level="INFO",
        )
    )
    logger = akquant.get_logger("strategy")
    logger.info(
        "json-file-check",
        extra=aklog.build_log_extra(
            phase="trade",
            strategy_id="alpha",
            slot="alpha",
            symbol="AAPL",
            order_id="order-1",
            client_order_id="coid-1",
            event_time_iso="2023-01-01T01:30:00Z",
        ),
    )

    payload = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert payload["message"] == "json-file-check"
    assert payload["logger"] == "akquant.strategy"
    assert payload["phase"] == "trade"
    assert payload["strategy_id"] == "alpha"
    assert payload["symbol"] == "AAPL"
    assert payload["order_id"] == "order-1"
    assert payload["client_order_id"] == "coid-1"
    assert payload["event_time_iso"] == "2023-01-01T01:30:00Z"

    akquant.register_logger(console=False, level="INFO")


def test_configure_logging_supports_json_console_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON console logging should emit structured JSON records."""
    import akquant.log as aklog

    akquant.configure_logging(
        akquant.LogConfig(
            console=True,
            console_json=True,
            level="INFO",
        )
    )
    logger = akquant.get_logger("gateway.live")
    logger.warning(
        "json-console-check",
        extra=aklog.build_log_extra(
            phase="gateway",
            strategy_id="beta",
            slot="beta",
            symbol="IF2406",
            client_order_id="coid-json-console",
        ),
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["message"] == "json-console-check"
    assert payload["logger"] == "akquant.gateway.live"
    assert payload["phase"] == "gateway"
    assert payload["strategy_id"] == "beta"
    assert payload["symbol"] == "IF2406"
    assert payload["client_order_id"] == "coid-json-console"

    akquant.register_logger(console=False, level="INFO")


def test_rust_warnings_bridge_into_python_logging(caplog: Any) -> None:
    """Rust log::warn! output should flow through the Python logging pipeline."""
    akquant.configure_logging(
        akquant.LogConfig(
            console=False,
            level="WARNING",
        )
    )

    with caplog.at_level(logging.WARNING, logger="akquant"):
        bars = akquant.from_arrays(
            np.array([1], dtype=np.int64),
            np.array([np.nan], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "AAPL",
            None,
            None,
        )

    assert bars[0].open == 0.0
    matching_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.data.columns"
        and "Invalid open price NaN, defaulting to 0.0" in record.getMessage()
    )
    assert matching_record.phase == "data"
    assert matching_record.symbol == "AAPL"
    assert matching_record.event_time_iso == "1970-01-01T00:00:00.000000001Z"
    assert any(
        record.name == "akquant.data.columns"
        and "Invalid open price NaN, defaulting to 0.0" in record.getMessage()
        for record in caplog.records
    )

    akquant.register_logger(console=False, level="INFO")


def test_run_backtest_invalid_volume_limit_bridges_rust_warning(caplog: Any) -> None:
    """Execution-layer Rust warnings should also enter the Python logging pipeline."""
    akquant.configure_logging(
        akquant.LogConfig(
            console=False,
            level="WARNING",
        )
    )

    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = akquant.run_backtest(
            strategy=NoopStrategy,
            data=_build_regression_bars("VOL_LIMIT_WARN"),
            symbols="VOL_LIMIT_WARN",
            show_progress=False,
            commission_rate=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            min_commission=0.0,
            volume_limit_pct=float("nan"),
        )

    assert result is not None
    matching_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.execution.simulated"
        and "Invalid volume limit NaN, defaulting to 0.0" in record.getMessage()
    )
    assert matching_record.phase == "execution"
    assert matching_record.symbol is None
    assert any(
        record.name == "akquant.execution.simulated"
        and "Invalid volume limit NaN, defaulting to 0.0" in record.getMessage()
        for record in caplog.records
    )

    akquant.register_logger(console=False, level="INFO")


def test_run_backtest_lot_size_rejection_bridges_rust_warning(caplog: Any) -> None:
    """Common execution rejects should bridge structured Rust warnings."""
    akquant.configure_logging(
        akquant.LogConfig(
            console=False,
            level="WARNING",
        )
    )

    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = akquant.run_backtest(
            strategy=SingleBuyStrategy,
            data=_build_regression_bars("LOT_WARN"),
            symbols="LOT_WARN",
            show_progress=False,
            fill_policy=CurrentClose(),
            lot_size=100,
        )

    assert result is not None
    matching_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.execution.common"
        and "Rejected order because Quantity 10 is not a multiple of lot size 100"
        in record.getMessage()
    )
    assert matching_record.phase == "execution"
    assert matching_record.symbol == "LOT_WARN"
    assert matching_record.event_time_iso == "2023-01-02T15:00:00Z"
    assert getattr(matching_record, "order_id", None)

    akquant.register_logger(console=False, level="INFO")


def test_run_backtest_futures_validation_rejection_bridges_rust_warning(
    caplog: Any,
) -> None:
    """Futures validation rejects should bridge structured Rust warnings."""

    class FractionalFuturesBuyStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self._submitted = False

        def on_bar(self, bar: akquant.Bar) -> None:
            if self._submitted:
                return
            self.buy(symbol=bar.symbol, quantity=1.5)
            self._submitted = True

    akquant.configure_logging(
        akquant.LogConfig(
            console=False,
            level="WARNING",
        )
    )

    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = akquant.run_backtest(
            strategy=FractionalFuturesBuyStrategy,
            data=_build_regression_bars("RB2310"),
            symbols="RB2310",
            show_progress=False,
            fill_policy=CurrentClose(),
            config=akquant.BacktestConfig(
                strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
                instruments_config=[
                    akquant.InstrumentConfig(
                        symbol="RB2310",
                        asset_type="FUTURES",
                        multiplier=10.0,
                        margin_ratio=0.1,
                        tick_size=0.2,
                    )
                ],
                china_futures=akquant.ChinaFuturesConfig(
                    enforce_lot_size=True,
                    enforce_tick_size=True,
                    enforce_sessions=False,
                ),
            ),
        )

    assert result is not None
    matching_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.execution.futures"
        and (
            "Rejected futures order because Quantity 1.5 is not a multiple of lot "
            "size 1"
        )
        in record.getMessage()
    )
    assert matching_record.phase == "execution"
    assert matching_record.symbol == "RB2310"
    assert matching_record.event_time_iso == "2023-01-02T15:00:00Z"
    assert getattr(matching_record, "order_id", None)

    akquant.register_logger(console=False, level="INFO")


def test_run_backtest_ioc_cancel_bridges_rust_warning(caplog: Any) -> None:
    """IOC/FOK auto-cancel should bridge structured Rust warnings."""

    class IOCNoFillStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self._submitted = False

        def on_bar(self, bar: akquant.Bar) -> None:
            if self._submitted:
                return
            self.buy(
                symbol=bar.symbol,
                quantity=10,
                price=1.0,
                time_in_force=akquant.TimeInForce.IOC,
            )
            self._submitted = True

    akquant.configure_logging(
        akquant.LogConfig(
            console=False,
            level="WARNING",
        )
    )

    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = akquant.run_backtest(
            strategy=IOCNoFillStrategy,
            data=_build_regression_bars("IOC_WARN"),
            symbols="IOC_WARN",
            show_progress=False,
            fill_policy=CurrentClose(),
        )

    assert result is not None
    matching_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.execution.common"
        and "Cancelled IOC order because it was not filled on current event"
        in record.getMessage()
    )
    assert matching_record.phase == "execution"
    assert matching_record.symbol == "IOC_WARN"
    assert matching_record.event_time_iso == "2023-01-02T15:00:00Z"
    assert getattr(matching_record, "order_id", None)

    akquant.register_logger(console=False, level="INFO")


def test_engine_tick_ioc_cancel_bridges_rust_warning(caplog: Any) -> None:
    """Tick-path IOC/FOK auto-cancel should bridge structured Rust warnings."""

    class TickIOCNoFillStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self._submitted = False

        def on_tick(self, tick: akquant.Tick) -> None:
            if self._submitted:
                return
            self.buy(
                symbol=tick.symbol,
                quantity=10,
                price=99.0,
                time_in_force=akquant.TimeInForce.IOC,
            )
            self._submitted = True

    symbol = "TICK_IOC_WARN"
    engine = akquant.Engine()
    engine.use_simple_market(0.0)
    engine.set_force_session_continuous(True)
    cast(Any, engine).set_fill_mode(akquant.ExecutionMode.CurrentClose, "same_cycle")
    engine.set_cash(100000.0)
    engine.set_stock_fee_rules(0.0, 0.0, 0.0, 0.0)
    engine.add_instrument(
        akquant.Instrument(
            symbol=symbol,
            asset_type=akquant.AssetType.Stock,
            multiplier=1.0,
            margin_ratio=1.0,
            tick_size=0.01,
            lot_size=1.0,
        )
    )
    feed = akquant.DataFeed()
    feed.add_tick(
        akquant.Tick(
            _ns(datetime(2024, 1, 3, 15, 0, tzinfo=timezone.utc)),
            100.0,
            1.0,
            symbol,
        )
    )
    feed.sort()
    engine.add_data(feed)

    akquant.configure_logging(
        akquant.LogConfig(
            console=False,
            level="WARNING",
        )
    )

    with caplog.at_level(logging.WARNING, logger="akquant"):
        summary = engine.run(TickIOCNoFillStrategy(), False)

    assert isinstance(summary, str)
    matching_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.execution.common"
        and "Cancelled IOC order because it was not filled on current event"
        in record.getMessage()
    )
    assert matching_record.phase == "execution"
    assert matching_record.symbol == symbol
    assert matching_record.event_time_iso == "2024-01-03T15:00:00Z"
    assert getattr(matching_record, "order_id", None)

    akquant.register_logger(console=False, level="INFO")


def test_run_backtest_stop_limit_deferred_bridges_rust_warning(caplog: Any) -> None:
    """Triggered stop-limit deferrals should bridge structured Rust warnings."""

    class StopLimitDeferredStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self._submitted = False

        def on_bar(self, bar: akquant.Bar) -> None:
            if self._submitted:
                return
            self.buy(
                symbol=bar.symbol,
                quantity=10,
                price=9.5,
                trigger_price=10.0,
            )
            self._submitted = True

    ts = _ns(datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc))
    bars = [
        akquant.Bar(
            ts,
            9.0,
            11.0,
            9.0,
            9.8,
            1000.0,
            "STOP_LIMIT_WARN",
        )
    ]

    akquant.configure_logging(
        akquant.LogConfig(
            console=False,
            level="WARNING",
        )
    )

    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = akquant.run_backtest(
            strategy=StopLimitDeferredStrategy,
            data=bars,
            symbols="STOP_LIMIT_WARN",
            show_progress=False,
            fill_policy=CurrentClose(),
        )

    assert result is not None
    matching_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.execution.common"
        and "Deferred triggered stop-limit order because trigger price 10"
        in record.getMessage()
    )
    assert matching_record.phase == "execution"
    assert matching_record.symbol == "STOP_LIMIT_WARN"
    assert matching_record.event_time_iso == "2024-01-02T15:00:00Z"
    assert getattr(matching_record, "order_id", None)

    akquant.register_logger(console=False, level="INFO")


def test_engine_csv_feed_bridges_rust_warning(caplog: Any, tmp_path: Path) -> None:
    """CSV-backed Rust warnings should enter the Python logging pipeline."""
    akquant.configure_logging(
        akquant.LogConfig(
            console=False,
            level="WARNING",
        )
    )

    csv_path = tmp_path / "bars.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerow([1, float("nan"), 10.0, 9.0, 9.5, 100.0])

    feed = akquant.DataFeed.from_csv(str(csv_path), "AAPL")
    engine = akquant.Engine()
    engine.add_data(feed)

    with caplog.at_level(logging.WARNING, logger="akquant"):
        summary = engine.run(NoopStrategy(), False)

    assert isinstance(summary, str)
    matching_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.data.client"
        and "Invalid open price NaN, defaulting to 0.0" in record.getMessage()
    )
    assert matching_record.phase == "data"
    assert matching_record.symbol == "AAPL"
    assert matching_record.event_time_iso == "1970-01-01T00:00:01Z"
    assert any(
        record.name == "akquant.data.client"
        and "Invalid open price NaN, defaulting to 0.0" in record.getMessage()
        for record in caplog.records
    )

    akquant.register_logger(console=False, level="INFO")


def test_rust_warning_json_output_includes_structured_context(tmp_path: Path) -> None:
    """Rust warnings should populate structured JSON fields after context extraction."""
    log_file = tmp_path / "rust-warning.json"
    akquant.configure_logging(
        akquant.LogConfig(
            console=False,
            filename=str(log_file),
            file_json=True,
            file_level="WARNING",
        )
    )

    _ = akquant.from_arrays(
        np.array([1], dtype=np.int64),
        np.array([np.nan], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        "AAPL",
        None,
        None,
    )

    payload = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert payload["logger"] == "akquant.data.columns"
    assert payload["message"] == "Invalid open price NaN, defaulting to 0.0"
    assert payload["phase"] == "data"
    assert payload["symbol"] == "AAPL"
    assert payload["event_time_iso"] == "1970-01-01T00:00:00.000000001Z"

    akquant.register_logger(console=False, level="INFO")


def test_rust_warning_live_profile_renders_structured_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Live profile should render Rust warning context in plain text output."""
    akquant.configure_logging(
        akquant.LogConfig(profile="live", console=True, level="WARNING")
    )

    _ = akquant.from_arrays(
        np.array([1], dtype=np.int64),
        np.array([np.nan], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        "AAPL",
        None,
        None,
    )

    captured = capsys.readouterr()
    assert "akquant.data.columns" in captured.out
    assert "Invalid open price NaN, defaulting to 0.0" in captured.out
    assert "phase=data" in captured.out
    assert "symbol=AAPL" in captured.out
    assert "event_time_iso=1970-01-01T00:00:00.000000001Z" in captured.out

    akquant.register_logger(console=False, level="INFO")


def test_get_logger_uses_null_handler_when_unconfigured() -> None:
    """The root AKQuant logger should stay quiet until handlers are configured."""
    import akquant.log as aklog

    logger = logging.getLogger("akquant")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    aklog.Logger._instance = None

    configured_logger = aklog.get_logger()

    assert configured_logger is logger
    assert any(isinstance(handler, logging.NullHandler) for handler in logger.handlers)
    assert not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler | logging.NullHandler)
        for handler in logger.handlers
    )


def test_rust_context_parser_ignores_malformed_payload() -> None:
    """Malformed Rust context payloads should not corrupt the rendered message."""
    import akquant.log as aklog

    message = 'broken-marker [akq_ctx={"phase":"data"'

    stripped, payload = aklog._parse_rust_context_message(message)

    assert stripped == message
    assert payload is None


def test_non_akquant_logger_keeps_literal_rust_context_marker() -> None:
    """Only AKQuant loggers should decode the embedded Rust context payload."""
    import akquant.log as aklog

    record = logging.LogRecord(
        name="thirdparty.lib",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg='raw-message [akq_ctx={"phase":"data","symbol":"AAPL"}]',
        args=(),
        exc_info=None,
    )

    aklog._extract_rust_context(record)

    assert (
        record.getMessage() == 'raw-message [akq_ctx={"phase":"data","symbol":"AAPL"}]'
    )
    assert not hasattr(record, "phase")
    assert not hasattr(record, "symbol")


def test_rust_context_parser_extracts_execution_order_fields() -> None:
    """AKQuant Rust payloads should restore execution/order business fields."""
    import akquant.log as aklog

    record = logging.LogRecord(
        name="akquant.execution.simulated",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg=(
            "Rejected order due to insufficient margin during execution "
            '[akq_ctx={"phase":"execution","symbol":"OPT_P","order_id":"ord-1",'
            '"strategy_id":"alpha","slot":"alpha",'
            '"event_time_iso":"1970-01-01T00:00:01Z"}]'
        ),
        args=(),
        exc_info=None,
    )

    aklog._extract_rust_context(record)
    typed_record = cast(Any, record)

    assert (
        record.getMessage()
        == "Rejected order due to insufficient margin during execution"
    )
    assert typed_record.phase == "execution"
    assert typed_record.symbol == "OPT_P"
    assert typed_record.order_id == "ord-1"
    assert typed_record.strategy_id == "alpha"
    assert typed_record.slot == "alpha"
    assert typed_record.event_time_iso == "1970-01-01T00:00:01Z"


def test_rust_context_parser_extracts_expired_order_fields() -> None:
    """Expired-order Rust payloads should restore execution/order business fields."""
    import akquant.log as aklog

    record = logging.LogRecord(
        name="akquant.execution.simulated",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg=(
            "Expired day order at session close "
            '[akq_ctx={"phase":"execution","symbol":"AAPL","order_id":"ord-exp",'
            '"strategy_id":"beta","slot":"beta",'
            '"event_time_iso":"1970-01-01T00:00:02Z"}]'
        ),
        args=(),
        exc_info=None,
    )

    aklog._extract_rust_context(record)
    typed_record = cast(Any, record)

    assert record.getMessage() == "Expired day order at session close"
    assert typed_record.phase == "execution"
    assert typed_record.symbol == "AAPL"
    assert typed_record.order_id == "ord-exp"
    assert typed_record.strategy_id == "beta"
    assert typed_record.slot == "beta"
    assert typed_record.event_time_iso == "1970-01-01T00:00:02Z"


def test_rust_context_parser_extracts_unknown_cancel_fields() -> None:
    """Unknown-cancel Rust payloads should restore execution/order fields."""
    import akquant.log as aklog

    record = logging.LogRecord(
        name="akquant.execution.simulated",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg=(
            "Ignored cancel request for unknown order "
            '[akq_ctx={"phase":"execution","order_id":"ghost-order"}]'
        ),
        args=(),
        exc_info=None,
    )

    aklog._extract_rust_context(record)
    typed_record = cast(Any, record)

    assert record.getMessage() == "Ignored cancel request for unknown order"
    assert typed_record.phase == "execution"
    assert typed_record.order_id == "ghost-order"
    assert typed_record.symbol is None


def test_rust_context_parser_extracts_non_cancellable_cancel_fields() -> None:
    """Non-cancellable cancel Rust payloads should restore full order context."""
    import akquant.log as aklog

    record = logging.LogRecord(
        name="akquant.execution.simulated",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg=(
            "Ignored cancel request because order is not cancellable in status Filled "
            '[akq_ctx={"phase":"execution","symbol":"AAPL","order_id":"ord-cancel",'
            '"strategy_id":"gamma","slot":"gamma",'
            '"event_time_iso":"1970-01-01T00:00:03Z"}]'
        ),
        args=(),
        exc_info=None,
    )

    aklog._extract_rust_context(record)
    typed_record = cast(Any, record)

    assert (
        record.getMessage()
        == "Ignored cancel request because order is not cancellable in status Filled"
    )
    assert typed_record.phase == "execution"
    assert typed_record.symbol == "AAPL"
    assert typed_record.order_id == "ord-cancel"
    assert typed_record.strategy_id == "gamma"
    assert typed_record.slot == "gamma"
    assert typed_record.event_time_iso == "1970-01-01T00:00:03Z"


def test_rust_context_parser_extracts_deferred_same_cycle_order_fields() -> None:
    """Deferred same-cycle Rust payloads should restore full order context."""
    import akquant.log as aklog

    record = logging.LogRecord(
        name="akquant.execution.simulated",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg=(
            "Deferred same-cycle order until cross-symbol reduce-first orders "
            "finish in current slice "
            '[akq_ctx={"phase":"execution","symbol":"MSFT","order_id":"ord-defer",'
            '"strategy_id":"delta","slot":"delta",'
            '"event_time_iso":"1970-01-01T00:00:04Z"}]'
        ),
        args=(),
        exc_info=None,
    )

    aklog._extract_rust_context(record)
    typed_record = cast(Any, record)

    assert (
        record.getMessage()
        == "Deferred same-cycle order until cross-symbol reduce-first orders "
        "finish in current slice"
    )
    assert typed_record.phase == "execution"
    assert typed_record.symbol == "MSFT"
    assert typed_record.order_id == "ord-defer"
    assert typed_record.strategy_id == "delta"
    assert typed_record.slot == "delta"
    assert typed_record.event_time_iso == "1970-01-01T00:00:04Z"


def test_rust_context_parser_extracts_ioc_cancel_fields() -> None:
    """IOC cancel Rust payloads should restore full order context."""
    import akquant.log as aklog

    record = logging.LogRecord(
        name="akquant.execution.common",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg=(
            "Cancelled IOC order because it was not filled on current event "
            '[akq_ctx={"phase":"execution","symbol":"IOC_WARN","order_id":"ord-ioc",'
            '"strategy_id":"epsilon","slot":"epsilon",'
            '"event_time_iso":"1970-01-01T00:00:05Z"}]'
        ),
        args=(),
        exc_info=None,
    )

    aklog._extract_rust_context(record)
    typed_record = cast(Any, record)

    assert (
        record.getMessage()
        == "Cancelled IOC order because it was not filled on current event"
    )
    assert typed_record.phase == "execution"
    assert typed_record.symbol == "IOC_WARN"
    assert typed_record.order_id == "ord-ioc"
    assert typed_record.strategy_id == "epsilon"
    assert typed_record.slot == "epsilon"
    assert typed_record.event_time_iso == "1970-01-01T00:00:05Z"


def test_rust_context_parser_extracts_stop_limit_deferred_fields() -> None:
    """Stop-limit deferral Rust payloads should restore full order context."""
    import akquant.log as aklog

    record = logging.LogRecord(
        name="akquant.execution.common",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg=(
            "Deferred triggered stop-limit order because trigger price 10 breached "
            'limit price 9.5 during in-bar activation [akq_ctx={"phase":"execution",'
            '"symbol":"STOP_LIMIT_WARN","order_id":"ord-stop","strategy_id":"zeta",'
            '"slot":"zeta","event_time_iso":"1970-01-01T00:00:06Z"}]'
        ),
        args=(),
        exc_info=None,
    )

    aklog._extract_rust_context(record)
    typed_record = cast(Any, record)

    assert (
        record.getMessage()
        == "Deferred triggered stop-limit order because trigger price 10 breached "
        "limit price 9.5 during in-bar activation"
    )
    assert typed_record.phase == "execution"
    assert typed_record.symbol == "STOP_LIMIT_WARN"
    assert typed_record.order_id == "ord-stop"
    assert typed_record.strategy_id == "zeta"
    assert typed_record.slot == "zeta"
    assert typed_record.event_time_iso == "1970-01-01T00:00:06Z"


def test_run_grid_search_parallel_warns_worker_log_visibility(caplog: Any) -> None:
    """Parallel grid search should log worker visibility warning once."""
    data = _build_benchmark_data(n=20, symbol="OPT_LOG_VISIBILITY")
    with caplog.at_level(logging.WARNING, logger="akquant"):
        _ = akquant.run_grid_search(
            strategy=NoopStrategy,
            param_grid={"dummy": [1, 2]},
            data=data,
            symbol="OPT_LOG_VISIBILITY",
            max_workers=2,
            return_df=True,
            show_progress=False,
        )
    assert "self.log() output may not be visible" in caplog.text


def test_run_grid_search_logs_dynamic_warmup_failures(caplog: Any) -> None:
    """Dynamic warmup calculation failures should flow through akquant.optimize."""
    data = _build_benchmark_data(n=20, symbol="OPT_WARMUP_WARN")

    def _broken_warmup(_: dict[str, Any]) -> int:
        raise RuntimeError("warmup failed")

    with caplog.at_level(logging.WARNING, logger="akquant.optimize"):
        _ = akquant.run_grid_search(
            strategy=NoopStrategy,
            param_grid={},
            data=data,
            symbol="OPT_WARMUP_WARN",
            max_workers=1,
            return_df=True,
            warmup_calc=_broken_warmup,
            show_progress=False,
        )

    assert "Failed to calculate dynamic warmup period: warmup failed" in caplog.text


def test_run_backtest_warns_on_suspicious_global_slippage(caplog: Any) -> None:
    """Large float slippage should warn because AKQuant treats it as percent."""
    data = _build_benchmark_data(n=5, symbol="SLIP_WARN")

    with pytest.warns(DeprecationWarning):
        with caplog.at_level(logging.WARNING, logger="akquant"):
            _ = akquant.run_backtest(
                strategy=NoopStrategy,
                data=data,
                symbols="SLIP_WARN",
                slippage=0.2,
                show_progress=False,
            )

    assert "Global slippage=0.2 uses percent semantics in AKQuant" in caplog.text
    assert "slippage={'type': 'fixed', 'value': 0.2}" in caplog.text
    assert any(record.name == "akquant.backtest" for record in caplog.records)


def test_run_backtest_does_not_warn_on_small_global_slippage(caplog: Any) -> None:
    """Typical bps-scale slippage should not trigger the suspicious warning."""
    data = _build_benchmark_data(n=5, symbol="SLIP_OK")

    with pytest.warns(DeprecationWarning):
        with caplog.at_level(logging.WARNING, logger="akquant"):
            _ = akquant.run_backtest(
                strategy=NoopStrategy,
                data=data,
                symbols="SLIP_OK",
                slippage=0.0002,
                show_progress=False,
            )

    assert "uses percent semantics in AKQuant" not in caplog.text


def test_run_backtest_live_profile_renders_risk_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Live profile should render structured context for risk warnings."""
    akquant.configure_logging(
        akquant.LogConfig(profile="live", console=True, level="INFO")
    )

    _ = akquant.run_backtest(
        data=_build_regression_bars("RISK_CONTEXT_LIVE"),
        strategy=SingleBuyStrategy,
        symbols="RISK_CONTEXT_LIVE",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        risk_config={"unknown_key": 1},
    )

    captured = capsys.readouterr()
    assert "akquant.backtest" in captured.out
    assert "Unknown risk config key: unknown_key" in captured.out
    assert "phase=risk" in captured.out
    assert "strategy_id=alpha" in captured.out
    assert "slot=alpha" in captured.out


def test_run_grid_search_parallel_forward_worker_logs_suppresses_visibility_warning(
    caplog: Any,
) -> None:
    """Forwarded worker logs should suppress visibility warning text."""
    akquant.register_logger(console=True, level="INFO")
    data = _build_benchmark_data(n=20, symbol="OPT_LOG_VISIBILITY_FORWARD")
    with caplog.at_level(logging.WARNING, logger="akquant"):
        _ = akquant.run_grid_search(
            strategy=NoopStrategy,
            param_grid={"dummy": [1, 2]},
            data=data,
            symbol="OPT_LOG_VISIBILITY_FORWARD",
            max_workers=2,
            return_df=True,
            show_progress=False,
            forward_worker_logs=True,
        )
    assert "self.log() output may not be visible" not in caplog.text


def test_run_grid_search_parallel_forward_worker_logs_warns_no_handler(
    caplog: Any,
) -> None:
    """Forwarding should warn explicitly when main process has no active handler."""
    akquant.register_logger(console=False, level="INFO")
    data = _build_benchmark_data(n=20, symbol="OPT_LOG_NO_HANDLER")
    with caplog.at_level(logging.WARNING, logger="akquant"):
        _ = akquant.run_grid_search(
            strategy=NoopStrategy,
            param_grid={"dummy": [1, 2]},
            data=data,
            symbol="OPT_LOG_NO_HANDLER",
            max_workers=2,
            return_df=True,
            show_progress=False,
            forward_worker_logs=True,
        )
    assert "forward_worker_logs=True but no active logger handler" in caplog.text
    assert "self.log() output may not be visible" not in caplog.text
    akquant.register_logger(console=True, level="INFO")


def test_run_grid_search_parallel_forward_worker_logs_to_main_process(
    tmp_path: Path,
) -> None:
    """Parallel optimization should forward worker strategy logs when enabled."""
    log_file = tmp_path / "parallel_worker_logs.log"
    akquant.register_logger(filename=str(log_file), console=False, level="INFO")
    data = _build_benchmark_data(n=20, symbol="OPT_LOG_FORWARD")
    _ = akquant.run_grid_search(
        strategy=WorkerLogStrategy,
        param_grid={"dummy": [1, 2]},
        data=data,
        symbol="OPT_LOG_FORWARD",
        max_workers=2,
        return_df=True,
        show_progress=False,
        forward_worker_logs=True,
    )
    time.sleep(0.2)
    logs_text = log_file.read_text(encoding="utf-8")
    assert "worker-log-1" in logs_text
    assert "worker-log-2" in logs_text
    akquant.register_logger(console=True, level="INFO")


def test_run_backtest_strict_default_does_not_inject_time_kwargs() -> None:
    """Default strict mode should not treat time filters as constructor kwargs."""

    class StrictNoTimeInitStrategy(akquant.Strategy):
        def on_bar(self, bar: akquant.Bar) -> None:
            return

    symbol = "STRICT_DEFAULT_TIME_FILTER"
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2023-01-01 00:00:00+00:00", periods=3, freq="D", tz="UTC"
            ),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
            "symbol": [symbol, symbol, symbol],
        }
    )

    result = akquant.run_backtest(
        data=data,
        strategy=StrictNoTimeInitStrategy,
        symbols=[symbol],
        start_time="2023-01-02 00:00:00+00:00",
        end_time="2023-01-03 00:00:00+00:00",
        show_progress=False,
    )

    assert result is not None


def test_run_backtest_start_time_preserves_pre_start_history() -> None:
    """start_time should not truncate preload bars needed by history APIs."""

    class PreStartHistoryStrategy(akquant.Strategy):
        def __init__(self) -> None:
            """Initialize captured callback state."""
            super().__init__()
            self.seen_times: list[int] = []
            self.first_history: np.ndarray | None = None

        def on_start(self) -> None:
            """Enable history tracking before active callbacks begin."""
            self.set_history_depth(3)

        def on_bar(self, bar: akquant.Bar) -> None:
            """Capture active callbacks and the first available history window."""
            self.seen_times.append(bar.timestamp)
            if self.first_history is None:
                self.first_history = self.get_history(count=3, symbol=bar.symbol)

    symbol = "START_HISTORY"
    bars = [
        akquant.Bar(
            _ns(datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)),
            10,
            10,
            10,
            10,
            1,
            symbol,
        ),
        akquant.Bar(
            _ns(datetime(2023, 1, 3, 15, 0, tzinfo=timezone.utc)),
            11,
            11,
            11,
            11,
            1,
            symbol,
        ),
        akquant.Bar(
            _ns(datetime(2023, 1, 4, 15, 0, tzinfo=timezone.utc)),
            12,
            12,
            12,
            12,
            1,
            symbol,
        ),
        akquant.Bar(
            _ns(datetime(2023, 1, 5, 15, 0, tzinfo=timezone.utc)),
            13,
            13,
            13,
            13,
            1,
            symbol,
        ),
    ]

    result = akquant.run_backtest(
        data=bars,
        strategy=PreStartHistoryStrategy,
        symbols=[symbol],
        start_time="2023-01-04 00:00:00+00:00",
        show_progress=False,
    )

    strategy = cast(PreStartHistoryStrategy, result.strategy)
    expected_active_times = [bars[2].timestamp, bars[3].timestamp]
    assert strategy.seen_times == expected_active_times
    assert strategy.first_history is not None
    assert np.allclose(strategy.first_history, np.array([10.0, 11.0, 12.0]))
    expected_start_time = pd.Timestamp(
        bars[2].timestamp, unit="ns", tz="UTC"
    ).tz_convert("Asia/Shanghai")
    assert result.metrics.start_time == expected_start_time


def test_run_backtest_accepts_camelcase_execution_mode_string() -> None:
    """run_backtest should reject removed CamelCase execution mode aliases."""
    symbol = "EXEC_CAMELCASE"
    bars = [
        akquant.Bar(
            pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value,
            10.0,
            10.0,
            10.0,
            10.0,
            1000.0,
            symbol,
        ),
        akquant.Bar(
            pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value,
            20.0,
            20.0,
            20.0,
            20.0,
            1000.0,
            symbol,
        ),
    ]
    strategy = BarOnlyCaptureStrategy()

    with pytest.raises(
        ValueError,
        match="run_backtest no longer accepts execution_mode/timer_execution_policy",
    ):
        legacy_kwargs: dict[str, Any] = {"execution_mode": "CurrentClose"}
        _ = akquant.run_backtest(
            data=bars,
            strategy=strategy,
            symbols=[symbol],
            initial_cash=100000.0,
            show_progress=False,
            **legacy_kwargs,
        )


def test_run_grid_search_single_worker_accepts_camelcase_execution_mode() -> None:
    """Grid search should surface error for removed CamelCase execution mode."""
    symbol = "OPT_EXEC_CAMELCASE"
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="min", tz="UTC"),
            "open": [10.0, 20.0],
            "high": [10.0, 20.0],
            "low": [10.0, 20.0],
            "close": [10.0, 20.0],
            "volume": [1000.0, 1000.0],
            "symbol": [symbol, symbol],
        }
    )

    with pytest.raises(
        ValueError,
        match="run_grid_search no longer accepts execution_mode/timer_execution_policy",
    ):
        _ = akquant.run_grid_search(
            strategy=SingleBuyStrategy,
            param_grid={},
            data=data,
            symbols=[symbol],
            execution_mode="CurrentClose",
            initial_cash=15.0,
            max_workers=1,
            return_df=True,
            show_progress=False,
        )


def test_run_grid_search_external_strategy_current_close_effective(
    tmp_path: Path,
) -> None:
    """External imported strategy should surface removed execution_mode error."""
    module_path = tmp_path / "external_strategy_module.py"
    module_path.write_text(
        "\n".join(
            [
                "import akquant",
                "",
                "class ExternalSingleBuyStrategy(akquant.Strategy):",
                "    def __init__(self, dummy: int = 0) -> None:",
                "        super().__init__()",
                "        self.dummy = int(dummy)",
                "        self._submitted = False",
                "",
                "    def on_bar(self, bar: akquant.Bar) -> None:",
                "        if self._submitted:",
                "            return",
                "        self.buy(symbol=bar.symbol, quantity=1)",
                "        self._submitted = True",
            ]
        ),
        encoding="utf-8",
    )
    spec = spec_from_file_location("external_strategy_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy_cls = getattr(module, "ExternalSingleBuyStrategy")

    symbol = "OPT_EXTERNAL_CURRENT_CLOSE"
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="min", tz="UTC"),
            "open": [10.0, 20.0],
            "high": [10.0, 20.0],
            "low": [10.0, 20.0],
            "close": [10.0, 20.0],
            "volume": [1000.0, 1000.0],
            "symbol": [symbol, symbol],
        }
    )

    with pytest.raises(
        ValueError,
        match="run_grid_search no longer accepts execution_mode/timer_execution_policy",
    ):
        _ = akquant.run_grid_search(
            strategy=cast(type[akquant.Strategy], strategy_cls),
            param_grid={"dummy": [1]},
            data=data,
            symbols=[symbol],
            execution_mode="current_close",
            initial_cash=15.0,
            max_workers=1,
            return_df=True,
            show_progress=False,
        )


def test_run_grid_search_db_path_serializes_timestamp_metrics(
    tmp_path: Path,
) -> None:
    """Grid search cache should serialize Timestamp metrics into JSON strings."""
    import json
    import sqlite3

    symbol = "OPT_DB_TS_SERIALIZE"
    data = _build_benchmark_data(n=40, symbol=symbol)
    db_path = tmp_path / "walk_forward_cache.db"

    results = akquant.run_grid_search(
        strategy=NoopStrategy,
        param_grid={"dummy": [1]},
        data=data,
        symbol=symbol,
        max_workers=1,
        return_df=True,
        show_progress=False,
        db_path=str(db_path),
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 1

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT metrics_json FROM optimization_results WHERE strategy_name = ?",
            (NoopStrategy.__name__,),
        ).fetchone()

    assert row is not None
    metrics = json.loads(cast(str, row[0]))
    assert isinstance(metrics.get("start_time"), str)
    assert isinstance(metrics.get("end_time"), str)


def test_run_grid_search_infers_symbols_from_dict_data() -> None:
    """Grid search should infer symbols from dict-form multisymbol data."""
    data = {
        "OPT_DICT_A": _build_benchmark_data(n=40, symbol="OPT_DICT_A"),
        "OPT_DICT_B": _build_benchmark_data(n=40, symbol="OPT_DICT_B"),
    }

    results = akquant.run_grid_search(
        strategy=NoopStrategy,
        param_grid={"dummy": [1]},
        data=data,
        max_workers=1,
        return_df=True,
        show_progress=False,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 1
    assert float(results.iloc[0]["total_bars"]) > 0.0
    assert pd.isna(results.iloc[0].get("error"))


def test_run_grid_search_dict_data_rejects_missing_symbols() -> None:
    """Grid search should fail fast when requested symbols are absent."""
    data = {
        "OPT_DICT_A": _build_benchmark_data(n=20, symbol="OPT_DICT_A"),
        "OPT_DICT_B": _build_benchmark_data(n=20, symbol="OPT_DICT_B"),
    }

    with pytest.raises(
        ValueError,
        match="Requested symbols are not available in optimization data",
    ):
        _ = akquant.run_grid_search(
            strategy=NoopStrategy,
            param_grid={"dummy": [1]},
            data=data,
            symbols=["OPT_DICT_C"],
            max_workers=1,
            return_df=True,
            show_progress=False,
        )


def test_run_walk_forward_supports_multisymbol_dict_data() -> None:
    """Walk-forward should slice dict-form multisymbol data by timeline."""
    data = {
        "WFO_DICT_A": _build_benchmark_data(n=24, symbol="WFO_DICT_A"),
        "WFO_DICT_B": _build_benchmark_data(n=24, symbol="WFO_DICT_B"),
    }

    results = akquant.run_walk_forward(
        strategy=NoopStrategy,
        param_grid={"dummy": [1]},
        data=data,
        train_period=10,
        test_period=5,
        initial_cash=100_000.0,
        max_tasks_per_child=1,
        show_progress=False,
    )

    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert "equity" in results.columns
    assert "train_start" in results.columns
    assert "train_end" in results.columns
    assert results["train_start"].iloc[0] < results["train_end"].iloc[0]


def test_run_walk_forward_multisymbol_dataframe_uses_timestamp_windows() -> None:
    """Walk-forward should slice multisymbol DataFrame input by unique timestamps."""
    symbols = ["WFO_DF_A", "WFO_DF_B"]
    data = _build_multisymbol_benchmark_data(n_timestamps=16, symbols=symbols)

    results = akquant.run_walk_forward(
        strategy=NoopStrategy,
        param_grid={"dummy": [1]},
        data=data,
        train_period=5,
        test_period=3,
        initial_cash=100_000.0,
        timezone="UTC",
        show_progress=False,
    )

    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    first_train_start = pd.Timestamp(results["train_start"].iloc[0])
    first_train_end = pd.Timestamp(results["train_end"].iloc[0])
    assert first_train_start == pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
    assert first_train_end == pd.Timestamp("2020-01-01 00:04:00", tz="UTC")


def test_run_walk_forward_filters_warmup_period_from_oos_equity() -> None:
    """Walk-forward output should exclude warmup timestamps from returned OOS curve."""
    symbol = "WFO_WARMUP_BOUNDARY"
    data = _build_benchmark_data(n=14, symbol=symbol)

    results = akquant.run_walk_forward(
        strategy=NoopStrategy,
        param_grid={"dummy": [1]},
        data=data,
        train_period=6,
        test_period=3,
        warmup_period=2,
        initial_cash=100_000.0,
        timezone="UTC",
        show_progress=False,
    )

    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    first_result_time = pd.Timestamp(results.index.min())
    assert first_result_time == pd.Timestamp("2020-01-01 00:06:00", tz="UTC")


def test_on_train_signal_runs_after_on_bar_for_trigger_bar() -> None:
    """Rolling training should execute after the trigger bar callback."""

    class RollingOrderProbeStrategy(akquant.Strategy):
        """Capture callback ordering for rolling training."""

        def __init__(self) -> None:
            """Initialize rolling callback probe state."""
            super().__init__()
            self.set_rolling_window(train_window=4, step=2)
            self.warmup_period = 4
            self.events: list[tuple[str, int]] = []

        def on_bar(self, bar: akquant.Bar) -> None:
            """Record bar callback order."""
            self.events.append(("bar", int(bar.close)))

        def on_train_signal(self, context: Any) -> None:
            """Record train callback order using the rolling window tail."""
            df, _ = self.get_rolling_data()
            closes = df["close"].dropna().astype(int).tolist()
            self.events.append(("train", int(closes[-1])))

    symbol = "ROLLING_ORDER"
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=8, freq="min", tz="UTC"),
            "open": np.arange(1, 9, dtype=float),
            "high": np.arange(1, 9, dtype=float),
            "low": np.arange(1, 9, dtype=float),
            "close": np.arange(1, 9, dtype=float),
            "volume": np.full(8, 1000.0),
            "symbol": [symbol] * 8,
        }
    )
    strategy = RollingOrderProbeStrategy()

    _ = akquant.run_backtest(
        data=data,
        strategy=strategy,
        symbols=[symbol],
        history_depth=4,
        show_progress=False,
    )

    assert strategy.events == [
        ("bar", 4),
        ("train", 4),
        ("bar", 5),
        ("bar", 6),
        ("train", 6),
        ("bar", 7),
        ("bar", 8),
        ("train", 8),
    ]


def test_ml_validation_training_schedule_uses_relative_rolling_step() -> None:
    """ML validation should retrain relative to first eligible train bar."""
    from akquant.ml.model import QuantModel, ValidationConfig

    class RecordingModel(QuantModel):
        """Minimal model stub that records fit calls."""

        fit_sizes: list[int] = []

        def __init__(self) -> None:
            """Initialize validation config and fit recorder."""
            super().__init__()
            self.validation_config = ValidationConfig(
                train_window=5,
                test_window=2,
                rolling_step=3,
                frequency="1m",
            )

        def clone(self) -> "RecordingModel":
            """Clone the test model while preserving validation config."""
            cloned = RecordingModel()
            cloned.validation_config = self.validation_config
            return cloned

        def fit(self, X: Any, y: Any) -> None:
            """Record fit sample size."""
            RecordingModel.fit_sizes.append(int(len(X)))

        def predict(self, X: Any) -> np.ndarray:
            """Return a deterministic empty prediction vector."""
            return np.zeros(len(X))

        def save(self, path: str) -> None:
            """Satisfy abstract model API for tests."""
            return

        def load(self, path: str) -> None:
            """Satisfy abstract model API for tests."""
            return

    class ValidationScheduleStrategy(akquant.Strategy):
        """Capture ML train bars under validation config."""

        def __init__(self) -> None:
            """Initialize model and training recorder."""
            super().__init__()
            self.model = RecordingModel()
            self.train_bars: list[int] = []

        def prepare_features(
            self, df: pd.DataFrame, mode: str = "training"
        ) -> tuple[pd.DataFrame, pd.Series]:
            """Return simple close-only features and aligned labels."""
            features = pd.DataFrame({"close": df["close"].fillna(0.0)})
            labels = pd.Series(np.zeros(len(features), dtype=int))
            return features, labels

        def on_bar(self, bar: akquant.Bar) -> None:
            """Ignore trading logic for schedule test."""
            return

        def on_train_signal(self, context: Any) -> None:
            """Record train bar index and delegate to default fit logic."""
            self.train_bars.append(int(self._bar_count))
            super().on_train_signal(context)

    symbol = "ML_RELATIVE_STEP"
    data = _build_benchmark_data(n=12, symbol=symbol)
    strategy = ValidationScheduleStrategy()

    _ = akquant.run_backtest(
        data=data,
        strategy=strategy,
        symbols=[symbol],
        show_progress=False,
    )

    assert strategy.train_bars == [5, 8, 11]
    assert RecordingModel.fit_sizes == [5, 5, 5]


def test_ml_validation_uses_test_window_when_rolling_step_is_zero() -> None:
    """ML validation should fall back to test_window when rolling_step is zero."""
    from akquant.ml.model import QuantModel, ValidationConfig

    class RecordingModel(QuantModel):
        """Minimal model stub that records fit calls."""

        fit_sizes: list[int] = []

        def __init__(self) -> None:
            """Initialize validation config and fit recorder."""
            super().__init__()
            self.validation_config = ValidationConfig(
                train_window=4,
                test_window=2,
                rolling_step=0,
                frequency="1m",
            )

        def clone(self) -> "RecordingModel":
            """Clone the test model while preserving validation config."""
            cloned = RecordingModel()
            cloned.validation_config = self.validation_config
            return cloned

        def fit(self, X: Any, y: Any) -> None:
            """Record fit sample size."""
            RecordingModel.fit_sizes.append(int(len(X)))

        def predict(self, X: Any) -> np.ndarray:
            """Return a deterministic empty prediction vector."""
            return np.zeros(len(X))

        def save(self, path: str) -> None:
            """Satisfy abstract model API for tests."""
            return

        def load(self, path: str) -> None:
            """Satisfy abstract model API for tests."""
            return

    class TestWindowFallbackStrategy(akquant.Strategy):
        """Capture ML train bars under test_window fallback scheduling."""

        def __init__(self) -> None:
            """Initialize model and training recorder."""
            super().__init__()
            self.model = RecordingModel()
            self.train_bars: list[int] = []

        def prepare_features(
            self, df: pd.DataFrame, mode: str = "training"
        ) -> tuple[pd.DataFrame, pd.Series]:
            """Return simple close-only features and aligned labels."""
            features = pd.DataFrame({"close": df["close"].fillna(0.0)})
            labels = pd.Series(np.zeros(len(features), dtype=int))
            return features, labels

        def on_bar(self, bar: akquant.Bar) -> None:
            """Ignore trading logic for schedule test."""
            return

        def on_train_signal(self, context: Any) -> None:
            """Record train bar index and delegate to default fit logic."""
            self.train_bars.append(int(self._bar_count))
            super().on_train_signal(context)

    symbol = "ML_TEST_WINDOW_STEP"
    data = _build_benchmark_data(n=8, symbol=symbol)
    strategy = TestWindowFallbackStrategy()

    _ = akquant.run_backtest(
        data=data,
        strategy=strategy,
        symbols=[symbol],
        show_progress=False,
    )

    assert strategy.train_bars == [4, 6, 8]
    assert RecordingModel.fit_sizes == [4, 4, 4]


def test_ml_validation_activates_new_model_on_next_bar() -> None:
    """Newly trained validation models should activate on the next bar."""
    from akquant.ml.model import QuantModel, ValidationConfig

    class VersionedModel(QuantModel):
        """Model stub that exposes fitted version ids in predictions."""

        next_version = 1

        def __init__(self, version: int = 0) -> None:
            """Initialize validation config and current version."""
            super().__init__()
            self.validation_config = ValidationConfig(
                train_window=4,
                test_window=2,
                rolling_step=2,
                frequency="1m",
            )
            self.version = version

        def clone(self) -> "VersionedModel":
            """Clone the model and preserve the current version state."""
            cloned = VersionedModel(version=self.version)
            cloned.validation_config = self.validation_config
            return cloned

        def fit(self, X: Any, y: Any) -> None:
            """Assign a new model version when a training window completes."""
            self.version = VersionedModel.next_version
            VersionedModel.next_version += 1

        def predict(self, X: Any) -> np.ndarray:
            """Return the current model version as prediction."""
            return np.full(len(X), self.version, dtype=float)

        def save(self, path: str) -> None:
            """Satisfy abstract model API for tests."""
            return

        def load(self, path: str) -> None:
            """Satisfy abstract model API for tests."""
            return

    class LifecycleStrategy(akquant.Strategy):
        """Capture active model versions and window metadata per bar."""

        def __init__(self) -> None:
            """Initialize model and lifecycle recorder."""
            super().__init__()
            self.model = VersionedModel()
            self.events: list[tuple[int, bool, int | None, int | None, int | None]] = []

        def prepare_features(
            self, df: pd.DataFrame, mode: str = "training"
        ) -> tuple[pd.DataFrame, pd.Series]:
            """Return simple close-only features and aligned labels."""
            features = pd.DataFrame({"close": df["close"].fillna(0.0)})
            labels = pd.Series(np.zeros(len(features), dtype=int))
            return features, labels

        def on_bar(self, bar: akquant.Bar) -> None:
            """Record the visible active model state for the current bar."""
            window = self.current_validation_window()
            version: int | None = None
            if self.is_model_ready() and self.model is not None:
                prediction = self.model.predict(pd.DataFrame({"close": [bar.close]}))
                version = int(prediction[0])
            self.events.append(
                (
                    int(self._bar_count),
                    bool(self.is_model_ready()),
                    version,
                    None if window is None else window["active_start_bar"],
                    None if window is None else window["active_end_bar"],
                )
            )

    symbol = "ML_LIFECYCLE"
    data = _build_benchmark_data(n=8, symbol=symbol)
    strategy = LifecycleStrategy()

    _ = akquant.run_backtest(
        data=data,
        strategy=strategy,
        symbols=[symbol],
        show_progress=False,
    )

    assert strategy.events == [
        (1, False, None, None, None),
        (2, False, None, None, None),
        (3, False, None, None, None),
        (4, False, None, None, None),
        (5, True, 1, 5, 6),
        (6, True, 1, 5, 6),
        (7, True, 2, 7, 8),
        (8, True, 2, 7, 8),
    ]


def test_run_backtest_expiry_date_str_is_rejected() -> None:
    """expiry_date should reject string input."""

    class Noop(akquant.Strategy):
        def on_bar(self, bar: akquant.Bar) -> None:
            return

    symbol = "OPT_EXPIRY_STR"
    bars = _build_regression_bars(symbol)
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="OPTION",
                option_type="CALL",
                strike_price=1.0,
                underlying_symbol="510050.SH",
                expiry_date=cast(Any, "2026-01-31"),
            )
        ],
    )

    with pytest.raises(TypeError, match="expiry_date no longer supports str"):
        akquant.run_backtest(
            data=bars,
            strategy=Noop,
            symbols=[symbol],
            config=config,
            show_progress=False,
        )


def test_strategy_get_instrument_config_snapshot() -> None:
    """Strategy should read instrument snapshot fields directly."""

    class CaptureInstrumentStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.snapshot: dict[str, Any] = {}

        def on_start(self) -> None:
            self.snapshot = {
                "single": self.get_instrument_field("OPT_META", "expiry_date"),
                "multi": self.get_instrument_config(
                    "OPT_META",
                    fields=[
                        "asset_type",
                        "option_type",
                        "option_margin_model",
                        "implied_volatility",
                        "reference_volatility",
                        "multiplier",
                    ],
                ),
                "all_count": len(self.get_instruments()),
                "symbol": self.get_instrument("OPT_META").symbol,
            }

        def on_bar(self, bar: akquant.Bar) -> None:
            return

    symbol = "OPT_META"
    bars = _build_regression_bars(symbol)
    strategy = CaptureInstrumentStrategy()
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="OPTION",
                option_type="CALL",
                strike_price=2.5,
                underlying_symbol="510050.SH",
                expiry_date=date(2026, 1, 31),
                option_margin_model="US_BROKER_SINGLE_LEG_VOL_ADJUSTED",
                implied_volatility=0.3,
                reference_volatility=0.2,
                multiplier=10.0,
            )
        ],
    )

    _ = akquant.run_backtest(
        data=bars,
        strategy=strategy,
        symbols=[symbol],
        config=config,
        show_progress=False,
    )

    assert strategy.snapshot["single"] == 20260131
    assert strategy.snapshot["all_count"] == 1
    assert strategy.snapshot["symbol"] == symbol
    multi = cast(dict[str, Any], strategy.snapshot["multi"])
    assert multi["asset_type"] == "OPTION"
    assert multi["option_type"] == "CALL"
    assert multi["option_margin_model"] == "US_BROKER_SINGLE_LEG_VOL_ADJUSTED"
    assert multi["implied_volatility"] == pytest.approx(0.3, rel=1e-12)
    assert multi["reference_volatility"] == pytest.approx(0.2, rel=1e-12)
    assert multi["multiplier"] == pytest.approx(10.0, rel=1e-12)


def test_run_backtest_settlement_price_mode_requires_price() -> None:
    """Futures settlement_price mode should require settlement_price."""

    class Noop(akquant.Strategy):
        def on_bar(self, bar: akquant.Bar) -> None:
            return

    symbol = "FUT_SETTLE_REQ"
    bars = _build_regression_bars(symbol)
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
                tick_size=0.2,
                expiry_date=date(2026, 1, 31),
                settlement_type="settlement_price",
            )
        ],
    )

    with pytest.raises(ValueError, match="settlement_price is required"):
        akquant.run_backtest(
            data=bars,
            strategy=Noop,
            symbols=[symbol],
            config=config,
            show_progress=False,
        )


def test_run_backtest_settlement_type_rejects_physical() -> None:
    """Futures settlement_type should reject physical mode."""
    with pytest.raises(ValueError, match="Unsupported settlement_type"):
        _ = akquant.InstrumentConfig(
            symbol="FUT_SETTLE_PHYSICAL",
            asset_type="FUTURES",
            multiplier=10.0,
            margin_ratio=0.1,
            tick_size=0.2,
            expiry_date=date(2026, 1, 31),
            settlement_type=cast(Any, "physical"),
        )


def test_run_backtest_calls_on_expiry_after_settlement() -> None:
    """Expiry callback should fire after positions are settled."""

    class ExpiryCaptureStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.expiry_events: list[dict[str, Any]] = []
            self.position_after_callback: float | None = None

        def on_bar(self, bar: akquant.Bar) -> None:
            target_ts = _ns(datetime(2026, 1, 30, 15, 0, tzinfo=timezone.utc))
            if int(bar.timestamp) == target_ts:
                self.buy(symbol=bar.symbol, quantity=1)

        def on_expiry(self, event: dict[str, Any]) -> None:
            self.expiry_events.append(dict(event))
            self.position_after_callback = self.get_position(event["symbol"])

    symbol = "FUT_EXP_CB"
    bars = [
        akquant.Bar(
            _ns(datetime(2026, 1, 30, 15, 0, tzinfo=timezone.utc)),
            100.0,
            100.0,
            100.0,
            100.0,
            1000.0,
            symbol,
        ),
        akquant.Bar(
            _ns(datetime(2026, 1, 31, 15, 0, tzinfo=timezone.utc)),
            110.0,
            110.0,
            110.0,
            110.0,
            1000.0,
            symbol,
        ),
        akquant.Bar(
            _ns(datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc)),
            109.0,
            109.0,
            109.0,
            109.0,
            1000.0,
            symbol,
        ),
    ]
    strategy = ExpiryCaptureStrategy()
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
                tick_size=0.2,
                expiry_date=date(2026, 1, 31),
                settlement_type="settlement_price",
                settlement_price=108.0,
            )
        ],
    )

    _ = akquant.run_backtest(
        data=bars,
        strategy=strategy,
        symbols=[symbol],
        config=config,
        show_progress=False,
    )

    assert len(strategy.expiry_events) == 1
    event = strategy.expiry_events[0]
    assert event["symbol"] == symbol
    assert event["asset_type"] == "FUTURES"
    assert event["expiry_date"] == 20260131
    assert event["settlement_type"] == "settlement_price"
    assert event["settlement_price"] == pytest.approx(108.0, rel=1e-12)
    assert event["quantity_before"] == pytest.approx(1.0, rel=1e-12)
    assert event["quantity_closed"] == pytest.approx(1.0, rel=1e-12)
    assert event["cash_flow"] == pytest.approx(1080.0, rel=1e-12)
    assert strategy.position_after_callback == pytest.approx(0.0, rel=1e-12)


def test_run_backtest_on_event_emits_expiry_event() -> None:
    """Unified stream should emit expiry events."""

    class BuyOnceStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.submitted = False

        def on_bar(self, bar: akquant.Bar) -> None:
            if not self.submitted:
                self.buy(symbol=bar.symbol, quantity=1)
                self.submitted = True

    symbol = "FUT_EXP_STREAM"
    bars = [
        akquant.Bar(
            _ns(datetime(2026, 1, 30, 15, 0, tzinfo=timezone.utc)),
            100.0,
            100.0,
            100.0,
            100.0,
            1000.0,
            symbol,
        ),
        akquant.Bar(
            _ns(datetime(2026, 1, 31, 15, 0, tzinfo=timezone.utc)),
            111.0,
            111.0,
            111.0,
            111.0,
            1000.0,
            symbol,
        ),
        akquant.Bar(
            _ns(datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc)),
            112.0,
            112.0,
            112.0,
            112.0,
            1000.0,
            symbol,
        ),
    ]
    events: list[akquant.BacktestStreamEvent] = []
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="FUTURES",
                multiplier=5.0,
                margin_ratio=0.1,
                tick_size=0.2,
                expiry_date=date(2026, 1, 31),
                settlement_type="settlement_price",
                settlement_price=110.0,
            )
        ],
    )

    _ = akquant.run_backtest(
        data=bars,
        strategy=BuyOnceStrategy,
        symbols=[symbol],
        config=config,
        on_event=events.append,
        show_progress=False,
    )

    expiry_events = [event for event in events if event["event_type"] == "expiry"]
    assert len(expiry_events) == 1
    payload = expiry_events[0]["payload"]
    assert payload["symbol"] == symbol
    assert payload["asset_type"] == "FUTURES"
    assert payload["expiry_date"] == "20260131"
    assert payload["settlement_type"] == "settlement_price"
    assert payload["settlement_price"] == "110"
    assert payload["owner_strategy_id"] == "_default"


def test_instrument_config_rejects_invalid_asset_type() -> None:
    """InstrumentConfig should reject unsupported asset_type."""
    with pytest.raises(ValueError, match="Unsupported asset_type"):
        _ = akquant.InstrumentConfig(
            symbol="BAD_ASSET",
            asset_type=cast(Any, "CRYPTO"),
        )


def test_instrument_config_rejects_invalid_option_type() -> None:
    """InstrumentConfig should reject unsupported option_type."""
    with pytest.raises(ValueError, match="Unsupported option_type"):
        _ = akquant.InstrumentConfig(
            symbol="BAD_OPT",
            asset_type="OPTION",
            option_type=cast(Any, "STRADDLE"),
        )


def test_instrument_config_rejects_invalid_option_margin_model() -> None:
    """InstrumentConfig should reject unsupported option_margin_model."""
    with pytest.raises(ValueError, match="Unsupported option_margin_model"):
        _ = akquant.InstrumentConfig(
            symbol="BAD_MARGIN_MODEL",
            asset_type="OPTION",
            option_margin_model=cast(Any, "BROKER_MAGIC"),
        )


def test_instrument_config_accepts_enum_inputs() -> None:
    """InstrumentConfig should accept public enum inputs."""
    conf = akquant.InstrumentConfig(
        symbol="ENUM_OK",
        asset_type=akquant.InstrumentAssetTypeEnum.FUTURES,
        option_type=akquant.InstrumentOptionTypeEnum.CALL,
        option_margin_model=(
            akquant.InstrumentOptionMarginModelEnum.US_BROKER_SINGLE_LEG_VOL_ADJUSTED
        ),
        settlement_type=akquant.InstrumentSettlementTypeEnum.CASH,
    )
    assert conf.asset_type == "FUTURES"
    assert conf.option_type == "CALL"
    assert conf.option_margin_model == "US_BROKER_SINGLE_LEG_VOL_ADJUSTED"
    assert conf.settlement_type == "cash"


def test_order_rejects_non_positive_quantity() -> None:
    """Order should reject zero quantity at constructor boundary."""
    with pytest.raises(ValueError, match=r"AKQ-ORDER-VALIDATION.*quantity must be > 0"):
        _ = akquant.Order(
            "o-invalid-qty",
            "AAPL",
            akquant.OrderSide.Buy,
            akquant.OrderType.Limit,
            0.0,
            100.0,
        )


def test_order_and_trade_time_iso_properties() -> None:
    """Order/Trade should expose UTC ISO 8601 timestamp strings."""
    ts = pd.Timestamp("2025-01-02 15:00:00", tz="Asia/Shanghai").value
    order = akquant.Order(
        "o-time-str",
        "AAPL",
        akquant.OrderSide.Buy,
        akquant.OrderType.Limit,
        10.0,
        100.0,
        created_at=ts,
    )
    trade = akquant.Trade(
        "t-time-str",
        "o-time-str",
        "AAPL",
        akquant.OrderSide.Buy,
        10.0,
        100.0,
        1.0,
        ts,
        0,
        None,
    )

    assert order.created_at_iso == "2025-01-02T07:00:00Z"
    assert order.updated_at_iso == "2025-01-02T07:00:00Z"
    assert trade.timestamp_iso == "2025-01-02T07:00:00Z"


def test_order_and_trade_position_effect_defaults() -> None:
    """Order/Trade should expose default and explicit position_effect values."""
    order = akquant.Order(
        "o-effect",
        "AAPL",
        akquant.OrderSide.Buy,
        akquant.OrderType.Limit,
        10.0,
        100.0,
        position_effect=akquant.PositionEffect.Close,
        reduce_only=True,
    )
    trade = akquant.Trade(
        "t-effect",
        "o-effect",
        "AAPL",
        akquant.OrderSide.Buy,
        10.0,
        100.0,
        1.0,
        0,
        0,
        position_effect=akquant.PositionEffect.Close,
    )

    assert order.position_effect == akquant.PositionEffect.Close
    assert order.reduce_only is True
    assert trade.position_effect == akquant.PositionEffect.Close


def test_position_effect_extended_close_variants() -> None:
    """Order/Trade should expose close_today and close_yesterday variants."""
    order = akquant.Order(
        "o-close-today",
        "IF2406",
        akquant.OrderSide.Sell,
        akquant.OrderType.Market,
        1.0,
        position_effect=akquant.PositionEffect.CloseToday,
    )
    trade = akquant.Trade(
        "t-close-yesterday",
        "o-close-yesterday",
        "IF2406",
        akquant.OrderSide.Sell,
        1.0,
        100.0,
        1.0,
        0,
        0,
        position_effect=akquant.PositionEffect.CloseYesterday,
    )

    assert order.position_effect == akquant.PositionEffect.CloseToday
    assert trade.position_effect == akquant.PositionEffect.CloseYesterday


def test_instrument_rejects_non_positive_tick_size() -> None:
    """Instrument should reject non-positive tick size."""
    with pytest.raises(
        ValueError, match=r"AKQ-INSTRUMENT-VALIDATION.*tick_size must be > 0"
    ):
        _ = akquant.Instrument("AAPL", akquant.AssetType.Stock, tick_size=0.0)


def test_instrument_option_rejects_empty_underlying_symbol() -> None:
    """Option instrument should require non-empty underlying symbol."""
    with pytest.raises(
        ValueError,
        match=r"AKQ-INSTRUMENT-VALIDATION.*underlying_symbol must not be empty",
    ):
        _ = akquant.Instrument(
            "OPT_BAD",
            akquant.AssetType.Option,
            expiry_date=20260101,
            underlying_symbol="",
        )


def test_corporate_action_split_rejects_non_positive_ratio() -> None:
    """CorporateAction split should reject non-positive ratio."""
    with pytest.raises(
        ValueError, match=r"AKQ-CORP-ACTION-VALIDATION.*split ratio must be > 0"
    ):
        _ = akquant.CorporateAction(
            "AAPL",
            date(2025, 1, 1),
            akquant.CorporateActionType.Split,
            0.0,
        )


def test_run_backtest_same_bar_sell_funds_buy_next_open() -> None:
    """Same-bar sell frees cash for a same-bar buy under next-open fill.

    Issue #307: under default next-open fill (bar_offset=1), a same-bar sell
    must free cash so a same-bar buy of another symbol is not rejected.
    """
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    price = 10.0

    def _mk(sym: str) -> pd.DataFrame:
        df = pd.DataFrame(index=dates)
        df.index.name = "date"
        for col in ("open", "high", "low", "close"):
            df[col] = price
        df["volume"] = 10_000_000
        df["symbol"] = sym
        return df

    data = {"A": _mk("A"), "B": _mk("B")}

    class SwitchStrategy(akquant.Strategy):
        """day1 buy A; day3 (A past T+1) sell A then buy B in the SAME callback."""

        def __init__(self) -> None:
            super().__init__()
            self._days: set = set()
            self.b_filled = 0.0
            self.rejects: list = []

        def on_bar(self, bar: akquant.Bar) -> None:
            day = pd.Timestamp(bar.timestamp).normalize()
            if day in self._days:
                return
            self._days.add(day)
            idx = len(self._days)
            if idx == 1:
                self.buy(symbol="A", quantity=5000)
            elif idx == 3 and self.get_position("A") > 0:
                self.sell(symbol="A", quantity=self.get_position("A"))
                self.buy(symbol="B", quantity=5000)

        def on_trade(self, trade: akquant.Trade) -> None:
            if trade.symbol == "B":
                self.b_filled += float(trade.quantity)

        def on_reject(self, order: akquant.Order) -> None:
            self.rejects.append(order.symbol)

    strat = SwitchStrategy()
    akquant.run_backtest(
        data=data,
        strategy=strat,
        symbols=["A", "B"],
        initial_cash=100_000.0,
        commission_rate=0.0,
        t_plus_one=True,
        lot_size=100,
        show_progress=False,
    )

    assert strat.rejects == []
    assert strat.b_filled == pytest.approx(5000.0)


def test_full_switch_sell_proceeds_fund_same_bar_buy_next_open() -> None:
    """Issue #307 core: fully switching A->B in one bar must let A's sale fund B.

    Selling all of A and reinvesting the whole resulting buying power into B
    (cost > residual cash) must fill under the default next-open policy. This is
    checked for both symbol orderings so the fix does not rely on feed order.
    """
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    price = 10.0

    def _mk(sym: str) -> pd.DataFrame:
        df = pd.DataFrame(index=dates)
        df.index.name = "date"
        for col in ("open", "high", "low", "close"):
            df[col] = price
        df["volume"] = 10_000_000
        df["symbol"] = sym
        return df

    def _run(symbols: list) -> tuple:
        data = {s: _mk(s) for s in symbols}

        class SwitchStrategy(akquant.Strategy):
            """day1 buy A x5000; day3 sell all A, reinvest full buying power in B."""

            def __init__(self) -> None:
                super().__init__()
                self._days: set = set()
                self.b_filled = 0.0
                self.rejects: list = []

            def on_bar(self, bar: akquant.Bar) -> None:
                day = pd.Timestamp(bar.timestamp).normalize()
                if day in self._days:
                    return
                self._days.add(day)
                idx = len(self._days)
                if idx == 1:
                    self.buy(symbol="A", quantity=5000)
                elif idx == 3 and self.get_position("A") > 0:
                    self.sell(symbol="A", quantity=self.get_position("A"))
                    qty_b = int(self.buying_power / price / 100) * 100
                    if qty_b > 0:
                        self.buy(symbol="B", quantity=qty_b)

            def on_trade(self, trade: akquant.Trade) -> None:
                if trade.symbol == "B":
                    self.b_filled += float(trade.quantity)

            def on_reject(self, order: akquant.Order) -> None:
                self.rejects.append(order.symbol)

        strat = SwitchStrategy()
        akquant.run_backtest(
            data=data,
            strategy=strat,
            symbols=symbols,
            initial_cash=100_000.0,
            commission_rate=0.0,
            t_plus_one=True,
            lot_size=100,
            show_progress=False,
        )
        return strat.rejects, strat.b_filled

    for order in (["A", "B"], ["B", "A"]):
        rejects, b_filled = _run(order)
        assert rejects == [], f"unexpected rejects for symbol order {order}: {rejects}"
        assert b_filled >= 9800.0, f"B underfilled for symbol order {order}: {b_filled}"


def test_after_trading_next_open_order_fills_next_trading_day() -> None:
    """Issue #324: a day-end next-open order must fill on the next trading day.

    A next-open order submitted in on_after_trading(T) must fill on T+1, not T+2.
    on_after_trading is a day-end hook. In the default (lazy) dispatch mode it
    used to fire only once T+1's bar arrived, so the engine clock was already at
    T+1 and the order's created_at was stamped at T+1; the #307 next-open guard
    (`event_ts <= created_at`) then skipped the T+1 bar and pushed the fill to
    T+2. The fix schedules the day-end boundary timer for on_after_trading in the
    default mode too, so the callback fires at T's session close (created_at in
    T) and the fill lands on T+1. Checked for both dispatch modes.
    """
    dates = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"])

    def _mk() -> pd.DataFrame:
        df = pd.DataFrame(index=dates)
        df.index.name = "date"
        for col in ("open", "high", "low", "close"):
            df[col] = 1.0
        df["volume"] = 1e7
        df["symbol"] = "AAA"
        return df

    def _run(precise: bool) -> tuple:
        class AfterTradingStrategy(akquant.Strategy):
            def __init__(self) -> None:
                super().__init__()
                if precise:
                    self.enable_precise_day_boundary_hooks = True
                self.bar_ts: list[
                    int
                ] = []  # per-trading-day bar timestamps (ns), in order
                self.ordered: bool = False
                self.fill_ts: int | None = None

            def on_bar(self, bar: akquant.Bar) -> None:
                self.bar_ts.append(int(bar.timestamp))

            def on_after_trading(self, trading_date: object, timestamp: int) -> None:
                # First call is for the first trading day (bar_ts[0]).
                if not self.ordered:
                    self.ordered = True
                    self.buy(
                        "AAA",
                        10000,
                        fill_mode=NextOpen(),
                    )

            def on_trade(self, trade: akquant.Trade) -> None:
                if self.fill_ts is None:
                    self.fill_ts = int(trade.timestamp)

        strat = AfterTradingStrategy()
        akquant.run_backtest(
            data={"AAA": _mk()},
            strategy=strat,
            symbols=["AAA"],
            initial_cash=1_000_000.0,
            commission_rate=0.0,
            lot_size=100,
            t_plus_one=True,
            show_progress=False,
        )
        return strat.bar_ts, strat.fill_ts

    # Order is submitted in on_after_trading of the first trading day (bar_ts[0]);
    # with a next-open policy it must fill at the open of the next trading day
    # (bar_ts[1]). The #324 regression fills at bar_ts[2] (T+2) in default mode.
    for precise in (False, True):
        bar_ts, fill_ts = _run(precise)
        assert fill_ts is not None, f"precise={precise}: order never filled"
        fill_index = bar_ts.index(fill_ts)
        assert fill_index == 1, (
            f"precise={precise}: on_after_trading(day 0) next-open order filled on "
            f"trading day index {fill_index}, expected 1 (next trading day). "
            f"index 2 is the #324 off-by-one."
        )


def test_pre_open_default_order_fills_same_day_open() -> None:
    """Issue #324 family: on_pre_open must fill on THIS day's open, not the next.

    on_pre_open is the "decide before open, fill on this open" hook. Its pre-open
    boundary timer used to be scheduled at the day's first bar timestamp (same
    instant as that bar), so an order submitted inside it had created_at equal to
    the bar timestamp and the next-open guard pushed the fill to the following
    day's open. The fix schedules the pre-open timer strictly before the first
    bar, so a default pre-open order (open / next bar) fills on the current day's
    open as documented.
    """
    dates = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"])

    def _mk() -> pd.DataFrame:
        df = pd.DataFrame(index=dates)
        df.index.name = "date"
        df["open"] = [10.0, 11.0, 12.0, 13.0]
        df["high"] = [10.0, 11.0, 12.0, 13.0]
        df["low"] = [10.0, 11.0, 12.0, 13.0]
        df["close"] = [10.0, 11.0, 12.0, 13.0]
        df["volume"] = 1e7
        df["symbol"] = "AAA"
        return df

    class PreOpenStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.bar_ts: list[int] = []
            self.ordered: bool = False
            self.order_bar_index: int | None = None
            self.fill_ts: int | None = None
            self.fill_price: float | None = None

        def on_start(self) -> None:
            self.subscribe("AAA")

        def on_bar(self, bar: akquant.Bar) -> None:
            self.bar_ts.append(int(bar.timestamp))

        def on_pre_open(self, event: dict) -> None:
            if not self.ordered:
                self.ordered = True
                # bars seen so far == index of the day this pre_open precedes.
                self.order_bar_index = len(self.bar_ts)
                self.buy("AAA", 100)  # default pre_open policy: open / this open

        def on_trade(self, trade: akquant.Trade) -> None:
            if self.fill_ts is None:
                self.fill_ts = int(trade.timestamp)
                self.fill_price = float(trade.price)

    strat = PreOpenStrategy()
    akquant.run_backtest(
        data={"AAA": _mk()},
        strategy=strat,
        symbols=["AAA"],
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        lot_size=100,
        show_progress=False,
    )

    assert strat.ordered, "on_pre_open never fired"
    assert strat.fill_ts is not None, "pre_open order never filled"
    fill_index = strat.bar_ts.index(strat.fill_ts)
    # pre_open(day D) fires before day D's bar (order_bar_index == D); the order
    # must fill on day D's own open, not the next day's.
    assert fill_index == strat.order_bar_index, (
        f"on_pre_open order filled on bar index {fill_index}, expected "
        f"{strat.order_bar_index} (this day's open). A later index is the #324 "
        f"off-by-one; the pre_open contract is 'fill on this open'."
    )


def test_strategy_buying_power_reflects_same_bar_pending_sell() -> None:
    """buying_power must include a same-callback pending sell's expected proceeds."""
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    price = 10.0

    def _mk(sym: str) -> pd.DataFrame:
        df = pd.DataFrame(index=dates)
        df.index.name = "date"
        for col in ("open", "high", "low", "close"):
            df[col] = price
        df["volume"] = 10_000_000
        df["symbol"] = sym
        return df

    data = {"A": _mk("A"), "B": _mk("B")}

    class BpStrategy(akquant.Strategy):
        """Record buying_power before and after a same-bar sell on day 3."""

        def __init__(self) -> None:
            super().__init__()
            self._days: set = set()
            self.bp_before: float = -1.0
            self.bp_after: float = -1.0

        def on_bar(self, bar: akquant.Bar) -> None:
            day = pd.Timestamp(bar.timestamp).normalize()
            if day in self._days:
                return
            self._days.add(day)
            idx = len(self._days)
            if idx == 1:
                self.buy(symbol="A", quantity=5000)
            elif idx == 3 and self.get_position("A") > 0:
                self.bp_before = self.buying_power
                self.sell(symbol="A", quantity=self.get_position("A"))
                self.bp_after = self.buying_power

    strat = BpStrategy()
    akquant.run_backtest(
        data=data,
        strategy=strat,
        symbols=["A", "B"],
        initial_cash=100_000.0,
        commission_rate=0.0,
        t_plus_one=True,
        lot_size=100,
        show_progress=False,
    )

    # buying_power mirrors the gate's `available` = free_margin * (1 - safety_margin);
    # default safety_margin 0.0001, so 50_000 -> 49_995 and 100_000 -> 99_990.
    assert strat.bp_before == pytest.approx(49_995.0, rel=1e-6)
    # After submitting the same-bar sell of 5000@10, proceeds roughly double it.
    assert strat.bp_after == pytest.approx(99_990.0, rel=1e-6)
    assert strat.bp_after == pytest.approx(strat.bp_before * 2, rel=1e-6)


def test_instrument_config_rejects_unsupported_sellable_after_days() -> None:
    """sellable_after_days>=2 is not yet supported (needs lot-aging)."""
    from akquant.config import InstrumentConfig

    InstrumentConfig(symbol="X", asset_type="STOCK", sellable_after_days=0)
    InstrumentConfig(symbol="Y", asset_type="STOCK", sellable_after_days=1)
    with pytest.raises(ValueError, match="sellable_after_days"):
        InstrumentConfig(symbol="Z", asset_type="STOCK", sellable_after_days=2)


def test_per_symbol_sellable_after_days_t0_vs_t1() -> None:
    """A T+0 instrument becomes sellable one trading day earlier than a T+1 one.

    Uses one single-symbol backtest per rule so the fill lands on the same bar
    in both runs; multi-symbol runs can skew per-symbol fill bars.
    """
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    price = 10.0

    def _mk(sym: str) -> pd.DataFrame:
        df = pd.DataFrame(index=dates)
        df.index.name = "date"
        for col in ("open", "high", "low", "close"):
            df[col] = price
        df["volume"] = 10_000_000
        df["symbol"] = sym
        return df

    def _first_available_day(n: int) -> int:
        class AvailStrategy(akquant.Strategy):
            """Buy 5000 on day 1; record available position each trading day."""

            def __init__(self) -> None:
                super().__init__()
                self._days: set = set()
                self.avail: dict[int, float] = {}

            def on_bar(self, bar: akquant.Bar) -> None:
                day = pd.Timestamp(bar.timestamp).normalize()
                if day in self._days:
                    return
                self._days.add(day)
                idx = len(self._days)
                self.avail[idx] = self.get_available_position("X")
                if idx == 1:
                    self.buy(symbol="X", quantity=5000)

        strat = AvailStrategy()
        akquant.run_backtest(
            data={"X": _mk("X")},
            strategy=strat,
            symbols=["X"],
            initial_cash=100_000.0,
            commission_rate=0.0,
            t_plus_one=True,
            lot_size=100,
            show_progress=False,
            instruments=[
                akquant.Instrument(
                    symbol="X",
                    asset_type=akquant.AssetType.Stock,
                    lot_size=100,
                    sellable_after_days=n,
                ),
            ],
        )
        return int(min(d for d, q in strat.avail.items() if q >= 5000.0))

    t0_day = _first_available_day(0)
    t1_day = _first_available_day(1)
    # T+1 shares become sellable exactly one trading day after T+0 shares.
    assert t1_day == t0_day + 1


def test_rebalance_weights_same_cycle_cross_symbol_sell_funds_buy() -> None:
    """Cross-symbol rebalance_weights: a same-cycle sell must fund the buy (#292).

    Regression for #292 (fixed under the #307 engine work): switching weights
    from {AAA, BBB} to {BBB, CCC} liquidates AAA and buys CCC in the SAME
    rebalance. The account is ~98% invested, so the CCC buy is only affordable
    if the AAA sale proceeds are released within the same slice. Before the fix
    the buy's affordability check saw pre-sale cash and was wrongly rejected,
    leaving the portfolio short of target.

    Covers the ``rebalance_weights`` + ``temporal="same_cycle"`` (``bar_offset=0``)
    path specifically; the #307 test exercises the low-level buy/sell +
    next-open path. Prices are constant and an explicit ``price_map`` is passed
    so weight sizing matches the fill price exactly — the test isolates the
    cross-symbol cash release, not weight-to-quantity drift.
    """
    dates = pd.to_datetime(
        ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09"]
    )
    price_map = {"AAA": 1.0, "BBB": 1.0, "CCC": 1.0}

    def _mk(sym: str) -> pd.DataFrame:
        df = pd.DataFrame(index=dates)
        df.index.name = "date"
        for col in ("open", "high", "low", "close"):
            df[col] = 1.0
        df["volume"] = 10_000_000.0
        df["symbol"] = sym
        return df

    data = {sym: _mk(sym) for sym in ("AAA", "BBB", "CCC")}

    class SwitchStrategy(akquant.Strategy):
        """day1 hold AAA+BBB; day3 (past T+1) drop AAA, keep BBB, add CCC."""

        def __init__(self) -> None:
            super().__init__()
            self._days: set = set()
            self.rejects: list = []
            self.final_positions: dict = {}

        def on_bar(self, bar: akquant.Bar) -> None:
            day = pd.Timestamp(bar.timestamp).normalize()
            if day in self._days:
                return
            self._days.add(day)
            idx = len(self._days)
            if idx == 1:
                self.rebalance_weights(
                    {"AAA": 0.49, "BBB": 0.49},
                    price_map=price_map,
                    liquidate_unmentioned=True,
                )
            elif idx == 3:  # AAA/BBB are now past T+1 and sellable
                self.rebalance_weights(
                    {"BBB": 0.49, "CCC": 0.49},
                    price_map=price_map,
                    liquidate_unmentioned=True,
                )

        def on_reject(self, order: akquant.Order) -> None:
            self.rejects.append(order.symbol)

        def on_stop(self) -> None:
            self.final_positions = {
                sym: float(qty) for sym, qty in dict(self.positions).items()
            }

    strat = SwitchStrategy()
    akquant.run_backtest(
        data=data,
        strategy=strat,
        symbols=["AAA", "BBB", "CCC"],
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        t_plus_one=True,
        lot_size=100,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    # The CCC buy funded by the AAA sale must not be rejected.
    assert strat.rejects == []
    # Portfolio reaches the target: BBB kept, CCC bought, AAA fully liquidated.
    assert strat.final_positions.get("BBB") == pytest.approx(490_000.0)
    assert strat.final_positions.get("CCC") == pytest.approx(490_000.0)
    assert strat.final_positions.get("AAA", 0.0) == pytest.approx(0.0)


def _rebalance_hook_price_frames() -> dict[str, pd.DataFrame]:
    """3-symbol daily frames with distinct open/close per day for hook tests.

    Symbol "a": open_i = 10.1 + 0.1*i, close_i = open_i + 0.01. So day 1 has
    open 10.10 / close 10.11 and day 2 has open 10.20 / close 10.21 — the fill
    price alone identifies which bar an order landed on.
    """
    dates = pd.to_datetime(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
    )
    frames: dict[str, pd.DataFrame] = {}
    for sym, base in (("a", 10.1), ("b", 20.1), ("c", 30.1)):
        opens = [base + 0.1 * i for i in range(len(dates))]
        df = pd.DataFrame(index=dates)
        df.index.name = "date"
        df["open"] = opens
        df["high"] = [o + 0.02 for o in opens]
        df["low"] = [o - 0.02 for o in opens]
        df["close"] = [o + 0.01 for o in opens]
        df["volume"] = 1_000_000.0
        df["symbol"] = sym
        frames[sym] = df
    return frames


def test_on_before_trading_no_lookahead_and_precise_hook_independent_fill() -> None:
    """`on_before_trading` contract (#291): no lookahead, precise-independent.

    The `enable_precise_day_boundary_hooks` toggle must not shift the fill bar.

    * On day 1 ``get_history("a", "close")`` must NOT reveal the current-day
      close (10.11); on day 2 it reveals day 1's close, i.e. history lags by a
      trading day (the day-boundary "previous info visible" semantics).
    * Buying on the first callback under next-open fill (``bar_offset=1``) fills
      at day 2's open (10.20) regardless of the precise-hooks setting. Issue
      #291 reported precise=True vs False diverging by a day here.
    """

    class RebalanceStrategy(akquant.Strategy):
        def __init__(self, precise: bool) -> None:
            super().__init__()
            self.set_history_depth(1)
            self.enable_precise_day_boundary_hooks = precise
            self._seen: set = set()
            self.hist_by_day: dict[int, float] = {}
            self._bought = False
            self.buy_price: float | None = None

        def on_before_trading(self, trading_date: Any, timestamp: int) -> None:
            day = pd.Timestamp(trading_date).normalize()
            if day in self._seen:
                return
            self._seen.add(day)
            idx = len(self._seen)
            hist = self.get_history(count=1, symbol="a", field="close")
            self.hist_by_day[idx] = float("nan") if len(hist) == 0 else float(hist[-1])
            if not self._bought:
                self.buy("a", 1)
                self._bought = True

        def on_trade(self, trade: akquant.Trade) -> None:
            if trade.symbol == "a" and trade.side == akquant.OrderSide.Buy:
                self.buy_price = float(trade.price)

    def _run(precise: bool) -> RebalanceStrategy:
        strat = RebalanceStrategy(precise=precise)
        akquant.run_backtest(
            data=_rebalance_hook_price_frames(),
            strategy=strat,
            symbols=["a", "b", "c"],
            initial_cash=1_000_000.0,
            commission_rate=0.0,
            stamp_tax_rate=0.0,
            show_progress=False,
            fill_policy=NextOpen(),
        )
        return strat

    precise_on = _run(True)
    precise_off = _run(False)

    # No lookahead: day 1 does not expose the current-day close (10.11).
    assert np.isnan(precise_on.hist_by_day[1])
    assert np.isnan(precise_off.hist_by_day[1])
    # History lags by one trading day: day 2 sees day 1's close.
    assert precise_on.hist_by_day[2] == pytest.approx(10.11)
    assert precise_off.hist_by_day[2] == pytest.approx(10.11)
    # Precise-hooks toggle must not move the fill: both land on day 2's open.
    assert precise_on.buy_price == pytest.approx(10.20)
    assert precise_off.buy_price == pytest.approx(10.20)


def test_on_cross_section_sees_current_day_and_same_cycle_fill() -> None:
    """`on_cross_section` contract (#291): current-day visibility.

    Current-day data is visible and a same-cycle close order fills at the
    current-day close. Fires after the day's first complete cross-symbol bar
    slice, so day 1's
    ``get_history("a", "close")`` already returns 10.11; under
    ``price_basis="close", bar_offset=0, temporal="same_cycle"`` the buy fills
    at that same-bar close (10.11), not the next bar.
    """

    class AfterBarStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.set_history_depth(1)
            self._seen: set = set()
            self.hist_by_day: dict[int, float] = {}
            self._bought = False
            self.buy_price: float | None = None

        def on_cross_section(self, trading_date: Any, timestamp: int) -> None:
            day = pd.Timestamp(trading_date).normalize()
            if day in self._seen:
                return
            self._seen.add(day)
            idx = len(self._seen)
            hist = self.get_history(count=1, symbol="a", field="close")
            self.hist_by_day[idx] = float("nan") if len(hist) == 0 else float(hist[-1])
            if not self._bought:
                self.buy("a", 1)
                self._bought = True

        def on_trade(self, trade: akquant.Trade) -> None:
            if trade.symbol == "a" and trade.side == akquant.OrderSide.Buy:
                self.buy_price = float(trade.price)

    strat = AfterBarStrategy()
    akquant.run_backtest(
        data=_rebalance_hook_price_frames(),
        strategy=strat,
        symbols=["a", "b", "c"],
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    # Current-day data is visible: day 1 already sees day 1's close.
    assert strat.hist_by_day[1] == pytest.approx(10.11)
    # Same-cycle close order fills at the current-day close, not the next bar.
    assert strat.buy_price == pytest.approx(10.11)


def test_get_history_multi_matches_per_field_get_history() -> None:
    """`get_history_multi` returns per-field arrays equal to `get_history` (#288).

    One batched accessor must be behaviorally identical to calling
    `get_history` once per field — same values and same left-NaN padding when
    ``count`` exceeds the available history — so it can replace the 5 separate
    FFI calls `get_history_df` makes without changing results.
    """
    dates = pd.to_datetime(
        ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
    )
    opens = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = pd.DataFrame(index=dates)
    df.index.name = "date"
    df["open"] = opens
    df["high"] = [o + 10.0 for o in opens]
    df["low"] = [o - 1.0 for o in opens]
    df["close"] = [o + 0.5 for o in opens]
    df["volume"] = [o * 100.0 for o in opens]
    df["symbol"] = "X"

    fields = ("open", "high", "low", "close", "volume")

    class MultiHistoryStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.set_history_depth(5)
            self._bars = 0
            self.match_ok: bool | None = None
            self.pad_ok: bool | None = None

        def on_bar(self, bar: akquant.Bar) -> None:
            # Check once enough history has accumulated (on the 5th bar).
            self._bars += 1
            if self._bars != len(dates):
                return

            per_field = {
                f: self.get_history(count=3, symbol="X", field=f) for f in fields
            }
            multi = self.get_history_multi(count=3, symbol="X", fields=fields)
            self.match_ok = set(multi.keys()) == set(fields) and all(
                np.array_equal(multi[f], per_field[f], equal_nan=True) for f in fields
            )

            # Padding: request more than available -> left-padded with NaN,
            # identical between the two accessors.
            per_pad = self.get_history(count=9, symbol="X", field="close")
            multi_pad = self.get_history_multi(count=9, symbol="X", fields=("close",))
            self.pad_ok = (
                len(multi_pad["close"]) == 9
                and np.isnan(multi_pad["close"][0])
                and np.array_equal(multi_pad["close"], per_pad, equal_nan=True)
            )

    strat = MultiHistoryStrategy()
    akquant.run_backtest(
        data={"X": df},
        strategy=strat,
        symbols=["X"],
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        show_progress=False,
    )

    assert strat.match_ok is True
    assert strat.pad_ok is True


def test_get_history_multi_matches_get_history_under_before_trading_cutoff() -> None:
    """`get_history_multi` equals per-field `get_history` on the cutoff path (#288).

    In day-boundary phases (`on_before_trading`) `get_history` applies a
    history-visibility cutoff that hides the current day. `get_history_multi`
    must resolve that cutoff identically, so `get_history_df` stays correct when
    called from a day-boundary hook — not only from `on_bar`.
    """
    dates = pd.to_datetime(
        ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
    )
    # Distinct per-day closes so the cutoff (current day hidden) is observable.
    closes = [10.0, 20.0, 30.0, 40.0, 50.0]
    df = pd.DataFrame(index=dates)
    df.index.name = "date"
    df["open"] = [c + 0.1 for c in closes]
    df["high"] = [c + 1.0 for c in closes]
    df["low"] = [c - 1.0 for c in closes]
    df["close"] = closes
    df["volume"] = [c * 10.0 for c in closes]
    df["symbol"] = "X"

    fields = ("open", "high", "low", "close", "volume")

    class RebalanceHistoryStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.set_history_depth(5)
            self._days = 0
            self.match_ok: bool | None = None
            self.cutoff_active: bool | None = None

        def on_before_trading(self, trading_date: Any, timestamp: int) -> None:
            # The 4th trading day has days 1-3 visible under the cutoff.
            self._days += 1
            if self._days != 4:
                return

            per_field = {
                f: self.get_history(count=2, symbol="X", field=f) for f in fields
            }
            multi = self.get_history_multi(count=2, symbol="X", fields=fields)
            self.match_ok = set(multi.keys()) == set(fields) and all(
                np.array_equal(multi[f], per_field[f], equal_nan=True) for f in fields
            )
            # Cutoff active: current day (close 40.0) is hidden; the last visible
            # close is the previous day's (30.0), proving we exercise the cutoff
            # branch rather than the on_bar path.
            self.cutoff_active = float(per_field["close"][-1]) == pytest.approx(30.0)

    strat = RebalanceHistoryStrategy()
    akquant.run_backtest(
        data={"X": df},
        strategy=strat,
        symbols=["X"],
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        show_progress=False,
    )

    assert strat.cutoff_active is True
    assert strat.match_ok is True
