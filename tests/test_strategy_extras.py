import logging
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
from akquant.akquant import Bar, StrategyContext, Tick
from akquant.strategy import Strategy


class MyStrategy(Strategy):
    """Test strategy."""

    def on_bar(self, bar: Bar) -> None:
        """Handle bar event."""
        self.log(f"Bar {self.symbol} Close: {self.close}")

    def on_tick(self, tick: Tick) -> None:
        """Handle tick event."""
        self.log(f"Tick {self.symbol} Price: {self.close}")


def test_strategy_logging(caplog: Any) -> None:
    """Test logging."""
    strategy = MyStrategy()

    # Mock context
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0

    # Mock Bar
    ts = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        symbol="AAPL",
    )

    with caplog.at_level(logging.INFO, logger="akquant"):
        strategy._on_bar_event(bar, ctx)

    assert "Bar AAPL Close: 102.0" in caplog.text
    assert "[2023-01-01 09:30:00]" in caplog.text


def test_strategy_properties() -> None:
    """Test properties."""
    strategy = MyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0

    # Test Bar Properties
    ts = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        symbol="AAPL",
    )
    strategy._on_bar_event(bar, ctx)

    assert strategy.symbol == "AAPL"
    assert strategy.close == 102.0
    assert strategy.open == 100.0
    assert strategy.high == 105.0
    assert strategy.low == 95.0
    assert strategy.volume == 1000.0

    # Test Tick Properties
    ts_tick = pd.Timestamp("2023-01-01 09:30:01", tz="Asia/Shanghai").value
    tick = Tick(timestamp=ts_tick, price=103.0, volume=500.0, symbol="GOOG")
    strategy._on_tick_event(tick, ctx)

    assert strategy.symbol == "GOOG"
    assert strategy.close == 103.0  # close maps to price in tick
    assert strategy.volume == 500.0
    # Open/High/Low should be 0.0 or handle gracefully?
    # Current implementation returns 0.0 if not current_bar
    assert strategy.open == 0.0
    assert strategy.high == 0.0
    assert strategy.low == 0.0
