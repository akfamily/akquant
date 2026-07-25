from typing import Any

import akquant
import pandas as pd
import pytest
from akquant.config import BacktestConfig, RiskConfig, StrategyConfig


class QuickstartLikeStrategy(akquant.Strategy):
    """Quickstart-equivalent strategy for stream/non-stream consistency checks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize strategy state."""
        super().__init__()
        self.bars_held: dict[str, int] = {}
        self.entry_prices: dict[str, float] = {}

    def on_bar(self, bar: akquant.Bar) -> None:
        """Handle bar data event."""
        symbol = bar.symbol
        pos = self.get_position(symbol)
        if pos > 0:
            if symbol not in self.bars_held:
                self.bars_held[symbol] = 0
            self.bars_held[symbol] += 1
        else:
            if symbol in self.bars_held:
                del self.bars_held[symbol]
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]

        if pos == 0:
            self.order_target_percent(target_percent=0.33, symbol=symbol)
            self.bars_held[symbol] = 0
            self.entry_prices[symbol] = bar.close
        elif pos > 0:
            entry_price = self.entry_prices.get(symbol, bar.close)
            current_bars_held = self.bars_held.get(symbol, 0)
            pnl_pct = (bar.close - entry_price) / entry_price
            if pnl_pct >= 0.10:
                self.sell(symbol, pos)
            elif current_bars_held >= 100:
                self.close_position()


def _build_symbol_df(symbol: str, base: float) -> pd.DataFrame:
    """Build deterministic daily bars for one symbol."""
    dates = pd.date_range("2025-01-01", periods=6, freq="D")
    close = [base, base * 1.02, base * 1.06, base * 1.11, base * 1.03, base * 1.08]
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1000.0] * len(dates),
            "symbol": [symbol] * len(dates),
        }
    )


def _build_quickstart_like_data() -> dict[str, pd.DataFrame]:
    """Build multi-symbol deterministic data mapping."""
    return {
        "600000": _build_symbol_df("600000", 10.0),
        "600004": _build_symbol_df("600004", 12.0),
        "600006": _build_symbol_df("600006", 8.0),
    }


def test_quickstart_like_stream_matches_non_stream() -> None:
    """Quickstart-like strategy should match between stream and non-stream runs."""
    data = _build_quickstart_like_data()
    config = BacktestConfig(
        strategy_config=StrategyConfig(risk=RiskConfig(safety_margin=0.0001))
    )
    common_args: dict[str, Any] = dict(
        strategy=QuickstartLikeStrategy,
        data=data,
        initial_cash=5000000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=5.0,
        lot_size=1,
        fill_policy=akquant.NextAverage(),
        config=config,
        start_time="20250101",
        end_time="20250106",
        symbols=["600000", "600004", "600006"],
        show_progress=False,
    )

    normal = akquant.run_backtest(**common_args)
    events: list[akquant.BacktestStreamEvent] = []
    stream = akquant.run_backtest(
        **common_args,
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=32,
        stream_max_buffer=256,
        stream_error_mode="continue",
    )

    assert len(stream.trades) == len(normal.trades)
    assert len(stream.orders) == len(normal.orders)
    assert len(stream.equity_curve) == len(normal.equity_curve)
    assert stream.metrics.total_return == pytest.approx(
        normal.metrics.total_return, rel=1e-9
    )
    assert stream.metrics.max_drawdown == pytest.approx(
        normal.metrics.max_drawdown, rel=1e-9
    )
    assert stream.metrics.end_market_value == pytest.approx(
        normal.metrics.end_market_value, rel=1e-9
    )
    assert events[0]["event_type"] == "started"
    assert events[-1]["event_type"] == "finished"
    seq_values = [event["seq"] for event in events]
    assert seq_values == sorted(seq_values)
