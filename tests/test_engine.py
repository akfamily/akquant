import time
from datetime import datetime, timezone

import akquant
import numpy as np
import pandas as pd
import pytest


def test_engine_initialization() -> None:
    """Test Engine initialization defaults."""
    engine = akquant.Engine()
    assert engine.portfolio.cash == 100000.0
    assert len(engine.trades) == 0
    assert len(engine.orders) == 0


class DummyStrategy(akquant.Strategy):
    """A dummy strategy for testing purposes."""

    pass


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

    def on_bar(self, bar: akquant.Bar) -> None:
        """Handle bar events without generating orders."""
        return


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


def _build_regression_bars(symbol: str) -> list[akquant.Bar]:
    day1 = _ns(datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc))
    day2 = _ns(datetime(2023, 1, 3, 15, 0, tzinfo=timezone.utc))
    day3 = _ns(datetime(2023, 1, 4, 15, 0, tzinfo=timezone.utc))
    return [
        akquant.Bar(day1, 10.0, 10.0, 10.0, 10.0, 1000.0, symbol),
        akquant.Bar(day2, 12.0, 12.0, 12.0, 12.0, 1000.0, symbol),
        akquant.Bar(day3, 11.0, 11.0, 11.0, 11.0, 1000.0, symbol),
    ]


def _build_benchmark_data(n: int, symbol: str) -> pd.DataFrame:
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


def test_backtest_regression_baseline() -> None:
    """Verify baseline equity curve and trade sequence."""
    symbol = "REGRESS"
    engine = akquant.Engine()
    engine.use_simple_market(0.0)
    engine.set_force_session_continuous(True)
    engine.set_execution_mode(akquant.ExecutionMode.CurrentClose)
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


def test_backtest_performance_baseline() -> None:
    """Verify minimum throughput for a no-op strategy."""
    data = _build_benchmark_data(n=3000, symbol="PERF")
    t0 = time.perf_counter()
    result = akquant.run_backtest(
        data=data,
        strategy=NoopStrategy,
        symbol="PERF",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        execution_mode="current_close",
        lot_size=1,
        show_progress=False,
    )
    duration = time.perf_counter() - t0
    throughput = len(data) / duration if duration > 0 else 0.0
    assert throughput >= 200.0
    assert result.metrics.initial_market_value == pytest.approx(100000.0, rel=1e-9)
