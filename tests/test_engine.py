import akquant


def test_engine_initialization() -> None:
    """Test Engine initialization defaults."""
    engine = akquant.Engine()
    assert engine.portfolio.cash == 100000.0
    assert len(engine.trades) == 0
    assert len(engine.orders) == 0


class DummyStrategy(akquant.Strategy):
    """A dummy strategy for testing purposes."""

    pass


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
