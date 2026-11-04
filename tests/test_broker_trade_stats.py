from akquant.akquant import LiveTradeStats


def test_live_trade_stats_smoke() -> None:
    """LiveTradeStats 冒烟：一笔亏损平仓与回撤."""
    s = LiveTradeStats(100_000.0)
    s.on_equity(1, 100_000.0)
    s.on_fill("AAA", "buy", 100.0, 10.0, 1)
    s.on_fill("AAA", "sell", 100.0, 9.0, 2)
    snap = s.snapshot(3, 99_900.0, {})
    assert snap["losing_trades"] == 1.0 and snap["win_rate"] == 0.0
    assert snap["max_drawdown_value"] == 100.0
