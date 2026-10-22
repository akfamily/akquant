"""交易时段底座（阶段 1）的 Python 侧验收."""

from __future__ import annotations

import akquant as aq


def test_trading_session_has_new_variants() -> None:
    """新时段类型可用；旧值本阶段仍保留（阶段 6 删除）."""
    names = {"OpeningAuction", "AuctionHold", "ClosingAuction"}
    for name in names:
        assert hasattr(aq.TradingSession, name), name
    assert aq.TradingSession.OpeningAuction != aq.TradingSession.ClosingAuction
    assert hasattr(aq.TradingSession, "Continuous")


def test_bar_interval_start_roundtrip() -> None:
    """Bar 可携带区间起点, 缺省为 None."""
    bar = aq.Bar(2_000, 1.0, 1.0, 1.0, 1.0, 10.0, "X")
    assert bar.interval_start is None
    with_start = aq.Bar(2_000, 1.0, 1.0, 1.0, 1.0, 10.0, "X", interval_start=1_000)
    assert with_start.interval_start == 1_000
    with_start.interval_start = None
    assert with_start.interval_start is None
