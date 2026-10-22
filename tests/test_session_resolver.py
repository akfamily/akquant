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
