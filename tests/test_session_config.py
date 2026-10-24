"""阶段 6：时段配置接口收敛与旧接口删除."""

from __future__ import annotations

import akquant as aq


def test_trading_session_has_only_new_values() -> None:
    """TradingSession 只保留 7 个新值."""
    names = {
        n
        for n in dir(aq.TradingSession)
        if n[0].isupper()
        and isinstance(getattr(aq.TradingSession, n), aq.TradingSession)
    }
    assert names == {
        "Closed",
        "OpeningAuction",
        "AuctionHold",
        "Continuous",
        "Break",
        "ClosingAuction",
        "PostClose",
    }


def test_engine_has_no_set_market_sessions() -> None:
    """旧的全局单表时段接口已删除."""
    assert not hasattr(aq.Engine(), "set_market_sessions")
