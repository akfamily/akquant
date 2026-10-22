"""交易时段底座（阶段 1）的 Python 侧验收."""

from __future__ import annotations

import datetime as dt

import akquant as aq
import pandas as pd
import pytest


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


def _ns(text: str) -> int:
    """上海本地时间文本转 UTC 纳秒."""
    return int(pd.Timestamp(text, tz="Asia/Shanghai").value)


def test_resolver_stock_auction_and_continuous() -> None:
    """股票集合竞价与连续竞价的时段解析."""
    r = aq.SessionResolver()
    day, session = r.resolve_instant(
        "600000.SH", _ns("2024-01-02 09:20"), aq.AssetType.Stock
    )
    assert day == dt.date(2024, 1, 2)
    assert session == aq.TradingSession.OpeningAuction
    _, session = r.resolve_bar(
        "600000.SH", _ns("2024-01-02 09:31"), asset_type=aq.AssetType.Stock
    )
    assert session == aq.TradingSession.Continuous


def test_resolver_daily_bar_and_night_session() -> None:
    """日线区间解析为连续竞价, 夜盘归属下一交易日."""
    r = aq.SessionResolver(sessions={"rb": "CN_FUTURES_NIGHT_23"})
    start, end = r.day_bounds("600000.SH", dt.date(2024, 1, 2), aq.AssetType.Stock)
    _, session = r.resolve_bar(
        "600000.SH", end, interval_start=start, asset_type=aq.AssetType.Stock
    )
    assert session == aq.TradingSession.Continuous
    day, session = r.resolve_instant(
        "rb2405", _ns("2024-01-05 22:00"), aq.AssetType.Futures
    )
    assert day == dt.date(2024, 1, 8)
    assert session == aq.TradingSession.Continuous
    assert r.uses_fallback_calendar is True


def test_resolver_explicit_calendar_and_errors() -> None:
    """显式日历生效, 越界与非法参数抛 ValueError."""
    days = [dt.date(2024, 2, 7), dt.date(2024, 2, 8), dt.date(2024, 2, 19)]
    r = aq.SessionResolver(
        sessions={"rb": "CN_FUTURES_NIGHT_23"}, trading_calendar=days
    )
    day, _ = r.resolve_instant("rb2405", _ns("2024-02-08 21:30"), aq.AssetType.Futures)
    assert day == dt.date(2024, 2, 19)
    assert r.uses_fallback_calendar is False
    with pytest.raises(ValueError, match="超出 trading_calendar"):
        r.resolve_instant("600000.SH", _ns("2024-03-01 10:00"), aq.AssetType.Stock)
    with pytest.raises(ValueError, match="未知时段模板"):
        aq.SessionResolver(sessions={"default": "NOPE"})
    with pytest.raises(ValueError, match="时区"):
        aq.SessionResolver(timezone="Mars/Olympus")


def test_resolver_templates_listed() -> None:
    """内置模板名可枚举."""
    names = aq.SessionResolver.templates()
    assert "CN_STOCK" in names and "CN_FUTURES_NIGHT_0230" in names


def test_resolve_bar_extreme_and_invalid_interval() -> None:
    """极端时间戳不得 panic, 非法区间抛 ValueError."""
    r = aq.SessionResolver()
    try:
        r.resolve_bar("BTC", -(2**63))
    except ValueError:
        pass
    except BaseException as exc:  # noqa: BLE001
        pytest.fail(f"不应抛出 {type(exc).__name__}: {exc}")
    with pytest.raises(ValueError, match="起点"):
        r.resolve_bar("BTC", timestamp=5, interval_start=10)
