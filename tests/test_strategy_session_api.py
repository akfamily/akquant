"""策略的交易日与时段接口：trading_day / session_of / is_market_open."""

from __future__ import annotations

import datetime as dt
from typing import Any, List, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest

TZ = "Asia/Shanghai"


def _minute(times: List[str], symbol: str) -> pd.DataFrame:
    idx = pd.to_datetime(times).tz_localize(TZ)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )


class _Probe(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[Tuple[str, Any, str, bool, bool]] = []

    def on_bar(self, bar: Bar) -> None:
        t = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ).strftime("%m-%d %H:%M")
        self.rows.append(
            (
                t,
                self.trading_day,
                str(self.session_of()).split(".")[-1],
                self.is_market_open(),
                self.is_market_open(include_auction=True),
            )
        )


def test_trading_day_and_session_api_on_stock_minutes() -> None:
    """股票分钟线：交易日、当前时段与是否开市（收盘集合竞价只在含竞价时算开市）."""
    df = _minute(
        ["2024-01-02 09:31", "2024-01-02 14:58", "2024-01-02 15:00"], "600000.SH"
    )
    r = run_backtest(
        strategy=_Probe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    assert r.strategy.rows == [
        ("01-02 09:31", dt.date(2024, 1, 2), "Continuous", True, True),
        ("01-02 14:58", dt.date(2024, 1, 2), "ClosingAuction", False, True),
        ("01-02 15:00", dt.date(2024, 1, 2), "ClosingAuction", False, True),
    ]


def test_night_session_trading_day_is_next_day() -> None:
    """夜盘 bar 的 trading_day 是下一交易日."""
    df = _minute(["2024-01-05 21:30"], "au2406")
    cfg = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=1e6),
        instruments_config=[
            aq.InstrumentConfig(
                symbol="au2406",
                asset_type="FUTURES",
                multiplier=1000.0,
                margin_ratio=0.1,
            )
        ],
    )
    r = run_backtest(
        strategy=_Probe,
        data=df,
        config=cfg,
        show_progress=False,
        sessions={"au": "CN_FUTURES_NIGHT_0230"},
    )
    assert r.strategy is not None
    assert r.strategy.rows[0][1] == dt.date(2024, 1, 8)  # 周五夜盘属下周一


def test_session_of_other_symbol_uses_resolver_at_now() -> None:
    """股票 10:20 bar 上查询期货（日盘 10:15～10:30 休息）的时段."""

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.got: List[str] = []

        def on_bar(self, bar: Bar) -> None:
            self.got.append(str(self.session_of("rb2405")).split(".")[-1])

    df = pd.concat(
        [
            _minute(["2024-01-02 10:20"], "600000.SH"),
            _minute(["2024-01-02 10:31"], "rb2405"),
        ]
    )
    cfg = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=1e6),
        instruments_config=[
            aq.InstrumentConfig(symbol="600000.SH", asset_type="STOCK"),
            aq.InstrumentConfig(
                symbol="rb2405",
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
            ),
        ],
    )
    r = run_backtest(strategy=P, data=df, config=cfg, show_progress=False)
    assert r.strategy is not None
    assert r.strategy.got[0] == "Break"


def test_user_timer_rejects_reserved_prefix() -> None:
    """用户定时器 payload 不能以引擎保留前缀 __akq_ 开头."""

    class P(Strategy):
        def on_start(self) -> None:
            self.schedule_daily("09:35", "__akq_lifecycle__|x")

    with pytest.raises(ValueError, match="__akq_"):
        run_backtest(
            strategy=P,
            data=_minute(["2024-01-02 09:31"], "600000.SH"),
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
        )


def test_session_of_own_symbol_matches_event_session_on_daily_bars() -> None:
    """日线 bar 覆盖整个交易日：session_of(本品种) 取本时间片的区间时段.

    与 ctx.session 一致，而不是按 15:00 前 1ns 求得的收盘集合竞价.
    """

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.got: List[Tuple[str, str]] = []

        def on_bar(self, bar: Bar) -> None:
            self.got.append(
                (
                    str(self.session_of()).split(".")[-1],
                    str(self.session_of(bar.symbol)).split(".")[-1],
                )
            )

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": "600000.SH",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )
    r = run_backtest(
        strategy=P,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    assert r.strategy.got == [("Continuous", "Continuous")] * 2
