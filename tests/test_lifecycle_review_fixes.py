"""阶段 4 终审修复（F4-1～F4-5）的回归测试."""

from __future__ import annotations

import datetime as dt
from typing import Any, List, Tuple

import akquant
import pandas as pd
from akquant import Bar, Strategy, run_backtest

TZ = "Asia/Shanghai"


def _option_daily(symbol: str, days: List[str], closes: List[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(days),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1000,
            "symbol": symbol,
        }
    )


def _run_option_expiry(strategy_cls: type) -> None:
    cfg = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=100_000.0),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol="OPT",
                asset_type="OPTION",
                multiplier=100.0,
                option_type="CALL",
                strike_price=100.0,
                expiry_date=20231201,
                underlying_symbol="UL",
            ),
            akquant.InstrumentConfig(symbol="UL", asset_type="STOCK"),
        ],
    )
    days = ["2023-11-29", "2023-11-30", "2023-12-01", "2023-12-04", "2023-12-05"]
    run_backtest(
        data={
            "OPT": _option_daily("OPT", days[:3], [1.0] * 3),
            "UL": _option_daily("UL", days, [100.0, 101.0, 120.0, 120.0, 120.0]),
        },
        strategy=strategy_cls,
        config=cfg,
        show_progress=False,
    )


class _ExpiryProbe(Strategy):
    events: List[Tuple[str, str]] = []

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol == "OPT" and str(bar.timestamp_iso)[:10] == "2023-11-29":
            self.buy(symbol="OPT", quantity=1)

    def on_expiry(self, event: Any) -> None:
        self.events.append(("on_expiry", str(event["symbol"])))

    def on_pre_open(self, event: Any) -> None:
        self.events.append(("pre_open", str(event["trading_date"])))

    def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
        self.events.append(("after", str(trading_date)))


def test_on_expiry_delivered_by_after_trading_of_expiry_settlement_day() -> None:
    """期权在 12-01 日终结算到期：on_expiry 不晚于 on_after_trading(12-01) 送达.

    此前生命周期定时器路径不调 check_expiry_events，on_expiry 要等到下一交易日
    首根 bar（晚于 12-04 的盘前与开始交易）才送达。
    """
    _ExpiryProbe.events = []
    _run_option_expiry(_ExpiryProbe)
    log = _ExpiryProbe.events
    assert log.count(("on_expiry", "OPT")) == 1, log
    expiry_idx = log.index(("on_expiry", "OPT"))
    assert expiry_idx < log.index(("after", "2023-12-01")), log
    assert expiry_idx < log.index(("pre_open", "2023-12-04")), log
