"""数据在到期日结束时, 收尾补结算也要派发 on_expiry 与流式 expiry 事件."""

from typing import Any

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy
from akquant.backtest import BacktestStreamEvent


def _daily(symbol: str, days: list[str], closes: list[float]) -> pd.DataFrame:
    ts = pd.to_datetime([f"{d} 15:00" for d in days])
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1000,
            "symbol": symbol,
        }
    )


class _Recorder(Strategy):
    """首日买 1 张期权, 记录 on_expiry."""

    expiries: list[dict[str, Any]]

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol == "OPT" and str(bar.timestamp_iso)[:10] == "2023-11-29":
            self.buy(symbol="OPT", quantity=1)

    def on_expiry(self, event: dict[str, Any]) -> None:
        self.expiries.append(dict(event))


def _run(days: list[str]) -> tuple[_Recorder, list[BacktestStreamEvent]]:
    strat = _Recorder()
    strat.expiries = []
    stream: list[BacktestStreamEvent] = []
    config = akquant.BacktestConfig(
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
    ul = [100.0, 101.0, 120.0, 120.0][: len(days)]
    akquant.run_backtest(
        data={
            "OPT": _daily("OPT", days, [1.0] * len(days)),
            "UL": _daily("UL", days, ul),
        },
        strategy=strat,
        config=config,
        show_progress=False,
        on_event=stream.append,
    )
    expiry_events = [e for e in stream if e.get("event_type") == "expiry"]
    return strat, expiry_events


def test_terminal_settlement_fires_on_expiry_and_stream_event() -> None:
    """数据在到期日 12/01 结束: 收尾补结算仍派发 1 次回调与 1 条流事件."""
    strat, stream = _run(["2023-11-29", "2023-11-30", "2023-12-01"])
    assert len(strat.expiries) == 1
    assert len(stream) == 1
    assert strat.expiries[0]["cash_flow"] == pytest.approx(2000.0)
    assert strat.expiries[0]["fee"] == pytest.approx(0.6)


def test_terminal_and_daily_paths_agree() -> None:
    """多一天数据(日常路径)与收尾路径给出相同的 cash_flow / fee."""
    short, _ = _run(["2023-11-29", "2023-11-30", "2023-12-01"])
    long, _ = _run(["2023-11-29", "2023-11-30", "2023-12-01", "2023-12-04"])
    for key in ("cash_flow", "fee", "quantity_closed"):
        assert short.expiries[0][key] == pytest.approx(long.expiries[0][key])
