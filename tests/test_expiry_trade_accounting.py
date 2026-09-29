"""到期平仓进入交易统计, 保证金账户期货到期按平仓盈亏入账."""

from typing import Any, Literal, Optional

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy
from akquant.config import RiskConfig


def _daily(
    symbol: str,
    days: list[str],
    closes: list[float],
    opens: Optional[list[float]] = None,
) -> pd.DataFrame:
    ts = pd.to_datetime([f"{d} 15:00" for d in days])
    o = opens if opens is not None else closes
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": o,
            "high": [max(a, b) for a, b in zip(o, closes)],
            "low": [min(a, b) for a, b in zip(o, closes)],
            "close": closes,
            "volume": 1000,
            "symbol": symbol,
        }
    )


class _OpenOnFirstDay(Strategy):
    """首个交易日对 target 下一张单, 方向由 side 决定."""

    target: str
    side: str
    first_day: str

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol == self.target and str(bar.timestamp_iso)[:10] == self.first_day:
            if self.get_position(self.target) == 0:
                if self.side == "buy":
                    self.buy(symbol=self.target, quantity=1)
                else:
                    self.sell(symbol=self.target, quantity=1)


def _strategy(target: str, side: str, first_day: str) -> _OpenOnFirstDay:
    s = _OpenOnFirstDay()
    s.target, s.side, s.first_day = target, side, first_day
    return s


OPT_DAYS = ["2023-11-29", "2023-11-30", "2023-12-01", "2023-12-04"]


def _run_option(
    side: str, option_type: Literal["CALL", "PUT"], ul_closes: list[float]
) -> Any:
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=100_000.0),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol="OPT",
                asset_type="OPTION",
                multiplier=100.0,
                option_type=option_type,
                strike_price=100.0,
                expiry_date=20231201,
                underlying_symbol="UL",
            ),
            akquant.InstrumentConfig(symbol="UL", asset_type="STOCK"),
        ],
    )
    return akquant.run_backtest(
        data={
            "OPT": _daily("OPT", OPT_DAYS, [1.0] * 4),
            "UL": _daily("UL", OPT_DAYS, ul_closes),
        },
        strategy=_strategy("OPT", side, OPT_DAYS[0]),
        config=config,
        show_progress=False,
    )


def test_itm_long_option_expiry_is_a_closed_trade() -> None:
    """实值多头: 1 笔已平仓交易, 盈亏 (20-1)*100, 行权费进 total_commission."""
    result = _run_option("buy", "CALL", [100.0, 101.0, 120.0, 120.0])
    trades = result.trades_df
    assert len(trades) == 1
    assert float(trades["pnl"].iloc[0]) == pytest.approx((20.0 - 1.0) * 100.0)
    assert float(result.trade_metrics.total_commission) == pytest.approx(6.6 + 0.6)


def test_otm_long_option_expiry_loses_premium() -> None:
    """虚值多头: 1 笔已平仓, 盈亏 = -权利金, 不收行权费."""
    result = _run_option("buy", "CALL", [100.0, 99.0, 90.0, 90.0])
    trades = result.trades_df
    assert len(trades) == 1
    assert float(trades["pnl"].iloc[0]) == pytest.approx(-1.0 * 100.0)
    assert float(result.trade_metrics.total_commission) == pytest.approx(6.6)


def test_assigned_short_option_expiry_is_a_losing_trade() -> None:
    """被指派的空头: 1 笔已平仓, 盈亏 = 权利金 - 内在价值 = (1-20)*100."""
    result = _run_option("sell", "CALL", [100.0, 101.0, 120.0, 120.0])
    trades = result.trades_df
    assert len(trades) == 1
    assert float(trades["pnl"].iloc[0]) == pytest.approx((1.0 - 20.0) * 100.0)


FUT_DAYS = ["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03"]


def _run_future(
    account_mode: str, closes: list[float], opens: Optional[list[float]] = None
) -> Any:
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(
            initial_cash=100_000.0,
            commission_rate=0.0,
            risk=RiskConfig(account_mode=account_mode),
        ),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol="F",
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
                expiry_date=20260130,
            )
        ],
    )
    return akquant.run_backtest(
        data={"F": _daily("F", FUT_DAYS, closes, opens)},
        strategy=_strategy("F", "buy", FUT_DAYS[0]),
        config=config,
        show_progress=False,
    )


@pytest.mark.parametrize("account_mode", ["cash", "margin"])
def test_futures_expiry_at_entry_price_is_flat(account_mode: str) -> None:
    """110 成交(次根 bar 开盘), 按 110 到期: 两种账户权益都不变, 且有 1 笔已平仓."""
    result = _run_future(account_mode, [100.0, 110.0, 110.0, 110.0])
    assert float(result.orders_df["avg_price"].iloc[0]) == pytest.approx(110.0)
    assert result.metrics.end_market_value == pytest.approx(100_000.0)
    assert len(result.trades_df) == 1


@pytest.mark.parametrize("account_mode", ["cash", "margin"])
def test_futures_expiry_profit(account_mode: str) -> None:
    """到期日 1/30 开盘 110 成交、收盘 120 结算: 两种账户都赚 (120-110)*10."""
    result = _run_future(
        account_mode,
        closes=[100.0, 120.0, 120.0, 120.0],
        opens=[100.0, 110.0, 120.0, 120.0],
    )
    assert float(result.orders_df["avg_price"].iloc[0]) == pytest.approx(110.0)
    assert result.metrics.end_market_value == pytest.approx(100_000.0 + 100.0)
