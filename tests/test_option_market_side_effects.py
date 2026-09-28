"""期权强制 ChinaMarket 不应改变同一回测里期货的手续费口径."""

from typing import Any, Optional

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy

FUT = "RB2401"


def _bars(symbol: str, price: float, n: int = 4) -> pd.DataFrame:
    ts = pd.date_range("2024-01-02 10:00", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 100000,
            "symbol": symbol,
        }
    )


class _BuyFutureOnce(Strategy):
    """只在期货的第一根 bar 上买 1 手."""

    def __init__(self) -> None:
        super().__init__()
        self._done = False

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol == FUT and not self._done:
            self._done = True
            self.buy(symbol=FUT, quantity=1)


def _future_conf(commission_rate: Optional[float] = None) -> akquant.InstrumentConfig:
    return akquant.InstrumentConfig(
        symbol=FUT,
        asset_type="FUTURES",
        multiplier=10.0,
        margin_ratio=0.1,
        tick_size=1.0,
        lot_size=1,
        commission_rate=commission_rate,
    )


def _option_conf() -> akquant.InstrumentConfig:
    return akquant.InstrumentConfig(
        symbol="OPT",
        asset_type="OPTION",
        option_type="CALL",
        strike_price=2.0,
        expiry_date=20991231,
        underlying_symbol="UL",
        multiplier=10000.0,
        tick_size=0.0001,
        lot_size=1,
    )


def _future_fee(
    instruments: list[akquant.InstrumentConfig],
    china_futures: Any = None,
    **strategy_kwargs: Any,
) -> float:
    data = {FUT: _bars(FUT, 4000.0)}
    if any(ic.symbol == "OPT" for ic in instruments):
        data["OPT"] = _bars("OPT", 0.1)
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(
            initial_cash=10_000_000.0, **strategy_kwargs
        ),
        instruments_config=instruments,
        china_futures=china_futures,
    )
    result = akquant.run_backtest(
        data=data,
        strategy=_BuyFutureOnce,
        config=config,
        show_progress=False,
    )
    orders = result.orders_df
    filled = orders[(orders["filled_quantity"] > 0) & (orders["symbol"] == FUT)]
    assert len(filled) == 1
    return float(filled["commission"].iloc[0])


def test_futures_fee_unchanged_when_option_instrument_added() -> None:
    """期权把整场切到 ChinaMarket 后, 期货仍按用户的 commission_rate 计费."""
    without_option = _future_fee([_future_conf()], commission_rate=0.001)
    with_option = _future_fee([_future_conf(), _option_conf()], commission_rate=0.001)
    # 成交额 4000 × 1 × 10 = 40000; 40000 × 0.001 = 40
    assert without_option == pytest.approx(40.0)
    assert with_option == pytest.approx(without_option)


def test_futures_per_instrument_override_beats_global_and_prefix_rule() -> None:
    """按品种覆盖(InstrumentConfig.commission_rate)同时压过全局费率和前缀规则."""
    china_futures = akquant.ChinaFuturesConfig(
        enforce_sessions=False,
        fee_by_symbol_prefix=[
            akquant.ChinaFuturesFeeConfig(symbol_prefix="RB", commission_rate=0.0005)
        ],
    )
    # 基线: 不设按品种覆盖时, 前缀规则生效 (40000 × 0.0005 = 20)
    assert _future_fee(
        [_future_conf()], china_futures=china_futures, commission_rate=0.001
    ) == pytest.approx(20.0)
    # 按品种覆盖: 40000 × 0.0002 = 8
    assert _future_fee(
        [_future_conf(commission_rate=0.0002)],
        china_futures=china_futures,
        commission_rate=0.001,
    ) == pytest.approx(8.0)
