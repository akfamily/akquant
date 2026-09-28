"""InstrumentConfig 的按品种费用字段必须真正生效."""

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy


def _bars(symbol: str, n: int = 4, price: float = 100.0) -> pd.DataFrame:
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


class _BuyEach(Strategy):
    """每个标的在它的第一根 bar 上各买一次.

    不用 ``self._bar_count``: 它是跨标的的全局计数, 多标的时只有第一个标的会下单。
    """

    def __init__(self) -> None:
        super().__init__()
        self._bought: set[str] = set()

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol not in self._bought:
            self._bought.add(bar.symbol)
            self.buy(symbol=bar.symbol, quantity=100)


def _fees(
    instruments: list[akquant.InstrumentConfig], t_plus_one: bool
) -> dict[str, float]:
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(
            initial_cash=10_000_000.0,
            commission_rate=0.0003,
            min_commission=5.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
        ),
        instruments_config=instruments,
    )
    result = akquant.run_backtest(
        data={ic.symbol: _bars(ic.symbol) for ic in instruments},
        strategy=_BuyEach,
        config=config,
        t_plus_one=t_plus_one,
        show_progress=False,
    )
    orders = result.orders_df
    filled = orders[orders["filled_quantity"] > 0]
    return {str(s): float(c) for s, c in zip(filled["symbol"], filled["commission"])}


@pytest.mark.parametrize("t_plus_one", [False, True])
def test_per_instrument_commission_overrides_global(t_plus_one: bool) -> None:
    """A 设了 commission_rate/min_commission, B 没设: 只有 A 的费用变化.

    t_plus_one=False 走 SimpleMarket, True 走 ChinaMarket, 两条路都要生效。
    """
    fees = _fees(
        [
            akquant.InstrumentConfig(
                symbol="A", commission_rate=0.001, min_commission=0.0
            ),
            akquant.InstrumentConfig(symbol="B"),
        ],
        t_plus_one,
    )
    # 成交额 100 × 100 = 10000
    assert fees["A"] == pytest.approx(10.0)  # 10000 × 0.001, 无最低佣金
    assert fees["B"] == pytest.approx(5.0)  # 10000 × 0.0003 = 3 < 最低 5


def test_fund_min_commission_override() -> None:
    """可转债/ETF(FUND)可以单独去掉最低佣金."""
    fees = _fees(
        [
            akquant.InstrumentConfig(
                symbol="CB",
                asset_type="FUND",
                commission_rate=0.0001,
                min_commission=0.0,
                lot_size=10,
            ),
        ],
        t_plus_one=True,
    )
    # 佣金 10000 × 0.0001 = 1 (最低佣金已覆盖为 0) + FUND 缺省过户费 10000 × 0.00001
    assert fees["CB"] == pytest.approx(1.0 + 10000 * 0.00001)
