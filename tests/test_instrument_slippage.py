"""InstrumentConfig.slippage 生效, 优先级: 订单 > 品种 > 策略 > 全局."""

from typing import Any, Optional

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy


def _bars() -> pd.DataFrame:
    ts = pd.date_range("2024-01-02 10:00", periods=3, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 100000,
            "symbol": "A",
        }
    )


class _BuyOnce(Strategy):
    order_slippage: Optional[dict[str, Any]]

    def on_bar(self, bar: Bar) -> None:
        if self.get_position("A") == 0 and not self.get_open_orders("A"):
            if self.order_slippage is None:
                self.buy(symbol="A", quantity=100)
            else:
                self.buy(symbol="A", quantity=100, slippage=self.order_slippage)


def _fill_price(
    instrument_slippage: Any = None,
    strategy_slippage: Any = None,
    order_slippage: Optional[dict[str, Any]] = None,
) -> float:
    strat = _BuyOnce()
    strat.order_slippage = order_slippage
    kwargs: dict[str, Any] = {}
    if strategy_slippage is not None:
        kwargs["strategy_slippage"] = {"_default": strategy_slippage}
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0, slippage=0.0),
        instruments_config=[
            akquant.InstrumentConfig(symbol="A", slippage=instrument_slippage)
        ],
    )
    result = akquant.run_backtest(
        data=_bars(), strategy=strat, config=config, show_progress=False, **kwargs
    )
    filled = result.orders_df[result.orders_df["filled_quantity"] > 0]
    return float(filled["avg_price"].iloc[0])


def test_instrument_slippage_applies() -> None:
    """全局滑点为 0 时, 品种级 InstrumentConfig.slippage 生效于成交价."""
    assert _fill_price({"type": "fixed", "value": 0.5}) == pytest.approx(100.5)


def test_instrument_beats_strategy_slippage() -> None:
    """品种级滑点优先于策略级 strategy_slippage."""
    price = _fill_price(
        {"type": "fixed", "value": 0.5},
        strategy_slippage={"type": "fixed", "value": 0.1},
    )
    assert price == pytest.approx(100.5)


def test_order_slippage_beats_instrument() -> None:
    """订单级显式 slippage 优先于品种级滑点."""
    price = _fill_price(
        {"type": "fixed", "value": 0.5}, order_slippage={"type": "fixed", "value": 0.2}
    )
    assert price == pytest.approx(100.2)


def test_ticks_slippage_uses_instrument_tick() -> None:
    """Ticks 类型品种滑点按该标的 tick_size 换算(股票 tick 0.01, 3 跳 = 0.03)."""
    assert _fill_price({"type": "ticks", "value": 3}) == pytest.approx(100.03)
