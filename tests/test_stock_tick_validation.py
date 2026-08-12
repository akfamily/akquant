"""股票 tick 校验的端到端行为(经 run_backtest)."""

from typing import Any

import pandas as pd
from akquant import Strategy, run_backtest
from akquant.config import (
    BacktestConfig,
    ChinaStockConfig,
    InstrumentConfig,
    StrategyConfig,
)


class _BuyAt2_8314(Strategy):
    """固定报一个非 tick 倍数的价格."""

    def on_bar(self, bar: Any) -> None:
        _ = bar
        if not getattr(self, "_done", False):
            self._done = True
            self.buy("600008.SH", 100, price=2.8314)


class _BuyAt2_83(Strategy):
    """固定报一个合法价格."""

    def on_bar(self, bar: Any) -> None:
        _ = bar
        if not getattr(self, "_done", False):
            self._done = True
            self.buy("600008.SH", 100, price=2.83)


def _feed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
            "open": 2.83,
            "high": 2.85,
            "low": 2.81,
            "close": 2.83,
            "volume": 100000.0,
            "symbol": "600008.SH",
        }
    )


def _config(enforce: bool = True) -> BacktestConfig:
    return BacktestConfig(
        strategy_config=StrategyConfig(initial_cash=100000.0, commission_rate=0.0),
        instruments_config=[
            InstrumentConfig(symbol="600008.SH", asset_type="STOCK", tick_size=0.01)
        ],
        china_stock=ChinaStockConfig(enforce_tick_size=enforce),
    )


def test_misaligned_limit_price_is_rejected() -> None:
    """2.8314 不是 0.01 的倍数, 撮合前校验应拒掉, 不产生成交."""
    result = run_backtest(
        data=_feed(),
        strategy=_BuyAt2_8314,
        show_progress=False,
        config=_config(),
    )
    assert result.executions_df.empty


def test_aligned_limit_price_fills() -> None:
    """2.83 合法, 正常成交(证明校验没有误伤)."""
    result = run_backtest(
        data=_feed(),
        strategy=_BuyAt2_83,
        show_progress=False,
        config=_config(),
    )
    assert not result.executions_df.empty


def test_validation_can_be_disabled() -> None:
    """enforce_tick_size=False 时放行(逃生舱: tick 元数据不准时不至于卡死)."""
    result = run_backtest(
        data=_feed(),
        strategy=_BuyAt2_8314,
        show_progress=False,
        config=_config(enforce=False),
    )
    assert not result.executions_df.empty
