"""交易所期权组合保证金(CHINA_COMMODITY_COMBINATION): 回测账户占用按组合策略计算.

数据取自 SimNow 苯乙烯 eb2701(交易单位 5, 标的期货保证金率 19%, 标的 8468): 单腿卖方
保证金 C8000(价 607.5)11082.10、C8500(价 313)9529.60, 已与 SimNow 核对; P8000(价 100)
按同一公式为 7374.60。组合公式按交易所规则, 未经实盘核对(SimNow 不计组合保证金)。
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

import akquant
import pandas as pd
import pytest
from akquant import BarBatch, Strategy

DAYS = pd.date_range("2026-11-17 15:00", periods=3, freq="B", tz="Asia/Shanghai")
UNDERLYING = "eb2701"
PRICES = {
    UNDERLYING: 8468.0,
    "eb2701-C-8000": 607.5,
    "eb2701-C-8500": 313.0,
    "eb2701-P-8000": 100.0,
}
KINDS: Dict[str, Tuple[Literal["CALL", "PUT"], float]] = {
    "eb2701-C-8000": ("CALL", 8000.0),
    "eb2701-C-8500": ("CALL", 8500.0),
    "eb2701-P-8000": ("PUT", 8000.0),
}


def _bars(symbol: str) -> pd.DataFrame:
    price = PRICES[symbol]
    return pd.DataFrame(
        {
            "date": DAYS,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1000,
            "symbol": symbol,
        }
    )


class _Open(Strategy):
    """首日按 ``holdings`` 开仓, 记录每日账户保证金."""

    holdings: Dict[str, int] = {}

    def __init__(self) -> None:
        super().__init__()
        self.margins: List[float] = []
        self.models: Dict[str, Optional[str]] = {}

    def on_bars(self, batch: BarBatch) -> None:
        if not self.margins:
            for symbol, qty in self.holdings.items():
                if qty > 0:
                    self.buy(symbol, qty)
                else:
                    self.sell(symbol, -qty)
            self.models = {s: self.get_instrument(s).option_margin_model for s in KINDS}
        self.margins.append(float(self.get_account()["margin"]))


def _run(
    holdings: Dict[str, int],
    combination: bool = True,
    model: Optional[str] = None,
) -> _Open:
    configs: List[Any] = [
        akquant.InstrumentConfig(
            symbol=UNDERLYING, asset_type="FUTURES", multiplier=5.0, margin_ratio=0.19
        )
    ]
    for symbol, (kind, strike) in KINDS.items():
        configs.append(
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="OPTION",
                multiplier=5.0,
                option_type=kind,
                strike_price=strike,
                expiry_date=20261218,
                underlying_symbol=UNDERLYING,
                option_margin_model=model,  # type: ignore[arg-type]
            )
        )
    strategy: _Open = type("_O", (_Open,), {"holdings": holdings})()
    akquant.run_backtest(
        data={s: _bars(s) for s in PRICES},
        strategy=strategy,
        config=akquant.BacktestConfig(
            strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
            instruments_config=configs,
            china_options=akquant.ChinaOptionsConfig(
                commission_per_contract=0.0,
                exchange_fee_per_contract=0.0,
                clearing_fee_per_contract=0.0,
                combination_margin=combination,
            ),
        ),
        freq="1d",
        show_progress=False,
    )
    return strategy


STRADDLE = {"eb2701-C-8000": -1, "eb2701-P-8000": -1}


def test_combination_margin_by_strategy() -> None:
    """跨式、卖出垂直价差、卖出期权期货组合按组合公式计占用."""
    cases = [
        # max(11082.10, 7374.60) + 认沽权利金 100 × 5
        (STRADDLE, 11582.10),
        # min(执行价差 500 × 5, 11082.10)
        ({"eb2701-C-8000": -1, "eb2701-C-8500": 1}, 2500.00),
        # 期货 8468 × 5 × 19% + 认购权利金 607.5 × 5
        ({"eb2701-C-8000": -1, UNDERLYING: 1}, 11082.10),
    ]
    for holdings, expected in cases:
        strategy = _run(holdings)
        assert set(strategy.models.values()) == {"CHINA_COMMODITY_COMBINATION"}
        assert round(strategy.margins[-1], 2) == expected, holdings


def test_combination_off_by_default() -> None:
    """未开启时缺省仍为 CHINA_COMMODITY, 按单腿之和计占用."""
    strategy = _run(STRADDLE, combination=False)
    assert set(strategy.models.values()) == {"CHINA_COMMODITY"}
    assert round(strategy.margins[-1], 2) == round(11082.10 + 7374.60, 2)


def test_explicit_model_requires_supported_exchange() -> None:
    """上期所品种与对照表外的品种不能用 CHINA_COMMODITY_COMBINATION."""
    for underlying, message in (
        ("cu2601", "属于上期所/上期能源, 期权没有组合保证金优惠"),
        ("xx2601", "不在组合保证金交易所对照表中"),
    ):
        with pytest.raises(ValueError, match=message):
            akquant.InstrumentConfig(
                symbol=f"{underlying}-C-1",
                asset_type="OPTION",
                option_type="CALL",
                strike_price=1.0,
                expiry_date=20261218,
                underlying_symbol=underlying,
                option_margin_model="CHINA_COMMODITY_COMBINATION",
            )
