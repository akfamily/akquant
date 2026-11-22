"""商品期货期权卖方保证金(CHINA_COMMODITY): 按交易所公式计算, 取不到输入时拒单写明原因.

期望值取自 SimNow 实测(交易日 20261120, 苯乙烯 eb2701, 交易单位 5, 标的期货保证金率 19%):
CTP 以期权、标的昨结算价计算的期权交易成本, 与实际卖开 1 手后的持仓占用一致。
"""

from typing import Any, Dict, List, Literal, Tuple

import akquant
import pandas as pd
from akquant import BarBatch, Strategy

OptionKind = Literal["CALL", "PUT"]

DAYS = pd.date_range("2026-11-17 15:00", periods=3, freq="B", tz="Asia/Shanghai")
UNDERLYING = "eb2701"
UNDERLYING_PRICE = 8468.0
# (期权代码, 类型, 行权价, 期权价, SimNow 1 手保证金)
OPTIONS: List[Tuple[str, OptionKind, float, float, float]] = [
    ("eb2701-C-8000", "CALL", 8000.0, 607.5, 11082.10),  # 实值
    ("eb2701-C-8500", "CALL", 8500.0, 313.0, 9529.60),  # 平值(虚值 32 点)
    ("eb2701-P-6000", "PUT", 6000.0, 2.5, 4034.80),  # 深度虚值, 取标的保证金一半
]


def _bars(symbol: str, price: float) -> pd.DataFrame:
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


def _option_config(
    symbol: str, kind: OptionKind, strike: float, underlying: str
) -> akquant.InstrumentConfig:
    return akquant.InstrumentConfig(
        symbol=symbol,
        asset_type="OPTION",
        multiplier=5.0,
        option_type=kind,
        strike_price=strike,
        expiry_date=20261216,
        underlying_symbol=underlying,
        option_margin_model="CHINA_COMMODITY",
    )


class _SellOne(Strategy):
    """首日卖开 ``target`` 1 手, 次日记录账户保证金."""

    target = ""

    def __init__(self) -> None:
        super().__init__()
        self.margins: List[float] = []

    def on_bars(self, batch: BarBatch) -> None:
        if not self.margins and self.get_position(self.target) == 0:
            self.sell(self.target, 1)
        self.margins.append(float(self.get_account()["margin"]))


def _run(target: str, data: Dict[str, pd.DataFrame], configs: List[Any]) -> Any:
    strategy = type("_Sell", (_SellOne,), {"target": target})()
    result = akquant.run_backtest(
        data=data,
        strategy=strategy,
        config=akquant.BacktestConfig(
            strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
            instruments_config=configs,
        ),
        freq="1d",
        show_progress=False,
    )
    return strategy, result


def test_short_option_margin_matches_simnow() -> None:
    """实值、平值、深度虚值认购/认沽各卖 1 手, 保证金与 SimNow 实测一致."""
    futures = akquant.InstrumentConfig(
        symbol=UNDERLYING, asset_type="FUTURES", multiplier=5.0, margin_ratio=0.19
    )
    for symbol, kind, strike, price, expected in OPTIONS:
        data = {
            UNDERLYING: _bars(UNDERLYING, UNDERLYING_PRICE),
            symbol: _bars(symbol, price),
        }
        configs = [futures, _option_config(symbol, kind, strike, UNDERLYING)]
        strategy, _ = _run(symbol, data, configs)
        assert strategy.margins[0] == 0.0
        assert round(strategy.margins[-1], 2) == expected, symbol


def test_short_option_rejected_when_underlying_futures_not_registered() -> None:
    """标的期货未登记时卖开被拒, 原因写明取不到标的期货保证金率."""
    symbol, kind, strike, price, _ = OPTIONS[1]
    data = {symbol: _bars(symbol, price)}
    configs = [_option_config(symbol, kind, strike, UNDERLYING)]
    strategy, result = _run(symbol, data, configs)
    orders = result.orders_df
    assert orders["status"].tolist() == ["rejected"]
    reason = str(orders["reject_reason"].iloc[0])
    assert (
        f"{symbol} 无法计算商品期权保证金: 标的期货 {UNDERLYING} 未登记为合约" in reason
    )
    assert strategy.margins[-1] == 0.0
