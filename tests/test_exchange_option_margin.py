"""期货交易所期权卖方保证金: 商品期权(CHINA_COMMODITY)与中金所股指期权(CHINA_INDEX).

期望值均取自实测:
- 商品期权: SimNow(交易日 20261120, 苯乙烯 eb2701, 交易单位 5, 标的期货保证金率 19%),
  CTP 以期权、标的昨结算价计算的期权交易成本, 与实际卖开 1 手后的持仓占用一致;
- 股指期权: 宏源期货实盘账户的 CTP 期权交易成本查询(交易日 20261123, 只查询),
  沪深 300 IO2612, 乘数 100, 合约保证金调整系数 12%。
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

import akquant
import pandas as pd
import pytest
from akquant import BarBatch, Strategy

OptionKind = Literal["CALL", "PUT"]
MarginModel = Literal["CHINA_COMMODITY", "CHINA_INDEX"]

DAYS = pd.date_range("2026-11-17 15:00", periods=3, freq="B", tz="Asia/Shanghai")
UNDERLYING = "eb2701"
UNDERLYING_PRICE = 8468.0
# (期权代码, 类型, 行权价, 期权价, SimNow 1 手保证金)
OPTIONS: List[Tuple[str, OptionKind, float, float, float]] = [
    ("eb2701-C-8000", "CALL", 8000.0, 607.5, 11082.10),  # 实值
    ("eb2701-C-8500", "CALL", 8500.0, 313.0, 9529.60),  # 平值(虚值 32 点)
    ("eb2701-P-6000", "PUT", 6000.0, 2.5, 4034.80),  # 深度虚值, 取标的保证金一半
]
INDEX = "000300"
INDEX_PRICE = 3866.4
# (期权代码, 类型, 行权价, 期权价, CTP 1 手保证金)
INDEX_OPTIONS: List[Tuple[str, OptionKind, float, float, float]] = [
    ("IO2612-C-3650", "CALL", 3650.0, 242.2, 70616.8),  # 实值
    ("IO2612-P-3850", "PUT", 3850.0, 78.8, 52636.8),  # 平值(虚值 16.4 点)
    ("IO2612-P-3550", "PUT", 3550.0, 14.0, 22700.0),  # 深度虚值, 下限按行权价
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
    symbol: str,
    kind: OptionKind,
    strike: float,
    underlying: str,
    model: Optional[MarginModel],
    multiplier: float = 5.0,
    margin_ratio: float = 1.0,
) -> akquant.InstrumentConfig:
    return akquant.InstrumentConfig(
        symbol=symbol,
        asset_type="OPTION",
        multiplier=multiplier,
        margin_ratio=margin_ratio,
        option_type=kind,
        strike_price=strike,
        expiry_date=20261218,
        underlying_symbol=underlying,
        option_margin_model=model,
    )


class _SellOne(Strategy):
    """首日卖开 ``target`` 1 手, 每日记录账户保证金与该合约的保证金模型."""

    target = ""

    def __init__(self) -> None:
        super().__init__()
        self.margins: List[float] = []
        self.margin_model: Optional[str] = None

    def on_bars(self, batch: BarBatch) -> None:
        if not self.margins and self.get_position(self.target) == 0:
            self.sell(self.target, 1)
            self.margin_model = self.get_instrument(self.target).option_margin_model
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


def test_commodity_short_option_margin_matches_simnow() -> None:
    """实值、平值、深度虚值各卖 1 手, 显式指定与缺省(标的为期货)都与 SimNow 一致."""
    futures = akquant.InstrumentConfig(
        symbol=UNDERLYING, asset_type="FUTURES", multiplier=5.0, margin_ratio=0.19
    )
    models: List[Optional[MarginModel]] = ["CHINA_COMMODITY", None]
    for model in models:
        for symbol, kind, strike, price, expected in OPTIONS:
            data = {
                UNDERLYING: _bars(UNDERLYING, UNDERLYING_PRICE),
                symbol: _bars(symbol, price),
            }
            configs = [futures, _option_config(symbol, kind, strike, UNDERLYING, model)]
            strategy, _ = _run(symbol, data, configs)
            assert strategy.margin_model == "CHINA_COMMODITY", (model, symbol)
            assert strategy.margins[0] == 0.0
            assert round(strategy.margins[-1], 2) == expected, (model, symbol)


def test_commodity_short_option_rejected_when_underlying_futures_not_registered() -> (
    None
):
    """标的期货未登记时卖开被拒, 原因写明取不到标的期货保证金率."""
    symbol, kind, strike, price, _ = OPTIONS[1]
    data = {symbol: _bars(symbol, price)}
    configs = [_option_config(symbol, kind, strike, UNDERLYING, "CHINA_COMMODITY")]
    strategy, result = _run(symbol, data, configs)
    orders = result.orders_df
    assert orders["status"].tolist() == ["rejected"]
    reason = str(orders["reject_reason"].iloc[0])
    assert f"{symbol} 无法计算期权保证金: 标的期货 {UNDERLYING} 未登记为合约" in reason
    assert strategy.margins[-1] == 0.0


def test_index_short_option_margin_matches_cffex() -> None:
    """中金所股指期权卖 1 手, 保证金与 CTP 查询一致; 标的为指数时缺省模型不变."""
    for symbol, kind, strike, price, expected in INDEX_OPTIONS:
        data = {INDEX: _bars(INDEX, INDEX_PRICE), symbol: _bars(symbol, price)}
        configs = [
            _option_config(symbol, kind, strike, INDEX, "CHINA_INDEX", 100.0, 0.12)
        ]
        strategy, _ = _run(symbol, data, configs)
        assert strategy.margin_model == "CHINA_INDEX"
        assert round(strategy.margins[-1], 2) == expected, symbol
    symbol, kind, strike, price, _ = INDEX_OPTIONS[0]
    data = {INDEX: _bars(INDEX, INDEX_PRICE), symbol: _bars(symbol, price)}
    strategy, _ = _run(
        symbol, data, [_option_config(symbol, kind, strike, INDEX, None, 100.0)]
    )
    assert strategy.margin_model == "CHINA_SINGLE_LEG"


def test_index_option_requires_explicit_adjustment_coefficient() -> None:
    """CHINA_INDEX 未填 margin_ratio 时配置报错, 不把缺省的 1.0 当调整系数."""
    symbol, kind, strike, _, _ = INDEX_OPTIONS[0]
    with pytest.raises(ValueError, match="须显式填写 margin_ratio"):
        akquant.InstrumentConfig(
            symbol=symbol,
            asset_type="OPTION",
            multiplier=100.0,
            option_type=kind,
            strike_price=strike,
            expiry_date=20261218,
            underlying_symbol=INDEX,
            option_margin_model="CHINA_INDEX",
        )


def test_index_short_option_rejected_without_adjustment_coefficient() -> None:
    """股指期权调整系数(margin_ratio)为 0 时卖开被拒并写明原因."""
    symbol, kind, strike, price, _ = INDEX_OPTIONS[0]
    data = {INDEX: _bars(INDEX, INDEX_PRICE), symbol: _bars(symbol, price)}
    configs = [_option_config(symbol, kind, strike, INDEX, "CHINA_INDEX", 100.0, 0.0)]
    _, result = _run(symbol, data, configs)
    orders = result.orders_df
    assert orders["status"].tolist() == ["rejected"]
    assert (
        f"{symbol} 无法计算期权保证金: 期权 {symbol} 的保证金调整系数(margin_ratio) 0"
        in str(orders["reject_reason"].iloc[0])
    )
