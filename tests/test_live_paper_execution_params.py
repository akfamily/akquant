"""run_live 的 paper 撮合参数: 费率、滑点、成交量限制.

paper 会话由 akquant 自己撮合(与回测同一个撮合器), 此前 ``run_live`` 不暴露这些
参数, 费率固定为 ``use_china_market()`` 的默认值、无滑点、不限量(#409 第 4 问)。
``broker_live`` 由柜台撮合, 传了这些参数必须报错, 不能让用户以为生效。
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import pytest
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar

_DAY1 = ["2023-01-03 09:31:00", "2023-01-03 09:32:00", "2023-01-03 09:33:00"]
_DAY2 = ["2023-01-04 09:31:00", "2023-01-04 09:32:00"]


def _instrument(symbol: str, asset_type: Any, tick_size: float = 0.01) -> Instrument:
    """构造一个整股可交易的标的."""
    return Instrument(
        symbol=symbol,
        asset_type=asset_type,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=tick_size,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bars(symbol: str, stamps: List[str], volume: float = 1e9) -> List[Bar]:
    """价格恒为 10 的 bar 序列."""
    return [
        Bar(
            timestamp=int(pd.Timestamp(text, tz="Asia/Shanghai").value),
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
            volume=volume,
            symbol=symbol,
        )
        for text in stamps
    ]


class _BuyThenSell(Strategy):
    """首根 bar 买入 ``qty``; 第二个交易日卖出(可选)."""

    def __init__(self) -> None:
        """记录成交; 参数由 ``_make`` 在构造后设置."""
        self.qty = 100_000.0
        self.sell_next_day = False
        self.trades: List[Any] = []
        self._bought = False
        self._sold = False

    def on_bar(self, bar: Bar) -> None:
        """第一根买, 次日第一根卖."""
        if not self._bought:
            self._bought = True
            self.buy(bar.symbol, self.qty)
            return
        day = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert("Asia/Shanghai").day
        if self.sell_next_day and not self._sold and day == 4:
            self._sold = True
            self.sell(bar.symbol, self.qty)

    def on_trade(self, trade: Any) -> None:
        """记录成交."""
        self.trades.append(trade)


def _make(qty: float = 100_000.0, sell_next_day: bool = False) -> _BuyThenSell:
    """构造策略实例并设置参数(不走构造参数, 避免策略参数声明告警)."""
    strat = _BuyThenSell()
    strat.qty = qty
    strat.sell_next_day = sell_next_day
    return strat


def _run(
    asset_type: Any = AssetType.Stock,
    *,
    strategy: Optional[_BuyThenSell] = None,
    bar_volume: float = 1e9,
    stamps: Optional[List[str]] = None,
    tick_size: float = 0.01,
    **params: Any,
) -> _BuyThenSell:
    """跑一个 replay paper 会话, 返回策略实例(带成交记录)."""
    symbol = "PAPER_A"
    strat = strategy or _BuyThenSell()
    run_live(
        strategy_cls=strat,
        instruments=[_instrument(symbol, asset_type, tick_size)],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars(symbol, stamps or _DAY1, bar_volume)},
        cash=10_000_000.0,
        show_progress=False,
        duration="60s",
        **params,
    )
    assert strat.trades, "paper 会话未产生成交"
    return strat


def _commission(trade: Any) -> float:
    return float(trade.commission)


@pytest.mark.parametrize("asset_type", [AssetType.Stock, AssetType.Fund])
def test_default_paper_fees_unchanged(asset_type: Any) -> None:
    """不传参数时沿用原默认费率: 万三佣金 + 十万分之一过户费(买入无印花税)."""
    trade = _run(asset_type).trades[0]

    # 100_000 股 × 10 元 = 1e6: 佣金 300 + 过户费 10
    assert _commission(trade) == pytest.approx(310.0)


@pytest.mark.parametrize("asset_type", [AssetType.Stock, AssetType.Fund])
def test_explicit_fees_apply_to_stock_and_fund(asset_type: Any) -> None:
    """显式费率对股票与基金(ETF)都生效."""
    trade = _run(
        asset_type,
        commission_rate=0.0001,
        transfer_fee_rate=0.0,
        min_commission=0.0,
    ).trades[0]

    assert _commission(trade) == pytest.approx(100.0)


def test_partial_fees_keep_other_defaults() -> None:
    """只传佣金率时, 过户费与最低佣金沿用默认值."""
    trade = _run(commission_rate=0.0001).trades[0]

    assert _commission(trade) == pytest.approx(110.0)


def test_min_commission_applies() -> None:
    """最低佣金生效: 1 股的佣金按最低佣金收."""
    trade = _run(
        strategy=_make(qty=1),
        commission_rate=0.0001,
        transfer_fee_rate=0.0,
        min_commission=1.0,
    ).trades[0]

    assert _commission(trade) == pytest.approx(1.0)


def test_stamp_tax_applies_to_stock_sell() -> None:
    """印花税只在股票卖出时收取."""
    strat = _run(
        strategy=_make(sell_next_day=True),
        stamps=_DAY1 + _DAY2,
        commission_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.001,
    )

    assert len(strat.trades) == 2
    assert _commission(strat.trades[0]) == pytest.approx(0.0)
    assert _commission(strat.trades[1]) == pytest.approx(1000.0)


def test_volume_limit_caps_fill() -> None:
    """volume_limit_pct 按 bar 成交量截断单笔成交."""
    strat = _run(
        strategy=_make(qty=1000),
        bar_volume=1000.0,
        volume_limit_pct=0.25,
    )

    assert float(strat.trades[0].quantity) == pytest.approx(250.0)


def test_no_volume_limit_by_default() -> None:
    """默认不限量, 与此前行为一致."""
    strat = _run(strategy=_make(qty=1000), bar_volume=1000.0)

    assert float(strat.trades[0].quantity) == pytest.approx(1000.0)


@pytest.mark.parametrize(
    ("slippage", "tick_size", "expected"),
    [
        ({"type": "percent", "value": 0.01}, 0.01, 10.1),
        ({"type": "fixed", "value": 0.05}, 0.01, 10.05),
        ({"type": "ticks", "value": 2}, 0.01, 10.02),
        ({"type": "zero"}, 0.01, 10.0),
    ],
)
def test_slippage_applies_to_buy_price(
    slippage: Dict[str, Any], tick_size: float, expected: float
) -> None:
    """滑点让买入成交价变差; ticks 按标的 tick_size 换算."""
    trade = _run(slippage=slippage, tick_size=tick_size).trades[0]

    assert float(trade.price) == pytest.approx(expected)


@pytest.mark.parametrize(
    "params",
    [
        {"commission_rate": 0.0001},
        {"stamp_tax_rate": 0.0},
        {"transfer_fee_rate": 0.0},
        {"min_commission": 0.0},
        {"slippage": {"type": "percent", "value": 0.001}},
        {"volume_limit_pct": 0.25},
    ],
)
def test_broker_live_rejects_paper_params(params: Dict[str, Any]) -> None:
    """broker_live 由柜台撮合, 传这些参数必须报错而不是静默忽略."""
    with pytest.raises(ValueError, match="paper"):
        run_live(
            strategy_cls=_BuyThenSell(),
            instruments=[_instrument("PAPER_A", AssetType.Stock)],
            broker="replay",
            trading_mode="broker_live",
            gateway_options={"bars": _bars("PAPER_A", _DAY1)},
            duration="60s",
            **params,
        )


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"commission_rate": -0.1}, ValueError),
        ({"min_commission": float("nan")}, ValueError),
        ({"volume_limit_pct": -0.25}, ValueError),
        ({"volume_limit_pct": float("inf")}, ValueError),
        ({"slippage": 0.001}, TypeError),
        ({"slippage": {"type": "bogus", "value": 1}}, ValueError),
    ],
)
def test_invalid_paper_params_rejected(
    params: Dict[str, Any], error: type[Exception]
) -> None:
    """非法取值在连网关之前报错."""
    with pytest.raises(error):
        run_live(
            strategy_cls=_BuyThenSell(),
            instruments=[_instrument("PAPER_A", AssetType.Stock)],
            broker="replay",
            trading_mode="paper",
            gateway_options={"bars": _bars("PAPER_A", _DAY1)},
            duration="60s",
            **params,
        )


def test_ticks_slippage_requires_single_tick_size() -> None:
    """Ticks 滑点要求所有标的 tick_size 相同(与回测同规则)."""
    with pytest.raises(ValueError, match="tick_size"):
        run_live(
            strategy_cls=_BuyThenSell(),
            instruments=[
                _instrument("PAPER_A", AssetType.Stock, 0.01),
                _instrument("PAPER_B", AssetType.Stock, 0.001),
            ],
            broker="replay",
            trading_mode="paper",
            gateway_options={"bars": _bars("PAPER_A", _DAY1)},
            duration="60s",
            slippage={"type": "ticks", "value": 1},
        )


def test_partial_fees_keep_stamp_and_min_defaults() -> None:
    """只传佣金率时, 印花税(卖出)与最低佣金也沿用默认值."""
    strat = _run(
        strategy=_make(sell_next_day=True),
        stamps=_DAY1 + _DAY2,
        commission_rate=0.0001,
    )
    # 卖出 1e6: 佣金 100 + 默认印花税 500 + 默认过户费 10
    assert _commission(strat.trades[1]) == pytest.approx(610.0)

    tiny = _run(strategy=_make(qty=1), commission_rate=0.0001).trades[0]
    # 1 股 × 10 元: 佣金按默认最低 5 元, 加过户费 0.0001
    assert _commission(tiny) == pytest.approx(5.0001)
