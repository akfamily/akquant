"""buy(quantity=None) 交给 sizer 定量时, 资金不足自动缩量 (提交时与成交时两道闸门)."""

from typing import Any, Optional

import pandas as pd
from akquant import Bar, Strategy, run_backtest
from akquant.sizer import AllInSizer, FixedSize, Sizer

SYMBOL = "RESIZE"


def _bars(prices: list[float]) -> list[Bar]:
    """北京时间 10:00(连续竞价)的 bar, 成交量充足."""
    stamps = pd.date_range(
        start="2023-01-03", periods=len(prices), freq="D", tz="Asia/Shanghai"
    )
    return [
        Bar(
            timestamp=(ts + pd.Timedelta(hours=10)).value,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1_000_000.0,
            symbol=SYMBOL,
        )
        for ts, price in zip(stamps, prices)
    ]


class _BuyOnce(Strategy):
    """首根 bar 买一次; sizer 为 None 时按 quantity 显式下单."""

    sizer_factory: Optional[type[Sizer]] = None
    sizer_args: tuple[Any, ...] = ()
    quantity: Optional[float] = None

    def __init__(self) -> None:
        super().__init__()
        self.sent = False

    def on_start(self) -> None:
        if self.sizer_factory is not None:
            self.sizer = self.sizer_factory(*self.sizer_args)

    def on_bar(self, bar: Bar) -> None:
        if not self.sent:
            self.sent = True
            self.buy(bar.symbol, self.quantity)


def _run(strategy: type[Strategy], prices: list[float]) -> Any:
    return run_backtest(
        data=_bars(prices),
        strategy=strategy,
        symbols=SYMBOL,
        initial_cash=31_000.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        lot_size=100,
        show_progress=False,
    )


class _AllIn(_BuyOnce):
    sizer_factory = AllInSizer


class _Fixed2700(_BuyOnce):
    sizer_factory = FixedSize
    sizer_args = (2700.0,)


class _Explicit3100(_BuyOnce):
    quantity = 3100.0


def test_sizer_order_resized_at_submission() -> None:
    """AllInSizer 按 10 元算出 3100 股, 留不出安全垫: 提交时缩到买得起的整手数."""
    result = _run(_AllIn, [10.0, 10.0, 10.0])
    fills = [float(t.quantity) for t in result.executions]
    assert fills == [3000.0]
    row = result.orders_df.iloc[0]
    assert row["status"] == "filled"
    assert float(row["quantity"]) == 3000.0


def test_sizer_order_resized_at_execution() -> None:
    """Sizer 给 2700 股, 按 10 元提交通过; 成交价涨到 12 时缩到 2500 股.

    订单数量、成交量与成交记录一致.
    """
    result = _run(_Fixed2700, [10.0, 12.0, 12.0])
    fills = [float(t.quantity) for t in result.executions]
    assert fills == [2500.0]
    row = result.orders_df.iloc[0]
    assert row["status"] == "filled"
    assert float(row["quantity"]) == 2500.0
    assert float(row["filled_quantity"]) == 2500.0


def test_explicit_quantity_is_not_resized() -> None:
    """显式给数量的买单不缩量, 资金不足照常拒单."""
    result = _run(_Explicit3100, [10.0, 10.0, 10.0])
    assert result.executions == []
    assert result.orders_df.iloc[0]["status"] == "rejected"
