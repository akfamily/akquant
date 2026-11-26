"""on_order 回调里收到的订单快照与最终订单记录一致.

当根成交（``CurrentClose``、模拟 tick 上的单）时，订单在同一步里成交并离开在途列表，
``on_order(Filled)`` 收到的曾是下单时的快照：已成交量 0、均价为空，成交时被缩量的
订单数量仍是原始数量。``result.orders_df`` 一直是对的。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, cast

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy, Tick
from akquant.sizer import FixedSize

SYMBOL = "600000.SH"
TZ = "Asia/Shanghai"

Snapshot = Tuple[str, float, float, Optional[float]]


def _daily(closes: List[float], volume: float = 1e9) -> pd.DataFrame:
    """日线：开 = 收 − 0.5，高 = 收 + 1，低 = 收 − 1."""
    idx = pd.date_range("2024-01-02 15:00", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": [c - 0.5 for c in closes],
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [volume] * len(closes),
        },
        index=idx.tz_localize(TZ),
    )


def _snap(order: Any) -> Snapshot:
    avg = order.average_filled_price
    return (
        str(order.status).split(".")[-1],
        float(order.quantity),
        float(order.filled_quantity),
        None if avg is None else float(avg),
    )


class _Recorder(Strategy):
    """首根 bar 买入；记录 on_order 收到的快照、回调里 get_order 读到的与回调顺序."""

    quantity: Optional[float] = 100.0
    sizer_size: Optional[float] = None

    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.orders: List[Snapshot] = []
        self.looked_up: List[Snapshot] = []
        self.sequence: List[str] = []

    def on_start(self) -> None:
        if self.sizer_size is not None:
            self.sizer = FixedSize(self.sizer_size)

    def on_bar(self, bar: Bar) -> None:
        if self.n == 0:
            self.buy(bar.symbol, self.quantity)
        self.n += 1

    def on_order(self, order: Any) -> None:
        self.orders.append(_snap(order))
        found = self.get_order(order.id)
        assert found is not None
        self.looked_up.append(_snap(found))
        self.sequence.append("order:" + self.orders[-1][0])

    def on_trade(self, trade: Any) -> None:
        self.sequence.append(f"trade:{float(trade.quantity):g}@{float(trade.price):g}")


def _run(
    policy: Any,
    *,
    closes: Optional[List[float]] = None,
    quantity: Optional[float] = 100.0,
    sizer_size: Optional[float] = None,
    volume: float = 1e9,
    cash: float = 100_000.0,
    **kwargs: Any,
) -> Any:
    strategy = type(
        "_Run", (_Recorder,), {"quantity": quantity, "sizer_size": sizer_size}
    )
    return aq.run_backtest(
        data={SYMBOL: _daily(closes or [10.0, 11.0, 12.0], volume)},
        strategy=strategy,
        symbols=SYMBOL,
        freq="1d",
        fill_policy=policy,
        initial_cash=cash,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        show_progress=False,
        **kwargs,
    )


def _final(result: Any) -> Snapshot:
    row = result.orders_df.iloc[0]
    return (
        str(row["status"]).capitalize(),
        float(row["quantity"]),
        float(row["filled_quantity"]),
        float(row["avg_price"]),
    )


def test_current_close_filled_snapshot() -> None:
    """CurrentClose：当根收盘价 10 成交 100 股，Filled 回调带上已成交量与均价."""
    result = _run(aq.CurrentClose())
    strategy = result.strategy
    assert strategy.orders == [
        ("New", 100.0, 0.0, None),
        ("Filled", 100.0, 100.0, 10.0),
    ]
    assert strategy.looked_up == strategy.orders
    assert strategy.orders[-1] == _final(result)
    assert strategy.sequence == ["order:New", "order:Filled", "trade:100@10"]


def test_next_open_filled_snapshot() -> None:
    """NextOpen：下一根开盘价 10.5 成交，本来就正确，顺序与次数不变."""
    result = _run(aq.NextOpen())
    strategy = result.strategy
    assert strategy.orders == [
        ("New", 100.0, 0.0, None),
        ("Filled", 100.0, 100.0, 10.5),
    ]
    assert strategy.looked_up == strategy.orders
    assert strategy.orders[-1] == _final(result)
    assert strategy.sequence == ["order:New", "order:Filled", "trade:100@10.5"]


def test_current_close_resized_order_reports_actual_quantity() -> None:
    """Sizer 给 20000 股、资金 10 万：按收盘价 10 缩到 9999 股（安全垫 0.0001）."""
    result = _run(aq.CurrentClose(), quantity=None, sizer_size=20000.0)
    strategy = result.strategy
    expected_qty = float(int(100_000.0 * (1 - 0.0001) / 10.0))
    assert expected_qty == 9999.0
    assert strategy.orders == [
        ("New", 20000.0, 0.0, None),
        ("Filled", expected_qty, expected_qty, 10.0),
    ]
    assert strategy.orders[-1] == _final(result)
    assert strategy.looked_up[-1] == strategy.orders[-1]


def test_next_open_resized_order_reports_actual_quantity() -> None:
    """NextOpen 下缩量后的回调本来就正确（对照）."""
    result = _run(aq.NextOpen(), quantity=None, sizer_size=20000.0)
    strategy = result.strategy
    assert strategy.orders[-1] == _final(result)
    assert strategy.orders[-1][0] == "Filled"
    assert strategy.orders[-1][1] == strategy.orders[-1][2] < 20000.0


@pytest.mark.parametrize(
    ("policy", "first_price", "second_price"),
    [(aq.CurrentClose, 10.0, 11.0), (aq.NextOpen, 10.5, 11.5)],
)
def test_partial_fill_under_volume_limit(
    policy: Any, first_price: float, second_price: float
) -> None:
    """量限 50%、每根成交量 1000：1000 股分两根各成交 500."""
    result = _run(policy(), quantity=1000.0, volume=1000.0, volume_limit_pct=0.5)
    strategy = result.strategy
    average = (first_price + second_price) / 2
    assert strategy.orders == [
        ("New", 1000.0, 0.0, None),
        ("PartiallyFilled", 1000.0, 500.0, first_price),
        ("Filled", 1000.0, 1000.0, average),
    ]
    assert strategy.looked_up == strategy.orders
    assert strategy.orders[-1] == _final(result)
    assert strategy.sequence == [
        "order:New",
        "order:PartiallyFilled",
        f"trade:500@{first_price:g}",
        "order:Filled",
        f"trade:500@{second_price:g}",
    ]


class _TickBuyer(Strategy):
    """第一个模拟 tick 上市价买 100 股."""

    def __init__(self) -> None:
        super().__init__()
        self.done = False
        self.orders: List[Snapshot] = []

    def on_tick(self, tick: Tick) -> None:
        if not self.done:
            self.done = True
            self.buy(tick.symbol, 100)

    def on_order(self, order: Any) -> None:
        self.orders.append(_snap(order))


def test_simulated_tick_order_filled_snapshot() -> None:
    """模拟 tick 上的市价单按该 tick 价（开盘 9.5）当场成交."""
    result = aq.run_backtest(
        data={SYMBOL: _daily([10.0, 11.0, 12.0])},
        strategy=_TickBuyer,
        symbols=SYMBOL,
        freq="1d",
        simulate_ticks=True,
        initial_cash=100_000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        show_progress=False,
    )
    strategy = cast(Any, result.strategy)
    assert strategy.orders == [
        ("New", 100.0, 0.0, None),
        ("Filled", 100.0, 100.0, 9.5),
    ]
    assert strategy.orders[-1] == _final(result)


class _PaperBuyer(Strategy):
    """首根 bar 按当根收盘价成交的市价单；记录 on_order."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.orders: List[Snapshot] = []

    def on_bar(self, bar: Bar) -> None:
        if self.n == 0:
            self.buy(bar.symbol, 100, fill_mode=aq.CurrentClose())
        self.n += 1

    def on_order(self, order: Any) -> None:
        self.orders.append(_snap(order))


def test_paper_current_close_filled_snapshot() -> None:
    """模拟盘：单笔 fill_mode=CurrentClose 当根收盘价 10 成交."""
    index = pd.date_range("2024-01-02 09:31", periods=4, freq="1min").tz_localize(TZ)
    bars = [Bar(int(ts.value), 10.0, 10.0, 10.0, 10.0, 1e6, SYMBOL) for ts in index]
    strategy = _PaperBuyer()
    aq.run_live(
        strategy_cls=strategy,
        instruments=[
            aq.Instrument(
                symbol=SYMBOL,
                asset_type=aq.AssetType.Stock,
                multiplier=1.0,
                margin_ratio=1.0,
                tick_size=0.01,
                lot_size=100,
                option_type=None,
                strike_price=None,
                expiry_date=None,
            )
        ],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars, "freq": "1min"},
        cash=1_000_000.0,
        show_progress=False,
        duration="60s",
    )
    assert strategy.orders[0] == ("New", 100.0, 0.0, None)
    assert strategy.orders[-1] == ("Filled", 100.0, 100.0, 10.0)
