"""short / cover / close_position 必须把订单标识返回给调用方.

三者此前签名硬写 ``-> None``: 内部确实把单报了出去(引擎里能看到对应订单与回调),
但**返回值被丢弃** —— 调用方拿不到 order id, 既无法 ``get_order`` 查询也无法
``cancel_order`` 撤单。同层的 ``buy`` / ``sell`` 返回 ``OrderReceipt``,
``order_target*`` 返回 ``str``, 三种返回形态并存正是测试反馈里
"报单异常 / 直接抛出的异常" 的一个来源(对返回值取 ``.primary`` 或 ``str`` 时炸)。
"""

from typing import Any, List, Optional

import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar

SYM = "600016.SH"
PRICE = 10.0


def _instrument() -> Instrument:
    return Instrument(
        symbol=SYM,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
    )


def _bars(days: tuple[int, ...]) -> List[Bar]:
    return [
        Bar(
            timestamp=int(
                pd.Timestamp(f"2023-01-{day:02d} 14:00:00", tz="Asia/Shanghai").value
            ),
            open=PRICE,
            high=PRICE + 0.2,
            low=PRICE - 0.2,
            close=PRICE,
            volume=1_000_000.0,
            symbol=SYM,
        )
        for day in days
    ]


class _ReturnValueProbe(Strategy):
    """在指定 bar 上调用一个下单 API, 记录其返回值与该 id 能否被查回."""

    def __init__(self, api: str = "short") -> None:
        self._api = api
        self.n = 0
        self.returned: Any = "<未调用>"
        self.lookup_ok: Optional[bool] = None

    def on_bar(self, bar: Bar) -> None:
        self.n += 1
        if self.n == 1:
            if self._api == "short":
                self.returned = self.short(SYM, 100, price=PRICE)
            elif self._api == "cover":
                self.returned = self.cover(SYM, 100, price=PRICE)
            elif self._api == "buy_then_close":
                self.buy(SYM, 100, price=PRICE)
            elif self._api == "close_without_holding":
                self.returned = self.close_position(SYM)
        elif self.n == 3 and self._api == "buy_then_close":
            self.returned = self.close_position(SYM)
        elif self.n == 4 and self.returned not in (None, "<未调用>"):
            # 拿到的标识必须真的能查回订单(否则"返回了个东西"没有意义)
            primary = getattr(self.returned, "primary", self.returned)
            self.lookup_ok = self.get_order(str(primary)) is not None


def _run(api: str, days: tuple[int, ...] = (3, 4, 5, 6, 9)) -> _ReturnValueProbe:
    strategy = _ReturnValueProbe(api)
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars(days)},
        cash=1_000_000.0,
        show_progress=False,
        duration="60s",
    )
    return strategy


def test_short_returns_order_identifier() -> None:
    """调用 short 报出单后, 必须返回可用的订单标识."""
    strategy = _run("short")
    assert strategy.returned is not None, "short 返回 None(但单已报出)"
    assert strategy.returned != "<未调用>"
    assert strategy.lookup_ok, f"short 返回的标识查不回订单: {strategy.returned!r}"


def test_cover_returns_order_identifier() -> None:
    """调用 cover 报出单后, 必须返回可用的订单标识."""
    strategy = _run("cover")
    assert strategy.returned is not None, "cover 返回 None(但单已报出)"
    assert strategy.returned != "<未调用>"
    assert strategy.lookup_ok, f"cover 返回的标识查不回订单: {strategy.returned!r}"


def test_close_position_returns_order_identifier() -> None:
    """有持仓时 close_position 必须返回它下出去的那张单的标识."""
    strategy = _run("buy_then_close")
    assert strategy.returned is not None, "close_position 返回 None(但平仓单已报出)"
    assert strategy.returned != "<未调用>"
    assert strategy.lookup_ok, (
        f"close_position 返回的标识查不回订单: {strategy.returned!r}"
    )


def test_close_position_without_holding_returns_none() -> None:
    """无持仓时没有单可下, 返回 None 是正确语义(不能凭空造一个标识)."""
    strategy = _run("close_without_holding")
    assert strategy.returned is None, (
        f"无持仓时 close_position 应返回 None, 实际: {strategy.returned!r}"
    )
