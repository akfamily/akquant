"""柜台不认 ``auto`` 时, 核心把 ``auto`` 委托定成明确的开平腿.

CTP 的报单必须带开平标志, 网关把 ``auto`` 当成开仓。此前核心只在卖出量**超过**反向
持仓时才拆成"平仓 + 开仓"; 卖出量在可平数量以内时原样传 ``auto``, 于是持多 5 手
卖 3 手不是平掉 3 手多头, 而是新开 3 手空头(锁仓)。
"""

from __future__ import annotations

from typing import Any

import pytest
from akquant.gateway.broker_models import BrokerCapability
from akquant.gateway.brokers.ctp.adapter import CTPTraderAdapter
from akquant.gateway.order_submitter import (
    PositionEffectUnresolved,
    resolve_live_order_legs,
)

SYMBOL = "rb2601"


class _Pos:
    def __init__(
        self,
        direction: str,
        quantity: float,
        available_today: float | None = None,
        available_yesterday: float | None = None,
    ) -> None:
        self.symbol = SYMBOL
        self.direction = direction
        self.quantity = quantity
        self.available_today_quantity = available_today
        self.available_yesterday_quantity = available_yesterday


class _Gw:
    def __init__(self, positions: list[_Pos] | None, fail: bool = False) -> None:
        self._positions = positions or []
        self._fail = fail

    def query_positions(self) -> list[_Pos]:
        if self._fail:
            raise RuntimeError("position query timed out")
        return self._positions


def _cap(explicit: bool = True) -> BrokerCapability:
    return BrokerCapability(
        broker_name="t",
        position_effect=True,
        position_details=True,
        supported_position_effects=(
            "auto",
            "open",
            "close",
            "close_today",
            "close_yesterday",
        ),
        features=frozenset({"explicit_offset"}) if explicit else frozenset(),
    )


def _legs(
    gw: _Gw, side: str, quantity: float, explicit: bool = True
) -> list[tuple[str, float]]:
    return resolve_live_order_legs(
        trader_gateway=gw,
        capability=_cap(explicit),
        symbol=SYMBOL,
        side=side,
        quantity=quantity,
        position_effect="auto",
        reduce_only=False,
        payload_field=lambda obj, name: getattr(obj, name, None),
    )


def test_ctp_declares_it_needs_explicit_offset() -> None:
    """CTP 网关声明自己不认 auto, 开平须由核心定."""
    adapter = CTPTraderAdapter.__new__(CTPTraderAdapter)
    assert "explicit_offset" in adapter.get_capabilities().features


def test_sell_within_long_position_closes_instead_of_opening() -> None:
    """持多 5 手(今 2 昨 3)卖 3 手: 平今 2 + 平昨 1, 没有开仓腿."""
    gw = _Gw([_Pos("long", 5.0, available_today=2.0, available_yesterday=3.0)])
    assert _legs(gw, "sell", 3.0) == [("close_today", 2.0), ("close_yesterday", 1.0)]


def test_buy_within_short_position_closes_instead_of_opening() -> None:
    """持空 4 手买 4 手: 全部是平仓腿."""
    gw = _Gw([_Pos("short", 4.0, available_today=0.0, available_yesterday=4.0)])
    assert _legs(gw, "buy", 4.0) == [("close_yesterday", 4.0)]


def test_order_exceeding_position_closes_then_opens() -> None:
    """持多 5 手卖 8 手: 平 5 + 开 3(原有的反手行为不变)."""
    gw = _Gw([_Pos("long", 5.0, available_today=5.0, available_yesterday=0.0)])
    assert _legs(gw, "sell", 8.0) == [("close_today", 5.0), ("open", 3.0)]


def test_no_opposite_position_opens() -> None:
    """没有可平的反向持仓(空仓, 或只有同向持仓): 开仓, 不留 auto."""
    assert _legs(_Gw([]), "buy", 2.0) == [("open", 2.0)]
    assert _legs(_Gw([_Pos("long", 5.0, 5.0, 0.0)]), "buy", 2.0) == [("open", 2.0)]


def test_fully_frozen_position_opens() -> None:
    """反向持仓全被平仓挂单冻结(可平为 0): 开仓, 与回测的可平口径一致."""
    gw = _Gw([_Pos("long", 5.0, available_today=0.0, available_yesterday=0.0)])
    assert _legs(gw, "sell", 6.0) == [("open", 6.0)]


def test_unknown_position_is_refused_not_guessed() -> None:
    """持仓查不到时不猜开平: 报单会被本地拒绝并写明原因."""
    with pytest.raises(PositionEffectUnresolved) as excinfo:
        _legs(_Gw(None, fail=True), "sell", 3.0)
    assert "position query timed out" in str(excinfo.value)


def test_brokers_that_resolve_auto_themselves_are_untouched() -> None:
    """没有声明 explicit_offset 的柜台(由柜台自己解析 auto): 可平量以内仍原样传 auto."""
    gw: Any = _Gw([_Pos("long", 5.0, available_today=2.0, available_yesterday=3.0)])
    assert _legs(gw, "sell", 3.0, explicit=False) == [("auto", 3.0)]
    assert _legs(_Gw(None, fail=True), "sell", 3.0, explicit=False) == [("auto", 3.0)]


def test_submitter_rejects_locally_when_offset_cannot_be_resolved() -> None:
    """下单器: 持仓查不到时不报单, 返回拒单回执, 策略经 on_reject 收到原因."""
    from akquant.gateway.order_submitter import BrokerOrderSubmitter

    placed: list[Any] = []
    rejects: list[Any] = []

    class _FailingGw(_Gw):
        def place_order(self, req: Any) -> str:
            placed.append(req)
            return "b1"

    class _Strategy:
        strategy_id = "s"
        current_bar = None
        current_tick = None
        _last_prices: dict[str, float] = {}

        def on_reject(self, order: Any) -> None:
            rejects.append(order)

    capability = _cap()
    submitter = BrokerOrderSubmitter(
        trader_gateway=_FailingGw(None, fail=True),
        strategy=_Strategy(),
        resolve_trader_capabilities=lambda _gw: capability,
        next_client_order_id=lambda: "c1",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda *_a, **_k: None,
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: capability.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
    )

    receipt = submitter.submit_order(
        symbol=SYMBOL, side="Sell", quantity=3.0, price=3500.0, order_type="Limit"
    )

    assert placed == []
    assert receipt.failure == "rejected"
    assert len(rejects) == 1
    assert "position query timed out" in str(rejects[0].reject_reason)
