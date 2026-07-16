from typing import Any

import pytest
from akquant.gateway.broker_models import BrokerCapability, UnifiedOrderRequest
from akquant.gateway.order_submitter import (
    BrokerOrderSubmitter,
    resolve_live_order_legs,
)


class _Strategy:
    """Bare strategy object for id/owner attributes."""

    _owner_strategy_id = "_default"


def _make_submitter(
    captured: list, capability: BrokerCapability
) -> BrokerOrderSubmitter:
    """Build a submitter whose gateway records the placed request."""

    class _Gw:
        def place_order(self, req: UnifiedOrderRequest) -> str:
            captured.append(req)
            return "b1"

    return BrokerOrderSubmitter(
        trader_gateway=_Gw(),
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


def test_submit_with_declared_extra_reaches_place_order() -> None:
    """Declared extra + asset_type are normalized onto the request."""
    captured: list = []
    cap = BrokerCapability(broker_name="qmf", broker_extra_fields=("entrust_oc",))
    sub = _make_submitter(captured, cap)
    sub.submit_order(
        symbol="10004321.SH",
        side="Buy",
        quantity=1,
        price=0.05,
        order_type="Limit",
        asset_type="opt",
        extra={"entrust_oc": "O"},
    )
    assert len(captured) == 1
    assert captured[0].asset_type == "option"  # 归一
    assert captured[0].extra == {"entrust_oc": "O"}


def test_submit_with_undeclared_extra_raises() -> None:
    """Undeclared extra keys are rejected."""
    captured: list = []
    cap = BrokerCapability(broker_name="qmf", broker_extra_fields=())
    sub = _make_submitter(captured, cap)
    with pytest.raises(RuntimeError):
        sub.submit_order(
            symbol="600000.SH",
            side="Buy",
            quantity=100,
            price=10.5,
            order_type="Limit",
            extra={"nope": "1"},
        )


class _Pos:
    def __init__(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        available_today_quantity: float | None = None,
        available_yesterday_quantity: float | None = None,
    ) -> None:
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.available_today_quantity = available_today_quantity
        self.available_yesterday_quantity = available_yesterday_quantity


class _GW:
    def __init__(self, positions: list[_Pos]) -> None:
        self._positions = positions

    def query_positions(self) -> list[_Pos]:
        return self._positions


def _field(pos: Any, name: str) -> Any:
    return getattr(pos, name, None)


def _cap(**kw: Any) -> BrokerCapability:
    base: dict[str, Any] = dict(
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
    )
    base.update(kw)
    return BrokerCapability(**base)


def test_auto_reverse_splits_close_plus_open() -> None:
    """持多 5，卖 6 → 平 5 + 开 1."""
    gw = _GW([_Pos("rb2410.SHFE", "long", 5.0)])
    legs = resolve_live_order_legs(
        trader_gateway=gw,
        capability=_cap(),
        symbol="rb2410.SHFE",
        side="sell",
        quantity=6.0,
        position_effect="auto",
        reduce_only=False,
        payload_field=_field,
    )
    assert legs[-1] == ("open", 1.0)
    assert sum(q for _, q in legs) == 6.0
    assert sum(q for e, q in legs if e != "open") == 5.0


def test_auto_reverse_uses_available_when_frozen() -> None:
    """部分冻结: raw 5, 可用 today0+yest3(冻2); 卖6 → 平3(可用)+开3, 无不可平腿."""
    pos = _Pos(
        "rb2410.SHFE",
        "long",
        5.0,
        available_today_quantity=0.0,
        available_yesterday_quantity=3.0,
    )
    gw = _GW([pos])
    legs = resolve_live_order_legs(
        trader_gateway=gw,
        capability=_cap(),
        symbol="rb2410.SHFE",
        side="sell",
        quantity=6.0,
        position_effect="auto",
        reduce_only=False,
        payload_field=_field,
    )
    assert legs[-1] == ("open", 3.0)
    assert sum(q for _, q in legs) == 6.0
    close_sum = sum(q for e, q in legs if e != "open")
    assert close_sum == 3.0
    assert all(e in ("close_today", "close_yesterday", "open") for e, _ in legs)


def test_auto_reverse_gated_off_when_broker_declares_feature() -> None:
    """当 broker 声明 auto_reverse 时，核心不拆腿."""
    gw = _GW([_Pos("rb2410.SHFE", "long", 5.0)])
    legs = resolve_live_order_legs(
        trader_gateway=gw,
        capability=_cap(features=frozenset({"auto_reverse"})),
        symbol="rb2410.SHFE",
        side="sell",
        quantity=6.0,
        position_effect="auto",
        reduce_only=False,
        payload_field=_field,
    )
    assert legs == [("auto", 6.0)]


def test_auto_no_split_when_within_position() -> None:
    """持多 5，卖 3 → 不反手（不产生 open 腿）."""
    gw = _GW([_Pos("rb2410.SHFE", "long", 5.0)])
    legs = resolve_live_order_legs(
        trader_gateway=gw,
        capability=_cap(),
        symbol="rb2410.SHFE",
        side="sell",
        quantity=3.0,
        position_effect="auto",
        reduce_only=False,
        payload_field=_field,
    )
    assert all(e != "open" for e, _ in legs)
    assert sum(q for _, q in legs) == 3.0
