from typing import Any, cast

from akquant.gateway.ctp_adapter import CTPTraderAdapter
from akquant.gateway.mapper import create_default_mapper
from akquant.gateway.models import UnifiedOrderRequest, UnifiedOrderStatus


def _build_adapter(
    connected: bool = True, execution_semantics_mode: str = "strict"
) -> CTPTraderAdapter:
    class Gateway:
        def __init__(self, connected_flag: bool) -> None:
            self.connected = connected_flag
            self.login_status = connected_flag
            self.ready_to_trade = connected_flag

        def can_trade(self) -> bool:
            return self.connected and self.login_status and self.ready_to_trade

        def insert_order(self, **kwargs: object) -> dict[str, object]:
            client_order_id = str(kwargs.get("client_order_id", ""))
            return {
                "broker_order_id": f"ctp-1-2-{client_order_id}",
                "order_ref": f"ref-{client_order_id}",
                "timestamp_ns": 1,
            }

        def cancel_order(self, broker_order_id: str) -> None:
            _ = broker_order_id

    adapter = CTPTraderAdapter.__new__(CTPTraderAdapter)
    adapter.mapper = create_default_mapper()
    adapter.order_callback = None
    adapter.trade_callback = None
    adapter.execution_callback = None
    adapter.orders = {}
    adapter.trades = []
    adapter.trade_ids = set()
    adapter.positions = {}
    adapter.account = None
    adapter.client_to_broker_order_ids = {}
    adapter.broker_to_client_order_ids = {}
    adapter.order_ref_to_client_order_ids = {}
    adapter.pending_reject_reasons = {}
    adapter.execution_semantics_mode = execution_semantics_mode
    adapter._order_seq = 0
    adapter.gateway = cast(Any, Gateway(connected))
    return adapter


def test_ctp_adapter_place_and_cancel_order() -> None:
    """Cancel request should rely on native order callback to reach terminal state."""
    adapter = _build_adapter(connected=True)
    order_ids: list[str] = []
    report_ids: list[str] = []
    adapter.on_order(lambda order: order_ids.append(order.broker_order_id))
    adapter.on_execution_report(
        lambda report: report_ids.append(report.broker_order_id)
    )

    broker_order_id = adapter.place_order(
        UnifiedOrderRequest(
            client_order_id="c1",
            symbol="au2606",
            side="Buy",
            quantity=1.0,
        )
    )
    adapter.cancel_order(broker_order_id)
    adapter._handle_native_order_event(
        {
            "broker_order_id": broker_order_id,
            "client_order_id": "c1",
            "symbol": "au2606",
            "status": "cancelled",
            "timestamp_ns": 2,
        }
    )

    assert broker_order_id.startswith("ctp-1-2-c1")
    assert order_ids[0] == broker_order_id
    assert report_ids[0] == broker_order_id
    assert report_ids[-1] == broker_order_id
    assert adapter.query_order(broker_order_id) is not None


def test_ctp_adapter_forwards_position_effect_to_native_gateway() -> None:
    """CTP adapter should pass position_effect through native order insert."""
    adapter = _build_adapter(connected=True)
    captured: dict[str, object] = {}

    def _insert_order(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "broker_order_id": "ctp-1-2-ceffect",
            "order_ref": "ref-ceffect",
            "timestamp_ns": 1,
        }

    adapter.gateway.insert_order = _insert_order  # type: ignore[method-assign]

    broker_order_id = adapter.place_order(
        UnifiedOrderRequest(
            client_order_id="ceffect",
            symbol="au2606",
            side="Buy",
            quantity=1.0,
            position_effect="close",
        )
    )

    assert broker_order_id == "ctp-1-2-ceffect"
    assert captured["position_effect"] == "close"
    snapshot = adapter.query_order(broker_order_id)
    assert snapshot is not None
    assert snapshot.position_effect == "close"


def test_ctp_adapter_exposes_capabilities() -> None:
    """CTP adapter should expose explicit position_effect capability."""
    adapter = _build_adapter(connected=True)

    capabilities = adapter.get_capabilities().as_execution_capabilities()

    assert capabilities["broker_name"] == "ctp"
    assert capabilities["position_effect"] is True
    assert capabilities["reduce_only"] is False
    assert capabilities["position_details"] is True
    assert capabilities["supports_short_sell"] is True
    assert capabilities["supported_position_effects"] == [
        "auto",
        "open",
        "close",
        "close_today",
        "close_yesterday",
    ]


def test_ctp_adapter_rejects_reduce_only() -> None:
    """CTP adapter should fail fast for unsupported reduce_only orders."""
    adapter = _build_adapter(connected=True)

    try:
        adapter.place_order(
            UnifiedOrderRequest(
                client_order_id="c-reduce-only",
                symbol="au2606",
                side="Buy",
                quantity=1.0,
                reduce_only=True,
            )
        )
    except RuntimeError as exc:
        assert "does not support reduce_only" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for reduce_only")


def test_ctp_adapter_forwards_close_today_to_native_gateway() -> None:
    """CTP adapter should pass close_today through native order insert."""
    adapter = _build_adapter(connected=True)
    captured: dict[str, object] = {}

    def _insert_order(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "broker_order_id": "ctp-1-2-ctoday",
            "order_ref": "ref-ctoday",
            "timestamp_ns": 1,
        }

    adapter.gateway.insert_order = _insert_order  # type: ignore[method-assign]

    broker_order_id = adapter.place_order(
        UnifiedOrderRequest(
            client_order_id="ctoday",
            symbol="au2606",
            side="Buy",
            quantity=1.0,
            position_effect="close_today",
        )
    )

    assert broker_order_id == "ctp-1-2-ctoday"
    assert captured["position_effect"] == "close_today"


def test_ctp_adapter_query_positions_preserves_today_yesterday_breakdown() -> None:
    """CTP adapter should preserve today/yesterday quantities from native payload."""
    adapter = _build_adapter(connected=True)

    def _query_positions() -> list[dict[str, object]]:
        return [
            {
                "symbol": "au2606",
                "direction": "Buy",
                "quantity": 5.0,
                "available_quantity": 3.0,
                "today_quantity": 2.0,
                "yesterday_quantity": 3.0,
                "available_today_quantity": 1.0,
                "available_yesterday_quantity": 2.0,
                "avg_price": 510.5,
                "timestamp_ns": 11,
            }
        ]

    setattr(cast(Any, adapter.gateway), "query_positions", _query_positions)

    positions = adapter.query_positions()

    assert len(positions) == 1
    position = positions[0]
    assert position.symbol == "au2606"
    assert position.direction == "Buy"
    assert position.quantity == 5.0
    assert position.available_quantity == 3.0
    assert position.today_quantity == 2.0
    assert position.yesterday_quantity == 3.0
    assert position.available_today_quantity == 1.0
    assert position.available_yesterday_quantity == 2.0


def test_ctp_adapter_query_positions_keeps_both_directions_for_same_symbol() -> None:
    """CTP adapter should not overwrite long/short positions for the same symbol."""
    adapter = _build_adapter(connected=True)

    def _query_positions() -> list[dict[str, object]]:
        return [
            {
                "symbol": "IF2406",
                "direction": "Buy",
                "quantity": 2.0,
                "available_quantity": 2.0,
            },
            {
                "symbol": "IF2406",
                "direction": "Sell",
                "quantity": 1.0,
                "available_quantity": 1.0,
            },
        ]

    setattr(cast(Any, adapter.gateway), "query_positions", _query_positions)

    positions = adapter.query_positions()

    assert len(positions) == 2
    assert {position.direction for position in positions} == {"Buy", "Sell"}
    assert len(adapter.positions) == 2


def test_ctp_adapter_deduplicates_active_client_order_id() -> None:
    """Duplicate active client_order_id should return existing broker order id."""
    adapter = _build_adapter(connected=True)
    req = UnifiedOrderRequest(
        client_order_id="dup-c1",
        symbol="ag2606",
        side="Buy",
        quantity=1.0,
    )
    first = adapter.place_order(req)
    second = adapter.place_order(req)

    assert first == second
    assert len(adapter.orders) == 1


def test_ctp_adapter_ingest_events_update_state() -> None:
    """Ingested order/trade events should update adapter state."""
    adapter = _build_adapter(connected=True)
    trades: list[str] = []
    adapter.on_trade(lambda trade: trades.append(trade.trade_id))

    snapshot = adapter.ingest_order_event(
        {
            "client_order_id": "c2",
            "broker_order_id": "b2",
            "symbol": "rb2605",
            "status": "submitted",
            "timestamp_ns": 1,
        }
    )
    trade = adapter.ingest_trade_event(
        {
            "trade_id": "t2",
            "broker_order_id": "b2",
            "client_order_id": "c2",
            "symbol": "rb2605",
            "side": "Buy",
            "quantity": 1.0,
            "price": 100.0,
            "timestamp_ns": 2,
        }
    )

    assert snapshot.broker_order_id == "b2"
    assert trade.trade_id == "t2"
    assert trades == ["t2"]
    assert adapter.query_trades()[-1].trade_id == "t2"


def test_ctp_adapter_query_account_maps_native_snapshot() -> None:
    """CTP adapter should map native account query payload to UnifiedAccount."""
    adapter = _build_adapter(connected=True)

    def _query_account() -> dict[str, object]:
        return {
            "account_id": "acct-002",
            "equity": 200000.0,
            "cash": 200000.0,
            "available_cash": 150000.0,
            "margin": 30000.0,
            "frozen_cash": 2500.0,
            "unrealized_pnl": 0.0,
            "timestamp_ns": 99,
        }

    setattr(cast(Any, adapter.gateway), "query_account", _query_account)

    account = adapter.query_account()

    assert account is not None
    assert account.account_id == "acct-002"
    assert account.equity == 200000.0
    assert account.available_cash == 150000.0
    assert account.margin == 30000.0
    assert account.frozen_cash == 2500.0
    # 0.0 是柜台给出的值, 不能当成"没给"
    assert account.unrealized_pnl == 0.0
    assert account.market_value is None


def test_ctp_adapter_account_without_optional_fields_leaves_them_unset() -> None:
    """原生快照不带保证金等字段(旧载荷)时留 None, 不编造 0."""
    adapter = _build_adapter(connected=True)
    setattr(
        cast(Any, adapter.gateway),
        "query_account",
        lambda: {"account_id": "a", "equity": 1.0, "cash": 1.0, "available_cash": 1.0},
    )
    account = adapter.query_account()
    assert account is not None
    assert (account.margin, account.frozen_cash, account.unrealized_pnl) == (
        None,
        None,
        None,
    )


def test_ctp_adapter_sync_today_trades_maps_and_deduplicates() -> None:
    """CTP adapter should map sync trades and avoid duplicate cache entries."""
    adapter = _build_adapter(connected=True)

    def _query_trades_today() -> list[dict[str, object]]:
        return [
            {
                "trade_id": "t-sync",
                "broker_order_id": "b-sync",
                "client_order_id": "c-sync",
                "symbol": "au2606",
                "side": "Buy",
                "quantity": 1.0,
                "price": 500.0,
                "timestamp_ns": 10,
                "position_effect": "close",
            }
        ]

    setattr(cast(Any, adapter.gateway), "query_trades_today", _query_trades_today)

    first_sync = adapter.sync_today_trades()
    second_sync = adapter.sync_today_trades()

    assert len(first_sync) == 1
    assert first_sync[0].trade_id == "t-sync"
    assert first_sync[0].position_effect == "close"
    assert len(second_sync) == 1
    assert len(adapter.query_trades()) == 1


def test_ctp_adapter_native_event_flow_advances_status() -> None:
    """Native order and trade events should update unified state."""
    adapter = _build_adapter(connected=True)
    adapter.order_ref_to_client_order_ids["42"] = "c42"

    adapter._handle_native_order_event(
        {
            "order_ref": "42",
            "broker_order_id": "ctp-1-2-42",
            "symbol": "au2606",
            "status": "partially_filled",
            "filled_quantity": 1.0,
            "avg_fill_price": 500.0,
            "timestamp_ns": 10,
        }
    )
    adapter._handle_native_trade_event(
        {
            "order_ref": "42",
            "trade_id": "t42",
            "broker_order_id": "ctp-1-2-42",
            "symbol": "au2606",
            "side": "Buy",
            "quantity": 1.0,
            "price": 500.0,
            "timestamp_ns": 11,
        }
    )
    adapter._handle_native_order_event(
        {
            "order_ref": "42",
            "broker_order_id": "ctp-1-2-42",
            "symbol": "au2606",
            "status": "filled",
            "filled_quantity": 1.0,
            "avg_fill_price": 500.0,
            "timestamp_ns": 12,
        }
    )

    snapshot = adapter.query_order("ctp-1-2-42")
    assert snapshot is not None
    assert snapshot.status == UnifiedOrderStatus.FILLED
    assert adapter.query_trades()[-1].trade_id == "t42"


def test_ctp_adapter_native_error_maps_to_rejected() -> None:
    """Native reject event should wait for order callback to mark rejected."""
    adapter = _build_adapter(connected=True)
    adapter.place_order(
        UnifiedOrderRequest(
            client_order_id="r1",
            symbol="ag2606",
            side="Buy",
            quantity=1.0,
        )
    )
    adapter._handle_native_error_event(
        {
            "broker_order_id": "ctp-1-2-r1",
            "client_order_id": "r1",
            "symbol": "ag2606",
            "error_message": "insufficient funds",
            "timestamp_ns": 13,
        }
    )
    snapshot_before = adapter.query_order("ctp-1-2-r1")
    assert snapshot_before is not None
    assert snapshot_before.status == UnifiedOrderStatus.SUBMITTED
    adapter._handle_native_order_event(
        {
            "broker_order_id": "ctp-1-2-r1",
            "client_order_id": "r1",
            "symbol": "ag2606",
            "status": "rejected",
            "timestamp_ns": 14,
        }
    )

    snapshot = adapter.query_order("ctp-1-2-r1")
    assert snapshot is not None
    assert snapshot.status == UnifiedOrderStatus.REJECTED
    assert snapshot.reject_reason == "insufficient funds"


def test_ctp_adapter_compatible_mode_rejects_on_error_event() -> None:
    """Compatible mode should immediately mark rejected on native error event."""
    adapter = _build_adapter(connected=True, execution_semantics_mode="compatible")
    adapter.place_order(
        UnifiedOrderRequest(
            client_order_id="r2",
            symbol="ag2606",
            side="Buy",
            quantity=1.0,
        )
    )
    adapter._handle_native_error_event(
        {
            "broker_order_id": "ctp-1-2-r2",
            "client_order_id": "r2",
            "symbol": "ag2606",
            "error_message": "risk blocked",
            "timestamp_ns": 15,
        }
    )

    snapshot = adapter.query_order("ctp-1-2-r2")
    assert snapshot is not None
    assert snapshot.status == UnifiedOrderStatus.REJECTED
    assert snapshot.reject_reason == "risk blocked"


def test_ctp_adapter_place_order_requires_connection() -> None:
    """Placing order without connection should fail fast."""
    adapter = _build_adapter(connected=False)
    try:
        adapter.place_order(
            UnifiedOrderRequest(
                client_order_id="c3",
                symbol="au2606",
                side="Sell",
                quantity=1.0,
            )
        )
    except RuntimeError as exc:
        assert "not connected" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for disconnected ctp trader")


def test_ctp_adapter_sync_open_orders_excludes_terminal_orders() -> None:
    """Cancelled orders should be excluded from open order sync."""
    adapter = _build_adapter(connected=True)
    broker_order_id = adapter.place_order(
        UnifiedOrderRequest(
            client_order_id="c4",
            symbol="ag2606",
            side="Buy",
            quantity=1.0,
        )
    )
    open_orders = adapter.sync_open_orders()
    assert any(order.broker_order_id == broker_order_id for order in open_orders)

    adapter.cancel_order(broker_order_id)
    adapter._handle_native_order_event(
        {
            "broker_order_id": broker_order_id,
            "client_order_id": "c4",
            "symbol": "ag2606",
            "status": "cancelled",
            "timestamp_ns": 3,
        }
    )
    open_orders_after_cancel = adapter.sync_open_orders()
    assert all(
        order.broker_order_id != broker_order_id for order in open_orders_after_cancel
    )


def test_ctp_adapter_order_snapshot_carries_side_quantity_and_price() -> None:
    """下单后的委托快照带上方向、委托数量、委托价(策略读挂单时用)."""
    adapter = _build_adapter(connected=True)
    adapter.gateway.insert_order = lambda **kwargs: {  # type: ignore[method-assign]
        "broker_order_id": "ctp-1-2-cside",
        "timestamp_ns": 1,
    }

    broker_order_id = adapter.place_order(
        UnifiedOrderRequest(
            client_order_id="cside",
            symbol="au2606",
            side="Sell",
            quantity=3.0,
            price=512.5,
            order_type="Limit",
        )
    )

    snapshot = adapter.query_order(broker_order_id)
    assert snapshot is not None
    assert (snapshot.side, snapshot.quantity, snapshot.price) == ("Sell", 3.0, 512.5)


def test_ctp_adapter_order_event_carries_side_quantity_and_price() -> None:
    """柜台推来的委托回报里有方向/数量/价格就带进快照, 没有留 None."""
    adapter = _build_adapter(connected=True)
    base = {"broker_order_id": "b1", "symbol": "au2606", "status": "submitted"}

    full = adapter.ingest_order_event(
        {**base, "side": "Buy", "quantity": 2.0, "price": 510.0}
    )
    assert (full.side, full.quantity, full.price) == ("Buy", 2.0, 510.0)

    bare = adapter.ingest_order_event(dict(base))
    assert (bare.side, bare.quantity, bare.price) == (None, None, None)


def test_ctp_adapter_position_carries_margin_and_position_profit() -> None:
    """持仓行里的占用保证金、持仓盈亏带进 UnifiedPosition; 没有留 None."""
    adapter = _build_adapter(connected=True)
    rows: list[dict[str, object]] = [
        {
            "symbol": "au2606",
            "direction": "Buy",
            "quantity": 5.0,
            "margin": 25000.0,
            "unrealized_pnl": -300.0,
        },
        {"symbol": "ag2606", "direction": "Sell", "quantity": 1.0},
    ]
    setattr(cast(Any, adapter.gateway), "query_positions", lambda: rows)

    reported, bare = adapter.query_positions()

    assert (reported.margin, reported.unrealized_pnl) == (25000.0, -300.0)
    assert (bare.margin, bare.unrealized_pnl) == (None, None)
