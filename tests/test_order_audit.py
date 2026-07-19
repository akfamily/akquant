"""Tests for order-lifecycle audit logging (RFC G1)."""

import json
import logging
from collections.abc import Generator
from pathlib import Path

import pytest
from akquant.gateway.broker_models import (
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedTrade,
)
from akquant.gateway.order_audit import (
    record_broker_event,
    record_cancel,
    record_reject,
    record_submit,
)
from akquant.log import (
    LogConfig,
    build_order_audit_extra,
    configure_logging,
)

_FILLED = UnifiedOrderStatus.FILLED


class _CaptureHandler(logging.Handler):
    """Collect emitted records for assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def capture_audit() -> Generator[_CaptureHandler, None, None]:
    """Attach a capture handler to the dedicated order-audit logger."""
    audit_logger = logging.getLogger("akquant.audit.order")
    handler = _CaptureHandler()
    audit_logger.addHandler(handler)
    previous_level = audit_logger.level
    audit_logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        audit_logger.removeHandler(handler)
        audit_logger.setLevel(previous_level)


class TestBuildOrderAuditExtra:
    """The structured payload keeps numbers numeric and ids normalized."""

    def test_numeric_fields_stay_numeric(self) -> None:
        """Price/quantity remain float for machine reconciliation."""
        extra = build_order_audit_extra(
            event="order_fill", price="10.5", quantity=100, symbol="600000.SH"
        )
        assert extra["price"] == 10.5
        assert extra["quantity"] == 100
        assert extra["event"] == "order_fill"
        assert extra["phase"] == "gateway"

    def test_trace_id_is_carried(self) -> None:
        """trace_id (logical-order correlation) is normalized and carried."""
        extra = build_order_audit_extra(event="order_submit", trace_id="  G1 ")
        assert extra["trace_id"] == "G1"

    def test_blank_fields_normalize_to_none(self) -> None:
        """Empty strings collapse to None so JSON omits them."""
        extra = build_order_audit_extra(event="order_submit", symbol="  ", reason="")
        assert extra["symbol"] is None
        assert extra["reason"] is None


class TestRecordFunctions:
    """Each transition emits one audit record with the right event tag."""

    def test_record_submit(self, capture_audit: _CaptureHandler) -> None:
        """A submit produces an order_submit record carrying both ids."""
        record_submit(
            strategy_id="alpha",
            symbol="600000.SH",
            side="Buy",
            quantity=100,
            price=10.5,
            client_order_id="C1",
            broker_order_id="B1",
            order_type="Limit",
        )
        assert len(capture_audit.records) == 1
        rec = capture_audit.records[0]
        assert getattr(rec, "event") == "order_submit"
        assert getattr(rec, "client_order_id") == "C1"
        assert getattr(rec, "order_id") == "B1"
        assert rec.levelno == logging.INFO

    def test_record_reject_is_warning(self, capture_audit: _CaptureHandler) -> None:
        """A reject is a WARNING with a reason."""
        record_reject(
            strategy_id="alpha",
            symbol="600000.SH",
            client_order_id="C1",
            reason="duplicate active client_order_id",
        )
        rec = capture_audit.records[0]
        assert getattr(rec, "event") == "order_reject"
        assert rec.levelno == logging.WARNING
        assert "duplicate" in str(getattr(rec, "reason"))

    def test_record_cancel(self, capture_audit: _CaptureHandler) -> None:
        """A cancel request records the broker order id."""
        record_cancel(broker_order_id="B1", strategy_id="alpha")
        rec = capture_audit.records[0]
        assert getattr(rec, "event") == "order_cancel"
        assert getattr(rec, "order_id") == "B1"

    def test_record_broker_event_trade_fill(
        self, capture_audit: _CaptureHandler
    ) -> None:
        """A trade payload maps to an order_fill with price/qty/trade_id."""
        trade = UnifiedTrade(
            trade_id="T1",
            broker_order_id="B1",
            client_order_id="C1",
            symbol="600000.SH",
            side="Buy",
            quantity=100.0,
            price=10.5,
            timestamp_ns=0,
        )
        record_broker_event("trade", trade, owner_strategy_id="alpha")
        rec = capture_audit.records[0]
        assert getattr(rec, "event") == "order_fill"
        assert getattr(rec, "trade_id") == "T1"
        assert getattr(rec, "price") == 10.5
        assert getattr(rec, "quantity") == 100.0

    def test_record_broker_event_order_update(
        self, capture_audit: _CaptureHandler
    ) -> None:
        """An order snapshot maps to an order_update with a status string."""
        snapshot = UnifiedOrderSnapshot(
            client_order_id="C1",
            broker_order_id="B1",
            symbol="600000.SH",
            status=_FILLED,
            filled_quantity=100.0,
            avg_fill_price=10.5,
        )
        record_broker_event("order", snapshot, owner_strategy_id="alpha")
        rec = capture_audit.records[0]
        assert getattr(rec, "event") == "order_update"
        assert getattr(rec, "order_status")  # non-empty status text
        assert getattr(rec, "quantity") == 100.0

    def test_unknown_event_is_ignored(self, capture_audit: _CaptureHandler) -> None:
        """Non-order events (e.g. account) produce no audit noise."""
        record_broker_event("account", {"account_id": "A1"}, owner_strategy_id="alpha")
        assert capture_audit.records == []

    def test_bad_payload_never_raises(self, capture_audit: _CaptureHandler) -> None:
        """A malformed payload must not break the trading path."""
        # object() has no fields and is not a dict — extraction must stay safe.
        record_broker_event("trade", object(), owner_strategy_id="alpha")
        # Still emits (best-effort), just with None fields — never raises.
        assert len(capture_audit.records) == 1


class TestOrderAuditFile:
    """configure_logging wires a dedicated JSON audit file (RFC G1)."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> Generator[None, None, None]:
        """Restore library-silent default after each test."""
        yield
        configure_logging(LogConfig(console=False, filename=None, reset_handlers=True))

    def test_audit_events_land_in_dedicated_file(self, tmp_path: Path) -> None:
        """submit/fill/cancel all appear as JSON lines with client_order_id."""
        audit_file = tmp_path / "orders_audit.log"
        configure_logging(
            LogConfig(
                console=False,
                filename=None,
                order_audit_file=str(audit_file),
                level="INFO",
            )
        )
        record_submit(
            strategy_id="alpha",
            symbol="600000.SH",
            side="Buy",
            quantity=100,
            price=10.5,
            client_order_id="C1",
            broker_order_id="B1",
            order_type="Limit",
        )
        record_broker_event(
            "trade",
            UnifiedTrade(
                trade_id="T1",
                broker_order_id="B1",
                client_order_id="C1",
                symbol="600000.SH",
                side="Buy",
                quantity=100.0,
                price=10.5,
                timestamp_ns=0,
            ),
            owner_strategy_id="alpha",
        )
        record_cancel(broker_order_id="B1", strategy_id="alpha")

        for handler in logging.getLogger("akquant.audit.order").handlers:
            handler.flush()
        lines = [
            json.loads(line)
            for line in audit_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        events = [entry["event"] for entry in lines]
        assert events == ["order_submit", "order_fill", "order_cancel"]
        assert all(entry.get("client_order_id") == "C1" for entry in lines[:2])
        # The full lifecycle is reconstructable from this file alone.
        assert lines[0]["order_id"] == "B1"
        assert lines[1]["price"] == 10.5

    def test_trace_id_shared_across_submit_and_fill(self, tmp_path: Path) -> None:
        """RFC G5: submit and fill of one logical order share one trace_id."""
        audit_file = tmp_path / "trace_audit.log"
        configure_logging(
            LogConfig(
                console=False,
                filename=None,
                order_audit_file=str(audit_file),
                level="INFO",
            )
        )
        trace = "GROUP-1"  # logical-order root id (group client_order_id)
        record_submit(
            strategy_id="alpha",
            symbol="600000.SH",
            side="Buy",
            quantity=100,
            price=10.5,
            client_order_id="GROUP-1-a-0",
            broker_order_id="B1",
            order_type="Limit",
            trace_id=trace,
        )
        # The fill push arrives with only the leg client_order_id; the bridge
        # resolves it back to the group trace_id via _lookup_group_id.
        record_broker_event(
            "trade",
            UnifiedTrade(
                trade_id="T1",
                broker_order_id="B1",
                client_order_id="GROUP-1-a-0",
                symbol="600000.SH",
                side="Buy",
                quantity=100.0,
                price=10.5,
                timestamp_ns=0,
            ),
            owner_strategy_id="alpha",
            trace_id=trace,
        )
        for handler in logging.getLogger("akquant.audit.order").handlers:
            handler.flush()
        lines = [
            json.loads(line)
            for line in audit_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        trace_ids = {entry["trace_id"] for entry in lines}
        assert trace_ids == {trace}  # one shared trace across submit + fill
        # grep-by-trace reconstructs the whole logical order
        assert [entry["event"] for entry in lines] == ["order_submit", "order_fill"]
