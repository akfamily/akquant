import json

import pytest

pytest.importorskip("websocket")

from akquant.gateway.brokers.qmf.ws import QMFPushClient


def _make_client() -> tuple[QMFPushClient, list, list]:
    """Build a push client capturing push events and status frames."""
    events: list = []
    statuses: list = []
    client = QMFPushClient(
        ws_url="ws://gw.test/api/v1/stream",
        token="gw-abc",
        on_push=lambda event, data: events.append((event, data)),
        on_status=lambda kind, frame: statuses.append(kind),
    )
    return client, events, statuses


def test_handle_push_frame() -> None:
    """A push frame dispatches (event, data) to on_push."""
    client, events, _ = _make_client()
    client._handle_message(
        json.dumps(
            {
                "type": "push",
                "event": "trade_update",
                "data": {"entrust_no": "100000001", "business_amount": "100"},
            }
        )
    )
    assert events == [
        ("trade_update", {"entrust_no": "100000001", "business_amount": "100"})
    ]


def test_handle_status_frame() -> None:
    """Status frames go to on_status, not on_push."""
    client, events, statuses = _make_client()
    client._handle_message(json.dumps({"type": "ready"}))
    client._handle_message(json.dumps({"type": "heartbeat"}))
    assert events == []
    assert statuses == ["ready", "heartbeat"]


def test_handle_malformed_frame_is_ignored() -> None:
    """A non-JSON frame is ignored without raising."""
    client, events, statuses = _make_client()
    client._handle_message("not-json")
    assert events == []
