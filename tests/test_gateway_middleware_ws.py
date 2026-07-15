import pytest

pytest.importorskip("websocket")

from akquant.gateway.brokers.middleware.ws import MiddlewarePushClient


def _client(seen: list, status: list | None = None) -> MiddlewarePushClient:
    """Build a push client whose callbacks append to the given lists."""
    return MiddlewarePushClient(
        ws_url="ws://gw.test/api/v1/ws?accounts=hengsheng:1:security",
        token="",
        on_push=lambda channel, data: seen.append((channel, data)),
        on_status=(None if status is None else (lambda kind, f: status.append(kind))),
    )


def test_book_order_frame_dispatched_to_on_push() -> None:
    """A book.order frame is dispatched with its data payload."""
    seen: list = []
    client = _client(seen)
    client._handle_message(
        '{"channel": "book.order", "data": {"broker_order_id": "123456"}}'
    )
    assert seen == [("book.order", {"broker_order_id": "123456"})]


def test_book_trade_frame_dispatched_to_on_push() -> None:
    """A book.trade frame is dispatched with its data payload."""
    seen: list = []
    client = _client(seen)
    client._handle_message('{"channel": "book.trade", "data": {"trade_id": "t-1"}}')
    assert seen == [("book.trade", {"trade_id": "t-1"})]


def test_status_frames_go_to_on_status_not_on_push() -> None:
    """Non-book frames (heartbeat/ack) route to on_status, never on_push."""
    seen: list = []
    status: list = []
    client = _client(seen, status)
    client._handle_message('{"channel": "heartbeat"}')
    client._handle_message('{"channel": "subscribed", "accounts": ["x"]}')
    assert seen == []
    assert status == ["heartbeat", "subscribed"]


def test_malformed_frames_are_ignored() -> None:
    """Non-JSON or non-dict frames are silently dropped."""
    seen: list = []
    client = _client(seen)
    client._handle_message("not json")
    client._handle_message("[1, 2, 3]")
    client._handle_message('{"channel": "book.order"}')  # missing data
    assert seen == []
