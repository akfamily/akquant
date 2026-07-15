"""中间件(TradeTools2.0)推送客户端（WS /api/v1/ws?accounts=）.

推送帧按 channel 路由：book.order / book.trade 走业务回调，其余（
heartbeat/ack/ready/subscribed/error/pong）走状态回调。
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable

import websocket

_PUSH_CHANNELS = {"book.order", "book.trade"}


class MiddlewarePushClient:
    """订阅中间件推送并按 channel 分发；断线自动重连."""

    def __init__(
        self,
        ws_url: str,
        token: str,
        on_push: Callable[[str, dict[str, Any]], None],
        on_status: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        """Bind the stream URL, token and dispatch callbacks."""
        self._ws_url = ws_url
        self._token = token
        self._on_push = on_push
        self._on_status = on_status
        self._app: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None

    def _handle_message(self, raw: str) -> None:
        try:
            frame = json.loads(raw)
        except (ValueError, TypeError):
            return
        if not isinstance(frame, dict):
            return
        channel = str(frame.get("channel", ""))
        if channel in _PUSH_CHANNELS:
            data = frame.get("data")
            if isinstance(data, dict):
                self._on_push(channel, data)
        elif self._on_status is not None:
            self._on_status(channel, frame)

    def start(self) -> None:
        """Open the stream on a daemon thread with auto-reconnect."""
        header = [f"Authorization: Bearer {self._token}"] if self._token else []
        self._app = websocket.WebSocketApp(
            self._ws_url,
            header=header,
            on_message=lambda _ws, msg: self._handle_message(msg),
        )
        self._thread = threading.Thread(
            target=lambda: self._app.run_forever(reconnect=5),
            daemon=True,
            name="middleware-push",
        )
        self._thread.start()

    def stop(self) -> None:
        """Close the stream and drop the worker thread."""
        if self._app is not None:
            self._app.close()
        self._app = None
        self._thread = None
