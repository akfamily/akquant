"""QMF 前置机网关 WebSocket 推送客户端（/api/v1/stream）."""

from __future__ import annotations

import json
import threading
from typing import Any, Callable

import websocket

_STATUS_FRAMES = {"ready", "heartbeat", "ack", "error", "pong"}


class QMFPushClient:
    """订阅推送帧并分发；断线自动重连."""

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
        frame_type = str(frame.get("type", ""))
        if frame_type == "push":
            event = str(frame.get("event", ""))
            data = frame.get("data") or {}
            if isinstance(data, dict):
                self._on_push(event, data)
        elif frame_type in _STATUS_FRAMES and self._on_status is not None:
            self._on_status(frame_type, frame)

    def start(self) -> None:
        """Open the stream on a daemon thread with auto-reconnect."""
        header = [f"Authorization: Bearer {self._token}"]
        self._app = websocket.WebSocketApp(
            self._ws_url,
            header=header,
            on_message=lambda _ws, msg: self._handle_message(msg),
        )
        self._thread = threading.Thread(
            target=lambda: self._app.run_forever(reconnect=5),
            daemon=True,
            name="qmf-push",
        )
        self._thread.start()

    def stop(self) -> None:
        """Close the stream and drop the worker thread."""
        if self._app is not None:
            self._app.close()
        self._app = None
        self._thread = None
