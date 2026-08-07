"""HTTP webhook 信号源(标准库实现, 零额外依赖).

**为何不用 FastAPI**: 这个端点能触发真实下单, 最需要在 CI 里被真实测到 —— 起服务、
发请求、断言下单。用可选依赖实现就只能靠 mock 覆盖。标准库 ``http.server`` 对
"接一个 webhook" 这个单一职责完全够用, 且让安全逻辑始终可验证。

高吞吐/复杂路由场景请改用 :class:`RedisSignalSource` + 独立的 Web 进程
(见 signal-ingestion-rfc.md 4.5 的部署建议)。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, List, Optional

from ...log import build_log_extra, get_logger
from ..models import Signal, SignalResult, SignalStatus
from ..protocols import SignalSourceBase
from ..security import AuthError, TokenAuth

logger = get_logger("signal.source.http")

LOCALHOST = "127.0.0.1"
DEFAULT_PATH = "/signal"


class HttpSignalSource(SignalSourceBase):
    """监听 HTTP POST 接收信号.

    安全默认(硬约束, 见 :mod:`akquant.signal.security`):

    - ``token`` 必填, 空值直接 ``ValueError``;
    - ``host`` 默认 ``127.0.0.1``, 绑非本机需显式 ``allow_remote=True`` 并打警告;
    - 传 ``secret`` 则额外要求 HMAC 签名 + 时间戳窗口(跨主机必开)。

    AKQuant **不承诺传输层安全**: HTTPS 与公网暴露请在反向代理层解决。
    """

    def __init__(
        self,
        *,
        token: str,
        host: str = LOCALHOST,
        port: int = 8765,
        path: str = DEFAULT_PATH,
        secret: Optional[str] = None,
        allow_remote: bool = False,
    ) -> None:
        """校验安全参数并准备服务器(不立即监听)."""
        super().__init__()
        self._auth = TokenAuth(token=token, secret=secret)
        if host != LOCALHOST and not allow_remote:
            raise ValueError(
                f"host={host!r} 会把下单端点暴露到本机之外。确认已在反向代理层"
                "做好 TLS 与访问控制后, 显式传 allow_remote=True"
            )
        if host != LOCALHOST:
            logger.warning(
                "信号端点绑定 %s(非本机): 请确认已在反向代理层配置 TLS 与访问控制,"
                " AKQuant 不承诺传输层安全",
                host,
                extra=build_log_extra(phase="signal"),
            )
        self._host = host
        self._port = port
        self._path = path if path.startswith("/") else f"/{path}"
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self.results: List[SignalResult] = []

    @property
    def bound_port(self) -> int:
        """实际监听端口(传 port=0 时用于取系统分配的端口)."""
        if self._server is None:
            return self._port
        return int(self._server.server_address[1])

    def start(self) -> None:
        """起 HTTP 服务线程, 确认就绪后返回(见 SignalSource 协议说明)."""
        if self._server is not None:
            return
        self._server = ThreadingHTTPServer(
            (self._host, self._port), self._make_handler()
        )
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._serve, name="akquant-signal-http", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            logger.warning(
                "信号 HTTP 服务 5 秒内未就绪", extra=build_log_extra(phase="signal")
            )
        logger.info(
            "Signal HTTP endpoint listening on http://%s:%s%s",
            self._host,
            self.bound_port,
            self._path,
            extra=build_log_extra(phase="signal"),
        )

    def stop(self) -> None:
        """关闭服务(幂等)."""
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
        self._ready.clear()

    def on_result(self, result: SignalResult) -> None:
        """收集回执(便于调用方查询/测试断言)."""
        self.results.append(result)

    def _serve(self) -> None:
        """服务循环."""
        self._ready.set()
        server = self._server
        if server is not None:
            server.serve_forever(poll_interval=0.1)

    def _handle_payload(self, raw: bytes) -> tuple[int, dict[str, Any]]:
        """解析并派发一条信号, 返回 (HTTP 状态码, 响应体)."""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return 400, {"error": f"invalid json: {exc}"}
        if not isinstance(payload, dict):
            return 400, {"error": "payload must be a JSON object"}
        try:
            signal = Signal(**payload)
        except Exception as exc:  # noqa: BLE001 — pydantic 校验失败即 400
            return 400, {"error": f"invalid signal: {exc}"}

        result = self.dispatch(signal)
        # 幂等重复不是错误: 平台重推本就该拿到 200 + duplicate, 否则会一直重试。
        status = 200 if result.status is not SignalStatus.ERROR else 500
        return status, result.as_dict()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        """造绑定到本实例的 handler 类."""
        source = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler 约定
                """处理信号投递."""
                if self.path.split("?")[0] != source._path:
                    self._reply(404, {"error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    self._reply(400, {"error": "bad content-length"})
                    return
                body = self.rfile.read(length) if length > 0 else b""
                try:
                    source._auth.verify(
                        authorization=self.headers.get("Authorization"),
                        body=body,
                        timestamp=self.headers.get("X-Signal-Timestamp"),
                        signature=self.headers.get("X-Signal-Signature"),
                    )
                except AuthError:
                    # 不回具体原因: 避免为攻击者做区分预言机。
                    logger.warning(
                        "信号请求鉴权失败(来源 %s)",
                        self.client_address[0],
                        extra=build_log_extra(phase="signal"),
                    )
                    self._reply(401, {"error": "unauthorized"})
                    return
                status, payload = source._handle_payload(body)
                self._reply(status, payload)

            def _reply(self, status: int, payload: dict[str, Any]) -> None:
                """回 JSON."""
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                """把 access log 接进 akquant 日志(默认会打到 stderr)."""
                logger.debug(
                    "signal http: " + format,
                    *args,
                    extra=build_log_extra(phase="signal"),
                )

        return _Handler
