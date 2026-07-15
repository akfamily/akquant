"""中间件(TradeTools2.0)标准 API 的 HTTP 客户端（/api/v1）."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx


class MiddlewareApiError(RuntimeError):
    """中间件业务错误（统一信封 success=false）."""

    def __init__(self, code: str, msg: str) -> None:
        """Record the error code and message."""
        super().__init__(f"[{code}] {msg}")
        self.code = code
        self.msg = msg


@dataclass
class MiddlewareClientConfig:
    """连接与登录参数."""

    base_url: str
    broker_id: str
    fund_account: str
    password: str
    account_type: str = "security"
    qmf_user_id: str = ""
    token: str = ""
    timeout: float = 10.0
    extra: dict[str, Any] = field(default_factory=dict)


class MiddlewareHttpClient:
    """薄 HTTP 客户端：会话/下单/撤单/查询，处理统一信封."""

    def __init__(
        self,
        config: MiddlewareClientConfig,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Build the underlying httpx client (optional transport for tests)."""
        self._cfg = config
        self._http = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout,
            transport=transport,
        )
        self.account_id: str = ""

    def _request(
        self, method: str, path: str, json_body: dict[str, Any] | None = None
    ) -> Any:
        headers = {}
        if self._cfg.token:
            headers["Authorization"] = f"Bearer {self._cfg.token}"
        resp = self._http.request(method, path, json=json_body, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        if not body.get("success"):
            raise MiddlewareApiError(
                str(body.get("code", "")), str(body.get("msg", ""))
            )
        return body.get("data")

    def _acct_path(self, tail: str) -> str:
        if not self.account_id:
            raise RuntimeError("未登录：account_id 为空")
        return f"/accounts/{quote(self.account_id, safe='')}/{tail}"

    def login(self) -> str:
        """登录并保存 account_id."""
        data = self._request(
            "POST",
            "/sessions",
            {
                "broker_id": self._cfg.broker_id,
                "fund_account": self._cfg.fund_account,
                "password": self._cfg.password,
                "account_type": self._cfg.account_type,
                "qmf_user_id": self._cfg.qmf_user_id,
                "extra": dict(self._cfg.extra),
            },
        )
        account = data.get("account", {}) if isinstance(data, dict) else {}
        self.account_id = str(account.get("account_id", ""))
        return self.account_id

    def logout(self) -> None:
        """登出（幂等，忽略错误）."""
        if not self.account_id:
            return
        try:
            self._request("DELETE", f"/sessions/{quote(self.account_id, safe='')}")
        except (MiddlewareApiError, httpx.HTTPError):
            pass

    def session_online(self) -> bool:
        """会话是否在线（heartbeat）."""
        try:
            data = self._request("GET", "/sessions")
        except (MiddlewareApiError, httpx.HTTPError):
            return False
        sessions = data.get("sessions", []) if isinstance(data, dict) else []
        for item in sessions:
            if str(item.get("account_id", "")) == self.account_id:
                return str(item.get("status", "")) in ("", "online")
        return False

    def place_order(self, body: dict[str, Any]) -> dict[str, Any]:
        """下单，返回标准 Order data."""
        return self._request("POST", self._acct_path("orders"), body)

    def cancel_order(self, body: dict[str, Any]) -> dict[str, Any]:
        """撤单，返回标准 Order data."""
        return self._request("POST", self._acct_path("cancel"), body)

    def query_positions(self) -> list[dict[str, Any]]:
        """查询持仓（列表）."""
        data = self._request("GET", self._acct_path("positions"))
        return list(data.get("positions", [])) if isinstance(data, dict) else []

    def query_trades(self) -> list[dict[str, Any]]:
        """查询成交（列表）."""
        data = self._request("GET", self._acct_path("trades"))
        return list(data.get("trades", [])) if isinstance(data, dict) else []

    def query_orders(self) -> list[dict[str, Any]]:
        """查询委托（列表）."""
        data = self._request("GET", self._acct_path("orders"))
        return list(data.get("orders", [])) if isinstance(data, dict) else []

    def query_summary(self) -> dict[str, Any]:
        """查询账户汇总（data 直接是 summary 对象）."""
        data = self._request("GET", self._acct_path("summary"))
        return data if isinstance(data, dict) else {}

    def close(self) -> None:
        """关闭底层连接."""
        self._http.close()
