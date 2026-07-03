"""QMF 前置机网关 HTTP 客户端（chibi_quant /api/v1）."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .crypto import encrypt_password


class QMFApiError(RuntimeError):
    """chibi_quant 业务错误（result != "0"）."""

    def __init__(self, result: str, msg: str) -> None:
        """Record the counter result code and message."""
        super().__init__(f"[{result}] {msg}")
        self.result = result
        self.msg = msg


@dataclass
class QMFClientConfig:
    """连接与登录参数."""

    base_url: str
    qmf_user_id: str
    account_content: str
    password: str
    input_content: str
    content_type: str
    password_key: str
    password_type: str = "2"
    asset_prop: str = "0"
    timeout: float = 10.0


class QMFHttpClient:
    """薄 HTTP 客户端：会话/下单/撤单/查询，处理统一信封."""

    def __init__(
        self, config: QMFClientConfig, transport: httpx.BaseTransport | None = None
    ) -> None:
        """Build the underlying httpx client (optional transport for tests)."""
        self._config = config
        self._http = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout,
            transport=transport,
        )
        self.token: str = ""
        self.fund_account: str = ""

    def _post(self, path: str, payload: dict[str, Any], auth: bool = True) -> Any:
        headers = {}
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = self._http.post(path, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        if str(body.get("result")) != "0":
            raise QMFApiError(str(body.get("result")), str(body.get("msg", "")))
        return body.get("data")

    def login(self) -> dict[str, Any]:
        """登录并保存 gateway token 与 fund_account."""
        payload = {
            "qmf_user_id": self._config.qmf_user_id,
            "account_content": self._config.account_content,
            "password": encrypt_password(
                self._config.password, self._config.password_key
            ),
            "input_content": self._config.input_content,
            "content_type": self._config.content_type,
            "password_type": self._config.password_type,
            "asset_prop": self._config.asset_prop,
        }
        data = self._post("/api/v1/auth/login", payload, auth=False)
        self.token = str(data.get("user_token", ""))
        self.fund_account = str(data.get("fund_account", ""))
        return data

    def auth_status(self, keepalive: bool = True) -> bool:
        """会话保活/状态查询；失败返回 False."""
        try:
            self._post("/api/v1/auth/status", {"keepalive": keepalive})
            return True
        except (QMFApiError, httpx.HTTPError):
            return False

    def place_order(self, order_fields: dict[str, Any]) -> dict[str, Any]:
        """下单（自动注入 fund_account），返回 data."""
        payload = {"fund_account": self.fund_account, **order_fields}
        return self._post("/api/v1/trading/order", payload)

    def cancel_order(
        self, entrust_no: str, exchange_type: str | None = None
    ) -> dict[str, Any]:
        """撤单."""
        payload: dict[str, Any] = {
            "fund_account": self.fund_account,
            "entrust_no": entrust_no,
        }
        if exchange_type is not None:
            payload["exchange_type"] = exchange_type
        return self._post("/api/v1/trading/cancel", payload)

    def query_funds(self) -> dict[str, Any]:
        """查询资金."""
        return self._post("/api/v1/account/funds", {"fund_account": self.fund_account})

    def query_positions(self) -> list[dict[str, Any]]:
        """查询持仓（列表）."""
        return list(
            self._post("/api/v1/account/positions", {"fund_account": self.fund_account})
        )

    def query_orders(self) -> list[dict[str, Any]]:
        """查询委托（列表）."""
        return list(
            self._post("/api/v1/account/orders", {"fund_account": self.fund_account})
        )

    def query_trades(self) -> list[dict[str, Any]]:
        """查询成交（列表）."""
        return list(
            self._post("/api/v1/account/trades", {"fund_account": self.fund_account})
        )

    def place_option_order(self, order_fields: dict[str, Any]) -> dict[str, Any]:
        """期权下单（自动注入 fund_account），返回 data."""
        payload = {"fund_account": self.fund_account, **order_fields}
        return self._post("/api/v1/option/order", payload)

    def cancel_option_order(
        self, entrust_no: str, exchange_type: str | None = None
    ) -> dict[str, Any]:
        """期权撤单."""
        payload: dict[str, Any] = {
            "fund_account": self.fund_account,
            "entrust_no": entrust_no,
        }
        if exchange_type is not None:
            payload["exchange_type"] = exchange_type
        return self._post("/api/v1/option/cancel", payload)

    def query_option_orders(self) -> list[dict[str, Any]]:
        """查询期权委托（列表）."""
        return list(
            self._post("/api/v1/option/orders", {"fund_account": self.fund_account})
        )

    def query_option_trades(self) -> list[dict[str, Any]]:
        """查询期权成交（列表）."""
        return list(
            self._post("/api/v1/option/trades", {"fund_account": self.fund_account})
        )

    def query_option_positions(self) -> list[dict[str, Any]]:
        """查询期权持仓（列表）."""
        return list(
            self._post("/api/v1/option/positions", {"fund_account": self.fund_account})
        )

    def query_option_assets(self, money_type: str = "0") -> dict[str, Any]:
        """查询期权资产."""
        return self._post(
            "/api/v1/option/assets",
            {"fund_account": self.fund_account, "money_type": money_type},
        )

    def query_settlements(
        self, start_date: str, end_date: str, stock_type: str | None = None
    ) -> list[dict[str, Any]]:
        """查询证券交割单（列表）."""
        payload: dict[str, Any] = {
            "fund_account": self.fund_account,
            "start_date": start_date,
            "end_date": end_date,
        }
        if stock_type is not None:
            payload["stock_type"] = stock_type
        return list(self._post("/api/v1/account/settlements", payload))

    def query_fund_flow(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> list[dict[str, Any]]:
        """查询证券资金流水（列表；日期可选）."""
        payload: dict[str, Any] = {"fund_account": self.fund_account}
        if start_date is not None:
            payload["start_date"] = start_date
        if end_date is not None:
            payload["end_date"] = end_date
        return list(self._post("/api/v1/account/fund-flow", payload))

    def query_option_history_orders(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """查询期权历史委托（列表）."""
        return list(
            self._post(
                "/api/v1/option/history-orders",
                {
                    "fund_account": self.fund_account,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        )

    def query_option_history_trades(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """查询期权历史成交（列表）."""
        return list(
            self._post(
                "/api/v1/option/history-trades",
                {
                    "fund_account": self.fund_account,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        )

    def query_option_history_settlements(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """查询期权历史交割单（列表）."""
        return list(
            self._post(
                "/api/v1/option/history-settlements",
                {
                    "fund_account": self.fund_account,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        )

    def close(self) -> None:
        """关闭底层连接."""
        self._http.close()
