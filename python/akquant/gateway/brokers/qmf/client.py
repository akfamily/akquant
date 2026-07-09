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

    def query_option_exercise_assignments(self) -> list[dict[str, Any]]:
        """查询期权行权指派（列表）."""
        return list(
            self._post(
                "/api/v1/option/exercise-assignments",
                {"fund_account": self.fund_account},
            )
        )

    def query_option_exercise_settlements(self) -> list[dict[str, Any]]:
        """查询期权行权交割（列表）."""
        return list(
            self._post(
                "/api/v1/option/exercise-settlements",
                {"fund_account": self.fund_account},
            )
        )

    def query_option_exercise_debts(self) -> list[dict[str, Any]]:
        """查询期权行权负债（列表）."""
        return list(
            self._post(
                "/api/v1/option/exercise-debts",
                {"fund_account": self.fund_account},
            )
        )

    def query_option_history_exercise_assignments(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """查询期权历史行权指派（列表）."""
        return list(
            self._post(
                "/api/v1/option/history-exercise-assignments",
                {
                    "fund_account": self.fund_account,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        )

    def query_option_history_exercise_settlements(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """查询期权历史行权交割（列表）."""
        return list(
            self._post(
                "/api/v1/option/history-exercise-settlements",
                {
                    "fund_account": self.fund_account,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        )

    def query_option_covered_shortages(self) -> list[dict[str, Any]]:
        """查询期权备兑不足（列表）."""
        return list(
            self._post(
                "/api/v1/option/covered-shortages",
                {"fund_account": self.fund_account},
            )
        )

    def query_option_covered_transferable(
        self, exchange_type: str, lock_direction: str, stock_code: str | None = None
    ) -> list[dict[str, Any]]:
        """查询期权可划转备兑证券（列表）."""
        payload: dict[str, Any] = {
            "fund_account": self.fund_account,
            "exchange_type": exchange_type,
            "lock_direction": lock_direction,
        }
        if stock_code is not None:
            payload["stock_code"] = stock_code
        return list(self._post("/api/v1/option/covered-transferable", payload))

    def covered_transfer(
        self,
        exchange_type: str,
        stock_code: str,
        entrust_amount: str,
        lock_direction: str,
    ) -> dict[str, Any]:
        """备兑证券划转（写；lock_direction 1=锁定 2=解锁），返回 data."""
        return self._post(
            "/api/v1/option/covered-transfer",
            {
                "fund_account": self.fund_account,
                "exchange_type": exchange_type,
                "stock_code": stock_code,
                "entrust_amount": entrust_amount,
                "lock_direction": lock_direction,
            },
        )

    def query_option_contracts(
        self, stock_code: str | None = None, option_code: str | None = None
    ) -> list[dict[str, Any]]:
        """查询期权合约（列表）."""
        payload: dict[str, Any] = {"fund_account": self.fund_account}
        if stock_code is not None:
            payload["stock_code"] = stock_code
        if option_code is not None:
            payload["option_code"] = option_code
        return list(self._post("/api/v1/option/contracts", payload))

    def query_option_underlyings(
        self, stock_code: str | None = None
    ) -> list[dict[str, Any]]:
        """查询期权标的证券（列表）."""
        payload: dict[str, Any] = {"fund_account": self.fund_account}
        if stock_code is not None:
            payload["stock_code"] = stock_code
        return list(self._post("/api/v1/option/underlyings", payload))

    def query_option_strategies(
        self, optcomb_code: str | None = None
    ) -> list[dict[str, Any]]:
        """查询期权组合策略定义（列表）."""
        payload: dict[str, Any] = {"fund_account": self.fund_account}
        if optcomb_code is not None:
            payload["optcomb_code"] = optcomb_code
        return list(self._post("/api/v1/option/strategies", payload))

    def query_option_position_limits(
        self, stock_code: str | None = None
    ) -> list[dict[str, Any]]:
        """查询期权持仓限额（列表）."""
        payload: dict[str, Any] = {"fund_account": self.fund_account}
        if stock_code is not None:
            payload["stock_code"] = stock_code
        return list(self._post("/api/v1/option/position-limits", payload))

    def query_option_contract_tips(self, money_type: str = "0") -> list[dict[str, Any]]:
        """查询期权合约提示（列表）."""
        return list(
            self._post(
                "/api/v1/option/contract-tips",
                {"fund_account": self.fund_account, "money_type": money_type},
            )
        )

    def query_option_enable_amount(
        self,
        exchange_type: str,
        option_code: str,
        opt_entrust_price: str,
        entrust_prop: str,
        entrust_bs: str,
        entrust_oc: str,
        covered_flag: str | None = None,
    ) -> dict[str, Any]:
        """查询期权可委托数量（下单前额度计算），返回 data."""
        payload: dict[str, Any] = {
            "fund_account": self.fund_account,
            "exchange_type": exchange_type,
            "option_code": option_code,
            "opt_entrust_price": opt_entrust_price,
            "entrust_prop": entrust_prop,
            "entrust_bs": entrust_bs,
            "entrust_oc": entrust_oc,
        }
        if covered_flag is not None:
            payload["covered_flag"] = covered_flag
        return self._post("/api/v1/option/enable-amount", payload)

    def query_option_underlying_amount_tip(
        self,
        exchange_type: str,
        option_code: str,
        entrust_amount: str,
        entrust_bs: str,
        entrust_oc: str,
    ) -> dict[str, Any]:
        """查询期权标的持仓数量提示（下单前提示），返回 data."""
        return self._post(
            "/api/v1/option/underlying-amount-tip",
            {
                "fund_account": self.fund_account,
                "exchange_type": exchange_type,
                "option_code": option_code,
                "entrust_amount": entrust_amount,
                "entrust_bs": entrust_bs,
                "entrust_oc": entrust_oc,
            },
        )

    def place_convertible_bond_order(
        self,
        stock_code: str,
        exchange_type: str,
        entrust_prop: str,
        entrust_amount: str,
        stock_account: str | None = None,
        stb_stock_property: str | None = None,
    ) -> dict[str, Any]:
        """可转债下单（写；自动注入 fund_account），返回 data."""
        payload: dict[str, Any] = {
            "fund_account": self.fund_account,
            "stock_code": stock_code,
            "exchange_type": exchange_type,
            "entrust_prop": entrust_prop,
            "entrust_amount": entrust_amount,
        }
        if stock_account is not None:
            payload["stock_account"] = stock_account
        if stb_stock_property is not None:
            payload["stb_stock_property"] = stb_stock_property
        return self._post("/api/v1/trading/convertible-bond-order", payload)

    def cancel_convertible_bond_order(self, entrust_no: str) -> dict[str, Any]:
        """可转债撤单（写），返回 data."""
        return self._post(
            "/api/v1/trading/convertible-bond-cancel",
            {"fund_account": self.fund_account, "entrust_no": entrust_no},
        )

    def query_convertible_bond_orders(
        self,
        stock_code: str | None = None,
        entrust_no: str | None = None,
        query_flag: str | None = None,
        en_entrust_prop: str | None = None,
    ) -> list[dict[str, Any]]:
        """查询可转债委托（列表）."""
        payload: dict[str, Any] = {"fund_account": self.fund_account}
        if stock_code is not None:
            payload["stock_code"] = stock_code
        if entrust_no is not None:
            payload["entrust_no"] = entrust_no
        if query_flag is not None:
            payload["query_flag"] = query_flag
        if en_entrust_prop is not None:
            payload["en_entrust_prop"] = en_entrust_prop
        return list(self._post("/api/v1/account/convertible-bond-orders", payload))

    def query_bond_putback_info(
        self, stock_code: str | None = None
    ) -> list[dict[str, Any]]:
        """查询可转债回售信息（列表）."""
        payload: dict[str, Any] = {"fund_account": self.fund_account}
        if stock_code is not None:
            payload["stock_code"] = stock_code
        return list(self._post("/api/v1/account/bond-putback-info", payload))

    def close(self) -> None:
        """关闭底层连接."""
        self._http.close()
