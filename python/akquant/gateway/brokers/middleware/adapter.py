"""中间件(TradeTools2.0) broker 适配器：实现 akquant TraderGateway 协议."""

from __future__ import annotations

from typing import Any

from ...broker_models import (
    BrokerCapability,
    UnifiedAccount,
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedPosition,
    UnifiedTrade,
)
from ...trader_base import TraderGatewayBase
from . import mapper
from .client import MiddlewareHttpClient
from .ws import MiddlewarePushClient


def default_capability(enable_options: bool = False) -> BrokerCapability:
    """中间件能力矩阵；position_effect 由标准 offset 承载."""
    return BrokerCapability(
        broker_name="middleware",
        position_effect=True,
        supports_short_sell=False,
        position_details=False,
        broker_extra_fields=(),
        features=frozenset({"options"}) if enable_options else frozenset(),
    )


class MiddlewareTraderGateway(TraderGatewayBase):
    """通过 TradeTools2.0 标准中间件交易（HTTP + WS 推送）."""

    def __init__(
        self,
        client: MiddlewareHttpClient,
        ws_url: str,
        capability: BrokerCapability | None = None,
        enable_options: bool = False,
    ) -> None:
        """Bind the HTTP client, stream URL and capability matrix."""
        super().__init__()
        self._client = client
        self._ws_url = ws_url
        self._capability = capability or default_capability(enable_options)
        self._push: MiddlewarePushClient | None = None

    # --- 生命周期 ---
    def connect(self) -> None:
        """登录中间件会话（存 account_id）."""
        self._client.login()

    def disconnect(self) -> None:
        """停止推送并关闭 HTTP 连接."""
        if self._push is not None:
            self._push.stop()
        self._client.close()

    def start(self) -> None:
        """建立推送长连并把 book.* 帧分发到回调."""
        self._push = MiddlewarePushClient(
            ws_url=self._ws_url,
            token=self._token(),
            on_push=self._dispatch_push,
        )
        self._push.start()

    def _token(self) -> str:
        cfg = getattr(self._client, "_cfg", None)
        return str(getattr(cfg, "token", "")) if cfg is not None else ""

    def get_capabilities(self) -> BrokerCapability:
        """返回能力矩阵."""
        return self._capability

    def heartbeat(self) -> bool:
        """会话保活：中间件会话是否在线."""
        return self._client.session_online()

    # --- 下单/撤单 ---
    def place_order(self, req: UnifiedOrderRequest) -> str:
        """下单，返回 broker_order_id 并登记 id 反查."""
        data = self._client.place_order(mapper.build_order_body(req))
        broker_order_id = str(data.get("broker_order_id") or data.get("order_id", ""))
        if broker_order_id:
            self.record_broker_order(broker_order_id, req.client_order_id)
        return broker_order_id

    def cancel_order(self, broker_order_id: str) -> None:
        """撤单（按 broker_order_id）."""
        self._client.cancel_order({"broker_order_id": str(broker_order_id)})

    # --- 查询 ---
    def query_order(self, broker_order_id: str) -> UnifiedOrderSnapshot | None:
        """按 broker_order_id 查询单笔委托快照."""
        target = str(broker_order_id)
        for row in self._client.query_orders():
            row_id = str(row.get("broker_order_id") or row.get("order_id", ""))
            if row_id == target:
                return mapper.parse_order(row, self.client_order_id_for(target))
        return None

    def query_trades(self, since: int | None = None) -> list[UnifiedTrade]:
        """查询成交列表."""
        _ = since
        return [
            mapper.parse_trade(
                row,
                self.client_order_id_for(
                    str(row.get("broker_order_id") or row.get("order_id", ""))
                ),
            )
            for row in self._client.query_trades()
        ]

    def query_account(self) -> UnifiedAccount | None:
        """查询资金账户（/summary）."""
        return mapper.parse_account(self._client.query_summary())

    def query_positions(self) -> list[UnifiedPosition]:
        """查询持仓列表."""
        return [mapper.parse_position(row) for row in self._client.query_positions()]

    # --- 断线补齐 ---
    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        """重新拉取当前委托（断线补齐）."""
        return [
            mapper.parse_order(
                row,
                self.client_order_id_for(
                    str(row.get("broker_order_id") or row.get("order_id", ""))
                ),
            )
            for row in self._client.query_orders()
        ]

    def sync_today_trades(self) -> list[UnifiedTrade]:
        """重新拉取今日成交（断线补齐）."""
        return self.query_trades()

    # --- 推送分发 ---
    def _dispatch_push(self, channel: str, data: dict[str, Any]) -> None:
        broker_order_id = str(data.get("broker_order_id") or data.get("order_id", ""))
        client_order_id = self.client_order_id_for(broker_order_id)
        if channel == "book.trade":
            self._emit_trade(mapper.parse_trade(data, client_order_id))
        elif channel == "book.order":
            snapshot = mapper.parse_order(data, client_order_id)
            self._emit_order(snapshot)
            self._emit_exec_from_order(snapshot)
