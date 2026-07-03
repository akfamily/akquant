"""QMF broker 适配器：实现 TraderGateway 协议，桥接 HTTP/WS 与 Unified 模型."""

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
from .client import QMFHttpClient
from .ws import QMFPushClient


def default_capability() -> BrokerCapability:
    """Phase 1 证券能力矩阵."""
    return BrokerCapability(
        broker_name="qmf",
        position_effect=False,
        supports_short_sell=False,
        position_details=False,
    )


class QMFTraderGateway(TraderGatewayBase):
    """通过 chibi_quant 前置机网关交易的 TraderGateway 实现."""

    def __init__(
        self,
        client: QMFHttpClient,
        ws_url: str,
        capability: BrokerCapability | None = None,
    ) -> None:
        """Bind the HTTP client, stream URL and capability matrix."""
        super().__init__()
        self._client = client
        self._ws_url = ws_url
        self._capability = capability or default_capability()
        self._push: QMFPushClient | None = None

    # --- 生命周期 ---
    def connect(self) -> None:
        """登录柜台并获取会话 token."""
        self._client.login()

    def disconnect(self) -> None:
        """停止推送并关闭 HTTP 连接."""
        if self._push is not None:
            self._push.stop()
        self._client.close()

    def start(self) -> None:
        """建立推送长连并把 push 帧分发到回调."""
        self._push = QMFPushClient(
            ws_url=self._ws_url,
            token=self._client.token,
            on_push=self._dispatch_push,
        )
        self._push.start()

    def get_capabilities(self) -> BrokerCapability:
        """返回能力矩阵."""
        return self._capability

    def heartbeat(self) -> bool:
        """会话保活."""
        return self._client.auth_status(keepalive=True)

    # --- 下单/撤单 ---
    def place_order(self, req: UnifiedOrderRequest) -> str:
        """下单，返回柜台 entrust_no 作为 broker_order_id."""
        data = self._client.place_order(mapper.build_order_payload(req))
        broker_order_id = str(data.get("entrust_no", ""))
        if broker_order_id:
            self.record_broker_order(broker_order_id, req.client_order_id)
        return broker_order_id

    def cancel_order(self, broker_order_id: str) -> None:
        """按 entrust_no 撤单."""
        self._client.cancel_order(broker_order_id)

    # --- 查询 ---
    def query_order(self, broker_order_id: str) -> UnifiedOrderSnapshot | None:
        """按 broker_order_id 查询单笔委托快照."""
        for row in self._client.query_orders():
            if str(row.get("entrust_no", "")) == str(broker_order_id):
                return mapper.parse_order(
                    row, self.client_order_id_for(str(broker_order_id))
                )
        return None

    def query_trades(self, since: int | None = None) -> list[UnifiedTrade]:
        """查询成交列表."""
        _ = since
        return [
            mapper.parse_trade(
                row, self.client_order_id_for(str(row.get("entrust_no", "")))
            )
            for row in self._client.query_trades()
        ]

    def query_account(self) -> UnifiedAccount | None:
        """查询资金账户."""
        return mapper.parse_account(self._client.query_funds())

    def query_positions(self) -> list[UnifiedPosition]:
        """查询持仓列表."""
        return [mapper.parse_position(row) for row in self._client.query_positions()]

    # --- 断线补齐 ---
    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        """重新拉取当前委托，用于断线补齐."""
        return [
            mapper.parse_order(
                row, self.client_order_id_for(str(row.get("entrust_no", "")))
            )
            for row in self._client.query_orders()
        ]

    def sync_today_trades(self) -> list[UnifiedTrade]:
        """重新拉取今日成交，用于断线补齐."""
        return self.query_trades()

    # --- 推送分发 ---
    def _dispatch_push(self, event: str, data: dict[str, Any]) -> None:
        broker_order_id = str(data.get("entrust_no", ""))
        client_order_id = self.client_order_id_for(broker_order_id)
        if event == "trade_update":
            self._emit_trade(mapper.parse_trade(data, client_order_id))
        elif event == "order_update":
            snapshot = mapper.parse_order(data, client_order_id)
            self._emit_order(snapshot)
            self._emit_exec_from_order(snapshot)
