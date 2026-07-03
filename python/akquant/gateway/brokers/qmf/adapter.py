"""QMF broker 适配器：实现 TraderGateway 协议，桥接 HTTP/WS 与 Unified 模型."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ...broker_models import (
    BrokerCapability,
    UnifiedAccount,
    UnifiedExecutionReport,
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedPosition,
    UnifiedTrade,
)
from . import mapper
from .client import QMFHttpClient
from .ws import QMFPushClient


class QMFMarketGateway:
    """空转行情网关：行情由 akquant 现有 feed 提供，本 broker 不接柜台行情."""

    def connect(self) -> None:
        """No-op: 无柜台行情连接."""
        return None

    def disconnect(self) -> None:
        """No-op."""
        return None

    def subscribe(self, symbols: Sequence[str]) -> None:
        """No-op：订阅由现有 feed 负责."""
        _ = symbols

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        """No-op."""
        _ = symbols

    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """No-op：无 tick 推送."""
        _ = callback

    def on_bar(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """No-op：无 bar 推送."""
        _ = callback

    def start(self) -> None:
        """No-op."""
        return None


def default_capability() -> BrokerCapability:
    """Phase 1 证券能力矩阵."""
    return BrokerCapability(
        broker_name="qmf",
        position_effect=False,
        supports_short_sell=False,
        position_details=False,
    )


class QMFTraderGateway:
    """通过 chibi_quant 前置机网关交易的 TraderGateway 实现."""

    def __init__(
        self,
        client: QMFHttpClient,
        ws_url: str,
        capability: BrokerCapability | None = None,
    ) -> None:
        """Bind the HTTP client, stream URL and capability matrix."""
        self._client = client
        self._ws_url = ws_url
        self._capability = capability or default_capability()
        self._push: QMFPushClient | None = None
        self._client_id_by_broker: dict[str, str] = {}
        self._on_order: Callable[[UnifiedOrderSnapshot], None] | None = None
        self._on_trade: Callable[[UnifiedTrade], None] | None = None
        self._on_exec: Callable[[UnifiedExecutionReport], None] | None = None

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
            self._client_id_by_broker[broker_order_id] = req.client_order_id
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
                    row, self._client_id_by_broker.get(str(broker_order_id), "")
                )
        return None

    def query_trades(self, since: int | None = None) -> list[UnifiedTrade]:
        """查询成交列表."""
        _ = since
        return [
            mapper.parse_trade(
                row, self._client_id_by_broker.get(str(row.get("entrust_no", "")), "")
            )
            for row in self._client.query_trades()
        ]

    def query_account(self) -> UnifiedAccount | None:
        """查询资金账户."""
        return mapper.parse_account(self._client.query_funds())

    def query_positions(self) -> list[UnifiedPosition]:
        """查询持仓列表."""
        return [mapper.parse_position(row) for row in self._client.query_positions()]

    # --- 回调注册 ---
    def on_order(self, callback: Callable[[UnifiedOrderSnapshot], None]) -> None:
        """注册委托状态回调."""
        self._on_order = callback

    def on_trade(self, callback: Callable[[UnifiedTrade], None]) -> None:
        """注册成交回调."""
        self._on_trade = callback

    def on_execution_report(
        self, callback: Callable[[UnifiedExecutionReport], None]
    ) -> None:
        """注册执行回报回调."""
        self._on_exec = callback

    # --- 断线补齐 ---
    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        """重新拉取当前委托，用于断线补齐."""
        return [
            mapper.parse_order(
                row, self._client_id_by_broker.get(str(row.get("entrust_no", "")), "")
            )
            for row in self._client.query_orders()
        ]

    def sync_today_trades(self) -> list[UnifiedTrade]:
        """重新拉取今日成交，用于断线补齐."""
        return self.query_trades()

    # --- 推送分发 ---
    def _dispatch_push(self, event: str, data: dict[str, Any]) -> None:
        broker_order_id = str(data.get("entrust_no", ""))
        client_order_id = self._client_id_by_broker.get(broker_order_id, "")
        if event == "trade_update" and self._on_trade is not None:
            self._on_trade(mapper.parse_trade(data, client_order_id))
        elif event == "order_update":
            snapshot = mapper.parse_order(data, client_order_id)
            if self._on_order is not None:
                self._on_order(snapshot)
            if self._on_exec is not None:
                self._on_exec(
                    UnifiedExecutionReport(
                        broker_order_id=snapshot.broker_order_id,
                        client_order_id=snapshot.client_order_id,
                        status=snapshot.status,
                        symbol=snapshot.symbol,
                        filled_quantity=snapshot.filled_quantity,
                        avg_fill_price=snapshot.avg_fill_price,
                        reject_reason=snapshot.reject_reason,
                    )
                )
