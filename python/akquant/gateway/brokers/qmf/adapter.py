"""QMF broker 适配器：证券 + 可选期权双会话，按 asset_type 路由."""

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

_OPTION_EXTRA_FIELDS = ("entrust_oc", "covered_flag", "entrust_prop")


def default_capability(enable_options: bool = False) -> BrokerCapability:
    """QMF 能力矩阵；启用期权时声明期权 extra 字段与 features."""
    return BrokerCapability(
        broker_name="qmf",
        position_effect=False,
        supports_short_sell=False,
        position_details=False,
        broker_extra_fields=_OPTION_EXTRA_FIELDS if enable_options else (),
        features=frozenset({"options"}) if enable_options else frozenset(),
    )


class QMFTraderGateway(TraderGatewayBase):
    """通过 chibi_quant 前置机网关交易（证券 + 可选期权）."""

    def __init__(
        self,
        client: QMFHttpClient,
        ws_url: str,
        capability: BrokerCapability | None = None,
        option_client: QMFHttpClient | None = None,
    ) -> None:
        """Bind securities client, optional option client, stream URL, capability."""
        super().__init__()
        self._client = client
        self._option_client = option_client
        self._ws_url = ws_url
        self._capability = capability or default_capability(option_client is not None)
        self._push: QMFPushClient | None = None
        self._option_broker_ids: set[str] = set()

    # --- 生命周期 ---
    def connect(self) -> None:
        """登录证券会话；启用期权时同时登录期权会话（fail-fast）."""
        self._client.login()
        if self._option_client is not None:
            self._option_client.login()

    def disconnect(self) -> None:
        """停止推送并关闭 HTTP 连接."""
        if self._push is not None:
            self._push.stop()
        self._client.close()
        if self._option_client is not None:
            self._option_client.close()

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
        """会话保活（证券会话）."""
        return self._client.auth_status(keepalive=True)

    # --- 下单/撤单 ---
    def place_order(self, req: UnifiedOrderRequest) -> str:
        """按 asset_type 路由下单，返回 entrust_no 作为 broker_order_id."""
        if req.asset_type == "option":
            if self._option_client is None:
                raise RuntimeError("期权交易需 enable_options 并配置期权会话")
            data = self._option_client.place_option_order(
                mapper.build_option_order_payload(req)
            )
            broker_order_id = str(data.get("entrust_no", ""))
            if broker_order_id:
                self.record_broker_order(broker_order_id, req.client_order_id)
                self._option_broker_ids.add(broker_order_id)
            return broker_order_id
        data = self._client.place_order(mapper.build_order_payload(req))
        broker_order_id = str(data.get("entrust_no", ""))
        if broker_order_id:
            self.record_broker_order(broker_order_id, req.client_order_id)
        return broker_order_id

    def cancel_order(self, broker_order_id: str) -> None:
        """按来源路由撤单（期权/证券）."""
        if (
            broker_order_id in self._option_broker_ids
            and self._option_client is not None
        ):
            self._option_client.cancel_option_order(broker_order_id)
        else:
            self._client.cancel_order(broker_order_id)

    # --- 查询（证券 + 可选期权合并）---
    def query_order(self, broker_order_id: str) -> UnifiedOrderSnapshot | None:
        """按 broker_order_id 查询单笔委托快照（先证券后期权）."""
        target = str(broker_order_id)
        for row in self._client.query_orders():
            if str(row.get("entrust_no", "")) == target:
                return mapper.parse_order(row, self.client_order_id_for(target))
        if self._option_client is not None:
            for row in self._option_client.query_option_orders():
                if str(row.get("entrust_no", "")) == target:
                    return mapper.parse_option_order(
                        row, self.client_order_id_for(target)
                    )
        return None

    def query_trades(self, since: int | None = None) -> list[UnifiedTrade]:
        """查询成交列表（证券 + 期权）."""
        _ = since
        trades = [
            mapper.parse_trade(
                row, self.client_order_id_for(str(row.get("entrust_no", "")))
            )
            for row in self._client.query_trades()
        ]
        if self._option_client is not None:
            trades.extend(
                mapper.parse_option_trade(
                    row, self.client_order_id_for(str(row.get("entrust_no", "")))
                )
                for row in self._option_client.query_option_trades()
            )
        return trades

    def query_account(self) -> UnifiedAccount | None:
        """查询资金账户（证券；期权资产本阶段不并入）."""
        return mapper.parse_account(self._client.query_funds())

    def query_positions(self) -> list[UnifiedPosition]:
        """查询持仓列表（证券 + 期权）."""
        positions = [
            mapper.parse_position(row) for row in self._client.query_positions()
        ]
        if self._option_client is not None:
            positions.extend(
                mapper.parse_option_position(row)
                for row in self._option_client.query_option_positions()
            )
        return positions

    # --- 断线补齐 ---
    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        """重新拉取当前委托（证券 + 期权），用于断线补齐."""
        snapshots = [
            mapper.parse_order(
                row, self.client_order_id_for(str(row.get("entrust_no", "")))
            )
            for row in self._client.query_orders()
        ]
        if self._option_client is not None:
            snapshots.extend(
                mapper.parse_option_order(
                    row, self.client_order_id_for(str(row.get("entrust_no", "")))
                )
                for row in self._option_client.query_option_orders()
            )
        return snapshots

    def sync_today_trades(self) -> list[UnifiedTrade]:
        """重新拉取今日成交（证券 + 期权），用于断线补齐."""
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
