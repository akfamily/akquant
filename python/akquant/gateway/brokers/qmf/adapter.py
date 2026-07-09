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
        self._option_push: QMFPushClient | None = None
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
        if self._option_push is not None:
            self._option_push.stop()
        self._client.close()
        if self._option_client is not None:
            self._option_client.close()

    def start(self) -> None:
        """建立推送长连并把 push 帧分发到回调（启用期权时含第二路期权 WS）."""
        self._push = QMFPushClient(
            ws_url=self._ws_url,
            token=self._client.token,
            on_push=self._dispatch_push,
        )
        self._push.start()
        if self._option_client is not None:
            self._option_push = QMFPushClient(
                ws_url=self._ws_url,
                token=self._option_client.token,
                on_push=self._dispatch_option_push,
            )
            self._option_push.start()

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
    def _option_order_rows(self) -> list[dict[str, Any]]:
        """拉取期权委托行，并登记其 entrust_no 以便（含断线补齐后）撤单判源."""
        if self._option_client is None:
            return []
        rows = self._option_client.query_option_orders()
        for row in rows:
            entrust_no = str(row.get("entrust_no", ""))
            if entrust_no:
                self._option_broker_ids.add(entrust_no)
        return rows

    def query_order(self, broker_order_id: str) -> UnifiedOrderSnapshot | None:
        """按 broker_order_id 查询单笔委托快照（先证券后期权）."""
        target = str(broker_order_id)
        for row in self._client.query_orders():
            if str(row.get("entrust_no", "")) == target:
                return mapper.parse_order(row, self.client_order_id_for(target))
        for row in self._option_order_rows():
            if str(row.get("entrust_no", "")) == target:
                return mapper.parse_option_order(row, self.client_order_id_for(target))
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
        """查询资金账户（启用期权时合并证券 + 期权资产）."""
        account = mapper.parse_account(self._client.query_funds())
        if self._option_client is not None:
            account = mapper.merge_option_assets(
                account, self._option_client.query_option_assets()
            )
        return account

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

    # --- 只读扩展查询（非协议；原始行透传）---
    def query_settlements(
        self, start_date: str, end_date: str, stock_type: str | None = None
    ) -> list[dict[str, Any]]:
        """证券交割单查询（原始行）."""
        return self._client.query_settlements(start_date, end_date, stock_type)

    def query_fund_flow(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> list[dict[str, Any]]:
        """证券资金流水查询（原始行）."""
        return self._client.query_fund_flow(start_date, end_date)

    def query_option_history_orders(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """期权历史委托查询（原始行）."""
        return self._require_option_client().query_option_history_orders(
            start_date, end_date
        )

    def query_option_history_trades(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """期权历史成交查询（原始行）."""
        return self._require_option_client().query_option_history_trades(
            start_date, end_date
        )

    def query_option_history_settlements(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """期权历史交割单查询（原始行）."""
        return self._require_option_client().query_option_history_settlements(
            start_date, end_date
        )

    def query_option_exercise_assignments(self) -> list[dict[str, Any]]:
        """期权行权指派查询（原始行）."""
        return self._require_option_client().query_option_exercise_assignments()

    def query_option_exercise_settlements(self) -> list[dict[str, Any]]:
        """期权行权交割查询（原始行）."""
        return self._require_option_client().query_option_exercise_settlements()

    def query_option_exercise_debts(self) -> list[dict[str, Any]]:
        """期权行权负债查询（原始行）."""
        return self._require_option_client().query_option_exercise_debts()

    def query_option_history_exercise_assignments(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """期权历史行权指派查询（原始行）."""
        return self._require_option_client().query_option_history_exercise_assignments(
            start_date, end_date
        )

    def query_option_history_exercise_settlements(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """期权历史行权交割查询（原始行）."""
        return self._require_option_client().query_option_history_exercise_settlements(
            start_date, end_date
        )

    def query_option_covered_shortages(self) -> list[dict[str, Any]]:
        """期权备兑不足查询（原始行）."""
        return self._require_option_client().query_option_covered_shortages()

    def query_option_covered_transferable(
        self, exchange_type: str, lock_direction: str, stock_code: str | None = None
    ) -> list[dict[str, Any]]:
        """期权可划转备兑证券查询（原始行）."""
        return self._require_option_client().query_option_covered_transferable(
            exchange_type, lock_direction, stock_code
        )

    def covered_transfer(
        self,
        exchange_type: str,
        stock_code: str,
        entrust_amount: str,
        lock_direction: str,
    ) -> dict[str, Any]:
        """备兑证券划转（写；lock_direction 1=锁定 2=解锁），返回原始 data."""
        return self._require_option_client().covered_transfer(
            exchange_type, stock_code, entrust_amount, lock_direction
        )

    def query_option_contracts(
        self, stock_code: str | None = None, option_code: str | None = None
    ) -> list[dict[str, Any]]:
        """期权合约查询（原始行）."""
        return self._require_option_client().query_option_contracts(
            stock_code, option_code
        )

    def query_option_underlyings(
        self, stock_code: str | None = None
    ) -> list[dict[str, Any]]:
        """期权标的证券查询（原始行）."""
        return self._require_option_client().query_option_underlyings(stock_code)

    def query_option_strategies(
        self, optcomb_code: str | None = None
    ) -> list[dict[str, Any]]:
        """期权组合策略定义查询（原始行）."""
        return self._require_option_client().query_option_strategies(optcomb_code)

    def query_option_position_limits(
        self, stock_code: str | None = None
    ) -> list[dict[str, Any]]:
        """期权持仓限额查询（原始行）."""
        return self._require_option_client().query_option_position_limits(stock_code)

    def query_option_contract_tips(self, money_type: str = "0") -> list[dict[str, Any]]:
        """期权合约提示查询（原始行）."""
        return self._require_option_client().query_option_contract_tips(money_type)

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
        """期权可委托数量查询（下单前额度，原始行）."""
        return self._require_option_client().query_option_enable_amount(
            exchange_type,
            option_code,
            opt_entrust_price,
            entrust_prop,
            entrust_bs,
            entrust_oc,
            covered_flag,
        )

    def query_option_underlying_amount_tip(
        self,
        exchange_type: str,
        option_code: str,
        entrust_amount: str,
        entrust_bs: str,
        entrust_oc: str,
    ) -> dict[str, Any]:
        """期权标的持仓数量提示查询（下单前提示，原始行）."""
        return self._require_option_client().query_option_underlying_amount_tip(
            exchange_type, option_code, entrust_amount, entrust_bs, entrust_oc
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
        """可转债下单（证券侧；原始 data）."""
        return self._client.place_convertible_bond_order(
            stock_code,
            exchange_type,
            entrust_prop,
            entrust_amount,
            stock_account,
            stb_stock_property,
        )

    def cancel_convertible_bond_order(self, entrust_no: str) -> dict[str, Any]:
        """可转债撤单（证券侧；原始 data）."""
        return self._client.cancel_convertible_bond_order(entrust_no)

    def query_convertible_bond_orders(
        self,
        stock_code: str | None = None,
        entrust_no: str | None = None,
        query_flag: str | None = None,
        en_entrust_prop: str | None = None,
    ) -> list[dict[str, Any]]:
        """可转债委托查询（证券侧；原始行）."""
        return self._client.query_convertible_bond_orders(
            stock_code, entrust_no, query_flag, en_entrust_prop
        )

    def query_bond_putback_info(
        self, stock_code: str | None = None
    ) -> list[dict[str, Any]]:
        """可转债回售信息查询（证券侧；原始行）."""
        return self._client.query_bond_putback_info(stock_code)

    def place_option_combo_order(
        self,
        exchange_type: str,
        optcomb_code: str,
        first_option_code: str,
        first_opthold_type: str,
        second_option_code: str,
        second_opthold_type: str,
        entrust_amount: str,
        comb_bs: str,
        optcomb_id: str | None = None,
    ) -> dict[str, Any]:
        """期权组合策略下单（写；两腿；原始 data）."""
        return self._require_option_client().place_option_combo_order(
            exchange_type,
            optcomb_code,
            first_option_code,
            first_opthold_type,
            second_option_code,
            second_opthold_type,
            entrust_amount,
            comb_bs,
            optcomb_id,
        )

    def confirm_option_combo(
        self,
        exchange_type: str,
        optcomb_code: str,
        comb_bs: str,
        first_option_code: str | None = None,
        first_opthold_type: str | None = None,
        second_option_code: str | None = None,
        second_opthold_type: str | None = None,
        optcomb_id: str | None = None,
    ) -> dict[str, Any]:
        """期权组合策略确认（写；原始 data）."""
        return self._require_option_client().confirm_option_combo(
            exchange_type,
            optcomb_code,
            comb_bs,
            first_option_code,
            first_opthold_type,
            second_option_code,
            second_opthold_type,
            optcomb_id,
        )

    def query_option_combo_orders(
        self, optcomb_code: str | None = None, optcomb_id: str | None = None
    ) -> list[dict[str, Any]]:
        """期权组合委托查询（原始行）."""
        return self._require_option_client().query_option_combo_orders(
            optcomb_code, optcomb_id
        )

    def query_option_combo_positions(
        self, optcomb_code: str | None = None, query_mode: str | None = None
    ) -> list[dict[str, Any]]:
        """期权组合持仓查询（原始行）."""
        return self._require_option_client().query_option_combo_positions(
            optcomb_code, query_mode
        )

    def query_option_history_combo_orders(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """期权历史组合委托查询（原始行）."""
        return self._require_option_client().query_option_history_combo_orders(
            start_date, end_date
        )

    def _require_option_client(self) -> QMFHttpClient:
        """返回期权会话客户端；未启用期权时抛清晰错误."""
        if self._option_client is None:
            raise RuntimeError("期权历史查询需 enable_options 并配置期权会话")
        return self._option_client

    # --- 断线补齐 ---
    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        """重新拉取当前委托（证券 + 期权），用于断线补齐."""
        snapshots = [
            mapper.parse_order(
                row, self.client_order_id_for(str(row.get("entrust_no", "")))
            )
            for row in self._client.query_orders()
        ]
        snapshots.extend(
            mapper.parse_option_order(
                row, self.client_order_id_for(str(row.get("entrust_no", "")))
            )
            for row in self._option_order_rows()
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

    def _dispatch_option_push(self, event: str, data: dict[str, Any]) -> None:
        """分发期权推送帧（issue_type 33011/33012）到成交/委托回调."""
        broker_order_id = str(data.get("entrust_no", ""))
        client_order_id = self.client_order_id_for(broker_order_id)
        if event == "trade_update":
            self._emit_trade(mapper.parse_option_trade(data, client_order_id))
        elif event == "order_update":
            snapshot = mapper.parse_option_order(data, client_order_id)
            self._emit_order(snapshot)
            self._emit_exec_from_order(snapshot)
