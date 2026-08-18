"""BrokerExecution：实盘执行后端，读/撤单走柜台，submit 走 submitter."""

from __future__ import annotations

from typing import Any, cast

from ..log import get_logger
from .broker_event_adapter import map_local_stop, map_order_snapshot
from .broker_state_cache import BrokerStateCache
from .broker_strategy_api import _account_to_dict, _resolve_symbol
from .local_stop_book import (
    LocalStopBook,
    LocalStopOrder,
    is_stop_order_type,
    underlying_order_type,
)
from .order_audit import record_cancel
from .order_receipt import OrderReceipt

logger = get_logger("gateway.live")

MAX_STOP_SUBMIT_ATTEMPTS = 3


class BrokerExecution:
    """broker_live 后端：柜台为唯一真相."""

    def __init__(
        self,
        strategy: Any,
        trader_gateway: Any,
        state_cache: BrokerStateCache,
        submitter: Any,
        record_stop_remap: Any = None,
        group_broker_ids: Any = None,
    ) -> None:
        """绑定策略实例、柜台网关、状态缓存、下单器、止损 remap 与 group 反查回调."""
        self._s = strategy
        self._gw = trader_gateway
        self._cache = state_cache
        self._submitter = submitter
        self._record_stop_remap = record_stop_remap
        self._group_broker_ids = group_broker_ids
        self._stop_book = LocalStopBook()
        self._stop_seq = 0
        self._warned_hold_bar = False

    def _owner_strategy_id(self) -> str | None:
        """本地止损/委托适配对象的归属策略 id(无则 None)."""
        return getattr(self._s, "strategy_id", None)

    def get_position(self, symbol: str | None = None) -> float:
        """获取指定标的持仓数量."""
        return self._cache.positions().get(_resolve_symbol(self._s, symbol), 0.0)

    def get_available_position(self, symbol: str | None = None) -> float:
        """获取指定标的可用持仓数量."""
        return self._cache.available_positions().get(
            _resolve_symbol(self._s, symbol), 0.0
        )

    def get_closable_position(self, symbol: str | None = None) -> float:
        """获取可平持仓; broker_live 下降级为结算仓(理由同 get_projected_position)."""
        return self.get_position(symbol)

    def get_projected_position(self, symbol: str | None = None) -> float:
        """获取投影持仓; broker_live 下降级为结算仓.

        回测侧该方法把同周期在途单计入,供 auto 拆腿与目标仓位算 delta(#361)。实盘
        无法照做: 柜台挂单快照 ``UnifiedOrderSnapshot`` 只有 ``position_effect``
        与 ``filled_quantity``, **没有** ``side`` 和 ``quantity``——
        ``map_order_snapshot`` 的这两个字段取自 ``request``, 而 ``get_open_orders``
        并不传 ``request``, 故算不出 ``remaining`` 也判不了方向。

        因此这里返回结算仓, 与 :meth:`get_position` 同值: 保持协议完整(不抛
        AttributeError)且不用缺失数据伪造投影。实盘开平通常已由柜台/网关侧
        转换(参见 vn.py ``OffsetConverter`` 的定位), 代价是实盘"先平后开"的
        反手第二腿仍可能被标成平仓。要真正修实盘需给
        ``UnifiedOrderSnapshot`` 补 ``side`` / ``quantity``, 会牵动各 broker 插件
        的 mapper, 不在本次范围内。
        """
        return self.get_position(symbol)

    def get_positions(self) -> dict[str, float]:
        """获取所有持仓信息."""
        return dict(self._cache.positions())

    def hold_bar(self, symbol: str | None = None) -> int:
        """获取当前持仓持有的 Bar 数量.

        柜台不提供持有 Bar 数, broker_live 下恒返回 0。首次调用发一次告警,
        避免依赖持仓时长的离场逻辑在实盘静默失效(回测 hold_bar 有真实值)。
        """
        if not self._warned_hold_bar:
            self._warned_hold_bar = True
            logger.warning(
                "broker_live 不提供 hold_bar(持有 Bar 数), 恒返回 0; "
                "依赖持仓时长的离场逻辑在实盘不会触发"
            )
        return 0

    def get_open_orders(self, symbol: str | None = None) -> list[Any]:
        """获取未完成订单列表(含本地止损).

        经 broker_event_adapter 适配为与回测 Order 同形状的 StrategyOrder
        (有 `.id`、status 为 OrderStatus 枚举), 以保证策略读路径两模式一致。
        """
        owner = self._owner_strategy_id()
        snaps = self._cache.open_orders()
        if symbol is not None:
            snaps = [o for o in snaps if getattr(o, "symbol", None) == symbol]
        orders = [map_order_snapshot(s, owner_strategy_id=owner) for s in snaps]
        stops = [
            map_local_stop(o, owner_strategy_id=owner)
            for o in self._stop_book.open_orders(symbol)
        ]
        return orders + stops

    def get_order(self, order_id: str) -> Any | None:
        """按订单号获取订单(适配为与回测同形状的 StrategyOrder).

        查找顺序: 本地止损簿(LSTOP-* id) → 柜台未完成委托(broker_order_id) →
        柜台回查已完成委托(query_order, 对齐回测能取到已成交/已撤单)。
        """
        oid = str(order_id)
        owner = self._owner_strategy_id()
        for stop in self._stop_book.open_orders():
            if stop.local_id == oid:
                return map_local_stop(stop, owner_strategy_id=owner)
        for snap in self._cache.open_orders():
            if getattr(snap, "broker_order_id", None) == oid:
                return map_order_snapshot(snap, owner_strategy_id=owner)
        query_order = getattr(self._gw, "query_order", None)
        if callable(query_order):
            try:
                snap = query_order(oid)
            except Exception:  # noqa: BLE001 柜台回查失败 → 视为查无此单
                snap = None
            if snap is not None:
                return map_order_snapshot(snap, owner_strategy_id=owner)
        return None

    def get_account(self) -> dict[str, Any]:
        """获取账户信息."""
        return _account_to_dict(self._cache.account())

    def get_portfolio_value(self) -> float:
        """获取组合总市值."""
        return float(getattr(self._cache.account(), "equity", 0.0) or 0.0)

    def get_cash(self) -> float:
        """获取现金."""
        return float(getattr(self._cache.account(), "cash", 0.0) or 0.0)

    def get_buying_power(self) -> float:
        """获取可用买入力（实盘暂以可用现金近似）."""
        return self.get_cash()

    def submit_order(self, **kwargs: Any) -> OrderReceipt:
        """提交订单，返回下单回执（OrderReceipt，含全部腿 id）.

        条件/止损单（trigger_price/trail_offset 或止损类 order_type）拦截入
        本地簿，不下发柜台；柜台不支持这类原生条件单——本地止损单只有单一
        本地 id，封装为单腿 OrderReceipt 以保持返回类型与柜台下单一致。

        time_in_force=None（Strategy.submit_order 的缺省值）会显式覆盖
        submitter.submit_order 签名默认值 "GTC"；这里丢弃 None，
        保留旧 broker_live 行为：未指定 TIF 时默认为 "GTC"。
        """
        if (
            kwargs.get("trigger_price") is not None
            or kwargs.get("trail_offset") is not None
            or is_stop_order_type(kwargs.get("order_type"))
        ):
            local_id = self._register_local_stop(**kwargs)
            return OrderReceipt.single(
                group_id=local_id,
                broker_order_id=local_id,
                position_effect=str(kwargs.get("position_effect") or "auto"),
                client_order_id=local_id,
            )
        if kwargs.get("time_in_force") is None:
            kwargs.pop("time_in_force", None)
        return cast(OrderReceipt, self._submitter.submit_order(**kwargs))

    def _next_local_stop_id(self) -> str:
        self._stop_seq += 1
        return f"LSTOP-{self._stop_seq}"

    def _register_local_stop(self, **kwargs: Any) -> str:
        local_id = self._next_local_stop_id()
        symbol = _resolve_symbol(self._s, kwargs.get("symbol"))
        known = {
            "symbol",
            "side",
            "quantity",
            "price",
            "order_type",
            "trigger_price",
            "trail_offset",
            "trail_reference_price",
            "time_in_force",
        }
        extra = {k: v for k, v in kwargs.items() if k not in known}
        self._stop_book.register(
            LocalStopOrder(
                local_id=local_id,
                symbol=symbol,
                side=str(kwargs.get("side", "Buy")),
                quantity=float(kwargs.get("quantity") or 0.0),
                order_type=str(kwargs.get("order_type") or "stopmarket")
                .strip()
                .lower(),
                trigger_price=kwargs.get("trigger_price"),
                price=kwargs.get("price"),
                trail_offset=kwargs.get("trail_offset"),
                trail_reference_price=kwargs.get("trail_reference_price"),
                time_in_force=kwargs.get("time_in_force"),
                extra=extra,
            )
        )
        return local_id

    def check_stop_triggers(
        self,
        symbol: str,
        last: float,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        """盯价触发本地止损; 命中提交底层单; 失败重试(上限)+on_error, 成功记 remap."""
        for order in self._stop_book.check(symbol, last, high, low):
            ut = underlying_order_type(order.order_type)
            kwargs: dict[str, Any] = dict(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price if ut == "Limit" else None,
                order_type=ut,
                **order.extra,
            )
            if order.time_in_force is not None:
                kwargs["time_in_force"] = order.time_in_force
            try:
                receipt = self._submitter.submit_order(**kwargs)
            except Exception as exc:  # noqa: BLE001 — 参数/能力校验等仍会在此抛出
                self._handle_stop_submit_failure(order, exc)
                continue
            if not receipt.primary:
                # submit_order 对可分类的柜台失败(明确拒单/状态未知)已不再抛异常
                # (见 order_submitter._handle_place_order_failure), 改回吐空回执;
                # 这里补上与旧的"异常穿透"等价的失败处置, 保住重试与
                # on_error(..., "stop_trigger", ...) 语义——不按分类区别对待。
                self._handle_stop_submit_failure(
                    order,
                    RuntimeError("止损单提交失败: 柜台拒单或状态未知, 回执为空"),
                )
                continue
            broker_order_id = str(receipt.primary)
            if self._record_stop_remap is not None:
                try:
                    self._record_stop_remap(order.local_id, broker_order_id)
                except Exception:  # noqa: BLE001
                    pass  # remap 记录失败不影响触发/不中断 run

    def _handle_stop_submit_failure(self, order: Any, exc: Exception) -> None:
        """止损单提交失败的统一处置: 重试(上限)+on_error, 与旧的异常穿透行为对等."""
        order.submit_attempts += 1
        if order.submit_attempts < MAX_STOP_SUBMIT_ATTEMPTS:
            self._stop_book.register(order)  # 重入簿, 下 tick 重试
        self._notify_stop_error(exc, order)

    def _notify_stop_error(self, exc: Exception, order: Any) -> None:
        """止损提交失败通知策略 on_error(可选实现), 异常吞掉不影响主流程."""
        on_error = getattr(self._s, "on_error", None)
        if callable(on_error):
            try:
                on_error(exc, "stop_trigger", order)
            except Exception:  # noqa: BLE001
                pass

    def cancel_order(self, order_id: str) -> None:
        """取消指定订单(本地止损优先, 否则走柜台)."""
        if self._stop_book.cancel(str(order_id)):
            return
        record_cancel(
            broker_order_id=str(order_id),
            strategy_id=self._owner_strategy_id(),
        )
        self._gw.cancel_order(str(order_id))

    def cancel_group(self, group_id: str) -> None:
        """撤销一个逻辑委托的全部腿（按 group_id）."""
        gid = str(group_id)
        if self._group_broker_ids is None:
            return
        for broker_order_id in self._group_broker_ids(gid):
            if broker_order_id:
                self.cancel_order(str(broker_order_id))

    def cancel_all_orders(self, symbol: str | None = None) -> None:
        """取消所有未完成订单(含本地止损)."""
        for order in self._stop_book.open_orders(symbol):
            self._stop_book.cancel(order.local_id)
        for order in self._gw.sync_open_orders():
            bid = getattr(order, "broker_order_id", "")
            if symbol is not None and getattr(order, "symbol", None) != symbol:
                continue
            if bid:
                self._gw.cancel_order(str(bid))

    def capabilities(self) -> dict[str, Any]:
        """返回当前执行后端支持的能力标记."""
        return dict(self._submitter._get_execution_capabilities())

    @property
    def state_cache(self) -> BrokerStateCache:
        """暴露底层状态缓存，便于测试/上层复用."""
        return self._cache
