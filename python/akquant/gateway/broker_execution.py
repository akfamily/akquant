"""BrokerExecution：实盘执行后端，读/撤单走柜台，submit 走 submitter."""

from __future__ import annotations

from typing import Any

from .broker_state_cache import BrokerStateCache
from .broker_strategy_api import _account_to_dict, _resolve_symbol


class BrokerExecution:
    """broker_live 后端：柜台为唯一真相."""

    def __init__(
        self,
        strategy: Any,
        trader_gateway: Any,
        state_cache: BrokerStateCache,
        submitter: Any,
    ) -> None:
        """绑定策略实例、柜台网关、状态缓存与下单器."""
        self._s = strategy
        self._gw = trader_gateway
        self._cache = state_cache
        self._submitter = submitter

    def get_position(self, symbol: str | None = None) -> float:
        """获取指定标的持仓数量."""
        return self._cache.positions().get(_resolve_symbol(self._s, symbol), 0.0)

    def get_available_position(self, symbol: str | None = None) -> float:
        """获取指定标的可用持仓数量."""
        return self._cache.available_positions().get(
            _resolve_symbol(self._s, symbol), 0.0
        )

    def get_positions(self) -> dict[str, float]:
        """获取所有持仓信息."""
        return dict(self._cache.positions())

    def hold_bar(self, symbol: str | None = None) -> int:
        """获取当前持仓持有的 Bar 数量."""
        return 0  # 柜台不提供持有 Bar 数；broker_live 下无意义

    def get_open_orders(self, symbol: str | None = None) -> list[Any]:
        """获取未完成订单列表."""
        orders = self._cache.open_orders()
        if symbol is not None:
            return [o for o in orders if getattr(o, "symbol", None) == symbol]
        return list(orders)

    def get_order(self, order_id: str) -> Any | None:
        """按订单号获取订单."""
        for o in self._cache.open_orders():
            if getattr(o, "broker_order_id", None) == order_id:
                return o
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

    def submit_order(self, **kwargs: Any) -> str:
        """提交订单，返回订单号."""
        return str(self._submitter.submit_order(**kwargs))

    def cancel_order(self, order_id: str) -> None:
        """取消指定订单."""
        self._gw.cancel_order(str(order_id))

    def cancel_all_orders(self, symbol: str | None = None) -> None:
        """取消当前所有未完成的订单."""
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
