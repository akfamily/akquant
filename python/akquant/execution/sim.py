"""SimExecution：回测执行后端，委托给 Rust ctx（经 strategy_trading_api 原语）."""

from __future__ import annotations

from typing import Any

from .. import strategy_trading_api as api


class SimExecution:
    """回测/paper 后端：所有原子操作走 strategy.ctx."""

    def __init__(self, strategy: Any) -> None:
        """绑定策略实例，读写均经由 strategy.ctx."""
        self._s = strategy

    def get_position(self, symbol: str | None = None) -> float:
        """获取指定标的持仓数量."""
        return api._sim_get_position(self._s, symbol)

    def get_available_position(self, symbol: str | None = None) -> float:
        """获取指定标的可用持仓数量."""
        return api._sim_get_available_position(self._s, symbol)

    def get_positions(self) -> dict[str, float]:
        """获取所有持仓信息."""
        return api._sim_get_positions(self._s)

    def hold_bar(self, symbol: str | None = None) -> int:
        """获取当前持仓持有的 Bar 数量."""
        return api._sim_get_holding_bars(self._s, symbol)

    def get_open_orders(self, symbol: str | None = None) -> list[Any]:
        """获取未完成订单列表."""
        return api._sim_get_open_orders(self._s, symbol=symbol)

    def get_order(self, order_id: str) -> Any | None:
        """按订单号获取订单."""
        return api.get_order(self._s, order_id)

    def get_account(self) -> dict[str, Any]:
        """获取账户信息."""
        return api._sim_get_account(self._s)

    def get_portfolio_value(self) -> float:
        """获取组合总市值."""
        return api._sim_get_portfolio_value(self._s)

    def get_cash(self) -> float:
        """获取现金."""
        return api._sim_get_cash(self._s)

    def submit_order(self, **kwargs: Any) -> str:
        """提交订单，返回订单号."""
        return api._sim_submit_order(self._s, **kwargs)

    def cancel_order(self, order_id: str) -> None:
        """取消指定订单."""
        api._sim_cancel_order(self._s, order_id)

    def cancel_all_orders(self, symbol: str | None = None) -> None:
        """取消当前所有未完成的订单."""
        for order in self.get_open_orders(symbol=symbol):
            self.cancel_order(order.id)

    def capabilities(self) -> dict[str, Any]:
        """返回当前执行后端支持的能力标记."""
        return api._sim_capabilities(self._s)
