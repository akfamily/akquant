"""ExecutionBackend 协议：策略执行面的统一原子接口."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExecutionBackend(Protocol):
    """策略读/下单/撤单只依赖本协议；回测与实盘各一实现."""

    def get_position(self, symbol: str | None = None) -> float:
        """获取指定标的持仓数量."""

    def get_available_position(self, symbol: str | None = None) -> float:
        """获取指定标的可用持仓数量."""

    def get_positions(self) -> dict[str, float]:
        """获取所有持仓信息."""

    def hold_bar(self, symbol: str | None = None) -> int:
        """获取当前持仓持有的 Bar 数量."""

    def get_open_orders(self, symbol: str | None = None) -> list[Any]:
        """获取未完成订单列表."""

    def get_order(self, order_id: str) -> Any | None:
        """按订单号获取订单."""

    def get_account(self) -> dict[str, Any]:
        """获取账户信息."""

    def get_portfolio_value(self) -> float:
        """获取组合总市值."""

    def get_cash(self) -> float:
        """获取现金."""

    def submit_order(self, **kwargs: Any) -> str:
        """提交订单，返回订单号."""

    def cancel_order(self, order_id: str) -> None:
        """取消指定订单."""

    def cancel_all_orders(self, symbol: str | None = None) -> None:
        """取消当前所有未完成的订单."""

    def capabilities(self) -> dict[str, Any]:
        """返回当前执行后端支持的能力标记."""
