"""ExecutionBackend 协议：策略执行面的统一原子接口."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable


class _AccountRequired(TypedDict):
    cash: float
    equity: float
    market_value: float
    notional_value: float
    frozen_cash: float
    margin: float
    used_margin: float
    free_margin: float
    unrealized_pnl: float
    borrowed_cash: float
    short_market_value: float
    maintenance_ratio: float
    account_mode: str
    accrued_interest: float
    daily_interest: float


class AccountSnapshot(_AccountRequired, total=False):
    """``get_account()`` 返回的字典的键; 回测与实盘两个执行后端都按它构造.

    两边各写一份字面量时, 加键、改名很容易只改到一边, 按回测写的策略到实盘就
    ``KeyError``。这里集中定义, 由类型检查保证两边不漏键、不多键。

    ``available_cash`` 只有经柜台交易时才有(柜台给出的可用资金), 回测没有这个键。
    对外的 ``get_account()`` 仍标注为普通字典, 不要求调用方改类型。
    """

    available_cash: float


@dataclass(frozen=True)
class PositionLeg:
    """一个标的在一个方向(多头或空头)上的持仓.

    数量都是非负数。``None`` 表示这一项拿不到(回测没有今昨仓和冻结量; 实盘取决于
    柜台给不给), 不要当成 0。
    """

    quantity: float = 0.0
    available: float = 0.0
    avg_price: float | None = None
    today_quantity: float | None = None
    yesterday_quantity: float | None = None
    frozen_quantity: float | None = None
    last_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    margin: float | None = None


@dataclass(frozen=True)
class PositionDetail:
    """一个标的的持仓明细: 净持仓加多空两边.

    ``quantity`` / ``available`` 带符号(多头正, 空头负), 与 ``get_position()`` /
    ``get_available_position()`` 相同。``avg_price`` 取净持仓所在一侧的成本。
    ``market_value`` / ``unrealized_pnl`` / ``margin`` 是多空两边的合计, 持仓的任何
    一边拿不到就是 ``None``。回测只有净持仓, 另一边恒为空。
    """

    symbol: str
    quantity: float = 0.0
    available: float = 0.0
    avg_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    margin: float | None = None
    long: PositionLeg = field(default_factory=PositionLeg)
    short: PositionLeg = field(default_factory=PositionLeg)


@runtime_checkable
class ExecutionBackend(Protocol):
    """策略读/下单/撤单只依赖本协议；回测与实盘各一实现."""

    def get_position(self, symbol: str | None = None) -> float:
        """获取指定标的持仓数量."""

    def get_available_position(self, symbol: str | None = None) -> float:
        """获取指定标的可用持仓数量."""

    def get_closable_position(self, symbol: str | None = None) -> float:
        """获取可平持仓：结算仓扣除在途平仓/减仓单占用后的剩余可平量.

        供 ``buy()`` / ``sell()`` 在 ``position_effect="auto"`` 下拆开平腿使用。
        结算仓不含同一 on_bar 内已提交未成交的在途单，直接用它拆腿会把"先平后开"
        的反手第二腿误判成平仓（issue #361）。只投影减仓方向的在途单。
        """

    def get_projected_position(self, symbol: str | None = None) -> float:
        """获取投影持仓：结算仓叠加全部在途单的预期效果.

        供 ``order_target*`` / ``close_position`` 算 delta 使用——它们问的是
        "仓位最终会落在哪"，故开仓与平仓在途单都要计入，否则同一 on_bar 内
        连续调用会按同一个结算仓重复下单。
        """

    def get_positions(self) -> dict[str, float]:
        """获取所有持仓信息."""

    def get_position_detail(self, symbol: str | None = None) -> PositionDetail:
        """获取指定标的的持仓明细(净持仓、多空两边、成本、盈亏等)."""

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

    def get_buying_power(self) -> float:
        """获取可用买入力."""

    def submit_order(self, **kwargs: Any) -> str:
        """提交订单，返回订单号."""

    def cancel_order(self, order_id: str) -> None:
        """取消指定订单."""

    def cancel_all_orders(self, symbol: str | None = None) -> None:
        """取消当前所有未完成的订单."""

    def capabilities(self) -> dict[str, Any]:
        """返回当前执行后端支持的能力标记."""
