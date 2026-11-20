from typing import Any, Optional

from .akquant import StrategyContext


class Position:
    """
    持仓信息辅助类 (Position Helper).

    允许通过属性访问特定标的的持仓信息.
    """

    def __init__(
        self, ctx: StrategyContext, symbol: str, execution: Optional[Any] = None
    ) -> None:
        """
        初始化持仓辅助对象.

        :param ctx: 策略上下文
        :param symbol: 标的代码
        :param execution: 执行后端. 给出时数量与可用数量经它读取, 与
            ``get_position()`` / ``get_available_position()`` 同源(经柜台交易时
            读柜台持仓); 不给则读策略上下文
        """
        self._ctx = ctx
        self._symbol = symbol
        self._execution = execution

    @property
    def size(self) -> float:
        """持仓数量."""
        if self._execution is not None:
            return float(self._execution.get_position(self._symbol))
        return self._ctx.get_position(self._symbol)

    @property
    def available(self) -> float:
        """可用持仓数量."""
        if self._execution is not None:
            return float(self._execution.get_available_position(self._symbol))
        return self._ctx.get_available_position(self._symbol)

    @property
    def entry_price(self) -> float:
        """持仓均价."""
        # 只有持仓以柜台为准的后端才绕到持仓明细取成本; 回测直接读上下文(热路径)。
        if getattr(self._execution, "tracks_broker_positions", False):
            detail = self._execution.get_position_detail(self._symbol)  # type: ignore[union-attr]
            return float(detail.avg_price or 0.0)
        return self._ctx.get_position_entry_price(self._symbol)

    @property
    def avg_price(self) -> float:
        """持仓均价别名."""
        return self.entry_price

    def __repr__(self) -> str:
        """返回持仓信息的字符串表示."""
        return (
            f"Position(symbol={self._symbol}, size={self.size}, "
            f"available={self.available}, entry_price={self.entry_price})"
        )
