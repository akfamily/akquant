from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .akquant import StrategyContext


class Sizer(ABC):
    """
    仓位管理基类 (Sizer Base Class).

    用于计算下单数量.
    """

    @abstractmethod
    def get_size(
        self, price: float, cash: float, context: "StrategyContext", symbol: str
    ) -> float:
        """
        计算下单数量.

        Args:
            price (float): 当前价格
            cash (float): 当前可用资金
            context (StrategyContext): 策略上下文
            symbol (str): 标的代码

        Returns:
            float: 下单数量
        """
        pass


class FixedSize(Sizer):
    """
    固定数量 Sizer.

    每次交易固定数量.
    """

    def __init__(self, size: float = 100.0):
        """
        Initialize FixedSize.

        :param size: The fixed size for each trade.
        """
        self.size = size

    def get_size(
        self, price: float, cash: float, context: "StrategyContext", symbol: str
    ) -> float:
        """Return the fixed size."""
        return self.size


class PercentSizer(Sizer):
    """
    百分比 Sizer.

    使用当前资金的一定百分比买入.
    """

    def __init__(self, percents: float = 10.0):
        """
        Initialize PercentSizer.

        Args:
            percents (float): 资金百分比 (0-100).
        """
        self.percents = percents

    def get_size(
        self, price: float, cash: float, context: "StrategyContext", symbol: str
    ) -> float:
        """Calculate order size based on percentage of cash."""
        if price <= 0:
            return 0.0

        target_cash = cash * (self.percents / 100.0)
        return int(target_cash / price)


class AllInSizer(Sizer):
    """
    全仓 Sizer.

    使用所有可用资金买入.
    """

    def get_size(
        self, price: float, cash: float, context: "StrategyContext", symbol: str
    ) -> float:
        """Calculate order size using all available cash."""
        if price <= 0:
            return 0.0
        return int(cash / price)
