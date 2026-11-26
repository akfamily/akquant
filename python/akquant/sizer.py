from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .akquant import StrategyContext


class Sizer(ABC):
    """
    仓位管理基类 (Sizer Base Class).

    用于计算下单数量.

    ``buy(symbol)`` 不给数量时调用 ``get_size(price, cash, context, symbol)``:

    - ``price``: 参考价(限价单为限价, 市价单为最新价)。
    - ``cash``: 期货为可用资金(权益 − 占用保证金 − 冻结资金, 同
      ``get_account()["available_cash"]``); 其余品种为 ``ctx.cash``。``broker_live``
      下为柜台现金 ``get_cash()``。

    子类的 ``get_size`` 若声明了名为 ``unit_cost`` 的参数, 调用方会把每单位占用资金
    (价格 × 合约乘数 × 保证金率, 保证金率只对期货生效)传进来; 没有声明则不传,
    原有的四参数写法照常工作。
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


def _divisor(price: float, unit_cost: Optional[float]) -> float:
    """每单位占用资金: 调用方给了 ``unit_cost`` 用它, 否则按价格(全额)算."""
    return price if unit_cost is None else unit_cost


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

    用可用资金的 ``percents``% 作为这笔开仓占用的资金: 股票、基金、期权买方按成交
    金额算, 期货按保证金(价格 × 合约乘数 × 保证金率)算.
    """

    def __init__(self, percents: float = 10.0):
        """
        Initialize PercentSizer.

        Args:
            percents (float): 资金百分比 (0-100).
        """
        self.percents = percents

    def get_size(
        self,
        price: float,
        cash: float,
        context: "StrategyContext",
        symbol: str,
        unit_cost: Optional[float] = None,
    ) -> float:
        """Calculate order size based on percentage of cash.

        :param unit_cost: 每单位占用资金, 由下单入口按合约快照算好传入; 不传时按
            ``price`` 算(直接调用本方法时与旧口径相同)。
        """
        divisor = _divisor(price, unit_cost)
        if divisor <= 0:
            return 0.0

        target_cash = cash * (self.percents / 100.0)
        return int(target_cash / divisor)


class AllInSizer(Sizer):
    """
    全仓 Sizer.

    用全部可用资金开仓, 即 ``PercentSizer(100)``.
    """

    def get_size(
        self,
        price: float,
        cash: float,
        context: "StrategyContext",
        symbol: str,
        unit_cost: Optional[float] = None,
    ) -> float:
        """Calculate order size using all available cash.

        :param unit_cost: 每单位占用资金, 含义同 :meth:`PercentSizer.get_size`。
        """
        divisor = _divisor(price, unit_cost)
        if divisor <= 0:
            return 0.0
        return int(cash / divisor)
