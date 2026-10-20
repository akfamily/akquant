"""指标脚本: 只声明指标、不交易的策略单元, 对标 Pine v5 的 ``indicator()``.

写法::

    class DualMA(aq.Indicator, title="双均线", overlay=True):
        def on_start(self):
            self.fast = self.I(aq.EMA(10))

复用多 slot 策略拓扑(``strategies_by_slot``), 不新建管线: 每个指标脚本就是一个
slot, 指标点带自己的 ``owner_strategy_id``(即 :meth:`Indicator.resolve_id`)。

与普通 ``Strategy`` 的区别: 全部交易 API 抛 :class:`IndicatorCannotTradeError`;
``self.I`` 默认画图, 位置由 ``overlay`` 决定。
"""

from __future__ import annotations

import re
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, Type, Union

from .strategy import Strategy

#: ``Strategy`` 上会产生/撤销委托或改仓位的公开方法。新增交易 API 时要同步。
TRADING_API_NAMES: tuple[str, ...] = (
    "buy",
    "sell",
    "short",
    "cover",
    "submit_order",
    "place_oco",
    "place_bracket",
    "place_trailing_stop",
    "place_trailing_stop_limit",
    "order_target",
    "order_target_value",
    "order_target_percent",
    "rebalance_weights",
    "rebalance_positions",
    "rebalance_to_topn",
    "close_position",
    "cancel_order",
    "cancel_group",
    "cancel_all_orders",
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


class IndicatorCannotTradeError(RuntimeError):
    """在 :class:`Indicator` 指标脚本上调用交易 API."""


def _snake_case(name: str) -> str:
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def _make_blocker(method_name: str) -> Callable[..., Any]:
    def blocked(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise IndicatorCannotTradeError(
            f"指标脚本 {type(self).__name__!r} 调用了交易 API {method_name}(): "
            "aq.Indicator 只声明指标、不交易。要下单请改继承 aq.Strategy"
        )

    blocked.__name__ = method_name
    blocked.__qualname__ = f"Indicator.{method_name}"
    blocked.__doc__ = (
        f"指标脚本不允许交易: 调用 ``{method_name}`` 一律抛 IndicatorCannotTradeError."
    )
    return blocked


class Indicator(Strategy):
    """指标脚本基类; 经 ``run_backtest/run_live(indicators=[...])`` 挂载.

    类定义行可写 ``title``(显示名)、``overlay``(默认 False=附图)、``id``(机器标识,
    省略为类名 snake_case)。在 ``on_start`` 里用 :meth:`Strategy.I` 声明指标。
    """

    _aq_is_indicator_script: ClassVar[bool] = True
    #: 类定义行 ``overlay=`` 的值; 省略时继承父类, 直到这里的 False。
    _aq_overlay: ClassVar[bool] = False

    def __init_subclass__(
        cls,
        *,
        overlay: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """在 Strategy 的 title/id 之外再接收 ``overlay``."""
        super().__init_subclass__(**kwargs)
        if overlay is not None:
            if not isinstance(overlay, bool):
                raise TypeError(
                    f"{cls.__name__}: overlay= 必须是 bool, 收到 {overlay!r}"
                )
            cls._aq_overlay = overlay

    @classmethod
    def resolve_id(cls) -> str:
        """机器标识: 类定义行 ``id``, 省略时为类名 snake_case."""
        return cls._aq_id or _snake_case(cls.__name__)

    @classmethod
    def resolve_overlay(cls) -> bool:
        """返回是否画主图: True 画主图, False(默认)画本脚本的附图."""
        return cls._aq_overlay


for _name in TRADING_API_NAMES:
    setattr(Indicator, _name, _make_blocker(_name))


IndicatorInput = Union[Type[Indicator], Indicator]


def merge_indicators_into_slots(
    strategy_id: Optional[str],
    strategies_by_slot: Optional[Dict[str, Any]],
    indicators: Optional[Iterable[IndicatorInput]],
) -> Optional[Dict[str, Any]]:
    """把 ``indicators`` 并入 ``strategies_by_slot``, 返回新的 slot 表.

    :raises TypeError: 元素不是 ``Indicator`` 子类/实例 —— 否则"不交易"的保证就没了
    :raises ValueError: 脚本 id 撞上 ``strategy_id`` 或已有 slot key, 不静默覆盖
    """
    if not indicators:
        return strategies_by_slot
    merged: Dict[str, Any] = dict(strategies_by_slot or {})
    taken = set(merged)
    if strategy_id:
        taken.add(str(strategy_id))
    for item in indicators:
        cls = item if isinstance(item, type) else type(item)
        if not (isinstance(cls, type) and issubclass(cls, Indicator)):
            raise TypeError(
                f"indicators 只接受 aq.Indicator 的子类或实例, 收到 {item!r}; "
                "会交易的策略请走 strategies_by_slot"
            )
        key = cls.resolve_id()
        if key in taken:
            raise ValueError(
                f"指标脚本 id {key!r} 与 strategy_id 或已有 slot key 重复; "
                "请在类定义行显式写 id="
            )
        taken.add(key)
        merged[key] = item
    return merged
