"""可插拔 study: 只声明指标、不交易的策略单元.

对标 TradingView 里 indicator 脚本与 strategy 脚本的分野: 一张图上可以叠 N 个
只画图的 indicator。akquant 的对应物是 :class:`Study` —— 复用多 slot 策略拓扑
(``strategies_by_slot``), **不新建管线**: 每个 study 就是一个 slot, 指标点带自己的
``owner_strategy_id``, 窗口订阅跨 slot 合并, 回测/实盘两条入口同构。

``Study`` 与普通 ``Strategy`` 的唯一区别是**全部交易 API 抛错**。这是硬保证而非
约定: study 在 ``on_bar`` 里误写 ``self.buy`` 会在引擎回调里直接抛
:class:`StudyCannotTradeError`, 不会静默成交。

设计见 ``docs/zh/meta/indicator-tradingview-rfc.md`` §2.4。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, Optional, Type, Union

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


class StudyCannotTradeError(RuntimeError):
    """在 :class:`Study` 上调用交易 API."""


def _snake_case(name: str) -> str:
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def _make_blocker(method_name: str) -> Callable[..., Any]:
    def blocked(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise StudyCannotTradeError(
            f"Study {type(self).__name__!r} 调用了交易 API {method_name}(): "
            "study 只声明指标、不交易。要下单请改继承 Strategy, "
            "或把交易逻辑放到主策略里"
        )

    blocked.__name__ = method_name
    blocked.__qualname__ = f"Study.{method_name}"
    blocked.__doc__ = (
        f"Study 不允许交易: 调用 ``{method_name}`` 一律抛 StudyCannotTradeError."
    )
    return blocked


class Study(Strategy):
    """只声明指标、不交易的策略单元; 经 ``run_backtest(studies=[...])`` 挂载.

    在 ``on_start`` 里用 :meth:`Strategy.I` 声明指标即可, 框架负责推进与上报;
    ``on_bar`` 可不覆写(基类空实现)。指标点的 ``owner_strategy_id`` 取
    :attr:`study_id`, 省略时为类名的 snake_case。
    """

    #: 该 study 在指标归属与 slot 表里的标识; 省略时按类名 snake_case 生成。
    study_id: Optional[str] = None

    @classmethod
    def resolve_study_id(cls) -> str:
        """返回本 study 的 slot 标识(显式 ``study_id`` 优先)."""
        explicit = str(cls.study_id or "").strip()
        return explicit or _snake_case(cls.__name__)


for _name in TRADING_API_NAMES:
    setattr(Study, _name, _make_blocker(_name))


StudyInput = Union[Type[Study], Study]


def merge_studies_into_slots(
    strategy_id: Optional[str],
    strategies_by_slot: Optional[Dict[str, Any]],
    studies: Optional[Iterable[StudyInput]],
) -> Optional[Dict[str, Any]]:
    """把 ``studies`` 并入 ``strategies_by_slot``, 返回新的 slot 表.

    :raises TypeError: 元素不是 ``Study`` 子类/实例 —— 否则"不交易"的保证就没了
    :raises ValueError: study_id 撞上 ``strategy_id`` 或已有 slot key, 不静默覆盖
    """
    if not studies:
        return strategies_by_slot
    merged: Dict[str, Any] = dict(strategies_by_slot or {})
    taken = set(merged)
    if strategy_id:
        taken.add(str(strategy_id))
    for item in studies:
        cls = item if isinstance(item, type) else type(item)
        if not (isinstance(cls, type) and issubclass(cls, Study)):
            raise TypeError(
                f"studies 只接受 Study 的子类或实例, 收到 {item!r}; "
                "会交易的策略请走 strategies_by_slot"
            )
        key = cls.resolve_study_id()
        if key in taken:
            raise ValueError(
                f"study_id {key!r} 与 strategy_id 或已有 slot key 重复; "
                "请给该 Study 显式设置 study_id"
            )
        taken.add(key)
        merged[key] = item
    return merged
