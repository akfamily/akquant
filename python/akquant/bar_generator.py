"""运行时多周期 Bar 聚合器.

把 on_bar 的 bar 流流式聚合成更高周期(minute/hour/day 的 N 倍)并回调;
回测/实盘同一句 update_bar。底层就是引擎 ``subscribe_bars`` 用的同一个 Rust
窗口聚合器(:class:`akquant.WindowAggregator`), 对同一数据、同一时段表产出相同
的窗口: 日内窗口按(交易日, 交易块)切分、标签为本地时钟 ceil 到周期但不超过块
结束(A 股 1h: 10:00、11:00、11:30、14:00、15:00); 日线窗口按交易日切分、标签为
交易日收盘时刻。
"""

from __future__ import annotations

import sys
import weakref
from typing import Any, Callable, Iterable, Mapping, Optional, Union

from .akquant import AssetType, Bar, SessionResolver, WindowAggregator

_INTERVAL_FREQ = {"minute": "min", "hour": "h", "day": "d"}

DEFAULT_TIMEZONE = "Asia/Shanghai"

AssetTypeLike = Union[AssetType, str]


def _to_asset_type(value: AssetTypeLike) -> AssetType:
    from .backtest.engine import _parse_asset_type_name

    if isinstance(value, AssetType):
        return value
    try:
        name = _parse_asset_type_name(value)
    except ValueError:
        raise ValueError(
            f"asset_type={value!r} 无法识别，请传 AssetType 或 "
            "'stock' / 'fund' / 'futures' / 'option'"
        ) from None
    return {
        "stock": AssetType.Stock,
        "fund": AssetType.Fund,
        "futures": AssetType.Futures,
        "option": AssetType.Option,
    }[name]


class BarGenerator:
    """策略内流式多周期聚合器(vnpy 式独立 helper)."""

    def __init__(
        self,
        on_window_bar: Callable[[Bar], None],
        window: int = 1,
        interval: str = "minute",
        *,
        session_windows: Optional[list[tuple[str, str]]] = None,
        timezone: Optional[str] = None,
        sessions: Optional[Mapping[str, str]] = None,
        trading_calendar: Optional[Iterable[Any]] = None,
        asset_type: Union[AssetTypeLike, Mapping[str, AssetTypeLike], None] = None,
        base_interval_min: Optional[int] = None,
    ) -> None:
        """绑定回调与目标周期.

        :param on_window_bar: 窗口闭合时回调, 收到聚合后的 Bar(带 ``freq`` 与
            ``interval_start``)。
        :param window: 周期倍数(>=1); ``interval="day"`` 时只能为 1。
        :param interval: "minute" | "hour" | "day"。
        :param session_windows: 手写本地交易时段 ``[(start, end), ...]``, 给定则
            覆盖时段表、窗口不跨段; ``end`` 早于 ``start`` 表示跨午夜。
        :param timezone: 市场时区, 默认 ``"Asia/Shanghai"``(与 ``run_backtest``
            一致)。bar 时间戳是 UTC 纳秒, 时钟对齐与交易时段都按该时区换算。
        :param sessions: 代码前缀 → 时段模板名, 同 ``run_backtest`` 的 ``sessions``。
        :param trading_calendar: 交易日列表, 同 ``run_backtest`` 的
            ``trading_calendar``。
        :param asset_type: 资产类型(决定默认时段表), 同 ``run_backtest`` 的
            ``asset_type``, 默认股票; 也可传 ``{symbol: asset_type}``, 未列出的
            品种按股票。在运行中的策略里创建(回调是策略方法, 或在策略方法内构造)
            且 ``timezone``/``sessions``/``trading_calendar``/``asset_type`` 都
            不传时, 沿用引擎的时区、时段表与合约资产类型, 与 ``subscribe_bars``
            的窗口一致(在第一根 bar 到达时绑定)。沿用要求在策略方法内直接构造
            ``BarGenerator``(回调是策略的绑定方法, 或构造处的调用帧里有策略 ``self``);
            在子类里经 ``super().__init__`` 间接构造、且回调不是策略方法时找不到所属
            策略, 不会沿用, 此时请显式传 ``timezone``/``sessions``/``asset_type``。
        :param base_interval_min: 喂入 bar 的周期(分钟)。给定(或 bar 带
            ``interval_start``)时, bar 区间终点到达窗口标签即闭合回调; 不给时等
            下一根落入新窗口的 bar 到达才闭合(收尾靠 :meth:`flush`)。
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        if interval not in _INTERVAL_FREQ:
            raise ValueError("interval must be one of: minute, hour, day")
        if interval == "day" and window != 1:
            raise ValueError(
                "interval='day' 只支持 window=1；多日聚合请用 "
                "akquant.feed_adapter 的 resample()"
            )
        from .backtest.engine import _normalize_trading_calendar

        self._cb = on_window_bar
        self._freq = f"{window}{_INTERVAL_FREQ[interval]}"
        self._session_windows = list(session_windows) if session_windows else None
        self._base_interval_min = base_interval_min
        # 未指定时区与时段配置时, 第一根 bar 到达时尝试沿用所属策略的引擎配置
        self._owner: Optional[Callable[[], Any]] = None
        if (
            timezone is None
            and sessions is None
            and trading_calendar is None
            and asset_type is None
        ):
            self._owner = _find_owner_strategy(on_window_bar)
        self._fed = False
        tz = timezone or DEFAULT_TIMEZONE
        resolver = SessionResolver(
            sessions=dict(sessions) if sessions is not None else None,
            trading_calendar=_normalize_trading_calendar(trading_calendar),
            timezone=tz,
        )
        self._symbol_types: dict[str, AssetType] = {}
        self._default_type: Optional[AssetType]
        if isinstance(asset_type, Mapping):
            self._default_type = AssetType.Stock
            self._symbol_types = {
                str(sym): _to_asset_type(v) for sym, v in asset_type.items()
            }
        else:
            self._default_type = (
                AssetType.Stock if asset_type is None else _to_asset_type(asset_type)
            )
        self._agg = WindowAggregator(
            self._freq,
            session_windows=self._session_windows,
            base_interval_min=base_interval_min,
            resolver=resolver,
            timezone=tz,
            asset_types=self._symbol_types,
        )

    def _bind_engine_sessions(self) -> None:
        """首根 bar 时: 所属策略已绑定引擎则改用引擎的解析器与资产类型."""
        owner_ref, self._owner = self._owner, None
        strategy = owner_ref() if owner_ref is not None else None
        ctx = getattr(strategy, "ctx", None)
        getter = getattr(ctx, "_window_session_context", None)
        context = getter() if callable(getter) else None
        if context is None:
            return
        resolver, asset_types = context
        self._agg = WindowAggregator(
            self._freq,
            session_windows=self._session_windows,
            base_interval_min=self._base_interval_min,
            resolver=resolver,
            asset_types=dict(asset_types),
        )
        # 引擎未登记的品种按解析器的前缀/默认表取时段(与 subscribe_bars 一致)
        self._default_type = None

    def update_bar(self, bar: Bar) -> None:
        """喂入一根 bar(时间戳非递减); 本步闭合的窗口依次回调."""
        if not self._fed:
            self._fed = True
            if self._owner is not None:
                self._bind_engine_sessions()
        symbol = bar.symbol
        if self._default_type is not None and not self._agg.has_asset_type(symbol):
            self._agg.set_asset_type(symbol, self._default_type)
        for window_bar in self._agg.update(bar):
            self._cb(window_bar)

    def flush(self) -> None:
        """强制闭合所有 symbol 的当前未满窗口(回测收尾 / 数据结束)."""
        for window_bar in self._agg.flush():
            self._cb(window_bar)

    def current(self, symbol: str) -> Optional[Bar]:
        """返回 symbol 正在形成(未闭合)的窗口 bar 快照; 无则 None。不触发回调."""
        return self._agg.current(symbol)


def _find_owner_strategy(callback: Any) -> Optional[Callable[[], Any]]:
    """回调所属的策略(绑定方法的 ``__self__``, 否则构造处的 ``self``), 弱引用."""
    from .strategy import Strategy

    candidates = [getattr(callback, "__self__", None)]
    try:
        # 0 = 本函数, 1 = BarGenerator.__init__, 2 = 构造 BarGenerator 的调用处
        candidates.append(sys._getframe(2).f_locals.get("self"))
    except ValueError:
        pass
    for obj in candidates:
        if isinstance(obj, Strategy):
            return weakref.ref(obj)
    return None
