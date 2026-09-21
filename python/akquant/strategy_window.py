"""引擎原生多周期: 策略侧订阅表、校验与窗口 bar 派发 (Internal).

配合 Rust `WindowAggregator`(src/data/window.rs): 策略在 ``__init__`` 里用
``subscribe_bars`` 声明周期, ``configure_engine_window_subscriptions`` 在引擎
启动前把订阅表下发给 Rust, 引擎在每根基础 bar 的 ``on_bar`` 之后把闭合的窗口
bar 回调到 ``on_window_bar_event``。
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from .akquant import Bar, StrategyContext
from .strategy_framework_hooks import call_window_bar_callback, ensure_framework_state

_FREQ_RE = re.compile(r"^(\d+)(min|h|d)$")

#: 回调定档前缀: 窗口回调期间 `_framework_current_callback` 取 "window:<freq>"。
WINDOW_CALLBACK_PREFIX = "window:"


@dataclass
class WindowSubscription:
    """一条 subscribe_bars 声明."""

    freq: str
    callback: Optional[Callable[[Bar], None]]
    symbols: Optional[frozenset[str]]
    session_windows: Optional[list[tuple[str, str]]]

    def matches(self, symbol: str) -> bool:
        """标的是否在本条订阅的范围内(``symbols=None`` 表示不限)."""
        return self.symbols is None or symbol in self.symbols


def parse_window_freq(freq: Any) -> tuple[str, int]:
    """规范化周期并返回 (label, 总分钟数).

    :raises ValueError: 非整数分钟/小时, 或非 '1d'
    """
    text = str(freq).strip().lower()
    match = _FREQ_RE.match(text)
    if match is None:
        raise ValueError(
            f"subscribe_bars(freq={freq!r}) 无法解析: 仅支持整数分钟 'Nmin'、"
            "整数小时 'Nh' 与日线 '1d'。秒级或周/月周期请改用 "
            "akquant.feed_adapter 的 resample()"
        )
    value = int(match.group(1))
    unit = match.group(2)
    if value <= 0:
        raise ValueError(f"subscribe_bars(freq={freq!r}) 数值必须为正")
    if unit == "d":
        if value != 1:
            raise ValueError(f"subscribe_bars(freq={freq!r}): 日周期仅支持 '1d'")
        return text, 1440
    return text, value * (60 if unit == "h" else 1)


def base_interval_minutes(base_freq: Optional[str]) -> Optional[int]:
    """把 ``self.freq`` 口径的基础周期转成分钟; 拿不到或不可解析返回 None."""
    if base_freq is None:
        return None
    try:
        return parse_window_freq(base_freq)[1]
    except ValueError:
        return None


def _validate_sessions(
    session_windows: Optional[Iterable[tuple[str, str]]],
) -> Optional[list[tuple[str, str]]]:
    if session_windows is None:
        return None
    out: list[tuple[str, str]] = []
    for start, end in session_windows:
        s, e = str(start).strip(), str(end).strip()
        if not (re.fullmatch(r"\d{2}:\d{2}", s) and re.fullmatch(r"\d{2}:\d{2}", e)):
            raise ValueError(f"session_windows 项 ({start!r}, {end!r}) 必须是 'HH:MM'")
        if e <= s:
            raise ValueError(
                f"session_windows 项 ({start!r}, {end!r}) 终点必须晚于起点"
            )
        out.append((s, e))
    # Rust 侧 dedup_and_validate 按序比较 sessions(见 src/data/window.rs), 排序后
    # 等价的 session 列表(元素顺序不同)才能比较相等, 否则会被误判为"不一致"而拒绝。
    out.sort(key=lambda pair: pair[0])
    return out


def subscribe_bars_impl(
    strategy: Any,
    freq: str,
    callback: Optional[Callable[[Bar], None]],
    symbols: Optional[Any],
    session_windows: Optional[Iterable[tuple[str, str]]],
) -> None:
    """``Strategy.subscribe_bars`` 的实现."""
    if getattr(strategy, "_window_subscriptions_frozen", False):
        raise RuntimeError(
            "subscribe_bars 必须在策略 __init__ 里调用: 引擎已按启动时的订阅表"
            "配置完成, 运行中新增订阅不会生效"
        )
    label, _ = parse_window_freq(freq)
    if callback is not None and not callable(callback):
        raise TypeError("subscribe_bars(callback=) 必须是可调用对象")
    symbol_filter: Optional[frozenset[str]]
    if symbols is None:
        symbol_filter = None
    elif isinstance(symbols, str):
        symbol_filter = frozenset({symbols})
    else:
        symbol_filter = frozenset(str(s) for s in symbols)
        if not symbol_filter:
            raise ValueError(
                "subscribe_bars(symbols=) 不能为空列表; 省略即覆盖全部标的"
            )
    sessions = _validate_sessions(session_windows)
    subs: list[WindowSubscription] = strategy._window_subscriptions
    for existing in subs:
        if (
            existing.freq == label
            and existing.symbols == symbol_filter
            and existing.session_windows != sessions
        ):
            raise ValueError(
                f"subscribe_bars({label!r}) 对同一标的集重复声明且 "
                "session_windows 不一致"
            )
    subs.append(WindowSubscription(label, callback, symbol_filter, sessions))


def window_freqs(strategy: Any) -> set[str]:
    """策略已声明订阅的全部周期标签集合."""
    return {s.freq for s in getattr(strategy, "_window_subscriptions", [])}


def _scopes_overlap(a: Optional[frozenset[str]], b: Optional[frozenset[str]]) -> bool:
    """``None`` 覆盖全部标的, 与任何范围都重叠; 两个具体集合仅交集非空时重叠."""
    if a is None or b is None:
        return True
    return bool(a & b)


def _merge_scoped_specs(
    entries: Iterable[
        tuple[Optional[frozenset[str]], str, Optional[list[tuple[str, str]]]]
    ],
) -> list[tuple[Optional[str], str, Optional[list[tuple[str, str]]]]]:
    """合并 (scope, freq, sessions) 三元组为下发 Rust 的 spec 列表.

    Rust 的 ``WindowAggregator::configure``(见 src/data/window.rs
    ``dedup_and_validate``)对同一 ``freq_label`` 的多条订阅只接受两种关系:
    逐字节完全相同(``symbol`` 与 ``sessions`` 都一致, 静默去重), 或者标的范围
    完全互斥; 范围有重叠但配置不完全相同(比如一条 ``symbols=None`` 一条
    ``symbols=["X"]``)会直接报错。

    但 Python 侧允许同一周期给「全部标的」挂一个通用回调、再给某个标的单独挂
    一个回调(``subscribe_bars("5min")`` + ``subscribe_bars("5min",
    callback=..., symbols=["X"])``)——这在语义上不是"重叠冲突", 而是"同一份
    聚合结果多个回调都要看"。因此这里必须在下发 Rust 之前按
    ``(freq, sessions)`` 分组合并: 组内只要有任意一条 ``symbols=None``, 整组
    只发一条 ``(None, freq, sessions)``(Rust 只聚合一次, 避免成交量翻倍这类
    双重聚合缺陷); 否则发按标的展开、去重后的多条 spec。

    组间(``freq`` 相同、``sessions`` 不同)若标的范围仍有重叠, 无法安全合并
    (两组要的窗口切法不同), 在这里就地报错, 不把 Rust 的报错留给用户当作
    第一现场。
    """
    _GroupKey = tuple[str, Optional[tuple[tuple[str, str], ...]]]
    groups: "OrderedDict[_GroupKey, list[Optional[frozenset[str]]]]" = OrderedDict()
    for entry_scope, freq, sessions in entries:
        sessions_key = tuple(sessions) if sessions is not None else None
        key = (freq, sessions_key)
        groups.setdefault(key, []).append(entry_scope)

    merged_scope: dict[
        tuple[str, Optional[tuple[tuple[str, str], ...]]], Optional[frozenset[str]]
    ] = {}
    by_freq: dict[
        str,
        list[tuple[Optional[tuple[tuple[str, str], ...]], Optional[frozenset[str]]]],
    ] = defaultdict(list)
    for key, scopes in groups.items():
        freq, sessions_key = key
        scope: Optional[frozenset[str]]
        if any(s is None for s in scopes):
            scope = None
        else:
            non_none_scopes: list[frozenset[str]] = [s for s in scopes if s is not None]
            scope = (
                frozenset[str]().union(*non_none_scopes)
                if non_none_scopes
                else frozenset()
            )
        merged_scope[key] = scope
        by_freq[freq].append((sessions_key, scope))

    for freq, group_list in by_freq.items():
        for i in range(len(group_list)):
            for j in range(i + 1, len(group_list)):
                sessions_i, scope_i = group_list[i]
                sessions_j, scope_j = group_list[j]
                if sessions_i == sessions_j:
                    continue
                if _scopes_overlap(scope_i, scope_j):
                    raise ValueError(
                        f"subscribe_bars({freq!r}) 同一周期对重叠标的的 "
                        "session_windows 必须一致: 请让重叠的订阅使用相同的 "
                        "session_windows, 或改成互斥的 symbols 范围"
                    )

    specs: list[tuple[Optional[str], str, Optional[list[tuple[str, str]]]]] = []
    for key, _scopes in groups.items():
        freq, sessions_key = key
        sessions = list(sessions_key) if sessions_key is not None else None
        scope = merged_scope[key]
        if scope is None:
            specs.append((None, freq, sessions))
        else:
            for symbol in sorted(scope):
                specs.append((symbol, freq, sessions))
    return specs


def freeze_window_subscriptions(strategies: Iterable[Any]) -> None:
    """在 on_start 之前冻结各策略的窗口订阅表: ``subscribe_bars`` 只认 ``__init__``.

    必须与 :func:`configure_engine_window_subscriptions` 分开调用——回测入口里
    真正调用 ``on_start`` 的时间点早于 ``Engine()`` 对象创建(见
    ``backtest/engine.py`` 里 "调用 on_start 获取订阅" 附近的注释), 而下发给
    Rust 需要已创建的 ``engine``。若只在后者(晚于 on_start)里冻结, ``on_start``
    里调用 ``subscribe_bars`` 会在没人察觉的情况下悄悄成功——直到引擎真正
    启动(第二次调用 on_start 时被 ``_start_initialized`` 挡住, 不会再触发)才
    被无声忽略, 用户毫无提示。这里提前到 on_start 之前冻结, 让这类误用当场
    抛 ``RuntimeError``。
    """
    for strategy in strategies:
        strategy._window_subscriptions_frozen = True


def configure_engine_window_subscriptions(
    engine: Any,
    strategies: Iterable[Any],
    base_freq: Optional[str],
    logger: logging.Logger,
) -> None:
    """把各策略的订阅表合并下发给引擎; 无订阅则什么都不做(零开销路径).

    跨策略(多 slot)一并合并——不能先各自按策略收集 spec 再简单去重: 若
    策略 A 订阅了 ``symbols=None``、策略 B 订阅了同周期的 ``symbols=["X"]``,
    两者展开后的 spec 会互相重叠但不完全相同, 直接下发会被 Rust 拒绝, 必须走
    同一套合并规则一次性处理。
    """
    strategy_list = list(strategies)
    entries: list[
        tuple[Optional[frozenset[str]], str, Optional[list[tuple[str, str]]]]
    ] = []
    for strategy in strategy_list:
        strategy._window_subscriptions_frozen = True
        for sub in strategy._window_subscriptions:
            entries.append((sub.symbols, sub.freq, sub.session_windows))
    if not entries:
        return
    specs = _merge_scoped_specs(entries)
    base_min = base_interval_minutes(base_freq)
    if base_min is None:
        logger.warning(
            "已订阅窗口周期 %s 但基础数据周期未知(self.freq=None): 窗口将在下一根落入"
            "新窗口的 bar 到达时才闭合(晚一根基础 bar)。回测请传 run_backtest(freq=), "
            "实盘请让行情网关声明 metadata['freq']",
            sorted({s[1] for s in specs}),
        )
    else:
        for _, freq, _ in specs:
            if parse_window_freq(freq)[1] <= base_min:
                raise ValueError(
                    f"subscribe_bars({freq!r}) 不高于基础周期 {base_freq!r}: "
                    "窗口周期必须严格大于输入数据周期"
                )
    engine.configure_window_subscriptions(specs, base_min)


def current_window_impl(
    strategy: Any, symbol: Optional[str], freq: str
) -> Optional[Bar]:
    """走 ctx 而非 Engine: 引擎 run() 期间再入 Engine pymethod 会 'Already borrowed'."""
    ctx: Optional[StrategyContext] = getattr(strategy, "ctx", None)
    if ctx is None:
        raise RuntimeError("current_window 只能在行情回调内调用(Context not ready)")
    label, _ = parse_window_freq(freq)
    if label not in window_freqs(strategy):
        raise ValueError(
            f"current_window(freq={freq!r}) 未订阅, 请先 subscribe_bars({label!r})"
        )
    resolved = strategy._resolve_symbol(symbol)
    return ctx.current_window(resolved, label)


def on_window_bar_event(strategy: Any, bar: Bar, ctx: StrategyContext) -> None:
    """引擎调用的窗口 bar 回调 (Internal).

    与 ``strategy_events.on_bar_event`` 同构但更薄: 不动 ``_last_prices`` /
    ``_bar_count`` / 持仓符号追踪(那些是基础行情的职责), 只做: 订单事件收尾 →
    active_start 与 warmup 门控 → 按周期驱动增量指标 → 回调。
    """
    from .strategy_events import _is_before_active_start  # 局部导入避免环

    ensure_framework_state(strategy)
    strategy.ctx = ctx
    strategy._check_order_events()
    freq = bar.freq
    if freq is None:
        return
    matched = [
        s
        for s in strategy._window_subscriptions
        if s.freq == freq and s.matches(bar.symbol)
    ]
    if not matched:
        return
    if _is_before_active_start(strategy, int(bar.timestamp)):
        return
    if hasattr(strategy, "_update_incremental_indicators"):
        strategy._update_incremental_indicators(bar)
    warmup = int(getattr(strategy, "warmup_period", 0) or 0)
    if warmup > 0 and int(strategy._symbol_bar_counts.get(bar.symbol, 0)) < warmup:
        return
    previous_bar = strategy.current_bar
    strategy.current_bar = bar
    try:
        for sub in matched:
            callback = (
                sub.callback if sub.callback is not None else strategy.on_window_bar
            )
            call_window_bar_callback(strategy, callback, bar)
    finally:
        strategy.current_bar = previous_bar
