"""同一时间片的多标的 bar 批量回调 ``on_bars(batch)``.

``on_bar`` 每根 bar 回调一次; 多标的策略要对同一时刻的全部标的统一处理时, 重写
``Strategy.on_bars``, 框架在该时间片的 bar 到齐后回调一次。

实现全部在 Python 框架层:

- 每根基础周期 bar 经 :func:`collect_bar` 收进当前时间片;
- 时间片的首根 bar 登记一个框架定时器, 到点由 :func:`dispatch_bar_slice_timer`
  派发批次。回测在 ``E + 1ns`` 触发(时间片恰好结束、下一片尚未开始); 实盘在
  ``E + bars_deadline_seconds`` 触发, 期间已见过的标的全部到齐则提前派发,
  先见到更晚时间片的 bar 也提前派发。

定时器 payload 复用截面定时器的 ``__framework_cross_section__|<标记>|<source_ts>``
格式: 引擎只解析末段 ``source_ts``, 据此把定时器归入该时间片(沿用本片各品种的
交易时段, 其中下的单按 ``source_ts`` 撮合), 与在本片最后一根 ``on_bar`` 里下单
同一口径。中间字段取 ``bars`` 以区别于每日一次的 ``on_cross_section``。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Iterator, Mapping, Optional, Set, Tuple

from .log import get_logger

logger = get_logger("strategy.bars")

#: 批次定时器 payload 前缀(末段为时间片时刻, 见模块文档).
BAR_SLICE_TIMER_PREFIX = "__framework_cross_section__|bars|"

#: 批次由回测的时间片边界派发.
FLUSH_BOUNDARY = "boundary"
#: 实盘: 此前见过的标的全部到齐.
FLUSH_COMPLETE = "complete"
#: 实盘: 先见到了更晚时间片的 bar.
FLUSH_WATERMARK = "watermark"
#: 实盘: 等到截止时间仍未到齐.
FLUSH_DEADLINE = "deadline"

#: 实盘截止时间的缺省值(秒): 须大于行情源自身的补发窗口。本地聚合的行情网关
#: 在区间终点 + 2 秒宽限后才补发没有后续报价的标的, 且每秒轮询一次。
DEFAULT_BARS_DEADLINE_SECONDS = 5.0


@dataclass(frozen=True)
class BarBatch:
    """同一时间片的一批 bar(``on_bars`` 的入参).

    :ivar timestamp: 时间片时刻(纳秒, 即其中每根 bar 的 ``timestamp``)
    :ivar bars: 本片的 bar, 按 ``symbol`` 升序
    :ivar missing: 此前出现过、本片没有可用 bar 的标的, 升序(停牌、无成交、迟到,
        或该标的仍在预热期)
    :ivar flush_reason: 派发原因: ``"boundary"``(回测时间片边界) /
        ``"complete"``(实盘到齐) / ``"watermark"``(实盘见到更晚的时间片) /
        ``"deadline"``(实盘等到截止时间)
    """

    timestamp: int
    bars: Tuple[Any, ...]
    missing: Tuple[str, ...] = ()
    flush_reason: str = FLUSH_BOUNDARY

    @cached_property
    def by_symbol(self) -> Mapping[str, Any]:
        """``symbol -> bar`` 映射."""
        return {bar.symbol: bar for bar in self.bars}

    @property
    def symbols(self) -> Tuple[str, ...]:
        """本片有 bar 的标的, 升序."""
        return tuple(bar.symbol for bar in self.bars)

    def get(self, symbol: str, default: Any = None) -> Any:
        """取某标的的 bar, 本片没有时返回 ``default``."""
        return self.by_symbol.get(symbol, default)

    def __iter__(self) -> Iterator[Any]:
        """按 ``symbol`` 升序遍历 bar."""
        return iter(self.bars)

    def __getitem__(self, symbol: str) -> Any:
        """取某标的的 bar, 本片没有时抛 ``KeyError``."""
        return self.by_symbol[symbol]

    def __contains__(self, symbol: object) -> bool:
        """本片是否有该标的的 bar."""
        return symbol in self.by_symbol

    def __len__(self) -> int:
        """本片 bar 的根数."""
        return len(self.bars)


@dataclass
class _BarSliceState:
    """一个策略实例的时间片收集状态(会话内派生态, 不随存档保存)."""

    slice_ts: Optional[int] = None
    bars: Dict[str, Any] = field(default_factory=dict)
    #: 本片已过预热期的标的: 只有它们的 bar 进批次(全在预热期内的时间片不派发)
    ready: Set[str] = field(default_factory=set)
    fired_ts: int = 0
    seen: Set[str] = field(default_factory=set)
    #: 本片开始之前见过的标的: 实盘据此判断"到齐"
    expected: frozenset[str] = frozenset()
    late_count: int = 0
    #: 实盘 on_cross_section: 当前交易日及其已出现过 bar 的标的
    cs_day: Any = None
    cs_seen: Set[str] = field(default_factory=set)
    late_warned: Set[str] = field(default_factory=set)


def wants_bar_batches(strategy: Any) -> bool:
    """策略是否重写了 ``on_bars``(按类判定, 结果缓存)."""
    cached = strategy.__dict__.get("_framework_wants_bar_batches")
    if cached is None:
        from .strategy_framework_hooks import _strategy_overrides_callback

        cached = _strategy_overrides_callback(strategy, "on_bars")
        strategy._framework_wants_bar_batches = cached
    return bool(cached)


def wants_live_cross_section(strategy: Any) -> bool:
    """是否由时间片批次驱动 ``on_cross_section``(实盘入口且重写了该钩子).

    回测的触发时刻由数据预先算出; 实盘(含 replay)事先不知道行情, 改在批次派发时
    判断, 见 :func:`_maybe_fire_cross_section`。
    """
    cached = strategy.__dict__.get("_framework_live_cross_section")
    if cached is None:
        from .strategy_framework_hooks import _strategy_overrides_callback

        cached = bool(
            getattr(strategy, "_live_market_data_owner", False)
        ) and _strategy_overrides_callback(strategy, "on_cross_section")
        strategy._framework_live_cross_section = cached
    return bool(cached)


def collects_bar_slices(strategy: Any) -> bool:
    """是否需要按时间片收集 bar."""
    return wants_bar_batches(strategy) or wants_live_cross_section(strategy)


def _state(strategy: Any) -> _BarSliceState:
    state = strategy.__dict__.get("_bar_slice_state")
    if state is None:
        state = _BarSliceState()
        strategy._bar_slice_state = state
    return state  # type: ignore[no-any-return]


def _is_live(strategy: Any) -> bool:
    """是否按实盘规则派发(到齐 / 截止时间).

    replay 行情走实盘入口, 但数据是预先排好序的历史 bar、时间片时刻与墙钟无关:
    按回测的时间片边界派发。
    """
    return bool(getattr(strategy, "_live_market_data_owner", False)) and not bool(
        getattr(strategy, "_live_replay_market", False)
    )


def _deadline_ns(strategy: Any) -> int:
    seconds = getattr(strategy, "bars_deadline_seconds", DEFAULT_BARS_DEADLINE_SECONDS)
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        value = DEFAULT_BARS_DEADLINE_SECONDS
    return max(int(value * 1_000_000_000), 1)


def _wall_clock_ns() -> int:
    return time.time_ns()


def note_bar_symbol(strategy: Any, symbol: str) -> None:
    """记下出现过 bar 的标的(含预加载的历史 bar), 供 ``missing`` 与到齐判断."""
    if collects_bar_slices(strategy):
        _state(strategy).seen.add(str(symbol))


def collect_bar(strategy: Any, bar: Any) -> bool:
    """把一根基础周期 bar 收进当前时间片, 返回是否已收入(迟到的不收).

    必须在该 bar 推进指标、调用 ``on_bar`` 之前调用: 实盘先见到更晚时间片的 bar 时,
    上一片在这里先行派发。
    """
    if not collects_bar_slices(strategy):
        return False
    state = _state(strategy)
    ts = int(bar.timestamp)
    symbol = str(bar.symbol)

    if state.slice_ts is not None and ts > state.slice_ts and state.bars:
        _fire(strategy, state, FLUSH_WATERMARK)

    if ts <= state.fired_ts or (state.slice_ts is not None and ts < state.slice_ts):
        # 所属时间片已派发(或已被更晚的时间片取代): 只走 on_bar, 不补进批次。
        state.seen.add(symbol)
        state.late_count += 1
        if symbol not in state.late_warned:
            state.late_warned.add(symbol)
            logger.warning(
                "标的 %s 时间戳为 %d 的 bar 在所属时间片的 on_bars 派发之后才到达，"
                "只触发 on_bar、不再补进批次（每个标的只提示一次）。实盘可调大 "
                "bars_deadline_seconds",
                symbol,
                ts,
            )
        return False

    if ts != state.slice_ts:
        state.slice_ts = ts
        state.bars = {}
        state.ready = set()
        state.expected = frozenset(state.seen)
        _schedule_slice_timer(strategy, ts)
    state.seen.add(symbol)
    state.bars[symbol] = bar
    return True


def mark_bar_ready(strategy: Any, bar: Any) -> None:
    """该 bar 已过预热期(``on_bar`` 会被调用): 进批次."""
    state = strategy.__dict__.get("_bar_slice_state")
    if state is not None and state.slice_ts == int(bar.timestamp):
        state.ready.add(str(bar.symbol))


def flush_if_complete(strategy: Any) -> None:
    """实盘: 此前见过的标的在本片全部到齐时立即派发, 不等截止时间."""
    state = strategy.__dict__.get("_bar_slice_state")
    if state is None or not state.bars or not _is_live(strategy):
        return
    # 首个时间片没有"此前见过的标的"可比, 只能等截止时间。
    if state.expected and state.expected.issubset(state.bars):
        _fire(strategy, state, FLUSH_COMPLETE)


def _schedule_slice_timer(strategy: Any, slice_ts: int) -> None:
    if _is_live(strategy):
        # 时间片时刻已远早于墙钟(回放、行情延迟)时改从当前时刻起算, 否则定时器
        # 登记即到期, 批次只会收到首根 bar。
        deadline = _deadline_ns(strategy)
        trigger = max(slice_ts, _wall_clock_ns()) + deadline
    else:
        trigger = slice_ts + 1
    strategy.ctx.schedule(trigger, f"{BAR_SLICE_TIMER_PREFIX}{slice_ts}")


def dispatch_bar_slice_timer(strategy: Any, payload: str) -> bool:
    """处理批次定时器, 返回是否已消费该 payload."""
    if not payload.startswith(BAR_SLICE_TIMER_PREFIX):
        return False
    try:
        slice_ts = int(payload[len(BAR_SLICE_TIMER_PREFIX) :])
    except ValueError:
        return True
    state = strategy.__dict__.get("_bar_slice_state")
    if state is None or state.slice_ts != slice_ts or not state.bars:
        return True  # 已提前派发(到齐 / 见到更晚的时间片)
    _fire(strategy, state, FLUSH_DEADLINE if _is_live(strategy) else FLUSH_BOUNDARY)
    return True


def flush_at_replay_end(strategy: Any) -> None:
    """Replay 会话收尾: 派发最后一个时间片的批次.

    replay 数据不在收盘时刻结束时, 会话在最后一根 bar 处理完后立即结束, 末片的
    批次定时器来不及触发。真实实盘不在收尾时派发: 停止时刻的不完整批次会让策略
    在会话结束的瞬间向柜台发单。
    """
    state = strategy.__dict__.get("_bar_slice_state")
    if state is None or not state.bars:
        return
    if not getattr(strategy, "_live_replay_market", False):
        return
    _fire(strategy, state, FLUSH_BOUNDARY)


def _fire(strategy: Any, state: _BarSliceState, reason: str) -> None:
    from .strategy_framework_hooks import _run_in_framework_phase

    slice_ts = int(state.slice_ts or 0)
    all_symbols = set(state.bars)
    bars = {symbol: bar for symbol, bar in state.bars.items() if symbol in state.ready}
    state.bars = {}
    state.ready = set()
    state.fired_ts = max(state.fired_ts, slice_ts)
    if wants_live_cross_section(strategy):
        _maybe_fire_cross_section(strategy, state, slice_ts, all_symbols, reason)
    if not bars or not wants_bar_batches(strategy):
        return
    batch = BarBatch(
        timestamp=slice_ts,
        bars=tuple(bars[symbol] for symbol in sorted(bars)),
        missing=tuple(sorted(state.seen.difference(bars))),
        flush_reason=reason,
    )
    _run_in_framework_phase(
        strategy,
        "bar_slice",
        slice_ts,
        "on_bars",
        batch,
        payload=batch,
        use_previous_account_snapshot=False,
    )


#: 这些时段里没有 bar 属正常(未开盘、午休、已收盘), 不是停牌.
_NOT_TRADING_SESSIONS = ("Closed", "Break", "PostClose")


def _maybe_fire_cross_section(
    strategy: Any,
    state: _BarSliceState,
    slice_ts: int,
    symbols: Set[str],
    reason: str,
) -> None:
    """实盘: 当日首个跨标的完整时间片派发后触发 ``on_cross_section``(每日一次).

    "完整"指此前见过的标的当日都已出现过 bar。当日尚未出现的标的若此刻不在交易
    时段(未开盘的品种, 如没有夜盘的期货)则继续等; 在交易时段却没有 bar 的按停牌
    处理、不等。先见到更晚时间片而派发的批次不触发(历史里已有那根更晚的 bar)。
    """
    from .strategy_framework_hooks import _run_in_framework_phase

    ctx = strategy.ctx
    day = getattr(ctx, "trading_day", None)
    if day is None:
        return
    if state.cs_day != day:
        state.cs_day = day
        state.cs_seen = set()
    state.cs_seen.update(symbols)
    if getattr(strategy, "_framework_cross_section_done_date", None) == day:
        return
    if reason == FLUSH_WATERMARK:
        return
    for symbol in state.seen.difference(state.cs_seen):
        try:
            session = str(ctx.session_of(symbol)).rsplit(".", 1)[-1]
        except Exception:
            continue
        if session in _NOT_TRADING_SESSIONS:
            return
    strategy._framework_cross_section_done_date = day
    _run_in_framework_phase(
        strategy,
        "cross_section",
        slice_ts,
        "on_cross_section",
        day,
        slice_ts,
        payload={"trading_date": day, "timestamp": slice_ts},
        use_previous_account_snapshot=False,
    )
