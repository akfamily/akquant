"""确定性回放行情网关.

把调用方给定的 ``Bar`` / ``Tick`` 序列按时间戳升序推入 live ``DataFeed``,
用于在没有真实柜台的环境下覆盖实盘数据通路(feed → 引擎 → on_bar/on_tick)。

**排序是本网关的硬责任**: ``RealtimeDataClient::sort()`` 是空实现
(``src/data/client.rs:374-376``, 注释 "Live data cannot be sorted"), 推送顺序
就是引擎看到的顺序。多品种必须交错推送。回测路径由 ``feed.sort()`` 兜底,
实盘没有这层保护。

**不覆盖用户 timer 语义**: live 引擎用墙钟判定 timer 是否到期
(``src/data/feed.rs`` 的 ``next_action``), 而回放数据带的是历史时间戳, 两条
时间线必然错位。``on_timer`` / ``schedule_daily`` 在回放会话中的行为不作保证。
交易日生命周期(盘前/开始交易/日终结算)例外: runner 按首个事件锚定
(``first_event_ns``)、按最后事件判定收尾(``last_event_ns``), 并在引擎开跑前
等推送完成(``wait_until_pushed``), 使其与同数据回测一致。

**行情源交易日**: ``gateway_options={"trading_day_hints": [...]}``,
每项为 ``(ts_ns, date)`` 或 ``(ts_ns, date, source)``,
经 ``DataFeed.add_trading_day_hint`` 按时间戳与行情合并推送
(同一时间戳 hint 先于行情), 模拟真实行情源(如 CTP 的
``TradingDay``)报告交易日。hint 不计入 ``bounded_event_total``,
也不参与首个/最后事件的判定。

**不模拟成交**: 本 broker 只提供行情, ``trader_gateway=None``。
"""

from __future__ import annotations

import datetime as dt
import threading
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

from ....akquant import Bar, Tick
from ....feed_adapter import DEFAULT_INPUT_TIMEZONE
from ....log import get_logger
from ...protocols import GatewayBundle

logger = get_logger("gateway.replay")

ReplayBars = Union[List[Bar], Any]  # list[Bar] 或 pandas.DataFrame
TradingDayHint = Tuple[int, dt.date, str]


def _coerce_hints(hints: Sequence[Any] | None) -> List[TradingDayHint]:
    """把 ``trading_day_hints`` 归一成按时间戳升序的 ``(ts_ns, date, source)``.

    每项为 ``(ts_ns, date)`` 或 ``(ts_ns, date, source)``; ``date`` 可为
    ``datetime.date`` 或 ``"YYYY-MM-DD"``, ``source`` 缺省 ``"replay"``。
    """
    result: List[TradingDayHint] = []
    for item in hints or []:
        if not isinstance(item, (tuple, list)) or len(item) not in (2, 3):
            raise ValueError(
                "trading_day_hints 的每一项须为 (ts_ns, date) 或 "
                f"(ts_ns, date, source), 收到 {item!r}"
            )
        ts, day = int(item[0]), item[1]
        if isinstance(day, str):
            day = dt.date.fromisoformat(day)
        elif isinstance(day, dt.datetime):
            day = day.date()
        if not isinstance(day, dt.date):
            raise ValueError(
                "trading_day_hints 的交易日须为 datetime.date 或 'YYYY-MM-DD', "
                f"收到 {day!r}"
            )
        source = str(item[2]) if len(item) == 3 else "replay"
        result.append((ts, day, source))
    result.sort(key=lambda hint: hint[0])
    return result


def _coerce_bars(
    bars: ReplayBars | None, symbols: Sequence[str], tz: str | None = None
) -> List[Bar]:
    """把入参归一成 ``list[Bar]``.

    DataFrame 走 ``normalize.dataframe_to_bars``。该函数要求时间戳列名为
    ``date``(或调用方自带 ``column_map``), 且多品种识别只认 ``股票代码`` 列
    (``normalize.py:254-255``), 两者都不满足时 symbol 会是 ``"UNKNOWN"``。

    因此: 订阅集只有一个标的、且 DataFrame 未自带 ``股票代码`` 列时, 用该标的名
    命名所有 bar——否则它们会带着 ``"UNKNOWN"`` 被 symbol 过滤器全部滤掉, 表现为
    "传了数据却一根都没推"。多品种 DataFrame 必须自带 ``股票代码`` 列, 或改用
    ``list[Bar]``。

    DataFrame 里 naive 的时间按 ``tz`` 本地化(``run_live(timezone=)`` 经
    runner 下发), 缺省 Asia/Shanghai。
    """
    if bars is None:
        return []
    if isinstance(bars, list):
        return list(bars)
    from ....normalize import dataframe_to_bars

    fallback_symbol: str | None = None
    if len(symbols) == 1 and "股票代码" not in list(getattr(bars, "columns", [])):
        fallback_symbol = str(symbols[0])
    return dataframe_to_bars(bars, symbol=fallback_symbol, tz=tz)


def _rewrite_date_only(
    bars: List[Bar], session_context: Mapping[str, Any] | None, tz: str | None
) -> List[Bar]:
    """只有日期(本地 0 点)的 bar 改写到会话收盘时刻, 与 ``run_backtest`` 同一实现.

    不改写时这类 bar 落在闭市, 订单永不成交。``session_context`` 由
    ``LiveRunner`` 按下发给引擎的同一配置写入(``sessions`` / ``trading_calendar``
    / ``instruments``); 缺省(直接构造网关)时不改写。
    """
    if session_context is None or not bars:
        return bars
    from ....backtest.engine import rewrite_date_only_bars_for_session

    return rewrite_date_only_bars_for_session(
        bars,
        instruments=list(session_context.get("instruments") or []),
        sessions=session_context.get("sessions"),
        trading_calendar=session_context.get("trading_calendar"),
        timezone=tz or DEFAULT_INPUT_TIMEZONE,
        logger=logger,
    )


def _sorted_events(
    bars: ReplayBars | None,
    ticks: Sequence[Tick] | None,
    symbols: Sequence[str],
    tz: str | None = None,
    session_context: Mapping[str, Any] | None = None,
) -> List[Union[Bar, Tick]]:
    """合并 bar/tick 并按时间戳升序排序.

    Python 的 sort 稳定, 因此同一时间戳下 bar 恒在 tick 之前(bars 先入列表),
    保证同一份输入产生逐字节相同的事件序列。
    """
    events: List[Union[Bar, Tick]] = []
    events.extend(
        _rewrite_date_only(_coerce_bars(bars, symbols, tz), session_context, tz)
    )
    events.extend(list(ticks or []))
    events.sort(key=lambda event: int(event.timestamp))
    return events


class ReplayMarketGateway:
    """实现 ``MarketGateway`` 协议的确定性回放行情网关."""

    def __init__(
        self,
        feed: Any,
        symbols: Sequence[str],
        bars: ReplayBars | None = None,
        ticks: Sequence[Tick] | None = None,
        tz: str | None = None,
        session_context: Mapping[str, Any] | None = None,
        trading_day_hints: Sequence[Any] | None = None,
    ) -> None:
        """按 ``symbols`` 过滤并预排序待推送事件."""
        self._feed = feed
        self._symbols: List[str] = [str(s) for s in symbols]
        self._all_events = _sorted_events(
            bars, ticks, self._symbols, tz, session_context
        )
        self._hints = _coerce_hints(trading_day_hints)
        self.tick_callback: Callable[[dict[str, Any]], None] | None = None
        self.bar_callback: Callable[[dict[str, Any]], None] | None = None
        self._pushed = threading.Event()

    @property
    def pending_events(self) -> List[Union[Bar, Tick]]:
        """当前订阅集下将被推送的事件(已排序、已过滤).

        始终按当前订阅集过滤, 不对空集合做"不过滤"短路: ``unsubscribe`` 可以把
        集合清空, 若此时放开过滤, 退订反而会推送**全部**品种——与订阅语义相反。
        退订全部即推送 0 条。
        """
        allowed = set(self._symbols)
        return [e for e in self._all_events if str(e.symbol) in allowed]

    def first_event_ns(self) -> Optional[int]:
        """当前订阅集下首个待推送事件的时间戳(纳秒); 无事件时为 ``None``.

        runner 据此锚定 replay 的交易日生命周期(首个事件所属交易日的盘前/开始交易
        在它之前触发, 同回测)。
        """
        pending = self.pending_events
        return int(pending[0].timestamp) if pending else None

    def last_event_ns(self) -> Optional[int]:
        """当前订阅集下最后一个待推送事件的时间戳(纳秒); 无事件时为 ``None``.

        runner 据此判定数据结束时是否已到该交易日收盘(到了才结算并触发
        on_after_trading 再结束会话, 同回测)。
        """
        pending = self.pending_events
        return int(pending[-1].timestamp) if pending else None

    def wait_until_pushed(self, timeout: float | None = None) -> bool:
        """等待 ``start()`` 把事件全部推入 feed(或推送中止), 返回是否已结束.

        replay 的生命周期定时器落在历史时刻, 按墙钟恒已到期: 推送线程尚未推完时
        feed 暂空, 引擎会先触发这些定时器(如末日收盘结算跑到当日行情之前)。
        runner 在 ``engine.run()`` 之前调用本方法, 使引擎取下一动作时全部事件已在
        feed 中, 定时器与行情按时间戳排序。
        """
        return self._pushed.wait(timeout)

    def connect(self) -> None:
        """no-op: 无外部连接."""

    def disconnect(self) -> None:
        """no-op: 无外部连接."""

    def subscribe(self, symbols: Sequence[str]) -> None:
        """替换订阅集(过滤集)."""
        self._symbols = [str(s) for s in symbols]

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        """从订阅集移除给定标的."""
        removed = {str(s) for s in symbols}
        self._symbols = [s for s in self._symbols if s not in removed]

    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """记录 tick 回调引用(与 CTPMarketAdapter 一致, 仅存不用)."""
        self.tick_callback = callback

    def on_bar(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """记录 bar 回调引用(与 CTPMarketAdapter 一致, 仅存不用)."""
        self.bar_callback = callback

    def start(self) -> None:
        """按时间戳升序把事件推入 feed.

        由 runner 在 daemon 线程上调用。异常只记日志后退出: 线程内向上抛无人
        接收, 会让回放静默中断而不留痕迹; ``duration`` 兜底避免主循环挂死。
        结束(含中止)时置位 ``wait_until_pushed`` 等待的事件。

        行情源交易日 hint 按时间戳合并: 时间戳不晚于某个行情事件的 hint 先于它推送,
        晚于全部行情的 hint 最后推送。
        """
        hints = list(self._hints)
        try:
            for event in self.pending_events:
                try:
                    while hints and hints[0][0] <= int(event.timestamp):
                        self._feed.add_trading_day_hint(*hints.pop(0))
                    if isinstance(event, Tick):
                        self._feed.add_tick(event)
                    else:
                        self._feed.add_bar(event)
                except Exception:
                    logger.exception("replay 推送事件失败, 回放中止")
                    return
            for hint in hints:
                self._feed.add_trading_day_hint(*hint)
        except Exception:
            logger.exception("replay 推送交易日 hint 失败, 回放中止")
        finally:
            self._pushed.set()


def build_replay_bundle(
    feed: Any,
    symbols: Sequence[str],
    use_aggregator: bool,
    **kwargs: Any,
) -> GatewayBundle:
    """构建回放 broker 的 ``GatewayBundle``.

    数据经 ``gateway_options`` 传入::

        run_live(
            broker="replay",
            trading_mode="paper",
            instruments=[...],
            gateway_options={"bars": [...]},   # 或 {"bars": df} / {"ticks": [...]}
        )

    ``use_aggregator`` 不适用(回放数据已是最终形态, 无需 tick→bar 聚合), 收下忽略。

    ``metadata["bounded_event_total"]`` 声明预期事件总数, ``LiveRunner`` 据此在
    事件放完后终止会话——live 循环本身在 channel 空时会无限等待。

    ``gateway_options={"freq": "1min"}`` 可声明回放数据的周期, 经
    ``metadata["freq"]`` 注入策略的只读属性 ``self.freq``(不传则为 ``None``,
    回放数据的周期无从推断)。取值用回测侧口径, 与 ``run_backtest(freq=)`` 一致。

    ``timezone`` 由 ``LiveRunner`` 按 ``run_live(timezone=)`` 写入, 用于 DataFrame
    里 naive 时间的本地化; 不要在 ``gateway_options`` 里另传。``session_context``
    同由 ``LiveRunner`` 写入(引擎的时段表、交易日历与合约), 用于把只有日期的 bar
    改写到会话收盘时刻(与 ``run_backtest`` 一致)。
    """
    _ = use_aggregator
    gateway = ReplayMarketGateway(
        feed=feed,
        symbols=symbols,
        bars=kwargs.get("bars"),
        ticks=kwargs.get("ticks"),
        tz=kwargs.get("timezone"),
        session_context=kwargs.get("session_context"),
        trading_day_hints=kwargs.get("trading_day_hints"),
    )
    pending = gateway.pending_events
    total = len(pending)
    if total == 0:
        raise ValueError(
            "broker='replay' 需要非空行情数据: 请通过 "
            "gateway_options={'bars': [...]} 或 {'ticks': [...]} 传入, "
            "且至少有一条数据的 symbol 落在 instruments 之内"
        )
    bad_event = next((e for e in pending if int(e.timestamp) <= 0), None)
    if bad_event is not None:
        raise ValueError(
            f"broker='replay' 检测到不可用时间戳: symbol={bad_event.symbol!r}, "
            f"timestamp={int(bad_event.timestamp)} (须为正数)。引擎会静默丢弃"
            "非正时间戳事件(见 src/pipeline/stages/data.rs 的 snapshot_time 比较), "
            "导致 bounded_event_total 计数永远达不到、会话挂死。常见原因: "
            "DataFrame 的日期列存在无法解析的值, pd.to_datetime(errors='coerce') "
            "会把它转成 NaT 进而产生非正时间戳——请检查 gateway_options 中传入的 "
            "'bars'/'ticks' 数据, 修正或剔除对应行"
        )
    raw_freq = kwargs.get("freq")
    freq = str(raw_freq).strip() or None if raw_freq is not None else None
    return GatewayBundle(
        market_gateway=gateway,
        trader_gateway=None,
        trader_capabilities=None,
        metadata={
            "broker": "replay",
            "bounded_event_total": total,
            "freq": freq,
            # 同时回放了 bar 与 tick 的标的是双流(见 LiveRunner._inject_dual_stream)
            "dual_stream": frozenset(
                {str(e.symbol) for e in pending if isinstance(e, Bar)}
                & {str(e.symbol) for e in pending if isinstance(e, Tick)}
            ),
        },
    )
