"""确定性回放行情网关.

把调用方给定的 ``Bar`` / ``Tick`` 序列按时间戳升序推入 live ``DataFeed``,
用于在没有真实柜台的环境下覆盖实盘数据通路(feed → 引擎 → on_bar/on_tick)。

**排序是本网关的硬责任**: ``RealtimeDataClient::sort()`` 是空实现
(``src/data/client.rs:374-376``, 注释 "Live data cannot be sorted"), 推送顺序
就是引擎看到的顺序。多品种必须交错推送。回测路径由 ``feed.sort()`` 兜底,
实盘没有这层保护。

**不覆盖 timer 语义**: live 引擎用墙钟判定 timer 是否到期
(``src/data/feed.rs:311-325``), 而回放数据带的是历史时间戳, 两条时间线必然
错位。``on_timer`` / ``schedule_daily`` 在回放会话中的行为不作保证。

**不模拟成交**: 本 broker 只提供行情, ``trader_gateway=None``。
"""

from __future__ import annotations

from typing import Any, Callable, List, Sequence, Union

from ....akquant import Bar, Tick
from ....log import get_logger
from ...protocols import GatewayBundle

logger = get_logger("gateway.replay")

ReplayBars = Union[List[Bar], Any]  # list[Bar] 或 pandas.DataFrame


def _coerce_bars(bars: ReplayBars | None, symbols: Sequence[str]) -> List[Bar]:
    """把入参归一成 ``list[Bar]``.

    DataFrame 走 ``normalize.dataframe_to_bars``。该函数要求时间戳列名为
    ``date``(或调用方自带 ``column_map``), 且多品种识别只认 ``股票代码`` 列
    (``normalize.py:254-255``), 两者都不满足时 symbol 会是 ``"UNKNOWN"``。

    因此: 订阅集只有一个标的、且 DataFrame 未自带 ``股票代码`` 列时, 用该标的名
    命名所有 bar——否则它们会带着 ``"UNKNOWN"`` 被 symbol 过滤器全部滤掉, 表现为
    "传了数据却一根都没推"。多品种 DataFrame 必须自带 ``股票代码`` 列, 或改用
    ``list[Bar]``。
    """
    if bars is None:
        return []
    if isinstance(bars, list):
        return list(bars)
    from ....normalize import dataframe_to_bars

    fallback_symbol: str | None = None
    if len(symbols) == 1 and "股票代码" not in list(getattr(bars, "columns", [])):
        fallback_symbol = str(symbols[0])
    return dataframe_to_bars(bars, symbol=fallback_symbol)


def _sorted_events(
    bars: ReplayBars | None,
    ticks: Sequence[Tick] | None,
    symbols: Sequence[str],
) -> List[Union[Bar, Tick]]:
    """合并 bar/tick 并按时间戳升序排序.

    Python 的 sort 稳定, 因此同一时间戳下 bar 恒在 tick 之前(bars 先入列表),
    保证同一份输入产生逐字节相同的事件序列。
    """
    events: List[Union[Bar, Tick]] = []
    events.extend(_coerce_bars(bars, symbols))
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
    ) -> None:
        """按 ``symbols`` 过滤并预排序待推送事件."""
        self._feed = feed
        self._symbols: List[str] = [str(s) for s in symbols]
        self._all_events = _sorted_events(bars, ticks, self._symbols)
        self.tick_callback: Callable[[dict[str, Any]], None] | None = None
        self.bar_callback: Callable[[dict[str, Any]], None] | None = None

    @property
    def pending_events(self) -> List[Union[Bar, Tick]]:
        """当前订阅集下将被推送的事件(已排序、已过滤)."""
        if not self._symbols:
            return list(self._all_events)
        allowed = set(self._symbols)
        return [e for e in self._all_events if str(e.symbol) in allowed]

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
        """
        for event in self.pending_events:
            try:
                if isinstance(event, Tick):
                    self._feed.add_tick(event)
                else:
                    self._feed.add_bar(event)
            except Exception:
                logger.exception("replay 推送事件失败, 回放中止")
                return


def build_replay_bundle(
    feed: Any,
    symbols: Sequence[str],
    bars: ReplayBars | None = None,
    ticks: Sequence[Tick] | None = None,
) -> GatewayBundle:
    """构建回放行情网关 bundle (MarketGateway 仅有, no trader)."""
    market_gateway = ReplayMarketGateway(
        feed=feed, symbols=symbols, bars=bars, ticks=ticks
    )
    return GatewayBundle(market_gateway=market_gateway, trader_gateway=None)
