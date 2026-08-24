from collections import deque
from typing import Any, Callable, Iterable

from ..log import build_log_extra, get_logger
from .order_audit import record_broker_event
from .symbol_match import normalize_symbol_for_match

logger = get_logger("gateway.live")

#: 会话级委托状态指纹表上限(有界 FIFO)。与回测侧
#: ``strategy_order_events._ORDER_EVENT_DEDUPE_LIMIT`` 取同一量级。
_ORDER_STATE_DEDUPE_LIMIT = 50000


class BrokerEventBridge:
    """Own broker event deduplication, state updates and callback fanout."""

    def __init__(
        self,
        *,
        event_lock: Any,
        event_store: list[tuple[str, Any]],
        event_keys: set[str],
        get_on_broker_event: Callable[[], Callable[[dict[str, Any]], None] | None],
        make_event_key: Callable[[str, Any], str],
        update_broker_state: Callable[[str, Any], None],
        resolve_owner_strategy_id: Callable[[Any], str],
        payload_to_dict: Callable[[Any], dict[str, Any]],
        safe_strategy_callback: Callable[[Any, str, Any], None],
        adapt_strategy_payload: Callable[[str, Any], Any],
        payload_field: Callable[[Any, str], Any],
        resolve_trace_id: Callable[[Any], str] | None = None,
        get_subscribed_symbols: Callable[[], set[str]] | None = None,
    ) -> None:
        """Bind the queue, state callbacks and observer fanout dependencies."""
        self._event_lock = event_lock
        self._event_store = event_store
        self._event_keys = event_keys
        self._get_on_broker_event = get_on_broker_event
        self._make_event_key = make_event_key
        self._update_broker_state = update_broker_state
        self._resolve_owner_strategy_id = resolve_owner_strategy_id
        self._resolve_trace_id = resolve_trace_id
        self._payload_to_dict = payload_to_dict
        self._safe_strategy_callback = safe_strategy_callback
        self._adapt_strategy_payload = adapt_strategy_payload
        # 会话级已入队 trade_id(不随 drain 清空): 防恢复循环重放同一成交
        # 导致 on_trade/_process_order_groups 重复触发。
        self._seen_trade_ids: set[str] = set()
        self._payload_field = payload_field
        # 会话级委托状态指纹(**不随 drain 清空**): 防 recovery 每轮重放同一状态的
        # 挂单导致 on_order 每轮重推。trade 侧早有 _seen_trade_ids, order 侧此前
        # 没有等价物——这就是"挂单频繁触发 order"那条反馈的活跃根因。
        self._seen_order_states: dict[str, str] = {}
        self._order_state_fifo: deque[str] = deque()
        self._dropped_duplicate_orders = 0
        # 由 Task 3(标的过滤)递增, 本任务先初始化以让 dropped_event_counts() 可读。
        self._dropped_foreign_symbols = 0
        self._get_subscribed_symbols = get_subscribed_symbols
        # 已告警过的外来标的(防刷屏): 首次 WARNING 点名, 之后降 DEBUG。
        self._warned_foreign_symbols: set[str] = set()

    def mark_trades_seen(self, trade_ids: Iterable[str]) -> None:
        """把 trade_id 灌入会话级 dedup 基线.

        已烘进持仓快照的成交, 后续重放经 queue_event 丢弃。
        """
        with self._event_lock:
            for tid in trade_ids:
                if tid:
                    self._seen_trade_ids.add(str(tid))

    def discard_pending_trades(self) -> None:
        """丢弃队列中待派发的成交事件并标记其 trade_id 已见.

        激活时: 这些成交已烘进持仓快照, 不应再 apply_fill/重放 on_trade。
        仅动 trade 事件, order/account 保留。
        """
        with self._event_lock:
            kept: list = []
            for event_name, payload in self._event_store:
                if event_name == "trade":
                    raw = getattr(payload, "trade_id", None)
                    if raw is None and isinstance(payload, dict):
                        raw = payload.get("trade_id")
                    if raw:
                        self._seen_trade_ids.add(str(raw))
                    continue  # 丢弃该 trade 事件
                kept.append((event_name, payload))
            self._event_store[:] = kept

    def _order_state_fingerprint(self, payload: Any) -> str:
        """委托状态指纹: 状态+已成交量+均价+拒单原因, **刻意不含时间戳**.

        口径对齐回测侧 ``strategy_order_events.order_event_key``: 含时间戳的键
        每次重推都会变、去重完全失效(这类缺陷最常见的写法); 只按订单号去重又会
        把 ``New -> PartiallyFilled -> Filled`` 这些真实的状态推进整批吞掉。

        :param payload: 委托快照(``UnifiedOrderSnapshot`` 或等价 dict)。
        :return: 可比较的指纹字符串。
        """
        fields = ("status", "filled_quantity", "avg_fill_price", "reject_reason")
        return "|".join(str(self._payload_field(payload, field)) for field in fields)

    def dropped_event_counts(self) -> dict[str, int]:
        """会话级丢弃计数(去重/过滤分开计), 供收尾摘要与诊断读取.

        :return: ``{"duplicate_order": N, "foreign_symbol": M}``。
        """
        with self._event_lock:
            return {
                "duplicate_order": self._dropped_duplicate_orders,
                "foreign_symbol": self._dropped_foreign_symbols,
            }

    def _accepts_symbol(self, event_name: str, payload: Any) -> bool:
        """判断事件的标的是否属于本会话挂载标的.

        柜台的 ``sync_open_orders`` / ``sync_today_trades`` 返回的是**全账户**
        委托与成交, 不限于本会话订阅的标的, 不过滤会让策略收到不属于自己的
        委托回报。

        所有边界情况一律放行(无访问器 / 订阅集为空 / payload 无 symbol):
        吞掉真实回报的代价远大于多派发一条。``account`` 事件无标的概念, 直接放行。

        :param event_name: 事件名。
        :param payload: 事件载荷。
        :return: 是否应当入队派发。
        """
        if event_name == "account":
            return True
        if self._get_subscribed_symbols is None:
            return True
        allowed = self._get_subscribed_symbols()
        if not allowed:
            return True
        normalized = normalize_symbol_for_match(self._payload_field(payload, "symbol"))
        if not normalized:
            return True
        if normalized in allowed:
            return True
        self._log_foreign_symbol(event_name, normalized)
        return False

    def _log_foreign_symbol(self, event_name: str, symbol: str) -> None:
        """外来标的事件的丢弃留痕: 首次 WARNING 点名, 之后同标的降 DEBUG."""
        first_time = symbol not in self._warned_foreign_symbols
        self._warned_foreign_symbols.add(symbol)
        log = logger.warning if first_time else logger.debug
        log(
            "Dropped %s event for unsubscribed symbol %s",
            event_name,
            symbol,
            extra=build_log_extra(phase="gateway", symbol=symbol),
        )

    def queue_event(self, event_name: str, payload: Any) -> None:
        """Add a broker event to the dispatch queue with semantic deduplication."""
        if not self._accepts_symbol(event_name, payload):
            with self._event_lock:
                self._dropped_foreign_symbols += 1
            return
        event_key = self._make_event_key(event_name, payload)
        trade_id = ""
        if event_name == "trade":
            # getattr 优先、dict 兜底: 与 _make_event_key 的 _payload_field 语义一致,
            # 且对 slotted payload 稳健(_payload_to_dict 依赖 __dict__ 会漏)。
            raw = getattr(payload, "trade_id", None)
            if raw is None and isinstance(payload, dict):
                raw = payload.get("trade_id")
            trade_id = str(raw) if raw else ""
        state_key = ""
        state_fingerprint = ""
        if event_name in ("order", "execution_report"):
            order_id = str(self._payload_field(payload, "broker_order_id") or "")
            if order_id:
                # 键带事件类型: order 与 execution_report 是两类独立回调, 内置
                # broker(ctp/miniqmt/ptrade)对同一次状态变化会用同一 payload 成对
                # 派发 order + execution_report, 四个指纹字段逐字相同; 共用命名
                # 空间会让第二个事件被误判"已派发过"而永久吞掉。
                state_key = f"{event_name}:{order_id}"
                state_fingerprint = self._order_state_fingerprint(payload)
        with self._event_lock:
            if trade_id:
                if trade_id in self._seen_trade_ids:
                    return  # 会话级: 该成交已入队(实盘推送/恢复重放), 丢弃
                self._seen_trade_ids.add(trade_id)
            if state_key:
                if self._seen_order_states.get(state_key) == state_fingerprint:
                    self._dropped_duplicate_orders += 1
                    return  # 会话级: 该委托的这个状态已派发过
                if state_key not in self._seen_order_states:
                    self._order_state_fifo.append(state_key)
                    while len(self._order_state_fifo) > _ORDER_STATE_DEDUPE_LIMIT:
                        stale = self._order_state_fifo.popleft()
                        self._seen_order_states.pop(stale, None)
                self._seen_order_states[state_key] = state_fingerprint
            if event_key in self._event_keys:
                return
            self._event_keys.add(event_key)
            self._event_store.append((event_name, payload))

    def drain_events(self, strategy: Any) -> None:
        """Drain queued broker events, update state and dispatch callbacks."""
        with self._event_lock:
            events = list(self._event_store)
            self._event_store.clear()
            self._event_keys.clear()
        for event_name, payload in events:
            adapted = self._adapt_strategy_payload(event_name, payload)
            self._update_broker_state(event_name, payload)
            self._emit_observer_event(event_name, payload)
            record_broker_event(
                event_name,
                payload,
                owner_strategy_id=self._resolve_owner_strategy_id(payload),
                trace_id=(
                    self._resolve_trace_id(payload)
                    if self._resolve_trace_id is not None
                    else None
                ),
            )
            self._dispatch_strategy_event(strategy, event_name, adapted)

    def emit_observer_event(self, event_name: str, payload: Any) -> None:
        """Emit a normalized event snapshot to the optional observer hook."""
        self._emit_observer_event(event_name, payload)

    def _emit_observer_event(self, event_name: str, payload: Any) -> None:
        on_broker_event = self._get_on_broker_event()
        if on_broker_event is None:
            return
        owner_strategy_id = self._resolve_owner_strategy_id(payload)
        payload_dict = self._payload_to_dict(payload)
        try:
            on_broker_event(
                {
                    "event_type": event_name,
                    "owner_strategy_id": owner_strategy_id,
                    "payload": payload_dict,
                }
            )
        except Exception as exc:
            logger.warning(
                "Broker event observer failed",
                exc_info=exc,
                extra=build_log_extra(
                    phase="gateway",
                    strategy_id=owner_strategy_id,
                    slot=owner_strategy_id if owner_strategy_id != "_default" else None,
                    symbol=str(payload_dict.get("symbol", "") or "").strip() or None,
                    order_id=payload_dict.get("broker_order_id")
                    or payload_dict.get("order_id"),
                    client_order_id=payload_dict.get("client_order_id"),
                ),
            )

    def _dispatch_strategy_event(
        self,
        strategy: Any,
        event_name: str,
        payload: Any,
    ) -> None:
        # `payload` is already adapted by `drain_events` (via `_adapt_strategy_payload`)
        # while the request cache was still populated, so it's dispatched as-is here.
        if event_name == "order":
            self._safe_strategy_callback(strategy, "on_order", payload)
        elif event_name == "trade":
            self._safe_strategy_callback(strategy, "on_trade", payload)
            # broker_live 下由真实成交驱动 OCO/Bracket 协调(回测由引擎驱动);
            # 经 _safe_strategy_callback 异常隔离, 无组时 no-op。
            self._safe_strategy_callback(strategy, "_process_order_groups", payload)
        elif event_name == "execution_report":
            self._safe_strategy_callback(strategy, "on_execution_report", payload)
        elif event_name == "account":
            self._safe_strategy_callback(strategy, "on_portfolio_update", payload)
