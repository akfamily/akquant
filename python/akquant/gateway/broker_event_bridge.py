from typing import Any, Callable

from ..log import build_log_extra, get_logger

logger = get_logger("gateway.live")


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
    ) -> None:
        """Bind the queue, state callbacks and observer fanout dependencies."""
        self._event_lock = event_lock
        self._event_store = event_store
        self._event_keys = event_keys
        self._get_on_broker_event = get_on_broker_event
        self._make_event_key = make_event_key
        self._update_broker_state = update_broker_state
        self._resolve_owner_strategy_id = resolve_owner_strategy_id
        self._payload_to_dict = payload_to_dict
        self._safe_strategy_callback = safe_strategy_callback
        self._adapt_strategy_payload = adapt_strategy_payload
        # 会话级已入队 trade_id(不随 drain 清空): 防恢复循环重放同一成交
        # 导致 on_trade/_process_order_groups 重复触发。
        self._seen_trade_ids: set[str] = set()

    def queue_event(self, event_name: str, payload: Any) -> None:
        """Add a broker event to the dispatch queue with semantic deduplication."""
        event_key = self._make_event_key(event_name, payload)
        trade_id = ""
        if event_name == "trade":
            # getattr 优先、dict 兜底: 与 _make_event_key 的 _payload_field 语义一致,
            # 且对 slotted payload 稳健(_payload_to_dict 依赖 __dict__ 会漏)。
            raw = getattr(payload, "trade_id", None)
            if raw is None and isinstance(payload, dict):
                raw = payload.get("trade_id")
            trade_id = str(raw) if raw else ""
        with self._event_lock:
            if trade_id:
                if trade_id in self._seen_trade_ids:
                    return  # 会话级: 该成交已入队(实盘推送/恢复重放), 丢弃
                self._seen_trade_ids.add(trade_id)
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
