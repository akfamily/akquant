from typing import Any, Callable

from .broker_event_bridge import BrokerEventBridge
from .broker_models import BrokerCapability
from .broker_recovery import BrokerRecovery
from .broker_strategy_api import wrap_state_invalidation
from .order_submitter import BrokerOrderSubmitter


class BrokerRuntime:
    """Coordinate broker-live submitter, event bridge and recovery helpers."""

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
        get_trader_gateway: Callable[[], Any],
        notify_strategy_error: Callable[[Any, Exception, str, Any], None],
        get_recovery_mode: Callable[[], str],
        get_last_error_key: Callable[[], str],
        set_last_error_key: Callable[[str], None],
        resolve_trader_capabilities: Callable[[Any], BrokerCapability],
        next_client_order_id: Callable[[], str],
        can_submit_client_order: Callable[[str], bool],
        sync_order_id_mapping: Callable[[str, str], None],
        bind_order_owner: Callable[[str, str, str], None],
        payload_field: Callable[[Any, str], Any],
        get_execution_capabilities: Callable[[], dict[str, Any]],
        record_order_request: Callable[[str, Any], None],
        adapt_strategy_payload: Callable[[str, Any], Any],
        record_stop_remap: Any = None,
        should_replay_trades: Callable[[], bool] | None = None,
        sync_group_mapping: Callable[[str, str], None] = lambda _c, _g: None,
        group_broker_ids: Callable[[str], list[str]] | None = None,
        resolve_trace_id: Callable[[Any], str] | None = None,
        get_subscribed_symbols: Callable[[], set[str]] | None = None,
        is_known_order: Callable[[str, str], bool] | None = None,
    ) -> None:
        """Assemble broker submitter, event bridge and recovery coordinators."""
        self._broker_state_caches: list[Any] = []
        update_broker_state = wrap_state_invalidation(
            update_broker_state, lambda: self._broker_state_caches
        )
        self._event_bridge = BrokerEventBridge(
            event_lock=event_lock,
            event_store=event_store,
            event_keys=event_keys,
            get_on_broker_event=get_on_broker_event,
            make_event_key=make_event_key,
            update_broker_state=update_broker_state,
            resolve_owner_strategy_id=resolve_owner_strategy_id,
            payload_to_dict=payload_to_dict,
            safe_strategy_callback=safe_strategy_callback,
            adapt_strategy_payload=adapt_strategy_payload,
            resolve_trace_id=resolve_trace_id,
            payload_field=payload_field,
            get_subscribed_symbols=get_subscribed_symbols,
            is_known_order=is_known_order,
        )
        self._recovery = BrokerRecovery(
            get_trader_gateway=get_trader_gateway,
            queue_broker_event=self._event_bridge.queue_event,
            notify_strategy_error=notify_strategy_error,
            get_on_broker_event=get_on_broker_event,
            get_recovery_mode=get_recovery_mode,
            get_last_error_key=get_last_error_key,
            set_last_error_key=set_last_error_key,
            should_replay_trades=should_replay_trades,
        )
        self._resolve_trader_capabilities = resolve_trader_capabilities
        self._next_client_order_id = next_client_order_id
        self._can_submit_client_order = can_submit_client_order
        self._sync_order_id_mapping = sync_order_id_mapping
        self._bind_order_owner = bind_order_owner
        self._notify_strategy_error = notify_strategy_error
        self._payload_field = payload_field
        self._get_execution_capabilities = get_execution_capabilities
        self._record_order_request = record_order_request
        self._record_stop_remap = record_stop_remap
        self._sync_group_mapping = sync_group_mapping
        self._group_broker_ids = group_broker_ids
        self._submitter: BrokerOrderSubmitter | None = None

    @property
    def event_bridge(self) -> BrokerEventBridge:
        """Expose the broker event bridge used by the runtime."""
        return self._event_bridge

    @property
    def recovery(self) -> BrokerRecovery:
        """Expose the broker recovery helper used by the runtime."""
        return self._recovery

    @property
    def submitter(self) -> BrokerOrderSubmitter | None:
        """Return the installed submitter, if broker live submit is enabled."""
        return self._submitter

    @property
    def state_caches(self) -> list[Any]:
        """Expose per-slot BrokerStateCache list (启动激活 seed 用)."""
        return self._broker_state_caches

    def install_submitter(
        self,
        trader_gateway: Any,
        strategy: Any,
        strategy_limits: dict[str, dict[str, float]] | None = None,
    ) -> BrokerOrderSubmitter:
        """Create and install the strategy-facing broker submitter.

        ``strategy_limits`` 透传给 submitter 做报单前的策略级限额风控;
        省略则不做该校验(仅测试桩)。
        """
        self._submitter = BrokerOrderSubmitter(
            trader_gateway=trader_gateway,
            strategy=strategy,
            strategy_limits=strategy_limits,
            resolve_trader_capabilities=self._resolve_trader_capabilities,
            next_client_order_id=self._next_client_order_id,
            can_submit_client_order=self._can_submit_client_order,
            sync_order_id_mapping=self._sync_order_id_mapping,
            bind_order_owner=self._bind_order_owner,
            notify_strategy_error=self._notify_strategy_error,
            payload_field=self._payload_field,
            get_execution_capabilities=self._get_execution_capabilities,
            record_order_request=self._record_order_request,
            sync_group_mapping=self._sync_group_mapping,
        )
        self._submitter.install()

        from .broker_execution import BrokerExecution
        from .broker_state_cache import BrokerStateCache

        # One cache per strategy target; a fill/order push must invalidate all of
        # them (single-field storage would only invalidate the last slot's cache).
        cache = BrokerStateCache(trader_gateway)
        self._broker_state_caches.append(cache)
        strategy.execution = BrokerExecution(
            strategy,
            trader_gateway,
            cache,
            self._submitter,
            record_stop_remap=self._record_stop_remap,
            group_broker_ids=self._group_broker_ids,
        )

        return self._submitter

    def queue_event(self, event_name: str, payload: Any) -> None:
        """Queue a broker event through the runtime-owned event bridge."""
        self._event_bridge.queue_event(event_name, payload)

    def drain_events(self, strategy: Any) -> None:
        """Drain queued broker events and dispatch them to the strategy."""
        self._event_bridge.drain_events(strategy)

    def run_recovery_cycle(
        self,
        strategy: Any | None = None,
        handle_error: Callable[[Any | None, str, Exception, dict[str, Any]], None]
        | None = None,
        *,
        sync_orders: bool = True,
        sync_trades: bool = True,
        refresh_account: bool = True,
    ) -> bool:
        """Run one recovery cycle through the runtime-owned recovery helper."""
        return self._recovery.run_cycle(
            strategy,
            handle_error=handle_error,
            sync_orders=sync_orders,
            sync_trades=sync_trades,
            refresh_account=refresh_account,
        )

    def handle_recovery_error(
        self,
        strategy: Any | None,
        source: str,
        error: Exception,
        payload: dict[str, Any],
    ) -> None:
        """Delegate recovery error handling to the recovery helper."""
        self._recovery.handle_error(strategy, source, error, payload)
