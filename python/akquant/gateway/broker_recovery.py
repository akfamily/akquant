from typing import Any, Callable


class BrokerRecovery:
    """Run broker recovery cycles and surface strict-mode recovery errors."""

    def __init__(
        self,
        *,
        get_trader_gateway: Callable[[], Any],
        queue_broker_event: Callable[[str, Any], None],
        notify_strategy_error: Callable[[Any, Exception, str, Any], None],
        get_on_broker_event: Callable[[], Callable[[dict[str, Any]], None] | None],
        get_recovery_mode: Callable[[], str],
        get_last_error_key: Callable[[], str],
        set_last_error_key: Callable[[str], None],
        should_replay_trades: Callable[[], bool] | None = None,
    ) -> None:
        """Bind gateway accessors and strict-mode error reporting hooks."""
        self._get_trader_gateway = get_trader_gateway
        self._queue_broker_event = queue_broker_event
        self._notify_strategy_error = notify_strategy_error
        self._get_on_broker_event = get_on_broker_event
        self._get_recovery_mode = get_recovery_mode
        self._get_last_error_key = get_last_error_key
        self._set_last_error_key = set_last_error_key
        self._should_replay_trades = should_replay_trades

    def run_cycle(
        self,
        strategy: Any | None = None,
        handle_error: Callable[[Any | None, str, Exception, dict[str, Any]], None]
        | None = None,
        *,
        sync_orders: bool = True,
        sync_trades: bool = True,
        refresh_account: bool = True,
    ) -> bool:
        """Run one broker recovery cycle with per-stage cadence control.

        心跳每拍都跑(便宜, 要快速发现断线); 全量 ``sync_*`` 与 ``query_account``
        由调用方按各自节奏决定是否本轮执行——每秒全量轮询柜台会把接口打爆。

        **重连成功后强制补齐**: 无论开关如何, 刚重连的那一轮一定跑全量 sync,
        这才是"断线补齐"真正该触发的时机。

        :return: 本轮是否发生过重连(调用方据此重置兜底计时)。
        """
        reconnected = False
        gateway = self._get_trader_gateway()
        if gateway is None:
            return reconnected
        heartbeat = getattr(gateway, "heartbeat", None)
        if callable(heartbeat):
            try:
                alive = heartbeat()
            except Exception as exc:
                self._dispatch_recovery_error(
                    strategy,
                    "broker_recovery.heartbeat",
                    exc,
                    {"stage": "heartbeat"},
                    handle_error,
                )
                return reconnected
            if not alive:
                connect = getattr(gateway, "connect", None)
                if callable(connect):
                    try:
                        connect()
                    except Exception as exc:
                        self._dispatch_recovery_error(
                            strategy,
                            "broker_recovery.connect",
                            exc,
                            {"stage": "connect"},
                            handle_error,
                        )
                        return reconnected
                    reconnected = True
        sync_open_orders = getattr(gateway, "sync_open_orders", None)
        if callable(sync_open_orders) and (sync_orders or reconnected):
            try:
                for order in sync_open_orders():
                    self._queue_broker_event("order", order)
            except Exception as exc:
                self._dispatch_recovery_error(
                    strategy,
                    "broker_recovery.sync_open_orders",
                    exc,
                    {"stage": "sync_open_orders"},
                    handle_error,
                )
                if self._get_recovery_mode() == "strict":
                    return reconnected
        sync_today_trades = getattr(gateway, "sync_today_trades", None)
        replay_ok = self._should_replay_trades is None or self._should_replay_trades()
        if callable(sync_today_trades) and replay_ok and (sync_trades or reconnected):
            try:
                for trade in sync_today_trades():
                    self._queue_broker_event("trade", trade)
            except Exception as exc:
                self._dispatch_recovery_error(
                    strategy,
                    "broker_recovery.sync_today_trades",
                    exc,
                    {"stage": "sync_today_trades"},
                    handle_error,
                )
                if self._get_recovery_mode() == "strict":
                    return reconnected
        query_account = getattr(gateway, "query_account", None)
        if callable(query_account) and refresh_account:
            try:
                account = query_account()
                if account is not None:
                    self._queue_broker_event("account", account)
            except Exception as exc:
                self._dispatch_recovery_error(
                    strategy,
                    "broker_recovery.query_account",
                    exc,
                    {"stage": "query_account"},
                    handle_error,
                )
                if self._get_recovery_mode() == "strict":
                    return reconnected
        self._set_last_error_key("")
        return reconnected

    def handle_error(
        self,
        strategy: Any | None,
        source: str,
        error: Exception,
        payload: dict[str, Any],
    ) -> None:
        """Surface broker recovery errors only in strict mode with deduplication."""
        if self._get_recovery_mode() != "strict":
            return
        error_key = f"{source}:{type(error).__name__}:{error}"
        if error_key == self._get_last_error_key():
            return
        self._set_last_error_key(error_key)
        if strategy is not None:
            self._notify_strategy_error(strategy, error, source, payload)
        on_broker_event = self._get_on_broker_event()
        if on_broker_event is not None:
            try:
                on_broker_event(
                    {
                        "event_type": "recovery_error",
                        "owner_strategy_id": "_default",
                        "payload": {
                            **dict(payload),
                            "source": source,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        },
                    }
                )
            except Exception:
                pass

    def _dispatch_recovery_error(
        self,
        strategy: Any | None,
        source: str,
        error: Exception,
        payload: dict[str, Any],
        handle_error: Callable[[Any | None, str, Exception, dict[str, Any]], None]
        | None,
    ) -> None:
        if handle_error is not None:
            handle_error(strategy, source, error, payload)
            return
        self.handle_error(strategy, source, error, payload)
