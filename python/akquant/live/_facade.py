"""Public ``run_live`` function facade, symmetric with ``run_backtest``.

``run_live`` mirrors the ergonomics of ``run_backtest``: a single top-level
function call instead of constructing a runner object. It forwards every
configuration argument to the underlying live session and starts it, and adds
the ``indicator_recorder`` / ``on_event`` extension points so live runs can
stream indicators exactly like backtests.
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union

from ..akquant import Bar, Instrument
from ..backtest import BacktestStreamEvent
from ..indicator_recording import IndicatorSink
from ..strategy import Strategy, StrategyRuntimeConfig
from ._runner import LiveRunner


def run_live(
    strategy_cls: Optional[
        Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]
    ] = None,
    instruments: Optional[List[Instrument]] = None,
    *,
    cash: float = 1_000_000.0,
    show_progress: bool = False,
    duration: Optional[str] = None,
    strategy_source: Optional[Union[str, bytes]] = None,
    strategy_loader: Optional[str] = None,
    strategy_loader_options: Optional[Dict[str, Any]] = None,
    strategy_id: Optional[str] = None,
    strategies_by_slot: Optional[
        Dict[str, Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]
    ] = None,
    md_front: str = "",
    td_front: Optional[str] = None,
    broker_id: str = "",
    user_id: str = "",
    password: str = "",
    app_id: str = "",
    auth_code: str = "",
    use_aggregator: bool = True,
    broker: str = "ctp",
    market_broker: Optional[str] = None,
    trader_broker: Optional[str] = None,
    trading_mode: str = "paper",
    gateway_options: Optional[Dict[str, Any]] = None,
    initialize: Optional[Callable[[Any], None]] = None,
    on_start: Optional[Callable[[Any], None]] = None,
    on_stop: Optional[Callable[[Any], None]] = None,
    on_tick: Optional[Callable[[Any, Any], None]] = None,
    on_order: Optional[Callable[[Any, Any], None]] = None,
    on_trade: Optional[Callable[[Any, Any], None]] = None,
    on_reject: Optional[Callable[[Any, Any], None]] = None,
    on_broker_connected: Optional[Callable[[Any], None]] = None,
    broker_ready_timeout: float = 30.0,
    broker_ready_required: bool = True,
    on_before_trading: Optional[Callable[[Any, Any, int], None]] = None,
    on_after_trading: Optional[Callable[[Any, Any, int], None]] = None,
    on_cross_section: Optional[Callable[[Any, Any, int], None]] = None,
    on_portfolio_update: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
    on_error: Optional[Callable[[Any, Exception, str, Any], None]] = None,
    on_timer: Optional[Callable[[Any, str], None]] = None,
    context: Optional[Dict[str, Any]] = None,
    strategy_max_order_value: Optional[Dict[str, float]] = None,
    strategy_max_order_size: Optional[Dict[str, float]] = None,
    strategy_max_position_size: Optional[Dict[str, float]] = None,
    strategy_max_daily_loss: Optional[Dict[str, float]] = None,
    strategy_max_drawdown: Optional[Dict[str, float]] = None,
    strategy_reduce_only_after_risk: Optional[Dict[str, bool]] = None,
    strategy_risk_cooldown_bars: Optional[Dict[str, int]] = None,
    strategy_priority: Optional[Dict[str, int]] = None,
    strategy_risk_budget: Optional[Dict[str, float]] = None,
    portfolio_risk_budget: Optional[float] = None,
    risk_budget_mode: str = "order_notional",
    risk_budget_reset_daily: bool = False,
    on_broker_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    signal_port_ready: Optional[Callable[[Any], None]] = None,
    signal_source: Optional[Any] = None,
    indicator_recorder: Optional[IndicatorSink] = None,
    on_event: Optional[Callable[[BacktestStreamEvent], None]] = None,
    strategy_runtime_config: Optional[
        Union[StrategyRuntimeConfig, Dict[str, Any]]
    ] = None,
    runtime_config_override: bool = True,
    session_tag: Optional[str] = None,
) -> None:
    """Run a live/paper trading session, symmetric with ``run_backtest``.

    :param strategy_cls: Strategy class/instance, or function-style on_bar callback.
    :param instruments: Instruments to trade.
    :param cash: Initial cash (default 1,000,000).
    :param show_progress: Whether to show a progress bar.
    :param duration: Optional run duration (e.g. ``"1m"``, ``"30s"``).
    :param broker: Broker supplying **both** market data and trading (default
        ``"ctp"``). Leave ``market_broker`` / ``trader_broker`` unset to use it.
    :param market_broker: Market-data source, overriding ``broker`` for that side.
        Must be given together with ``trader_broker``; passing only one raises
        ``ValueError``. When both are given, ``broker`` is not used at all.
    :param trader_broker: Trading source, the counterpart of ``market_broker``.
        Use this pair to combine two single-sided brokers — ``replay`` only
        provides market data (cannot place orders), while some broker plugins
        only provide a trading channel (no market data, so ``on_bar`` /
        ``on_tick`` never fire and ``current_tick`` stays ``None``)::

            run_live(..., market_broker="replay", trader_broker="qmf")
    :param broker_ready_timeout: Max seconds to poll ``trader_gateway.heartbeat()``
        before giving up (default 30.0).
    :param broker_ready_required: If True (default), abort the run when the broker
        is not ready within ``broker_ready_timeout``. Only polled in
        ``trading_mode='broker_live'``; paper runs are unaffected. Pass False to
        restore the pre-0.3.25 behaviour of warning and continuing.
    :param signal_port_ready: Optional callback that receives a ``SignalPort`` just
        before the engine loop starts. Hand the port to an external signal source
        (HTTP webhook / MQ consumer) running on its own thread to inject orders
        without going through strategy callbacks::

            def bind(port):
                threading.Thread(target=consume, args=(port,), daemon=True).start()

            run_live(..., trading_mode="paper", signal_port_ready=bind)

        Only meaningful for ``trading_mode='paper'``: under ``broker_live`` the
        engine's realtime executor does not forward orders to the broker, so
        injected orders pass risk checks and sit in active orders forever. A
        warning is logged if used there.

        **The callback runs synchronously right before the engine loop starts.**
        Once it returns, the main thread enters the Rust loop and holds the GIL
        for long stretches. If you spawn a thread here, wait until it has
        actually started before returning — otherwise it may never get
        scheduled::

            def bind(port):
                running = threading.Event()

                def worker():
                    running.set()
                    ...  # port.submit(...)

                threading.Thread(target=worker, daemon=True).start()
                running.wait(timeout=5.0)   # 关键: 确认线程已就绪

        Also note an injected order only fills when a **subsequent** market
        event arrives — an order injected after the last bar stays ``New``.
    :param signal_source: Optional :class:`~akquant.signal.SignalSource` feeding
        external trading instructions into the session. Its lifecycle is managed
        here: ``bind`` → ``start`` (before the engine loop) → ``stop`` (on exit).
        Orders are created directly from signals, **not** through strategy
        callbacks::

            from akquant.signal import QueueSignalSource, Signal

            source = QueueSignalSource()
            source.put(Signal(signal_id="s-1", symbol="000001.SZ",
                              action="buy", quantity=100, price=10.5))
            run_live(..., trading_mode="paper", signal_source=source)

        Risk coverage differs by mode and this is an engine-architecture fact,
        not a setting: ``paper`` injects through the engine event channel and
        gets the **full** risk pipeline; ``broker_live`` goes through the broker
        channel where only the three strategy-level limits (order value / order
        size / position size) are enforced up front. If the source fails to
        start the session aborts — it is the only order origin, so continuing
        would silently produce a session that never trades.
    :param indicator_recorder: Optional public :class:`IndicatorSink` to collect
        indicator points; defaults to a streaming sink when ``on_event`` is set.
    :param on_event: Optional stream event callback for realtime indicator flow.
    :param strategy_runtime_config: Runtime behaviour config (object or dict),
        dispatched to the primary strategy **and** every slot strategy, exactly as
        ``run_backtest`` does. Lets you flip behaviour from the entry point without
        editing strategy code — e.g. run the same strategy with
        ``error_mode="raise"`` in backtests to surface bugs, and
        ``error_mode="continue"`` live so a callback error does not halt trading.
        Left unset, whatever the strategy assigned to ``self.runtime_config``
        stands untouched.
    :param runtime_config_override: Whether to override a ``runtime_config`` the
        strategy already set itself (default True). Either way a conflict is
        logged; False keeps the strategy's own value and only warns.
    :param session_tag: Optional session marker embedded in every
        ``client_order_id`` this session generates (``{broker}-{session_tag}-{seq}``),
        replacing the default random uuid. **Recommended: pass your platform's
        task ID.** Enables per-task order-push isolation when multiple tasks
        trade the **same** symbol under the same account — pushes whose
        ``client_order_id`` carries a different session tag are actively
        rejected (counted as ``foreign_task``, separate from the pre-existing
        ``foreign_symbol`` filter), instead of only being filtered by
        subscribed symbol.

        **Strict rejection only activates when this parameter is explicitly
        passed.** Leaving it unset keeps behaviour identical to prior
        releases: the session still gets a random tag for
        ``client_order_id`` uniqueness across restarts, but a mismatched
        prefix is never treated as "not mine" — filtering still falls back
        to the existing order-ownership / subscribed-symbol checks. This is
        deliberate: if the broker truncates or rewrites ``client_order_id``
        (length limits, character filtering), this session's own prefix
        would also stop matching, and unconditional rejection would drop
        every one of its own callbacks — orders would appear to submit
        successfully yet no callback would ever arrive. Only enable this
        once you've confirmed your broker echoes ``client_order_id`` back
        unmodified in push frames (check by comparing a live
        ``client_order_id`` against this session's prefix in the logs).

        Validation is fail-fast, not normalized: ``session_tag`` must match
        ``[A-Za-z0-9_]+`` (no ``-``, which is the ``client_order_id`` field
        separator) and be at most 32 characters, or ``ValueError`` is raised
        with a ready-to-use suggestion. Values are not silently rewritten —
        that would let the platform believe isolation is active when it
        silently is not.
    :return: ``None``.
    """
    if instruments is None:
        raise ValueError("run_live requires instruments")

    runner = LiveRunner(
        strategy_cls=strategy_cls,
        instruments=instruments,
        strategy_source=strategy_source,
        strategy_loader=strategy_loader,
        strategy_loader_options=strategy_loader_options,
        strategy_id=strategy_id,
        strategies_by_slot=strategies_by_slot,
        md_front=md_front,
        td_front=td_front,
        broker_id=broker_id,
        user_id=user_id,
        password=password,
        app_id=app_id,
        auth_code=auth_code,
        use_aggregator=use_aggregator,
        broker=broker,
        market_broker=market_broker,
        trader_broker=trader_broker,
        trading_mode=trading_mode,
        gateway_options=gateway_options,
        initialize=initialize,
        on_start=on_start,
        on_stop=on_stop,
        on_tick=on_tick,
        on_order=on_order,
        on_trade=on_trade,
        on_reject=on_reject,
        on_broker_connected=on_broker_connected,
        broker_ready_timeout=broker_ready_timeout,
        broker_ready_required=broker_ready_required,
        on_before_trading=on_before_trading,
        on_after_trading=on_after_trading,
        on_cross_section=on_cross_section,
        on_portfolio_update=on_portfolio_update,
        on_error=on_error,
        on_timer=on_timer,
        context=context,
        strategy_max_order_value=strategy_max_order_value,
        strategy_max_order_size=strategy_max_order_size,
        strategy_max_position_size=strategy_max_position_size,
        strategy_max_daily_loss=strategy_max_daily_loss,
        strategy_max_drawdown=strategy_max_drawdown,
        strategy_reduce_only_after_risk=strategy_reduce_only_after_risk,
        strategy_risk_cooldown_bars=strategy_risk_cooldown_bars,
        strategy_priority=strategy_priority,
        strategy_risk_budget=strategy_risk_budget,
        portfolio_risk_budget=portfolio_risk_budget,
        risk_budget_mode=risk_budget_mode,
        risk_budget_reset_daily=risk_budget_reset_daily,
        on_broker_event=on_broker_event,
        signal_port_ready=signal_port_ready,
        signal_source=signal_source,
        strategy_runtime_config=strategy_runtime_config,
        runtime_config_override=runtime_config_override,
        session_tag=session_tag,
    )
    runner.set_indicator_stream(
        indicator_recorder=indicator_recorder, on_event=on_event
    )
    runner.run(cash=cash, show_progress=show_progress, duration=duration)
