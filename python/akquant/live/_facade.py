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
from ..strategy import Strategy
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
    indicator_recorder: Optional[IndicatorSink] = None,
    on_event: Optional[Callable[[BacktestStreamEvent], None]] = None,
) -> None:
    """Run a live/paper trading session, symmetric with ``run_backtest``.

    :param strategy_cls: Strategy class/instance, or function-style on_bar callback.
    :param instruments: Instruments to trade.
    :param cash: Initial cash (default 1,000,000).
    :param show_progress: Whether to show a progress bar.
    :param duration: Optional run duration (e.g. ``"1m"``, ``"30s"``).
    :param broker_ready_timeout: Max seconds to poll ``trader_gateway.heartbeat()``
        before giving up (default 30.0).
    :param broker_ready_required: If True (default), abort the run when the broker
        is not ready within ``broker_ready_timeout``. Only polled in
        ``trading_mode='broker_live'``; paper runs are unaffected. Pass False to
        restore the pre-0.3.25 behaviour of warning and continuing.
    :param indicator_recorder: Optional public :class:`IndicatorSink` to collect
        indicator points; defaults to a streaming sink when ``on_event`` is set.
    :param on_event: Optional stream event callback for realtime indicator flow.
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
    )
    runner.set_indicator_stream(
        indicator_recorder=indicator_recorder, on_event=on_event
    )
    runner.run(cash=cash, show_progress=show_progress, duration=duration)
