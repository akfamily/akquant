# -*- coding: utf-8 -*-
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

from .akquant import Bar, DataFeed, Engine, Instrument
from .gateway.broker_runtime import BrokerRuntime
from .gateway.factory import create_gateway_bundle
from .gateway.models import BrokerCapability
from .gateway.order_submitter import (
    build_live_order_client_ids,
    find_live_close_position,
    resolve_live_order_legs,
    validate_live_order_client_ids,
)
from .log import build_log_extra, get_logger
from .strategy import Strategy
from .strategy_loader import resolve_strategy_input
from .utils import format_metric_value

logger = get_logger("gateway.live")

_LEGACY_GATEWAY_OPTION_KEYS = (
    "md_front",
    "td_front",
    "broker_id",
    "user_id",
    "password",
    "app_id",
    "auth_code",
)


class _StrategyCallbackFanout:
    def __init__(self, strategies: List[Strategy]):
        self._strategies = strategies

    def on_order(self, order: Any) -> None:
        for strategy in self._strategies:
            callback = getattr(strategy, "on_order", None)
            if callback is None:
                continue
            try:
                callback(order)
            except Exception as exc:
                on_error = getattr(strategy, "on_error", None)
                if on_error is not None:
                    try:
                        on_error(exc, "on_order", order)
                    except Exception:
                        pass

    def on_trade(self, trade: Any) -> None:
        for strategy in self._strategies:
            callback = getattr(strategy, "on_trade", None)
            if callback is None:
                continue
            try:
                callback(trade)
            except Exception as exc:
                on_error = getattr(strategy, "on_error", None)
                if on_error is not None:
                    try:
                        on_error(exc, "on_trade", trade)
                    except Exception:
                        pass

    def on_execution_report(self, report: Any) -> None:
        for strategy in self._strategies:
            callback = getattr(strategy, "on_execution_report", None)
            if callback is None:
                continue
            try:
                callback(report)
            except Exception as exc:
                on_error = getattr(strategy, "on_error", None)
                if on_error is not None:
                    try:
                        on_error(exc, "on_execution_report", report)
                    except Exception:
                        pass


class LiveRunner:
    """
    Live/Paper Trading Runner.

    Encapsulates the boilerplate code for setting up the engine, data feed,
    instruments, and gateways for live or paper trading.
    """

    def __init__(
        self,
        strategy_cls: Optional[
            Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]
        ],
        instruments: List[Instrument],
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
        trading_mode: str = "paper",
        gateway_options: Optional[Dict[str, Any]] = None,
        initialize: Optional[Callable[[Any], None]] = None,
        on_start: Optional[Callable[[Any], None]] = None,
        on_stop: Optional[Callable[[Any], None]] = None,
        on_tick: Optional[Callable[[Any, Any], None]] = None,
        on_order: Optional[Callable[[Any, Any], None]] = None,
        on_trade: Optional[Callable[[Any, Any], None]] = None,
        on_reject: Optional[Callable[[Any, Any], None]] = None,
        on_session_start: Optional[Callable[[Any, Any, int], None]] = None,
        on_broker_connected: Optional[Callable[[Any], None]] = None,
        broker_ready_timeout: float = 10.0,
        broker_ready_required: bool = False,
        on_session_end: Optional[Callable[[Any, Any, int], None]] = None,
        on_before_trading: Optional[Callable[[Any, Any, int], None]] = None,
        on_after_trading: Optional[Callable[[Any, Any, int], None]] = None,
        on_daily_rebalance: Optional[Callable[[Any, Any, int], None]] = None,
        on_daily_rebalance_after_bar: Optional[Callable[[Any, Any, int], None]] = None,
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
    ):
        """
        Initialize the LiveRunner.

        :param strategy_cls: Strategy class/instance, or function-style on_bar callback.
        :param strategy_id: Primary strategy id for slot ownership (default "_default").
        :param strategies_by_slot: Optional slot->strategy mapping
                                   for multi-slot runtime.
        :param instruments: List of instruments to trade.
        :param md_front: CTP Market Data Front URL.
        :param td_front: CTP Trade Front URL (optional).
        :param broker_id: CTP Broker ID (optional).
        :param user_id: CTP User ID (optional).
        :param password: CTP Password (optional).
        :param app_id: CTP App ID (optional).
        :param auth_code: CTP Auth Code (optional).
        :param use_aggregator: Whether to use BarAggregator (default True).
        :param initialize: Optional function-style initialize callback.
        :param on_start: Optional function-style on_start callback.
        :param on_stop: Optional function-style on_stop callback.
        :param on_tick: Optional function-style on_tick callback.
        :param on_order: Optional function-style on_order callback.
        :param on_trade: Optional function-style on_trade callback.
        :param on_reject: Optional function-style on_reject callback.
        :param on_session_start: Optional function-style on_session_start callback.
        :param on_broker_connected: Optional callback fired once the broker
            trader gateway reports readiness (heartbeat) after connect+start.
        :param broker_ready_timeout: Max seconds to poll trader_gateway.heartbeat()
            before giving up (default 10.0).
        :param broker_ready_required: If True, raise when broker readiness is not
            reached within broker_ready_timeout (default False, only warns).
        :param on_session_end: Optional function-style on_session_end callback.
        :param on_before_trading: Optional function-style on_before_trading callback.
        :param on_after_trading: Optional function-style on_after_trading callback.
        :param on_daily_rebalance: Optional function-style on_daily_rebalance callback.
        :param on_daily_rebalance_after_bar:
            Optional function-style on_daily_rebalance_after_bar callback.
        :param on_portfolio_update:
            Optional function-style on_portfolio_update callback.
        :param on_error: Optional function-style on_error callback.
        :param on_timer: Optional function-style on_timer callback.
        :param context: Optional context dict injected into function-style strategy.
        :param on_broker_event: Optional broker event observer callback.
        """
        self.strategy_cls = strategy_cls
        self.strategy_source = strategy_source
        self.strategy_loader = strategy_loader
        self.strategy_loader_options = strategy_loader_options
        self.strategy_id = (strategy_id or "_default").strip() or "_default"
        self.strategies_by_slot = strategies_by_slot or {}
        self.instruments = instruments
        self.gateway_options = self._normalize_gateway_options(
            gateway_options=gateway_options,
            md_front=md_front,
            td_front=td_front,
            broker_id=broker_id,
            user_id=user_id,
            password=password,
            app_id=app_id,
            auth_code=auth_code,
        )
        self.md_front = cast(str, self.gateway_options.get("md_front", ""))
        self.td_front = cast(Optional[str], self.gateway_options.get("td_front"))
        self.broker_id = cast(str, self.gateway_options.get("broker_id", ""))
        self.user_id = cast(str, self.gateway_options.get("user_id", ""))
        self.password = cast(str, self.gateway_options.get("password", ""))
        self.app_id = cast(str, self.gateway_options.get("app_id", ""))
        self.auth_code = cast(str, self.gateway_options.get("auth_code", ""))
        self.use_aggregator = use_aggregator
        self.broker = broker
        self.trading_mode = trading_mode
        self.initialize = initialize
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_tick = on_tick
        self.on_order = on_order
        self.on_trade = on_trade
        self.on_reject = on_reject
        self.on_session_start = on_session_start
        self.on_broker_connected = on_broker_connected
        self.broker_ready_timeout = broker_ready_timeout
        self.broker_ready_required = broker_ready_required
        self.on_session_end = on_session_end
        self.on_before_trading = on_before_trading
        self.on_after_trading = on_after_trading
        self.on_daily_rebalance = on_daily_rebalance
        self.on_daily_rebalance_after_bar = on_daily_rebalance_after_bar
        self.on_portfolio_update = on_portfolio_update
        self.on_error = on_error
        self.on_timer = on_timer
        self.context = context or {}
        self.strategy_max_order_value = self._normalize_strategy_float_map(
            strategy_max_order_value
        )
        self.strategy_max_order_size = self._normalize_strategy_float_map(
            strategy_max_order_size
        )
        self.strategy_max_position_size = self._normalize_strategy_float_map(
            strategy_max_position_size
        )
        self.strategy_max_daily_loss = self._normalize_strategy_float_map(
            strategy_max_daily_loss
        )
        self.strategy_max_drawdown = self._normalize_strategy_float_map(
            strategy_max_drawdown
        )
        self.strategy_reduce_only_after_risk = self._normalize_strategy_bool_map(
            strategy_reduce_only_after_risk
        )
        self.strategy_risk_cooldown_bars = self._normalize_strategy_int_map(
            strategy_risk_cooldown_bars
        )
        self.strategy_priority = self._normalize_strategy_int_map(strategy_priority)
        self.strategy_risk_budget = self._normalize_strategy_float_map(
            strategy_risk_budget
        )
        self.portfolio_risk_budget = (
            float(portfolio_risk_budget) if portfolio_risk_budget is not None else None
        )
        self.risk_budget_mode = risk_budget_mode
        self.risk_budget_reset_daily = bool(risk_budget_reset_daily)
        self.on_broker_event = on_broker_event
        self._init_broker_bridge_state()

        self.feed = DataFeed.create_live()  # type: ignore
        self.engine = Engine()

    def run(
        self,
        cash: float = 1_000_000.0,
        show_progress: bool = False,
        duration: Optional[str] = None,
    ) -> None:
        """
        Run the live/paper trading session.

        :param cash: Initial cash (default 1,000,000).
        :param show_progress: Whether to show progress bar (default False).
        :param duration: Optional run duration string (e.g., "1m", "1h", "60s").
                         If set, strategy will stop after this duration.
        """
        logger.info(
            "Configuring live engine",
            extra=self._runner_log_extra(phase="live"),
        )
        self.engine.add_data(self.feed)
        self.engine.set_cash(cash)

        for instrument in self.instruments:
            self.engine.add_instrument(instrument)

        if self.trading_mode == "broker_live":
            self.engine.use_realtime_execution()
        else:
            self.engine.use_simulated_execution()

        self.engine.use_china_futures_market()
        self.engine.set_force_session_continuous(True)

        symbols = [inst.symbol for inst in self.instruments]
        gateway_kwargs = self._build_gateway_kwargs()
        bundle = create_gateway_bundle(
            broker=self.broker,
            feed=self.feed,
            symbols=symbols,
            use_aggregator=self.use_aggregator,
            **gateway_kwargs,
        )
        self._broker_capabilities = bundle.trader_capabilities

        if bundle.market_gateway is not None:
            logger.info(
                "Starting %s market gateway",
                self.broker,
                extra=self._runner_log_extra(phase="gateway"),
            )
            self._start_gateway_thread(
                bundle.market_gateway.start, f"{self.broker}-market"
            )

        if self.trading_mode == "broker_live":
            if bundle.trader_gateway is None:
                raise ValueError(
                    "trading_mode='broker_live' requires a trader gateway configuration"
                )
            logger.info(
                "Connecting+starting %s trader gateway",
                self.broker,
                extra=self._runner_log_extra(phase="gateway"),
            )
            self._connect_and_start_trader(bundle.trader_gateway)

        time.sleep(2.0)

        # Create Strategy Instances
        strategy_instance, slot_strategy_instances, effective_strategy_id = (
            self._build_strategy_topology()
        )
        self._configure_strategy_slots(
            strategy_instance, slot_strategy_instances, effective_strategy_id
        )
        if bundle.trader_gateway is not None:
            strategy_targets = [strategy_instance, *slot_strategy_instances.values()]
            callback_target: Any = (
                _StrategyCallbackFanout(strategy_targets)
                if len(strategy_targets) > 1
                else strategy_instance
            )
            self._bind_broker_callbacks(bundle.trader_gateway, callback_target)
            for target in strategy_targets:
                self._install_broker_order_submitter(bundle.trader_gateway, target)
            self._await_broker_ready(bundle.trader_gateway, strategy_targets)

        # Apply duration limit if specified
        if duration:
            logger.info(
                "Live auto-stop enabled: %s",
                duration,
                extra=self._runner_log_extra(phase="live"),
            )
            self._apply_time_limit(strategy_instance, duration)

        logger.info(
            "Running live strategy loop",
            extra=self._runner_log_extra(phase="live"),
        )
        try:
            self.engine.run(strategy_instance, show_progress=show_progress)
        except KeyboardInterrupt:
            logger.info(
                "Stopping live runner by user interrupt or duration limit",
                extra=self._runner_log_extra(phase="live"),
            )
        except Exception as exc:
            logger.exception(
                "Stopping live runner due to error: %s",
                exc,
                extra=self._runner_log_extra(phase="live"),
            )
        finally:
            self._stop_broker_dispatcher()
            self._print_summary()

    def _build_strategy_instance(self, strategy_input: Any) -> Strategy:
        resolved_strategy_input = resolve_strategy_input(
            strategy=cast(
                Optional[Union[type[Strategy], Strategy, Callable[[Any, Bar], None]]],
                strategy_input,
            ),
            strategy_source=getattr(self, "strategy_source", None),
            strategy_loader=getattr(self, "strategy_loader", None),
            strategy_loader_options=getattr(self, "strategy_loader_options", None),
        )
        if isinstance(resolved_strategy_input, type) and issubclass(
            resolved_strategy_input, Strategy
        ):
            return cast(Strategy, resolved_strategy_input())
        if isinstance(resolved_strategy_input, Strategy):
            return resolved_strategy_input
        if callable(resolved_strategy_input):
            from akquant.backtest import FunctionalStrategy

            return FunctionalStrategy(
                initialize=getattr(self, "initialize", None),
                on_bar=cast(Callable[[Any, Bar], None], resolved_strategy_input),
                on_start=getattr(self, "on_start", None),
                on_stop=getattr(self, "on_stop", None),
                on_tick=getattr(self, "on_tick", None),
                on_order=getattr(self, "on_order", None),
                on_trade=getattr(self, "on_trade", None),
                on_reject=getattr(self, "on_reject", None),
                on_session_start=getattr(self, "on_session_start", None),
                on_session_end=getattr(self, "on_session_end", None),
                on_before_trading=getattr(self, "on_before_trading", None),
                on_after_trading=getattr(self, "on_after_trading", None),
                on_daily_rebalance=getattr(self, "on_daily_rebalance", None),
                on_daily_rebalance_after_bar=getattr(
                    self, "on_daily_rebalance_after_bar", None
                ),
                on_portfolio_update=getattr(self, "on_portfolio_update", None),
                on_error=getattr(self, "on_error", None),
                on_timer=getattr(self, "on_timer", None),
                context=getattr(self, "context", {}),
            )
        raise TypeError("strategy must be Strategy type/instance or callable")

    def _build_strategy_topology(self) -> tuple[Strategy, Dict[str, Strategy], str]:
        strategy_instance = self._build_strategy_instance(self.strategy_cls)
        slot_strategy_instances: Dict[str, Strategy] = {}
        for slot_key, slot_input in self.strategies_by_slot.items():
            slot_key_str = str(slot_key).strip()
            if not slot_key_str:
                raise ValueError("strategy slot id cannot be empty")
            slot_strategy_instances[slot_key_str] = self._build_strategy_instance(
                slot_input
            )
        return strategy_instance, slot_strategy_instances, self.strategy_id

    def _configure_strategy_slots(
        self,
        strategy_instance: Strategy,
        slot_strategy_instances: Dict[str, Strategy],
        effective_strategy_id: str,
    ) -> None:
        configured_slot_ids = [effective_strategy_id]
        for slot_id in slot_strategy_instances.keys():
            if slot_id not in configured_slot_ids:
                configured_slot_ids.append(slot_id)

        setattr(strategy_instance, "_owner_strategy_id", effective_strategy_id)
        for slot_id, slot_strategy in slot_strategy_instances.items():
            setattr(slot_strategy, "_owner_strategy_id", slot_id)

        default_ready = getattr(self, "trading_mode", "paper") != "broker_live"
        for target in [strategy_instance, *slot_strategy_instances.values()]:
            setattr(target, "broker_ready", default_ready)

        strategy_targets = [strategy_instance, *slot_strategy_instances.values()]
        if self.context:
            for target in strategy_targets:
                if hasattr(target, "_context"):
                    continue
                for key, value in self.context.items():
                    setattr(target, key, value)

        if hasattr(self.engine, "set_strategy_slots"):
            cast(Any, self.engine).set_strategy_slots(configured_slot_ids)
        if hasattr(self.engine, "set_default_strategy_id"):
            cast(Any, self.engine).set_default_strategy_id(effective_strategy_id)
        if hasattr(self.engine, "set_strategy_for_slot"):
            for slot_index, slot_id in enumerate(configured_slot_ids):
                assigned = (
                    strategy_instance
                    if slot_id == effective_strategy_id
                    else slot_strategy_instances[slot_id]
                )
                cast(Any, self.engine).set_strategy_for_slot(slot_index, assigned)
        self._apply_strategy_risk_controls(configured_slot_ids)

    def _normalize_strategy_float_map(
        self, values: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        if values is None:
            return {}
        if not isinstance(values, dict):
            raise TypeError("strategy map must be a dict when provided")
        normalized: Dict[str, float] = {}
        for key, value in values.items():
            key_str = str(key).strip()
            if not key_str:
                raise ValueError("strategy id cannot be empty")
            normalized[key_str] = float(value)
        return normalized

    def _normalize_strategy_int_map(
        self, values: Optional[Dict[str, int]]
    ) -> Dict[str, int]:
        if values is None:
            return {}
        if not isinstance(values, dict):
            raise TypeError("strategy map must be a dict when provided")
        normalized: Dict[str, int] = {}
        for key, value in values.items():
            key_str = str(key).strip()
            if not key_str:
                raise ValueError("strategy id cannot be empty")
            normalized[key_str] = int(value)
        return normalized

    def _normalize_strategy_bool_map(
        self, values: Optional[Dict[str, bool]]
    ) -> Dict[str, bool]:
        if values is None:
            return {}
        if not isinstance(values, dict):
            raise TypeError("strategy map must be a dict when provided")
        normalized: Dict[str, bool] = {}
        for key, value in values.items():
            key_str = str(key).strip()
            if not key_str:
                raise ValueError("strategy id cannot be empty")
            normalized[key_str] = bool(value)
        return normalized

    def _validate_strategy_map_keys(
        self, values: Dict[str, Any], configured_slot_ids: List[str], field_name: str
    ) -> None:
        if not values:
            return
        unknown = sorted(set(values.keys()).difference(set(configured_slot_ids)))
        if unknown:
            unknown_text = ", ".join(unknown)
            raise ValueError(
                f"{field_name} contains unknown strategy ids: {unknown_text}"
            )

    def _apply_strategy_risk_controls(self, configured_slot_ids: List[str]) -> None:
        strategy_max_order_value = cast(
            Dict[str, float], getattr(self, "strategy_max_order_value", {})
        )
        strategy_max_order_size = cast(
            Dict[str, float], getattr(self, "strategy_max_order_size", {})
        )
        strategy_max_position_size = cast(
            Dict[str, float], getattr(self, "strategy_max_position_size", {})
        )
        strategy_max_daily_loss = cast(
            Dict[str, float], getattr(self, "strategy_max_daily_loss", {})
        )
        strategy_max_drawdown = cast(
            Dict[str, float], getattr(self, "strategy_max_drawdown", {})
        )
        strategy_reduce_only_after_risk = cast(
            Dict[str, bool], getattr(self, "strategy_reduce_only_after_risk", {})
        )
        strategy_risk_cooldown_bars = cast(
            Dict[str, int], getattr(self, "strategy_risk_cooldown_bars", {})
        )
        strategy_priority = cast(Dict[str, int], getattr(self, "strategy_priority", {}))
        strategy_risk_budget = cast(
            Dict[str, float], getattr(self, "strategy_risk_budget", {})
        )
        portfolio_risk_budget = cast(
            Optional[float], getattr(self, "portfolio_risk_budget", None)
        )
        risk_budget_mode = str(getattr(self, "risk_budget_mode", "order_notional"))
        risk_budget_reset_daily = bool(getattr(self, "risk_budget_reset_daily", False))

        self._validate_strategy_map_keys(
            strategy_max_order_value,
            configured_slot_ids,
            "strategy_max_order_value",
        )
        self._validate_strategy_map_keys(
            strategy_max_order_size,
            configured_slot_ids,
            "strategy_max_order_size",
        )
        self._validate_strategy_map_keys(
            strategy_max_position_size,
            configured_slot_ids,
            "strategy_max_position_size",
        )
        self._validate_strategy_map_keys(
            strategy_max_daily_loss,
            configured_slot_ids,
            "strategy_max_daily_loss",
        )
        self._validate_strategy_map_keys(
            strategy_max_drawdown,
            configured_slot_ids,
            "strategy_max_drawdown",
        )
        self._validate_strategy_map_keys(
            strategy_reduce_only_after_risk,
            configured_slot_ids,
            "strategy_reduce_only_after_risk",
        )
        self._validate_strategy_map_keys(
            strategy_risk_cooldown_bars,
            configured_slot_ids,
            "strategy_risk_cooldown_bars",
        )
        self._validate_strategy_map_keys(
            strategy_priority,
            configured_slot_ids,
            "strategy_priority",
        )
        self._validate_strategy_map_keys(
            strategy_risk_budget,
            configured_slot_ids,
            "strategy_risk_budget",
        )

        if strategy_max_order_value and hasattr(
            self.engine, "set_strategy_max_order_value_limits"
        ):
            cast(Any, self.engine).set_strategy_max_order_value_limits(
                strategy_max_order_value
            )
        if strategy_max_order_size and hasattr(
            self.engine, "set_strategy_max_order_size_limits"
        ):
            cast(Any, self.engine).set_strategy_max_order_size_limits(
                strategy_max_order_size
            )
        if strategy_max_position_size and hasattr(
            self.engine, "set_strategy_max_position_size_limits"
        ):
            cast(Any, self.engine).set_strategy_max_position_size_limits(
                strategy_max_position_size
            )
        if strategy_max_daily_loss and hasattr(
            self.engine, "set_strategy_max_daily_loss_limits"
        ):
            cast(Any, self.engine).set_strategy_max_daily_loss_limits(
                strategy_max_daily_loss
            )
        if strategy_max_drawdown and hasattr(
            self.engine, "set_strategy_max_drawdown_limits"
        ):
            cast(Any, self.engine).set_strategy_max_drawdown_limits(
                strategy_max_drawdown
            )
        if strategy_reduce_only_after_risk and hasattr(
            self.engine, "set_strategy_reduce_only_after_risk"
        ):
            cast(Any, self.engine).set_strategy_reduce_only_after_risk(
                strategy_reduce_only_after_risk
            )
        if strategy_risk_cooldown_bars and hasattr(
            self.engine, "set_strategy_risk_cooldown_bars"
        ):
            cast(Any, self.engine).set_strategy_risk_cooldown_bars(
                strategy_risk_cooldown_bars
            )
        if strategy_priority and hasattr(self.engine, "set_strategy_priorities"):
            cast(Any, self.engine).set_strategy_priorities(strategy_priority)
        if strategy_risk_budget and hasattr(
            self.engine, "set_strategy_risk_budget_limits"
        ):
            cast(Any, self.engine).set_strategy_risk_budget_limits(strategy_risk_budget)
        if hasattr(self.engine, "set_portfolio_risk_budget_limit"):
            cast(Any, self.engine).set_portfolio_risk_budget_limit(
                portfolio_risk_budget
            )
        if risk_budget_mode not in {"order_notional", "trade_notional"}:
            raise ValueError(
                "risk_budget_mode must be 'order_notional' or 'trade_notional'"
            )
        if hasattr(self.engine, "set_risk_budget_mode"):
            cast(Any, self.engine).set_risk_budget_mode(risk_budget_mode)
        if hasattr(self.engine, "set_risk_budget_reset_daily"):
            cast(Any, self.engine).set_risk_budget_reset_daily(risk_budget_reset_daily)

    def _normalize_gateway_options(
        self,
        gateway_options: Optional[Dict[str, Any]],
        **legacy_values: Any,
    ) -> Dict[str, Any]:
        normalized = dict(gateway_options or {})
        for key in _LEGACY_GATEWAY_OPTION_KEYS:
            if key in normalized:
                continue
            value = legacy_values.get(key)
            if value is None or value == "":
                continue
            normalized[key] = value
        return normalized

    def _build_gateway_kwargs(self) -> Dict[str, Any]:
        return dict(self.gateway_options)

    def _start_gateway_thread(self, target: Any, name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()

    def _connect_and_start_trader(self, trader_gateway: Any) -> None:
        """Run connect() then start() on the gateway thread.

        connect() may block (CTP's connect() == start() enters the event loop
        via Join()), so it must NOT run on the main thread. Running both on the
        daemon thread fixes the QMF login gap (connect()=login before start()=WS)
        without blocking run(); readiness is confirmed by the heartbeat poll.
        """

        def _run() -> None:
            try:
                connect = getattr(trader_gateway, "connect", None)
                if callable(connect):
                    connect()
                trader_gateway.start()
            except Exception:
                # 登录/启动失败：记清晰日志（否则仅一条泛化的"未就绪"告警）；
                # 就绪由 heartbeat 轮询裁定，broker_ready 保持 False。
                logger.exception(
                    "trader gateway connect/start failed",
                    extra=self._runner_log_extra(phase="gateway"),
                )

        self._start_gateway_thread(_run, f"{self.broker}-trader")

    def _runner_log_extra(self, *, phase: str) -> dict[str, Any]:
        strategy_id = (
            str(getattr(self, "strategy_id", "_default")).strip() or "_default"
        )
        return build_log_extra(
            phase=phase,
            strategy_id=strategy_id,
            slot=strategy_id if strategy_id != "_default" else None,
        )

    def _init_broker_bridge_state(self) -> None:
        if not hasattr(self, "on_broker_event"):
            self.on_broker_event = None
        if not hasattr(self, "gateway_options"):
            self.gateway_options = {}
        self._broker_event_lock = threading.Lock()
        self._broker_events: list[tuple[str, Any]] = []
        self._broker_event_keys: set[str] = set()
        self._broker_order_states: dict[str, Any] = {}
        self._broker_account_state: Any = None
        self._client_to_broker_order_ids: dict[str, str] = {}
        self._broker_to_client_order_ids: dict[str, str] = {}
        self._client_to_strategy_ids: dict[str, str] = {}
        self._broker_to_strategy_ids: dict[str, str] = {}
        self._closed_broker_order_ids: set[str] = set()
        self._broker_trade_keys: set[str] = set()
        self._broker_report_keys: set[str] = set()
        self._broker_dispatch_stop: threading.Event | None = None
        self._broker_dispatch_thread: threading.Thread | None = None
        self._broker_recovery_stop: threading.Event | None = None
        self._broker_recovery_thread: threading.Thread | None = None
        self._broker_recovery_interval_sec = 1.0
        self._broker_trader_gateway: Any = None
        self._broker_runtime: Any = None
        self._broker_order_submitter: Any = None
        self._broker_event_bridge: Any = None
        self._broker_recovery: Any = None
        self._broker_submit_seq = 0
        self._broker_submit_lock = threading.Lock()
        self._broker_recovery_mode = self._normalize_broker_recovery_mode(
            getattr(self, "gateway_options", {}).get("recovery_mode", "compatible")
        )
        self._broker_recovery_last_error_key = ""
        self._broker_runtime = BrokerRuntime(
            event_lock=self._broker_event_lock,
            event_store=self._broker_events,
            event_keys=self._broker_event_keys,
            get_on_broker_event=lambda: self.on_broker_event,
            make_event_key=self._make_event_key,
            update_broker_state=self._update_broker_state,
            resolve_owner_strategy_id=self._resolve_owner_strategy_id,
            payload_to_dict=self._payload_to_dict,
            safe_strategy_callback=self._safe_strategy_callback,
            get_trader_gateway=lambda: self._broker_trader_gateway,
            notify_strategy_error=self._notify_strategy_error,
            get_recovery_mode=lambda: self._broker_recovery_mode,
            get_last_error_key=lambda: self._broker_recovery_last_error_key,
            set_last_error_key=self._set_broker_recovery_last_error_key,
            resolve_trader_capabilities=self._resolve_trader_capabilities,
            next_client_order_id=self._next_client_order_id,
            can_submit_client_order=self.can_submit_client_order,
            sync_order_id_mapping=self._sync_order_id_mapping,
            bind_order_owner=self._bind_order_owner,
            payload_field=self._payload_field,
            get_execution_capabilities=self.get_execution_capabilities,
        )
        self._broker_event_bridge = self._broker_runtime.event_bridge
        self._broker_recovery = self._broker_runtime.recovery

    def _bind_broker_callbacks(self, trader_gateway: Any, strategy: Strategy) -> None:
        self._broker_trader_gateway = trader_gateway
        if hasattr(trader_gateway, "on_order"):
            trader_gateway.on_order(
                lambda order: self._queue_broker_event("order", order)
            )
        if hasattr(trader_gateway, "on_trade"):
            trader_gateway.on_trade(
                lambda trade: self._queue_broker_event("trade", trade)
            )
        if hasattr(trader_gateway, "on_execution_report"):
            trader_gateway.on_execution_report(
                lambda report: self._queue_broker_event("execution_report", report)
            )
        self._start_broker_dispatcher(strategy)

    def _install_broker_order_submitter(
        self, trader_gateway: Any, strategy: Strategy
    ) -> None:
        self._broker_capabilities = self._resolve_trader_capabilities(trader_gateway)
        self._broker_order_submitter = self._broker_runtime.install_submitter(
            trader_gateway,
            strategy,
        )

    def _await_broker_ready(self, trader_gateway: Any, targets: list) -> None:
        """Poll heartbeat() until ready/timeout; set broker_ready, fire callback."""
        deadline = time.monotonic() + float(self.broker_ready_timeout)
        ready = False
        while time.monotonic() < deadline:
            try:
                if trader_gateway.heartbeat():
                    ready = True
                    break
            except Exception:  # noqa: BLE001 heartbeat errors count as not-ready
                pass
            time.sleep(0.2)
        for target in targets:
            setattr(target, "broker_ready", ready)
        if ready:
            self._dispatch_broker_connected(targets)
        else:
            logger.warning(
                "broker not ready within %ss",
                self.broker_ready_timeout,
                extra=self._runner_log_extra(phase="gateway"),
            )
            if self.broker_ready_required:
                raise RuntimeError(
                    f"broker not ready within {self.broker_ready_timeout}s"
                )

    def _dispatch_broker_connected(self, targets: list) -> None:
        """Fire on_broker_connected: strategy method(s) plus function-style callback."""
        for target in targets:
            method = getattr(target, "on_broker_connected", None)
            if callable(method):
                try:
                    method()
                except Exception:  # noqa: BLE001
                    logger.exception("strategy on_broker_connected failed")
            if self.on_broker_connected is not None:
                try:
                    self.on_broker_connected(target)
                except Exception:  # noqa: BLE001
                    logger.exception("on_broker_connected callback failed")

    def _resolve_live_order_legs(
        self,
        trader_gateway: Any,
        capability: Any,
        symbol: str,
        side: str,
        quantity: float,
        position_effect: str,
        reduce_only: bool,
    ) -> list[tuple[str, float]]:
        return resolve_live_order_legs(
            trader_gateway=trader_gateway,
            capability=capability,
            symbol=symbol,
            side=side,
            quantity=quantity,
            position_effect=position_effect,
            reduce_only=reduce_only,
            payload_field=self._payload_field,
        )

    def _find_live_close_position(
        self, positions: Any, symbol: str, side: str
    ) -> Any | None:
        return find_live_close_position(
            positions=positions,
            symbol=symbol,
            side=side,
            payload_field=self._payload_field,
        )

    def _build_live_order_client_ids(
        self, request_client_order_id: str, order_legs: list[tuple[str, float]]
    ) -> list[str]:
        return build_live_order_client_ids(request_client_order_id, order_legs)

    def _validate_live_order_client_ids(
        self,
        strategy: Strategy,
        client_order_ids: list[str],
        symbol: str,
        side: str,
        quantity: float,
    ) -> None:
        validate_live_order_client_ids(
            strategy=strategy,
            client_order_ids=client_order_ids,
            symbol=symbol,
            side=side,
            quantity=quantity,
            can_submit_client_order=self.can_submit_client_order,
            notify_strategy_error=self._notify_strategy_error,
        )

    def _start_broker_dispatcher(self, strategy: Strategy) -> None:
        self._stop_broker_dispatcher()
        self._broker_dispatch_stop = threading.Event()
        self._broker_dispatch_thread = threading.Thread(
            target=self._broker_dispatch_loop,
            args=(strategy,),
            name=f"{self.broker}-broker-dispatch",
            daemon=True,
        )
        self._broker_dispatch_thread.start()
        self._start_broker_recovery(strategy)

    def _stop_broker_dispatcher(self) -> None:
        self._stop_broker_recovery()
        if self._broker_dispatch_stop is not None:
            self._broker_dispatch_stop.set()
        if self._broker_dispatch_thread is not None:
            self._broker_dispatch_thread.join(timeout=1.0)
        self._broker_dispatch_stop = None
        self._broker_dispatch_thread = None

    def _start_broker_recovery(self, strategy: Strategy) -> None:
        self._stop_broker_recovery()
        self._broker_recovery_stop = threading.Event()
        self._broker_recovery_thread = threading.Thread(
            target=self._broker_recovery_loop,
            args=(strategy,),
            name=f"{self.broker}-broker-recovery",
            daemon=True,
        )
        self._broker_recovery_thread.start()

    def _stop_broker_recovery(self) -> None:
        if self._broker_recovery_stop is not None:
            self._broker_recovery_stop.set()
        if self._broker_recovery_thread is not None:
            self._broker_recovery_thread.join(timeout=1.0)
        self._broker_recovery_stop = None
        self._broker_recovery_thread = None

    def _queue_broker_event(self, event_name: str, payload: Any) -> None:
        self._broker_runtime.queue_event(event_name, payload)

    def _broker_dispatch_loop(self, strategy: Strategy) -> None:
        while (
            self._broker_dispatch_stop is not None
            and not self._broker_dispatch_stop.is_set()
        ):
            self._drain_broker_events(strategy)
            time.sleep(0.05)
        self._drain_broker_events(strategy)

    def _drain_broker_events(self, strategy: Strategy) -> None:
        self._broker_runtime.drain_events(strategy)

    def _broker_recovery_loop(self, strategy: Strategy) -> None:
        while (
            self._broker_recovery_stop is not None
            and not self._broker_recovery_stop.is_set()
        ):
            self._run_broker_recovery_cycle(strategy)
            self._drain_broker_events(strategy)
            time.sleep(self._broker_recovery_interval_sec)

    def _run_broker_recovery_cycle(self, strategy: Strategy | None = None) -> None:
        self._broker_runtime.run_recovery_cycle(
            strategy,
            handle_error=self._handle_broker_recovery_error,
        )

    def _update_broker_state(self, event_name: str, payload: Any) -> None:
        if event_name == "order":
            broker_order_id = str(self._payload_field(payload, "broker_order_id"))
            client_order_id = str(self._payload_field(payload, "client_order_id"))
            self._sync_order_id_mapping(client_order_id, broker_order_id)
            if broker_order_id:
                self._broker_order_states[broker_order_id] = payload
                status = self._payload_field(payload, "status")
                if self._is_terminal_status(status):
                    self._close_order_mapping(client_order_id, broker_order_id)
        elif event_name == "trade":
            trade_key = str(self._payload_field(payload, "trade_id"))
            broker_order_id = str(self._payload_field(payload, "broker_order_id"))
            client_order_id = str(self._payload_field(payload, "client_order_id"))
            if not client_order_id and broker_order_id:
                client_order_id = self._resolve_client_order_id(broker_order_id)
            self._sync_order_id_mapping(client_order_id, broker_order_id)
            if trade_key:
                self._broker_trade_keys.add(trade_key)
        elif event_name == "execution_report":
            broker_order_id = str(self._payload_field(payload, "broker_order_id"))
            client_order_id = str(self._payload_field(payload, "client_order_id"))
            self._sync_order_id_mapping(client_order_id, broker_order_id)
            status = self._payload_field(payload, "status")
            if self._is_terminal_status(status):
                self._close_order_mapping(client_order_id, broker_order_id)
            report_key = (
                f"{self._payload_field(payload, 'broker_order_id')}-"
                f"{self._payload_field(payload, 'status')}-"
                f"{self._payload_field(payload, 'timestamp_ns')}"
            )
            if report_key:
                self._broker_report_keys.add(report_key)
        elif event_name == "account":
            self._broker_account_state = payload

    def _make_event_key(self, event_name: str, payload: Any) -> str:
        if event_name == "trade":
            trade_id = str(self._payload_field(payload, "trade_id"))
            if trade_id:
                return f"trade:{trade_id}"
        if event_name == "order":
            broker_order_id = str(self._payload_field(payload, "broker_order_id"))
            status = str(self._payload_field(payload, "status"))
            filled_quantity = str(self._payload_field(payload, "filled_quantity"))
            timestamp_ns = str(self._payload_field(payload, "timestamp_ns"))
            return f"order:{broker_order_id}:{status}:{filled_quantity}:{timestamp_ns}"
        if event_name == "execution_report":
            broker_order_id = str(self._payload_field(payload, "broker_order_id"))
            status = str(self._payload_field(payload, "status"))
            timestamp_ns = str(self._payload_field(payload, "timestamp_ns"))
            return f"execution_report:{broker_order_id}:{status}:{timestamp_ns}"
        if event_name == "account":
            account_id = str(self._payload_field(payload, "account_id"))
            timestamp_ns = str(self._payload_field(payload, "timestamp_ns"))
            return f"account:{account_id}:{timestamp_ns}"
        return f"{event_name}:{id(payload)}"

    def _payload_field(self, payload: Any, field: str) -> Any:
        if isinstance(payload, dict):
            return payload.get(field, "")
        return getattr(payload, field, "")

    def _next_client_order_id(self) -> str:
        with self._broker_submit_lock:
            self._broker_submit_seq += 1
            return f"{self.broker}-coid-{self._broker_submit_seq}"

    def _sync_order_id_mapping(
        self,
        client_order_id: str,
        broker_order_id: str,
    ) -> None:
        if client_order_id and broker_order_id:
            self._client_to_broker_order_ids[client_order_id] = broker_order_id
            self._broker_to_client_order_ids[broker_order_id] = client_order_id

    def _bind_order_owner(
        self, client_order_id: str, broker_order_id: str, owner_strategy_id: str
    ) -> None:
        if client_order_id:
            self._client_to_strategy_ids[client_order_id] = owner_strategy_id
        if broker_order_id:
            self._broker_to_strategy_ids[broker_order_id] = owner_strategy_id

    def _resolve_owner_strategy_id(self, payload: Any) -> str:
        owner_strategy_id = str(
            self._payload_field(payload, "owner_strategy_id")
        ).strip()
        if owner_strategy_id:
            return owner_strategy_id
        broker_order_id = str(self._payload_field(payload, "broker_order_id")).strip()
        client_order_id = str(self._payload_field(payload, "client_order_id")).strip()
        if not client_order_id and broker_order_id:
            client_order_id = self._resolve_client_order_id(broker_order_id)
        if client_order_id:
            mapped = self._client_to_strategy_ids.get(client_order_id, "").strip()
            if mapped:
                return mapped
        if broker_order_id:
            mapped = self._broker_to_strategy_ids.get(broker_order_id, "").strip()
            if mapped:
                return mapped
        return "_default"

    def _payload_to_dict(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return dict(payload)
        if hasattr(payload, "__dict__"):
            return dict(getattr(payload, "__dict__"))
        return {}

    def _resolve_client_order_id(self, broker_order_id: str) -> str:
        return self._broker_to_client_order_ids.get(broker_order_id, "")

    def _resolve_broker_order_id(self, client_order_id: str) -> str:
        return self._client_to_broker_order_ids.get(client_order_id, "")

    def _close_order_mapping(self, client_order_id: str, broker_order_id: str) -> None:
        if client_order_id:
            self._client_to_broker_order_ids.pop(client_order_id, None)
            self._client_to_strategy_ids.pop(client_order_id, None)
        if broker_order_id:
            self._broker_to_client_order_ids.pop(broker_order_id, None)
            self._broker_to_strategy_ids.pop(broker_order_id, None)
            self._closed_broker_order_ids.add(broker_order_id)

    def _is_terminal_status(self, status: Any) -> bool:
        status_text = str(status).strip().lower()
        return status_text in {"filled", "cancelled", "canceled", "rejected"}

    def can_submit_client_order(self, client_order_id: str) -> bool:
        """Check whether a client order id can be submitted again."""
        broker_order_id = self._resolve_broker_order_id(client_order_id)
        if not broker_order_id:
            return True
        if broker_order_id in self._closed_broker_order_ids:
            return True
        snapshot = self._broker_order_states.get(broker_order_id)
        if snapshot is None:
            return False
        status = self._payload_field(snapshot, "status")
        return self._is_terminal_status(status)

    def get_execution_capabilities(self) -> dict[str, Any]:
        """Return execution capabilities for broker live mode."""
        capability = self._broker_capabilities
        if capability is None:
            gateway = getattr(self, "_broker_trader_gateway", None)
            capability = self._resolve_trader_capabilities(gateway)
            self._broker_capabilities = capability
        return capability.as_execution_capabilities()

    def _resolve_trader_capabilities(self, trader_gateway: Any) -> BrokerCapability:
        get_capabilities = getattr(trader_gateway, "get_capabilities", None)
        if callable(get_capabilities):
            return BrokerCapability.from_value(
                get_capabilities(),
                broker_name=str(getattr(self, "broker", "broker")),
            )
        existing = getattr(self, "_broker_capabilities", None)
        if isinstance(existing, BrokerCapability):
            return existing
        return BrokerCapability(
            broker_name=str(getattr(self, "broker", "broker")),
            broker_live=True,
            client_order_id=True,
            order_type=True,
            time_in_force_str=True,
            position_effect=True,
            reduce_only=True,
            position_details=False,
            supported_position_effects=(
                "auto",
                "open",
                "close",
                "close_today",
                "close_yesterday",
            ),
        )

    def _normalize_broker_recovery_mode(self, mode: Any) -> str:
        normalized = str(mode or "compatible").strip().lower()
        if normalized not in {"compatible", "strict"}:
            raise ValueError(
                "gateway_options.recovery_mode must be 'compatible' or 'strict'"
            )
        return normalized

    def _handle_broker_recovery_error(
        self,
        strategy: Strategy | None,
        source: str,
        error: Exception,
        payload: Dict[str, Any],
    ) -> None:
        self._broker_runtime.handle_recovery_error(strategy, source, error, payload)

    def _set_broker_recovery_last_error_key(self, error_key: str) -> None:
        self._broker_recovery_last_error_key = error_key

    def _notify_strategy_error(
        self,
        strategy: Strategy,
        error: Exception,
        source: str,
        payload: Any,
    ) -> None:
        on_error = getattr(strategy, "on_error", None)
        if on_error is None:
            return
        try:
            on_error(error, source, payload)
        except Exception:
            pass

    def _safe_strategy_callback(
        self,
        strategy: Strategy,
        callback_name: str,
        payload: Any,
    ) -> None:
        callback = getattr(strategy, callback_name, None)
        if callback is None:
            return
        try:
            callback(payload)
        except Exception as exc:
            owner_strategy_id = (
                str(getattr(strategy, "_owner_strategy_id", "_default")).strip()
                or "_default"
            )
            payload_dict = self._payload_to_dict(payload)
            logger.warning(
                "Strategy broker callback failed",
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
            on_error = getattr(strategy, "on_error", None)
            if on_error is not None and callback_name != "on_error":
                on_error(exc, callback_name, payload)

    def _apply_time_limit(self, strategy: Strategy, duration_str: str) -> None:
        """Inject time check into strategy methods."""
        import re

        # Parse duration
        duration_sec = 0
        match = re.match(r"^(\d+)([smh]?)$", duration_str)
        if match:
            val, unit = match.groups()
            val = int(val)
            if unit == "s" or unit == "":
                duration_sec = val
            elif unit == "m":
                duration_sec = val * 60
            elif unit == "h":
                duration_sec = val * 3600
        else:
            logger.warning(
                "Ignored invalid live duration format: %s",
                duration_str,
                extra=self._runner_log_extra(phase="live"),
            )
            return

        start_time = time.time()

        # Patch on_bar
        original_on_bar = strategy.on_bar

        def wrapped_on_bar(bar: Bar) -> None:
            if time.time() - start_time > duration_sec:
                raise KeyboardInterrupt(f"Duration {duration_str} reached")
            original_on_bar(bar)

        # Use setattr to bypass mypy method assignment check
        setattr(strategy, "on_bar", wrapped_on_bar)

        # Patch on_tick if it exists/is overridden
        if hasattr(strategy, "on_tick"):
            original_on_tick = strategy.on_tick

            def wrapped_on_tick(tick: Any) -> None:
                if time.time() - start_time > duration_sec:
                    raise KeyboardInterrupt(f"Duration {duration_str} reached")
                original_on_tick(tick)

            setattr(strategy, "on_tick", wrapped_on_tick)

    def _print_summary(self) -> None:
        try:
            results = self.engine.get_results()
            total_return_display = format_metric_value(
                "total_return_pct", results.metrics.total_return_pct
            )
            annualized_return_display = format_metric_value(
                "annualized_return", results.metrics.annualized_return
            )
            max_drawdown_display = format_metric_value(
                "max_drawdown_pct", results.metrics.max_drawdown_pct
            )
            win_rate_display = format_metric_value("win_rate", results.metrics.win_rate)
            summary_lines = [
                "=" * 50,
                "TRADING SUMMARY (Manual Stop)",
                "=" * 50,
                f"Total Return: {total_return_display}",
                f"Annualized Return: {annualized_return_display}",
                f"Max Drawdown: {max_drawdown_display}",
                f"Sharpe Ratio: {results.metrics.sharpe_ratio:.4f}",
                f"Win Rate: {win_rate_display}",
                f"Total Trades: {len(results.trades)}",
                "=" * 50,
            ]

            # Print Current Positions if available
            if results.snapshots:
                last_snapshots = results.snapshots[-1][1]
                summary_lines.extend(["", "Current Positions:"])
                has_pos = False
                for s in last_snapshots:
                    if abs(s.quantity) > 0:
                        summary_lines.append(f"  {s.symbol}: {s.quantity}")
                        has_pos = True
                if not has_pos:
                    summary_lines.append("  (None)")

            logger.info(
                "%s",
                "\n".join(summary_lines),
                extra=self._runner_log_extra(phase="live"),
            )
        except Exception:
            logger.warning(
                "Failed to generate live trading summary",
                exc_info=True,
                extra=self._runner_log_extra(phase="live"),
            )
