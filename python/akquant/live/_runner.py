# -*- coding: utf-8 -*-
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

from ..akquant import Bar, DataFeed, Engine, Instrument
from ..backtest import BacktestStreamEvent
from ..gateway.broker_event_adapter import map_order_snapshot, map_trade
from ..gateway.broker_runtime import BrokerRuntime
from ..gateway.factory import create_gateway_bundle
from ..gateway.models import BrokerCapability
from ..gateway.order_submitter import (
    build_live_order_client_ids,
    find_live_close_position,
    resolve_live_order_legs,
    validate_live_order_client_ids,
)
from ..gateway.symbol_match import normalize_symbol_for_match
from ..gateway.trader_base import TraderGatewayBase
from ..indicator_recording import IndicatorSink
from ..log import build_log_extra, get_logger
from ..strategy import InstrumentSnapshot, Strategy, StrategyRuntimeConfig
from ..strategy_loader import resolve_strategy_input
from ..strategy_runtime_config import apply_strategy_runtime_config
from ..utils import format_metric_value
from ._gateway_setup import (
    LEGACY_GATEWAY_OPTION_KEYS,
    normalize_gateway_options,
    start_gateway_thread,
)
from ._payload_utils import (
    is_terminal_status,
    normalize_broker_recovery_mode,
    payload_field,
    payload_to_dict,
)
from ._strategy_maps import (
    normalize_strategy_bool_map,
    normalize_strategy_float_map,
    normalize_strategy_int_map,
    validate_strategy_map_keys,
)
from ._stream_sink import StreamingIndicatorSink

logger = get_logger("gateway.live")

# Backward-compatible alias; canonical definition lives in _gateway_setup.
_LEGACY_GATEWAY_OPTION_KEYS = LEGACY_GATEWAY_OPTION_KEYS


def _instruments_to_snapshots(
    instruments: List[Instrument],
) -> Dict[str, InstrumentSnapshot]:
    """把 live 的 ``Instrument`` 列表转成策略侧可读的标的快照.

    回测在 ``run_backtest`` 里把 ``InstrumentConfig`` 灌进
    ``Strategy._set_instrument_snapshots``；live 此前完全没有这一步，
    ``_instrument_snapshots`` 恒为空 dict，导致 ``get_instrument()`` /
    ``get_instrument_field()`` 在实盘对**任何** symbol 都抛 KeyError。

    覆盖范围受 live 入口形态限制: ``run_live`` 只接 ``Instrument``(Rust 对象)，
    而它仅暴露 symbol/asset_type/multiplier/margin_ratio/tick_size/lot_size/
    option_margin_model/implied_volatility/reference_volatility 九个可读字段。
    option_type/strike_price/expiry_date/underlying_symbol/settlement_* 只在
    构造入参上存在、不可回读，故实盘快照里保持 None——期权策略若依赖这些字段，
    仍需自行传入，不能指望 ``get_instrument()``。
    """
    from ..backtest.engine import (
        _asset_type_to_upper_name,
        _option_margin_model_to_upper_name,
    )

    snapshots: Dict[str, InstrumentSnapshot] = {}
    for inst in instruments or []:
        symbol = str(inst.symbol)
        snapshots[symbol] = InstrumentSnapshot(
            symbol=symbol,
            asset_type=_asset_type_to_upper_name(inst.asset_type),
            multiplier=float(inst.multiplier),
            margin_ratio=float(inst.margin_ratio),
            tick_size=float(inst.tick_size),
            lot_size=float(inst.lot_size),
            option_margin_model=_option_margin_model_to_upper_name(
                inst.option_margin_model
            ),
            implied_volatility=(
                float(inst.implied_volatility)
                if inst.implied_volatility is not None
                else None
            ),
            reference_volatility=(
                float(inst.reference_volatility)
                if inst.reference_volatility is not None
                else None
            ),
        )
    return snapshots


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
        strategy_runtime_config: Optional[
            Union[StrategyRuntimeConfig, Dict[str, Any]]
        ] = None,
        runtime_config_override: bool = True,
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
        :param broker: Broker supplying **both** market data and trading
            (default ``"ctp"``). Used when ``market_broker`` / ``trader_broker``
            are both unset.
        :param market_broker: Market-data source, overriding ``broker`` for that
            side. Must be given together with ``trader_broker``; passing only one
            raises ``ValueError``. When both are given, ``broker`` is unused.
        :param trader_broker: Trading source, the counterpart of
            ``market_broker``. Use the pair to combine two single-sided brokers
            (``replay`` is market-data-only; some plugins are trading-only).
        :param initialize: Optional function-style initialize callback.
        :param on_start: Optional function-style on_start callback.
        :param on_stop: Optional function-style on_stop callback.
        :param on_tick: Optional function-style on_tick callback.
        :param on_order: Optional function-style on_order callback.
        :param on_trade: Optional function-style on_trade callback.
        :param on_reject: Optional function-style on_reject callback.
        :param on_broker_connected: Optional callback fired once the broker
            trader gateway reports readiness (heartbeat) after connect+start.
        :param broker_ready_timeout: Max seconds to poll trader_gateway.heartbeat()
            before giving up (default 30.0, sized for a real broker login such as
            QMF's dual-session login plus WS connect).
        :param broker_ready_required: If True (default), raise when broker
            readiness is not reached within broker_ready_timeout; pass False to
            only warn. Readiness is only polled in ``trading_mode='broker_live'``,
            so paper runs are unaffected. Defaulting to True is deliberate: while
            not ready, every submit raises RuntimeError anyway (see
            ``order_submitter``), so continuing to run buys no capability — it only
            trades one easily-missed warning for a stream of hard-to-attribute
            exceptions.
        :param on_before_trading: Optional function-style on_before_trading callback.
        :param on_after_trading: Optional function-style on_after_trading callback.
        :param on_cross_section:
            Optional function-style on_cross_section callback.
        :param on_portfolio_update:
            Optional function-style on_portfolio_update callback.
        :param on_error: Optional function-style on_error callback.
        :param on_timer: Optional function-style on_timer callback.
        :param context: Optional context dict injected into function-style strategy.
        :param on_broker_event: Optional broker event observer callback.
        :param signal_port_ready: Optional callback receiving a ``SignalPort`` right
            before the engine loop starts. Use it to hand the port to an external
            signal source (HTTP webhook / MQ consumer) running on its own thread.
            Only meaningful for ``trading_mode='paper'``.
        """
        self.strategy_cls = strategy_cls
        self.strategy_source = strategy_source
        self.strategy_loader = strategy_loader
        self.strategy_loader_options = strategy_loader_options
        self.strategy_id = (strategy_id or "_default").strip() or "_default"
        self.strategies_by_slot = strategies_by_slot or {}
        self.instruments = instruments
        self._subscribed_symbols_cache: set[str] | None = None
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
        # 行情源与交易源可分别覆盖 broker: 补齐单边 broker
        # (qmf 只有交易 / replay 只有行情)。
        self.market_broker = market_broker
        self.trader_broker = trader_broker
        self.trading_mode = trading_mode
        self.initialize = initialize
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_tick = on_tick
        self.on_order = on_order
        self.on_trade = on_trade
        self.on_reject = on_reject
        self.on_broker_connected = on_broker_connected
        self.broker_ready_timeout = broker_ready_timeout
        self.broker_ready_required = broker_ready_required
        self.on_before_trading = on_before_trading
        self.on_after_trading = on_after_trading
        self.on_cross_section = on_cross_section
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
        self.signal_port_ready = signal_port_ready
        self.signal_source = signal_source
        self.strategy_runtime_config = strategy_runtime_config
        self.runtime_config_override = bool(runtime_config_override)
        self._signal_dispatcher: Any = None
        # Indicator streaming wiring (set via set_indicator_stream / run_live).
        self._indicator_recorder_override: Optional[IndicatorSink] = None
        self._stream_on_event: Optional[Callable[[BacktestStreamEvent], None]] = None
        self._init_broker_bridge_state()

        self.feed = DataFeed.create_live()  # type: ignore
        self.engine = Engine()

        # 收尾派发的目标: run() 里策略创建成功后写入。预置 None/空是因为
        # run() 在创建策略之前就可能抛(网关工厂、broker_live 校验), 那时
        # 收尾派发也会被调用。
        self._stop_target_strategy: Optional[Strategy] = None
        self._stop_target_slot_strategies: Dict[str, Strategy] = {}

    def set_indicator_stream(
        self,
        *,
        indicator_recorder: Optional[IndicatorSink] = None,
        on_event: Optional[Callable[[BacktestStreamEvent], None]] = None,
    ) -> None:
        """Configure realtime indicator streaming for this live session.

        :param indicator_recorder: Optional public :class:`IndicatorSink` to
            collect indicator points; when omitted a streaming sink is used if
            ``on_event`` is provided.
        :param on_event: Optional stream event callback receiving
            ``indicator_point`` / ``indicator_snapshot`` events.
        """
        self._indicator_recorder_override = indicator_recorder
        self._stream_on_event = on_event

    def _all_instruments_are_futures(self) -> bool:
        """本次会话的标的是否全为期货.

        决定用哪种中国市场配置: ``use_china_futures_market()`` 只配期货费率,
        ``stock`` / ``fund`` / ``option`` 均为 ``None``, Rust 侧撮合遇到非期货
        订单会 panic(``src/market/china.rs``); ``use_china_market()`` 配齐四类。
        与回测的资产类型分支(``backtest/engine.py``)对齐。

        无标的时返回 True, 保持改动前的行为不变。
        """
        from ..backtest.engine import _asset_type_to_upper_name

        instruments = getattr(self, "instruments", None) or []
        if not instruments:
            return True
        return all(
            _asset_type_to_upper_name(inst.asset_type) == "FUTURES"
            for inst in instruments
        )

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

        # 市场配置按资产类型分支: 只配期货费率时, 股票/基金/期权订单会在 Rust
        # 撮合层 panic(找不到对应配置)。回测早有此分支, live 此前一直无条件用
        # 期货配置, 导致任何含股票标的的 paper 会话下单即 panic。
        if self._all_instruments_are_futures():
            self.engine.use_china_futures_market()
        else:
            self.engine.use_china_market()
        self.engine.set_force_session_continuous(True)

        symbols = [inst.symbol for inst in self.instruments]
        gateway_kwargs = self._build_gateway_kwargs()
        bundle = create_gateway_bundle(
            broker=self.broker,
            market_broker=getattr(self, "market_broker", None),
            trader_broker=getattr(self, "trader_broker", None),
            feed=self.feed,
            symbols=symbols,
            use_aggregator=self.use_aggregator,
            **gateway_kwargs,
        )
        self._broker_capabilities = bundle.trader_capabilities
        # 供 _configure_strategy_slots 装配 subscribe() 转发用: 行情网关在此刻已
        # 建好, 但策略的 on_start(里面才调 subscribe)要等到 engine.run() 才触发,
        # 所以只能存下网关引用, 到 subscribe 发生的当刻再运行期下发。
        self._market_gateway = bundle.market_gateway

        if bundle.market_gateway is not None:
            logger.info(
                "Starting %s market gateway",
                self.broker,
                extra=self._runner_log_extra(phase="gateway"),
            )
            self._start_gateway_thread(
                bundle.market_gateway.start, f"{self.broker}-market"
            )
        else:
            self._warn_if_no_market_gateway(bundle)

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
        # 记录收尾目标: finally 里不能用局部变量 —— 策略创建之前就可能抛,
        # 那时局部名尚未绑定。
        self._stop_target_strategy = strategy_instance
        self._stop_target_slot_strategies = slot_strategy_instances
        # 运行时配置下发放在最前: indicator_mode 决定指标注册走哪条路, 而指标在
        # on_start 里注册(主策略由 Rust 在 engine.run() 内触发, 槽位由
        # _dispatch_slot_strategy_start 触发) —— 两者都在下面。
        self._apply_runtime_config(
            [strategy_instance, *slot_strategy_instances.values()]
        )
        self._configure_strategy_slots(
            strategy_instance, slot_strategy_instances, effective_strategy_id
        )
        self._inject_data_freq(
            [strategy_instance, *slot_strategy_instances.values()], bundle
        )
        self._attach_indicator_stream(
            [strategy_instance, *slot_strategy_instances.values()]
        )
        # Broker callbacks + submit/state/cancel overrides are broker_live-only.
        # A non-broker_live run (e.g. paper) uses simulated execution, so must not
        # route submit/reads/cancel to a real trader gateway even if one was built.
        if self.trading_mode == "broker_live" and bundle.trader_gateway is not None:
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

        # slot 子策略的 on_start: Rust 只对主策略调 _on_start_internal, 回测
        # 在 Python 侧显式补(backtest/engine.py:3172-3176), 实盘此前没有 ——
        # slot 的 on_start/subscribe/指标注册全部不触发。放在 duration 处理
        # 之前: on_start 里可能注册 timer, 早于 deadline 设置更符合"启动完成
        # 再计时"。
        self._dispatch_slot_strategy_start(slot_strategy_instances)

        # Apply duration limit if specified
        if duration:
            logger.info(
                "Live auto-stop enabled: %s",
                duration,
                extra=self._runner_log_extra(phase="live"),
            )
            self._apply_time_limit(strategy_instance, duration)

        # 有界会话: broker 在 metadata 声明预期事件总数时, 事件放完即结束。
        # 键名不含具体 broker 名, 任何有界 live 数据源均可复用(另一个读
        # bundle.metadata 的地方是 _inject_data_freq 的 "freq" 键)。duration 仍作
        # 安全网: 若该总数因故永远达不到(例如数据含引擎会丢弃的事件), 墙钟兜底
        # 避免挂死。
        bounded_total = 0
        if bundle.metadata:
            try:
                bounded_total = int(bundle.metadata.get("bounded_event_total", 0) or 0)
            except (TypeError, ValueError):
                bounded_total = 0
        if bounded_total > 0:
            logger.info(
                "Bounded live session: will stop after %s market events",
                bounded_total,
                extra=self._runner_log_extra(phase="live"),
            )
            self._apply_bounded_event_limit(strategy_instance, bounded_total)

        # 外部信号端口: 必须在 engine.run() 之前取。run() 会独占可变借用引擎对象,
        # 之后再取会撞 "Already borrowed"(见 signal-ingestion-rfc.md 4.2.1)。
        # 取到的是 channel sender 的克隆, 与引擎对象解耦, 可安全交给任意线程。
        if self.signal_port_ready is not None:
            self._deliver_signal_port(effective_strategy_id)

        if self.signal_source is not None:
            self._start_signal_source(
                effective_strategy_id,
                [strategy_instance, *slot_strategy_instances.values()],
            )

        logger.info(
            "Running live strategy loop",
            extra=self._runner_log_extra(phase="live"),
        )
        # 收尾摘要的标题要如实反映停止原因: 因错误中止的会话若仍自称
        # "Manual Stop", 看起来就像正常手动停止, 而那条 CRITICAL 往往已被
        # 几十行日志淹没(实测反馈里"任务看着跑完了其实没跑"即此)。
        stop_reason = "Manual Stop"
        try:
            self.engine.run(strategy_instance, show_progress=show_progress)
        except KeyboardInterrupt:
            logger.info(
                "Stopping live runner by user interrupt or duration limit",
                extra=self._runner_log_extra(phase="live"),
            )
        except Exception as exc:
            # 系统级致命: 实盘 runner 整体因未捕获异常停止, 交易中断, 需人工立即介入。
            stop_reason = "ABORTED ON ERROR"
            logger.critical(
                "Stopping live runner due to error: %s",
                exc,
                exc_info=True,
                extra=self._runner_log_extra(phase="live"),
            )
        finally:
            # 策略收尾排最前: on_stop 里的撤单/平仓需要 broker 通道仍然活着,
            # 而摘要排最后才能反映 on_stop 落盘后的最终状态。
            self._dispatch_strategy_stop()
            self._stop_signal_source()
            self._stop_broker_dispatcher()
            self._print_summary(stop_reason)

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
                on_before_trading=getattr(self, "on_before_trading", None),
                on_after_trading=getattr(self, "on_after_trading", None),
                on_cross_section=getattr(self, "on_cross_section", None),
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

    def _dispatch_slot_strategy_start(
        self, slot_strategies: Dict[str, Strategy]
    ) -> None:
        """触发 slot 子策略的 on_start.

        主策略**刻意不在这里调**: 它的 `on_start` 由 Rust 在 `engine.run()`
        内触发(`src/engine/python.rs:1167`)。若在 Python 侧提前调, `on_start`
        里的 `subscribe()` / timer 注册会挪到 `_apply_time_limit` 之前发生,
        改变现有实盘时序(Rust 那次因 `_start_initialized` 幂等不会重复执行,
        但顺序变了)。slot 没有这个约束 —— 它此前一次都不触发。

        异常**不吞**: 启动期失败应当让会话起不来, 与收尾期(见
        `_dispatch_strategy_stop`)吞异常的取舍相反。对齐回测
        `backtest/engine.py:3172-3176` 同样不包 try/except 的写法。
        """
        for slot_strategy in slot_strategies.values():
            if hasattr(slot_strategy, "_on_start_internal"):
                slot_strategy._on_start_internal()
            elif hasattr(slot_strategy, "on_start"):
                slot_strategy.on_start()

    def _dispatch_strategy_stop(self) -> None:
        """会话收尾: 触发主策略与 slot 子策略的 on_stop(异常隔离).

        与回测(`backtest/engine.py:4757`)的差异有两处, 都是刻意的:

        1. 走 `_on_stop_live_internal` 而非 `_on_stop_internal` —— 后者串了
           三个回测语义的数据覆盖校验, 实盘会误报, 且其中
           `_check_incremental_hl_bar_coverage` 会抛异常打断收尾。
        2. **不重抛** `StrategyConfigurationError` —— 会抛它的那三个校验根本
           不在实盘收尾路径里; 而实盘的 `finally` 抛出会盖掉 `try` 里真正的
           停止原因, 包括标记 `ABORTED ON ERROR` 的 CRITICAL, 使"任务看着
           跑完了其实没跑"重现。一律吞成 ERROR。

        主策略与每个 slot 各自独立 try/except: 一个 slot 的 on_stop 抛错不能
        阻止其余 slot 收尾。
        """
        targets: List[tuple[str, Strategy]] = []
        main_strategy = getattr(self, "_stop_target_strategy", None)
        if main_strategy is not None:
            targets.append(("on_stop", main_strategy))
        for slot_strategy in getattr(self, "_stop_target_slot_strategies", {}).values():
            targets.append(("slot on_stop", slot_strategy))

        for label, target in targets:
            try:
                if hasattr(target, "_on_stop_live_internal"):
                    target._on_stop_live_internal()
                elif hasattr(target, "on_stop"):
                    target.on_stop()
            except Exception as exc:  # noqa: BLE001 — 收尾失败不改变会话结果
                logger.error(
                    "Error in %s: %s",
                    label,
                    exc,
                    exc_info=True,
                    extra=self._runner_log_extra(phase="strategy"),
                )

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
            # 标记实盘: 让 Strategy.subscribe() 能告知用户它在实盘不生效
            # (实盘订阅集来自 run_live(instruments=...))。
            mark_live = getattr(target, "_set_live_market_data_owner", None)
            if callable(mark_live):
                mark_live()
            # 无条件置空 _symbol_whitelist(而非只在它已是 None 时跳过): 与
            # run_from_checkpoint 对 _symbol_whitelist / _strategy_fill_policy_map
            # 等字段的处理同一类问题(见该函数与 commit ac277ae 的同名注释)——
            # `strategy_cls` 允许直接传一个已构造好的 Strategy 实例(见
            # `_build_strategy_instance`), 该实例若来自 `load_checkpoint()`
            # 恢复(pickle 整体恢复 `__dict__`), 会带着上一段回测遗留的非 None
            # `_symbol_whitelist`。`Strategy.subscribe()` 对白名单的校验不区分
            # 回测/实盘(见该方法文档"实盘不做白名单校验", 但代码本身没有按
            # `_live_market_data_owner` 分支跳过这段检查), 残留的白名单会让
            # `on_start`/`initialize` 里的正常 `subscribe()` 被上一次回测的
            # 白名单误拦, 是与已修三个 Critical 同一类"pickle 残留污染新会话"
            # 的问题。实盘本就不设白名单, 无条件清空是安全的。
            setattr(target, "_symbol_whitelist", None)

        strategy_targets = [strategy_instance, *slot_strategy_instances.values()]
        self._install_subscription_forwarder(strategy_targets)
        # getattr 兜底与同方法内 trading_mode 一致: 允许绕过 __init__ 的调用方。
        instrument_snapshots = _instruments_to_snapshots(
            getattr(self, "instruments", None) or []
        )
        for target in strategy_targets:
            target._set_instrument_snapshots(instrument_snapshots)
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

    def _warn_if_no_market_gateway(self, bundle: Any) -> None:
        """trader-only broker 无行情网关时告警.

        ``broker='middleware'``/``'qmf'`` 这类只有交易通道的 broker 会让
        ``market_gateway`` 为 ``None``: ``on_bar``/``on_tick`` 永不触发,
        ``current_tick`` 恒为 ``None``。这是**预期行为**(见 ``run_live``
        docstring), 但启动阶段此前完全静默, 用户只看到"策略没反应", 无从判断
        是配置错了还是本就没有行情。

        已显式配 ``market_broker`` 却仍没有行情网关时不告警: 那说明所配的行情
        broker 自己没给出网关, 属于配置错误, 该由 factory/该 broker 报, 这里
        再劝一句"去配 market_broker"只会误导。
        """
        if getattr(bundle, "market_gateway", None) is not None:
            return
        if str(getattr(self, "market_broker", "") or "").strip():
            logger.warning(
                "broker '%s' 未提供行情网关: on_bar/on_tick 不会触发, "
                "current_tick 恒为 None",
                self.broker,
                extra=self._runner_log_extra(phase="gateway"),
            )
            return
        logger.warning(
            "broker '%s' 只提供交易通道(无行情网关): on_bar/on_tick 不会触发, "
            "current_tick 恒为 None。若策略依赖行情, 用 "
            "run_live(market_broker='<行情源>', trader_broker='%s') "
            "把行情与交易分开指定",
            self.broker,
            self.broker,
            extra=self._runner_log_extra(phase="gateway"),
        )

    def _apply_runtime_config(self, targets: list[Strategy]) -> None:
        """把入口传入的 ``strategy_runtime_config`` 下发到主策略与各槽位策略.

        与回测的 ``run_backtest`` 对称(``backtest/engine.py`` 里同一个
        ``apply_strategy_runtime_config``), 冲突检测与告警去重因此两侧一致。
        入口没传就整个跳过 —— 策略自设的 ``self.runtime_config`` 保持不动。

        **必须早于任何 ``on_start``**: ``runtime_config`` 含 ``indicator_mode``,
        它决定指标走增量还是预计算, 而指标在 ``on_start`` 里注册。主策略的
        ``on_start`` 由 Rust 在 ``engine.run()`` 内触发, 槽位策略的由
        ``_dispatch_slot_strategy_start`` 触发, 本方法在两者之前调用。

        :param targets: 主策略与各槽位策略实例
        """
        if self.strategy_runtime_config is None:
            return
        for target in targets:
            if not isinstance(target, Strategy):
                continue
            apply_strategy_runtime_config(
                target,
                self.strategy_runtime_config,
                self.runtime_config_override,
                logger,
            )

    def _inject_data_freq(self, targets: list[Strategy], bundle: Any) -> None:
        """把行情网关声明的数据周期注入 ``self.freq``(只读属性).

        周期从 ``GatewayBundle.metadata['freq']`` 取, 由各 broker 的 builder 回填
        (klinedata 把自己的 ``period="M1"`` 转成回测侧口径 ``"1min"``)。表示统一用
        回测侧写法, 策略代码因此无需按 broker 写兼容分支 —— 这是把转换放在网关侧
        而非策略侧的理由。

        **没有声明就保持 ``None``**: CTP 等只有逐笔源的网关、trader-only broker
        (无行情通道)都属于这种情形。此处刻意不做任何推断或兜底默认值——错误的
        周期比未知的周期更危险(按周期折年化会静默错一个数量级)。

        :param targets: 主策略与各槽位策略实例
        :param bundle: 已构建的 ``GatewayBundle``
        """
        metadata = getattr(bundle, "metadata", None)
        freq = None
        if metadata:
            raw = metadata.get("freq")
            if raw is not None:
                freq = str(raw).strip() or None
        for target in targets:
            setattr(target, "_framework_freq", freq)
        if freq is None:
            logger.debug(
                "行情网关未声明数据周期: self.freq 为 None",
                extra=self._runner_log_extra(phase="gateway"),
            )
        else:
            logger.info(
                "数据周期: self.freq = %r",
                freq,
                extra=self._runner_log_extra(phase="gateway"),
            )

    def _install_subscription_forwarder(self, targets: list[Strategy]) -> None:
        """给各策略装上 subscribe() → 行情网关的运行期转发器.

        没有行情网关时(如 ``broker='qmf'`` 的 ``market_gateway=None``)不装,
        ``Strategy.subscribe()`` 便退回告警——那类 broker 确实无处可下发。

        下发的是 ``instruments ∪ 全部 slot 的 _subscriptions`` 的**并集**:
        网关的 ``subscribe()`` 是整集替换语义, 只发增量会静默退订其余标的。
        """
        market_gateway = getattr(self, "_market_gateway", None)
        if market_gateway is None:
            return
        subscribe = getattr(market_gateway, "subscribe", None)
        if not callable(subscribe):
            return

        def _forward() -> None:
            symbols: list[str] = []
            for instrument in getattr(self, "instruments", None) or []:
                symbol = str(getattr(instrument, "symbol", "") or "").strip()
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
            for target in targets:
                for symbol in getattr(target, "_subscriptions", None) or []:
                    symbol = str(symbol).strip()
                    if symbol and symbol not in symbols:
                        symbols.append(symbol)
            if not symbols:
                return
            try:
                subscribe(symbols)
            except Exception:  # noqa: BLE001 订阅失败不该打断策略回调
                logger.exception(
                    "forwarding subscribe(%s) to %s market gateway failed",
                    symbols,
                    self.broker,
                    extra=self._runner_log_extra(phase="gateway"),
                )

        for target in targets:
            setattr(target, "_live_subscription_forwarder", _forward)

    def _attach_indicator_stream(self, targets: list[Strategy]) -> None:
        """Attach an indicator sink to strategies and flush snapshots per cycle.

        Mirrors the backtest indicator recorder attachment. Uses the injected
        ``indicator_recorder`` when provided, else a non-accumulating
        :class:`StreamingIndicatorSink` when ``on_event`` is set. When neither is
        configured, live runs record no indicators (unchanged legacy behavior).
        """
        recorder: IndicatorSink | None = self._indicator_recorder_override
        if recorder is None and self._stream_on_event is not None:
            recorder = StreamingIndicatorSink(
                self._stream_on_event,
                run_id=str(self.strategy_id),
            )
        if recorder is None:
            return
        for target in targets:
            setattr(target, "_indicator_recorder", recorder)
        flush = getattr(recorder, "flush_stream_snapshot", None)
        if not callable(flush):
            return
        for target in targets:
            self._wrap_callback_with_flush(target, "on_bar", flush)
            if hasattr(target, "on_tick"):
                self._wrap_callback_with_flush(target, "on_tick", flush)

    @staticmethod
    def _wrap_callback_with_flush(
        strategy: Strategy, method_name: str, flush: Callable[[], None]
    ) -> None:
        """Wrap a strategy callback so the indicator snapshot flushes after it."""
        original = getattr(strategy, method_name, None)
        if not callable(original):
            return

        def wrapped(event: Any, _original: Any = original) -> None:
            _original(event)
            flush()

        setattr(strategy, method_name, wrapped)

    def _normalize_strategy_float_map(
        self, values: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        return normalize_strategy_float_map(values)

    def _normalize_strategy_int_map(
        self, values: Optional[Dict[str, int]]
    ) -> Dict[str, int]:
        return normalize_strategy_int_map(values)

    def _normalize_strategy_bool_map(
        self, values: Optional[Dict[str, bool]]
    ) -> Dict[str, bool]:
        return normalize_strategy_bool_map(values)

    def _validate_strategy_map_keys(
        self, values: Dict[str, Any], configured_slot_ids: List[str], field_name: str
    ) -> None:
        validate_strategy_map_keys(values, configured_slot_ids, field_name)

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
        return normalize_gateway_options(gateway_options, **legacy_values)

    def _build_gateway_kwargs(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], dict(self.gateway_options))

    def _start_gateway_thread(self, target: Any, name: str) -> None:
        start_gateway_thread(target, name)

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
        self._client_to_group_ids: dict[str, str] = {}
        self._order_requests: dict[str, Any] = {}
        self._broker_to_local_stop_id: dict[str, str] = {}
        self._client_to_strategy_ids: dict[str, str] = {}
        self._broker_to_strategy_ids: dict[str, str] = {}
        self._closed_broker_order_ids: set[str] = set()
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
        # 会话标记: 让 client_order_id 跨进程/跨重启唯一(见 _next_client_order_id)。
        self._broker_session_tag = uuid.uuid4().hex[:8]
        self._broker_submit_lock = threading.Lock()
        self._broker_recovery_mode = self._normalize_broker_recovery_mode(
            getattr(self, "gateway_options", {}).get("recovery_mode", "compatible")
        )
        self._broker_recovery_last_error_key = ""
        self._broker_baseline_done = False
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
            record_order_request=self._record_order_request,
            adapt_strategy_payload=self._adapt_strategy_payload,
            record_stop_remap=self._record_stop_remap,
            should_replay_trades=lambda: self._broker_baseline_done,
            sync_group_mapping=self._sync_group_mapping,
            group_broker_ids=self._broker_order_ids_for_group,
            resolve_trace_id=self._lookup_group_id,
            get_subscribed_symbols=self._subscribed_symbol_set,
            is_known_order=self._is_known_broker_order,
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
            strategy_limits={
                "max_order_value": dict(
                    getattr(self, "strategy_max_order_value", {}) or {}
                ),
                "max_order_size": dict(
                    getattr(self, "strategy_max_order_size", {}) or {}
                ),
                "max_position_size": dict(
                    getattr(self, "strategy_max_position_size", {}) or {}
                ),
            },
        )

    def _start_signal_source(
        self, effective_strategy_id: str | None, strategies: list[Any]
    ) -> None:
        """按 trading_mode 选下单出口, 装好 dispatcher 并启动信号源.

        两种模式的风控覆盖面不同(见 akquant/signal/__init__.py 的说明):
        paper 经引擎注入走完整风控; broker_live 经柜台通道, 只有策略级三项限额
        前置生效 —— 因为引擎的实盘执行器不向柜台报单。
        """
        from ..signal.dispatcher import SignalDispatcher
        from ..signal.sinks import BrokerOrderSink, PaperOrderSink

        source: Any = self.signal_source
        if source is None:
            return
        try:
            if self.trading_mode == "broker_live":
                submitter = getattr(self, "_broker_order_submitter", None)
                if submitter is None:
                    raise RuntimeError(
                        "broker_live 下 signal_source 需要已装配的柜台下单器; "
                        "请检查 trader_broker 配置"
                    )
                sink: Any = BrokerOrderSink(submitter)
            else:
                port = cast(Any, self.engine).signal_port(effective_strategy_id)
                sink = PaperOrderSink(port)

            dispatcher = SignalDispatcher(
                sink, on_result=getattr(source, "on_result", None)
            )
            self._signal_dispatcher = dispatcher
            self._bind_signal_reject_hook(strategies, dispatcher)
            source.bind(dispatcher.dispatch)
            source.start()
            logger.info(
                "Signal source started (%s sink)",
                sink.mode,
                extra=self._runner_log_extra(phase="signal"),
            )
        except Exception as exc:  # noqa: BLE001 — 装配失败必须显式中止, 见下
            logger.critical(
                "信号源启动失败, 会话中止: %s",
                exc,
                exc_info=True,
                extra=self._runner_log_extra(phase="signal"),
            )
            # 与 SignalPort 交付失败不同: 那只是少一个可选能力, 而信号源是本次
            # 会话的**唯一订单来源**, 静默继续会跑出一个永不下单的空会话。
            raise

    def _bind_signal_reject_hook(self, strategies: list[Any], dispatcher: Any) -> None:
        """把 dispatcher.handle_reject 串到各策略的 on_reject 前面.

        用包装而非替换: 用户自己的 on_reject 仍要照常触发。
        """
        for strategy in strategies:
            if strategy is None:
                continue
            original = getattr(strategy, "on_reject", None)

            def wrapped(order: Any, _original: Any = original) -> None:
                dispatcher.handle_reject(order)
                if callable(_original):
                    _original(order)

            setattr(strategy, "on_reject", wrapped)

    def _stop_signal_source(self) -> None:
        """停掉信号源(会话结束时调用, 异常隔离)."""
        source = self.signal_source
        if source is None:
            return
        try:
            source.stop()
        except Exception:  # noqa: BLE001 — 收尾失败不改变会话结果
            logger.warning(
                "信号源停止时抛出异常",
                exc_info=True,
                extra=self._runner_log_extra(phase="signal"),
            )

    def _deliver_signal_port(self, effective_strategy_id: str | None) -> None:
        """把 SignalPort 交给调用方(异常隔离: 交付失败不应中断会话启动).

        broker_live 下额外告警: 该模式的 `RealtimeExecutionClient` 不报柜台,
        经端口注入的订单会过风控进 active_orders 却永不成交, 静默失效风险高。
        """
        callback = self.signal_port_ready
        if callback is None:
            return
        if self.trading_mode == "broker_live":
            logger.warning(
                "signal_port_ready 在 trading_mode='broker_live' 下不能用于真实下单: "
                "引擎实盘执行器不报柜台, 经 SignalPort 注入的订单会通过风控并进入"
                "活动委托, 但既不撮合也不到柜台。broker_live 的外部信号请走 "
                "broker 下单通道",
                extra=self._runner_log_extra(phase="live"),
            )
        try:
            port = cast(Any, self.engine).signal_port(effective_strategy_id)
            callback(port)
        except Exception as exc:  # noqa: BLE001 — 交付失败不拖垮会话
            logger.error(
                "交付 SignalPort 失败: %s",
                exc,
                exc_info=True,
                extra=self._runner_log_extra(phase="live"),
            )

    def _baseline_broker_state(self, trader_gateway: Any) -> None:
        """就绪激活: 先丢弃待派发成交, 再急切 seed 各 slot 持仓, 最后灌 dedup 基线.

        闭合 pre-seed 入队/post-seed drain 过计窗口。dispatch/recovery 线程在
        ready 前已启动, 回调也已绑定, 故激活前可能已有实盘
        推送的成交事件排队等待派发; 若在 seed 之后才被 dispatch 线程 drain, 会对
        eager-seed 过的缓存 apply_fill 一次 delta, 而该成交其实已经烘进了快照 →
        持仓过计。丢弃在先: 这些事件是历史(已含在快照里), 策略尚未激活, 不应
        apply_fill/触发 on_trade 重放。

        seed 先/基线后 → 两查询间隙成交残留为轻微低计(较高计安全)。此后仅激活后
        新成交(新 trade_id)才 apply_fill; 基线前成交的恢复重放经 _seen 丢弃(不 apply、
        不重放 on_trade)。恢复的成交重放由 _broker_baseline_done 门控, 激活前不跑。
        """
        try:
            self._broker_event_bridge.discard_pending_trades()
        except Exception:  # noqa: BLE001 保守: 失败不阻塞后续 seed
            logger.exception("broker baseline discard pending trades failed")
        for cache in self._broker_runtime.state_caches:
            try:
                cache.positions()
            except Exception:  # noqa: BLE001 保守: 失败交给懒 seed
                logger.exception("broker baseline seed positions failed")
        sync = getattr(trader_gateway, "sync_today_trades", None)
        if callable(sync):
            try:
                self._broker_event_bridge.mark_trades_seen(
                    getattr(t, "trade_id", None) for t in sync()
                )
            except Exception:  # noqa: BLE001
                logger.exception("broker baseline sync_today_trades failed")
        self._broker_baseline_done = True

    def _warn_if_heartbeat_not_overridden(self, trader_gateway: Any) -> None:
        """Warn when a gateway inherits the always-true default heartbeat.

        ``TraderGatewayBase.heartbeat`` returns True unconditionally, so a custom
        broker that forgets to override it is always judged ready and
        ``broker_ready_required`` silently has no effect on it — a failed login
        would still be let through to order submission.
        """
        base_heartbeat = getattr(TraderGatewayBase, "heartbeat", None)
        gateway_heartbeat = getattr(type(trader_gateway), "heartbeat", None)
        if base_heartbeat is None or gateway_heartbeat is not base_heartbeat:
            return
        logger.warning(
            "%s 未覆写 heartbeat(), 将始终判定为已就绪; broker_ready_required "
            "对该 gateway 无效——请实现 heartbeat() 返回真实登录/连接状态",
            type(trader_gateway).__name__,
            extra=self._runner_log_extra(phase="gateway"),
        )

    def _await_broker_ready(self, trader_gateway: Any, targets: list) -> None:
        """Poll heartbeat() until ready/timeout; set broker_ready, fire callback."""
        self._warn_if_heartbeat_not_overridden(trader_gateway)
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
            self._baseline_broker_state(trader_gateway)
            self._dispatch_broker_connected(targets)
        else:
            logger.warning(
                "broker not ready within %ss",
                self.broker_ready_timeout,
                extra=self._runner_log_extra(phase="gateway"),
            )
            if self.broker_ready_required:
                # dispatcher/recovery 线程已在 _bind_broker_callbacks 中启动, 而本次
                # raise 发生在 run() 的 try 之外(finally 的清理不会执行)。此处不收尾,
                # 一次失败的启动就会留下一个每秒轮询柜台的 recovery 线程。
                self._stop_broker_dispatcher()
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
        return cast(
            "list[tuple[str, float]]",
            resolve_live_order_legs(
                trader_gateway=trader_gateway,
                capability=capability,
                symbol=symbol,
                side=side,
                quantity=quantity,
                position_effect=position_effect,
                reduce_only=reduce_only,
                payload_field=self._payload_field,
            ),
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
        return cast(
            "list[str]",
            build_live_order_client_ids(request_client_order_id, order_legs),
        )

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
            if not client_order_id and broker_order_id:
                client_order_id = self._resolve_client_order_id(broker_order_id)
            self._sync_order_id_mapping(client_order_id, broker_order_id)
            if broker_order_id:
                self._broker_order_states[broker_order_id] = payload
                status = self._payload_field(payload, "status")
                if self._is_terminal_status(status):
                    self._close_order_mapping(client_order_id, broker_order_id)
        elif event_name == "trade":
            broker_order_id = str(self._payload_field(payload, "broker_order_id"))
            client_order_id = str(self._payload_field(payload, "client_order_id"))
            if not client_order_id and broker_order_id:
                client_order_id = self._resolve_client_order_id(broker_order_id)
            self._sync_order_id_mapping(client_order_id, broker_order_id)
        elif event_name == "execution_report":
            broker_order_id = str(self._payload_field(payload, "broker_order_id"))
            client_order_id = str(self._payload_field(payload, "client_order_id"))
            self._sync_order_id_mapping(client_order_id, broker_order_id)
            status = self._payload_field(payload, "status")
            if self._is_terminal_status(status):
                self._close_order_mapping(client_order_id, broker_order_id)
        elif event_name == "account":
            self._broker_account_state = payload

    def _subscribed_symbol_set(self) -> set[str]:
        """本会话挂载标的的归一化集合(惰性构建并缓存).

        数据源是 ``self.instruments``, 与网关订阅集同源(``symbols`` 就是从它
        派生的), 保证过滤口径与订阅口径一致。``run_live`` 只要求
        ``instruments`` 非 ``None``(见 ``_facade.py:181-182``), ``[]`` 能合法
        通过校验, 因此空集合是可能出现的边界而非纯异常路径; 首次构建出空
        集合时打一条 WARNING, 否则标的过滤会悄悄退化成"全放行"却无人知晓。

        用 ``getattr`` 兜底缺失属性: 部分测试用
        ``LiveRunner.__new__(LiveRunner)`` + ``_init_broker_bridge_state()``
        绕开 ``__init__`` 直接构造替身, ``instruments`` / 缓存字段不存在;
        按裁决三"无访问器场景一律放行", 缺失时当空集处理。

        :return: 归一化后的标的集合。
        """
        cached = getattr(self, "_subscribed_symbols_cache", None)
        if cached is None:
            cached = {
                normalize_symbol_for_match(getattr(inst, "symbol", ""))
                for inst in (getattr(self, "instruments", None) or [])
            }
            cached.discard("")
            if not cached:
                logger.warning(
                    "Subscribed symbol set is empty; broker event symbol "
                    "filter has no effect (everything will pass through)",
                    extra=build_log_extra(phase="gateway"),
                )
            self._subscribed_symbols_cache = cached
        return cached

    def _is_known_broker_order(
        self, broker_order_id: str, client_order_id: str
    ) -> bool:
        """判断委托是否为本会话已知(已建立 id 映射)的订单.

        ``order_submitter.py`` 报单成功即调用 ``_sync_order_id_mapping`` 建立
        broker_order_id/client_order_id 互查映射; 命中即视为"这是我自己报出
        的单", 不论其标的是否落在 ``instruments`` 挂载集合内——外部信号源经
        ``BrokerOrderSink`` 直调 ``submitter.submit_order`` 天然会报出挂载
        集合之外的标的(不经引擎合约登记表), 见 ``signal/sinks.py:69``。
        跨会话恢复的老挂单不在映射表里, 交给标的判据兜底, 行为不变。

        :param broker_order_id: 柜台委托号(可能为空)。
        :param client_order_id: 客户端委托号(可能为空)。
        :return: 是否命中本会话已知订单映射。
        """
        broker_to_client = getattr(self, "_broker_to_client_order_ids", None) or {}
        client_to_broker = getattr(self, "_client_to_broker_order_ids", None) or {}
        if broker_order_id and broker_order_id in broker_to_client:
            return True
        if client_order_id and client_order_id in client_to_broker:
            return True
        return False

    def _make_event_key(self, event_name: str, payload: Any) -> str:
        if event_name == "trade":
            trade_id = str(self._payload_field(payload, "trade_id"))
            if trade_id:
                return f"trade:{trade_id}"
        if event_name == "order":
            broker_order_id = str(self._payload_field(payload, "broker_order_id"))
            status = str(self._payload_field(payload, "status"))
            filled_quantity = str(self._payload_field(payload, "filled_quantity"))
            # 刻意不含 timestamp_ns: 含它则每次重推键都变、批内去重也失效。
            # 跨轮去重由 BrokerEventBridge._seen_order_states 承担。
            return f"order:{broker_order_id}:{status}:{filled_quantity}"
        if event_name == "execution_report":
            broker_order_id = str(self._payload_field(payload, "broker_order_id"))
            status = str(self._payload_field(payload, "status"))
            filled_quantity = str(self._payload_field(payload, "filled_quantity"))
            # 与 order 键对称补回 filled_quantity: UnifiedExecutionReport 没有
            # 单独的本次成交量字段, 缺它同批内两条 status 相同但成交量不同的
            # 执行回报(连续部分成交)会被误判批内重复丢弃第二条。
            return f"execution_report:{broker_order_id}:{status}:{filled_quantity}"
        if event_name == "account":
            account_id = str(self._payload_field(payload, "account_id"))
            timestamp_ns = str(self._payload_field(payload, "timestamp_ns"))
            return f"account:{account_id}:{timestamp_ns}"
        return f"{event_name}:{id(payload)}"

    def _payload_field(self, payload: Any, field: str) -> Any:
        return payload_field(payload, field)

    def _next_client_order_id(self) -> str:
        """生成 client_order_id: ``{broker}-{会话标记}-{会话内序号}``.

        序号是实例字段, 每次 run_live 从 0 起。若只用序号(旧格式
        ``{broker}-coid-{seq}``), 重启后第一笔单又是 ``...-coid-1``, 会与柜台
        里同名的历史委托撞号——把 client_order_id 当幂等键的柜台会直接拒单
        (实测中间件返回 409 Conflict)。加 8 位会话标记后, 同一进程内递增序号
        保证顺序可读, 跨重启与多进程并行则由标记区分。
        """
        with self._broker_submit_lock:
            self._broker_submit_seq += 1
            tag = getattr(self, "_broker_session_tag", "") or "nosess"
            return f"{self.broker}-{tag}-{self._broker_submit_seq}"

    def _sync_order_id_mapping(
        self,
        client_order_id: str,
        broker_order_id: str,
    ) -> None:
        if client_order_id and broker_order_id:
            self._client_to_broker_order_ids[client_order_id] = broker_order_id
            self._broker_to_client_order_ids[broker_order_id] = client_order_id

    def _sync_group_mapping(self, client_order_id: str, group_id: str) -> None:
        if client_order_id and group_id:
            self._client_to_group_ids[client_order_id] = group_id

    def _lookup_group_id(self, payload: Any) -> str:
        cid = str(self._payload_field(payload, "client_order_id") or "").strip()
        if not cid:
            bid = str(self._payload_field(payload, "broker_order_id") or "").strip()
            cid = self._broker_to_client_order_ids.get(bid, "") if bid else ""
        return self._client_to_group_ids.get(cid, cid)

    def _broker_order_ids_for_group(self, group_id: str) -> list[str]:
        gid = str(group_id)
        result: list[str] = []
        # 用 list(...) 对 dict 做一次性快照: list() 构造在 CPython 下是单个
        # C 级原子操作, 中途不会释放 GIL, 因此不会观察到 broker 派发线程
        # (_close_order_mapping) 并发 pop 导致的 "dictionary changed size
        # during iteration"; 直接 for 一个存活的 dict 则会在每次迭代之间让出
        # GIL, 存在竞态崩溃风险。与本文件其余映射表的单操作原子性约定一致,
        # 无需额外加锁。
        for cid, mapped in list(self._client_to_group_ids.items()):
            if mapped != gid:
                continue
            bid = self._client_to_broker_order_ids.get(cid, "")
            if bid:
                result.append(bid)
        # group_id 本身即根 client id: 正常情况下 Task 5 已将根 leg 自身的
        # client_order_id 同步进 _client_to_group_ids, 上面的主循环即可收集到
        # 其 broker id; 这里保留作为防御性兜底 (belt-and-suspenders), 以防
        # 未来同步路径变化导致根 leg 未被计入, 而非当前逻辑所必需。
        root_bid = self._client_to_broker_order_ids.get(gid, "")
        if root_bid and root_bid not in result:
            result.insert(0, root_bid)
        return result

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
        return payload_to_dict(payload)

    def _resolve_client_order_id(self, broker_order_id: str) -> str:
        return self._broker_to_client_order_ids.get(broker_order_id, "")

    def _resolve_broker_order_id(self, client_order_id: str) -> str:
        return self._client_to_broker_order_ids.get(client_order_id, "")

    def _close_order_mapping(self, client_order_id: str, broker_order_id: str) -> None:
        if client_order_id:
            self._client_to_broker_order_ids.pop(client_order_id, None)
            self._client_to_strategy_ids.pop(client_order_id, None)
            self._order_requests.pop(str(client_order_id), None)
            # 注意: 不弹出 _client_to_group_ids —— 乱序到达的末笔成交仍需据此
            # 解析 group_id; 条目极小且随会话生命周期存在, 保留以保证 group
            # 关联正确 (见 #317 反手 open 腿)。
        if broker_order_id:
            self._broker_to_client_order_ids.pop(broker_order_id, None)
            self._broker_to_strategy_ids.pop(broker_order_id, None)
            self._closed_broker_order_ids.add(broker_order_id)
            self._broker_to_local_stop_id.pop(str(broker_order_id), None)

    # 线程安全说明: _order_requests / _broker_to_local_stop_id 由行情线程写
    # (_record_order_request/_record_stop_remap 经 submit / check_stop_triggers)、
    # 由 broker drain 线程读+清 (_lookup_* / _close_order_mapping)。每处都是单个
    # dict get/set/pop——CPython GIL 下原子, 无跨线程复合读改写, 故无需加锁; lookup
    # 与 pop 竞争只会得到 "命中" 或 None, 两者都是合法的事件时序结果。
    def _record_order_request(self, client_order_id: str, request: Any) -> None:
        self._order_requests[str(client_order_id)] = request

    def _lookup_order_request(self, payload: Any) -> Any:
        cid = self._payload_field(payload, "client_order_id")
        if not cid:
            bid = self._payload_field(payload, "broker_order_id")
            cid = self._broker_to_client_order_ids.get(str(bid)) if bid else None
        return self._order_requests.get(str(cid)) if cid else None

    def _record_stop_remap(self, local_id: str, broker_order_id: str) -> None:
        self._broker_to_local_stop_id[str(broker_order_id)] = str(local_id)

    def _lookup_stop_local_id(self, payload: Any) -> Any:
        bid = self._payload_field(payload, "broker_order_id")
        return self._broker_to_local_stop_id.get(str(bid)) if bid else None

    def _adapt_strategy_payload(self, event_name: str, payload: Any) -> Any:
        if event_name == "order":
            return map_order_snapshot(
                payload,
                request=self._lookup_order_request(payload),
                owner_strategy_id=self._resolve_owner_strategy_id(payload),
                local_id=self._lookup_stop_local_id(payload),
                group_id=self._lookup_group_id(payload),
            )
        if event_name == "trade":
            return map_trade(
                payload,
                request=self._lookup_order_request(payload),
                owner_strategy_id=self._resolve_owner_strategy_id(payload),
                local_id=self._lookup_stop_local_id(payload),
                group_id=self._lookup_group_id(payload),
            )
        return payload

    def _is_terminal_status(self, status: Any) -> bool:
        return is_terminal_status(status)

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
        return cast("dict[str, Any]", capability.as_execution_capabilities())

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
        return normalize_broker_recovery_mode(mode)

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
                # on_error 自身可能抛错; 不能让它逃出 → 否则会杀掉 broker drain 循环。
                try:
                    on_error(exc, callback_name, payload)
                except Exception as on_error_exc:
                    logger.warning(
                        "Strategy on_error handler raised",
                        exc_info=on_error_exc,
                        extra=build_log_extra(
                            phase="gateway",
                            strategy_id=owner_strategy_id,
                        ),
                    )

    def _apply_bounded_event_limit(self, strategy: Any, total_events: int) -> None:
        """让有界 live 会话在收到 ``total_events`` 个行情事件后自行结束.

        live 循环在 channel 空时返回 ``FeedAction::Wait`` 并无限循环
        (``src/data/feed.rs:358-366``), 而 ``Engine`` 未向 Python 暴露 ``stop()``。
        因此复用 ``duration`` 已验证的模式: 抛 ``KeyboardInterrupt``, 由 ``run()``
        的 ``except`` 接住并走正常收尾。

        计数点选在框架入口 ``_on_bar_event_and_flush`` /
        ``_on_tick_event_and_flush``(Rust 调用点 ``src/engine/core.rs:1280`` 与
        ``:1334``): 这两个是每个事件必经之路, warmup 与
        ``_active_start_time_ns`` 预热过滤只影响用户的 ``on_bar``, 不影响这一层,
        因此计数精确。

        本机制不含具体 broker 语义: 任何在 ``GatewayBundle.metadata`` 中声明
        ``bounded_event_total`` 的 broker 都能复用。
        """
        if total_events <= 0:
            return

        state = {"count": 0}

        def _make_wrapper(original: Any) -> Any:
            def wrapped(event: Any, ctx: Any) -> Any:
                # 先处理完本事件再判断, 使第 N 个事件不被截断。
                result = original(event, ctx)
                state["count"] += 1
                if state["count"] >= total_events:
                    raise KeyboardInterrupt(
                        f"bounded live session complete: {total_events} events"
                    )
                return result

            return wrapped

        for method_name in ("_on_bar_event_and_flush", "_on_tick_event_and_flush"):
            original = getattr(strategy, method_name, None)
            if original is None:
                continue
            setattr(strategy, method_name, _make_wrapper(original))

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

        # 墙钟兜底(主路径): 把截止时刻下沉到引擎的等待循环, 使会话在行情停摆时
        # 也能到点结束。此前只有下面的回调 patch, 而它在无行情时永不触发 ——
        # 行情一停就挂死(见 docs/zh/meta/signal-ingestion-rfc.md 4.6)。
        set_deadline = getattr(self.engine, "set_session_deadline_ns", None)
        if callable(set_deadline):
            set_deadline(int((start_time + duration_sec) * 1_000_000_000))
        else:
            logger.warning(
                "引擎不支持 set_session_deadline_ns, duration 在行情停摆时不会生效",
                extra=self._runner_log_extra(phase="live"),
            )

        # 回调 patch(保留): 有行情时能更早地在事件边界处停下, 与上面互补。
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

    def _print_summary(self, stop_reason: str = "Manual Stop") -> None:
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
                f"TRADING SUMMARY ({stop_reason})",
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
