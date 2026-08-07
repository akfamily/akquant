from typing import Any, Callable, Optional, cast

from ..akquant import OrderStatus, check_strategy_limits
from ..log import build_log_extra, get_logger
from .broker_event_adapter import StrategyOrder
from .broker_models import (
    BrokerCapability,
    UnifiedOrderRequest,
    normalize_asset_type,
    validate_broker_extra,
    validate_execution_semantics,
)
from .broker_strategy_api import _resolve_symbol
from .order_audit import record_reject, record_submit
from .order_receipt import OrderLeg, OrderReceipt

logger = get_logger("gateway.live")


def find_live_close_position(
    positions: Any,
    symbol: str,
    side: str,
    payload_field: Callable[[Any, str], Any],
) -> Any | None:
    """Find the position that should be reduced by the given order side."""
    target_symbol = str(symbol).strip()
    normalized_side = str(side).strip().lower()
    for position in positions or ():
        position_symbol = str(payload_field(position, "symbol")).strip()
        if position_symbol != target_symbol:
            continue
        position_direction = str(payload_field(position, "direction")).strip().lower()
        position_quantity = float(payload_field(position, "quantity") or 0.0)
        if normalized_side == "sell" and (
            position_direction in {"buy", "long"} or position_quantity > 0
        ):
            return position
        if normalized_side == "buy" and (
            position_direction in {"sell", "short"} or position_quantity < 0
        ):
            return position
    return None


def _split_close_legs(
    trader_gateway: Any,
    capability: BrokerCapability,
    symbol: str,
    side: str,
    quantity: float,
    supported_effects: set[str],
    payload_field: Callable[[Any, str], Any],
) -> list[tuple[str, float]]:
    """把一段平仓量拆成 close_today/close_yesterday；能力不足则整段 close."""
    if not capability.position_details:
        return [("close", quantity)]
    if (
        "close_today" not in supported_effects
        or "close_yesterday" not in supported_effects
    ):
        return [("close", quantity)]
    query_positions = getattr(trader_gateway, "query_positions", None)
    if not callable(query_positions):
        return [("close", quantity)]
    try:
        positions = query_positions()
    except Exception:
        return [("close", quantity)]
    target_position = find_live_close_position(positions, symbol, side, payload_field)
    if target_position is None:
        return [("close", quantity)]
    available_today = max(
        0.0, float(getattr(target_position, "available_today_quantity", 0.0) or 0.0)
    )
    available_yesterday = max(
        0.0, float(getattr(target_position, "available_yesterday_quantity", 0.0) or 0.0)
    )
    split_quantity = min(quantity, available_today + available_yesterday)
    legs: list[tuple[str, float]] = []
    close_today_quantity = min(split_quantity, available_today)
    if close_today_quantity > 0:
        legs.append(("close_today", close_today_quantity))
    remaining_quantity = max(split_quantity - close_today_quantity, 0.0)
    close_yesterday_quantity = min(remaining_quantity, available_yesterday)
    if close_yesterday_quantity > 0:
        legs.append(("close_yesterday", close_yesterday_quantity))
    unresolved_quantity = max(quantity - split_quantity, 0.0)
    if unresolved_quantity > 0 or not legs:
        legs.append(
            ("close", unresolved_quantity if unresolved_quantity > 0 else quantity)
        )
    return legs


def resolve_live_order_legs(
    trader_gateway: Any,
    capability: BrokerCapability,
    symbol: str,
    side: str,
    quantity: float,
    position_effect: str,
    reduce_only: bool,
    payload_field: Callable[[Any, str], Any],
) -> list[tuple[str, float]]:
    """Resolve close orders into close_today/close_yesterday legs when supported.

    Also resolves `auto` reversal orders (sell/buy quantity exceeding the opposite
    position) into close leg(s) plus a trailing `open` leg, unless the broker
    declares it handles reversal itself via `"auto_reverse" in capability.features`.
    """
    normalized_effect = str(position_effect).strip().lower()
    if quantity <= 0 or reduce_only:
        return [(normalized_effect, quantity)]

    supported_effects = {
        str(item).strip().lower() for item in capability.supported_position_effects
    }

    # 反手：auto 单卖出量超过反向持仓可用量时，核心拆成 平仓 + 开仓。
    # broker 若声明自身会反手(auto_reverse)，则不拆。
    if normalized_effect == "auto":
        if "auto_reverse" in capability.features:
            return [(normalized_effect, quantity)]
        if "open" not in supported_effects or "close" not in supported_effects:
            return [(normalized_effect, quantity)]
        query_positions = getattr(trader_gateway, "query_positions", None)
        if not callable(query_positions):
            return [(normalized_effect, quantity)]
        try:
            positions = query_positions()
        except Exception:
            return [(normalized_effect, quantity)]
        target = find_live_close_position(positions, symbol, side, payload_field)
        if target is None:
            return [(normalized_effect, quantity)]
        raw_today = payload_field(target, "available_today_quantity")
        raw_yesterday = payload_field(target, "available_yesterday_quantity")
        raw_available = payload_field(target, "available_quantity")
        has_availability = any(
            value is not None for value in (raw_today, raw_yesterday, raw_available)
        )
        available_split = max(0.0, float(raw_today or 0.0)) + max(
            0.0, float(raw_yesterday or 0.0)
        )
        available_quantity = max(0.0, float(raw_available or 0.0))
        raw_quantity = abs(float(payload_field(target, "quantity") or 0.0))
        # 可平口径与 _split_close_legs 一致: broker 报可用量时优先用今昨可用量,
        # 其次 available_quantity(即便为 0 也不回退——全冻结应交由下方 closable<=0
        # 门槛退回单腿 auto); 仅当 broker 完全不提供可用量字段时才退回原始持仓。
        if has_availability:
            closable = available_split if available_split > 0 else available_quantity
        else:
            closable = raw_quantity
        if closable <= 0 or quantity <= closable:
            return [(normalized_effect, quantity)]
        close_qty = closable
        open_qty = quantity - closable
        legs = _split_close_legs(
            trader_gateway,
            capability,
            symbol,
            side,
            close_qty,
            supported_effects,
            payload_field,
        )
        legs.append(("open", open_qty))
        return legs

    if normalized_effect != "close":
        return [(normalized_effect, quantity)]

    return _split_close_legs(
        trader_gateway,
        capability,
        symbol,
        side,
        quantity,
        supported_effects,
        payload_field,
    )


def build_live_order_client_ids(
    request_client_order_id: str, order_legs: list[tuple[str, float]]
) -> list[str]:
    """Build per-leg client order ids while keeping the first leg stable."""
    if len(order_legs) <= 1:
        return [request_client_order_id]
    client_order_ids = [request_client_order_id]
    for leg_index, (position_effect, _) in enumerate(order_legs[1:], start=2):
        suffix = str(position_effect).replace("_", "-")
        client_order_ids.append(f"{request_client_order_id}-{suffix}-{leg_index}")
    return client_order_ids


def validate_live_order_client_ids(
    strategy: Any,
    client_order_ids: list[str],
    symbol: str,
    side: str,
    quantity: float,
    can_submit_client_order: Callable[[str], bool],
    notify_strategy_error: Callable[[Any, Exception, str, Any], None],
    group_id: str | None = None,
) -> None:
    """Ensure all generated client order ids are available before submit."""
    owner_strategy_id = str(getattr(strategy, "_owner_strategy_id", "_default")).strip()
    owner_strategy_id = owner_strategy_id or "_default"
    for client_order_id in client_order_ids:
        if can_submit_client_order(client_order_id):
            continue
        exc = RuntimeError(f"duplicate active client_order_id: {client_order_id}")
        logger.warning(
            "Rejected live submit_order because client_order_id is already active",
            extra=build_log_extra(
                phase="gateway",
                strategy_id=owner_strategy_id,
                slot=owner_strategy_id if owner_strategy_id != "_default" else None,
                symbol=symbol,
                client_order_id=client_order_id,
            ),
        )
        record_reject(
            strategy_id=owner_strategy_id,
            symbol=symbol,
            client_order_id=client_order_id,
            reason="duplicate active client_order_id",
            side=side,
            quantity=quantity,
            trace_id=group_id,
        )
        notify_strategy_error(
            strategy,
            exc,
            "submit_order",
            {
                "client_order_id": client_order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
        )
        raise exc


class BrokerOrderSubmitter:
    """Inject broker-live submit_order helpers into a strategy instance."""

    def __init__(
        self,
        *,
        trader_gateway: Any,
        strategy: Any,
        resolve_trader_capabilities: Callable[[Any], BrokerCapability],
        next_client_order_id: Callable[[], str],
        can_submit_client_order: Callable[[str], bool],
        sync_order_id_mapping: Callable[[str, str], None],
        bind_order_owner: Callable[[str, str, str], None],
        notify_strategy_error: Callable[[Any, Exception, str, Any], None],
        payload_field: Callable[[Any, str], Any],
        get_execution_capabilities: Callable[[], dict[str, Any]],
        record_order_request: Callable[[str, Any], None],
        sync_group_mapping: Callable[[str, str], None] = lambda _c, _g: None,
        strategy_limits: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Bind strategy injection hooks, id mapping and capability callbacks.

        ``strategy_limits`` 形如 ``{"max_order_value": {"_default": 500.0}, ...}``,
        用于报单前的策略级限额前置风控(见 :meth:`_check_risk`)。省略则不做该校验
        —— 生产 broker_live 必须传入, 否则 `strategy_max_*` 在实盘静默失效
        (见 docs/zh/meta/signal-ingestion-rfc.md 3.3)。

        这里刻意收的是**限额快照而非 engine 引用**: 报单在策略回调内发生, 那一刻
        引擎正被 `run(&mut self)` 独占借用, 持有并调用 engine 必然
        `RuntimeError: Already borrowed`。
        """
        self._trader_gateway = trader_gateway
        self._strategy = strategy
        self._strategy_limits = strategy_limits or {}
        self._risk_reject_seq = 0
        self._resolve_trader_capabilities = resolve_trader_capabilities
        self._next_client_order_id = next_client_order_id
        self._can_submit_client_order = can_submit_client_order
        self._sync_order_id_mapping = sync_order_id_mapping
        self._bind_order_owner = bind_order_owner
        self._notify_strategy_error = notify_strategy_error
        self._payload_field = payload_field
        self._get_execution_capabilities = get_execution_capabilities
        self._record_order_request = record_order_request
        self._sync_group_mapping = sync_group_mapping
        self._warned_ignored_params: set[str] = set()

    def _resolve_quantity(
        self,
        symbol: str,
        side: str,
        price: float | None,
        quantity: float | None,
    ) -> float:
        """把 ``quantity=None`` 解析为具体下单量, 对齐回测的缺省量语义.

        回测两侧口径不同(见 `strategy_trading_api`):
        买入侧 (`_submit_buy_side_orders`) 走 `sizer.get_size(...)`;
        卖出侧 (`_submit_sell_side_orders`) 不走 sizer, 而是全平当前持仓。
        broker_live 此前完全不做这一步, `quantity=None` 直接透传到
        `resolve_live_order_legs`, 在 `quantity <= 0` 处以 TypeError 崩掉
        —— 等价于 `set_sizer()` 在实盘静默失效。

        与回测的一处有意偏离: 卖出全平取**可用**持仓而非总持仓。A 股 T+1
        下总持仓含当日不可卖部分, 按总量报单会被柜台整单拒绝(硬失败),
        取可用量则退化为部分卖出(软降级), 更贴近用户意图。
        """
        if quantity is not None:
            return float(quantity)

        execution = getattr(self._strategy, "execution", None)
        if str(side).strip().lower() == "sell":
            if execution is None:
                raise RuntimeError(
                    "quantity 未指定且执行后端未就绪, 无法按持仓推断卖出量"
                )
            return float(execution.get_available_position(symbol) or 0.0)

        sizer = getattr(self._strategy, "sizer", None)
        if sizer is None:
            raise RuntimeError("quantity 未指定且策略未配置 sizer, 无法推断买入量")
        ref_price = price
        if ref_price is None:
            last_prices = getattr(self._strategy, "_last_prices", None) or {}
            ref_price = float(last_prices.get(symbol, 0.0) or 0.0)
        cash = float(execution.get_cash()) if execution is not None else 0.0
        # 实盘无回测 ctx(为 None); 自定义 sizer 若依赖 context 需自行兼容。
        size = sizer.get_size(
            float(ref_price), cash, getattr(self._strategy, "ctx", None), symbol
        )
        if size is None:
            raise RuntimeError(
                f"sizer {type(sizer).__name__}.get_size() 返回 None "
                "(get_size 是否漏了 return?)"
            )
        return float(size)

    def _check_risk(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None,
        owner_strategy_id: str,
    ) -> str | None:
        """报单前校验策略级限额, 返回拒单原因(None 表示通过).

        broker_live 的订单不经 `ChannelProcessor`, 故 `strategy_max_*` 这组限额
        此前在实盘完全不生效(见 docs/zh/meta/signal-ingestion-rfc.md 3.3)。

        **为何不直接调 engine**: 报单发生在策略回调内, 那一刻
        `Engine::run(&mut self)` 正独占借用引擎对象, 任何经 Python 侧触达 `Engine`
        的调用都会 `RuntimeError: Already borrowed`。故改用无状态的
        `check_strategy_limits` —— 它与 `Engine::check_strategy_*_limit` 共用同一批
        Rust 自由函数, 拒单文案逐字一致, 不产生第二套规则。

        **覆盖范围**: 只含 order_value / order_size / position_size 三项(不依赖引擎
        运行态)。daily_loss / drawdown / risk_budget 依赖引擎累计盈亏与预算用量,
        无法在此校验, 仍是 broker_live 的缺口。
        """
        limits = self._strategy_limits
        if not limits:
            return None
        key = owner_strategy_id or "_default"
        max_order_value = limits.get("max_order_value", {}).get(key)
        max_order_size = limits.get("max_order_size", {}).get(key)
        max_position_size = limits.get("max_position_size", {}).get(key)
        if (
            max_order_value is None
            and max_order_size is None
            and max_position_size is None
        ):
            return None

        positions: dict[str, float] = {}
        if max_position_size is not None:
            execution = getattr(self._strategy, "execution", None)
            get_positions = getattr(execution, "get_positions", None)
            if callable(get_positions):
                try:
                    positions = {
                        str(k): float(v) for k, v in (get_positions() or {}).items()
                    }
                except Exception:  # noqa: BLE001 — 拿不到持仓不应阻断下单
                    logger.warning(
                        "前置风控读取持仓失败, 本笔跳过持仓上限校验",
                        exc_info=True,
                        extra=build_log_extra(phase="gateway"),
                    )
                    positions = {}

        try:
            return cast(
                Optional[str],
                check_strategy_limits(
                    strategy_id=key,
                    symbol=symbol,
                    side=str(side),
                    quantity=float(quantity),
                    price=price,
                    current_positions=positions,
                    max_order_value=max_order_value,
                    max_order_size=max_order_size,
                    max_position_size=max_position_size,
                ),
            )
        except Exception:  # noqa: BLE001 — 风控自身异常不应吞掉下单能力
            logger.warning(
                "前置风控校验异常, 本笔跳过策略级限额",
                exc_info=True,
                extra=build_log_extra(phase="gateway"),
            )
            return None

    def _emit_risk_reject(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None,
        owner_strategy_id: str,
        reason: str,
    ) -> None:
        """把前置风控拒单以 `Rejected` 委托事件回吐给策略.

        走 `on_order` + `on_reject` 而非错误回调: 与回测同口径——回测的风控拒单
        经 `Event::ExecutionReport(Rejected)` 落到 `_emit_order_callback`, 策略靠
        `on_reject` 感知。实盘若改用 `on_error`, 同一份策略代码就得写两套感知逻辑。
        """
        self._risk_reject_seq += 1
        order = StrategyOrder(
            id=f"RISKREJ-{self._risk_reject_seq}",
            symbol=symbol,
            status=OrderStatus.Rejected,
            filled_quantity=0.0,
            average_filled_price=None,
            reject_reason=reason,
            updated_at=0,
            position_effect=None,
            side=side,
            quantity=float(quantity),
            price=price,
            owner_strategy_id=owner_strategy_id or "_default",
        )
        # broker_live 的事件桥直调 `on_order`(broker_event_bridge.py), 不走回测那套
        # `_emit_order_callback` 的 Rejected→on_reject 级联, 故这里显式派发两者。
        for callback_name in ("on_order", "on_reject"):
            callback = getattr(self._strategy, callback_name, None)
            if not callable(callback):
                continue
            try:
                callback(order)
            except Exception as exc:  # noqa: BLE001 — 用户回调异常不改变拒单结果
                self._notify_strategy_error(self._strategy, exc, callback_name, order)

    def install(self) -> None:
        """No-op: install now happens via `strategy.execution = BrokerExecution(...)`.

        Kept for call-site compatibility (`BrokerRuntime.install_submitter` still
        calls it) and as an extension point should submitter-only setup be
        needed again without touching the strategy object directly.
        """

    def submit_order(
        self,
        symbol: str | None = None,
        side: str = "Buy",
        quantity: float | None = None,
        price: float | None = None,
        client_order_id: str | None = None,
        order_type: str = "Market",
        time_in_force: str = "GTC",
        trigger_price: float | None = None,
        tag: str | None = None,
        position_effect: str = "auto",
        reduce_only: bool = False,
        extra: dict[str, Any] | None = None,
        asset_type: str = "stock",
        fill_mode: Any = None,
        slippage: Any = None,
        commission: Any = None,
        trail_offset: float | None = None,
        trail_reference_price: float | None = None,
        broker_options: dict[str, Any] | None = None,
    ) -> OrderReceipt:
        """Submit a live broker order using the unified strategy-facing signature.

        Returns an `OrderReceipt`, not a single id string: a logical order can be
        split into multiple legs (e.g. close_today/close_yesterday, or a reversal's
        close+open legs), each with its own client/broker order id. `receipt.primary`
        is the *first leg's* `broker_order_id` (the value production call sites use
        as "the" order id); `str(receipt)` is the `group_id` (client order id) and is
        a *different* id space in production brokers — do not use them
        interchangeably.

        `symbol=None` / `quantity=None` 按回测同口径解析(见
        :meth:`_resolve_quantity`): symbol 取当前 bar/tick, 买入量走
        `strategy.sizer`, 卖出量取可用持仓。解析后 `quantity <= 0` 返回空回执,
        不向柜台报单。
        """
        if not getattr(self._strategy, "broker_ready", True):
            raise RuntimeError(
                "broker 尚未就绪，请在 broker_ready=True"
                "(on_broker_connected 之后)再下单"
            )
        _ = tag  # 元数据，收下忽略
        _ = trail_reference_price  # 仅随 trail_offset 生效；下方已拦截追踪单
        order_type = order_type or "Market"  # buy()/sell() 缺省传 None → 用签名默认
        if trigger_price is not None:
            raise RuntimeError("broker_live 暂不支持条件/止损触发单(trigger_price)")
        if trail_offset is not None or str(order_type).strip().lower() in {
            "stoptrail",
            "stoptraillimit",
        }:
            raise RuntimeError("broker_live 暂不支持追踪止损单(trail_offset/StopTrail)")
        for _name, _val in (
            ("fill_mode", fill_mode),
            ("slippage", slippage),
            ("commission", commission),
            ("broker_options", broker_options),
        ):
            if _val is not None and _name not in self._warned_ignored_params:
                self._warned_ignored_params.add(_name)
                logger.warning(
                    "broker_live 忽略回测模拟参数 %s",
                    _name,
                    extra=build_log_extra(phase="gateway"),
                )
        capability = self._resolve_trader_capabilities(self._trader_gateway)
        validate_broker_extra(capability, extra)
        # symbol/quantity 的缺省解析: 回测在 strategy_trading_api 内完成,
        # broker_live 此前整条缺失(仅本地止损路径解析过 symbol), 这里补齐。
        symbol = _resolve_symbol(self._strategy, symbol)
        quantity = self._resolve_quantity(
            symbol=symbol, side=side, price=price, quantity=quantity
        )
        if quantity <= 0:
            # 回测对 quantity<=0 返回空单(不报单); 此前实盘会把 0 手单发给柜台。
            return OrderReceipt(group_id="", order_ids=(), legs=())
        normalized_asset_type = normalize_asset_type(asset_type)
        normalized_position_effect = validate_execution_semantics(
            capability,
            position_effect,
            reduce_only,
            side=side,
        )
        owner_strategy_id = str(
            getattr(self._strategy, "_owner_strategy_id", "_default")
        )
        # 前置风控: 在拆腿与报单之前。放这里而非拆腿之后, 是因为限额语义针对的是
        # 用户提交的那一笔逻辑委托(名义/数量), 而非拆出的每条腿——按腿校验会让
        # 一笔超限单因拆成两腿而各自"合规"从而整体漏过。
        reject_reason = self._check_risk(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            owner_strategy_id=owner_strategy_id,
        )
        if reject_reason is not None:
            record_reject(
                strategy_id=owner_strategy_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                client_order_id="",
                reason=reject_reason,
            )
            self._emit_risk_reject(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                owner_strategy_id=owner_strategy_id,
                reason=reject_reason,
            )
            return OrderReceipt(group_id="", order_ids=(), legs=())
        order_legs = resolve_live_order_legs(
            trader_gateway=self._trader_gateway,
            capability=capability,
            symbol=symbol,
            side=side,
            quantity=quantity,
            position_effect=normalized_position_effect,
            reduce_only=reduce_only,
            payload_field=self._payload_field,
        )
        request_client_order_id = client_order_id or self._next_client_order_id()
        client_order_ids = build_live_order_client_ids(
            request_client_order_id=request_client_order_id,
            order_legs=order_legs,
        )
        validate_live_order_client_ids(
            strategy=self._strategy,
            client_order_ids=client_order_ids,
            symbol=symbol,
            side=side,
            quantity=quantity,
            can_submit_client_order=self._can_submit_client_order,
            notify_strategy_error=self._notify_strategy_error,
            group_id=request_client_order_id,
        )
        broker_order_ids: list[str] = []
        legs: list[OrderLeg] = []
        for leg_index, (leg_position_effect, leg_quantity) in enumerate(order_legs):
            request = UnifiedOrderRequest(
                client_order_id=client_order_ids[leg_index],
                symbol=symbol,
                side=side,
                quantity=leg_quantity,
                price=price,
                order_type=order_type,
                time_in_force=time_in_force,
                position_effect=leg_position_effect,
                reduce_only=reduce_only,
                asset_type=normalized_asset_type,
                extra=dict(extra or {}),
            )
            broker_order_id = str(self._trader_gateway.place_order(request))
            broker_order_ids.append(broker_order_id)
            record_submit(
                strategy_id=owner_strategy_id,
                symbol=symbol,
                side=side,
                quantity=leg_quantity,
                price=price,
                client_order_id=request.client_order_id,
                broker_order_id=broker_order_id,
                order_type=order_type,
                trace_id=request_client_order_id,
            )
            self._sync_order_id_mapping(request.client_order_id, broker_order_id)
            self._record_order_request(request.client_order_id, request)
            self._bind_order_owner(
                request.client_order_id, broker_order_id, owner_strategy_id
            )
            self._sync_group_mapping(request.client_order_id, request_client_order_id)
            legs.append(
                OrderLeg(
                    position_effect=leg_position_effect,
                    quantity=leg_quantity,
                    client_order_id=request.client_order_id,
                    broker_order_id=broker_order_id,
                )
            )
        return OrderReceipt(
            group_id=request_client_order_id,
            order_ids=tuple(broker_order_ids),
            legs=tuple(legs),
        )
