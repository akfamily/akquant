import warnings
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union, cast

from .akquant import OrderStatus, OrderType, PositionEffect, TimeInForce

if TYPE_CHECKING:
    from .backtest.fill_mode import FillMode
from .gateway.broker_models import normalize_asset_type
from .gateway.order_receipt import OrderLeg, OrderReceipt
from .log import get_logger

logger = get_logger("trading")

OrderFillPolicy = Dict[str, Any]
OrderSlippage = Dict[str, Any]
OrderCommission = Dict[str, Any]


def _reject_legacy_fill_mode(fill_mode: Any) -> None:
    """公开下单入口的硬切断:``fill_mode`` 仅接受 :class:`FillMode`,dict → TypeError.

    内部 dict 注入路径(pre-open / strategy map)不经此关卡,继续以 dict 流转。
    """
    from .backtest.fill_mode import FillMode

    if fill_mode is not None and not isinstance(fill_mode, FillMode):
        from .backtest.engine import _LEGACY_FILL_POLICY_DICT_MSG

        raise TypeError(_LEGACY_FILL_POLICY_DICT_MSG)


def resolve_symbol(strategy: Any, symbol: Optional[str]) -> str:
    """解析标的代码，默认使用当前处理的 bar/tick 标的."""
    if symbol is None:
        if strategy._last_event_type == "tick" and strategy.current_tick:
            symbol = strategy.current_tick.symbol
        elif strategy._last_event_type == "bar" and strategy.current_bar:
            symbol = strategy.current_bar.symbol
        elif strategy.current_bar:
            symbol = strategy.current_bar.symbol
        elif strategy.current_tick:
            symbol = strategy.current_tick.symbol
        else:
            raise ValueError("Symbol must be provided")
    return symbol


def get_position(strategy: Any, symbol: Optional[str] = None) -> float:
    """获取指定标的持仓（经执行后端）."""
    return float(strategy.execution.get_position(symbol))


def get_available_position(strategy: Any, symbol: Optional[str] = None) -> float:
    """获取可用持仓（经执行后端）."""
    return float(strategy.execution.get_available_position(symbol))


def get_holding_bars(strategy: Any, symbol: Optional[str] = None) -> int:
    """获取持仓持有 Bar 数（经执行后端）."""
    return int(strategy.execution.hold_bar(symbol))


def get_positions(strategy: Any) -> Dict[str, float]:
    """获取所有持仓（经执行后端）."""
    return cast(Dict[str, float], strategy.execution.get_positions())


def get_last_target_positions_plan(strategy: Any) -> Dict[str, Any]:
    """获取最近一次 rebalance_positions() 生成的调仓计划."""
    plan = getattr(strategy, "_last_target_positions_plan", None)
    if isinstance(plan, dict):
        return cast(Dict[str, Any], plan)
    return {}


def get_open_orders(strategy: Any, symbol: Optional[str] = None) -> List[Any]:
    """获取当前未完成的订单（经执行后端）."""
    return cast(List[Any], strategy.execution.get_open_orders(symbol=symbol))


def get_order(strategy: Any, order_id: str) -> Optional[Any]:
    """获取指定订单详情."""
    canceled_order_ids = {
        str(oid)
        for oid in getattr(getattr(strategy, "ctx", None), "canceled_order_ids", [])
        if oid
    }
    pending_canceled_ids: Set[str] = getattr(
        strategy, "_pending_canceled_order_ids", set()
    )
    if not isinstance(pending_canceled_ids, set):
        pending_canceled_ids = set()
        setattr(strategy, "_pending_canceled_order_ids", pending_canceled_ids)
    canceled_order_ids.update(str(oid) for oid in pending_canceled_ids if oid)

    if order_id in strategy._known_orders:
        order = strategy._known_orders[order_id]
        if order_id in canceled_order_ids:
            try:
                order.status = OrderStatus.Cancelled
            except Exception:
                pass
        _attach_broker_options(strategy, order_id, order)
        return order

    if strategy.ctx:
        for o in strategy.ctx.active_orders:
            if o.id == order_id:
                if order_id in canceled_order_ids:
                    try:
                        o.status = OrderStatus.Cancelled
                    except Exception:
                        pass
                _attach_broker_options(strategy, order_id, o)
                return o

    # 终态订单已离开在途账本(_known_orders / ctx.active_orders), 回退查留档,
    # 否则订单一旦成交/撤单/被拒就永久查不到。
    finalized = getattr(strategy, "_finalized_orders", None)
    if isinstance(finalized, dict):
        order = finalized.get(order_id)
        if order is not None:
            if order_id in canceled_order_ids:
                try:
                    order.status = OrderStatus.Cancelled
                except Exception:
                    pass
            _attach_broker_options(strategy, order_id, order)
            return order

    return None


def _record_broker_options(
    strategy: Any,
    order_ids: Optional[Union[str, List[str]]],
    broker_options: Optional[Dict[str, Any]],
) -> None:
    if not order_ids or not broker_options:
        return
    if not isinstance(broker_options, dict):
        raise TypeError("broker_options must be a dict when provided")
    normalized_order_ids: List[str]
    if isinstance(order_ids, list):
        normalized_order_ids = [str(order_id) for order_id in order_ids if order_id]
    else:
        normalized_order_ids = [str(order_ids)]
    if not normalized_order_ids:
        return
    store = getattr(strategy, "_broker_options_by_order_id", None)
    if not isinstance(store, dict):
        store = {}
        setattr(strategy, "_broker_options_by_order_id", store)
    normalized = dict(broker_options)
    for order_id in normalized_order_ids:
        store[order_id] = dict(normalized)
        order = get_order(strategy, order_id)
        if order is not None:
            _attach_broker_options(strategy, order_id, order)


def _attach_broker_options(strategy: Any, order_id: str, order: Any) -> None:
    store = getattr(strategy, "_broker_options_by_order_id", None)
    if not isinstance(store, dict):
        return
    options = store.get(str(order_id))
    if not isinstance(options, dict):
        return
    try:
        setattr(order, "broker_options", dict(options))
    except Exception:
        return


def cancel_order(strategy: Any, order_id: Any) -> None:
    """取消指定订单（经执行后端）；order_id 可为 str 或 OrderReceipt（取 .primary）."""
    strategy.execution.cancel_order(str(getattr(order_id, "primary", order_id)))


def cancel_group(strategy: Any, group_id: Any) -> None:
    """按 group_id（或 OrderReceipt）撤销一个逻辑委托的全部腿（经执行后端）."""
    execution = strategy.execution
    gid = str(getattr(group_id, "group_id", group_id))
    cancel_group_method = getattr(execution, "cancel_group", None)
    if callable(cancel_group_method):
        cancel_group_method(gid)
    else:  # 回测后端无 group 概念：退化为撤单
        execution.cancel_order(gid)


def cancel_all_orders(strategy: Any, symbol: Optional[str] = None) -> None:
    """取消所有未完成订单（经执行后端）."""
    strategy.execution.cancel_all_orders(symbol=symbol)


def buy(
    strategy: Any,
    symbol: Optional[str] = None,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: Optional[TimeInForce] = None,
    trigger_price: Optional[float] = None,
    tag: Optional[str] = None,
    order_type: Optional[str] = None,
    trail_offset: Optional[float] = None,
    trail_reference_price: Optional[float] = None,
    fill_mode: Optional["FillMode"] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: Union[PositionEffect, str, None] = None,
    reduce_only: bool = False,
) -> OrderReceipt:
    """买入下单."""
    _reject_legacy_fill_mode(fill_mode)
    submit_order_method = getattr(strategy, "submit_order", None)
    if callable(submit_order_method):
        return cast(
            OrderReceipt,
            submit_order_method(
                symbol=symbol,
                side="Buy",
                quantity=quantity,
                price=price,
                time_in_force=time_in_force,
                trigger_price=trigger_price,
                tag=tag,
                order_type=order_type,
                trail_offset=trail_offset,
                trail_reference_price=trail_reference_price,
                fill_mode=fill_mode,
                slippage=slippage,
                commission=commission,
                position_effect=_normalize_position_effect(position_effect, "auto"),
                reduce_only=reduce_only,
            ),
        )
    order_type_enum = _parse_order_type(order_type)
    return _submit_buy_side(
        strategy=strategy,
        symbol=symbol,
        quantity=quantity,
        price=price,
        time_in_force=time_in_force,
        trigger_price=trigger_price,
        tag=tag,
        order_type=order_type_enum,
        trail_offset=trail_offset,
        trail_reference_price=trail_reference_price,
        fill_mode=fill_mode,
        slippage=slippage,
        commission=commission,
        position_effect=_normalize_position_effect(position_effect, "auto"),
        reduce_only=reduce_only,
    )


def sell(
    strategy: Any,
    symbol: Optional[str] = None,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: Optional[TimeInForce] = None,
    trigger_price: Optional[float] = None,
    tag: Optional[str] = None,
    order_type: Optional[str] = None,
    trail_offset: Optional[float] = None,
    trail_reference_price: Optional[float] = None,
    fill_mode: Optional["FillMode"] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: Union[PositionEffect, str, None] = None,
    reduce_only: bool = False,
) -> OrderReceipt:
    """卖出下单."""
    _reject_legacy_fill_mode(fill_mode)
    submit_order_method = getattr(strategy, "submit_order", None)
    if callable(submit_order_method):
        return cast(
            OrderReceipt,
            submit_order_method(
                symbol=symbol,
                side="Sell",
                quantity=quantity,
                price=price,
                time_in_force=time_in_force,
                trigger_price=trigger_price,
                tag=tag,
                order_type=order_type,
                trail_offset=trail_offset,
                trail_reference_price=trail_reference_price,
                fill_mode=fill_mode,
                slippage=slippage,
                commission=commission,
                position_effect=_normalize_position_effect(position_effect, "auto"),
                reduce_only=reduce_only,
            ),
        )
    order_type_enum = _parse_order_type(order_type)
    return _submit_sell_side(
        strategy=strategy,
        symbol=symbol,
        quantity=quantity,
        price=price,
        time_in_force=time_in_force,
        trigger_price=trigger_price,
        tag=tag,
        order_type=order_type_enum,
        trail_offset=trail_offset,
        trail_reference_price=trail_reference_price,
        fill_mode=fill_mode,
        slippage=slippage,
        commission=commission,
        position_effect=_normalize_position_effect(position_effect, "auto"),
        reduce_only=reduce_only,
    )


def _submit_buy_side(
    strategy: Any,
    symbol: Optional[str],
    quantity: Optional[float],
    price: Optional[float],
    time_in_force: Optional[TimeInForce],
    trigger_price: Optional[float],
    tag: Optional[str],
    order_type: Optional[Any] = None,
    trail_offset: Optional[float] = None,
    trail_reference_price: Optional[float] = None,
    fill_mode: Optional["FillMode"] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: str = "auto",
    reduce_only: bool = False,
) -> OrderReceipt:
    return _orders_to_receipt(
        _submit_buy_side_orders(
            strategy=strategy,
            symbol=symbol,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            tag=tag,
            order_type=order_type,
            trail_offset=trail_offset,
            trail_reference_price=trail_reference_price,
            fill_mode=fill_mode,
            slippage=slippage,
            commission=commission,
            position_effect=position_effect,
            reduce_only=reduce_only,
        ),
        position_effect=position_effect,
    )


def _submit_buy_side_orders(
    strategy: Any,
    symbol: Optional[str],
    quantity: Optional[float],
    price: Optional[float],
    time_in_force: Optional[TimeInForce],
    trigger_price: Optional[float],
    tag: Optional[str],
    order_type: Optional[Any] = None,
    trail_offset: Optional[float] = None,
    trail_reference_price: Optional[float] = None,
    fill_mode: Optional["FillMode"] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: str = "auto",
    reduce_only: bool = False,
) -> List[str]:
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")

    symbol = resolve_symbol(strategy, symbol)

    ref_price = price
    if ref_price is None:
        ref_price = strategy._last_prices.get(symbol, 0.0)

    allow_quantity_auto_resize = quantity is None
    if quantity is None:
        quantity = strategy.sizer.get_size(
            ref_price, strategy.ctx.cash, strategy.ctx, symbol
        )
    if quantity <= 0:
        return []

    if position_effect == "auto":
        # 用可平持仓而非结算仓：同一 on_bar 内已提交未成交的平仓单必须扣除，
        # 否则"先平后开"的反手第二腿会被误判成平仓（#361）。这里刻意不投影在途
        # 开仓单，偏向判为开仓即偏向多预留保证金（与 vn.py / RQAlpha 一致）。
        current_position = _position_from_execution(
            strategy, symbol, "get_closable_position"
        )
        legs = _resolve_auto_position_effect_legs(
            "buy", current_position, float(quantity), reduce_only
        )
        order_ids: List[str] = []
        for leg_effect, leg_quantity, leg_reduce_only in legs:
            order_ids.extend(
                _submit_buy_side_orders(
                    strategy=strategy,
                    symbol=symbol,
                    quantity=leg_quantity,
                    price=price,
                    time_in_force=time_in_force,
                    trigger_price=trigger_price,
                    tag=tag,
                    order_type=order_type,
                    trail_offset=trail_offset,
                    trail_reference_price=trail_reference_price,
                    fill_mode=fill_mode,
                    slippage=slippage,
                    commission=commission,
                    position_effect=leg_effect,
                    reduce_only=leg_reduce_only,
                )
            )
        return order_ids

    explicit_fill_dict = _fill_mode_to_dict(fill_mode)
    effective_fill_policy = _resolve_effective_order_fill_policy(
        strategy, explicit_fill_dict
    )
    fill_mode_enum, fill_timer_timing = _normalize_order_fill_policy(
        effective_fill_policy
    )
    effective_slippage = _resolve_effective_order_slippage(strategy, slippage)
    fill_slippage_type, fill_slippage_value = _normalize_order_slippage(
        strategy,
        symbol,
        effective_slippage,
    )
    effective_commission = _resolve_effective_order_commission(strategy, commission)
    fill_commission_type, fill_commission_value = _normalize_order_commission(
        effective_commission
    )
    if (
        order_type is None
        and trail_offset is None
        and trail_reference_price is None
        and effective_fill_policy is None
        and effective_slippage is None
        and effective_commission is None
    ):
        return [
            cast(
                str,
                strategy.ctx.buy(
                    symbol,
                    quantity,
                    price,
                    time_in_force,
                    trigger_price,
                    tag or "",
                    position_effect=_position_effect_enum(position_effect),
                    reduce_only=reduce_only,
                    allow_quantity_auto_resize=allow_quantity_auto_resize,
                ),
            )
        ]
    return [
        cast(
            str,
            strategy.ctx.buy(
                symbol,
                quantity,
                price,
                time_in_force,
                trigger_price,
                tag or "",
                order_type,
                trail_offset,
                trail_reference_price,
                fill_mode_enum,
                fill_timer_timing,
                fill_slippage_type,
                fill_slippage_value,
                fill_commission_type,
                fill_commission_value,
                allow_quantity_auto_resize,
                _position_effect_enum(position_effect),
                reduce_only,
            ),
        )
    ]


def _submit_sell_side(
    strategy: Any,
    symbol: Optional[str],
    quantity: Optional[float],
    price: Optional[float],
    time_in_force: Optional[TimeInForce],
    trigger_price: Optional[float],
    tag: Optional[str],
    order_type: Optional[Any] = None,
    trail_offset: Optional[float] = None,
    trail_reference_price: Optional[float] = None,
    fill_mode: Optional["FillMode"] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: str = "auto",
    reduce_only: bool = False,
) -> OrderReceipt:
    return _orders_to_receipt(
        _submit_sell_side_orders(
            strategy=strategy,
            symbol=symbol,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            tag=tag,
            order_type=order_type,
            trail_offset=trail_offset,
            trail_reference_price=trail_reference_price,
            fill_mode=fill_mode,
            slippage=slippage,
            commission=commission,
            position_effect=position_effect,
            reduce_only=reduce_only,
        ),
        position_effect=position_effect,
    )


def _submit_sell_side_orders(
    strategy: Any,
    symbol: Optional[str],
    quantity: Optional[float],
    price: Optional[float],
    time_in_force: Optional[TimeInForce],
    trigger_price: Optional[float],
    tag: Optional[str],
    order_type: Optional[Any] = None,
    trail_offset: Optional[float] = None,
    trail_reference_price: Optional[float] = None,
    fill_mode: Optional["FillMode"] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: str = "auto",
    reduce_only: bool = False,
) -> List[str]:
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")

    symbol = resolve_symbol(strategy, symbol)

    if quantity is None:
        pos = strategy.ctx.get_position(symbol)
        if pos > 0:
            quantity = pos
        else:
            return []
    if quantity <= 0:
        return []

    if position_effect == "auto":
        # 与 buy 侧对称：可平持仓已扣除在途平仓单，避免反手第二腿误判（#361）。
        current_position = _position_from_execution(
            strategy, symbol, "get_closable_position"
        )
        legs = _resolve_auto_position_effect_legs(
            "sell", current_position, float(quantity), reduce_only
        )
        order_ids: List[str] = []
        for leg_effect, leg_quantity, leg_reduce_only in legs:
            order_ids.extend(
                _submit_sell_side_orders(
                    strategy=strategy,
                    symbol=symbol,
                    quantity=leg_quantity,
                    price=price,
                    time_in_force=time_in_force,
                    trigger_price=trigger_price,
                    tag=tag,
                    order_type=order_type,
                    trail_offset=trail_offset,
                    trail_reference_price=trail_reference_price,
                    fill_mode=fill_mode,
                    slippage=slippage,
                    commission=commission,
                    position_effect=leg_effect,
                    reduce_only=leg_reduce_only,
                )
            )
        return order_ids

    explicit_fill_dict = _fill_mode_to_dict(fill_mode)
    effective_fill_policy = _resolve_effective_order_fill_policy(
        strategy, explicit_fill_dict
    )
    fill_mode_enum, fill_timer_timing = _normalize_order_fill_policy(
        effective_fill_policy
    )
    effective_slippage = _resolve_effective_order_slippage(strategy, slippage)
    fill_slippage_type, fill_slippage_value = _normalize_order_slippage(
        strategy,
        symbol,
        effective_slippage,
    )
    effective_commission = _resolve_effective_order_commission(strategy, commission)
    fill_commission_type, fill_commission_value = _normalize_order_commission(
        effective_commission
    )
    if (
        order_type is None
        and trail_offset is None
        and trail_reference_price is None
        and effective_fill_policy is None
        and effective_slippage is None
        and effective_commission is None
    ):
        return [
            cast(
                str,
                strategy.ctx.sell(
                    symbol,
                    quantity,
                    price,
                    time_in_force,
                    trigger_price,
                    tag or "",
                    position_effect=_position_effect_enum(position_effect),
                    reduce_only=reduce_only,
                ),
            )
        ]
    return [
        cast(
            str,
            strategy.ctx.sell(
                symbol,
                quantity,
                price,
                time_in_force,
                trigger_price,
                tag or "",
                order_type,
                trail_offset,
                trail_reference_price,
                fill_mode_enum,
                fill_timer_timing,
                fill_slippage_type,
                fill_slippage_value,
                fill_commission_type,
                fill_commission_value,
                _position_effect_enum(position_effect),
                reduce_only,
            ),
        )
    ]


def get_execution_capabilities(strategy: Any) -> Dict[str, Any]:
    """获取当前执行环境能力描述（经执行后端）."""
    execution = getattr(strategy, "execution", None)
    if execution is not None:
        return cast(Dict[str, Any], execution.capabilities())
    return _sim_capabilities(strategy)


def _require_execution_ready(strategy: Any) -> None:
    """Fail fast when neither a backtest ctx nor a live broker backend is ready.

    broker_live binds a ready BrokerExecution while ctx stays None, so only
    raise when ctx is None AND the backend is not broker_live-capable — this
    preserves the old "Context not ready" fail-fast for backtest strategies
    that call sizing helpers before the engine binds ctx (e.g. in __init__).
    """
    if getattr(strategy, "ctx", None) is None and not get_execution_capabilities(
        strategy
    ).get("broker_live"):
        raise RuntimeError("Context not ready")


def _normalize_position_effect(
    position_effect: Union[PositionEffect, str, None], default: str = "auto"
) -> str:
    if position_effect is None:
        return default
    if isinstance(position_effect, str):
        value = position_effect.strip().lower()
    else:
        value = str(position_effect).split(".")[-1].strip().lower()
    if value not in {"auto", "open", "close", "close_today", "close_yesterday"}:
        raise ValueError(
            "position_effect must be one of: auto, open, close, "
            "close_today, close_yesterday"
        )
    return value


def _orders_to_receipt(order_ids: List[str], position_effect: str) -> OrderReceipt:
    """将回测拆腿产生的全部订单 id 封装为 OrderReceipt（每 id 一腿）."""
    ids = tuple(str(o) for o in order_ids if o)
    legs = tuple(
        OrderLeg(
            position_effect=position_effect,
            quantity=0.0,
            client_order_id=oid,
            broker_order_id=oid,
        )
        for oid in ids
    )
    return OrderReceipt(
        group_id=ids[0] if ids else "",
        order_ids=ids,
        legs=legs,
    )


def _position_effect_enum(position_effect: str) -> PositionEffect:
    mapping = {
        "auto": PositionEffect.Auto,
        "open": PositionEffect.Open,
        "close": PositionEffect.Close,
        "close_today": PositionEffect.CloseToday,
        "close_yesterday": PositionEffect.CloseYesterday,
    }
    return mapping[str(position_effect).strip().lower()]


def _resolve_auto_position_effect_legs(
    side: str,
    current_position: float,
    quantity: float,
    reduce_only: bool,
) -> List[Tuple[str, float, bool]]:
    if quantity <= 0:
        return []
    normalized_side = str(side).strip().lower()
    legs: List[Tuple[str, float, bool]] = []
    if normalized_side == "buy":
        close_qty = (
            min(quantity, abs(current_position)) if current_position < 0 else 0.0
        )
    else:
        close_qty = min(quantity, current_position) if current_position > 0 else 0.0
    if close_qty > 0:
        legs.append(("close", float(close_qty), reduce_only))
    open_qty = float(max(quantity - close_qty, 0.0))
    if open_qty > 0 and not reduce_only:
        legs.append(("open", open_qty, False))
    return legs


def submit_order(strategy: Any, **kwargs: Any) -> OrderReceipt:
    """统一下单接口（经执行后端）."""
    return cast(OrderReceipt, strategy.execution.submit_order(**kwargs))


def _parse_order_type(order_type: Optional[str]) -> Tuple[Optional[str], Optional[Any]]:
    if order_type is None:
        return None, None
    key = str(order_type).strip().lower()
    mapping: Dict[str, str] = {
        "market": "Market",
        "limit": "Limit",
        "stop": "StopMarket",
        "stopmarket": "StopMarket",
        "stop_limit": "StopLimit",
        "stoplimit": "StopLimit",
        "stoptrail": "StopTrail",
        "stoptraillimit": "StopTrailLimit",
    }
    if key not in mapping:
        raise RuntimeError(
            f"order_type {order_type!r} is not supported in current execution mode"
        )
    attr_name = mapping[key]
    return key, getattr(OrderType, attr_name, None)


def _fill_mode_to_dict(fill_mode: Optional["FillMode"]) -> Optional[OrderFillPolicy]:
    """把公开传入的 ``FillMode`` 转为内部 dict,与内部注入 dict 统一流转.

    ``None`` 透传(交由 ``_resolve_effective_order_fill_policy`` 决定 pre-open /
    strategy map 注入)。硬切断(dict → TypeError)已在公开入口
    :func:`_reject_legacy_fill_mode` 完成,此处仅翻译 FillMode。
    """
    if fill_mode is None:
        return None
    price_basis, bar_offset, temporal = fill_mode._to_core()
    return {
        "price_basis": price_basis,
        "bar_offset": bar_offset,
        "temporal": temporal,
    }


def _normalize_order_fill_policy(
    fill_policy: Optional[OrderFillPolicy],
) -> Tuple[Optional[Any], Optional[str]]:
    """内部 dict → ``(ExecutionMode, timer_timing)`` 的唯一收口.

    内部注入路径(pre-open / strategy map)与公开 ``fill_mode`` 翻译后的 dict
    在此统一。合法性由 :func:`fill_mode_from_core` 在枚举层保证(非法三元组不可
    表达),故此处不再重复 ``open|ohlc4|hl2 requires bar_offset=1`` 校验。
    """
    if fill_policy is None:
        return None, None
    if not isinstance(fill_policy, dict):
        raise TypeError("internal: fill_policy must be a dict when provided")
    from .backtest.fill_mode import fill_mode_from_core

    raw_basis = str(fill_policy.get("price_basis", "open")).strip().lower()
    raw_offset = int(fill_policy.get("bar_offset", 0 if raw_basis == "close" else 1))
    raw_temporal = str(fill_policy.get("temporal", "same_cycle")).strip().lower()
    mode = fill_mode_from_core(raw_basis, raw_offset, raw_temporal)
    mode_enum, timer_timing = mode.to_execution_mode()
    return mode_enum, timer_timing


def _normalize_order_slippage(
    strategy: Any,
    symbol: str,
    slippage: Optional[Union[OrderSlippage, float, int]],
) -> Tuple[Optional[str], Optional[float]]:
    if slippage is None:
        return None, None
    if isinstance(slippage, (int, float)):
        raw_type = "percent"
        raw_value = float(slippage)
        if raw_value != 0.0:
            warnings.warn(
                "Passing order slippage as a bare number is deprecated in AKQuant. "
                "Use an explicit policy such as "
                "slippage={'type': 'percent', 'value': 0.0002} or "
                "slippage={'type': 'fixed', 'value': 0.2}.",
                DeprecationWarning,
                stacklevel=3,
            )
    else:
        if not isinstance(slippage, dict):
            raise TypeError("slippage must be a dict when provided")
        raw_type = str(slippage.get("type", "percent")).strip().lower()
        raw_value = slippage.get("value", 0.0)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError("slippage.value must be a number >= 0") from None
    if value < 0:
        raise ValueError("slippage.value must be >= 0")
    if raw_type in {"percent", "fixed"}:
        return raw_type, value
    if raw_type == "zero":
        return "fixed", 0.0
    if raw_type == "ticks":
        tick_size = float(strategy.get_instrument(symbol).tick_size)
        if tick_size <= 0:
            raise ValueError("slippage.type='ticks' requires tick_size > 0")
        return "fixed", value * tick_size
    raise ValueError("slippage.type must be one of: percent, fixed, ticks, zero")


def _normalize_order_commission(
    commission: Optional[OrderCommission],
) -> Tuple[Optional[str], Optional[float]]:
    if commission is None:
        return None, None
    if not isinstance(commission, dict):
        raise TypeError("commission must be a dict when provided")
    raw_type = str(commission.get("type", "percent")).strip().lower()
    if raw_type not in {"percent", "fixed", "per_unit"}:
        raise ValueError("commission.type must be one of: percent, fixed, per_unit")
    raw_value = commission.get("value", 0.0)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError("commission.value must be a number >= 0") from None
    if value < 0:
        raise ValueError("commission.value must be >= 0")
    return raw_type, value


def _resolve_effective_order_fill_policy(
    strategy: Any, fill_policy: Optional[OrderFillPolicy]
) -> Optional[OrderFillPolicy]:
    if fill_policy is not None:
        return fill_policy
    if bool(getattr(strategy, "_framework_in_pre_open_phase", False)):
        return {
            "price_basis": "open",
            "bar_offset": 1,
            "temporal": "same_cycle",
        }
    owner_strategy_id = str(getattr(strategy, "_owner_strategy_id", "") or "").strip()
    if not owner_strategy_id:
        owner_strategy_id = "_default"
    policy_map = cast(
        Optional[Dict[str, OrderFillPolicy]],
        getattr(strategy, "_strategy_fill_policy_map", None),
    )
    if not policy_map:
        return None
    policy = policy_map.get(owner_strategy_id)
    if policy is None and owner_strategy_id != "_default":
        policy = policy_map.get("_default")
    if policy is None:
        return None
    return dict(policy)


def _resolve_effective_order_slippage(
    strategy: Any, slippage: Optional[Union[OrderSlippage, float, int]]
) -> Optional[Union[OrderSlippage, float, int]]:
    if slippage is not None:
        return slippage
    owner_strategy_id = str(getattr(strategy, "_owner_strategy_id", "") or "").strip()
    if not owner_strategy_id:
        owner_strategy_id = "_default"
    slippage_map = cast(
        Optional[Dict[str, OrderSlippage]],
        getattr(strategy, "_strategy_slippage_map", None),
    )
    if not slippage_map:
        return None
    resolved = slippage_map.get(owner_strategy_id)
    if resolved is None and owner_strategy_id != "_default":
        resolved = slippage_map.get("_default")
    if resolved is None:
        return None
    return dict(resolved)


def _resolve_effective_order_commission(
    strategy: Any, commission: Optional[OrderCommission]
) -> Optional[OrderCommission]:
    if commission is not None:
        return commission
    owner_strategy_id = str(getattr(strategy, "_owner_strategy_id", "") or "").strip()
    if not owner_strategy_id:
        owner_strategy_id = "_default"
    commission_map = cast(
        Optional[Dict[str, OrderCommission]],
        getattr(strategy, "_strategy_commission_map", None),
    )
    if not commission_map:
        return None
    resolved = commission_map.get(owner_strategy_id)
    if resolved is None and owner_strategy_id != "_default":
        resolved = commission_map.get("_default")
    if resolved is None:
        return None
    return dict(resolved)


def get_portfolio_value(strategy: Any) -> float:
    """计算当前投资组合总价值 (现金 + 持仓市值)（经执行后端）."""
    return float(strategy.execution.get_portfolio_value())


def _resolve_mark_price(strategy: Any, symbol: str) -> float:
    price = float(strategy._last_prices.get(symbol, 0.0))
    if price > 0.0:
        return price
    if strategy.current_bar and strategy.current_bar.symbol == symbol:
        return float(strategy.current_bar.close)
    if strategy.current_tick and strategy.current_tick.symbol == symbol:
        return float(strategy.current_tick.price)
    return 0.0


def _is_margin_account(strategy: Any) -> bool:
    if strategy.ctx is None:
        return False
    risk_config = getattr(strategy.ctx, "risk_config", None)
    account_mode = str(getattr(risk_config, "account_mode", "cash")).strip().lower()
    return account_mode == "margin"


def _supports_short_targets(
    strategy: Any, capabilities: Optional[Dict[str, Any]] = None
) -> bool:
    capability_map = (
        capabilities
        if capabilities is not None
        else get_execution_capabilities(strategy)
    )
    if bool(capability_map.get("supports_short_sell", False)):
        return True
    account_mode = str(capability_map.get("account_mode", "")).strip().lower()
    if account_mode:
        return account_mode == "margin" and bool(
            capability_map.get("supports_short_sell", False)
        )
    if strategy.ctx is None:
        return False
    risk_config = getattr(strategy.ctx, "risk_config", None)
    return bool(getattr(risk_config, "enable_short_sell", False))


def get_account(strategy: Any) -> Dict[str, Any]:
    """获取账户资金详情快照（经执行后端）."""
    return cast(Dict[str, Any], strategy.execution.get_account())


def _lot_size_from_strategy(strategy: Any, symbol: str) -> int:
    """读策略属性 ``self.lot_size``(int 或 ``Dict[str, int]``), 缺省 1."""
    lot_size = getattr(strategy, "lot_size", 1)
    if isinstance(lot_size, dict):
        value = lot_size.get(symbol, lot_size.get("DEFAULT", 1))
    else:
        value = lot_size
    if value is None:
        return 1
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 1


def _lot_size_from_instrument(strategy: Any, symbol: str) -> int:
    """读标的登记值 ``Instrument.lot_size``, 取不到返回 0(表示"无登记值")."""
    snapshots = getattr(strategy, "_instrument_snapshots", None) or {}
    snapshot = snapshots.get(symbol)
    if snapshot is None:
        return 0
    try:
        return int(float(getattr(snapshot, "lot_size", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _resolve_lot_size(strategy: Any, symbol: str) -> int:
    """解析下单取整用的最小交易单位.

    **口径必须与撮合层校验一致**: Rust 侧按 ``Instrument.lot_size`` 校验买单
    (``execution/common.rs``), 下单侧若按别的粒度取整, 算出的数量会被自己的风控以
    ``Quantity X is not a multiple of lot size Y`` 拒掉。此前这里只读策略属性
    ``self.lot_size``(缺省 **1**), 于是登记了 ``lot_size=100`` 的 A 股标的按 1 股
    取整 —— 实盘尤其无解, ``run_live`` 没有 ``lot_size`` 参数, 除了手写
    ``self.lot_size = 100`` 没有任何途径让取整逻辑知道登记值。

    优先级:

    1. 标的**未登记** lot_size(拿不到或 <=0): 用 ``self.lot_size``, 保持既有行为;
    2. ``self.lot_size`` 比登记值**更粗且是其整数倍**(如登记 100 而策略要按 200
       下单): 尊重策略的显式意图 —— 200 的倍数必然也是 100 的倍数, 不会被拒;
    3. 其余情况(含缺省的 1、以及与登记值不成倍数的值): 用登记值, 避免下出必然
       被自己风控拒掉的单。

    :param strategy: 策略实例。
    :param symbol: 标的代码。
    :return: 用于取整的最小交易单位(>=1 时才生效)。
    """
    explicit = _lot_size_from_strategy(strategy, symbol)
    registered = _lot_size_from_instrument(strategy, symbol)
    if registered <= 0:
        return explicit
    if explicit > registered and explicit % registered == 0:
        return explicit
    return registered


def calculate_max_buy_qty(
    strategy: Any, symbol: str, price: float, cash: float
) -> float:
    """计算考虑费率后的最大可买数量."""
    if price <= 0 or cash <= 0:
        return 0.0

    total_rate = float(strategy.commission_rate) + float(strategy.transfer_fee_rate)

    safety_margin = 0.0001
    if strategy.ctx and hasattr(strategy.ctx, "risk_config"):
        safety_margin = float(strategy.ctx.risk_config.safety_margin)

    safe_cash = float(cash) * (1.0 - float(safety_margin))
    est_qty = safe_cash / (float(price) * (1 + float(total_rate)))
    est_commission = est_qty * float(price) * float(strategy.commission_rate)

    if est_commission < float(strategy.min_commission):
        remaining_cash = safe_cash - float(strategy.min_commission)
        if remaining_cash <= 0:
            return 0.0
        est_qty = remaining_cash / (
            float(price) * (1 + float(strategy.transfer_fee_rate))
        )

    current_lot_size = _resolve_lot_size(strategy, symbol)

    if current_lot_size > 0:
        est_qty = (est_qty // current_lot_size) * current_lot_size

    return float(est_qty)


def order_target(
    strategy: Any,
    symbol: Optional[str] = None,
    target: Optional[float] = None,
    price: Optional[float] = None,
    **kwargs: Any,
) -> Optional[str]:
    """调整仓位到目标数量.

    Returns:
        本次调仓产生的订单 ID; 若无需交易 (已在目标) 则返回 None.
    """
    return _target_to_orders(strategy, symbol, target, price, **kwargs)


def _position_from_execution(strategy: Any, symbol: str, method: str) -> float:
    """按名取执行后端的持仓口径，缺失则退回 ``get_position``.

    ``get_closable_position`` / ``get_projected_position`` 是 #361 新增到
    :class:`ExecutionBackend` 协议上的方法。协议是公开且 ``runtime_checkable``
    的，第三方自定义后端不会有这两个方法，故此处降级而非抛错——代价是这类后端
    维持旧行为（结算仓口径），需自行实现新方法才能获得修复。
    """
    execution = strategy.execution
    getter = getattr(execution, method, None)
    if callable(getter):
        return float(getter(symbol))
    return float(execution.get_position(symbol))


def _target_to_orders(
    strategy: Any,
    symbol: Optional[str] = None,
    target_qty: Optional[float] = None,
    price: Optional[float] = None,
    round_to_lot: bool = True,
    **kwargs: Any,
) -> Optional[str]:
    """目标持仓 → 下单的共享核心：按 lot_size 取整 delta（可关闭），不撤单."""
    if target_qty is None:
        raise ValueError("target requires a target quantity (目标持仓数量)")
    symbol = resolve_symbol(strategy, symbol)
    # 目标仓位问的是"仓位最终会落在哪"，故按投影持仓（含全部在途单）算 delta。
    # 用结算仓会让同一 on_bar 内的连续调用按同一基准重复下单——例如先
    # close_position 再 order_target_percent，会在全平单之外再补一笔卖单造成
    # 超卖（与 #361 同源，均是"结算仓不含在途单"）。
    current_qty = _position_from_execution(strategy, symbol, "get_projected_position")
    delta_qty = target_qty - current_qty

    if round_to_lot:
        current_lot_size = _resolve_lot_size(strategy, symbol)
        if current_lot_size > 0:
            if delta_qty > 0:
                delta_qty = (delta_qty // current_lot_size) * current_lot_size
            elif delta_qty < 0:
                delta_qty = -((abs(delta_qty) // current_lot_size) * current_lot_size)

    # buy()/sell() 现返回 OrderReceipt；order_target 系列对外仍以 str 订单号
    # 为契约（不属于 Task 7 变更范围），故在此边界取 .primary 落地为字符串。
    # 空回执(前置风控/柜台拒单/状态未知)返回 None。
    if delta_qty > 0:
        receipt = buy(strategy, symbol, delta_qty, price, **kwargs)
        return str(getattr(receipt, "primary", receipt)) or None
    elif delta_qty < 0:
        receipt = sell(strategy, symbol, abs(delta_qty), price, **kwargs)
        return str(getattr(receipt, "primary", receipt)) or None
    return None


def order_target_value(
    strategy: Any,
    symbol: Optional[str] = None,
    target_value: Optional[float] = None,
    price: Optional[float] = None,
    **kwargs: Any,
) -> Optional[str]:
    """调整仓位到目标价值.

    Returns:
        本次调仓产生的订单 ID; 若无需交易或无法定价则返回 None.
    """
    if target_value is None:
        raise ValueError("order_target_value requires 'target_value' (目标持仓价值)")
    symbol = resolve_symbol(strategy, symbol)

    if price is not None:
        current_price = price
    else:
        current_price = strategy._last_prices.get(symbol, 0.0)

    if current_price == 0.0:
        if strategy.current_bar and strategy.current_bar.symbol == symbol:
            current_price = strategy.current_bar.close
        elif strategy.current_tick and strategy.current_tick.symbol == symbol:
            current_price = strategy.current_tick.price
        else:
            logger.warning(
                "Cannot determine price for %s, skipping order_target_value", symbol
            )
            return None

    target_qty = target_value / current_price
    return _target_to_orders(strategy, symbol, target_qty, price, **kwargs)


def order_target_percent(
    strategy: Any,
    symbol: Optional[str] = None,
    target_percent: Optional[float] = None,
    price: Optional[float] = None,
    **kwargs: Any,
) -> Optional[str]:
    """调整仓位到目标百分比.

    Returns:
        本次调仓产生的订单 ID; 若无需交易则返回 None.
    """
    if target_percent is None:
        raise ValueError(
            "order_target_percent requires 'target_percent' (目标持仓比例)"
        )
    portfolio_value = strategy.execution.get_portfolio_value()
    target_value = portfolio_value * float(target_percent)
    return order_target_value(strategy, symbol, target_value, price, **kwargs)


def rebalance_weights(
    strategy: Any,
    target_weights: Dict[str, float],
    price_map: Optional[Dict[str, float]] = None,
    liquidate_unmentioned: bool = False,
    allow_leverage: bool = False,
    rebalance_tolerance: float = 0.0,
    **kwargs: Any,
) -> List[str]:
    """按多标的目标权重调仓.

    Returns:
        本次调仓产生的所有订单 ID 列表 (无交易时为空列表).
    """
    _require_execution_ready(strategy)
    if rebalance_tolerance < 0:
        raise ValueError("rebalance_tolerance must be >= 0")

    normalized_weights: Dict[str, float] = {}
    for symbol, weight in target_weights.items():
        if not symbol:
            raise ValueError("symbol in target_weights must be non-empty")
        normalized_weight = float(weight)
        if normalized_weight < 0:
            raise ValueError(f"target weight for {symbol} must be >= 0")
        normalized_weights[symbol] = normalized_weight

    total_weight = sum(normalized_weights.values())
    if not allow_leverage and total_weight > 1.0 + 1e-8:
        raise ValueError(
            f"sum of target_weights ({total_weight:.6f}) exceeds 1.0; "
            "set allow_leverage=True to permit this"
        )

    if liquidate_unmentioned:
        for symbol, qty in strategy.execution.get_positions().items():
            if float(qty) != 0.0 and symbol not in normalized_weights:
                normalized_weights[symbol] = 0.0

    if not normalized_weights:
        return []

    portfolio_value = strategy.execution.get_portfolio_value()
    abs_tolerance_value = abs(float(portfolio_value)) * float(rebalance_tolerance)
    planned: List[Tuple[str, float, float]] = []

    for symbol, weight in normalized_weights.items():
        target_value = float(portfolio_value) * float(weight)
        current_qty = float(strategy.execution.get_position(symbol))

        current_price = strategy._last_prices.get(symbol, 0.0)
        if current_price == 0.0:
            if strategy.current_bar and strategy.current_bar.symbol == symbol:
                current_price = strategy.current_bar.close
            elif strategy.current_tick and strategy.current_tick.symbol == symbol:
                current_price = strategy.current_tick.price

        current_value = current_qty * float(current_price)
        delta_value = target_value - current_value
        if abs(delta_value) <= abs_tolerance_value:
            continue
        planned.append((symbol, target_value, delta_value))

    if not planned:
        return []

    sell_legs = [item for item in planned if item[2] < 0]
    buy_legs = [item for item in planned if item[2] >= 0]

    order_ids: List[str] = []
    for symbol, target_value, _ in sorted(
        sell_legs,
        key=lambda item: (float(item[2]), str(item[0])),
    ):
        leg_price = price_map.get(symbol) if price_map else None
        oid = order_target_value(strategy, symbol, target_value, leg_price, **kwargs)
        if oid is not None:
            order_ids.append(oid)

    for symbol, target_value, _ in sorted(
        buy_legs,
        key=lambda item: (-float(item[2]), str(item[0])),
    ):
        leg_price = price_map.get(symbol) if price_map else None
        oid = order_target_value(strategy, symbol, target_value, leg_price, **kwargs)
        if oid is not None:
            order_ids.append(oid)

    return order_ids


def rebalance_positions(
    strategy: Any,
    target_positions: Dict[str, float],
    price_map: Optional[Dict[str, float]] = None,
    liquidate_unmentioned: bool = False,
    rebalance_tolerance: float = 0.0,
    allow_short: Optional[bool] = None,
    strict_short_capability: bool = True,
    missing_price_mode: str = "ignore",
    **kwargs: Any,
) -> List[str]:
    """按多标的目标持仓数量调仓，支持正负目标仓位.

    Returns:
        本次调仓产生的所有订单 ID 列表 (无交易时为空列表).
    """
    _require_execution_ready(strategy)
    if rebalance_tolerance < 0:
        raise ValueError("rebalance_tolerance must be >= 0")
    normalized_missing_price_mode = str(missing_price_mode).strip().lower()
    if normalized_missing_price_mode not in {"ignore", "skip", "fail"}:
        raise ValueError("missing_price_mode must be one of: ignore, skip, fail")

    normalized_targets: Dict[str, float] = {}
    for symbol, target_qty in target_positions.items():
        if not symbol:
            raise ValueError("symbol in target_positions must be non-empty")
        normalized_targets[str(symbol)] = float(target_qty)

    has_short_target = any(
        float(target_qty) < 0.0 for target_qty in normalized_targets.values()
    )
    plan: Dict[str, Any] = {
        "requested_targets": dict(normalized_targets),
        "liquidate_unmentioned": bool(liquidate_unmentioned),
        "rebalance_tolerance": float(rebalance_tolerance),
        "allow_short": allow_short,
        "strict_short_capability": bool(strict_short_capability),
        "missing_price_mode": normalized_missing_price_mode,
        "reduce_legs": [],
        "increase_legs": [],
        "skipped_legs": [],
        "submitted_legs": [],
    }
    setattr(strategy, "_last_target_positions_plan", plan)
    if has_short_target:
        capabilities = get_execution_capabilities(strategy)
        plan["execution_capabilities"] = dict(capabilities)
        inferred_allow_short = _supports_short_targets(strategy, capabilities)
        effective_allow_short = (
            inferred_allow_short if allow_short is None else bool(allow_short)
        )
        if not effective_allow_short:
            reject_reason = (
                "negative target positions require allow_short=True "
                "and a short-enabled execution environment"
            )
            plan["status"] = "rejected"
            plan["reject_reason"] = reject_reason
            raise ValueError(reject_reason)
        if strict_short_capability and not inferred_allow_short:
            broker_name = str(capabilities.get("broker_name", "")).strip()
            broker_hint = f" for broker '{broker_name}'" if broker_name else ""
            plan["status"] = "rejected"
            plan["reject_reason"] = (
                "current execution environment does not advertise short-sell support"
                f"{broker_hint}"
            )
            raise RuntimeError(
                "current execution environment does not advertise short-sell support"
                f"{broker_hint}"
            )

    if liquidate_unmentioned:
        for symbol, qty in strategy.execution.get_positions().items():
            if float(qty) != 0.0 and symbol not in normalized_targets:
                normalized_targets[str(symbol)] = 0.0

    if not normalized_targets:
        plan["status"] = "noop"
        return []

    reduce_legs: List[Tuple[str, float, float]] = []
    increase_legs: List[Tuple[str, float, float]] = []

    for symbol, target_qty in normalized_targets.items():
        current_qty = float(strategy.execution.get_position(symbol))
        delta_qty = float(target_qty) - current_qty
        if abs(delta_qty) <= float(rebalance_tolerance):
            continue
        is_reduction_or_reversal = current_qty != 0.0 and (
            float(target_qty) == 0.0
            or current_qty * float(target_qty) < 0.0
            or abs(float(target_qty)) < abs(current_qty)
        )
        if is_reduction_or_reversal:
            reduce_legs.append((symbol, target_qty, abs(delta_qty)))
        else:
            increase_legs.append((symbol, target_qty, abs(delta_qty)))

    plan["reduce_legs"] = [
        {
            "symbol": symbol,
            "target_quantity": float(target_qty),
            "delta_quantity": float(target_qty)
            - float(strategy.execution.get_position(symbol)),
            "phase": "reduce",
        }
        for symbol, target_qty, _ in reduce_legs
    ]
    plan["increase_legs"] = [
        {
            "symbol": symbol,
            "target_quantity": float(target_qty),
            "delta_quantity": float(target_qty)
            - float(strategy.execution.get_position(symbol)),
            "phase": "increase",
        }
        for symbol, target_qty, _ in increase_legs
    ]

    if not reduce_legs and not increase_legs:
        plan["status"] = "noop"
        return []

    order_ids: List[str] = []
    for symbol, target_qty, _ in sorted(
        reduce_legs,
        key=lambda item: (float(item[2]), str(item[0])),
        reverse=True,
    ):
        if price_map is not None and symbol not in price_map:
            if normalized_missing_price_mode == "skip":
                plan["skipped_legs"].append(
                    {
                        "symbol": symbol,
                        "target_quantity": float(target_qty),
                        "reason": "missing_price_map",
                        "phase": "reduce",
                    }
                )
                continue
            if normalized_missing_price_mode == "fail":
                missing_price_error = (
                    f"missing price_map entry for symbol '{symbol}' "
                    "in rebalance_positions"
                )
                plan["status"] = "rejected"
                plan["reject_reason"] = missing_price_error
                raise RuntimeError(missing_price_error)
        leg_price = price_map.get(symbol) if price_map else None
        oid = _target_to_orders(strategy, symbol, target_qty, leg_price, **kwargs)
        plan["submitted_legs"].append(
            {
                "symbol": symbol,
                "target_quantity": float(target_qty),
                "price": leg_price,
                "phase": "reduce",
                "order_id": oid,
            }
        )
        if oid is not None:
            order_ids.append(oid)

    for symbol, target_qty, _ in sorted(
        increase_legs,
        key=lambda item: (float(item[2]), str(item[0])),
        reverse=True,
    ):
        if price_map is not None and symbol not in price_map:
            if normalized_missing_price_mode == "skip":
                plan["skipped_legs"].append(
                    {
                        "symbol": symbol,
                        "target_quantity": float(target_qty),
                        "reason": "missing_price_map",
                        "phase": "increase",
                    }
                )
                continue
            if normalized_missing_price_mode == "fail":
                missing_price_error = (
                    f"missing price_map entry for symbol '{symbol}' "
                    "in rebalance_positions"
                )
                plan["status"] = "rejected"
                plan["reject_reason"] = missing_price_error
                raise RuntimeError(missing_price_error)
        leg_price = price_map.get(symbol) if price_map else None
        oid = _target_to_orders(strategy, symbol, target_qty, leg_price, **kwargs)
        plan["submitted_legs"].append(
            {
                "symbol": symbol,
                "target_quantity": float(target_qty),
                "price": leg_price,
                "phase": "increase",
                "order_id": oid,
            }
        )
        if oid is not None:
            order_ids.append(oid)
    plan["status"] = "submitted"
    return order_ids


def close_position(strategy: Any, symbol: Optional[str] = None) -> Optional[str]:
    """平掉当前持仓（全平，含 A 股零股；不按手数取整）.

    :return: 平仓单的订单号; 当前无持仓(无需交易)时返回 ``None``。返回类型与
        同层的 ``order_target*`` 对齐(它们共用 ``_target_to_orders``) —— 此前
        这里丢弃了返回值, 调用方拿不到 order id, 既查不了也撤不了。
    """
    symbol = resolve_symbol(strategy, symbol)
    return _target_to_orders(strategy, symbol=symbol, target_qty=0, round_to_lot=False)


def short(
    strategy: Any,
    symbol: Optional[str] = None,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: Optional[TimeInForce] = None,
    trigger_price: Optional[float] = None,
    tag: Optional[str] = None,
    fill_mode: Optional["FillMode"] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    reduce_only: bool = False,
) -> Optional[OrderReceipt]:
    """卖出开空 (Short Sell).

    :return: 委托回执; 当 ``quantity`` 经 sizer 计算后不为正(没有单可下)时返回
        ``None``。返回类型与同层的 :func:`buy` / :func:`sell` 对齐 —— 此前这里
        丢弃了下层的返回值, 调用方拿不到 order id, 既查不了也撤不了。
    """
    _reject_legacy_fill_mode(fill_mode)
    submit_order_method = getattr(strategy, "submit_order", None)
    if callable(submit_order_method):
        return cast(
            OrderReceipt,
            submit_order_method(
                symbol=symbol,
                side="Sell",
                quantity=quantity,
                price=price,
                time_in_force=time_in_force,
                trigger_price=trigger_price,
                tag=tag,
                fill_mode=fill_mode,
                slippage=slippage,
                commission=commission,
                position_effect="open",
                reduce_only=reduce_only,
            ),
        )
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")

    symbol = resolve_symbol(strategy, symbol)

    ref_price = price
    if ref_price is None:
        if strategy.current_bar:
            ref_price = strategy.current_bar.close
        elif strategy.current_tick:
            ref_price = strategy.current_tick.price
        else:
            ref_price = 0.0

    if quantity is None:
        quantity = strategy.sizer.get_size(
            ref_price, strategy.ctx.cash, strategy.ctx, symbol
        )

    if quantity > 0:
        return _submit_sell_side(
            strategy=strategy,
            symbol=symbol,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            tag=tag,
            fill_mode=fill_mode,
            slippage=slippage,
            commission=commission,
            position_effect="open",
            reduce_only=reduce_only,
        )
    return None


def cover(
    strategy: Any,
    symbol: Optional[str] = None,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: Optional[TimeInForce] = None,
    trigger_price: Optional[float] = None,
    tag: Optional[str] = None,
    fill_mode: Optional["FillMode"] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    reduce_only: bool = False,
) -> Optional[OrderReceipt]:
    """买入平空 (Buy to Cover).

    :return: 委托回执; 当前无空头持仓、或 ``quantity`` 不为正(没有单可下)时返回
        ``None``。返回类型与同层的 :func:`buy` / :func:`sell` 对齐 —— 此前这里
        丢弃了下层的返回值, 调用方拿不到 order id, 既查不了也撤不了。
    """
    _reject_legacy_fill_mode(fill_mode)
    submit_order_method = getattr(strategy, "submit_order", None)
    if callable(submit_order_method):
        return cast(
            OrderReceipt,
            submit_order_method(
                symbol=symbol,
                side="Buy",
                quantity=quantity,
                price=price,
                time_in_force=time_in_force,
                trigger_price=trigger_price,
                tag=tag,
                fill_mode=fill_mode,
                slippage=slippage,
                commission=commission,
                position_effect="close",
                reduce_only=reduce_only,
            ),
        )
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")

    symbol = resolve_symbol(strategy, symbol)

    if quantity is None:
        pos = strategy.execution.get_position(symbol)
        if pos < 0:
            quantity = abs(pos)
        else:
            return None

    if quantity > 0:
        return _submit_buy_side(
            strategy=strategy,
            symbol=symbol,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            tag=tag,
            fill_mode=fill_mode,
            slippage=slippage,
            commission=commission,
            position_effect="close",
            reduce_only=reduce_only,
        )
    return None


def get_cash(strategy: Any) -> float:
    """获取现金（经执行后端）."""
    return float(strategy.execution.get_cash())


def get_buying_power(strategy: Any) -> float:
    """获取可用买入力（经执行后端）."""
    return float(strategy.execution.get_buying_power())


# --- SimExecution 后端原语（Task 1 引入；Task 3/4 让公共函数 delegate 到 execution）---
def _sim_get_position(strategy: Any, symbol: Optional[str] = None) -> float:
    if strategy.ctx is None:
        return 0.0
    return float(strategy.ctx.get_position(resolve_symbol(strategy, symbol)))


def _sim_get_available_position(strategy: Any, symbol: Optional[str] = None) -> float:
    if strategy.ctx is None:
        return 0.0
    return float(strategy.ctx.get_available_position(resolve_symbol(strategy, symbol)))


def _sim_get_closable_position(strategy: Any, symbol: Optional[str] = None) -> float:
    """可平持仓：结算仓 − 在途平仓/减仓单占用（auto 拆腿用，见 #361）."""
    if strategy.ctx is None:
        return 0.0
    return float(strategy.ctx.get_closable_position(resolve_symbol(strategy, symbol)))


def _sim_get_projected_position(strategy: Any, symbol: Optional[str] = None) -> float:
    """投影持仓：结算仓 + 全部在途单效果（目标仓位算 delta 用）."""
    if strategy.ctx is None:
        return 0.0
    return float(strategy.ctx.get_projected_position(resolve_symbol(strategy, symbol)))


def _sim_get_positions(strategy: Any) -> Dict[str, float]:
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")
    return cast(Dict[str, float], strategy.ctx.positions)


def _sim_get_holding_bars(strategy: Any, symbol: Optional[str] = None) -> int:
    if strategy.ctx is None:
        return 0
    return int(strategy._hold_bars[resolve_symbol(strategy, symbol)])


def _sim_get_cash(strategy: Any) -> float:
    if strategy.ctx is None:
        return 0.0
    return float(strategy.ctx.cash)


def _sim_get_buying_power(strategy: Any) -> float:
    if strategy.ctx is None:
        return 0.0
    return float(strategy.ctx.buying_power)


def _sim_get_open_orders(strategy: Any, symbol: Optional[str] = None) -> List[Any]:
    """获取当前未完成的订单（_sim_ 原语，复制自 get_open_orders，逻辑一字不改）."""
    if strategy.ctx is None:
        return []

    canceled_order_ids = {
        str(order_id)
        for order_id in getattr(strategy.ctx, "canceled_order_ids", [])
        if order_id
    }
    pending_canceled_ids: Set[str] = getattr(
        strategy, "_pending_canceled_order_ids", set()
    )
    if not isinstance(pending_canceled_ids, set):
        pending_canceled_ids = set()
        setattr(strategy, "_pending_canceled_order_ids", pending_canceled_ids)
    canceled_order_ids.update(
        str(order_id) for order_id in pending_canceled_ids if order_id
    )

    orders = [
        o
        for o in strategy.ctx.active_orders
        if getattr(o, "id", "") not in canceled_order_ids
        if o.status
        in (OrderStatus.New, OrderStatus.Submitted, OrderStatus.PartiallyFilled)
    ]
    if symbol:
        return [o for o in orders if o.symbol == symbol]
    return orders


def _sim_get_portfolio_value(strategy: Any) -> float:
    """计算组合总价值（_sim_ 原语，复制自 get_portfolio_value，逻辑一字不改）."""
    if strategy.ctx is None:
        return 0.0
    use_previous_snapshot = bool(
        getattr(strategy, "_framework_use_previous_account_snapshot", False)
    )
    ctx_equity = (
        getattr(strategy.ctx, "previous_account_equity", None)
        if use_previous_snapshot
        else getattr(strategy.ctx, "account_equity", None)
    )
    if isinstance(ctx_equity, (int, float)):
        return float(ctx_equity)
    engine = getattr(strategy, "_engine", None)
    get_metrics = getattr(engine, "get_account_metrics", None)
    if callable(get_metrics):
        equity, _, _, _, _, _ = get_metrics()
        return float(equity)

    total_value = float(strategy.ctx.cash)
    for sym, qty in strategy.ctx.positions.items():
        if qty == 0:
            continue

        price = strategy._last_prices.get(sym, 0.0)
        if price == 0.0:
            if strategy.current_bar and strategy.current_bar.symbol == sym:
                price = strategy.current_bar.close
            elif strategy.current_tick and strategy.current_tick.symbol == sym:
                price = strategy.current_tick.price

        total_value += float(qty) * price
    return total_value


def _sim_get_account(strategy: Any) -> Dict[str, Any]:
    """获取账户资金详情快照（_sim_ 原语，复制自 get_account，逻辑一字不改）."""
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")

    use_previous_snapshot = bool(
        getattr(strategy, "_framework_use_previous_account_snapshot", False)
    )
    cash_source = (
        getattr(strategy.ctx, "previous_cash", strategy.ctx.cash)
        if use_previous_snapshot
        else strategy.ctx.cash
    )
    cash = float(cash_source)
    prefix = "previous_account_" if use_previous_snapshot else "account_"
    ctx_market_value = getattr(strategy.ctx, f"{prefix}market_value", None)
    ctx_notional_value = getattr(strategy.ctx, f"{prefix}notional_value", None)
    ctx_used_margin = getattr(strategy.ctx, f"{prefix}used_margin", None)
    ctx_unrealized_pnl = getattr(strategy.ctx, f"{prefix}unrealized_pnl", None)
    ctx_maintenance_ratio = getattr(strategy.ctx, f"{prefix}maintenance_ratio", None)
    ctx_equity = getattr(strategy.ctx, f"{prefix}equity", None)
    engine = getattr(strategy, "_engine", None)
    get_metrics = getattr(engine, "get_account_metrics", None)
    notional_value = (
        float(ctx_notional_value) if ctx_notional_value is not None else 0.0
    )
    unrealized_pnl = (
        float(ctx_unrealized_pnl) if ctx_unrealized_pnl is not None else 0.0
    )
    if isinstance(ctx_equity, (int, float)):
        equity = float(ctx_equity)
        market_value = (
            float(ctx_market_value)
            if isinstance(ctx_market_value, (int, float))
            else equity - cash
        )
        margin = (
            float(ctx_used_margin) if isinstance(ctx_used_margin, (int, float)) else 0.0
        )
        maintenance_ratio = (
            float(ctx_maintenance_ratio)
            if isinstance(ctx_maintenance_ratio, (int, float))
            else 0.0
        )
    elif callable(get_metrics):
        (
            equity,
            market_value,
            notional_value,
            margin,
            unrealized_pnl,
            maintenance_ratio,
        ) = get_metrics()
        equity = float(equity)
        market_value = float(market_value)
        margin = float(margin)
        maintenance_ratio = float(maintenance_ratio)
    else:
        equity = float(strategy.equity)
        market_value = float(equity - cash)
        margin = float(getattr(strategy.ctx, "account_used_margin", 0.0) or 0.0)
        maintenance_ratio = 0.0
    previous_details = (
        getattr(strategy, "_framework_previous_account_details", None)
        if use_previous_snapshot
        else None
    )
    # frozen_cash / short_market_value are authoritative Rust values on the
    # StrategyContext. On the previous-snapshot path they are read from the
    # framework cache (which was populated from those same Rust values one
    # period earlier); the Python re-implementations were removed.
    if isinstance(previous_details, dict):
        frozen_cash = float(previous_details.get("frozen_cash", 0.0))
        short_market_value = float(previous_details.get("short_market_value", 0.0))
    else:
        frozen_cash = float(getattr(strategy.ctx, "account_frozen_cash", 0.0))
        short_market_value = float(
            getattr(strategy.ctx, "account_short_market_value", 0.0)
        )
    borrowed_cash = float(max(-cash, 0.0))
    if ctx_equity is None and not callable(get_metrics):
        denominator = market_value + short_market_value
        maintenance_ratio = float(equity / denominator) if denominator > 0.0 else 0.0
    account_mode = "margin" if _is_margin_account(strategy) else "cash"
    if isinstance(previous_details, dict):
        accrued_interest = float(previous_details.get("margin_accrued_interest", 0.0))
        daily_interest = float(previous_details.get("margin_daily_interest", 0.0))
    else:
        accrued_interest = float(getattr(strategy.ctx, "margin_accrued_interest", 0.0))
        daily_interest = float(getattr(strategy.ctx, "margin_daily_interest", 0.0))
    # 可用保证金 = 总权益 - 已占用保证金.
    # 与开仓资金校验(拒单信息中的 Available)口径一致.
    # 期货保证金账户下, cash 仅为现金余额(开仓不扣保证金).
    # 真正可用于新开仓的是 free_margin.
    free_margin = equity - margin
    return {
        "cash": cash,
        "equity": equity,
        "market_value": market_value,
        "notional_value": float(notional_value),
        "frozen_cash": frozen_cash,
        "margin": margin,
        "used_margin": margin,
        "free_margin": free_margin,
        "unrealized_pnl": float(unrealized_pnl),
        "borrowed_cash": borrowed_cash,
        "short_market_value": float(short_market_value),
        "maintenance_ratio": maintenance_ratio,
        "account_mode": account_mode,
        "accrued_interest": accrued_interest,
        "daily_interest": daily_interest,
    }


def _sim_capabilities(strategy: Any) -> Dict[str, Any]:
    risk_config = getattr(getattr(strategy, "ctx", None), "risk_config", None)
    account_mode = str(getattr(risk_config, "account_mode", "cash")).strip().lower()
    supports_short_sell = bool(getattr(risk_config, "enable_short_sell", False))
    return {
        "broker_live": False,
        "client_order_id": False,
        "order_type": True,
        "time_in_force_str": False,
        "position_effect": True,
        "reduce_only": True,
        "position_details": False,
        "account_mode": account_mode,
        "supports_short_sell": supports_short_sell,
        "broker_extra_fields": [],
    }


def _remember_pending_cancel(strategy: Any, order_id: str) -> None:
    """记录撤单意图, 按 FIFO 淘汰(同 remember_trade_key).

    ``ctx.canceled_order_ids`` 只是当拍增量, 该集合负责让撤单意图跨拍生效;
    paper 模式走 SimExecution 且可长跑, 因此必须有上限。
    """
    pending: Set[str] = getattr(strategy, "_pending_canceled_order_ids", set())
    if not isinstance(pending, set):
        pending = set()
        setattr(strategy, "_pending_canceled_order_ids", pending)
    order_ids = getattr(strategy, "_pending_canceled_order_id_queue", None)
    if not isinstance(order_ids, deque):
        order_ids = deque()
        setattr(strategy, "_pending_canceled_order_id_queue", order_ids)

    if order_id not in pending:
        pending.add(order_id)
        order_ids.append(order_id)

    raw_limit = getattr(strategy, "pending_cancel_cache_size", 50000)
    try:
        limit = max(1, int(raw_limit))
    except (TypeError, ValueError):
        limit = 50000
    while len(order_ids) > limit:
        oldest = order_ids.popleft()
        pending.discard(oldest)


def _sim_cancel_order(strategy: Any, order_id: str) -> None:
    """取消指定订单（_sim_ 原语）.

    撤单意图记到 ``_pending_canceled_order_ids`` 后即透传引擎。不再尝试直接改写
    ``ctx.active_orders`` 里的 ``status``: 该字段是 Rust 侧 ``Vec<Order>`` 经
    ``#[pyo3(get)]`` 暴露的, 每次访问都克隆出新的 Python 包装对象, 写入不落回引擎
    (下次读到的仍是撤单前状态)。状态呈现由 get_order / get_open_orders 按撤单意图
    集合裁定。
    """
    if strategy.ctx:
        _remember_pending_cancel(strategy, order_id)
        strategy.ctx.cancel_order(order_id)


def _sim_submit_order(
    strategy: Any,
    symbol: Optional[str] = None,
    side: str = "Buy",
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: Optional[TimeInForce | str] = None,
    trigger_price: Optional[float] = None,
    tag: Optional[str] = None,
    client_order_id: Optional[str] = None,
    order_type: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    broker_options: Optional[Dict[str, Any]] = None,
    trail_offset: Optional[float] = None,
    trail_reference_price: Optional[float] = None,
    fill_mode: Optional["FillMode"] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: Union[PositionEffect, str, None] = None,
    reduce_only: bool = False,
    asset_type: str = "stock",
) -> OrderReceipt:
    """统一下单接口（_sim_ 原语，复制自 submit_order:755-845，逻辑一字不改）."""
    _reject_legacy_fill_mode(fill_mode)
    capabilities = get_execution_capabilities(strategy)
    if client_order_id and not bool(capabilities.get("client_order_id", False)):
        raise RuntimeError("client_order_id is not supported in current execution mode")
    if extra:
        raise RuntimeError(
            "extra broker fields require broker_live mode "
            "(not available in simulated/backtest execution)"
        )
    if normalize_asset_type(asset_type) != "stock":
        raise RuntimeError(
            "non-stock asset_type requires broker_live mode "
            "(not available in simulated/backtest execution)"
        )
    order_type_key, order_type_enum = _parse_order_type(order_type)
    if time_in_force is not None and not isinstance(time_in_force, TimeInForce):
        raise RuntimeError(
            "time_in_force string is not supported in current execution mode"
        )
    if order_type_key in {"stoptrail", "stoptraillimit"}:
        if trail_offset is None or trail_offset <= 0:
            raise RuntimeError("trail_offset must be > 0 for trailing orders")
    if order_type_key == "stoptraillimit" and price is None:
        raise RuntimeError("price must be provided for StopTrailLimit order")
    if order_type_key in {"stoptrail", "stoptraillimit"} and order_type_enum is None:
        raise RuntimeError("trailing order requires runtime with StopTrail support")

    side_text = side.strip().lower()
    normalized_position_effect = _normalize_position_effect(position_effect, "auto")
    if side_text == "buy":
        order_ids = _submit_buy_side_orders(
            strategy=strategy,
            symbol=symbol,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            tag=tag,
            order_type=order_type_enum,
            trail_offset=trail_offset,
            trail_reference_price=trail_reference_price,
            fill_mode=fill_mode,
            slippage=slippage,
            commission=commission,
            position_effect=normalized_position_effect,
            reduce_only=reduce_only,
        )
        _record_broker_options(strategy, order_ids, broker_options)
        return _orders_to_receipt(order_ids, position_effect=normalized_position_effect)
    if side_text == "sell":
        order_ids = _submit_sell_side_orders(
            strategy=strategy,
            symbol=symbol,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            tag=tag,
            order_type=order_type_enum,
            trail_offset=trail_offset,
            trail_reference_price=trail_reference_price,
            fill_mode=fill_mode,
            slippage=slippage,
            commission=commission,
            position_effect=normalized_position_effect,
            reduce_only=reduce_only,
        )
        _record_broker_options(strategy, order_ids, broker_options)
        return _orders_to_receipt(order_ids, position_effect=normalized_position_effect)
    raise ValueError(f"Unsupported side: {side}")
