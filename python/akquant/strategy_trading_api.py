import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from .akquant import OrderStatus, OrderType, PositionEffect, TimeInForce
from .gateway.broker_models import normalize_asset_type

OrderFillPolicy = Dict[str, Any]
OrderSlippage = Dict[str, Any]
OrderCommission = Dict[str, Any]


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
    """获取指定标的的持仓数量."""
    if strategy.ctx is None:
        return 0.0
    symbol = resolve_symbol(strategy, symbol)
    return float(strategy.ctx.get_position(symbol))


def get_available_position(strategy: Any, symbol: Optional[str] = None) -> float:
    """获取指定标的的可用持仓数量."""
    if strategy.ctx is None:
        return 0.0
    symbol = resolve_symbol(strategy, symbol)
    return float(strategy.ctx.get_available_position(symbol))


def hold_bar(strategy: Any, symbol: Optional[str] = None) -> int:
    """获取当前持仓持有的 Bar 数量."""
    if strategy.ctx is None:
        return 0
    symbol = resolve_symbol(strategy, symbol)
    return int(strategy._hold_bars[symbol])


def get_positions(strategy: Any) -> Dict[str, float]:
    """获取所有持仓信息."""
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")
    return cast(Dict[str, float], strategy.ctx.positions)


def get_last_target_positions_plan(strategy: Any) -> Dict[str, Any]:
    """获取最近一次 order_target_positions() 生成的调仓计划."""
    plan = getattr(strategy, "_last_target_positions_plan", None)
    if isinstance(plan, dict):
        return cast(Dict[str, Any], plan)
    return {}


def get_open_orders(strategy: Any, symbol: Optional[str] = None) -> List[Any]:
    """获取当前未完成的订单."""
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


def cancel_order(strategy: Any, order_id: str) -> None:
    """取消指定订单."""
    if strategy.ctx:
        pending_canceled_ids: Set[str] = getattr(
            strategy, "_pending_canceled_order_ids", set()
        )
        if not isinstance(pending_canceled_ids, set):
            pending_canceled_ids = set()
            setattr(strategy, "_pending_canceled_order_ids", pending_canceled_ids)
        pending_canceled_ids.add(order_id)
        strategy.ctx.cancel_order(order_id)
        for order in strategy.ctx.active_orders:
            if getattr(order, "id", "") != order_id:
                continue
            if getattr(order, "status", None) not in (
                OrderStatus.New,
                OrderStatus.Submitted,
                OrderStatus.PartiallyFilled,
            ):
                continue
            try:
                order.status = OrderStatus.Cancelled
            except Exception:
                pass
            break


def cancel_all_orders(strategy: Any, symbol: Optional[str] = None) -> None:
    """取消当前所有未完成的订单."""
    for order in get_open_orders(strategy, symbol=symbol):
        cancel_order(strategy, order.id)


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
    fill_policy: Optional[OrderFillPolicy] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: Union[PositionEffect, str, None] = None,
    reduce_only: bool = False,
) -> str:
    """买入下单."""
    submit_order_method = getattr(strategy, "submit_order", None)
    if callable(submit_order_method):
        return cast(
            str,
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
                fill_policy=fill_policy,
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
        fill_policy=fill_policy,
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
    fill_policy: Optional[OrderFillPolicy] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: Union[PositionEffect, str, None] = None,
    reduce_only: bool = False,
) -> str:
    """卖出下单."""
    submit_order_method = getattr(strategy, "submit_order", None)
    if callable(submit_order_method):
        return cast(
            str,
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
                fill_policy=fill_policy,
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
        fill_policy=fill_policy,
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
    fill_policy: Optional[OrderFillPolicy] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: str = "auto",
    reduce_only: bool = False,
) -> str:
    return _first_order_id(
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
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
            position_effect=position_effect,
            reduce_only=reduce_only,
        )
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
    fill_policy: Optional[OrderFillPolicy] = None,
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
        current_position = float(strategy.ctx.get_position(symbol))
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
                    fill_policy=fill_policy,
                    slippage=slippage,
                    commission=commission,
                    position_effect=leg_effect,
                    reduce_only=leg_reduce_only,
                )
            )
        return order_ids

    effective_fill_policy = _resolve_effective_order_fill_policy(strategy, fill_policy)
    fill_price_basis, fill_bar_offset, fill_temporal = _normalize_order_fill_policy(
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
                fill_price_basis,
                fill_bar_offset,
                fill_temporal,
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
    fill_policy: Optional[OrderFillPolicy] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: str = "auto",
    reduce_only: bool = False,
) -> str:
    return _first_order_id(
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
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
            position_effect=position_effect,
            reduce_only=reduce_only,
        )
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
    fill_policy: Optional[OrderFillPolicy] = None,
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
        current_position = float(strategy.ctx.get_position(symbol))
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
                    fill_policy=fill_policy,
                    slippage=slippage,
                    commission=commission,
                    position_effect=leg_effect,
                    reduce_only=leg_reduce_only,
                )
            )
        return order_ids

    effective_fill_policy = _resolve_effective_order_fill_policy(strategy, fill_policy)
    fill_price_basis, fill_bar_offset, fill_temporal = _normalize_order_fill_policy(
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
                fill_price_basis,
                fill_bar_offset,
                fill_temporal,
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
    """获取当前执行环境能力描述."""
    injected_method = getattr(
        getattr(strategy, "__dict__", {}), "get", lambda *_: None
    )("get_execution_capabilities")
    if callable(injected_method):
        return cast(Dict[str, Any], injected_method())
    execution = getattr(strategy, "execution", None)
    if execution is not None:
        return cast(Dict[str, Any], execution.capabilities())
    return _sim_capabilities(strategy)


def _reject_target_orders_in_broker_live(strategy: Any) -> None:
    """Raise clearly for buy_all/order_target* helpers when running broker_live.

    These helpers size off the (simulated) backtest ctx, which is not the
    broker's source of truth in broker_live — raise instead of silently
    mis-sizing.
    """
    if get_execution_capabilities(strategy).get("broker_live"):
        raise RuntimeError(
            "组合目标类下单(buy_all/order_target*)在 broker_live 暂不支持;"
            "请用 submit_order/buy 并按 get_account()/get_position() 手动 sizing"
        )


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


def _first_order_id(order_ids: List[str]) -> str:
    for order_id in order_ids:
        if order_id:
            return str(order_id)
    return ""


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


def submit_order(
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
    fill_policy: Optional[OrderFillPolicy] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: Union[PositionEffect, str, None] = None,
    reduce_only: bool = False,
    asset_type: str = "stock",
) -> str:
    """统一下单接口."""
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
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
            position_effect=_normalize_position_effect(position_effect, "auto"),
            reduce_only=reduce_only,
        )
        _record_broker_options(strategy, order_ids, broker_options)
        return _first_order_id(order_ids)
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
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
            position_effect=_normalize_position_effect(position_effect, "auto"),
            reduce_only=reduce_only,
        )
        _record_broker_options(strategy, order_ids, broker_options)
        return _first_order_id(order_ids)
    raise ValueError(f"Unsupported side: {side}")


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


def _normalize_order_fill_policy(
    fill_policy: Optional[OrderFillPolicy],
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    if fill_policy is None:
        return None, None, None
    if not isinstance(fill_policy, dict):
        raise TypeError("fill_policy must be a dict when provided")
    raw_basis = str(fill_policy.get("price_basis", "open")).strip().lower()
    raw_temporal = str(fill_policy.get("temporal", "same_cycle")).strip().lower()
    if raw_basis not in {"open", "close", "ohlc4", "hl2"}:
        raise ValueError(
            "fill_policy.price_basis must be one of: open, close, ohlc4, hl2"
        )
    if raw_temporal not in {"same_cycle", "next_event"}:
        raise ValueError("fill_policy.temporal must be one of: same_cycle, next_event")
    raw_offset_value = fill_policy.get(
        "bar_offset",
        0 if raw_basis == "close" else 1,
    )
    try:
        raw_offset = int(raw_offset_value)
    except (TypeError, ValueError):
        raise ValueError("fill_policy.bar_offset must be 0 or 1") from None
    if raw_offset not in {0, 1}:
        raise ValueError("fill_policy.bar_offset must be 0 or 1")
    if raw_basis in {"open", "ohlc4", "hl2"} and raw_offset != 1:
        raise ValueError(f"fill_policy({raw_basis}) requires bar_offset=1")
    return raw_basis, raw_offset, raw_temporal


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


def stop_buy(
    strategy: Any,
    symbol: Optional[str] = None,
    trigger_price: float = 0.0,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: Optional[TimeInForce] = None,
) -> None:
    """发送止损买入单."""
    buy(strategy, symbol, quantity, price, time_in_force, trigger_price=trigger_price)


def stop_sell(
    strategy: Any,
    symbol: Optional[str] = None,
    trigger_price: float = 0.0,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: Optional[TimeInForce] = None,
) -> None:
    """发送止损卖出单."""
    sell(strategy, symbol, quantity, price, time_in_force, trigger_price=trigger_price)


def get_portfolio_value(strategy: Any) -> float:
    """计算当前投资组合总价值 (现金 + 持仓市值)."""
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
    """获取账户资金详情快照."""
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

    current_lot_size = 1
    if isinstance(strategy.lot_size, int):
        current_lot_size = int(strategy.lot_size)
    elif isinstance(strategy.lot_size, dict):
        val = strategy.lot_size.get(symbol, strategy.lot_size.get("DEFAULT", 1))
        current_lot_size = int(val) if val is not None else 1

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
    _reject_target_orders_in_broker_live(strategy)
    return _order_target_core(strategy, symbol, target, price, **kwargs)


def _order_target_core(
    strategy: Any,
    symbol: Optional[str] = None,
    target: Optional[float] = None,
    price: Optional[float] = None,
    **kwargs: Any,
) -> Optional[str]:
    """order_target 的无守卫核心：供已自带能力校验的调仓入口内部复用."""
    if target is None:
        raise ValueError("order_target requires 'target' (目标持仓数量)")
    symbol = resolve_symbol(strategy, symbol)

    current_qty = 0.0
    if strategy.ctx:
        current_qty = float(strategy.ctx.get_position(symbol))

    delta_qty = target - current_qty

    if delta_qty > 0:
        return buy(strategy, symbol, delta_qty, price, **kwargs)
    elif delta_qty < 0:
        return sell(strategy, symbol, abs(delta_qty), price, **kwargs)
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
    _reject_target_orders_in_broker_live(strategy)
    if target_value is None:
        raise ValueError("order_target_value requires 'target_value' (目标持仓价值)")
    symbol = resolve_symbol(strategy, symbol)

    cancel_all_orders(strategy, symbol=symbol)

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
            print(
                f"Warning: Cannot determine price for {symbol}, "
                "skipping order_target_value"
            )
            return None

    current_qty = 0.0
    if strategy.ctx:
        current_qty = float(strategy.ctx.get_position(symbol))

    target_qty = target_value / current_price
    delta_qty = target_qty - current_qty

    current_lot_size = 1
    if isinstance(strategy.lot_size, int):
        current_lot_size = strategy.lot_size
    elif isinstance(strategy.lot_size, dict):
        val = strategy.lot_size.get(symbol, strategy.lot_size.get("DEFAULT", 1))
        current_lot_size = int(val) if val is not None else 1

    if current_lot_size > 0:
        if delta_qty > 0:
            delta_qty = (delta_qty // current_lot_size) * current_lot_size
        elif delta_qty < 0:
            delta_qty = -((abs(delta_qty) // current_lot_size) * current_lot_size)

    if delta_qty > 0:
        return buy(strategy, symbol, delta_qty, price, **kwargs)
    elif delta_qty < 0:
        return sell(strategy, symbol, abs(delta_qty), price, **kwargs)
    return None


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
    _reject_target_orders_in_broker_live(strategy)
    if target_percent is None:
        raise ValueError(
            "order_target_percent requires 'target_percent' (目标持仓比例)"
        )
    portfolio_value = get_portfolio_value(strategy)
    target_value = portfolio_value * float(target_percent)
    return order_target_value(strategy, symbol, target_value, price, **kwargs)


def order_target_weights(
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
    _reject_target_orders_in_broker_live(strategy)
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")

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
        for symbol, qty in strategy.ctx.positions.items():
            if float(qty) != 0.0 and symbol not in normalized_weights:
                normalized_weights[symbol] = 0.0

    if not normalized_weights:
        return []

    portfolio_value = get_portfolio_value(strategy)
    abs_tolerance_value = abs(float(portfolio_value)) * float(rebalance_tolerance)
    planned: List[Tuple[str, float, float]] = []

    for symbol, weight in normalized_weights.items():
        target_value = float(portfolio_value) * float(weight)
        current_qty = float(strategy.ctx.get_position(symbol))

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


def order_target_positions(
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
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")

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
        for symbol, qty in strategy.ctx.positions.items():
            if float(qty) != 0.0 and symbol not in normalized_targets:
                normalized_targets[str(symbol)] = 0.0

    if not normalized_targets:
        plan["status"] = "noop"
        return []

    reduce_legs: List[Tuple[str, float, float]] = []
    increase_legs: List[Tuple[str, float, float]] = []

    for symbol, target_qty in normalized_targets.items():
        current_qty = float(strategy.ctx.get_position(symbol))
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
            - float(strategy.ctx.get_position(symbol)),
            "phase": "reduce",
        }
        for symbol, target_qty, _ in reduce_legs
    ]
    plan["increase_legs"] = [
        {
            "symbol": symbol,
            "target_quantity": float(target_qty),
            "delta_quantity": float(target_qty)
            - float(strategy.ctx.get_position(symbol)),
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
                    "in order_target_positions"
                )
                plan["status"] = "rejected"
                plan["reject_reason"] = missing_price_error
                raise RuntimeError(missing_price_error)
        leg_price = price_map.get(symbol) if price_map else None
        oid = _order_target_core(strategy, symbol, target_qty, leg_price, **kwargs)
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
                    "in order_target_positions"
                )
                plan["status"] = "rejected"
                plan["reject_reason"] = missing_price_error
                raise RuntimeError(missing_price_error)
        leg_price = price_map.get(symbol) if price_map else None
        oid = _order_target_core(strategy, symbol, target_qty, leg_price, **kwargs)
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


def buy_all(strategy: Any, symbol: Optional[str] = None) -> None:
    """全仓买入 (Buy All)."""
    _reject_target_orders_in_broker_live(strategy)
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")

    symbol = resolve_symbol(strategy, symbol)

    price = 0.0
    if strategy.current_bar and strategy.current_bar.symbol == symbol:
        price = strategy.current_bar.close
    elif strategy.current_tick and strategy.current_tick.symbol == symbol:
        price = strategy.current_tick.price

    if price <= 0:
        return

    cash = strategy.ctx.cash
    quantity = int(cash / price)

    if quantity > 0:
        buy(strategy, symbol=symbol, quantity=quantity)


def close_position(strategy: Any, symbol: Optional[str] = None) -> None:
    """平仓 (Close Position)."""
    symbol = resolve_symbol(strategy, symbol)
    position = get_position(strategy, symbol)

    if position > 0:
        sell(strategy, symbol=symbol, quantity=position)
    elif position < 0:
        buy(strategy, symbol=symbol, quantity=abs(position))


def short(
    strategy: Any,
    symbol: Optional[str] = None,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: Optional[TimeInForce] = None,
    trigger_price: Optional[float] = None,
    tag: Optional[str] = None,
    fill_policy: Optional[OrderFillPolicy] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    reduce_only: bool = False,
) -> None:
    """卖出开空 (Short Sell)."""
    submit_order_method = getattr(strategy, "submit_order", None)
    if callable(submit_order_method):
        submit_order_method(
            symbol=symbol,
            side="Sell",
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            tag=tag,
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
            position_effect="open",
            reduce_only=reduce_only,
        )
        return
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
        _submit_sell_side(
            strategy=strategy,
            symbol=symbol,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            tag=tag,
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
            position_effect="open",
            reduce_only=reduce_only,
        )


def cover(
    strategy: Any,
    symbol: Optional[str] = None,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: Optional[TimeInForce] = None,
    trigger_price: Optional[float] = None,
    tag: Optional[str] = None,
    fill_policy: Optional[OrderFillPolicy] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    reduce_only: bool = False,
) -> None:
    """买入平空 (Buy to Cover)."""
    submit_order_method = getattr(strategy, "submit_order", None)
    if callable(submit_order_method):
        submit_order_method(
            symbol=symbol,
            side="Buy",
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            tag=tag,
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
            position_effect="close",
            reduce_only=reduce_only,
        )
        return
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")

    symbol = resolve_symbol(strategy, symbol)

    if quantity is None:
        pos = strategy.ctx.get_position(symbol)
        if pos < 0:
            quantity = abs(pos)
        else:
            return

    if quantity > 0:
        _submit_buy_side(
            strategy=strategy,
            symbol=symbol,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            tag=tag,
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
            position_effect="close",
            reduce_only=reduce_only,
        )


def get_cash(strategy: Any) -> float:
    """获取现金."""
    if strategy.ctx is None:
        return 0.0
    return float(strategy.ctx.cash)


# --- SimExecution 后端原语（Task 1 引入；Task 3/4 让公共函数 delegate 到 execution）---
def _sim_get_position(strategy: Any, symbol: Optional[str] = None) -> float:
    if strategy.ctx is None:
        return 0.0
    return float(strategy.ctx.get_position(resolve_symbol(strategy, symbol)))


def _sim_get_available_position(strategy: Any, symbol: Optional[str] = None) -> float:
    if strategy.ctx is None:
        return 0.0
    return float(strategy.ctx.get_available_position(resolve_symbol(strategy, symbol)))


def _sim_get_positions(strategy: Any) -> Dict[str, float]:
    if strategy.ctx is None:
        raise RuntimeError("Context not ready")
    return cast(Dict[str, float], strategy.ctx.positions)


def _sim_hold_bar(strategy: Any, symbol: Optional[str] = None) -> int:
    if strategy.ctx is None:
        return 0
    return int(strategy._hold_bars[resolve_symbol(strategy, symbol)])


def _sim_get_cash(strategy: Any) -> float:
    if strategy.ctx is None:
        return 0.0
    return float(strategy.ctx.cash)


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


def _sim_cancel_order(strategy: Any, order_id: str) -> None:
    """取消指定订单（_sim_ 原语，复制自 cancel_order:178-202，逻辑一字不改）."""
    if strategy.ctx:
        pending_canceled_ids: Set[str] = getattr(
            strategy, "_pending_canceled_order_ids", set()
        )
        if not isinstance(pending_canceled_ids, set):
            pending_canceled_ids = set()
            setattr(strategy, "_pending_canceled_order_ids", pending_canceled_ids)
        pending_canceled_ids.add(order_id)
        strategy.ctx.cancel_order(order_id)
        for order in strategy.ctx.active_orders:
            if getattr(order, "id", "") != order_id:
                continue
            if getattr(order, "status", None) not in (
                OrderStatus.New,
                OrderStatus.Submitted,
                OrderStatus.PartiallyFilled,
            ):
                continue
            try:
                order.status = OrderStatus.Cancelled
            except Exception:
                pass
            break


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
    fill_policy: Optional[OrderFillPolicy] = None,
    slippage: Optional[Union[OrderSlippage, float, int]] = None,
    commission: Optional[OrderCommission] = None,
    position_effect: Union[PositionEffect, str, None] = None,
    reduce_only: bool = False,
    asset_type: str = "stock",
) -> str:
    """统一下单接口（_sim_ 原语，复制自 submit_order:755-845，逻辑一字不改）."""
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
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
            position_effect=_normalize_position_effect(position_effect, "auto"),
            reduce_only=reduce_only,
        )
        _record_broker_options(strategy, order_ids, broker_options)
        return _first_order_id(order_ids)
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
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
            position_effect=_normalize_position_effect(position_effect, "auto"),
            reduce_only=reduce_only,
        )
        _record_broker_options(strategy, order_ids, broker_options)
        return _first_order_id(order_ids)
    raise ValueError(f"Unsupported side: {side}")
