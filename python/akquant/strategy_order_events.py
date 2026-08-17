from collections import deque
from typing import Any, Set, Tuple

from .akquant import OrderStatus
from .strategy_framework_hooks import (
    call_user_callback,
    ensure_framework_state,
    mark_portfolio_dirty,
)

_TERMINAL_ORDER_STATUSES = (
    OrderStatus.Filled,
    OrderStatus.Cancelled,
    OrderStatus.Rejected,
    OrderStatus.Expired,
)

#: on_order 去重缓存上限(与成交侧 trade_dedupe_cache_size 同量级)。
#: 不做成对外可配参数: 订单事件量级远小于成交, 固定上限足够, 不必扩大 API 面。
_ORDER_EVENT_DEDUPE_LIMIT = 50000


def _is_terminal_order(order: Any) -> bool:
    """判断订单是否已处于终态."""
    return getattr(order, "status", None) in _TERMINAL_ORDER_STATUSES


def remember_finalized_order(strategy: Any, order: Any) -> None:
    """把进入终态的订单存入留档, 使 get_order() 在其离开在途账本后仍可查到.

    容量按 FIFO 淘汰(同 remember_trade_key), 避免实盘长跑内存无界增长。
    已按终态落档的订单不会被后来的非终态快照覆盖: 撤单当拍订单尚未进入
    _known_orders, 需在 ctx.active_orders 上直接定终态落档, 而下一拍它从
    active 消失时兜底路径持有的仍是撤单前的 New 快照。
    """
    order_id = str(getattr(order, "id", "") or "")
    if not order_id:
        return

    finalized = getattr(strategy, "_finalized_orders", None)
    if finalized is None:
        finalized = {}
        strategy._finalized_orders = finalized
    order_ids = getattr(strategy, "_finalized_order_ids", None)
    if order_ids is None:
        order_ids = deque()
        strategy._finalized_order_ids = order_ids

    existing = finalized.get(order_id)
    if existing is not None:
        if _is_terminal_order(existing) and not _is_terminal_order(order):
            return
    else:
        order_ids.append(order_id)
    finalized[order_id] = order

    limit = finalized_order_cache_limit(strategy)
    while len(order_ids) > limit:
        oldest = order_ids.popleft()
        finalized.pop(oldest, None)


def finalized_order_cache_limit(strategy: Any) -> int:
    """获取终态订单留档上限."""
    raw_limit = getattr(strategy, "finalized_order_cache_size", 10000)
    try:
        return max(1, int(raw_limit))
    except (TypeError, ValueError):
        return 10000


def check_order_events(strategy: Any) -> None:
    """检查订单和成交事件并触发回调."""
    ensure_framework_state(strategy)
    if strategy.ctx is None:
        return

    if hasattr(strategy.ctx, "canceled_order_ids"):
        for oid in strategy.ctx.canceled_order_ids:
            if oid in strategy._known_orders:
                order = strategy._known_orders[oid]
                try:
                    order.status = OrderStatus.Cancelled
                except Exception:
                    pass

                _emit_order_callback(strategy, order)
                remember_finalized_order(strategy, order)
                del strategy._known_orders[oid]
                continue

            # 撤单当拍订单可能还没被本函数后段从 active_orders 收录进
            # _known_orders, 此时直接在在途账本上定终态落档; 否则下一拍它从
            # active 消失时, 兜底路径只拿得到撤单前的 New 快照。
            active_order = next(
                (o for o in getattr(strategy.ctx, "active_orders", []) if o.id == oid),
                None,
            )
            if active_order is None:
                continue
            try:
                active_order.status = OrderStatus.Cancelled
            except Exception:
                pass
            remember_finalized_order(strategy, active_order)

    if hasattr(strategy.ctx, "recent_rejected_orders"):
        for order in strategy.ctx.recent_rejected_orders:
            oid = getattr(order, "id", "")
            if not oid:
                continue
            strategy._known_orders[oid] = order
            _emit_order_callback(strategy, order)

    pending_order_ids: set[str] = set()
    if hasattr(strategy.ctx, "orders"):
        for order in strategy.ctx.orders:
            oid = getattr(order, "id", "")
            if not oid or oid in strategy._known_orders:
                continue
            pending_order_ids.add(oid)
            strategy._known_orders[oid] = order
            _emit_order_callback(strategy, order)

    current_active_ids: set[str] = set()
    if hasattr(strategy.ctx, "active_orders"):
        for order in strategy.ctx.active_orders:
            current_active_ids.add(order.id)
            oid = order.id

            if oid not in strategy._known_orders:
                strategy._known_orders[oid] = order
                _emit_order_callback(strategy, order)
            else:
                known = strategy._known_orders[oid]
                status_changed = known.status != order.status
                qty_changed = known.filled_quantity != order.filled_quantity
                if status_changed or qty_changed:
                    strategy._known_orders[oid] = order
                    _emit_order_callback(strategy, order)

    recent_trade_order_ids: set[str] = set()
    if hasattr(strategy.ctx, "recent_trades"):
        for t in strategy.ctx.recent_trades:
            recent_trade_order_ids.add(t.order_id)

    for oid in list(strategy._known_orders.keys()):
        if oid not in current_active_ids:
            if oid in recent_trade_order_ids:
                order = strategy._known_orders[oid]
                try:
                    order.status = OrderStatus.Filled
                except Exception:
                    pass
                _emit_order_callback(strategy, order)
                remember_finalized_order(strategy, order)
                del strategy._known_orders[oid]
            elif oid in pending_order_ids:
                continue
            else:
                remember_finalized_order(strategy, strategy._known_orders[oid])
                del strategy._known_orders[oid]

    if hasattr(strategy.ctx, "recent_trades"):
        for t in strategy.ctx.recent_trades:
            key = trade_event_key(strategy, t)
            if not remember_trade_key(strategy, key):
                continue
            strategy._framework_emit_previous_portfolio_snapshot = True
            strategy._framework_use_previous_account_snapshot = True
            try:
                call_user_callback(strategy, "on_trade", t, payload=t)
            finally:
                strategy._framework_use_previous_account_snapshot = False
            process_order_groups(strategy, t)
            analyzer_manager = getattr(strategy, "_analyzer_manager", None)
            if analyzer_manager is not None:
                try:
                    # 已知限制: 与 on_bar_event 里 warmup 门槛挡住
                    # analyzer_manager.on_bar 不同, 这里的 on_trade 不受该门槛约束
                    # ——预热期内的成交回报仍会正常送达。按 bar 计数做分母/索引的
                    # 自定义 analyzer 在多标的+预热场景下会因此偏小/错位, 详见
                    # docs/zh/advanced/analyzer_plugin_spec.md「已知限制」。
                    analyzer_manager.on_trade(
                        {
                            "strategy": strategy,
                            "trade": t,
                            "engine": getattr(strategy, "_engine", None),
                            "ctx": strategy.ctx,
                            "owner_strategy_id": str(
                                getattr(strategy.ctx, "strategy_id", None)
                                or getattr(strategy, "_owner_strategy_id", "_default")
                            ),
                        }
                    )
                except Exception:
                    pass
            mark_portfolio_dirty(strategy)


def check_expiry_events(strategy: Any) -> None:
    """检查到期事件并触发回调."""
    ensure_framework_state(strategy)
    if strategy.ctx is None or not hasattr(strategy.ctx, "recent_expiry_events"):
        return

    for event in strategy.ctx.recent_expiry_events:
        key = expiry_event_key(strategy, event)
        seen: Set[Tuple[Any, ...]] = getattr(
            strategy, "_framework_expiry_event_keys", set()
        )
        if key in seen:
            continue
        seen.add(key)
        strategy._framework_expiry_event_keys = seen
        payload = {
            "symbol": getattr(event, "symbol", None),
            "asset_type": _enum_name(getattr(event, "asset_type", None)),
            "trading_date": getattr(event, "trading_date", None),
            "expiry_date": getattr(event, "expiry_date", None),
            "quantity_before": getattr(event, "quantity_before", None),
            "quantity_closed": getattr(event, "quantity_closed", None),
            "cash_flow": getattr(event, "cash_flow", None),
            "settlement_type": getattr(event, "settlement_type", None),
            "settlement_price": getattr(event, "settlement_price", None),
            "reason": getattr(event, "reason", None),
            "description": getattr(event, "description", None),
            "owner_strategy_id": str(
                getattr(strategy.ctx, "strategy_id", None)
                or getattr(strategy, "_owner_strategy_id", "_default")
            ),
        }
        call_user_callback(strategy, "on_expiry", payload, payload=payload)
        mark_portfolio_dirty(strategy)


def order_event_key(order: Any) -> Tuple[Any, ...]:
    """生成订单事件去重 Key(**状态指纹**).

    键 = 订单标识 + 状态 + 已成交量 + 成交均价 + 拒单原因, **刻意不含时间戳**:

    - 含时间戳(``updated_at`` / ``timestamp_ns``)的键在每次重推时都会变,
      去重会完全失效 —— 这是这类缺陷最常见的写法;
    - 只按订单号去重又会把 ``New -> PartiallyFilled -> Filled`` 这些**真实的**
      状态推进整批吞掉, 比重复推送更糟。

    :param order: 订单对象。
    :return: 可稳定哈希的状态指纹。
    """
    return (
        key_value(getattr(order, "id", None)),
        key_value(getattr(order, "status", None)),
        key_value(getattr(order, "filled_quantity", None)),
        key_value(getattr(order, "average_filled_price", None)),
        key_value(getattr(order, "reject_reason", None)),
    )


def remember_order_event_key(strategy: Any, key: Tuple[Any, ...]) -> bool:
    """记录订单事件 Key, 返回是否为首次出现(有界 FIFO, 模式同成交侧).

    :param strategy: 策略实例。
    :param key: :func:`order_event_key` 产出的状态指纹。
    :return: 首次出现返回 ``True``(应当派发回调), 重复出现返回 ``False``。
    """
    seen: Set[Tuple[Any, ...]] = strategy._framework_order_event_keys
    if key in seen:
        return False
    seen.add(key)
    key_order = strategy._framework_order_event_key_order
    key_order.append(key)
    while len(key_order) > _ORDER_EVENT_DEDUPE_LIMIT:
        seen.discard(key_order.popleft())
    return True


def _emit_order_callback(strategy: Any, order: Any) -> None:
    # check_order_events 每个 bar/tick 事件都会重扫一遍在途与终态订单, 而
    # ctx.recent_rejected_orders / orders 里的同一张单会在多拍里持续存在 ⇒
    # 不去重就会每拍重推一次同样的 on_order(表现为"全量推送"、"盘后还在推")。
    # 按状态指纹去重: 状态没变的重复推送丢弃, 状态一变立刻放行。
    if not remember_order_event_key(strategy, order_event_key(order)):
        return
    call_user_callback(strategy, "on_order", order, payload=order)
    mark_portfolio_dirty(strategy)

    if getattr(order, "status", None) == OrderStatus.Rejected:
        order_id = getattr(order, "id", "")
        if order_id and order_id not in strategy._framework_rejected_order_ids:
            strategy._framework_rejected_order_ids.add(order_id)
            call_user_callback(strategy, "on_reject", order, payload=order)


def expiry_event_key(strategy: Any, event: Any) -> Tuple[Any, ...]:
    """生成到期事件去重 Key."""
    return (
        key_value(getattr(event, "symbol", None)),
        key_value(getattr(event, "trading_date", None)),
        key_value(getattr(event, "expiry_date", None)),
        key_value(getattr(event, "quantity_closed", None)),
        key_value(getattr(event, "reason", None)),
    )


def _enum_name(value: Any) -> Any:
    """将枚举值转换为可读名称."""
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    if value is None:
        return None
    text = str(value)
    return text.split(".")[-1].upper() if "." in text else text.upper()


def trade_event_key(strategy: Any, trade: Any) -> Tuple[Any, ...]:
    """生成成交事件去重 Key."""
    return (
        key_value(getattr(trade, "trade_id", None)),
        key_value(getattr(trade, "id", None)),
        key_value(getattr(trade, "order_id", None)),
        key_value(getattr(trade, "timestamp", None)),
        key_value(getattr(trade, "symbol", None)),
        key_value(getattr(trade, "side", None)),
        key_value(getattr(trade, "quantity", None)),
        key_value(getattr(trade, "price", None)),
    )


def key_value(value: Any) -> Any:
    """将复杂对象转换为可稳定哈希的值."""
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    return str(value)


def remember_trade_key(strategy: Any, key: Tuple[Any, ...]) -> bool:
    """记录成交 Key，返回是否为首次出现."""
    if key in strategy._seen_trade_keys:
        return False

    strategy._seen_trade_keys.add(key)
    strategy._seen_trade_key_order.append(key)

    limit = trade_dedupe_cache_limit(strategy)
    while len(strategy._seen_trade_key_order) > limit:
        oldest = strategy._seen_trade_key_order.popleft()
        strategy._seen_trade_keys.discard(oldest)
    return True


def trade_dedupe_cache_limit(strategy: Any) -> int:
    """获取成交去重缓存上限."""
    raw_limit = getattr(strategy, "trade_dedupe_cache_size", 50000)
    try:
        return max(1, int(raw_limit))
    except (TypeError, ValueError):
        return 50000


def process_order_groups(strategy: Any, trade: Any) -> None:
    """处理策略内部订单组联动逻辑."""
    handler = getattr(strategy, "_process_order_groups", None)
    if callable(handler):
        handler(trade)
