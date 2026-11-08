from typing import Any, Set, Tuple

from .strategy_framework_hooks import (
    call_user_callback,
    ensure_framework_state,
    mark_portfolio_dirty,
)


def finalized_order_cache_limit(strategy: Any) -> int:
    """获取终态订单留档上限."""
    raw_limit = getattr(strategy, "finalized_order_cache_size", 10000)
    try:
        return max(1, int(raw_limit))
    except (TypeError, ValueError):
        return 10000


def check_order_events(strategy: Any) -> None:
    """检查订单和成交事件并触发回调.

    比对、去重、留档都在 Rust 侧的本槽订单账本里完成
    (``StrategyContext.order_events``), 这里只按顺序派发回调。分两段: 第一段是
    主动撤单 / 拒单 / 过期; 第一段有回调时, 回调里下的新单要在第二段(新单 /
    在途变化 / 离开在途 / 成交)里通知, 所以等第一段派发完再取第二段。

    上下文没有 ``order_events``(测试里的鸭子类型假上下文)时视为无事件,
    与旧实现对缺属性上下文的 ``hasattr`` 容忍一致。
    """
    ensure_framework_state(strategy)
    ctx = strategy.ctx
    order_events_of = getattr(ctx, "order_events", None)
    if order_events_of is None:
        return
    finalized_limit = finalized_order_cache_limit(strategy)
    trade_limit = trade_dedupe_cache_limit(strategy)
    result = order_events_of(0, finalized_limit, trade_limit)
    if result is None:
        return
    order_events, trades, done = result
    _dispatch_order_events(strategy, order_events)
    if not done:
        second = order_events_of(1, finalized_limit, trade_limit)
        if second is None:
            return
        order_events, trades, _ = second
        _dispatch_order_events(strategy, order_events)
    _dispatch_trades(strategy, trades)


def _dispatch_order_events(strategy: Any, order_events: Any) -> None:
    for order, fire_reject in order_events:
        call_user_callback(strategy, "on_order", order, payload=order)
        mark_portfolio_dirty(strategy)
        if fire_reject:
            call_user_callback(strategy, "on_reject", order, payload=order)


def _dispatch_trades(strategy: Any, trades: Any) -> None:
    for t in trades:
        strategy._framework_emit_previous_portfolio_snapshot = True
        strategy._framework_use_previous_account_snapshot = True
        try:
            call_user_callback(strategy, "on_trade", t, payload=t)
        finally:
            strategy._framework_use_previous_account_snapshot = False
        process_order_groups(strategy, t)
        analyzer_manager = getattr(strategy, "_analyzer_manager", None)
        if analyzer_manager is not None:
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
            "fee": getattr(event, "fee", None),
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


def flush_terminal_expiry_events(strategy: Any, ctx: Any) -> None:
    """收尾补结算之后派发 on_expiry(没有下一根 bar 来触发常规入口)."""
    ensure_framework_state(strategy)
    strategy.ctx = ctx
    check_expiry_events(strategy)


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


def key_value(value: Any) -> Any:
    """将复杂对象转换为可稳定哈希的值."""
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    return str(value)


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
