"""broker_live 下策略读/撤单相关的共享 helper（经 `BrokerExecution` 调用）."""

from __future__ import annotations

from typing import Any, Callable


def _account_to_dict(acct: Any) -> dict[str, Any]:
    """Map a broker UnifiedAccount to the backtest get_account dict shape.

    Keys the broker cannot source (margin/PnL/interest 等) default to 0.0 so a
    strategy written against backtest does not KeyError in broker_live (parity).
    """
    cash = float(getattr(acct, "cash", 0.0) or 0.0)
    equity = float(getattr(acct, "equity", 0.0) or 0.0)
    available = float(getattr(acct, "available_cash", 0.0) or 0.0)
    return {
        "cash": cash,
        "available_cash": available,
        "equity": equity,
        "market_value": equity - cash,
        "notional_value": 0.0,
        "frozen_cash": 0.0,
        "margin": 0.0,
        "used_margin": 0.0,
        "free_margin": equity,
        "unrealized_pnl": 0.0,
        "borrowed_cash": 0.0,
        "short_market_value": 0.0,
        "maintenance_ratio": 0.0,
        "account_mode": "cash",
        "accrued_interest": 0.0,
        "daily_interest": 0.0,
    }


def _resolve_symbol(strategy: Any, symbol: str | None) -> str:
    """Resolve a symbol, defaulting to the current bar/tick (as backtest does)."""
    if symbol is not None:
        return str(symbol)
    bar = getattr(strategy, "current_bar", None)
    if bar is not None and getattr(bar, "symbol", None):
        return str(bar.symbol)
    tick = getattr(strategy, "current_tick", None)
    if tick is not None and getattr(tick, "symbol", None):
        return str(tick.symbol)
    raise ValueError("Symbol must be provided")


def _trade_field(payload: Any, name: str) -> Any:
    """从 trade payload 取字段(getattr 优先, dict 兜底)."""
    if payload is None:
        return None
    val = getattr(payload, name, None)
    if val is None and isinstance(payload, dict):
        val = payload.get(name)
    return val


def _signed_fill_qty(payload: Any) -> float:
    """成交带符号数量: Buy 正 / Sell 负; 数量缺失或方向无法识别 → 0.0."""
    raw = _trade_field(payload, "quantity")
    try:
        qty = float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0
    side = _trade_field(payload, "side")
    side_name = str(getattr(side, "name", side)).strip().lower()
    if side_name == "buy":
        return qty
    if side_name == "sell":
        return -qty
    return 0.0  # 方向无法识别 → 不动持仓(不臆测为卖出)


def wrap_state_invalidation(
    update_broker_state: Callable[[str, Any], None],
    get_caches: Callable[[], list[Any] | None],
) -> Callable[[str, Any], None]:
    """Wrap the broker update callback: trade 幂等叠总持仓 delta, order 失效委托/资金.

    多 slot broker_live 各 target 一缓存, 都镜像同一账户持仓; 一笔成交的 delta
    应用到 ALL 缓存(账户级)。trade 不失效总持仓(改为同步 delta), 失效可用/资金/委托;
    order 只失效委托/资金(挂撤单不改持仓)。总是先调原回调。

    **按 trade_id 去重(关键)**: 恢复循环每周期 `sync_today_trades()` 会重放当日成交,
    `apply_fill` 是加性的, 若不去重会每周期重复入账→持仓无界漂移。故用会话级
    `applied_fill_ids` 保证每个 trade_id 只叠一次。无 trade_id 时无法去重→退回幂等
    `invalidate()`(重查柜台)以防漂移。
    """
    applied_fill_ids: set[str] = set()

    def _wrapped(event_name: str, payload: Any) -> None:
        update_broker_state(event_name, payload)
        caches = get_caches() or ()
        if event_name == "trade":
            trade_id = _trade_field(payload, "trade_id")
            trade_id = str(trade_id) if trade_id else ""
            if trade_id and trade_id in applied_fill_ids:
                return  # 重复成交(恢复/重连重放): 已入账, 不再叠 delta
            symbol = _trade_field(payload, "symbol")
            signed = _signed_fill_qty(payload)
            for cache in caches:
                if cache is None:
                    continue
                if not trade_id:
                    # 无 trade_id 无法去重 → 退回幂等 invalidate(重查), 防漂移
                    cache.invalidate()
                    continue
                if symbol:
                    cache.apply_fill(str(symbol), signed)
                cache.invalidate_available()
                cache.invalidate_account()
                cache.invalidate_open_orders()
            if trade_id:
                applied_fill_ids.add(trade_id)
        elif event_name == "order":
            for cache in caches:
                if cache is None:
                    continue
                cache.invalidate_open_orders()
                cache.invalidate_account()

    return _wrapped
