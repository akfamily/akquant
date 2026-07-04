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


def wrap_state_invalidation(
    update_broker_state: Callable[[str, Any], None],
    get_caches: Callable[[], list[Any] | None],
) -> Callable[[str, Any], None]:
    """Wrap the broker update callback so order/trade events invalidate caches.

    In multi-slot broker_live each strategy target owns its own cache; a fill/
    order push must invalidate ALL of them, not just the last installed one.
    Always calls the original callback first; `get_caches()` may be None/empty.
    """

    def _wrapped(event_name: str, payload: Any) -> None:
        update_broker_state(event_name, payload)
        if event_name in ("order", "trade"):
            for cache in get_caches() or ():
                if cache is not None:
                    cache.invalidate()

    return _wrapped
