"""broker_live 下把策略读/撤单方法 setattr 覆盖到柜台版（独立可测）."""

from __future__ import annotations

from typing import Any

from .broker_state_cache import BrokerStateCache


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


def install_broker_state_reads(strategy: Any, cache: BrokerStateCache) -> None:
    """Override the strategy's state-read methods to consult the broker cache."""

    def _get_position(symbol: str | None = None) -> float:
        return cache.positions().get(_resolve_symbol(strategy, symbol), 0.0)

    def _get_available_position(symbol: str | None = None) -> float:
        return cache.available_positions().get(_resolve_symbol(strategy, symbol), 0.0)

    def _get_account() -> dict[str, Any]:
        return _account_to_dict(cache.account())

    def _get_portfolio_value() -> float:
        return float(getattr(cache.account(), "equity", 0.0) or 0.0)

    def _get_open_orders(symbol: str | None = None) -> list[Any]:
        orders = cache.open_orders()
        if symbol is not None:
            return [o for o in orders if getattr(o, "symbol", None) == symbol]
        return list(orders)

    strategy.get_position = _get_position
    strategy.get_available_position = _get_available_position
    strategy.get_account = _get_account
    strategy.get_portfolio_value = _get_portfolio_value
    strategy.get_open_orders = _get_open_orders
