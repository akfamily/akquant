from typing import Any

import pytest
from akquant.execution.sim import SimExecution
from akquant.strategy_trading_api import submit_order


class _SimStrategy:
    """Strategy stub whose execution mode reports simulated (no broker_live)."""

    def __init__(self) -> None:
        """Bind SimExecution, whose capabilities() always report simulated mode."""
        self.ctx = None
        self.current_bar: Any | None = None
        self.current_tick: Any | None = None
        self.execution = SimExecution(self)


def test_sim_rejects_extra_with_clear_message() -> None:
    """Raise a clear broker_live-required error for extra in simulated mode."""
    with pytest.raises(RuntimeError) as exc:
        submit_order(
            _SimStrategy(),
            symbol="600000.SH",
            side="Buy",
            quantity=100,
            price=10.5,
            order_type="Limit",
            extra={"entrust_oc": "O"},
        )
    assert "broker_live" in str(exc.value)


def test_sim_rejects_non_stock_asset_type() -> None:
    """Reject non-stock asset_type in simulated mode."""
    with pytest.raises(RuntimeError):
        submit_order(
            _SimStrategy(),
            symbol="600000.SH",
            side="Buy",
            quantity=100,
            price=10.5,
            order_type="Limit",
            asset_type="option",
        )


def test_strategy_submit_order_forwards_asset_type(monkeypatch: Any) -> None:
    """Strategy.submit_order exposes and forwards asset_type to the impl."""
    from akquant import strategy as strat_mod
    from akquant.strategy import Strategy

    captured: dict[str, Any] = {}

    def _fake_impl(_self: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "order-id"

    monkeypatch.setattr(strat_mod, "_submit_order_impl", _fake_impl)
    result = Strategy.submit_order(
        Strategy.__new__(Strategy),
        symbol="10004321.SH",
        side="Buy",
        quantity=1,
        asset_type="option",
    )
    assert result == "order-id"
    assert captured["asset_type"] == "option"
