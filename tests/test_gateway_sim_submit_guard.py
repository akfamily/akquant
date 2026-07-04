import pytest
from akquant.execution.sim import SimExecution
from akquant.strategy_trading_api import submit_order


class _SimStrategy:
    """Strategy stub whose execution mode reports simulated (no broker_live)."""

    def __init__(self) -> None:
        """Inject simulated-mode capabilities on the instance dict.

        ``get_execution_capabilities`` only honors an override found in the
        instance ``__dict__`` (see strategy_trading_api), so a class method
        would be ignored; inject it here to truly exercise simulated mode.
        """
        self.__dict__["get_execution_capabilities"] = lambda: {
            "client_order_id": False,
            "broker_live": False,
        }
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


def test_strategy_submit_order_forwards_asset_type(monkeypatch) -> None:
    """Strategy.submit_order exposes and forwards asset_type to the impl."""
    from akquant import strategy as strat_mod
    from akquant.strategy import Strategy

    captured: dict = {}

    def _fake_impl(_self, **kwargs):
        captured.update(kwargs)
        return "order-id"

    monkeypatch.setattr(strat_mod, "_submit_order_impl", _fake_impl)
    result = Strategy.submit_order(
        object(), symbol="10004321.SH", side="Buy", quantity=1, asset_type="option"
    )
    assert result == "order-id"
    assert captured["asset_type"] == "option"
