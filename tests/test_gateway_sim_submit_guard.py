import pytest
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
