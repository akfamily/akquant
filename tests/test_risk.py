import typing
import uuid

from akquant import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    RiskManager,
    TimeInForce,
)


def create_dummy_order(
    symbol: str, quantity: float, price: typing.Optional[float] = None
) -> Order:
    """Create a dummy order for testing."""
    return Order(
        id=str(uuid.uuid4()),
        symbol=symbol,
        side=OrderSide.Buy,
        order_type=OrderType.Limit,
        quantity=quantity,
        price=price,
        time_in_force=TimeInForce.Day,
        trigger_price=None,
        status=OrderStatus.New,
        filled_quantity=0.0,
        average_filled_price=None,
    )


def test_risk_restricted_list() -> None:
    """Test restricted list check."""
    risk = RiskManager()
    risk.config.restricted_list = ["BANNED"]
    risk.config.active = True

    portfolio = Portfolio(100000.0)
    order = create_dummy_order("BANNED", 100.0)

    error = risk.check(order, portfolio)
    assert error is not None
    assert "restricted" in error


def test_risk_max_order_size() -> None:
    """Test max order size check."""
    risk = RiskManager()
    risk.config.max_order_size = 1000.0
    risk.config.active = True

    portfolio = Portfolio(100000.0)
    order = create_dummy_order("AAPL", 2000.0)

    error = risk.check(order, portfolio)
    assert error is not None
    assert "quantity" in error


def test_risk_max_order_value() -> None:
    """Test max order value check."""
    risk = RiskManager()
    risk.config.max_order_value = 50000.0
    risk.config.active = True

    portfolio = Portfolio(100000.0)
    order = create_dummy_order("AAPL", 100.0, 600.0)  # Value = 60000

    error = risk.check(order, portfolio)
    assert error is not None
    assert "value" in error


def test_risk_max_position_size() -> None:
    """Test max position size check."""
    risk = RiskManager()
    risk.config.max_position_size = 500.0
    risk.config.active = True

    portfolio = Portfolio(100000.0)
    portfolio.positions["AAPL"] = 400.0

    order = create_dummy_order("AAPL", 200.0)  # Resulting pos = 600

    error = risk.check(order, portfolio)
    assert error is not None
    assert "position" in error
