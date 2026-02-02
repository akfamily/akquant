import typing
import uuid
from datetime import datetime

import akquant
from akquant import (
    Order,
    OrderSide,
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
    )


def test_risk_restricted_list() -> None:
    """Test restricted list check."""
    risk = RiskManager()
    config = risk.config
    config.restricted_list = ["BANNED"]
    config.active = True
    risk.config = config

    portfolio = Portfolio(100000.0)
    order = create_dummy_order("BANNED", 100.0)

    error = risk.check(order, portfolio, {}, [])
    assert error is not None
    assert "restricted" in error


def test_risk_max_order_size() -> None:
    """Test max order size check."""
    risk = RiskManager()
    config = risk.config
    config.max_order_size = 1000.0
    config.active = True
    risk.config = config

    portfolio = Portfolio(100000.0)
    order = create_dummy_order("AAPL", 2000.0)

    error = risk.check(order, portfolio, {}, [])
    assert error is not None
    assert "quantity" in error


def test_risk_max_order_value() -> None:
    """Test max order value check."""
    risk = RiskManager()
    config = risk.config
    config.max_order_value = 50000.0
    config.active = True
    risk.config = config

    portfolio = Portfolio(100000.0)
    order = create_dummy_order("AAPL", 100.0, 600.0)  # Value = 60000

    error = risk.check(order, portfolio, {}, [])
    assert error is not None
    assert "value" in error


def test_risk_max_position_size() -> None:
    """Test max position size check."""
    risk = RiskManager()
    config = risk.config
    config.max_position_size = 500.0
    config.active = True
    risk.config = config

    class PositionStrategy(akquant.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.count = 0

        def on_bar(self, bar: typing.Any) -> None:
            if self.count == 0:
                self.buy("AAPL", 400.0)
            self.count += 1

    engine = akquant.Engine()
    engine.use_simple_market(0.0)
    engine.set_force_session_continuous(True)
    engine.set_execution_mode(akquant.ExecutionMode.CurrentClose)
    engine.set_t_plus_one(False)
    engine.set_cash(100000.0)

    instr = akquant.Instrument(
        symbol="AAPL",
        asset_type=akquant.AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        option_type=None,
        strike_price=None,
        expiry_date=None,
        lot_size=1.0,
    )
    engine.add_instrument(instr)

    bar = akquant.Bar(
        int(datetime(2023, 1, 2, 15, 0).timestamp() * 1e9),
        10.0,
        10.0,
        10.0,
        10.0,
        1000.0,
        "AAPL",
    )
    engine.add_bars([bar])

    strategy = PositionStrategy()
    engine.run(strategy, show_progress=False)

    portfolio = engine.portfolio
    order = create_dummy_order("AAPL", 200.0)
    error = risk.check(order, portfolio, {}, [])
    assert error is not None
    assert "position" in error
