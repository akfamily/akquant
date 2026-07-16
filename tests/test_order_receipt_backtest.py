"""Test backtest order-submission exits wrap ids into an OrderReceipt.

Covers _submit_buy_side/_submit_sell_side/submit_order wrapping the full
order id list into an OrderReceipt instead of collapsing it to a single id.
"""

from akquant.gateway.order_receipt import OrderReceipt
from akquant.strategy_trading_api import _orders_to_receipt


def test_first_orders_wrapped_into_receipt() -> None:
    """Test that multiple order ids are all preserved in the receipt."""
    receipt = _orders_to_receipt(["o1", "o2"], position_effect="auto")
    assert isinstance(receipt, OrderReceipt)
    assert receipt.order_ids == ("o1", "o2")
    assert receipt.group_id == "o1"
    assert len(receipt.legs) == 2
    assert receipt.legs[0].client_order_id == "o1"
    assert receipt.legs[0].broker_order_id == "o1"
    assert receipt.legs[1].client_order_id == "o2"
    assert receipt.legs[1].broker_order_id == "o2"
    assert all(leg.position_effect == "auto" for leg in receipt.legs)
    assert all(leg.quantity == 0.0 for leg in receipt.legs)


def test_empty_orders_receipt() -> None:
    """Test that an empty order id list produces an empty receipt."""
    receipt = _orders_to_receipt([], position_effect="auto")
    assert receipt.order_ids == ()
    assert receipt.group_id == ""
    assert receipt.legs == ()


def test_falsy_order_ids_are_filtered() -> None:
    """Test that falsy ids (empty string / None) are dropped before wrapping."""
    receipt = _orders_to_receipt(["o1", "", "o2"], position_effect="close")
    assert receipt.order_ids == ("o1", "o2")
    assert receipt.group_id == "o1"
    assert len(receipt.legs) == 2
