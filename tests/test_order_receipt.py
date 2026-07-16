"""Test OrderReceipt and OrderLeg value objects."""

from akquant.gateway.order_receipt import OrderLeg, OrderReceipt


def test_receipt_str_is_group_id() -> None:
    """Test that __str__ returns group_id."""
    r = OrderReceipt(group_id="g1", order_ids=("b1", "b2"), legs=())
    assert str(r) == "g1"


def test_receipt_iter_and_len_and_primary() -> None:
    """Test __iter__, __len__, and primary property."""
    r = OrderReceipt(group_id="g1", order_ids=("b1", "b2"), legs=())
    assert list(r) == ["b1", "b2"]
    assert len(r) == 2
    assert r.primary == "b1"


def test_receipt_primary_empty_when_no_ids() -> None:
    """Test primary returns empty string when no ids."""
    r = OrderReceipt(group_id="g1", order_ids=(), legs=())
    assert r.primary == ""


def test_single_builds_one_leg() -> None:
    """Test single() class method builds one-leg receipt."""
    r = OrderReceipt.single(
        group_id="c1",
        broker_order_id="b1",
        position_effect="open",
        quantity=3.0,
        client_order_id="c1",
    )
    assert r.order_ids == ("b1",)
    assert r.legs == (
        OrderLeg(
            position_effect="open",
            quantity=3.0,
            client_order_id="c1",
            broker_order_id="b1",
        ),
    )
    assert r.primary == "b1"


def test_exported_from_package() -> None:
    """Test classes are exported from akquant package."""
    import akquant

    assert akquant.OrderReceipt is OrderReceipt
    assert akquant.OrderLeg is OrderLeg
