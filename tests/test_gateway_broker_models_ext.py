import pytest
from akquant.gateway.broker_models import (
    BrokerCapability,
    UnifiedOrderRequest,
    normalize_asset_type,
    validate_broker_extra,
)


def test_order_request_defaults() -> None:
    """New order requests default to stock asset_type and empty extra."""
    req = UnifiedOrderRequest(
        client_order_id="c1", symbol="600000.SH", side="Buy", quantity=100
    )
    assert req.asset_type == "stock"
    assert req.extra == {}


def test_normalize_asset_type_aliases_and_passthrough() -> None:
    """Aliases map to canonical names; unknown values pass through lowercased."""
    assert normalize_asset_type("OPT") == "option"
    assert normalize_asset_type("stk") == "stock"
    assert normalize_asset_type("Future") == "future"
    assert normalize_asset_type("etf") == "fund"
    assert normalize_asset_type("  perp ") == "perp"  # 规范集外，放行
    assert normalize_asset_type(None) == "stock"


def test_validate_broker_extra() -> None:
    """Extra keys must be declared in capability.broker_extra_fields."""
    cap = BrokerCapability(
        broker_name="x", broker_extra_fields=("entrust_oc", "covered_flag")
    )
    validate_broker_extra(cap, {"entrust_oc": "O"})  # ok
    validate_broker_extra(cap, {})  # empty ok
    validate_broker_extra(cap, None)  # None ok
    with pytest.raises(RuntimeError):
        validate_broker_extra(cap, {"nope": "1"})


def test_capability_features_roundtrip() -> None:
    """Features default empty; surfaced in the dict; restored by from_value."""
    cap = BrokerCapability(broker_name="x", features=frozenset({"iceberg", "oco"}))
    d = cap.as_execution_capabilities()
    assert d["features"] == ["iceberg", "oco"]
    restored = BrokerCapability.from_value(
        {"broker_name": "x", "features": ["oco", "iceberg"]}
    )
    assert restored.features == frozenset({"iceberg", "oco"})
    assert BrokerCapability(broker_name="y").features == frozenset()
