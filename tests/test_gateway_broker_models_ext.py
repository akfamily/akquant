import pytest
from akquant.gateway.broker_models import (
    BrokerCapability,
    UnifiedOrderRequest,
    normalize_asset_type,
    validate_broker_extra,
    validate_execution_semantics,
)


def test_short_sell_rejection_explains_why() -> None:
    """卖空拒单文案要让策略作者看懂, 不能只抛 supports_short_sell=False.

    原文案是给框架开发者看的内部标志名; 用户拿到它无法判断是"框架限制"还是
    "账户类型限制"。现券账户不能融券卖空是市场规则, 要说出来。
    """
    capability = BrokerCapability(
        broker_name="middleware",
        position_effect=True,
        supports_short_sell=False,
        supported_position_effects=("auto", "open", "close"),
    )
    with pytest.raises(RuntimeError) as exc:
        validate_execution_semantics(capability, "open", side="sell")
    message = str(exc.value)
    assert "middleware" in message
    assert "supports_short_sell=False" in message  # 保留可 grep 的标志名
    assert "融券" in message or "卖空" in message  # 但要有人话解释


def test_short_sell_allowed_when_capability_declares_it() -> None:
    """声明支持卖空时不拦(回归保护: 改文案别改判断)."""
    capability = BrokerCapability(
        broker_name="ctp",
        position_effect=True,
        supports_short_sell=True,
        supported_position_effects=("auto", "open", "close"),
    )
    assert validate_execution_semantics(capability, "open", side="sell") == "open"


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


def test_group_id_defaults_empty_and_settable() -> None:
    """Test that group_id defaults to empty string and is settable."""
    from akquant.gateway.broker_models import (
        UnifiedExecutionReport,
        UnifiedOrderSnapshot,
        UnifiedOrderStatus,
        UnifiedTrade,
    )

    trade = UnifiedTrade(
        trade_id="t1",
        broker_order_id="b1",
        client_order_id="c1",
        symbol="rb2410.SHFE",
        side="Sell",
        quantity=1.0,
        price=100.0,
        timestamp_ns=0,
    )
    assert trade.group_id == ""
    trade.group_id = "g1"
    assert trade.group_id == "g1"

    snap = UnifiedOrderSnapshot(
        client_order_id="c1",
        broker_order_id="b1",
        symbol="rb2410.SHFE",
        status=UnifiedOrderStatus.SUBMITTED,
    )
    assert snap.group_id == ""

    report = UnifiedExecutionReport(
        broker_order_id="b1",
        client_order_id="c1",
        status=UnifiedOrderStatus.FILLED,
        symbol="rb2410.SHFE",
    )
    assert report.group_id == ""
