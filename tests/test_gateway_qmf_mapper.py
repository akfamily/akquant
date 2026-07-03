from akquant.gateway.broker_models import (
    UnifiedErrorType,
    UnifiedOrderRequest,
    UnifiedOrderStatus,
)
from akquant.gateway.brokers.qmf import mapper


def test_split_symbol() -> None:
    """split_symbol maps akquant symbols to (exchange_type, stock_code)."""
    assert mapper.split_symbol("600000.SH") == ("1", "600000")
    assert mapper.split_symbol("000001.SZ") == ("2", "000001")


def test_split_symbol_invalid() -> None:
    """split_symbol rejects symbols without an exchange suffix."""
    import pytest

    with pytest.raises(ValueError):
        mapper.split_symbol("600000")


def test_join_symbol_roundtrip() -> None:
    """join_symbol reconstructs the akquant symbol from counter fields."""
    assert mapper.join_symbol("1", "600000") == "600000.SH"
    assert mapper.join_symbol("2", "000001") == "000001.SZ"


def test_build_order_payload_limit_buy() -> None:
    """A limit buy maps to the counter OrderRequest fields (no fund_account)."""
    req = UnifiedOrderRequest(
        client_order_id="c1",
        symbol="600000.SH",
        side="Buy",
        quantity=100,
        price=10.5,
        order_type="Limit",
    )
    payload = mapper.build_order_payload(req)
    assert payload == {
        "exchange_type": "1",
        "stock_code": "600000",
        "entrust_bs": "1",
        "entrust_prop": "0",
        "entrust_price": "10.5",
        "entrust_amount": "100",
    }
    assert "fund_account" not in payload


def test_build_order_payload_market_unsupported() -> None:
    """Phase 1 only supports Limit orders; Market raises ValueError."""
    import pytest

    req = UnifiedOrderRequest(
        client_order_id="c1",
        symbol="600000.SH",
        side="Buy",
        quantity=100,
        price=None,
        order_type="Market",
    )
    with pytest.raises(ValueError):
        mapper.build_order_payload(req)


def test_map_order_status() -> None:
    """Counter entrust_status/error_no map to UnifiedOrderStatus."""
    assert mapper.map_order_status("1") == UnifiedOrderStatus.SUBMITTED
    assert mapper.map_order_status("2") == UnifiedOrderStatus.PARTIALLY_FILLED
    assert mapper.map_order_status("8") == UnifiedOrderStatus.FILLED
    assert mapper.map_order_status("6") == UnifiedOrderStatus.CANCELLED
    assert mapper.map_order_status("1", error_no="11") == UnifiedOrderStatus.REJECTED


def test_classify_error() -> None:
    """error_no/error_info classify into the UnifiedErrorType buckets."""
    assert mapper.classify_error("0", "") == UnifiedErrorType.RETRYABLE
    assert mapper.classify_error("1", "触发风控限制") == UnifiedErrorType.RISK_REJECTED
    assert mapper.classify_error("1", "连接超时") == UnifiedErrorType.RETRYABLE
    assert mapper.classify_error("1", "资金不足") == UnifiedErrorType.NON_RETRYABLE
