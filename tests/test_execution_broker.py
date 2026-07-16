"""BrokerExecution：读走 cache、submit 走 submitter、cancel 走 gateway."""

from typing import Any

from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_models import UnifiedAccount, UnifiedPosition
from akquant.gateway.broker_state_cache import BrokerStateCache
from akquant.gateway.order_receipt import OrderLeg, OrderReceipt


class _Gw:
    def __init__(self) -> None:
        self.canceled: list[str] = []

    def query_positions(self) -> list[UnifiedPosition]:
        return [
            UnifiedPosition(symbol="600000.SH", quantity=1000, available_quantity=800)
        ]

    def query_account(self) -> UnifiedAccount:
        return UnifiedAccount(
            account_id="a", equity=1500.0, cash=1000.0, available_cash=850.0
        )

    def sync_open_orders(self) -> list[object]:
        return []

    def cancel_order(self, bid: str) -> None:
        self.canceled.append(bid)


class _Submitter:
    def __init__(self) -> None:
        self.submitted: dict[str, Any] | None = None

    def submit_order(self, **kwargs: Any) -> OrderReceipt:
        self.submitted = kwargs
        # group_id(client) 与 broker_order_id 故意不同: submit_order() 断言锁定
        # broker_order_id(.primary), 避免 str(receipt)==broker_order_id 掩盖回归。
        return OrderReceipt.single(group_id="CID-1", broker_order_id="BID-1")

    def _get_execution_capabilities(self) -> dict[str, bool]:
        return {"broker_live": True, "client_order_id": True}


class _S:
    current_bar = None
    current_tick = None


def test_broker_execution_reads_and_writes() -> None:
    """读走 cache、submit 走 submitter、cancel 走 gateway."""
    gw = _Gw()
    ex = BrokerExecution(_S(), gw, BrokerStateCache(gw), _Submitter())
    assert ex.get_position("600000.SH") == 1000.0
    assert ex.get_available_position("600000.SH") == 800.0
    assert ex.get_account()["cash"] == 1000.0
    assert ex.get_cash() == 1000.0
    assert ex.get_portfolio_value() == 1500.0
    assert ex.capabilities()["broker_live"] is True
    receipt = ex.submit_order(symbol="600000.SH", side="Buy", quantity=100)
    assert isinstance(receipt, OrderReceipt)
    assert receipt.primary == "BID-1"
    ex.cancel_order("BID-1")
    assert gw.canceled == ["BID-1"]


def test_broker_execution_submit_order_drops_none_time_in_force() -> None:
    """time_in_force=None 不应覆盖 submitter 的 "GTC" 默认.

    回归：Strategy.submit_order 默认转发 time_in_force=None；若原样透传给
    submitter.submit_order(**kwargs)，会覆盖其签名默认值 "GTC"，破坏重构前
    broker_live 未指定 TIF 时默认为 "GTC" 的行为。
    """
    gw = _Gw()
    submitter = _Submitter()
    ex = BrokerExecution(_S(), gw, BrokerStateCache(gw), submitter)
    ex.submit_order(symbol="600000.SH", side="Buy", quantity=100, time_in_force=None)
    assert submitter.submitted is not None
    assert "time_in_force" not in submitter.submitted


def test_broker_execution_submit_order_forwards_explicit_time_in_force() -> None:
    """显式指定的 time_in_force 应原样透传给 submitter（不被丢弃）."""
    gw = _Gw()
    submitter = _Submitter()
    ex = BrokerExecution(_S(), gw, BrokerStateCache(gw), submitter)
    ex.submit_order(symbol="600000.SH", side="Buy", quantity=100, time_in_force="IOC")
    assert submitter.submitted is not None
    assert submitter.submitted["time_in_force"] == "IOC"


class _MultiLegSubmitter:
    """反手拆腿场景的下单器桩：一次逻辑委托拆成 close+open 两腿."""

    def submit_order(self, **kwargs: Any) -> OrderReceipt:
        legs = (
            OrderLeg(
                position_effect="close",
                quantity=50.0,
                client_order_id="CID-1-close",
                broker_order_id="BID-1-close",
            ),
            OrderLeg(
                position_effect="open",
                quantity=50.0,
                client_order_id="CID-1-open",
                broker_order_id="BID-1-open",
            ),
        )
        return OrderReceipt(
            group_id="CID-1",
            order_ids=("BID-1-close", "BID-1-open"),
            legs=legs,
        )

    def _get_execution_capabilities(self) -> dict[str, bool]:
        return {"broker_live": True, "client_order_id": True}


def test_broker_execution_submit_order_returns_full_receipt_for_multi_leg() -> None:
    """实盘 BrokerExecution.submit_order 不得把多腿 OrderReceipt 收窄为单一 id 字符串.

    回归：#317 的诉求是 buy/sell/submit_order "返回全部 id"；此前实盘路径
    做了 `str(...primary)`，丢弃了反手/开平拆腿产生的其余腿 id。
    """
    gw = _Gw()
    ex = BrokerExecution(_S(), gw, BrokerStateCache(gw), _MultiLegSubmitter())
    result = ex.submit_order(symbol="600000.SH", side="Buy", quantity=100)
    assert isinstance(result, OrderReceipt)
    assert len(result.order_ids) > 1
    assert result.order_ids == ("BID-1-close", "BID-1-open")
    assert result.primary == "BID-1-close"
    assert str(result) == "CID-1"


def test_broker_execution_submit_order_local_stop_returns_receipt() -> None:
    """本地止损分支（不经 submitter）也应返回 OrderReceipt，而非裸 str 本地 id."""
    gw = _Gw()
    ex = BrokerExecution(_S(), gw, BrokerStateCache(gw), _Submitter())
    result = ex.submit_order(
        symbol="600000.SH", side="Sell", quantity=100, trigger_price=9.5
    )
    assert isinstance(result, OrderReceipt)
    assert result.primary == "LSTOP-1"
    assert str(result) == "LSTOP-1"
