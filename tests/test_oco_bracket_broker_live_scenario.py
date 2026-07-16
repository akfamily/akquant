"""broker_live OCO/Bracket 端到端: 组建→一腿成交→撤对手/激活."""

from typing import cast

from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_state_cache import BrokerStateCache
from akquant.gateway.order_receipt import OrderReceipt
from akquant.strategy import Strategy


class _Cache:
    def positions(self) -> dict:
        return {}

    def available_positions(self) -> dict:
        return {}

    def open_orders(self) -> list:
        return []

    def account(self) -> None:
        return None


class _Gw:
    def __init__(self) -> None:
        self.canceled: list = []

    def cancel_order(self, bid: str) -> None:
        self.canceled.append(bid)

    def sync_open_orders(self) -> list:
        return []


class _Sub:
    def __init__(self) -> None:
        self.n = 0

    def submit_order(self, **kw: object) -> OrderReceipt:
        self.n += 1
        # group_id(client) 与 broker_order_id 故意不同: OCO 组里登记的应是
        # broker_order_id(.primary), 避免 str(receipt)==broker_order_id 掩盖回归。
        return OrderReceipt.single(
            group_id=f"CID-{self.n}", broker_order_id=f"BID-{self.n}"
        )


class _Trade:
    def __init__(self, order_id: str, symbol: str = "X", quantity: float = 100.0):
        self.order_id, self.symbol, self.quantity = order_id, symbol, quantity


def _strategy() -> Strategy:
    s = Strategy.__new__(Strategy)
    s.execution = BrokerExecution(s, _Gw(), cast(BrokerStateCache, _Cache()), _Sub())
    s._oco_groups = {}
    s._oco_order_to_group = {}
    s._pending_brackets = {}
    s._use_engine_oco = False
    s._use_engine_bracket = False
    s.current_bar = None
    s.current_tick = None
    return s


def test_oco_peer_cancel_broker_live() -> None:
    """OCO 一腿成交→撤对手(柜台单)."""
    s = _strategy()
    # 两柜台限价单 a,b 绑 OCO
    s._oco_groups = {"g": {"BID-a", "BID-b"}}
    s._oco_order_to_group = {"BID-a": "g", "BID-b": "g"}
    s._process_order_groups(_Trade("BID-a"))  # a 成交
    assert s.execution._gw.canceled == ["BID-b"]  # 撤对手 b(柜台)


def test_bracket_activates_on_entry_fill_then_stop_leg_is_local() -> None:
    """Bracket 入场成交→止损进本地簿(LSTOP)+止盈走柜台+OCO."""
    s = _strategy()
    # 模拟 place_bracket 已登记(entry 柜台单 BID-E)
    s._pending_brackets["BID-E"] = dict(
        symbol="X",
        quantity=100.0,
        stop_trigger_price=9.0,
        take_profit_price=11.0,
        time_in_force=None,
        stop_tag=None,
        take_profit_tag=None,
    )
    s._process_order_groups(_Trade("BID-E"))  # 入场成交→激活
    # 止损腿进本地簿(LSTOP), 止盈腿走柜台
    # 读路径已统一适配为 StrategyOrder: 本地止损单以 `.id`(LSTOP-*)标识
    opens = s.execution.get_open_orders("X")
    assert any(
        str(getattr(o, "id", "")).startswith("LSTOP-") for o in opens
    )  # 止损=本地
    # 止盈腿是柜台单(submitter 被调过)
    assert s.execution._submitter.n >= 1
    # 止损(LSTOP)与止盈(BID)已绑 OCO；止盈腿登记的须是 broker_order_id(.primary)
    # ="BID-1", 而非 group_id/client_order_id="CID-1"。
    assert len(s._oco_groups) == 1
    assert "BID-1" in s._oco_order_to_group
    assert "CID-1" not in s._oco_order_to_group
