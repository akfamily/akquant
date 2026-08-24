"""broker_live 标的过滤: 只派发本会话挂载标的的委托/成交."""

import threading

from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.live._payload_utils import payload_field


class _Strat:
    def __init__(self) -> None:
        self.orders: list = []
        self.trades: list = []

    def on_order(self, o: object) -> None:
        self.orders.append(o)

    def on_trade(self, t: object) -> None:
        self.trades.append(t)


def _bridge(store: list, allowed: set[str] | None) -> BrokerEventBridge:
    def safe(strategy: object, name: str, payload: object) -> None:
        fn = getattr(strategy, name, None)
        if fn is not None:
            fn(payload)

    return BrokerEventBridge(
        event_lock=threading.Lock(),
        event_store=store,
        event_keys=set(),
        get_on_broker_event=lambda: None,
        make_event_key=lambda n, p: f"{n}:{id(p)}",
        update_broker_state=lambda n, p: None,
        resolve_owner_strategy_id=lambda p: "",
        payload_to_dict=lambda p: dict(p) if isinstance(p, dict) else {},
        safe_strategy_callback=safe,
        adapt_strategy_payload=lambda n, p: p,
        payload_field=payload_field,
        get_subscribed_symbols=None if allowed is None else (lambda: allowed),
    )


def _order(symbol: str, oid: str = "O1") -> dict:
    return {
        "broker_order_id": oid,
        "symbol": symbol,
        "status": "submitted",
        "filled_quantity": 0.0,
        "avg_fill_price": 0.0,
        "reject_reason": "",
    }


def test_foreign_symbol_order_dropped() -> None:
    """账户里其他标的的挂单(sync_open_orders 返回全账户)不派发给策略."""
    store: list = []
    b, s = _bridge(store, {"600008.SH"}), _Strat()

    b.queue_event("order", _order("600008.SH", "MINE"))
    b.queue_event("order", _order("000651.SZ", "FOREIGN"))
    b.drain_events(s)

    assert [o["broker_order_id"] for o in s.orders] == ["MINE"]
    assert b.dropped_event_counts()["foreign_symbol"] == 1


def test_suffix_case_mismatch_still_matches() -> None:
    """登记小写后缀、柜台推大写后缀, 本会话自己的回报**必须**放行.

    这是 2026-08-17 踩过的坑: 精确比较会把自己的单判成别人的并静默丢弃,
    表现为"下单成功却收不到回调"。
    """
    store: list = []
    b, s = _bridge(store, {"000012.SZ"}), _Strat()

    b.queue_event("order", _order("000012.sz"))
    b.drain_events(s)

    assert len(s.orders) == 1
    assert b.dropped_event_counts()["foreign_symbol"] == 0


def test_foreign_symbol_trade_dropped() -> None:
    """sync_today_trades 同样返回全账户, trade 也要过滤."""
    store: list = []
    b, s = _bridge(store, {"600008.SH"}), _Strat()

    b.queue_event("trade", {"trade_id": "T1", "symbol": "600008.SH"})
    b.queue_event("trade", {"trade_id": "T2", "symbol": "000651.SZ"})
    b.drain_events(s)

    assert [t["trade_id"] for t in s.trades] == ["T1"]


def test_empty_subscription_set_passes_everything() -> None:
    """订阅集为空: 全放行(宁可多派发, 不吞真实回报)."""
    store: list = []
    b, s = _bridge(store, set()), _Strat()

    b.queue_event("order", _order("000651.SZ"))
    b.drain_events(s)

    assert len(s.orders) == 1


def test_no_accessor_passes_everything() -> None:
    """未提供访问器(旧调用方/测试替身): 行为与改动前一致."""
    store: list = []
    b, s = _bridge(store, None), _Strat()

    b.queue_event("order", _order("000651.SZ"))
    b.drain_events(s)

    assert len(s.orders) == 1


def test_payload_without_symbol_passes() -> None:
    """Payload 没有 symbol 字段: 放行, 不猜归属."""
    store: list = []
    b, s = _bridge(store, {"600008.SH"}), _Strat()

    b.queue_event("order", {"broker_order_id": "O1", "status": "submitted"})
    b.drain_events(s)

    assert len(s.orders) == 1


def test_account_event_never_filtered() -> None:
    """Account 事件没有 symbol 概念, 永不过滤."""
    store: list = []
    b, s = _bridge(store, {"600008.SH"}), _Strat()

    b.queue_event("account", {"account_id": "A1"})
    b.drain_events(s)

    assert b.dropped_event_counts()["foreign_symbol"] == 0
