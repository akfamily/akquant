"""broker_live 标的过滤: 只派发本会话挂载标的的委托/成交."""

import threading
from types import SimpleNamespace
from typing import Any, Callable, cast

from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.live._payload_utils import payload_field
from akquant.live._runner import LiveRunner


class _Strat:
    def __init__(self) -> None:
        self.orders: list = []
        self.trades: list = []
        self.reports: list = []

    def on_order(self, o: object) -> None:
        self.orders.append(o)

    def on_trade(self, t: object) -> None:
        self.trades.append(t)

    def on_execution_report(self, r: object) -> None:
        self.reports.append(r)


def _safe(strategy: object, name: str, payload: object) -> None:
    fn = getattr(strategy, name, None)
    if fn is not None:
        fn(payload)


def _const(value: set[str]) -> Callable[[], set[str]]:
    """Return a zero-arg accessor that always yields ``value``."""
    return lambda: value


def _bridge(
    store: list,
    allowed: set[str] | None,
    *,
    is_known_order: Callable[[str, str], bool] | None = None,
    get_subscribed_symbols: Callable[[], set[str]] | None = None,
) -> BrokerEventBridge:
    resolved_get_subscribed_symbols = get_subscribed_symbols
    if resolved_get_subscribed_symbols is None and allowed is not None:
        resolved_get_subscribed_symbols = _const(allowed)

    return BrokerEventBridge(
        event_lock=threading.Lock(),
        event_store=store,
        event_keys=set(),
        get_on_broker_event=lambda: None,
        make_event_key=lambda n, p: f"{n}:{id(p)}",
        update_broker_state=lambda n, p: None,
        resolve_owner_strategy_id=lambda p: "",
        payload_to_dict=lambda p: dict(p) if isinstance(p, dict) else {},
        safe_strategy_callback=_safe,
        adapt_strategy_payload=lambda n, p: p,
        payload_field=payload_field,
        get_subscribed_symbols=resolved_get_subscribed_symbols,
        is_known_order=is_known_order,
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


def test_dropped_foreign_symbol_names_lists_the_blocked_symbols() -> None:
    """被挡掉的标的名可读出来: 盘中汇总据此判断"是否出现了新的外来标的".

    计数本身在稳态下必然线性增长(每轮全量 sync 对同一笔外来挂单重复 +1),
    唯一有诊断价值的是**被挡的标的是谁** —— 若点名的标的是本任务挂载的,
    才是配置或匹配问题。见 ``LiveRunner._report_dropped_event_counts_if_changed``。
    """
    store: list = []
    b, s = _bridge(store, {"600008.SH"}), _Strat()

    b.queue_event("order", _order("600008.SH", "MINE"))
    b.queue_event("order", _order("000651.SZ", "FOREIGN"))
    b.queue_event("order", _order("600519.SH", "FOREIGN2"))
    b.drain_events(s)

    assert b.dropped_foreign_symbol_names() == {"000651.SZ", "600519.SH"}


def test_dropped_foreign_symbol_names_returns_a_snapshot() -> None:
    """返回的是快照: 调用方改它不会污染 bridge 的内部集合."""
    store: list = []
    b = _bridge(store, {"600008.SH"})

    b.queue_event("order", _order("000651.SZ", "FOREIGN"))
    snapshot = b.dropped_foreign_symbol_names()
    snapshot.add("999999.SH")

    assert b.dropped_foreign_symbol_names() == {"000651.SZ"}


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


def test_execution_report_foreign_symbol_dropped() -> None:
    """execution_report 与 order/trade 同样按标的过滤."""
    store: list = []
    b, s = _bridge(store, {"600008.SH"}), _Strat()

    b.queue_event(
        "execution_report",
        {"broker_order_id": "R1", "symbol": "600008.SH", "status": "submitted"},
    )
    b.queue_event(
        "execution_report",
        {"broker_order_id": "R2", "symbol": "000651.SZ", "status": "submitted"},
    )
    b.drain_events(s)

    assert [r["broker_order_id"] for r in s.reports] == ["R1"]
    assert b.dropped_event_counts()["foreign_symbol"] == 1


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
    """Account 事件没有 symbol 概念, 永不过滤.

    payload 刻意带一个外来 symbol: 否则该用例是恒真的
    (``{"account_id": ...}`` 本身没有 symbol 字段, 落到"无 symbol 放行"分支,
    删掉 ``event_name == "account"`` 短路依然会通过)。
    """
    store: list = []
    b, s = _bridge(store, {"600008.SH"}), _Strat()

    b.queue_event("account", {"account_id": "A1", "symbol": "000651.SZ"})
    b.drain_events(s)

    assert b.dropped_event_counts()["foreign_symbol"] == 0


def test_known_order_passes_even_with_foreign_symbol() -> None:
    """已知本会话订单(broker_order_id 命中映射)一律放行, 不论标的是否挂载.

    覆盖 ``BrokerOrderSink`` 经外部信号源直调 ``submitter.submit_order``、
    不经引擎合约登记表就能合法报出挂载集合之外标的的场景。
    """
    store: list = []
    b = _bridge(store, {"600008.SH"}, is_known_order=lambda bid, cid: bid == "KNOWN")
    s = _Strat()

    b.queue_event("order", _order("000651.SZ", "KNOWN"))
    b.drain_events(s)

    assert [o["broker_order_id"] for o in s.orders] == ["KNOWN"]
    assert b.dropped_event_counts()["foreign_symbol"] == 0


def test_unknown_order_with_foreign_symbol_still_dropped() -> None:
    """未命中已知订单映射的外来标的委托仍被过滤(标的判据兜底)."""
    store: list = []
    b = _bridge(store, {"600008.SH"}, is_known_order=lambda bid, cid: False)
    s = _Strat()

    b.queue_event("order", _order("000651.SZ", "UNKNOWN"))
    b.drain_events(s)

    assert s.orders == []
    assert b.dropped_event_counts()["foreign_symbol"] == 1


def test_get_subscribed_symbols_exception_allows_through(caplog: Any) -> None:
    """``get_subscribed_symbols`` 抛异常时放行并降噪留痕, 不吞真实回报.

    不兜底的话异常会顺着 ``queue_event`` 一路炸到 ``broker_recovery`` 的
    ``sync_open_orders``/``sync_today_trades`` 循环, 被外层宽
    ``except Exception`` 吞掉整批剩余委托; 默认 ``recovery_mode="compatible"``
    下连日志都没有——后果比放行一条外来事件严重得多。
    """

    def boom() -> set[str]:
        raise RuntimeError("accessor exploded")

    store: list = []
    b = _bridge(store, None, get_subscribed_symbols=boom)
    s = _Strat()

    with caplog.at_level("WARNING", logger="akquant.gateway.live"):
        b.queue_event("order", _order("000651.SZ"))
        b.drain_events(s)

    assert len(s.orders) == 1
    assert b.dropped_event_counts()["foreign_symbol"] == 0
    assert any(
        "Symbol filter accessor raised" in record.getMessage()
        for record in caplog.records
    )


def test_is_known_order_exception_allows_through() -> None:
    """``is_known_order`` 抛异常时同样放行, 不因归属判据出错吞事件."""

    def boom(_bid: str, _cid: str) -> bool:
        raise RuntimeError("ownership lookup exploded")

    store: list = []
    b = _bridge(store, {"600008.SH"}, is_known_order=boom)
    s = _Strat()

    b.queue_event("order", _order("000651.SZ"))
    b.drain_events(s)

    assert len(s.orders) == 1
    assert b.dropped_event_counts()["foreign_symbol"] == 0


def test_runner_subscribed_symbol_set_normalizes_and_caches() -> None:
    """``_subscribed_symbol_set``: 从 instruments 派生归一化集合并只建一次."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner.instruments = cast(
        Any,
        [
            SimpleNamespace(symbol="000012.sz"),
            SimpleNamespace(symbol=""),
        ],
    )
    runner._init_broker_bridge_state()

    first = runner._subscribed_symbol_set()
    assert first == {"000012.SZ"}

    # 缓存语义: 事后追加 instruments 不影响已缓存结果(惰性构建只发生一次)。
    runner.instruments.append(cast(Any, SimpleNamespace(symbol="600000.SH")))
    assert runner._subscribed_symbol_set() is first
    assert runner._subscribed_symbol_set() == {"000012.SZ"}


def test_dropped_counts_are_reported_together() -> None:
    """两类丢弃分开计数, 供收尾摘要一次读出."""
    store: list = []
    b, s = _bridge(store, {"600008.SH"}), _Strat()

    b.queue_event("order", _order("000651.SZ", "FOREIGN"))
    b.queue_event("order", _order("600008.SH", "MINE"))
    b.drain_events(s)
    b.queue_event("order", _order("600008.SH", "MINE"))  # 同状态重放

    counts = b.dropped_event_counts()
    assert counts["foreign_symbol"] == 1
    assert counts["duplicate_order"] == 1


def test_runner_dispatches_own_subscribed_symbol_order() -> None:
    """8/17 事故防回归: 小写登记 -> 归一化大写集合, 柜台推大写回报仍派发到策略."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner.instruments = cast(Any, [SimpleNamespace(symbol="000012.sz")])
    runner._init_broker_bridge_state()
    strategy = _Strat()

    runner._queue_broker_event("order", _order("000012.SZ"))
    runner._drain_broker_events(cast(Any, strategy))

    assert len(strategy.orders) == 1
