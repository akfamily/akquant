"""多任务委托回报隔离: 会话标记前缀判据(设计文档第 1/2 层) + foreign_task 计数.

覆盖 ``docs/superpowers/specs/2026-08-25-per-task-order-isolation-design.md``
"测试"一节列的 9 条。核心防线是测试 4——未传 ``session_tag`` 时严格拒绝
(第 2 层)绝不生效, 否则柜台一旦截断/改写 ``client_order_id``, 本任务自己
的回报会被误判成"别的任务的单"全部吞掉("下单成功却收不到回调")。
"""

import threading
from typing import Any, Callable

import pytest
from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.live._payload_utils import payload_field
from akquant.live._runner import LiveRunner, _validate_session_tag


class _Strat:
    def __init__(self) -> None:
        self.orders: list = []

    def on_order(self, o: object) -> None:
        self.orders.append(o)


def _safe(strategy: object, name: str, payload: object) -> None:
    fn = getattr(strategy, name, None)
    if fn is not None:
        fn(payload)


def _bridge(
    *,
    own_session_prefix: str = "",
    strict_task_isolation: bool = False,
    allowed: set[str] | None = None,
) -> BrokerEventBridge:
    get_subscribed_symbols: Callable[[], set[str]] | None = (
        (lambda: allowed) if allowed is not None else None
    )
    return BrokerEventBridge(
        event_lock=threading.Lock(),
        event_store=[],
        event_keys=set(),
        get_on_broker_event=lambda: None,
        make_event_key=lambda n, p: f"{n}:{id(p)}",
        update_broker_state=lambda n, p: None,
        resolve_owner_strategy_id=lambda p: "",
        payload_to_dict=lambda p: dict(p) if isinstance(p, dict) else {},
        safe_strategy_callback=_safe,
        adapt_strategy_payload=lambda n, p: p,
        payload_field=payload_field,
        get_subscribed_symbols=get_subscribed_symbols,
        own_session_prefix=own_session_prefix,
        strict_task_isolation=strict_task_isolation,
    )


def _order(client_order_id: str, symbol: str = "600008.SH", oid: str = "O1") -> dict:
    return {
        "broker_order_id": oid,
        "client_order_id": client_order_id,
        "symbol": symbol,
        "status": "submitted",
        "filled_quantity": 0.0,
        "avg_fill_price": 0.0,
        "reject_reason": "",
    }


def test_session_tag_shapes_client_order_id() -> None:
    """测试 1: 传 session_tag="task_42" 后, client_order_id 形如 {broker}-task_42-1."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.session_tag = "task_42"
    runner._init_broker_bridge_state()

    cid = runner._next_client_order_id()

    assert cid == "ctp-task_42-1"
    assert runner._broker_strict_task_isolation is True


def test_prefix_match_passes_even_with_unmounted_symbol() -> None:
    """测试 2: 前缀匹配本会话 -> 放行, 且标的不在挂载列表内也放行(第 1 层优先)."""
    b = _bridge(
        own_session_prefix="ctp-task_42-",
        strict_task_isolation=True,
        allowed={"600008.SH"},
    )
    s = _Strat()

    b.queue_event("order", _order("ctp-task_42-1", symbol="000651.SZ"))
    b.drain_events(s)

    assert len(s.orders) == 1
    counts = b.dropped_event_counts()
    assert counts["foreign_symbol"] == 0
    assert counts["foreign_task"] == 0


def test_mismatched_prefix_rejected_when_strict() -> None:
    """测试 3: 前缀不匹配 + 严格模式启用 -> 拒绝, 计入 foreign_task."""
    b = _bridge(
        own_session_prefix="ctp-task_42-",
        strict_task_isolation=True,
        allowed={"600008.SH"},
    )
    s = _Strat()

    b.queue_event("order", _order("ctp-task_7-1", symbol="600008.SH"))
    b.drain_events(s)

    assert s.orders == []
    counts = b.dropped_event_counts()
    assert counts["foreign_task"] == 1
    assert counts["foreign_symbol"] == 0


def test_mismatched_prefix_passes_when_not_strict() -> None:
    """测试 4(核心防线): 前缀不匹配 + 未传 session_tag -> 放行(向后兼容)."""
    b = _bridge(
        own_session_prefix="ctp-abc123-",
        strict_task_isolation=False,
        allowed={"600008.SH"},
    )
    s = _Strat()

    b.queue_event("order", _order("ctp-task_7-1", symbol="600008.SH"))
    b.drain_events(s)

    assert len(s.orders) == 1
    assert b.dropped_event_counts()["foreign_task"] == 0


def test_empty_client_order_id_falls_back_to_symbol_layer() -> None:
    """测试 5: client_order_id 为空 -> 落第 3 层, 行为与 v0.3.51 一致."""
    b = _bridge(
        own_session_prefix="ctp-task_42-",
        strict_task_isolation=True,
        allowed={"600008.SH"},
    )
    s = _Strat()

    b.queue_event("order", _order("", symbol="600008.SH"))
    b.drain_events(s)

    assert len(s.orders) == 1
    assert b.dropped_event_counts()["foreign_task"] == 0


def test_similar_prefixes_do_not_cross_match() -> None:
    """测试 6: task_4 与 task_42 互不命中(尾部 '-' 保证不误匹配)."""
    b = _bridge(
        own_session_prefix="ctp-task_4-",
        strict_task_isolation=True,
        allowed={"600008.SH"},
    )
    s = _Strat()

    b.queue_event("order", _order("ctp-task_42-1", symbol="600008.SH"))
    b.drain_events(s)

    assert s.orders == []
    assert b.dropped_event_counts()["foreign_task"] == 1


@pytest.mark.parametrize(
    "bad_tag",
    ["has-dash", "bad char!", "a" * 33, ""],
    ids=["contains-dash", "invalid-char", "too-long", "empty"],
)
def test_invalid_session_tag_raises(bad_tag: str) -> None:
    """测试 7: 含 '-'、含非法字符、超长(或空)各自 raise ValueError."""
    with pytest.raises(ValueError):
        _validate_session_tag(bad_tag)


def test_valid_session_tag_passes() -> None:
    """合法 session_tag(字母/数字/下划线, <=32 长度)不报错."""
    _validate_session_tag("task_42")
    _validate_session_tag("a" * 32)


def test_foreign_task_and_foreign_symbol_counted_separately() -> None:
    """测试 8: foreign_task 与 foreign_symbol 分别计数, 互不污染."""
    b = _bridge(
        own_session_prefix="ctp-task_42-",
        strict_task_isolation=True,
        allowed={"600008.SH"},
    )
    s = _Strat()

    b.queue_event("order", _order("ctp-task_7-1", symbol="600008.SH", oid="A"))
    b.queue_event("order", _order("", symbol="000651.SZ", oid="B"))
    b.drain_events(s)

    counts = b.dropped_event_counts()
    assert counts["foreign_task"] == 1
    assert counts["foreign_symbol"] == 1


def test_periodic_summary_not_triggered_by_foreign_task_growth(caplog: Any) -> None:
    """测试 9: 盘中周期汇总不因 foreign_task 增长而触发(防稳态刷屏回归).

    与 duplicate_order 同理: 多任务稳态下 foreign_task 必然持续增长(同账户
    同标的下必有别的任务的单), 纳入触发条件会重现"每 30s 稳态刷屏"这个刚
    修掉的失败模式(见 2026-08-25-broker-order-push-design.md 的 M4 订正段)。
    触发条件必须**只看 foreign_symbol**。
    """
    from types import SimpleNamespace

    counts = {"foreign_symbol": 0, "duplicate_order": 0, "foreign_task": 0}
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.strategy_id = "_default"
    runner._init_broker_bridge_state()
    runner._broker_event_bridge = SimpleNamespace(
        dropped_event_counts=lambda: dict(counts)
    )

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        for _ in range(5):
            counts["foreign_task"] += 3
            runner._report_dropped_event_counts_if_changed()

    assert [r for r in caplog.records if "Broker event drops" in r.getMessage()] == []


def test_no_session_tag_keeps_random_prefix_and_disables_strict_mode() -> None:
    """不传 session_tag: 仍生成前缀(跨重启唯一), 但严格模式不启用."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner._init_broker_bridge_state()

    assert runner._broker_strict_task_isolation is False
    cid = runner._next_client_order_id()
    assert cid.startswith("ctp-") and cid.endswith("-1")
