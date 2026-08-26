"""多任务委托回报隔离: 会话标记前缀判据(设计文档第 1/2 层) + foreign_task 计数.

覆盖 ``docs/superpowers/specs/2026-08-25-per-task-order-isolation-design.md``
"测试"一节列的 9 条。核心防线是测试 4——未传 ``session_tag`` 时严格拒绝
(第 2 层)绝不生效, 否则柜台一旦截断/改写 ``client_order_id``, 本任务自己
的回报会被误判成"别的任务的单"全部吞掉("下单成功却收不到回调")。
"""

import re
import threading
from typing import Any, Callable

import akquant.live._runner as live_module
import pytest
from akquant import run_live
from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.live._payload_utils import payload_field
from akquant.live._runner import LiveRunner, _validate_session_tag


class _DummyDataFeed:
    """真实 ``LiveRunner.__init__`` 需要的 DataFeed/Engine 替身.

    I-2 三处接线测试要从真实构造出发(而非 ``LiveRunner.__new__`` 绕开
    ``__init__``), 但不需要真的连行情/引擎——只替掉这两个依赖, 让
    ``__init__`` 能跑到底、把 ``_init_broker_bridge_state()`` 真正执行一遍。
    """

    @staticmethod
    def create_live() -> object:
        return object()


class _DummyEngine:
    pass


def _patch_live_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_module, "DataFeed", _DummyDataFeed)
    monkeypatch.setattr(live_module, "Engine", _DummyEngine)


#: 生成端格式是 ``{broker}-{tag}-{run_salt}{seq}``, salt 固定 4 位十六进制。
_SALTED_SEQ_RE = re.compile(r"^[0-9a-f]{4}[0-9]+$")


class _Strat:
    def __init__(self) -> None:
        self.orders: list = []
        self.portfolio_updates: list = []

    def on_order(self, o: object) -> None:
        self.orders.append(o)

    def on_portfolio_update(self, o: object) -> None:
        self.portfolio_updates.append(o)


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
    """测试 1: 传 session_tag="task_42" 后, client_order_id 前缀为 ctp-task_42-.

    序号段(``{run_salt}{seq}``)每次 run_live 独立带 4 位随机盐(见 C-1 修复:
    固定 session_tag 场景下若序号段不带盐, 重启后第一笔单会与柜台里同名的
    历史委托撞号), 前缀段本身不受影响。
    """
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.session_tag = "task_42"
    runner._init_broker_bridge_state()

    cid = runner._next_client_order_id()

    assert cid.startswith("ctp-task_42-")
    suffix = cid[len("ctp-task_42-") :]
    assert _SALTED_SEQ_RE.match(suffix), suffix
    assert suffix.endswith("1")
    assert runner._broker_strict_task_isolation is True


def test_run_salt_avoids_collision_with_leftover_active_mapping() -> None:
    """C-1 回退验证配套: 固定 session_tag 重启后, 若序号段不加运行盐会撞号.

    review 指出的设计漏洞: ``_broker_submit_seq`` 每次 run_live 从 0 起,
    固定 session_tag 场景下重启后第一笔单又是 ``...-1``。若不带运行盐,
    这个新 client_order_id 会与"重启后 sync_open_orders() 把上一轮仍活跃
    的挂单重放进 _client_to_broker_order_ids"这个动作留下的旧 id 完全相同
    ——can_submit_client_order 判定为重复且非终态, 本地直接拒绝提交,
    下单根本到不了柜台。带盐后新旧 id 不同, 不会被这条本地去重挡住。
    """
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.session_tag = "task_42"
    runner._init_broker_bridge_state()

    # 模拟重启后 sync_open_orders() 把上一轮仍活跃的旧单重放进映射: 若不带
    # 运行盐, 新会话首笔单会生成与它完全相同的 client_order_id。
    stale_cid = "ctp-task_42-1"
    runner._client_to_broker_order_ids[stale_cid] = "BO-STALE"
    runner._broker_order_states["BO-STALE"] = {"status": "submitted"}

    new_cid = runner._next_client_order_id()

    assert new_cid != stale_cid
    assert runner.can_submit_client_order(new_cid) is True


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
    "bad_tag,expected_match",
    [
        ("has-dash", "不能包含"),
        ("bad char!", "只能包含字母"),
        ("a" * 33, "超过上限"),
        ("", "不能是空字符串"),
    ],
    ids=["contains-dash", "invalid-char", "too-long", "empty"],
)
def test_invalid_session_tag_raises(bad_tag: str, expected_match: str) -> None:
    """测试 7: 含 '-'、含非法字符、超长(或空)各自 raise ValueError, 各命中专属分支.

    ``match=`` 断言消息里出现该分支特有的措辞——不加这个, 四个参数化用例
    可能全命中同一条错误分支(比如都被空字符串分支的检查拦下)而仍然全绿,
    "消息给出可照抄写法"这条设计要求就完全没被验证到。
    """
    with pytest.raises(ValueError, match=expected_match):
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
    assert cid.startswith("ctp-")
    assert cid.endswith("1")  # 序号仍以 1 结尾, 前面是 4 位运行盐(非 '-')


def test_account_event_bypasses_session_layer_even_if_strict(
    caplog: Any,
) -> None:
    """Minor #3: account 事件显式绕过第 1/2 层, 不是"巧合放行".

    即便 payload 里带了一个形态上像"别的任务"的 client_order_id、且严格模式
    已启用, account 事件也必须放行且不计入 foreign_task——这是设计要求的
    显式排除, 不是"account payload 恰好没有 client_order_id 字段所以落
    None"这种巧合(见 review Minor #3)。
    """
    b = _bridge(
        own_session_prefix="ctp-task_42-",
        strict_task_isolation=True,
        allowed={"600008.SH"},
    )
    s = _Strat()
    account_payload = {
        "client_order_id": "ctp-task_7-9999",  # 形似别的任务, 应被忽略
        "cash": 100000.0,
    }

    b.queue_event("account", account_payload)
    b.drain_events(s)

    assert len(s.portfolio_updates) == 1
    counts = b.dropped_event_counts()
    assert counts["foreign_task"] == 0
    assert counts["foreign_symbol"] == 0


def test_log_foreign_task_warns_once_then_debug(caplog: Any) -> None:
    """I-1/Minor #5: 同一外来任务前缀首次 WARNING, 之后降 DEBUG."""
    b = _bridge(
        own_session_prefix="ctp-task_42-",
        strict_task_isolation=True,
        allowed={"600008.SH"},
    )
    s = _Strat()

    with caplog.at_level("DEBUG", logger="akquant.gateway.live"):
        for i in range(3):
            b.queue_event(
                "order", _order(f"ctp-task_7-{i}", symbol="600008.SH", oid=f"O{i}")
            )
        b.drain_events(s)

    assert b.dropped_event_counts()["foreign_task"] == 3
    warnings = [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and "foreign task" in r.getMessage()
    ]
    debugs = [
        r
        for r in caplog.records
        if r.levelname == "DEBUG" and "foreign task" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert len(debugs) == 2
    assert len(b._warned_foreign_task_prefixes) == 1


def test_unstructured_foreign_ids_share_one_dedupe_bucket(caplog: Any) -> None:
    """I-1: 不含两个 '-' 的外来 id(非本框架系统下的单)共用一个去重桶.

    形如 ``A0000001`` 的 id 不能按 ``rsplit("-", 1)`` 分组(没有足够的 '-'),
    若仍逐单生成告警键会导致(a) 每笔外来单都是一条新 WARNING, (b)
    ``_warned_foreign_task_prefixes`` 无界增长——都是刚修掉的"稳态刷屏"的
    同形复现。修复后应退化到固定桶, 只告警一次。
    """
    b = _bridge(
        own_session_prefix="ctp-task_42-",
        strict_task_isolation=True,
        allowed={"600008.SH"},
    )
    s = _Strat()

    with caplog.at_level("DEBUG", logger="akquant.gateway.live"):
        for i in range(5):
            b.queue_event(
                "order", _order(f"A000000{i}", symbol="600008.SH", oid=f"P{i}")
            )
        b.drain_events(s)

    assert b.dropped_event_counts()["foreign_task"] == 5
    warnings = [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and "foreign task" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert len(b._warned_foreign_task_prefixes) == 1


def test_real_construction_wires_matching_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-2 接线测试 1/3: 真实 __init__ 构造出的生成端与判据端必须同源.

    这条同时锁住两件事: (a) 生产端 ``own_session_prefix`` 真的带尾部 '-'
    (``_runner.py`` 里若被改成不带尾部 '-' 拼接, 本测试会失败——参见
    review I-2 的第一处回退实验); (b) ``_next_client_order_id()`` 的生成端
    与 ``BrokerEventBridge`` 的判据端读的是同一个 ``session_tag``/broker 名,
    不会各自独立漂移。
    """
    _patch_live_deps(monkeypatch)

    runner = LiveRunner(
        strategy_cls=None,
        instruments=[],
        session_tag="task_42",
    )

    bridge = runner._broker_event_bridge
    assert bridge._own_session_prefix == "ctp-task_42-"
    assert bridge._strict_task_isolation is True
    cid = runner._next_client_order_id()
    assert cid.startswith(bridge._own_session_prefix)


def test_run_live_facade_forwards_session_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """I-2 接线测试 2/3: run_live() 门面必须把 session_tag 转发进 LiveRunner.

    通过 monkeypatch ``LiveRunner.run`` 拦下真正的阻塞运行, 只捕获构造好的
    runner 实例——``__init__`` 及其内部的 ``_init_broker_bridge_state()``
    已经真实跑过, 足以验证 ``run_live(session_tag=...)`` -> ``LiveRunner(...,
    session_tag=...)`` 这一条转发(review I-2 的第二处回退实验: 删掉
    ``_facade.py`` 里的 ``session_tag=session_tag`` 转发, 本测试应失败)。
    """
    _patch_live_deps(monkeypatch)

    captured: dict[str, Any] = {}

    def _fake_run(self: LiveRunner, **_: Any) -> None:
        captured["runner"] = self

    monkeypatch.setattr(LiveRunner, "run", _fake_run)

    run_live(
        strategy_cls=None,
        instruments=[],
        session_tag="task_facade",
    )

    runner = captured["runner"]
    assert runner.session_tag == "task_facade"
    assert runner._broker_strict_task_isolation is True
    assert runner._broker_event_bridge._own_session_prefix == "ctp-task_facade-"


def test_live_runner_rejects_invalid_session_tag_via_real_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-2 接线测试 3/3: 真实 __init__ 必须真的调用 _validate_session_tag.

    既有测试只直接调用 ``_validate_session_tag()``, 从未经过真实
    ``LiveRunner(session_tag=...)`` 构造去验证——若 ``__init__`` 里那行校验
    调用被删掉(review I-2 的第三处回退实验), 非法 session_tag 会被悄悄接受,
    直到运行期才在别处以更隐蔽的方式出问题。
    """
    _patch_live_deps(monkeypatch)

    with pytest.raises(ValueError, match="不能包含"):
        LiveRunner(
            strategy_cls=None,
            instruments=[],
            session_tag="bad-tag",
        )
