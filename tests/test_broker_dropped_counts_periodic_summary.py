"""盘中周期性丢弃计数汇总: 挂在全量 sync 那一档, 只在 foreign_symbol 有增量时打 INFO.

覆盖 M4 补的可观测性缺口 ——
``BrokerEventBridge.dropped_event_counts()`` 此前唯一的消费点是会话收尾摘要,
盘中完全不可见。见
``docs/superpowers/specs/2026-08-25-broker-order-push-design.md`` 成分 C。

**触发条件只看 ``foreign_symbol``**(M4 复审订正): 账户有挂单且状态未变是
实盘最常见的稳态, 每轮全量 sync 都会让 ``duplicate_order`` 必然 +N(N = 挂单
数) —— 把它也纳入触发条件, 等于把"每 tick 刷屏"稀释成"每 30s 刷屏", 不是
消除。``duplicate_order`` 只作为日志上下文与收尾总计, 不参与触发判断。
"""

from types import SimpleNamespace
from typing import Any, Callable, cast

from akquant.live._runner import LiveRunner


def _runner_with_bridge(
    dropped_event_counts: Callable[[], Any],
    dropped_foreign_symbol_names: Callable[[], Any] | None = None,
) -> LiveRunner:
    """构造一个只装了 broker event bridge 替身的 ``LiveRunner``.

    ``dropped_foreign_symbol_names`` 省略时替身**不带**该访问器, 用于覆盖
    "bridge 没有这个方法"的兼容分支。
    """
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.strategy_id = "_default"
    runner._init_broker_bridge_state()
    attrs: dict[str, Any] = {"dropped_event_counts": dropped_event_counts}
    if dropped_foreign_symbol_names is not None:
        attrs["dropped_foreign_symbol_names"] = dropped_foreign_symbol_names
    runner._broker_event_bridge = SimpleNamespace(**attrs)
    return runner


def test_silent_when_the_same_foreign_symbols_keep_being_dropped(caplog: Any) -> None:
    """稳态防线: 外来标的集合不变、``foreign_symbol`` 计数持续增长 -> 完全静默.

    ``foreign_symbol`` 的计数点在 ``queue_event`` 最前面, **早于一切去重**,
    而 recovery 每档全量 sync 都会把柜台返回的**全账户**未完成委托重推一遍
    ⇒ 同一笔外来挂单每轮 +1。同账户下只要存在非挂载标的的挂单(别的任务、
    人工下单终端、隔夜挂单), 该计数就必然线性增长 —— 与 ``duplicate_order``
    是**同形**的稳态, 拿它当触发条件就是每 30s 准时刷一条, 正是
    ``test_silent_when_only_duplicate_order_grows`` 要防的那个失败模式。
    触发条件必须看"是否出现了**新的**外来标的"。
    """
    counts = {"foreign_symbol": 6, "duplicate_order": 5}
    names = {"600519.SH", "000001.SZ"}
    runner = _runner_with_bridge(lambda: dict(counts), lambda: set(names))
    runner._report_dropped_event_counts_if_changed()

    caplog.clear()
    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        for _ in range(5):
            counts["foreign_symbol"] += 2  # 同两个标的, 每轮 sync 重复计数
            runner._report_dropped_event_counts_if_changed()

    assert [r for r in caplog.records if "Broker event drops" in r.getMessage()] == []


def test_reports_only_the_newly_seen_foreign_symbols(caplog: Any) -> None:
    """出现此前没见过的外来标的 -> 打一条 INFO 并只点名新出现的那个."""
    counts = {"foreign_symbol": 2, "duplicate_order": 5}
    names = {"600519.SH"}
    runner = _runner_with_bridge(lambda: dict(counts), lambda: set(names))
    runner._report_dropped_event_counts_if_changed()  # 首轮把 600519.SH 报掉

    caplog.clear()
    counts["foreign_symbol"] = 4
    names.add("000001.SZ")
    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._report_dropped_event_counts_if_changed()

    records = [r for r in caplog.records if "Broker event drops" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "000001.SZ" in message
    # 老标的在首轮已点名过, 再报一次等于把稳态噪声按标的数放大。
    assert "600519.SH" not in message


def test_message_does_not_claim_the_count_should_stay_zero(caplog: Any) -> None:
    """文案不得断言"配置正确应恒为 0", 也不得把用户引向归一化 bug.

    ``session_tag`` 未传时**不启用**严格任务隔离(见
    ``BrokerEventBridge._session_layer_verdict``), 同账户下别的任务/人工
    终端的委托不走 ``foreign_task`` 而是落进 ``foreign_symbol`` ⇒ 它在默认
    配置下是正常的过滤工作量, 不是故障信号。旧文案("若配置正确应恒为 0,
    出现增长请怀疑标的归一化把自己的回报也挡掉了")把对接方引向一个不存在
    的 bug —— 2026-08-26 的反馈原文就是照抄这句在排查。
    """
    runner = _runner_with_bridge(
        lambda: {"foreign_symbol": 2, "duplicate_order": 0},
        lambda: {"600519.SH"},
    )
    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._report_dropped_event_counts_if_changed()

    message = next(
        r.getMessage() for r in caplog.records if "Broker event drops" in r.getMessage()
    )
    assert "恒为 0" not in message
    assert "归一化" not in message


def test_bridge_without_symbol_names_accessor_stays_silent(caplog: Any) -> None:
    """Bridge 没有标的名访问器 -> 静默且留一条 debug(不退回按计数触发).

    退回按计数触发就等于保留刚修掉的稳态刷屏; 会话收尾摘要仍有完整总计,
    因此静默是安全的一侧。
    """
    runner = _runner_with_bridge(lambda: {"foreign_symbol": 7, "duplicate_order": 0})

    with caplog.at_level("DEBUG", logger="akquant.gateway.live"):
        runner._report_dropped_event_counts_if_changed()

    assert [r for r in caplog.records if "Broker event drops" in r.getMessage()] == []
    assert [r for r in caplog.records if r.levelname == "DEBUG"] != []


def test_reported_message_carries_both_running_totals_as_context(caplog: Any) -> None:
    """报出时两类计数都作为上下文出现在文本里(排查不用再翻别处).

    它们只是上下文: 触发与否由标的集合决定, 见
    ``test_silent_when_the_same_foreign_symbols_keep_being_dropped``。
    """
    runner = _runner_with_bridge(
        lambda: {"foreign_symbol": 3, "duplicate_order": 5},
        lambda: {"600519.SH"},
    )

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._report_dropped_event_counts_if_changed()

    records = [r for r in caplog.records if "Broker event drops" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "foreign_symbol total=3" in message
    assert "duplicate_order total=5" in message
    assert "600519.SH" in message


def test_silent_when_nothing_changed(caplog: Any) -> None:
    """计数与标的集合都无变化 -> 完全不打日志(防噪声回归的关键).

    先跑一轮建立基线, 再原样跑第二轮; 第二轮不应产生任何
    "Broker event drops" 日志。
    """
    counts = {"foreign_symbol": 3, "duplicate_order": 5}
    names = {"600519.SH"}
    runner = _runner_with_bridge(lambda: dict(counts), lambda: set(names))

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._report_dropped_event_counts_if_changed()
    caplog.clear()

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._report_dropped_event_counts_if_changed()

    assert [r for r in caplog.records if "Broker event drops" in r.getMessage()] == []


def test_silent_when_only_duplicate_order_grows(caplog: Any) -> None:
    """稳态防线: 无外来标的、duplicate_order 持续增长 -> 完全不打日志.

    账户挂单且状态未变时, 每轮全量 sync 都会让 ``duplicate_order`` +N
    (N = 挂单数), 若把它当触发条件, 会在整个挂单周期里每 30s 准时打一条
    日志——把"每 tick 刷屏"稀释成"每 30s 刷屏", 不是真正的静默。
    """
    counts = {"foreign_symbol": 0, "duplicate_order": 5}
    runner = _runner_with_bridge(lambda: dict(counts), set)
    runner._report_dropped_event_counts_if_changed()

    caplog.clear()
    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        for _ in range(5):
            counts["duplicate_order"] += 5  # 模拟连续几轮全量 sync 重放同样的挂单
            runner._report_dropped_event_counts_if_changed()

    assert [r for r in caplog.records if "Broker event drops" in r.getMessage()] == []


def test_running_totals_are_recorded_even_when_nothing_is_reported() -> None:
    """累计值每轮都刷新(供收尾摘要与下次日志读), 与是否打日志无关."""
    counts = {"foreign_symbol": 0, "duplicate_order": 5}
    runner = _runner_with_bridge(lambda: dict(counts), set)

    runner._report_dropped_event_counts_if_changed()
    assert runner._broker_last_reported_drop_counts == {
        "foreign_symbol": 0,
        "duplicate_order": 5,
    }

    counts["duplicate_order"] = 9
    runner._report_dropped_event_counts_if_changed()
    assert runner._broker_last_reported_drop_counts == {
        "foreign_symbol": 0,
        "duplicate_order": 9,
    }


def test_dropped_event_counts_exception_does_not_break_recovery_cycle(
    caplog: Any,
) -> None:
    """``dropped_event_counts()`` 抛异常 -> 恢复循环不受影响, 只有一条 debug.

    构造与 ``test_broker_recovery_cadence.py`` 同款的假 gateway, 按
    ``_broker_recovery_loop`` 里 do_sync 那一拍的真实调用顺序——先跑
    ``_run_broker_recovery_cycle``, 再跑 ``_report_dropped_event_counts_if_changed``
    ——验证后者内部炸掉不会向外抛, 也不影响前者已经完成的柜台同步。
    """

    class _Gateway:
        def __init__(self) -> None:
            self.heartbeats = 0
            self.sync_orders_calls = 0
            self.sync_trades_calls = 0
            self.account_calls = 0

        def heartbeat(self) -> bool:
            self.heartbeats += 1
            return True

        def sync_open_orders(self) -> list[Any]:
            self.sync_orders_calls += 1
            return []

        def sync_today_trades(self) -> list[Any]:
            self.sync_trades_calls += 1
            return []

        def query_account(self) -> None:
            self.account_calls += 1
            return None

    def boom() -> dict[str, int]:
        raise RuntimeError("dropped_event_counts exploded")

    runner = _runner_with_bridge(boom, set)
    gateway = _Gateway()
    runner._broker_trader_gateway = gateway
    runner._broker_baseline_done = True

    with caplog.at_level("DEBUG", logger="akquant.gateway.live"):
        reconnected = runner._run_broker_recovery_cycle(
            cast(Any, None), sync_orders=True, sync_trades=True, refresh_account=True
        )
        runner._report_dropped_event_counts_if_changed()

    # 该轮其余阶段(心跳/资金/全量 sync)照常执行, 不受汇总失败影响。
    assert reconnected is False
    assert gateway.heartbeats == 1
    assert gateway.sync_orders_calls == 1
    assert gateway.sync_trades_calls == 1
    assert gateway.account_calls == 1

    debug_records = [
        r
        for r in caplog.records
        if "dropped_event_counts()/dropped_foreign_symbol_names() failed"
        in r.getMessage()
    ]
    assert len(debug_records) == 1
    assert debug_records[0].levelname == "DEBUG"
    info_records = [r for r in caplog.records if "Broker event drops" in r.getMessage()]
    assert info_records == []


def test_no_bridge_is_a_noop() -> None:
    """``_broker_event_bridge`` 不存在时(绕开 __init__ 的替身)不报错."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.strategy_id = "_default"
    # 刻意不调用 _init_broker_bridge_state, 模拟测试替身没有该属性的场景。
    runner._report_dropped_event_counts_if_changed()
