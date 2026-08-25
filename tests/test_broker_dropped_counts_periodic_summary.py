"""盘中周期性丢弃计数汇总: 挂在全量 sync 那一档, 只在有增量时才打 INFO.

覆盖 M4 补的可观测性缺口 ——
``BrokerEventBridge.dropped_event_counts()`` 此前唯一的消费点是会话收尾摘要,
盘中完全不可见。见
``docs/superpowers/specs/2026-08-25-broker-order-push-design.md`` 成分 C。
"""

from types import SimpleNamespace
from typing import Any, Callable, cast

from akquant.live._runner import LiveRunner


def _runner_with_bridge(dropped_event_counts: Callable[[], Any]) -> LiveRunner:
    """构造一个只装了 broker event bridge 替身的 ``LiveRunner``."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.strategy_id = "_default"
    runner._init_broker_bridge_state()
    runner._broker_event_bridge = SimpleNamespace(
        dropped_event_counts=dropped_event_counts
    )
    return runner


def test_reports_when_counts_increase(caplog: Any) -> None:
    """计数有增长 -> 汇总日志被打出, 且文本含两类计数与增量."""
    runner = _runner_with_bridge(lambda: {"foreign_symbol": 3, "duplicate_order": 5})

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._report_dropped_event_counts_if_changed()

    records = [r for r in caplog.records if "Broker event drops" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "foreign_symbol total=3" in message
    assert "(+3;" in message
    assert "duplicate_order total=5" in message
    assert "(+5;" in message


def test_silent_when_counts_unchanged(caplog: Any) -> None:
    """计数无变化 -> 完全不打日志(防噪声回归的关键).

    先跑一轮建立基线(3/5), 再原样跑第二轮; 第二轮不应产生任何
    "Broker event drops" 日志 —— 这是防止把 duplicate_order 每轮重放的
    预期增量误当成故障刷屏的关键回归点。
    """
    counts = {"foreign_symbol": 3, "duplicate_order": 5}
    runner = _runner_with_bridge(lambda: dict(counts))

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._report_dropped_event_counts_if_changed()
    caplog.clear()

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._report_dropped_event_counts_if_changed()

    assert [r for r in caplog.records if "Broker event drops" in r.getMessage()] == []


def test_reports_only_the_increment_not_the_running_total_again() -> None:
    """第二轮只在真的有新增量时才报, 且上次上报值被记住(增量口径正确)."""
    counts = {"foreign_symbol": 0, "duplicate_order": 5}
    runner = _runner_with_bridge(lambda: dict(counts))

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

    runner = _runner_with_bridge(boom)
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
        r for r in caplog.records if "dropped_event_counts() failed" in r.getMessage()
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
