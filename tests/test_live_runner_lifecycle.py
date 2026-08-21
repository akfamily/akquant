"""实盘会话生命周期回调的补发测试.

与 ``test_live_runner.py`` 的区别: 那边全是 ``LiveRunner.__new__`` 绕过
``__init__`` 的装配级单测, 这里首次驱动完整的 ``LiveRunner.run()``,
验证收尾期回调真的被触发。
"""

from types import SimpleNamespace
from typing import Any, cast

import akquant.live._runner as live_module
import pytest
from akquant.live._runner import LiveRunner
from akquant.strategy import Strategy


class _FakeEngine:
    """驱动 run() 的最小引擎替身.

    ``run()`` 里显式调 ``_on_start_internal`` 是在模拟 Rust 侧
    ``src/engine/python.rs:1167`` 的行为 —— 主策略的 on_start 由 Rust
    触发, 不是 Python 侧调的。
    """

    def __init__(self, exc: BaseException | None = None) -> None:
        self.exc = exc
        self.events: list[str] = []

    def add_data(self, *a: Any) -> None:
        pass

    def set_cash(self, *a: Any) -> None:
        pass

    def add_instrument(self, *a: Any) -> None:
        pass

    def use_simulated_execution(self) -> None:
        pass

    def use_realtime_execution(self) -> None:
        pass

    def use_china_market(self) -> None:
        pass

    def use_china_futures_market(self) -> None:
        pass

    def set_force_session_continuous(self, *a: Any) -> None:
        pass

    def run(self, strategy: Any, show_progress: bool = False) -> None:
        if hasattr(strategy, "_on_start_internal"):
            strategy._on_start_internal()
        self.events.append("engine.run")
        if self.exc is not None:
            raise self.exc

    def get_results(self) -> Any:
        raise RuntimeError("fake engine has no results")


@pytest.fixture
def patched_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 run() 的外部依赖替换成替身: 网关工厂与启动等待."""
    monkeypatch.setattr(
        live_module,
        "create_gateway_bundle",
        lambda **kw: SimpleNamespace(
            market_gateway=None,
            trader_gateway=None,
            trader_capabilities=None,
            metadata=None,
        ),
    )
    monkeypatch.setattr(live_module.time, "sleep", lambda s: None)


def test_live_run_triggers_slot_strategy_on_start(patched_live: None) -> None:
    """实盘 slot 子策略的 on_start 必须被触发.

    Rust 侧只对主策略调 ``_on_start_internal``
    (``src/engine/python.rs:1167``), slot 策略在 Rust 里仅参与事件分发
    (``src/pipeline/stages/strategy.rs:71``)。回测在 Python 侧显式补这一步
    (``backtest/engine.py:3172-3176``), 实盘此前完全没有。
    """
    events: list[str] = []

    class Main(Strategy):
        def on_start(self) -> None:
            events.append("main_start")

        def on_bar(self, bar: Any) -> None:
            pass

    class Slot(Strategy):
        def on_start(self) -> None:
            events.append("slot_start")

        def on_bar(self, bar: Any) -> None:
            pass

    runner = LiveRunner(
        strategy_cls=Main,
        instruments=[],
        broker="ctp",
        strategies_by_slot={"slot-a": Slot},
    )
    runner.engine = cast(Any, _FakeEngine())
    runner.run(cash=1000.0)

    assert events.count("slot_start") == 1
    assert events.count("main_start") == 1


def test_on_stop_live_internal_skips_backtest_only_coverage_checks() -> None:
    """实盘收尾只补发结束钩子 + on_stop, 不跑三个回测语义的数据覆盖校验.

    三个校验(``_check_symbol_data_coverage`` / ``_check_warmup_symbol_coverage``
    / ``_check_incremental_hl_bar_coverage``)在实盘分别会误报、误报、以及
    **抛** ``StrategyConfigurationError`` 打断收尾 —— 盘中无成交标的、warmup
    未攒满、纯 tick 会话在实盘都是常态。
    """
    called: list[str] = []

    class Strat(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

        def on_stop(self) -> None:
            called.append("on_stop")

        def _check_symbol_data_coverage(self) -> None:
            called.append("symbol_data")

        def _check_warmup_symbol_coverage(self) -> None:
            called.append("warmup")

        def _check_incremental_hl_bar_coverage(self) -> None:
            called.append("hl_bar")

    strategy = Strat()
    strategy._on_stop_live_internal()

    assert called == ["on_stop"]


def test_on_stop_live_internal_is_idempotent() -> None:
    """重复派发实盘收尾不应重复执行用户 on_stop.

    ``_framework_stop_flushed`` 只护住 ``on_after_trading`` 的补发,
    护不住 ``on_stop`` 本身, 所以需要独立标志。
    """
    calls: list[int] = []

    class Strat(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

        def on_stop(self) -> None:
            calls.append(1)

    strategy = Strat()
    strategy._on_stop_live_internal()
    strategy._on_stop_live_internal()

    assert len(calls) == 1


def test_live_stop_flag_not_carried_by_checkpoint() -> None:
    """实盘收尾标志属会话内派生态: 不随存档带走, 续跑能再收尾一次."""

    class Strat(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

    strategy = Strat()
    strategy._on_stop_live_internal()
    assert strategy._framework_live_stop_dispatched is True

    state = strategy.__getstate__()
    assert "_framework_live_stop_dispatched" not in state


def _make_recording_strategy(events: list[str], stop_exc: bool = False) -> type:
    class Strat(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

        def on_stop(self) -> None:
            events.append("on_stop")
            if stop_exc:
                raise RuntimeError("user on_stop boom")

    return Strat


def test_live_run_normal_return_triggers_on_stop(patched_live: None) -> None:
    """run() 正常返回时(含 duration 经 Rust deadline 结束)触发 on_stop 一次."""
    events: list[str] = []
    runner = LiveRunner(
        strategy_cls=_make_recording_strategy(events), instruments=[], broker="ctp"
    )
    runner.engine = cast(Any, _FakeEngine())
    runner.run(cash=1000.0)

    assert events.count("on_stop") == 1


def test_live_run_keyboard_interrupt_triggers_on_stop(patched_live: None) -> None:
    """Duration 的回调 patch 路径(抛 KeyboardInterrupt)也要触发 on_stop.

    见 ``_runner.py`` 的 ``_apply_time_limit``: wrapped_on_bar 到点抛
    ``KeyboardInterrupt``。
    """
    events: list[str] = []
    runner = LiveRunner(
        strategy_cls=_make_recording_strategy(events), instruments=[], broker="ctp"
    )
    runner.engine = cast(Any, _FakeEngine(exc=KeyboardInterrupt("duration reached")))
    runner.run(cash=1000.0)

    assert events.count("on_stop") == 1


def test_live_run_error_abort_triggers_on_stop_and_keeps_summary(
    patched_live: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """异常中止路径下 on_stop 仍触发, 且 ABORTED ON ERROR 摘要不被顶掉."""
    events: list[str] = []
    reasons: list[str] = []
    runner = LiveRunner(
        strategy_cls=_make_recording_strategy(events), instruments=[], broker="ctp"
    )
    runner.engine = cast(Any, _FakeEngine(exc=RuntimeError("engine exploded")))
    monkeypatch.setattr(
        runner, "_print_summary", lambda reason="Manual Stop": reasons.append(reason)
    )
    runner.run(cash=1000.0)

    assert events.count("on_stop") == 1
    assert reasons == ["ABORTED ON ERROR"]


def test_live_run_user_on_stop_error_is_isolated(
    patched_live: None, caplog: pytest.LogCaptureFixture
) -> None:
    """用户 on_stop 抛异常只记 ERROR, 不外泄, 后续收尾步骤照常执行.

    实盘的 finally 抛出会盖掉 try 里真正的停止原因(包括标记
    ABORTED ON ERROR 的 CRITICAL), 使"任务看着跑完了其实没跑"重现。
    """
    events: list[str] = []
    summaries: list[str] = []
    runner = LiveRunner(
        strategy_cls=_make_recording_strategy(events, stop_exc=True),
        instruments=[],
        broker="ctp",
    )
    runner.engine = cast(Any, _FakeEngine())
    runner._print_summary = lambda reason="Manual Stop": summaries.append(reason)  # type: ignore[assignment]

    with caplog.at_level("ERROR"):
        runner.run(cash=1000.0)

    assert events.count("on_stop") == 1
    assert summaries == ["Manual Stop"]
    assert any("on_stop" in r.getMessage() for r in caplog.records)


def test_live_run_slot_on_stop_error_does_not_block_others(
    patched_live: None, caplog: pytest.LogCaptureFixture
) -> None:
    """一个 slot 的 on_stop 抛错不能阻止主策略与其余 slot 收尾."""
    events: list[str] = []

    class Main(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

        def on_stop(self) -> None:
            events.append("main")

    class BadSlot(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

        def on_stop(self) -> None:
            events.append("bad")
            raise RuntimeError("slot boom")

    class GoodSlot(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

        def on_stop(self) -> None:
            events.append("good")

    runner = LiveRunner(
        strategy_cls=Main,
        instruments=[],
        broker="ctp",
        strategies_by_slot={"bad": BadSlot, "good": GoodSlot},
    )
    runner.engine = cast(Any, _FakeEngine())
    with caplog.at_level("ERROR"):
        runner.run(cash=1000.0)

    assert sorted(events) == ["bad", "good", "main"]


def test_live_run_on_stop_precedes_gateway_shutdown_and_summary(
    patched_live: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """收尾顺序: on_stop -> 停信号源/broker 分发 -> 摘要.

    on_stop 排在停网关之前, 才能让用户在 on_stop 里撤单/平仓时通道仍可用;
    摘要排最后, 才能反映 on_stop 落盘后的最终状态。
    """
    order: list[str] = []

    class Strat(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

        def on_stop(self) -> None:
            order.append("on_stop")

    runner = LiveRunner(strategy_cls=Strat, instruments=[], broker="ctp")
    runner.engine = cast(Any, _FakeEngine())
    monkeypatch.setattr(
        runner, "_stop_signal_source", lambda: order.append("stop_signal")
    )
    monkeypatch.setattr(
        runner, "_stop_broker_dispatcher", lambda: order.append("stop_dispatcher")
    )
    monkeypatch.setattr(
        runner, "_print_summary", lambda reason="Manual Stop": order.append("summary")
    )
    runner.run(cash=1000.0)

    assert order == ["on_stop", "stop_signal", "stop_dispatcher", "summary"]


def test_live_run_stop_dispatch_survives_failure_before_strategy_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """策略尚未创建就失败时, 收尾派发不能自己炸掉.

    ``run()`` 里策略实例在网关装配之后才创建, 之前的
    ``create_gateway_bundle`` / broker_live 校验都可能先抛。
    """
    monkeypatch.setattr(
        live_module,
        "create_gateway_bundle",
        lambda **kw: (_ for _ in ()).throw(ValueError("gateway boom")),
    )
    monkeypatch.setattr(live_module.time, "sleep", lambda s: None)

    class Strat(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

    runner = LiveRunner(strategy_cls=Strat, instruments=[], broker="ctp")
    runner.engine = cast(Any, _FakeEngine())

    with pytest.raises(ValueError, match="gateway boom"):
        runner.run(cash=1000.0)

    runner._dispatch_strategy_stop()
