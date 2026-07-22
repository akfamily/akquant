from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import akquant.live._runner as live_module
import pytest
from akquant.live._runner import LiveRunner
from akquant.strategy import Strategy


def test_live_runner_init_normalizes_legacy_gateway_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LiveRunner should fold legacy broker args into gateway_options at init."""

    class _DummyDataFeed:
        @staticmethod
        def create_live() -> object:
            return object()

    class _DummyEngine:
        pass

    monkeypatch.setattr(live_module, "DataFeed", _DummyDataFeed)
    monkeypatch.setattr(live_module, "Engine", _DummyEngine)

    runner = LiveRunner(
        strategy_cls=None,
        instruments=[],
        md_front="tcp://md-front",
        broker_id="9999",
        user_id="trader-a",
        password="secret",
        gateway_options={"recovery_mode": "strict", "custom": "value"},
    )

    assert runner.gateway_options == {
        "recovery_mode": "strict",
        "custom": "value",
        "md_front": "tcp://md-front",
        "broker_id": "9999",
        "user_id": "trader-a",
        "password": "secret",
    }
    assert runner.md_front == "tcp://md-front"
    assert runner.broker_id == "9999"
    assert runner.user_id == "trader-a"
    assert runner.password == "secret"
    assert runner._build_gateway_kwargs() == runner.gateway_options


def test_live_runner_init_prefers_gateway_options_over_legacy_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit gateway_options values should win over legacy init args."""

    class _DummyDataFeed:
        @staticmethod
        def create_live() -> object:
            return object()

    class _DummyEngine:
        pass

    monkeypatch.setattr(live_module, "DataFeed", _DummyDataFeed)
    monkeypatch.setattr(live_module, "Engine", _DummyEngine)

    runner = LiveRunner(
        strategy_cls=None,
        instruments=[],
        md_front="tcp://legacy-md-front",
        broker_id="legacy-broker",
        user_id="legacy-user",
        gateway_options={
            "md_front": "tcp://explicit-md-front",
            "broker_id": "explicit-broker",
            "user_id": "explicit-user",
        },
    )

    assert runner.gateway_options["md_front"] == "tcp://explicit-md-front"
    assert runner.gateway_options["broker_id"] == "explicit-broker"
    assert runner.gateway_options["user_id"] == "explicit-user"
    assert runner.md_front == "tcp://explicit-md-front"
    assert runner.broker_id == "explicit-broker"
    assert runner.user_id == "explicit-user"


def test_live_runner_builds_strategy_instance_from_class() -> None:
    """Build strategy instance from class input."""

    class _DummyStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = _DummyStrategy
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    strategy = runner._build_strategy_instance(runner.strategy_cls)
    assert isinstance(strategy, _DummyStrategy)


def test_live_runner_builds_strategy_instance_from_existing_instance() -> None:
    """Reuse provided strategy instance input."""

    class _DummyStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    instance = _DummyStrategy()
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = instance
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    strategy = runner._build_strategy_instance(runner.strategy_cls)
    assert strategy is instance


def test_live_runner_builds_functional_strategy_instance() -> None:
    """Build functional strategy wrapper from callable input."""
    events: list[str] = []

    def initialize(ctx: Any) -> None:
        events.append("initialize")
        ctx.seed = 7

    def on_start(ctx: Any) -> None:
        _ = ctx
        events.append("on_start")

    def on_stop(ctx: Any) -> None:
        _ = ctx
        events.append("on_stop")

    def on_bar(ctx: Any, bar: Any) -> None:
        _ = bar
        events.append(f"bar:{getattr(ctx, 'seed', 0)}")

    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = on_bar
    runner.initialize = initialize
    runner.on_start = on_start
    runner.on_stop = on_stop
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {"flag": "ok"}
    strategy = runner._build_strategy_instance(runner.strategy_cls)

    assert isinstance(strategy, Strategy)
    assert getattr(strategy, "flag") == "ok"
    assert events == ["initialize"]
    strategy.on_start()
    strategy.on_bar(cast(Any, SimpleNamespace(symbol="TEST")))
    strategy.on_stop()
    assert events == ["initialize", "on_start", "bar:7", "on_stop"]


def test_live_runner_builds_strategy_instance_from_strategy_source(
    tmp_path: Path,
) -> None:
    """Build strategy instance from configured strategy_source."""
    strategy_file = tmp_path / "live_source_strategy.py"
    strategy_file.write_text(
        "\n".join(
            [
                "from akquant.strategy import Strategy",
                "",
                "class Strategy(Strategy):",
                "    def __init__(self):",
                "        self.calls = 0",
                "",
                "    def on_bar(self, bar):",
                "        self.calls += 1",
            ]
        ),
        encoding="utf-8",
    )

    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = None
    runner.strategy_source = str(strategy_file)
    runner.strategy_loader = "python_plain"
    runner.strategy_loader_options = None
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    strategy = runner._build_strategy_instance(runner.strategy_cls)

    assert isinstance(strategy, Strategy)
    assert type(strategy).__name__ == "Strategy"


def test_live_runner_builds_strategy_from_encrypted_external_loader() -> None:
    """Build strategy instance using encrypted_external loader callback."""

    class _LoadedStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    def _decrypt_loader(source: Any, options: dict[str, Any]) -> type[Strategy]:
        _ = source
        _ = options
        return _LoadedStrategy

    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = None
    runner.strategy_source = b"cipher"
    runner.strategy_loader = "encrypted_external"
    runner.strategy_loader_options = {"decrypt_and_load": _decrypt_loader}
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    strategy = runner._build_strategy_instance(runner.strategy_cls)

    assert isinstance(strategy, _LoadedStrategy)


def test_live_runner_rejects_missing_strategy_and_source() -> None:
    """Live runner should fail when both strategy and source are missing."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = None
    runner.strategy_source = None
    runner.strategy_loader = None
    runner.strategy_loader_options = None
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    with pytest.raises(ValueError, match="Strategy must be provided"):
        runner._build_strategy_instance(runner.strategy_cls)


def test_live_runner_builds_strategy_topology_with_slots() -> None:
    """Build primary and slot strategies with explicit strategy ids."""

    def on_bar(ctx: Any, bar: Any) -> None:
        _ = ctx
        _ = bar

    def slot_on_bar(ctx: Any, bar: Any) -> None:
        _ = ctx
        _ = bar

    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = on_bar
    runner.strategy_id = "alpha"
    runner.strategies_by_slot = {"beta": slot_on_bar}
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    strategy, slots, strategy_id = runner._build_strategy_topology()

    assert isinstance(strategy, Strategy)
    assert strategy_id == "alpha"
    assert set(slots.keys()) == {"beta"}
    assert isinstance(slots["beta"], Strategy)


def test_live_runner_configures_engine_slots_for_primary_and_secondary() -> None:
    """Configure slot metadata and strategy binding on engine."""

    class _DummyEngine:
        def __init__(self) -> None:
            self.slot_ids: list[str] = []
            self.default_strategy_id = ""
            self.slot_strategies: dict[int, Any] = {}

        def set_strategy_slots(self, slot_ids: list[str]) -> None:
            self.slot_ids = slot_ids

        def set_default_strategy_id(self, strategy_id: str) -> None:
            self.default_strategy_id = strategy_id

        def set_strategy_for_slot(self, slot_index: int, strategy: Any) -> None:
            self.slot_strategies[slot_index] = strategy

    class _DummyStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    runner = LiveRunner.__new__(LiveRunner)
    runner.engine = cast(Any, _DummyEngine())
    runner.context = {"shared_flag": "ok"}
    primary = _DummyStrategy()
    secondary = _DummyStrategy()
    runner._configure_strategy_slots(primary, {"beta": secondary}, "alpha")
    engine = cast(_DummyEngine, runner.engine)

    assert engine.slot_ids == ["alpha", "beta"]
    assert engine.default_strategy_id == "alpha"
    assert engine.slot_strategies[0] is primary
    assert engine.slot_strategies[1] is secondary
    assert getattr(primary, "_owner_strategy_id") == "alpha"
    assert getattr(secondary, "_owner_strategy_id") == "beta"
    assert getattr(primary, "shared_flag") == "ok"
    assert getattr(secondary, "shared_flag") == "ok"


def test_live_runner_applies_strategy_risk_controls_for_slots() -> None:
    """Apply strategy-level risk controls using configured slot ids."""

    class _DummyEngine:
        def __init__(self) -> None:
            self.slot_ids: list[str] = []
            self.default_strategy_id = ""
            self.slot_strategies: dict[int, Any] = {}
            self.max_order_value_limits: dict[str, float] = {}
            self.max_order_size_limits: dict[str, float] = {}
            self.max_position_size_limits: dict[str, float] = {}
            self.max_daily_loss_limits: dict[str, float] = {}
            self.max_drawdown_limits: dict[str, float] = {}
            self.reduce_only_flags: dict[str, bool] = {}
            self.cooldown_bars: dict[str, int] = {}
            self.strategy_priorities: dict[str, int] = {}
            self.strategy_risk_budget_limits: dict[str, float] = {}
            self.portfolio_risk_budget_limit: float | None = None
            self.risk_budget_mode = ""
            self.risk_budget_reset_daily = False

        def set_strategy_slots(self, slot_ids: list[str]) -> None:
            self.slot_ids = slot_ids

        def set_default_strategy_id(self, strategy_id: str) -> None:
            self.default_strategy_id = strategy_id

        def set_strategy_for_slot(self, slot_index: int, strategy: Any) -> None:
            self.slot_strategies[slot_index] = strategy

        def set_strategy_max_order_value_limits(self, limits: dict[str, float]) -> None:
            self.max_order_value_limits = limits

        def set_strategy_max_order_size_limits(self, limits: dict[str, float]) -> None:
            self.max_order_size_limits = limits

        def set_strategy_max_position_size_limits(
            self, limits: dict[str, float]
        ) -> None:
            self.max_position_size_limits = limits

        def set_strategy_max_daily_loss_limits(self, limits: dict[str, float]) -> None:
            self.max_daily_loss_limits = limits

        def set_strategy_max_drawdown_limits(self, limits: dict[str, float]) -> None:
            self.max_drawdown_limits = limits

        def set_strategy_reduce_only_after_risk(self, flags: dict[str, bool]) -> None:
            self.reduce_only_flags = flags

        def set_strategy_risk_cooldown_bars(self, bars: dict[str, int]) -> None:
            self.cooldown_bars = bars

        def set_strategy_priorities(self, priorities: dict[str, int]) -> None:
            self.strategy_priorities = priorities

        def set_strategy_risk_budget_limits(self, limits: dict[str, float]) -> None:
            self.strategy_risk_budget_limits = limits

        def set_portfolio_risk_budget_limit(self, limit: float | None) -> None:
            self.portfolio_risk_budget_limit = limit

        def set_risk_budget_mode(self, mode: str) -> None:
            self.risk_budget_mode = mode

        def set_risk_budget_reset_daily(self, enabled: bool) -> None:
            self.risk_budget_reset_daily = enabled

    class _DummyStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    runner = LiveRunner.__new__(LiveRunner)
    runner.engine = cast(Any, _DummyEngine())
    runner.context = {}
    runner.strategy_max_order_value = {"alpha": 1000.0, "beta": 2000.0}
    runner.strategy_max_order_size = {"alpha": 10.0, "beta": 20.0}
    runner.strategy_max_position_size = {"alpha": 100.0, "beta": 200.0}
    runner.strategy_max_daily_loss = {"alpha": 0.02, "beta": 0.03}
    runner.strategy_max_drawdown = {"alpha": 0.1, "beta": 0.15}
    runner.strategy_reduce_only_after_risk = {"alpha": True, "beta": False}
    runner.strategy_risk_cooldown_bars = {"alpha": 3, "beta": 5}
    runner.strategy_priority = {"alpha": 1, "beta": 2}
    runner.strategy_risk_budget = {"alpha": 50000.0, "beta": 60000.0}
    runner.portfolio_risk_budget = 120000.0
    runner.risk_budget_mode = "order_notional"
    runner.risk_budget_reset_daily = True
    primary = _DummyStrategy()
    secondary = _DummyStrategy()
    runner._configure_strategy_slots(primary, {"beta": secondary}, "alpha")
    engine = cast(_DummyEngine, runner.engine)

    assert engine.max_order_value_limits == {"alpha": 1000.0, "beta": 2000.0}
    assert engine.max_order_size_limits == {"alpha": 10.0, "beta": 20.0}
    assert engine.max_position_size_limits == {"alpha": 100.0, "beta": 200.0}
    assert engine.max_daily_loss_limits == {"alpha": 0.02, "beta": 0.03}
    assert engine.max_drawdown_limits == {"alpha": 0.1, "beta": 0.15}
    assert engine.reduce_only_flags == {"alpha": True, "beta": False}
    assert engine.cooldown_bars == {"alpha": 3, "beta": 5}
    assert engine.strategy_priorities == {"alpha": 1, "beta": 2}
    assert engine.strategy_risk_budget_limits == {"alpha": 50000.0, "beta": 60000.0}
    assert engine.portfolio_risk_budget_limit == 120000.0
    assert engine.risk_budget_mode == "order_notional"
    assert engine.risk_budget_reset_daily is True


def test_live_runner_rejects_unknown_strategy_ids_in_risk_controls() -> None:
    """Reject strategy-level maps containing ids outside configured slots."""

    class _DummyEngine:
        def set_strategy_slots(self, slot_ids: list[str]) -> None:
            _ = slot_ids

        def set_default_strategy_id(self, strategy_id: str) -> None:
            _ = strategy_id

        def set_strategy_for_slot(self, slot_index: int, strategy: Any) -> None:
            _ = slot_index
            _ = strategy

    class _DummyStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    runner = LiveRunner.__new__(LiveRunner)
    runner.engine = cast(Any, _DummyEngine())
    runner.context = {}
    runner.strategy_max_order_value = {"ghost": 123.0}
    runner.strategy_max_order_size = {}
    runner.strategy_max_position_size = {}
    runner.strategy_max_daily_loss = {}
    runner.strategy_max_drawdown = {}
    runner.strategy_reduce_only_after_risk = {}
    runner.strategy_risk_cooldown_bars = {}
    runner.strategy_priority = {}
    runner.strategy_risk_budget = {}
    runner.portfolio_risk_budget = None
    runner.risk_budget_mode = "order_notional"
    runner.risk_budget_reset_daily = False
    primary = _DummyStrategy()
    secondary = _DummyStrategy()

    try:
        runner._configure_strategy_slots(primary, {"beta": secondary}, "alpha")
        assert False, "expected ValueError for unknown strategy id"
    except ValueError as exc:
        assert "unknown strategy ids: ghost" in str(exc)


def test_live_runner_logs_invalid_duration_format(caplog: Any) -> None:
    """Invalid live duration should be logged instead of printed."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_id = "alpha"

    with caplog.at_level("WARNING", logger="akquant.gateway.live"):
        runner._apply_time_limit(cast(Any, SimpleNamespace()), "not-a-duration")

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Ignored invalid live duration format: not-a-duration"
    )
    assert record.phase == "live"
    assert record.strategy_id == "alpha"
    assert record.slot == "alpha"


def test_live_runner_logs_summary_with_structured_context(caplog: Any) -> None:
    """Live summary should be emitted through the gateway logger."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_id = "beta"
    runner.engine = cast(
        Any,
        SimpleNamespace(
            get_results=lambda: SimpleNamespace(
                metrics=SimpleNamespace(
                    total_return_pct=0.12,
                    annualized_return=0.08,
                    max_drawdown_pct=-0.04,
                    sharpe_ratio=1.23,
                    win_rate=0.67,
                ),
                trades=[object(), object()],
                snapshots=[
                    (
                        0,
                        [
                            SimpleNamespace(symbol="IF2406", quantity=2.0),
                            SimpleNamespace(symbol="rb2406", quantity=0.0),
                        ],
                    )
                ],
            )
        ),
    )

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._print_summary()

    record = next(
        record for record in caplog.records if "TRADING SUMMARY" in record.getMessage()
    )
    assert record.phase == "live"
    assert record.strategy_id == "beta"
    assert record.slot == "beta"
    assert "Current Positions:" in record.getMessage()
    assert "IF2406: 2.0" in record.getMessage()
