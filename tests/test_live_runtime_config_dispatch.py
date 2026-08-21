"""实盘侧把入口传入的 ``strategy_runtime_config`` 下发到主策略与各槽位策略.

五个开关本来在实盘就生效 —— 消费点 ``strategy_framework_hooks._runtime_option``
直读 ``strategy.runtime_config``, 与运行模式无关。缺的是从入口统一下发的能力:
回测有 ``run_backtest(strategy_runtime_config=...)``, 实盘此前没有对应参数, 只能
在策略内部自己写 ``self.runtime_config = ...``。

下发复用与回测同一个 ``apply_strategy_runtime_config``, 冲突检测与告警去重两侧
因此一致。时序上必须早于任何 ``on_start``: ``indicator_mode`` 决定指标注册走增量
还是预计算, 而指标在 ``on_start`` 里注册。
"""

import logging

import pandas as pd
import pytest
from akquant import AssetType, Instrument, run_live
from akquant.akquant import Bar
from akquant.live._runner import LiveRunner
from akquant.strategy import Strategy, StrategyRuntimeConfig


class _Reader(Strategy):
    """只用来读 runtime_config 的空策略."""


def _runner(runtime_config: object = None, override: bool = True) -> LiveRunner:
    """构造裸 LiveRunner(不走 __init__), 与 freq_injection 测试同一手法."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "test"
    runner.strategy_runtime_config = runtime_config  # type: ignore[assignment]
    runner.runtime_config_override = override
    return runner


def test_dispatches_dict_to_strategy() -> None:
    """字典形态被归一化后写到 runtime_config."""
    strategy = _Reader()
    _runner({"error_mode": "continue"})._apply_runtime_config([strategy])
    assert strategy.runtime_config.error_mode == "continue"


def test_dispatches_config_object() -> None:
    """对象形态同样生效."""
    strategy = _Reader()
    cfg = StrategyRuntimeConfig(portfolio_update_eps=0.5)
    _runner(cfg)._apply_runtime_config([strategy])
    assert strategy.runtime_config.portfolio_update_eps == 0.5


def test_dispatches_into_every_slot() -> None:
    """多槽位下主策略与每个槽位策略都要拿到 —— 这是回测 :2903/:2910 的对称行为."""
    targets: list[Strategy] = [_Reader(), _Reader(), _Reader()]
    _runner({"indicator_mode": "incremental"})._apply_runtime_config(targets)
    assert [t.runtime_config.indicator_mode for t in targets] == ["incremental"] * 3


def test_each_target_gets_its_own_instance() -> None:
    """下发的是副本: 改一个实例的配置不能串到其他实例."""
    a, b = _Reader(), _Reader()
    _runner({"error_mode": "continue"})._apply_runtime_config([a, b])
    assert a.runtime_config is not b.runtime_config


def test_noop_when_entry_did_not_pass_config() -> None:
    """入口没传就完全不动策略自设的值."""
    strategy = _Reader()
    strategy.runtime_config = StrategyRuntimeConfig(error_mode="legacy")
    _runner(None)._apply_runtime_config([strategy])
    assert strategy.runtime_config.error_mode == "legacy"


def test_override_true_replaces_strategy_own_value() -> None:
    """override=True(默认)时入口值胜出."""
    strategy = _Reader()
    strategy.runtime_config = StrategyRuntimeConfig(error_mode="legacy")
    _runner({"error_mode": "continue"}, override=True)._apply_runtime_config([strategy])
    assert strategy.runtime_config.error_mode == "continue"


def test_override_false_keeps_strategy_own_value() -> None:
    """override=False 时策略自设值保留, 入口值被忽略."""
    strategy = _Reader()
    strategy.runtime_config = StrategyRuntimeConfig(error_mode="legacy")
    _runner({"error_mode": "continue"}, override=False)._apply_runtime_config(
        [strategy]
    )
    assert strategy.runtime_config.error_mode == "legacy"


def test_conflict_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    """冲突要告警 —— 静默覆盖会让用户不知道自己的设置被顶掉了."""
    strategy = _Reader()
    strategy.runtime_config = StrategyRuntimeConfig(error_mode="legacy")
    with caplog.at_level(logging.WARNING):
        _runner({"error_mode": "continue"})._apply_runtime_config([strategy])
    assert "overrides strategy runtime_config" in caplog.text
    assert "error_mode: legacy -> continue" in caplog.text


def test_ignored_conflict_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    """override=False 的忽略也要告警, 否则入口参数静默失效."""
    strategy = _Reader()
    strategy.runtime_config = StrategyRuntimeConfig(error_mode="legacy")
    with caplog.at_level(logging.WARNING):
        _runner({"error_mode": "continue"}, override=False)._apply_runtime_config(
            [strategy]
        )
    assert "runtime_config_override=False" in caplog.text


def test_repeated_dispatch_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """同一实例反复下发同一份配置只警告一次, 否则多槽位会话会刷屏."""
    strategy = _Reader()
    strategy.runtime_config = StrategyRuntimeConfig(error_mode="legacy")
    runner = _runner({"error_mode": "continue"})
    with caplog.at_level(logging.WARNING):
        runner._apply_runtime_config([strategy])
        strategy.runtime_config = StrategyRuntimeConfig(error_mode="legacy")
        runner._apply_runtime_config([strategy])
    assert caplog.text.count("overrides strategy runtime_config") == 1


def test_no_conflict_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """入口值与策略自设值一致时不该有告警."""
    strategy = _Reader()
    with caplog.at_level(logging.WARNING):
        _runner({"error_mode": "raise"})._apply_runtime_config([strategy])
    assert "runtime_config" not in caplog.text


def test_unknown_field_rejected() -> None:
    """未知字段要报错并列出允许的字段, 不能静默丢弃."""
    strategy = _Reader()
    with pytest.raises(ValueError, match="unknown fields: nope"):
        _runner({"nope": 1})._apply_runtime_config([strategy])


def test_invalid_value_rejected() -> None:
    """非法取值走 dataclass 校验."""
    strategy = _Reader()
    with pytest.raises(ValueError, match="error_mode must be one of"):
        _runner({"error_mode": "bogus"})._apply_runtime_config([strategy])


def test_wrong_type_rejected() -> None:
    """既不是对象也不是字典时报 TypeError."""
    strategy = _Reader()
    with pytest.raises(TypeError, match="must be StrategyRuntimeConfig"):
        _runner("continue")._apply_runtime_config([strategy])


def test_run_live_exposes_both_parameters() -> None:
    """``run_live`` 的参数名与 ``run_backtest`` 逐字对称, 便于两侧切换."""
    import inspect

    from akquant.live import run_live

    params = inspect.signature(run_live).parameters
    assert "strategy_runtime_config" in params
    assert params["runtime_config_override"].default is True


def test_dispatch_happens_before_slot_on_start() -> None:
    """下发必须早于槽位 on_start —— indicator_mode 决定指标注册走哪条路.

    源码顺序守卫: 匹配 ``self.`` 前缀的真实调用点, 并先剥掉注释行 —— 否则注释里
    提到的方法名会被当成调用点(这个断言最初就是这么误报的)。
    """
    import inspect

    body = "\n".join(
        line
        for line in inspect.getsource(LiveRunner.run).splitlines()
        if not line.lstrip().startswith("#")
    )
    assert body.index("self._apply_runtime_config(") < body.index(
        "self._dispatch_slot_strategy_start("
    )


def _instrument(symbol: str) -> Instrument:
    """构造一个股票标的(与 replay 端到端测试同一手法)."""
    return Instrument(
        symbol=symbol,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bar(ts: int, symbol: str, close: float) -> Bar:
    """构造一根 bar."""
    return Bar(
        timestamp=ts,
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=1000.0,
        symbol=symbol,
    )


class _ConfigProbe(Strategy):
    """在 on_start / on_bar 里抓自己看到的 runtime_config."""

    def __init__(self) -> None:
        """初始化抓取容器."""
        self.seen_at_start: str | None = None
        self.seen_at_bar: list[str] = []

    def on_start(self) -> None:
        """记录 on_start 时刻的 error_mode —— 下发必须已经生效."""
        self.seen_at_start = str(self.runtime_config.error_mode)

    def on_bar(self, bar: Bar) -> None:
        """记录每根 bar 时刻的 error_mode."""
        self.seen_at_bar.append(str(self.runtime_config.error_mode))


def test_run_live_end_to_end_dispatch() -> None:
    """端到端: 参数从 ``run_live`` 穿过 LiveRunner 落到策略, 且 on_start 时已生效.

    走 ``broker="replay"`` 真跑一次会话 —— 单元测试都是裸构造 ``LiveRunner.__new__``
    打的, 不会覆盖 ``run_live`` → ``LiveRunner.__init__`` → ``run()`` 这条透传链。
    """
    strategy = _ConfigProbe()
    stamps = [
        int(pd.Timestamp("2023-01-03 09:30:00", tz="Asia/Shanghai").value),
        int(pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai").value),
    ]

    run_live(
        strategy_cls=strategy,
        instruments=[_instrument("REPLAY_RC")],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": [_bar(ts, "REPLAY_RC", 10.0) for ts in stamps]},
        cash=100_000.0,
        show_progress=False,
        duration="60s",
        strategy_runtime_config={"error_mode": "continue"},
    )

    assert strategy.seen_at_start == "continue", (
        f"on_start 时下发未生效: {strategy.seen_at_start}"
    )
    assert strategy.seen_at_bar == ["continue"] * len(stamps)


def test_run_live_end_to_end_default_is_untouched() -> None:
    """不传该参数时策略自设的 runtime_config 原样保留."""
    strategy = _ConfigProbe()
    strategy.runtime_config = StrategyRuntimeConfig(error_mode="legacy")
    stamp = int(pd.Timestamp("2023-01-03 09:30:00", tz="Asia/Shanghai").value)

    run_live(
        strategy_cls=strategy,
        instruments=[_instrument("REPLAY_RC")],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": [_bar(stamp, "REPLAY_RC", 10.0)]},
        cash=100_000.0,
        show_progress=False,
        duration="60s",
    )

    assert strategy.seen_at_start == "legacy"
