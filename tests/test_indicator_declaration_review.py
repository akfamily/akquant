"""code-review 抓出的声明式指标层缺陷回归(2026-09-21).

每条测试对应一条 review finding, docstring 里写清失败场景; 修复前全部失败。
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, List, cast

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Indicator, Strategy, run_backtest
from akquant.akquant import Tick
from akquant.indicator_declaration import IndicatorBinding

SYMBOL = "REVIEW"


class _PySma:
    """可 pickle 的纯 Python SMA(Rust 指标不能 pickle, warm start 用不了)."""

    def __init__(self, period: int) -> None:
        self.period = period
        self.buf: deque[float] = deque(maxlen=period)

    def update(self, value: float) -> None:
        self.buf.append(float(value))

    @property
    def value(self) -> Any:
        if len(self.buf) < self.period:
            return None
        return sum(self.buf) / self.period


def _make_py_sma3() -> _PySma:
    return _PySma(3)


class _AutoNamedProbe(Strategy):
    """省略 name 的声明; 模块级才能被 pickle."""

    first_values: List[Any] = []

    def on_start(self) -> None:
        self.sma = self.I(factory=_make_py_sma3, source="close")

    def on_bar(self, bar: Bar) -> None:
        _AutoNamedProbe.first_values.append(self.sma[0])


class _ClashProbe(Strategy):
    """binding 存在 self.fast, 声明名 'period' 与 self.period=20 撞名."""

    def on_start(self) -> None:
        self.period = 20
        self.fast = self.I(factory=_make_py_sma3, name="period", source="close")

    def on_bar(self, bar: Bar) -> None:
        pass


def _bars(start: str, periods: int, start_price: float = 100.0) -> List[Bar]:
    idx = pd.date_range(start=start, periods=periods, freq="D")
    return [
        Bar(
            timestamp=int(ts.value),
            open=start_price + i,
            high=start_price + i + 1,
            low=start_price + i - 1,
            close=start_price + i,
            volume=100.0,
            symbol=SYMBOL,
        )
        for i, ts in enumerate(idx)
    ]


def _md(n: int, symbol: str = SYMBOL) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "symbol": symbol,
            "open": [10.0 + i for i in range(n)],
            "high": [11.0 + i for i in range(n)],
            "low": [9.0 + i for i in range(n)],
            "close": [10.0 + i for i in range(n)],
            "volume": 100.0,
        }
    )


# --------------------------------------------------------------------------
# finding 1: 省略 name 的声明在 warm start 恢复后要命中同一条, 不能孤立预热状态
# --------------------------------------------------------------------------
def test_auto_named_declaration_is_stable_across_warm_start(tmp_path: Path) -> None:
    """恢复后 on_start 重跑, 省略 name 的 I() 必须解析成同一个名字并复用已恢复状态.

    修复前: 字典非空时 _resolve_indicator_name 永远追加后缀 → "…_2" → 冷启动新指标,
    恢复的预热状态被孤立, 用户拿到的 binding 未预热。
    """
    from akquant import run_from_checkpoint, save_checkpoint

    Probe = _AutoNamedProbe
    Probe.first_values = []
    result1 = run_backtest(
        data=_bars("2023-01-01", 3),
        strategy=Probe,
        symbols=SYMBOL,
        initial_cash=1e5,
        show_progress=False,
    )
    checkpoint = tmp_path / "auto_name.pkl"
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    Probe.first_values = []
    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=_bars("2023-01-04", 2, start_price=103.0),
        symbols=SYMBOL,
        show_progress=False,
    )
    restored = result2.strategy
    assert restored is not None

    assert set(restored._incremental_indicators) == {"make_py_sma3"}
    # 已恢复的 SMA(3) 已攒满 3 根, 第二阶段首根 bar 就有值; 冷启动的会是 None
    assert Probe.first_values[0] is not None


# --------------------------------------------------------------------------
# finding 6: pickle 按"持有 binding 的属性"重建, 不能按声明名 setattr 覆盖用户属性
# --------------------------------------------------------------------------
def test_pickle_restores_binding_on_user_attribute_not_declaration_name(
    tmp_path: Path,
) -> None:
    """用户把 binding 存在 self.fast, 声明名 'period' 与用户的 self.period=20 撞名.

    修复前: __setstate__ 按声明名 setattr → self.period 被覆盖成 binding;
    self.fast 上的 binding 反而没被重建。走与用户一致的 checkpoint 路径(直接
    pickle 策略会带上不可 pickle 的 Engine)。
    """
    from akquant import load_checkpoint, save_checkpoint

    result = run_backtest(
        data=_bars("2023-01-01", 3),
        strategy=_ClashProbe,
        symbols=SYMBOL,
        initial_cash=1e5,
        show_progress=False,
    )
    strategy = result.strategy
    assert strategy is not None

    path = tmp_path / "clash.pkl"
    save_checkpoint(result.engine, strategy, str(path))  # type: ignore[arg-type]
    _, restored = load_checkpoint(str(path))

    assert restored.period == 20
    assert isinstance(restored.fast, IndicatorBinding)
    assert restored.fast._name == "period"


# --------------------------------------------------------------------------
# finding 3: 预计算指标没有增量状态, warmup_bars 对它无意义且会让 bootstrap 崩
# --------------------------------------------------------------------------
def test_precomputed_indicator_rejects_warmup_bars() -> None:
    """self.I(Indicator(...), warmup_bars=5) 应直接报错.

    而不是等到数据准备阶段才抛 NotImplementedError.
    """

    class Probe(Strategy):
        def on_start(self) -> None:
            self.mom = self.I(
                Indicator("mom", lambda df: df["close"].diff()), warmup_bars=5
            )

        def on_bar(self, bar: Bar) -> None:
            pass

    with pytest.raises(ValueError, match="warmup_bars"):
        run_backtest(
            data=_md(6),
            strategy=Probe,
            symbols=[SYMBOL],
            initial_cash=1e5,
            show_progress=False,
            timezone="UTC",
            start_time=pd.Timestamp("2024-01-03", tz="UTC"),
        )


# --------------------------------------------------------------------------
# finding 5: 预计算 Indicator 天然按 symbol 缓存, 多标的下代理属性不能报"共享实例"错
# --------------------------------------------------------------------------
def test_precomputed_indicator_proxies_work_across_symbols() -> None:
    """两只标的 + 预计算指标: get_instance()/代理属性对第二只标的不抛 ValueError."""
    seen: List[Any] = []

    class Probe(Strategy):
        def on_start(self) -> None:
            self.mom = self.I(
                Indicator("mom", lambda df: df["close"].diff()), name="mom"
            )

        def on_bar(self, bar: Bar) -> None:
            inst = self.mom.get_instance()
            seen.append((bar.symbol, self.mom[0], inst.name))

    a = _md(4, "AAA")
    b = _md(4, "BBB")
    b["close"] = b["close"] * 10
    b["open"] = b["open"] * 10
    run_backtest(
        data=pd.concat([a, b], ignore_index=True),
        strategy=Probe,
        symbols=["AAA", "BBB"],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
    )

    assert {s for s, _, _ in seen} == {"AAA", "BBB"}
    assert all(name == "mom" for _, _, name in seen)
    # 各自的 diff: AAA 每天 +1, BBB 每天 +10
    assert any(v == 1.0 for s, v, _ in seen if s == "AAA")
    assert any(v == 10.0 for s, v, _ in seen if s == "BBB")


# --------------------------------------------------------------------------
# finding 4: 旧协议 sink(无 confirmed、无 **kwargs)必须仍能收确认点
# --------------------------------------------------------------------------
class _LegacySink:
    """严格按 1.2 版协议签名实现的第三方 sink: 没有 confirmed, 也没有 **kwargs."""

    def __init__(self) -> None:
        self.points: List[dict[str, Any]] = []

    def record(
        self,
        *,
        name: str,
        value: Any,
        symbol: str,
        timestamp: Any,
        owner_strategy_id: str,
        display_name: Any = None,
        pane: int = 0,
        render_type: str = "line",
        unit: Any = None,
        precision: Any = None,
        color: Any = None,
        meta: Any = None,
        reference_lines: Any = None,
        scale_group: Any = None,
        warmup: bool = False,
    ) -> None:
        self.points.append({"name": name, "value": value, "symbol": symbol})

    def build_payload(self) -> dict[str, list[dict[str, Any]]]:
        return {"definitions": [], "instances": [], "points": []}

    def flush_stream_snapshot(self) -> None:
        return None

    def set_stream_emitter(self, emitter: Any) -> None:
        return None


def test_legacy_sink_without_confirmed_still_receives_confirmed_points() -> None:
    """record_indicator 不能无条件传 confirmed=: 旧签名 sink 会 TypeError."""
    sink = _LegacySink()

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), name="sma", plot=True)

        def on_bar(self, bar: Bar) -> None:
            pass

    run_backtest(
        data=_md(4),
        strategy=Probe,
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
        indicator_recorder=cast(Any, sink),
    )

    assert [p["value"] for p in sink.points] == [10.5, 11.5, 12.5]


def test_legacy_sink_skips_provisional_points_instead_of_crashing() -> None:
    """开了 intrabar 的策略配旧 sink: 临时点跳过(告警), 确认点照常, 会话不崩."""
    sink = _LegacySink()

    class Probe(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.subscribe_bars("5min")

        window_closes: List[float] = []

        def on_start(self) -> None:
            self.sma = self.I(
                aq.SMA(1), name="sma", freq="5min", intrabar=True, plot=True
            )

        def on_bar(self, bar: Bar) -> None:
            pass

        def on_window_bar(self, bar: Bar) -> None:
            Probe.window_closes.append(bar.close)

    base = int(pd.Timestamp("2024-03-01 09:31", tz="Asia/Shanghai").value)
    ticks = [
        Tick(
            timestamp=base + i * 60_000_000_000,
            price=10.0 + i,
            volume=1.0,
            symbol=SYMBOL,
        )
        for i in range(11)
    ]
    run_backtest(
        data=ticks,
        freq="1min",
        strategy=Probe,
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        indicator_recorder=cast(Any, sink),
    )

    assert sink.points  # 确认点到了
    # 旧 sink 分不清临时点, 所以干脆不给它: 收到的每个值都必须是某个 5min 窗口的
    # 收盘价(SMA(1) 就是 close), 中途基础 bar 的价格一个都不该出现。
    # 期望取运行时真实的窗口收盘价(含尾部 flush), 不手算 bar 边界。
    window_closes = set(Probe.window_closes)
    assert len(window_closes) >= 2
    received = {p["value"] for p in sink.points}
    assert received == window_closes, (received, window_closes)


# --------------------------------------------------------------------------
# finding 2: 纯 tick 会话(无 bar)下 intrabar 基础指标永不确认, 必须在会话末报错
# --------------------------------------------------------------------------
def test_intrabar_base_indicator_in_pure_tick_session_fails_at_end() -> None:
    """没有 bar 流就没有"确认"; 之前静默地全程只试算, 且绕过了 H/L 覆盖率核验."""
    from akquant.strategy import StrategyConfigurationError

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), name="sma", intrabar=True)

        def on_tick(self, tick: Tick) -> None:
            pass

    base = int(pd.Timestamp("2024-03-01 09:31", tz="Asia/Shanghai").value)
    ticks = [
        Tick(
            timestamp=base + i * 30_000_000_000,
            price=10.0 + i,
            volume=1.0,
            symbol=SYMBOL,
        )
        for i in range(6)
    ]
    with pytest.raises(StrategyConfigurationError, match="intrabar"):
        run_backtest(
            data=ticks,
            strategy=Probe,
            symbols=[SYMBOL],
            initial_cash=1e5,
            show_progress=False,
        )
