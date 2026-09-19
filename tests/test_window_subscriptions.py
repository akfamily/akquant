"""引擎原生多周期: subscribe_bars / on_window_bar / get_history(freq=) / 指标按周期."""

import pandas as pd
import pytest
from akquant import Strategy, run_backtest
from akquant.akquant import Bar


def test_bar_freq_defaults_to_none_and_is_settable() -> None:
    """Bar.freq 默认 None, 可读写, 构造函数支持 freq 关键字参数."""
    bar = Bar(1_700_000_000_000_000_000, 1.0, 2.0, 0.5, 1.5, 100.0, "X")
    assert bar.freq is None
    bar.freq = "5min"
    assert bar.freq == "5min"
    tagged = Bar(1, 1.0, 1.0, 1.0, 1.0, 1.0, "X", freq="1d")
    assert tagged.freq == "1d"
    assert "freq=1d" in repr(tagged)


def _minute_bars(symbol: str, start: str, n: int) -> list[Bar]:
    idx = pd.date_range(start, periods=n, freq="1min", tz="Asia/Shanghai")
    return [
        Bar(int(ts.value), 10.0 + i, 10.5 + i, 9.5 + i, 10.1 + i, 100.0, symbol)
        for i, ts in enumerate(idx)
    ]


class _EngineDispatchProbe(Strategy):
    """绕过 Task 5 的 Python API, 直接验证引擎会调 _on_window_bar_event_and_flush."""

    def __init__(self) -> None:
        super().__init__()
        self.window_calls: list[
            tuple[int, str, int]
        ] = []  # (ts, freq, 触发时基础 bar 数)
        self.base_seen = 0

    def on_bar(self, bar: Bar) -> None:
        self.base_seen += 1

    def _on_window_bar_event_and_flush(self, bar: Bar, ctx: object) -> None:
        self.window_calls.append((int(bar.timestamp), str(bar.freq), self.base_seen))
        return None


def test_engine_dispatches_closed_window_bars_after_base_on_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """绕过 Task 5 的 Python API, 直接验证引擎会同步派发闭合的窗口 bar.

    Task 5 之前 Python 侧还没有 subscribe_bars, 用工厂替换 engine.py 里的 Engine 类,
    在引擎创建后、run() 之前直接配置订阅。不能在 on_start 里配置: run() 期间再入
    Engine pymethod 会 'Already borrowed'。
    """
    import akquant.backtest.engine as engine_module

    real_engine_cls = engine_module.Engine

    def _factory() -> object:
        eng = real_engine_cls()
        eng.configure_window_subscriptions([(None, "5min", None)], 1)
        return eng

    monkeypatch.setattr(engine_module, "Engine", _factory)
    bars = _minute_bars("X", "2024-01-02 09:31:00", 7)  # 09:31..09:37
    strategy = _EngineDispatchProbe()
    run_backtest(data=bars, strategy=strategy, symbols="X", show_progress=False)
    # 09:35 闭合一根(基础周期已知 → 即时), 尾部 09:36-09:37 在结束 flush 时闭合一根
    assert [c[1] for c in strategy.window_calls] == ["5min", "5min"]
    assert strategy.window_calls[0][0] == int(
        pd.Timestamp("2024-01-02 09:35", tz="Asia/Shanghai").value
    )
    assert strategy.window_calls[0][2] == 5, "应在第 5 根基础 bar 的 on_bar 之后派发"
    assert strategy.window_calls[1][2] == 7, "尾部窗口在全部 bar 处理完后 flush"
