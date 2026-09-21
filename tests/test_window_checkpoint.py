"""多周期窗口序列的 checkpoint 往返与旧存档告警."""

import pickle
import warnings
from pathlib import Path

import pandas as pd
import pytest
from akquant import Strategy, run_backtest
from akquant.akquant import Bar
from akquant.backtest import run_from_checkpoint
from akquant.checkpoint import load_checkpoint, save_checkpoint


def _minute_bars(symbol: str, start: str, n: int) -> list[Bar]:
    idx = pd.date_range(start, periods=n, freq="1min", tz="Asia/Shanghai")
    return [
        Bar(int(ts.value), 10.0 + i, 10.5 + i, 9.5 + i, 10.1 + i, 100.0, symbol)
        for i, ts in enumerate(idx)
    ]


class WindowHistoryStrategy(Strategy):
    """订阅 5 分钟窗口周期, 每根窗口 bar 记录当时的 3 根窗口收盘历史."""

    def __init__(self) -> None:
        """订阅 5min 窗口并初始化窗口历史记录列表."""
        super().__init__()
        self.subscribe_bars("5min")
        self.seen_window_hist: list[list[float]] = []

    def on_start(self) -> None:
        """设置历史深度以便 get_history 能取回窗口序列."""
        self.set_history_depth(50)

    def on_window_bar(self, bar: Bar) -> None:
        """记录当前窗口 bar 派发时刻可见的 3 根窗口收盘历史."""
        arr = self.get_history(3, bar.symbol, "close", freq="5min")
        self.seen_window_hist.append([float(x) for x in arr])


class _PlainStrategy(Strategy):
    """无窗口订阅的策略, 用于验证旧存档告警不误伤无关策略."""


def test_window_history_survives_checkpoint(tmp_path: Path) -> None:
    """窗口历史序列随快照往返; 续跑后 get_history(freq="5min") 能看到阶段 1 的窗口.

    注意 run_backtest 结束时已 flush 尾部窗口(09:36-09:37 → 标签 09:40 入历史), 此时
    保存的聚合器在形成状态为空, 续跑的 09:38-09:40 会重新形成一根同标签 09:40 窗口。
    这是「结束即 flush」与「续跑」叠加的既定行为, 文档写明(Task 8), 测试只断言历史
    跨存档保留, 不断言半根窗口合并。
    """
    ckpt = tmp_path / "w.pkl"
    # 阶段 1: 09:31..09:37 → 09:35 闭合一根(延迟, 在 09:36 派发),
    # 尾部 flush 出标签 09:40
    phase1 = _minute_bars("X", "2024-01-02 09:31:00", 7)
    r1 = run_backtest(
        data=phase1, strategy=WindowHistoryStrategy, symbols="X", show_progress=False
    )
    strat1 = r1.strategy
    assert strat1 is not None
    # 阶段 1 应收到 2 根窗口: 09:35(在 09:36 派发) 与尾部 flush 的 09:40
    n1 = len(strat1.seen_window_hist)
    assert n1 == 2, strat1.seen_window_hist
    save_checkpoint(r1.engine, strat1, str(ckpt))  # type: ignore[arg-type]
    # 阶段 2: 09:38..09:41
    # (参数形式参照 tests/test_strategy_extras.py 里 run_from_checkpoint 的既有调用)
    phase2 = _minute_bars("X", "2024-01-02 09:38:00", 4)
    r2 = run_from_checkpoint(str(ckpt), data=phase2, symbols="X", show_progress=False)
    strat = r2.strategy
    assert strat is not None
    # 策略对象随 pickle 往返, seen_window_hist 里前 n1 条是阶段 1 的; 取续跑后的第一条
    assert len(strat.seen_window_hist) > n1, "续跑必须收到窗口 bar"
    first_resumed = strat.seen_window_hist[n1]
    # 续跑首根窗口(09:38-09:40, 在 09:41 派发)回调里取 3 根历史
    # = [14.1, 16.1, 12.1]: 阶段 1 的 09:35 窗口收盘(14.1)、
    # 阶段 1 尾部 flush 的 09:40 窗口收盘(16.1)、本根(12.1, 阶段 2 的
    # _minute_bars 索引从 0 重新计, 09:38/09:39/09:40 三根收盘为
    # 10.1/11.1/12.1, 本窗口取窗内最后一根 bar 的收盘 = 12.1)
    assert first_resumed == pytest.approx([14.1, 16.1, 12.1]), (
        f"窗口历史未跨 checkpoint 保留: {first_resumed}"
    )


def test_load_checkpoint_warns_for_pre_window_series_snapshot(tmp_path: Path) -> None:
    """旧存档缺 history_window_series 标记: 有窗口订阅的策略恢复告警, 无订阅的不告警."""
    ckpt = tmp_path / "old.pkl"
    r1 = run_backtest(
        data=_minute_bars("X", "2024-01-02 09:31:00", 6),
        strategy=WindowHistoryStrategy,
        symbols="X",
        show_progress=False,
    )
    save_checkpoint(r1.engine, r1.strategy, str(ckpt))  # type: ignore[arg-type]
    with ckpt.open("rb") as fh:
        snap = pickle.load(fh)
    assert snap["snapshot_features"]["history_window_series"] is True
    snap["snapshot_features"].pop("history_window_series")
    with ckpt.open("wb") as fh:
        pickle.dump(snap, fh)
    with pytest.warns(RuntimeWarning, match="window series"):
        load_checkpoint(str(ckpt))
    # 无订阅的策略恢复同一旧存档不告警
    r2 = run_backtest(
        data=_minute_bars("X", "2024-01-02 09:31:00", 3),
        strategy=_PlainStrategy,
        symbols="X",
        show_progress=False,
    )
    plain = tmp_path / "plain.pkl"
    save_checkpoint(r2.engine, r2.strategy, str(plain))  # type: ignore[arg-type]
    with plain.open("rb") as fh:
        snap = pickle.load(fh)
    snap["snapshot_features"].pop("history_window_series")
    with plain.open("wb") as fh:
        pickle.dump(snap, fh)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_checkpoint(str(plain))
