"""多标的下 warmup_period 门槛必须按 symbol 独立计数.

根因: strategy_events.py 里驱动 warmup 门槛的 ``strategy._bar_count`` 是跨
symbol 的全局计数器, 但历史缓冲区是 per-symbol 的。M 个 symbol 交替产生 bar
事件时, 每个 symbol 实际只攒到约 ``ceil(warmup_period / M)`` 根真实历史就被
放行, 官方示例里 ``warmup_period = long_window`` 这类写法在多标的下全部失效
(参见 docs/zh/guide/examples.md、strategy.md、quant_basics.md)。

这里覆盖三个场景:
1. 多标的且各标的数据条数相同: 首次 on_bar 时该标的历史必须真正攒够
   warmup_period 根, 且值与从数据序列独立推导的结果精确匹配(不是拿程序
   输出反推)。
2. 单标的: 原本正确的行为不能变(回归底线)。
3. 标的上市时间不同(数据条数不等): 更极端的场景 —— 后来者不能"蹭"到先行
   者攒下的全局计数, 必须等自己攒够。
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import cast

import pandas as pd
from akquant import Bar, Strategy, run_backtest, run_from_checkpoint, save_checkpoint


def _df(closes: list[float], start: str, periods: int, symbol: str) -> pd.DataFrame:
    """构造单标的日线 DataFrame, 时间戳与 close 一一对应."""
    dates = pd.date_range(start, periods=periods, freq="D", tz="Asia/Shanghai")
    assert len(closes) == periods
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * periods,
            "symbol": [symbol] * periods,
        }
    )


class _FirstBarHistoryCollector(Strategy):
    """每个 symbol 首次触发 on_bar 时, 记录当时的历史窗口(count=warmup_period)."""

    warmup_period = 3

    def on_start(self) -> None:
        """重置采集状态."""
        self.seen: set[str] = set()
        self.captured: dict[str, list[float]] = {}

    def on_bar(self, bar: Bar) -> None:
        """每个 symbol 首次收到 bar 时取一次历史(不覆盖)."""
        if bar.symbol in self.seen:
            return
        self.seen.add(bar.symbol)
        self.captured[bar.symbol] = [
            float(x)
            for x in self.get_history(
                count=self.warmup_period, symbol=bar.symbol, field="close"
            )
        ]


def test_warmup_period_reaches_full_depth_per_symbol_with_multiple_symbols() -> None:
    """两个标的、条数相同、时间戳对齐: 各自应在攒够 3 根真实历史后才首次触发.

    独立推导的期望值: X 的前 3 根收盘价是 [10, 11, 12], Y 的前 3 根是
    [100, 101, 102]。修复前, 全局计数器在两个 symbol 交替下 3 次 bar 事件
    (即 X 的第 2 根、随后 Y 的第 2 根) 就会跨过门槛, 此时历史里会混入 nan
    占位, 且值不精确匹配 [10, 11, 12] / [100, 101, 102]。
    """
    x_closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    y_closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    data = {
        "X": _df(x_closes, "2024-01-01", 10, "X"),
        "Y": _df(y_closes, "2024-01-01", 10, "Y"),
    }

    result = run_backtest(
        data=data,
        strategy=_FirstBarHistoryCollector,
        symbols=["X", "Y"],
        initial_cash=1e5,
        show_progress=False,
    )
    strategy = cast(_FirstBarHistoryCollector, result.strategy)

    assert strategy.captured["X"] == [10.0, 11.0, 12.0]
    assert strategy.captured["Y"] == [100.0, 101.0, 102.0]
    for values in strategy.captured.values():
        assert all(not math.isnan(v) for v in values)


def test_warmup_period_single_symbol_unaffected() -> None:
    """单标的场景下, 修复前后行为必须一致(回归底线)."""
    x_closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    data = {"X": _df(x_closes, "2024-01-01", 10, "X")}

    result = run_backtest(
        data=data,
        strategy=_FirstBarHistoryCollector,
        symbols=["X"],
        initial_cash=1e5,
        show_progress=False,
    )
    strategy = cast(_FirstBarHistoryCollector, result.strategy)

    assert strategy.captured["X"] == [10.0, 11.0, 12.0]


def test_warmup_period_per_symbol_independent_with_staggered_listing() -> None:
    """标的上市时间不同(数据条数不等)时, 后来者不能蹭到先行者攒下的全局计数.

    X 从第 1 天开始有 10 根 bar; Y 从第 6 天才开始上市, 只有 5 根 bar。修复前
    全局计数器在 Y 上市前就已被 X 推过 warmup_period=3 的门槛, Y 的第 1 根
    bar 就会直接触发 on_bar, 历史几乎全是 nan。独立推导的期望: Y 必须等到
    自己攒够 3 根(第 6、7、8 天), 历史精确等于 [200, 201, 202]。
    """
    x_closes = [10.0 + i for i in range(10)]  # 10..19, day1..day10
    y_closes = [200.0, 201.0, 202.0, 203.0, 204.0]  # day6..day10

    data = {
        "X": _df(x_closes, "2024-01-01", 10, "X"),
        "Y": _df(y_closes, "2024-01-06", 5, "Y"),
    }

    result = run_backtest(
        data=data,
        strategy=_FirstBarHistoryCollector,
        symbols=["X", "Y"],
        initial_cash=1e5,
        show_progress=False,
    )
    strategy = cast(_FirstBarHistoryCollector, result.strategy)

    assert strategy.captured["Y"] == [200.0, 201.0, 202.0]
    assert all(not math.isnan(v) for v in strategy.captured["Y"])


# ---------------------------------------------------------------------------
# 旧存档(缺 ``_symbol_bar_counts``)热启动: 不得重放已完成的 warmup, 且真正
# 处于预热期中途的旧存档必须从实际恢复深度继续、而不是无条件放行。
#
# ``_symbol_bar_counts`` 是随 per-symbol warmup 计数一起引入的新字段。
# ``Strategy.__new__`` 会用空 ``defaultdict(int)`` 预填充它, 若不特殊处理,
# 早于该字段存在的旧存档恢复后该 symbol 会被当成"从未见过", 从 0 重新计数,
# 把已经攒够、Rust 历史缓冲区里其实已经完整恢复的 warmup 期又重放一遍。
# ---------------------------------------------------------------------------


class _WarmupGateObserver(Strategy):
    """记录实际触发 on_bar 的 symbol 序列, 用于判断 warmup 门槛是否被重放."""

    warmup_period = 4

    def on_start(self) -> None:
        """每次(冷启动/热启动后)重置观测列表."""
        self.on_bar_symbols: list[str] = []

    def on_bar(self, bar: Bar) -> None:
        """记录本次实际放行的 on_bar 调用."""
        self.on_bar_symbols.append(bar.symbol)


def _strip_symbol_bar_counts(checkpoint_path: Path) -> None:
    """就地改写 checkpoint, 移除 ``_symbol_bar_counts`` 字段模拟旧存档.

    直接 ``del`` 整个 key(而不是清空成空 dict), 才能让反序列化时
    ``"_symbol_bar_counts" not in state`` 成立——这正是
    ``Strategy.__setstate__`` 用来判断"这是一份早于该字段存在的旧存档"的
    条件, 也是本测试要模拟的真实场景。
    """
    with checkpoint_path.open("rb") as fh:
        snapshot = pickle.load(fh)
    strategy_state = snapshot["strategy"].__dict__
    assert "_symbol_bar_counts" in strategy_state
    del strategy_state["_symbol_bar_counts"]
    with checkpoint_path.open("wb") as fh:
        pickle.dump(snapshot, fh)


def test_legacy_checkpoint_without_symbol_bar_counts_skips_replay_single_symbol(
    tmp_path: Path,
) -> None:
    """单标的: warmup 在存档前已完整攒够, 旧存档恢复后不应重放.

    warmup_period=4, phase1 恰好 4 根 bar(存档时该标的历史已攒满、on_bar
    在 phase1 第 4 根就已触发过一次)。旧存档缺 ``_symbol_bar_counts``:
    若从 0 重新计数(未修复前的行为), phase2 的前几根 bar 会被再次挡在
    warmup 门槛之外; 修复后应查询 Rust 已恢复的真实历史深度, 判定该标的
    早已攒满, phase2 的每一根 bar 都应立即触发 on_bar。
    """
    checkpoint = tmp_path / "legacy_single.pkl"
    phase1 = _df([10.0, 11.0, 12.0, 13.0], "2024-01-01", 4, "X")
    phase2 = _df([14.0, 15.0, 16.0], "2024-01-05", 3, "X")

    result1 = run_backtest(
        data={"X": phase1},
        strategy=_WarmupGateObserver,
        symbols=["X"],
        initial_cash=1e5,
        show_progress=False,
    )
    strategy1 = cast(_WarmupGateObserver, result1.strategy)
    assert strategy1.on_bar_symbols == ["X"]  # phase1 内仅第 4 根触发过一次

    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]
    _strip_symbol_bar_counts(checkpoint)

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data={"X": phase2},
        symbols=["X"],
        show_progress=False,
    )
    strategy2 = cast(_WarmupGateObserver, result2.strategy)

    assert strategy2.on_bar_symbols == ["X", "X", "X"]


def test_legacy_checkpoint_without_symbol_bar_counts_skips_replay_multi_symbol(
    tmp_path: Path,
) -> None:
    """多标的: 两个标的 warmup 均已在存档前攒够, 旧存档恢复后都不应重放."""
    checkpoint = tmp_path / "legacy_multi.pkl"
    phase1 = {
        "X": _df([10.0, 11.0, 12.0, 13.0], "2024-01-01", 4, "X"),
        "Y": _df([100.0, 101.0, 102.0, 103.0], "2024-01-01", 4, "Y"),
    }
    phase2 = {
        "X": _df([14.0, 15.0, 16.0], "2024-01-05", 3, "X"),
        "Y": _df([104.0, 105.0, 106.0], "2024-01-05", 3, "Y"),
    }

    result1 = run_backtest(
        data=phase1,
        strategy=_WarmupGateObserver,
        symbols=["X", "Y"],
        initial_cash=1e5,
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]
    _strip_symbol_bar_counts(checkpoint)

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols=["X", "Y"],
        show_progress=False,
    )
    strategy2 = cast(_WarmupGateObserver, result2.strategy)

    assert strategy2.on_bar_symbols.count("X") == 3
    assert strategy2.on_bar_symbols.count("Y") == 3
    assert len(strategy2.on_bar_symbols) == 6


def test_legacy_checkpoint_without_symbol_bar_counts_resumes_partial_warmup(
    tmp_path: Path,
) -> None:
    """存档时真实处于预热期中途: 旧存档必须按实际恢复深度继续预热, 而不是无条件放行.

    warmup_period=4, phase1 只有 2 根 bar(存档时该标的历史确实不足,
    on_bar 一次都还没触发过)。旧存档缺 ``_symbol_bar_counts``, 若直接
    "查不到就当作已攒满"无条件放行, phase2 第一根就会立刻触发 —— 这正是
    行为要求里明确禁止的 (b) 场景。修复后应查询到真实深度只有 2, 需要
    phase2 再攒 2 根(共 4 根)才满足门槛: phase2 第 1 根仍应被挡住, 第
    2~5 根才应触发。
    """
    checkpoint = tmp_path / "legacy_partial.pkl"
    phase1 = _df([10.0, 11.0], "2024-01-01", 2, "X")
    phase2 = _df([12.0, 13.0, 14.0, 15.0, 16.0], "2024-01-03", 5, "X")

    result1 = run_backtest(
        data={"X": phase1},
        strategy=_WarmupGateObserver,
        symbols=["X"],
        initial_cash=1e5,
        show_progress=False,
    )
    strategy1 = cast(_WarmupGateObserver, result1.strategy)
    assert strategy1.on_bar_symbols == []  # 存档时确实还没攒够, 一次都没触发过

    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]
    _strip_symbol_bar_counts(checkpoint)

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data={"X": phase2},
        symbols=["X"],
        show_progress=False,
    )
    strategy2 = cast(_WarmupGateObserver, result2.strategy)

    # 第 1 根仍在预热(2 旧 + 1 新 = 3 < 4), 第 2~5 根(累计 >= 4)开始触发。
    assert strategy2.on_bar_symbols == ["X", "X", "X", "X"]
