"""set_rolling_window 不带 validation_config 时, 训练触发必须是阈值语义而非取模.

根因: ``strategy_ml.should_trigger_training`` 在没有 ``validation_config`` 的
分支此前用 ``_bar_count % _rolling_step == 0`` 判断是否触发训练。这对"任何
一次被跳过的 bar 事件"都不健壮——per-symbol warmup 门槛(见
``tests/test_warmup_period_per_symbol.py``)会让某个 symbol 的部分 bar 事件
在 warmup 完成前直接 return, 不经过 ``should_trigger_training``。若全局
``_bar_count`` 恰好在被跳过的那次到达 ``_rolling_step`` 的倍数, 下一次事件
``_bar_count`` 已经往前走了一格, 模值再也凑不回 0, 这次训练**永久丢失**而
非延后。复现: 3 个标的各 10 根 bar, ``warmup_period=8``——``step=20`` 时
``model.fit`` 全程一次都不会被调用, 回测在没有任何模型的情况下悄悄跑完。

修复把判断改成阈值语义: 新增 ``_last_train_bar_count`` 状态(初值 0), 触发
条件是 ``_bar_count - _last_train_bar_count >= _rolling_step``, 在
``consume_training_trigger`` 消费触发时推进为当前 ``_bar_count``。跳过的
bar 只会让触发顺延到下一根满足条件的 bar, 不会丢。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from akquant import Strategy, run_backtest
from akquant.ml.model import QuantModel


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


class _CountingModel(QuantModel):
    """不带 validation_config 的最小模型, 只记录 fit 调用次数."""

    def __init__(self) -> None:
        """初始化, 显式保持 validation_config=None(命中取模/阈值分支)."""
        super().__init__()
        self.fit_calls = 0

    def clone(self) -> "_CountingModel":
        """不会被调用(无 validation_config 时训练生命周期不 clone), 保留以满足接口."""
        cloned = _CountingModel()
        cloned.fit_calls = self.fit_calls
        return cloned

    def fit(self, X: Any, y: Any) -> None:
        """记录一次训练发生, 不做真实拟合."""
        self.fit_calls += 1

    def predict(self, X: Any) -> np.ndarray:
        """返回全零占位预测, 满足抽象接口."""
        return np.zeros(len(X))

    def save(self, path: str) -> None:
        """满足抽象接口, 测试不需要落盘."""
        return

    def load(self, path: str) -> None:
        """满足抽象接口, 测试不需要加载."""
        return


class _NoValidationConfigMLStrategy(Strategy):
    """挂载 ``_CountingModel`` 并调用 ``set_rolling_window``(不经 validation_config).

    有参 ``__init__`` 在当前版本不再是外部参数入口(见 ``params.py`` 的迁移提示),
    因此这里改用类属性 + 零参构造函数的写法, 每种场景用一个子类覆盖
    ``warmup_period`` / ``train_window`` / ``step`` 三个类属性。
    """

    warmup_period = 0
    train_window = 1
    step = 1

    def __init__(self) -> None:
        """挂载模型与滚动训练窗口, 并记录每次训练触发时的全局 bar 计数."""
        super().__init__()
        self.model = _CountingModel()
        self.set_rolling_window(train_window=self.train_window, step=self.step)
        self.train_bar_counts: list[int] = []

    def prepare_features(
        self, df: pd.DataFrame, mode: str = "training"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """用收盘价当特征, 标签占位为 0(测试只关心 fit 有没有被正确调用)."""
        features = pd.DataFrame({"close": df["close"]})
        labels = pd.Series(np.zeros(len(features), dtype=int))
        return features, labels

    def on_bar(self, bar: Any) -> None:
        """无需交易逻辑."""
        return

    def on_train_signal(self, strategy: Any) -> None:
        """记录本次训练触发时的全局 bar 计数, 并调用默认实现真正执行 model.fit."""
        self.train_bar_counts.append(int(self._bar_count))
        super().on_train_signal(strategy)


def _three_symbol_ten_bar_data() -> dict[str, pd.DataFrame]:
    """3 个标的、各 10 根对齐 bar 的复现数据集(reviewer 原始复现规模)."""
    return {
        "A": _df([10.0 + i for i in range(10)], "2024-01-01", 10, "A"),
        "B": _df([100.0 + i for i in range(10)], "2024-01-01", 10, "B"),
        "C": _df([1000.0 + i for i in range(10)], "2024-01-01", 10, "C"),
    }


class _Step20Strategy(_NoValidationConfigMLStrategy):
    """3 标的 x 10 bar 复现规模, warmup_period=8, step=20(修复前零训练场景)."""

    warmup_period = 8
    train_window = 5
    step = 20


class _Step5Strategy(_NoValidationConfigMLStrategy):
    """同一复现规模, step=5."""

    warmup_period = 8
    train_window = 5
    step = 5


class _SingleSymbolStep3Strategy(_NoValidationConfigMLStrategy):
    """单标的回归场景, warmup_period=3, step=3."""

    warmup_period = 3
    train_window = 3
    step = 3


class _StateAdvanceGuardStrategy(_NoValidationConfigMLStrategy):
    """守护 consume_training_trigger 状态推进的场景, warmup_period=2, step=3."""

    warmup_period = 2
    train_window = 2
    step = 3


def test_multi_symbol_rolling_step_20_now_trains_at_least_once() -> None:
    """critical: step=20 此前一次都不训练(warmup 跳过的那次恰好撞上取模边界).

    3 标的 x 10 bar, warmup_period=8: 全局 bar 事件共 30 次, 三个标的各自的
    warmup 完成点(第 8 根自有 bar)分散在全局第 22/23/24 次事件, 之后才可能
    触发训练。修复前, 取模判断在 warmup 门槛内被跳过的那次 _bar_count 撞上
    20 的倍数后永久错过, ``model.fit`` 全程零调用; 修复后阈值语义保证触发
    只会顺延、不会丢——至少应有一次训练发生。
    """
    result = run_backtest(
        data=_three_symbol_ten_bar_data(),
        strategy=_Step20Strategy,
        symbols=["A", "B", "C"],
        initial_cash=1e5,
        show_progress=False,
    )
    fitted_strategy = result.strategy
    assert fitted_strategy is not None
    assert fitted_strategy.model.fit_calls == 1
    assert len(fitted_strategy.train_bar_counts) == 1


def test_multi_symbol_rolling_step_5_trains_expected_number_of_times() -> None:
    """step=5 场景: 修复后仍应正常触发训练(非零), 且触发间隔满足阈值语义.

    该数据形状下修复前后触发**次数**恰好都是 2(取模与阈值在这组具体跳过点上
    巧合地给出相同的触发计数), 但触发发生的全局 bar 计数点不同——本测试不
    断言具体触发点(那是内部实现细节), 只断言：训练确实发生、且任意两次
    触发之间的全局 bar 计数间隔不小于 step(阈值语义的核心不变式，也是
    ``consume_training_trigger`` 必须正确推进 ``_last_train_bar_count`` 的
    直接证据——若状态没有推进, 触发会挤在一起而不是保持 >= step 的间隔)。
    """
    result = run_backtest(
        data=_three_symbol_ten_bar_data(),
        strategy=_Step5Strategy,
        symbols=["A", "B", "C"],
        initial_cash=1e5,
        show_progress=False,
    )
    fitted_strategy = result.strategy
    assert fitted_strategy is not None
    assert fitted_strategy.model.fit_calls == 2
    assert fitted_strategy.model.fit_calls == len(fitted_strategy.train_bar_counts)
    for earlier, later in zip(
        fitted_strategy.train_bar_counts, fitted_strategy.train_bar_counts[1:]
    ):
        assert later - earlier >= 5


def test_single_symbol_rolling_training_trigger_count_unchanged() -> None:
    """单标的回归底线: 触发次数与修复前完全一致(单标的下无 warmup 跳过, 阈值与取模等价).

    10 根 bar, warmup_period=3, step=3: 全局 bar 计数与 per-symbol 计数完全
    重合(单标的没有交替), 取模与阈值化在没有任何跳过的情况下首次触发时机
    完全一致, 之后每隔 step 根必然再次触发——不应因本次修复产生任何变化。
    """
    closes = [10.0 + i for i in range(10)]
    result = run_backtest(
        data={"X": _df(closes, "2024-01-01", 10, "X")},
        strategy=_SingleSymbolStep3Strategy,
        symbols=["X"],
        initial_cash=1e5,
        show_progress=False,
    )
    fitted_strategy = result.strategy
    assert fitted_strategy is not None
    assert fitted_strategy.train_bar_counts == [3, 6, 9]
    assert fitted_strategy.model.fit_calls == 3


def test_consume_training_trigger_advances_state_and_does_not_fire_every_bar() -> None:
    """守护 consume_training_trigger 的早退坑: 状态不推进会导致之后每根 bar 都再次触发.

    单标的, warmup_period=2, step=3, 12 根 bar。若 ``_last_train_bar_count``
    在无 ``validation_config`` 分支里没有被正确更新(停留在 0), 阈值判断
    ``_bar_count - 0 >= step`` 一旦满足就会永远为真, 首次触发之后**每一根**
    后续 bar 都会再次触发训练——本测试直接断言这不会发生: 训练次数远少于
    参与训练判断的 bar 数, 且相邻两次触发之间的间隔严格等于 step。
    """
    closes = [10.0 + i for i in range(12)]
    result = run_backtest(
        data={"X": _df(closes, "2024-01-01", 12, "X")},
        strategy=_StateAdvanceGuardStrategy,
        symbols=["X"],
        initial_cash=1e5,
        show_progress=False,
    )
    fitted_strategy = result.strategy
    assert fitted_strategy is not None
    # 12 根 bar、warmup 之后仍有 10 根参与判断; 若状态不推进, fit_calls 会
    # 逼近这个数量级(每根都触发)。阈值语义下应恰好是每 3 根一次(bar=3/6/9/12)。
    assert fitted_strategy.model.fit_calls == 4
    assert fitted_strategy.train_bar_counts == [3, 6, 9, 12]
