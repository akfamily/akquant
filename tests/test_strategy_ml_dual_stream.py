"""双流(on_bar + on_tick 同时触发)下 ML 自动滚动训练不能被静默关闭.

背景: `get_rolling_data()` 固定取 open/high/low/close/volume 五个字段, 而
`get_history`/`get_history_multi` 的既定规则是——同一 symbol 若同时存在 bar
与 tick 两条历史序列(双流), 省略 `freq` 会显式报 `ValueError`, 要求调用方
指定取哪条(不静默选一条)。`strategy_ml.on_train_signal` 此前硬写了不带
`freq` 的 `get_rolling_data()` 调用, 双流下这个 `ValueError` 会被外层
`except Exception` 吞掉降级成一行 WARNING, 训练一次也不会真正发生, 但回测
照常跑完、照常出报告——完全静默。

修复后 `strategy_ml._fetch_rolling_data_for_training` 会在捕获到这个特定的
"双流歧义"错误时自动退回 `freq='bar'` 重试(ML 训练需要真实 OHLCV, tick 序列
没有真实 open/high/low, 双流下只能选 bar); 单流场景(只有 bar 或只有 tick)
不会触发这个 except 分支, 行为与此前完全一致。
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from akquant import BarAggregator, DataFeed, Strategy, Tick, run_backtest
from akquant.ml.model import QuantModel, ValidationConfig

T0 = pd.Timestamp("2024-01-02 09:30:00", tz="Asia/Shanghai").value


def _dual_stream_feed(n_minutes: int = 12) -> DataFeed:
    """构造双流 feed: 每分钟 4 个 tick + 1 根聚合 bar, 价格段可区分.

    分钟 n(从 1 开始)的 4 个 tick 价格为 n*10+0..3, 该分钟 bar 的收盘价为
    n*10+3, 与 tick 价格明显不同, 用于分辨训练用到的到底是 bar 还是 tick。
    """
    feed = DataFeed()
    aggregator = BarAggregator(
        feed, 1, volume_is_cumulative=False, stamp_bar_at_interval_end=True
    )
    for i in range(n_minutes * 4):
        timestamp = T0 + 15_000_000_000 * i
        price = float((i // 4 + 1) * 10 + (i % 4))
        aggregator.on_tick("X", price, 10.0, timestamp)
        feed.add_tick(Tick(timestamp=timestamp, price=price, volume=10.0, symbol="X"))
    return feed


class _RecordingModel(QuantModel):
    """记录每次 fit 调用的样本量与收盘价序列, 供测试断言."""

    fit_calls: int = 0
    fit_closes: list = []

    def __init__(self) -> None:
        """初始化验证配置与录制状态."""
        super().__init__()
        self.validation_config = ValidationConfig(
            train_window=5,
            test_window=2,
            rolling_step=3,
            frequency="1m",
        )

    def clone(self) -> "_RecordingModel":
        """克隆模型副本, 保留验证配置(滚动训练每窗口都会 clone 一次)."""
        cloned = _RecordingModel()
        cloned.validation_config = self.validation_config
        return cloned

    def fit(self, X: Any, y: Any) -> None:
        """记录本次训练的样本量与收盘价, 不做真实拟合."""
        _RecordingModel.fit_calls += 1
        _RecordingModel.fit_closes.append(list(X["close"]))

    def predict(self, X: Any) -> np.ndarray:
        """返回全零占位预测, 满足抽象接口."""
        return np.zeros(len(X))

    def save(self, path: str) -> None:
        """满足抽象接口, 测试不需要落盘."""
        return

    def load(self, path: str) -> None:
        """满足抽象接口, 测试不需要加载."""
        return


class _MLStrategy(Strategy):
    """挂载 `_RecordingModel` 并允许 on_bar/on_tick 同时触发的最小策略."""

    warmup_period = 3

    def __init__(self) -> None:
        """初始化模型, 并在构造期显式声明滚动窗口.

        显式调用 `set_rolling_window` (而不是只靠 `validation_config` 在首个
        bar 才自动推导)是为了让 `_history_depth` 在引擎按容量分配历史缓冲区
        之前就已生效——这是与本次修复无关的既有框架行为, 不显式声明的话,
        引擎会按构造期已知的(更小的)深度分配缓冲区, 训练窗口早期会被 NaN
        左填充, 与本测试要验证的"真实 bar 收盘价"这件事无关, 但会污染断言,
        故这里按推荐用法显式声明以拿到确定、无 NaN 的窗口。
        """
        super().__init__()
        self.model = _RecordingModel()
        self.set_rolling_window(train_window=5, step=3)

    def prepare_features(
        self, df: pd.DataFrame, mode: str = "training"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """用收盘价当特征, 标签占位为 0(测试只关心 fit 有没有被正确调用)."""
        features = pd.DataFrame({"close": df["close"]})
        labels = pd.Series(np.zeros(len(features), dtype=int))
        return features, labels

    def on_bar(self, bar: Any) -> None:
        """双流下 on_bar 必须能触发, 无需交易逻辑."""
        return

    def on_tick(self, tick: Any) -> None:
        """双流下 on_tick 必须能触发, 无需交易逻辑."""
        return


def test_dual_stream_ml_auto_training_actually_fits_with_real_bar_ohlc() -> None:
    """双流下自动滚动训练必须真的调用 fit, 且喂给它的是 bar 收盘价而非 tick 价.

    这是本次修复要证明的核心事实: 此前双流下 `get_rolling_data()` 撞上双流
    歧义 `ValueError`, 被外层 `except Exception` 吞掉降级成 WARNING, `fit`
    一次都不会被调用。修复后应自动退回 `freq='bar'`, `fit` 真的被调用,
    且喂给它的收盘价是 bar 序列(如 13.0/23.0/...), 不是 tick 序列
    (10.0/11.0/12.0/13.0/...)。
    """
    _RecordingModel.fit_calls = 0
    _RecordingModel.fit_closes = []

    run_backtest(
        strategy=_MLStrategy,
        data=_dual_stream_feed(),
        symbols=["X"],
        initial_cash=1e5,
        show_progress=False,
    )

    assert _RecordingModel.fit_calls > 0, (
        "双流下自动滚动训练必须真的触发 fit, 而不是被双流歧义错误静默吞掉"
    )
    # 每次训练喂给模型的收盘价都必须落在 bar 收盘价的取值集合({n*10+3})上,
    # 不能混入 tick 价格(n*10+0/1/2)。
    for closes in _RecordingModel.fit_closes:
        for value in closes:
            assert value % 10 == 3, f"fit 收到了非 bar 收盘价的样本: {value}"


def test_dual_stream_ml_training_fetch_error_is_logged_at_error_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """双流歧义之外的取数/配置类错误必须用 ERROR 级别记录.

    如非法 rolling window: 且要点破"本次训练窗口已跳过", 而不是与训练本身
    失败合用同一条 WARNING——否则用户很容易以为训练正常发生了(这正是本次
    要修的根本原因)。
    """

    class BrokenWindowModel(QuantModel):
        fit_calls = 0

        def clone(self) -> "BrokenWindowModel":
            return BrokenWindowModel()

        def fit(self, X: Any, y: Any) -> None:
            BrokenWindowModel.fit_calls += 1

        def predict(self, X: Any) -> np.ndarray:
            return np.zeros(len(X))

        def save(self, path: str) -> None:
            return

        def load(self, path: str) -> None:
            return

    class BrokenWindowStrategy(Strategy):
        """train_window=0 触发的非法 rolling window 配置错误.

        这是一个与双流歧义完全无关的配置类错误(get_rolling_data 的
        'Invalid rolling window length' ValueError), 用来验证它走的是
        ERROR 分支而不是训练失败的 WARNING 分支。
        """

        warmup_period = 0

        def __init__(self) -> None:
            super().__init__()
            self.model = BrokenWindowModel()
            self.set_rolling_window(train_window=0, step=1)

        def on_bar(self, bar: Any) -> None:
            return

    BrokenWindowModel.fit_calls = 0
    with caplog.at_level("INFO", logger="akquant.ml"):
        run_backtest(
            strategy=BrokenWindowStrategy,
            data=pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2020-01-01", periods=4, freq="min", tz="UTC"
                    ),
                    "open": [1.0, 2.0, 3.0, 4.0],
                    "high": [1.0, 2.0, 3.0, 4.0],
                    "low": [1.0, 2.0, 3.0, 4.0],
                    "close": [1.0, 2.0, 3.0, 4.0],
                    "volume": [100.0] * 4,
                    "symbol": ["Y"] * 4,
                }
            ),
            symbols=["Y"],
            initial_cash=1e5,
            show_progress=False,
        )

    assert BrokenWindowModel.fit_calls == 0
    error_records = [r for r in caplog.records if r.levelno >= 40]
    assert error_records, "取数/配置类错误必须至少有一条 ERROR 级别日志"
    assert any("已跳过" in r.getMessage() for r in error_records)
    # 不应该和"训练本身失败"共用同一条 WARNING 消息模板。
    assert not any("Auto-training failed" in r.getMessage() for r in caplog.records)


def test_dual_stream_ml_training_execution_failure_still_only_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """训练本身失败(如用户模型 fit 抛错)仍保持 WARNING 级降级、不打断回测.

    这项验证"配置/取数类错误"与"训练执行失败"分流后, 后者的既有容错预期
    (回测不因一次训练失败而中断)没有被破坏。
    """

    class RaisingModel(QuantModel):
        def clone(self) -> "RaisingModel":
            return RaisingModel()

        def fit(self, X: Any, y: Any) -> None:
            raise RuntimeError("boom")

        def predict(self, X: Any) -> np.ndarray:
            return np.zeros(len(X))

        def save(self, path: str) -> None:
            return

        def load(self, path: str) -> None:
            return

    class RaisingStrategy(Strategy):
        warmup_period = 2

        def __init__(self) -> None:
            super().__init__()
            self.model = RaisingModel()
            self.set_rolling_window(train_window=2, step=2)

        def prepare_features(
            self, df: pd.DataFrame, mode: str = "training"
        ) -> tuple[pd.DataFrame, pd.Series]:
            """提供最小特征, 让训练流程真正跑到 `model.fit`(才会触发它抛错)."""
            features = pd.DataFrame({"close": df["close"]})
            return features, pd.Series(np.zeros(len(features)))

        def on_bar(self, bar: Any) -> None:
            return

    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=6, freq="min", tz="UTC"),
            "open": np.arange(1, 7, dtype=float),
            "high": np.arange(1, 7, dtype=float),
            "low": np.arange(1, 7, dtype=float),
            "close": np.arange(1, 7, dtype=float),
            "volume": np.full(6, 100.0),
            "symbol": ["Y"] * 6,
        }
    )

    with caplog.at_level("INFO", logger="akquant.ml"):
        result = run_backtest(
            strategy=RaisingStrategy,
            data=data,
            symbols=["Y"],
            initial_cash=1e5,
            show_progress=False,
        )

    assert result is not None
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("Auto-training failed" in r.getMessage() for r in warning_records)
    # 训练执行失败不应该被记成"取数已跳过"的 ERROR 分支。
    assert not any(r.levelno >= 40 for r in caplog.records)
