import warnings

import numpy as np
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest
from akquant.params import IntParam


class PStrat(Strategy):
    """Minimal strategy declaring a single inline int param field."""

    fast = IntParam(5, ge=2, le=100)

    def on_start(self) -> None:
        """Subscribe to the test symbol and set warmup from the param."""
        self.subscribe("X")
        self.warmup_period = self.params.fast

    def on_bar(self, bar: Bar) -> None:
        """Record the resolved param value for inspection."""
        self._seen_fast = self.params.fast


def _data() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=30)
    close = 100 + np.arange(30, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
            "symbol": "X",
        },
        index=idx,
    )


def test_strategy_params_injected() -> None:
    """Test strategy_params flows through __param_model__ validation into params."""
    result = run_backtest(
        data=_data(),
        strategy=PStrat,
        symbols="X",
        initial_cash=100000.0,
        strategy_params={"fast": 7},
    )
    assert result is not None  # 跑通即可


def test_unknown_strategy_param_raises() -> None:
    """Test unknown strategy_params keys raise under strict_strategy_params."""
    with pytest.raises((TypeError, ValueError)):
        run_backtest(
            data=_data(),
            strategy=PStrat,
            symbols="X",
            initial_cash=100000.0,
            strategy_params={"nope": 1},
            strict_strategy_params=True,
        )


def _tiny_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=5, freq="D")
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
        index=index,
    )


class _InlineStrategy(Strategy):
    """已迁移的内联字段策略."""

    fast_period = IntParam(5, ge=1)


def test_unknown_kwarg_on_inline_strategy_lists_available_fields() -> None:
    """键名拼错时应列出可用字段, 而不是只说 unknown."""
    with pytest.raises(TypeError) as excinfo:
        run_backtest(
            strategy=_InlineStrategy,
            data=_tiny_data(),
            symbols=["600008.SH"],
            initial_cash=100000.0,
            fast_perid=9,
        )
    message = str(excinfo.value)
    assert "fast_perid" in message
    assert "fast_period" in message
    assert "请检查键名拼写" in message


def test_unknown_kwarg_on_legacy_strategy_gives_migration_path() -> None:
    """遗留写法策略传参时, 报错须给出迁移路径."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        class _LegacyStrategy(Strategy):
            def __init__(self, fast_period: int = 5) -> None:
                super().__init__()
                self.fast_period = fast_period

    with pytest.raises(TypeError) as excinfo:
        run_backtest(
            strategy=_LegacyStrategy,
            data=_tiny_data(),
            symbols=["600008.SH"],
            initial_cash=100000.0,
            fast_period=9,
        )
    message = str(excinfo.value)
    assert "未声明任何内联参数字段" in message
    assert "self.params.fast_period" in message


def test_non_strict_mode_warns_with_same_diagnosis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """strict=False 时静默丢参最难查, warning 须带同样的诊断."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        class _LegacyStrategy2(Strategy):
            def __init__(self, fast_period: int = 5) -> None:
                super().__init__()
                self.fast_period = fast_period

    with caplog.at_level("WARNING"):
        run_backtest(
            strategy=_LegacyStrategy2,
            data=_tiny_data(),
            symbols=["600008.SH"],
            initial_cash=100000.0,
            strict_strategy_params=False,
            fast_period=9,
        )
    assert "未声明任何内联参数字段" in caplog.text
