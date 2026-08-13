import warnings

import pandas as pd
import pytest
from akquant import Strategy, run_grid_search
from akquant.optimize import _validate_strategy_param_grid_keys
from akquant.params import IntParam


class OStrat(Strategy):
    """带内联 ParamSpec 字段的测试策略."""

    fast = IntParam(10, ge=2, le=100)


def test_valid_grid_passes() -> None:
    """合法网格应通过校验."""
    _validate_strategy_param_grid_keys(OStrat, {"fast": [5, 10, 20]})


def test_unknown_key_rejected() -> None:
    """未知键应被拒绝."""
    with pytest.raises((TypeError, ValueError)):
        _validate_strategy_param_grid_keys(OStrat, {"nope": [1, 2]})


def test_out_of_range_value_rejected() -> None:
    """越界候选值应被拒绝."""
    with pytest.raises((TypeError, ValueError)):
        _validate_strategy_param_grid_keys(OStrat, {"fast": [1, 200]})  # 越界


def test_mixed_valid_and_unknown_keys_rejected_for_unknown_only() -> None:
    """合法字段键(fast)与非法键(nope)混合时，应仅因非法键被拒绝.

    区别于未声明任何字段的策略(如 StrictParamStrategy)——那种情况下任意键都会
    被判定为非法；这里 fast 是 OStrat 实际声明的字段，报错信息应只点名 nope，
    从而证明校验确有"合法 vs 非法"区分度，而非对任意 kwargs 一律拒绝。
    """
    with pytest.raises(
        TypeError, match=r"Unknown strategy param\(s\) in param_grid: nope"
    ):
        _validate_strategy_param_grid_keys(OStrat, {"fast": [5, 10], "nope": [1, 2]})


def _tiny_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=5, freq="D")
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
        index=index,
    )


class _GridInlineStrategy(Strategy):
    """已迁移的内联字段策略."""

    fast_period = IntParam(5, ge=1)


def test_param_grid_unknown_key_lists_available_fields() -> None:
    """param_grid 键名拼错应列出可用字段."""
    with pytest.raises(TypeError) as excinfo:
        run_grid_search(
            strategy=_GridInlineStrategy,
            param_grid={"fast_perid": [5, 10]},
            data=_tiny_data(),
            symbols=["600008.SH"],
            initial_cash=100000.0,
            max_workers=1,
            show_progress=False,
        )
    message = str(excinfo.value)
    assert "param_grid" in message
    assert "fast_period" in message
    assert "请检查键名拼写" in message


def test_param_grid_on_partial_migration_strategy_gives_migration_path() -> None:
    """半迁移态策略的 param_grid 键命中未迁移的 __init__ 参数, 应走迁移分支.

    _GridMixedStrategy 已声明 fast 字段, 但 __init__ 里的 slow 忘了迁——网格键
    slow 应诊断为"未迁移", 而不是被当成拼写错误去列可用字段名。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        class _GridMixedStrategy(Strategy):
            fast = IntParam(5, ge=1)

            def __init__(self, slow: int = 20) -> None:
                super().__init__()
                self.slow = slow

    with pytest.raises(TypeError) as excinfo:
        run_grid_search(
            strategy=_GridMixedStrategy,
            param_grid={"slow": [10, 20]},
            data=_tiny_data(),
            symbols=["600008.SH"],
            initial_cash=100000.0,
            max_workers=1,
            show_progress=False,
        )
    message = str(excinfo.value)
    assert "param_grid" in message
    assert "slow" in message
    assert "__init__" in message
    assert "请检查键名拼写" not in message


def test_param_grid_on_legacy_strategy_gives_migration_path() -> None:
    """遗留写法策略做参数优化时, 报错须给出迁移路径."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        class _GridLegacyStrategy(Strategy):
            def __init__(self, fast_period: int = 5) -> None:
                super().__init__()
                self.fast_period = fast_period

    with pytest.raises(TypeError) as excinfo:
        run_grid_search(
            strategy=_GridLegacyStrategy,
            param_grid={"fast_period": [5, 10]},
            data=_tiny_data(),
            symbols=["600008.SH"],
            initial_cash=100000.0,
            max_workers=1,
            show_progress=False,
        )
    message = str(excinfo.value)
    assert "param_grid" in message
    assert "未声明任何内联参数字段" in message
    assert "self.params.fast_period" in message
