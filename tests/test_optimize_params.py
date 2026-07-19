import pytest
from akquant import Strategy
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
