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
