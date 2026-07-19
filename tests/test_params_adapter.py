import pytest
from akquant import Strategy
from akquant.params import IntParam
from akquant.params_adapter import (
    build_param_grid_from_search_space,
    get_strategy_param_schema,
    resolve_param_model,
    validate_strategy_params,
)
from pydantic import ValidationError


class S(Strategy):
    """带内联 ParamSpec 字段的测试策略."""

    fast = IntParam(10, ge=2, le=200, title="快线")


class NoParam(Strategy):
    """无内联参数字段的测试策略."""


def test_resolve_returns_param_model() -> None:
    """resolve_param_model 应恒返回 cls.__param_model__（含无字段策略）."""
    assert resolve_param_model(S) is S.__param_model__
    assert resolve_param_model(NoParam) is NoParam.__param_model__


def test_schema_from_model() -> None:
    """Schema 应来自参数模型，且保留字段 title."""
    schema = get_strategy_param_schema(S)
    assert "fast" in schema["properties"]
    assert schema["properties"]["fast"]["title"] == "快线"


def test_validate_ok() -> None:
    """合法参数应校验通过并原样返回."""
    assert validate_strategy_params(S, {"fast": 20}) == {"fast": 20}


def test_validate_rejects_unknown() -> None:
    """未知字段应被模型拒绝."""
    with pytest.raises(ValidationError):
        validate_strategy_params(S, {"nope": 1})


def test_validate_rejects_out_of_range() -> None:
    """越界字段应被模型拒绝."""
    with pytest.raises(ValidationError):
        validate_strategy_params(S, {"fast": 1})


def test_no_param_schema_empty() -> None:
    """无字段策略的 schema 应无 properties."""
    schema = get_strategy_param_schema(NoParam)
    assert schema.get("properties", {}) == {}


def test_build_param_grid() -> None:
    """search_space 应原样归一化为 param_grid."""
    grid = build_param_grid_from_search_space({"fast": [5, 10]})
    assert grid == {"fast": [5, 10]}
