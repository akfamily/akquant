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
    """未知字段应被拒绝, 且诊断可执行(不再是裸 pydantic extra_forbidden).

    异常类型自本次起为 ``TypeError``, 与 ``run_backtest`` / ``run_grid_search``
    对未知参数的报错保持一致; 越界值仍走 ``ValidationError``(见下一个用例)。
    """
    with pytest.raises(TypeError) as excinfo:
        validate_strategy_params(S, {"nope": 1})
    assert "nope" in str(excinfo.value)


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


def test_validate_strategy_params_unknown_key_on_legacy_gives_migration_path() -> None:
    """遗留写法策略传参时须给出迁移路径, 而非裸 pydantic extra_forbidden.

    这是文档推荐的「页面化参数输入」入口(见 docs/zh/guide/examples.md 的流程图),
    前端在这里拿到的报错必须和 run_backtest / run_grid_search 一样可执行。
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        class LegacyAdapterStrategy(Strategy):
            def __init__(self, fast_period: int = 5) -> None:
                super().__init__()
                self.fast_period = fast_period

    with pytest.raises(TypeError) as excinfo:
        validate_strategy_params(LegacyAdapterStrategy, {"fast_period": 9})
    message = str(excinfo.value)
    assert "未声明任何内联参数字段" in message
    assert "self.params.fast_period" in message


def test_validate_strategy_params_typo_lists_available_fields() -> None:
    """已迁移策略拼错键名时应列出可用字段."""
    with pytest.raises(TypeError) as excinfo:
        validate_strategy_params(S, {"fsat": 20})
    message = str(excinfo.value)
    assert "fsat" in message
    assert "fast" in message
    assert "请检查键名拼写" in message


def test_validate_strategy_params_out_of_range_still_raises_validation_error() -> None:
    """越界值必须仍抛 ValidationError —— optimize 的逐候选值校验依赖该行为."""
    with pytest.raises(ValidationError):
        validate_strategy_params(S, {"fast": 999})


def test_extract_runtime_kwargs_unknown_key_gives_migration_path() -> None:
    """extract_runtime_kwargs 与 validate_strategy_params 同源, 诊断须一致."""
    from akquant.params_adapter import extract_runtime_kwargs

    with pytest.raises(TypeError) as excinfo:
        extract_runtime_kwargs(S, {"fsat": 20})
    assert "请检查键名拼写" in str(excinfo.value)
