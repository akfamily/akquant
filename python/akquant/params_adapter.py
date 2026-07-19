"""策略参数适配层：连接内联参数模型、前端 schema 与回测入口."""

from typing import Any, Mapping, Sequence, cast

from .params import ParamModel, model_to_schema, to_runtime_kwargs, validate_payload
from .strategy import Strategy


def resolve_param_model(strategy_cls: type[Strategy]) -> type[ParamModel]:
    """
    解析策略的参数模型.

    :param strategy_cls: 策略类
    :return: ParamModel 子类（恒非空；无内联字段时为空模型）
    """
    return cast("type[ParamModel]", strategy_cls.__param_model__)


def get_strategy_param_schema(strategy_cls: type[Strategy]) -> dict[str, Any]:
    """
    获取策略参数 schema.

    :param strategy_cls: 策略类
    :return: 参数 schema
    """
    return model_to_schema(resolve_param_model(strategy_cls))


def validate_strategy_params(
    strategy_cls: type[Strategy], payload: Mapping[str, Any]
) -> dict[str, Any]:
    """
    校验策略参数.

    :param strategy_cls: 策略类
    :param payload: 待校验参数
    :return: 可直接注入 strategy_params 的参数字典
    """
    model = validate_payload(resolve_param_model(strategy_cls), payload)
    return cast(dict[str, Any], model.model_dump())


def extract_runtime_kwargs(
    strategy_cls: type[Strategy], payload: Mapping[str, Any]
) -> dict[str, Any]:
    """
    提取运行时参数.

    :param strategy_cls: 策略类
    :param payload: 待校验参数
    :return: 可透传 run_backtest 的 runtime kwargs
    """
    model = validate_payload(resolve_param_model(strategy_cls), payload)
    return to_runtime_kwargs(model)


def build_param_grid_from_search_space(
    search_space: Mapping[str, Sequence[Any]],
) -> dict[str, list[Any]]:
    """
    将上层 search space 归一化为 param_grid.

    :param search_space: 搜索空间
    :return: param_grid
    :raises TypeError: 值不是序列
    :raises ValueError: 候选为空
    """
    grid: dict[str, list[Any]] = {}
    for key, values in search_space.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"search_space[{key}] must be a sequence")
        candidate_values = list(values)
        if not candidate_values:
            raise ValueError(f"search_space[{key}] cannot be empty")
        grid[str(key)] = candidate_values
    return grid


__all__ = [
    "resolve_param_model",
    "get_strategy_param_schema",
    "validate_strategy_params",
    "extract_runtime_kwargs",
    "build_param_grid_from_search_space",
]
