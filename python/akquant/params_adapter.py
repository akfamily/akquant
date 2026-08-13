"""策略参数适配层：连接内联参数模型、前端 schema 与回测入口."""

from typing import Any, Mapping, Sequence, cast

from .params import (
    ParamModel,
    ValidationError,
    model_to_schema,
    to_runtime_kwargs,
    unknown_param_message,
    validate_payload,
)
from .strategy import Strategy, _legacy_init_arg_names


def _reraise_unknown_params_with_diagnosis(
    strategy_cls: type[Strategy], error: ValidationError
) -> None:
    """把「纯未知键」的校验失败换成带迁移指引的 ``TypeError``.

    本模块是文档推荐的「页面化参数输入」入口（见 docs/zh/guide/examples.md 的
    参数分流流程图），前端在这里拿到的诊断必须与 ``run_backtest`` /
    ``run_grid_search`` 一致；裸 pydantic 的 ``extra_forbidden`` 只说
    "Extra inputs are not permitted"，既不指出该策略有哪些可用字段，也不告诉
    遗留写法的用户该怎么迁移。

    只在**全部** error 都是 ``extra_forbidden`` 时改写。越界（``ge``/``le``）、
    类型不符等其它成因原样交回 —— ``optimize._validate_strategy_param_grid_keys``
    的逐候选值校验依赖那条 ``ValidationError`` 路径来报越界。

    :param strategy_cls: 被校验的策略类
    :param error: pydantic 抛出的原始校验错误
    :raises TypeError: 当且仅当所有 error 都是未知键时
    """
    errors = error.errors()
    if not errors or any(item.get("type") != "extra_forbidden" for item in errors):
        return
    unknown = sorted({str(item["loc"][0]) for item in errors if item.get("loc")})
    if not unknown:
        return
    own_init = vars(strategy_cls).get("__init__")
    init_arg_names = _legacy_init_arg_names(own_init) if own_init is not None else []
    raise TypeError(
        unknown_param_message(
            unknown_keys=unknown,
            declared_fields=sorted(resolve_param_model(strategy_cls).model_fields),
            strategy_label=f"{strategy_cls.__module__}.{strategy_cls.__name__}",
            init_signature_names=init_arg_names,
        )
    ) from error


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
    try:
        model = validate_payload(resolve_param_model(strategy_cls), payload)
    except ValidationError as exc:
        _reraise_unknown_params_with_diagnosis(strategy_cls, exc)
        raise
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
    try:
        model = validate_payload(resolve_param_model(strategy_cls), payload)
    except ValidationError as exc:
        _reraise_unknown_params_with_diagnosis(strategy_cls, exc)
        raise
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
