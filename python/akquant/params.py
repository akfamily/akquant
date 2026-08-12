"""
策略参数建模与校验工具.

该模块提供轻量的参数 DSL，底层基于 Pydantic。
"""

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ParamModel(BaseModel):
    """
    策略参数基类.

    :cvar model_config: Pydantic 配置，默认禁止未知字段。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    if TYPE_CHECKING:
        # 仅供 mypy 静态检查使用：内联参数字段（如 ``fast = IntParam(10)``）
        # 是在 Strategy 子类上动态声明的，mypy 无法从 ParamModel 基类推断出
        # 具体字段名。加上这个 `__getattr__` 让 `self.params.<field>` 在类型
        # 检查时退化为 ``Any``，避免误报 "ParamModel" has no attribute "xxx"。
        # 运行期该分支不会被执行，不影响 Pydantic 的实际属性访问与校验行为。
        def __getattr__(self, name: str) -> Any: ...  # noqa: D105


class DateRange(BaseModel):
    """
    日期区间类型.

    :ivar start: 区间开始
    :ivar end: 区间结束
    """

    start: dt.date | dt.datetime
    end: dt.date | dt.datetime

    @model_validator(mode="after")
    def _validate_range(self) -> "DateRange":
        """
        校验区间有效性.

        :return: 当前对象
        :raises ValueError: 结束时间早于开始时间
        """
        if _as_datetime(self.end) < _as_datetime(self.start):
            raise ValueError("date_range.end must be >= date_range.start")
        return self


@dataclass(frozen=True)
class ParamSpec:
    """内联参数字段规格：携带 python 类型与 pydantic FieldInfo.

    :ivar field_info: 实际运行期总是 ``pydantic.fields.FieldInfo``；
        标注为 ``Any`` 是因为 ``pydantic.Field()`` 的静态返回类型会
        随默认值的类型（``int``/``float``/``bool``/``str``/``DateRange | None``
        等）变化而变化，与固定的 ``FieldInfo`` 标注冲突，此处放宽以消除
        误报，不影响运行期行为。
    """

    python_type: type
    field_info: Any


def IntParam(
    default: int,
    *,
    ge: int | None = None,
    le: int | None = None,
    title: str | None = None,
    description: str | None = None,
) -> ParamSpec:
    """
    声明整型参数字段.

    :param default: 默认值
    :param ge: 最小值约束
    :param le: 最大值约束
    :param title: 展示标题
    :param description: 描述
    :return: 参数规格
    """
    return ParamSpec(
        int, Field(default, ge=ge, le=le, title=title, description=description)
    )


def FloatParam(
    default: float,
    *,
    ge: float | None = None,
    le: float | None = None,
    title: str | None = None,
    description: str | None = None,
) -> ParamSpec:
    """
    声明浮点参数字段.

    :param default: 默认值
    :param ge: 最小值约束
    :param le: 最大值约束
    :param title: 展示标题
    :param description: 描述
    :return: 参数规格
    """
    return ParamSpec(
        float, Field(default, ge=ge, le=le, title=title, description=description)
    )


def BoolParam(
    default: bool,
    *,
    title: str | None = None,
    description: str | None = None,
) -> ParamSpec:
    """
    声明布尔参数字段.

    :param default: 默认值
    :param title: 展示标题
    :param description: 描述
    :return: 参数规格
    """
    return ParamSpec(bool, Field(default, title=title, description=description))


def ChoiceParam(
    default: str,
    *,
    choices: Sequence[str],
    title: str | None = None,
    description: str | None = None,
) -> ParamSpec:
    """
    声明枚举参数字段.

    :param default: 默认值
    :param choices: 可选值列表
    :param title: 展示标题
    :param description: 描述
    :return: 参数规格
    :raises ValueError: choices 为空
    """
    if not choices:
        raise ValueError("choices cannot be empty")
    return ParamSpec(
        str,
        Field(
            default,
            title=title,
            description=description,
            json_schema_extra={"enum": list(choices)},
        ),
    )


def DateRangeParam(
    *,
    default: DateRange | None = None,
    title: str | None = None,
    description: str | None = None,
) -> ParamSpec:
    """
    声明日期区间参数字段.

    :param default: 默认值
    :param title: 展示标题
    :param description: 描述
    :return: 参数规格
    """
    return ParamSpec(DateRange, Field(default, title=title, description=description))


def model_to_schema(model_cls: type[ParamModel]) -> dict[str, Any]:
    """
    导出参数模型 JSON Schema.

    :param model_cls: 参数模型类
    :return: JSON Schema
    """
    return cast(dict[str, Any], model_cls.model_json_schema())


def validate_payload(
    model_cls: type[ParamModel],
    payload: Mapping[str, Any],
) -> ParamModel:
    """
    校验参数载荷并返回模型实例.

    :param model_cls: 参数模型类
    :param payload: 输入参数
    :return: 校验后的模型实例
    :raises ValidationError: 参数不合法
    """
    return cast(ParamModel, model_cls.model_validate(payload))


def to_runtime_kwargs(model: ParamModel) -> dict[str, Any]:
    """
    将参数模型映射为运行时 kwargs.

    当前规则：
    - 若存在 ``date_range`` 字段，自动展开为 ``start_time`` / ``end_time``。

    :param model: 参数模型实例
    :return: 运行时参数字典
    """
    runtime_kwargs: dict[str, Any] = {}
    data = model.model_dump()
    date_range_obj = data.get("date_range")
    if isinstance(date_range_obj, dict):
        start_value = date_range_obj.get("start")
        end_value = date_range_obj.get("end")
        if start_value is not None:
            runtime_kwargs["start_time"] = _to_iso(start_value)
        if end_value is not None:
            runtime_kwargs["end_time"] = _to_iso(end_value)
    return runtime_kwargs


def _as_datetime(value: dt.date | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    return dt.datetime.combine(value, dt.time.min)


def _to_iso(value: Any) -> str:
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value)


#: 参数声明文档锚点, 所有迁移提示共用.
PARAM_DOC_ANCHOR = "docs/zh/guide/strategy.md#param-declaration"

#: 报错文案里最多列出的已声明字段数, 超出截断.
_MAX_LISTED_FIELDS = 20


def _migration_hint(sample: str) -> str:
    """拼出「怎么改」的那一句, 供报错与告警共用.

    :param sample: 用作示范的参数名
    :return: 迁移提示片段
    """
    return (
        f"迁移: 类体声明 {sample} = IntParam(默认值), "
        f"读取处 self.{sample} -> self.params.{sample}; 详见 {PARAM_DOC_ANCHOR}"
    )


def unknown_param_message(
    *,
    unknown_keys: Sequence[str],
    declared_fields: Sequence[str],
    strategy_label: str,
    context: str = "",
) -> str:
    """拼出「未知策略参数」的可执行报错文案.

    按成因分流: ``declared_fields`` 为空说明该策略一个内联字段都没声明, 判定为
    遗留的构造函数签名写法, 给出完整迁移路径; 否则是键名不在已声明字段中(拼错或
    多传), 列出可用字段名。单行拼接——``TypeError`` 的 ``str()`` 会把换行转义成
    可见字面量, 多行在终端里反而更难读。

    :param unknown_keys: 未被接受的参数名
    :param declared_fields: 该策略已声明的内联字段名
    :param strategy_label: 形如 ``module.ClassName`` 的策略标识
    :param context: 来源上下文, 如 ``param_grid``; 为空表示策略构造期
    :return: 单行报错文案
    """
    keys_text = ", ".join(str(k) for k in unknown_keys)
    where = f" in {context}" if context else ""
    head = f"Unknown strategy param(s){where}: {keys_text}. Strategy={strategy_label}"
    if not declared_fields:
        sample = str(unknown_keys[0]) if unknown_keys else "my_param"
        return (
            f"{head}; 该策略未声明任何内联参数字段, 0.3.x 起构造函数签名不再是参数"
            f"入口, 故这些参数无法从外部传入; {_migration_hint(sample)}"
        )
    listed = sorted(str(f) for f in declared_fields)
    shown = ", ".join(listed[:_MAX_LISTED_FIELDS])
    suffix = " ..." if len(listed) > _MAX_LISTED_FIELDS else ""
    return (
        f"{head}; 该策略已声明 {len(listed)} 个参数字段: {shown}{suffix}; "
        "请检查键名拼写, 或改用上述字段名"
    )


def legacy_init_warning_message(
    *, strategy_name: str, init_arg_names: Sequence[str]
) -> str:
    """拼出类定义期「遗留写法」告警文案.

    :param strategy_name: 策略类名
    :param init_arg_names: ``__init__`` 中除 self/*args/**kwargs 外的命名参数
    :return: 单行告警文案
    """
    args_text = ", ".join(str(a) for a in init_arg_names)
    sample = str(init_arg_names[0]) if init_arg_names else "my_param"
    return (
        f"策略 {strategy_name} 定义了带参 __init__({args_text}), 但未声明任何内联"
        f"参数字段; 0.3.x 起构造函数签名不再是参数入口, 这些参数无法从 "
        f"run_backtest / run_grid_search 外部传入, 将始终取默认值; "
        f"{_migration_hint(sample)}"
    )


__all__ = [
    "ParamModel",
    "ParamSpec",
    "DateRange",
    "IntParam",
    "FloatParam",
    "BoolParam",
    "ChoiceParam",
    "DateRangeParam",
    "ValidationError",
    "model_to_schema",
    "validate_payload",
    "to_runtime_kwargs",
    "PARAM_DOC_ANCHOR",
    "unknown_param_message",
    "legacy_init_warning_message",
]
