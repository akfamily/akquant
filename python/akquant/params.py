"""
策略参数建模与校验工具.

该模块提供轻量的参数 DSL，底层基于 Pydantic。
"""

import datetime as dt
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.fields import FieldInfo


class ParamModel(BaseModel):
    """
    策略参数基类.

    :cvar model_config: Pydantic 配置，默认禁止未知字段。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


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
    """内联参数字段规格：携带 python 类型与 pydantic FieldInfo."""

    python_type: type
    field_info: FieldInfo


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
]
