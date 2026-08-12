import datetime as dt

import pytest
from akquant.params import (
    PARAM_DOC_ANCHOR,
    BoolParam,
    ChoiceParam,
    DateRange,
    DateRangeParam,
    FloatParam,
    IntParam,
    ParamModel,
    ParamSpec,
    legacy_init_warning_message,
    unknown_param_message,
)
from pydantic import ValidationError


def test_int_param_returns_paramspec_with_type() -> None:
    """Test IntParam returns ParamSpec with correct python_type."""
    spec = IntParam(10, ge=2, le=200, title="快线")
    assert isinstance(spec, ParamSpec)
    assert spec.python_type is int
    assert spec.field_info.default == 10


def test_field_helpers_carry_python_types() -> None:
    """Test all field helpers carry correct python_type."""
    assert FloatParam(1.0).python_type is float
    assert BoolParam(True).python_type is bool
    assert ChoiceParam("a", choices=["a", "b"]).python_type is str
    assert DateRangeParam().python_type is DateRange


def test_choice_param_records_enum() -> None:
    """Test ChoiceParam records choices in json_schema_extra."""
    spec = ChoiceParam("a", choices=["a", "b"])
    assert spec.field_info.json_schema_extra == {"enum": ["a", "b"]}


def test_choice_param_rejects_empty_choices() -> None:
    """Test ChoiceParam raises ValueError for empty choices."""
    with pytest.raises(ValueError):
        ChoiceParam("a", choices=[])


def test_parammodel_is_frozen() -> None:
    """Test ParamModel instances are frozen."""

    class M(ParamModel):
        x: int = 1

    m = M()
    with pytest.raises(ValidationError):
        m.x = 2


def test_daterange_validates_order() -> None:
    """Test DateRange validates that end >= start."""
    with pytest.raises(ValidationError):
        DateRange(start=dt.date(2023, 2, 1), end=dt.date(2023, 1, 1))


def test_unknown_param_message_empty_model_gives_migration_path() -> None:
    """空模型说明是遗留写法, 文案须给出迁移路径与文档锚点."""
    msg = unknown_param_message(
        unknown_keys=["fast_period", "slow_period"],
        declared_fields=[],
        strategy_label="user_strat.MyStrategy",
    )
    assert "fast_period, slow_period" in msg
    assert "user_strat.MyStrategy" in msg
    assert "未声明任何内联参数字段" in msg
    assert "self.params.fast_period" in msg
    assert PARAM_DOC_ANCHOR in msg


def test_unknown_param_message_declared_model_lists_available_fields() -> None:
    """已声明字段时应列出可用字段名, 便于分辨拼错还是多传."""
    msg = unknown_param_message(
        unknown_keys=["fast_perid"],
        declared_fields=["fast_period", "slow_period"],
        strategy_label="user_strat.MyStrategy",
    )
    assert "fast_perid" in msg
    assert "fast_period, slow_period" in msg
    assert "未声明任何内联参数字段" not in msg


def test_unknown_param_message_truncates_long_field_list() -> None:
    """字段过多时截断并加省略号, 避免刷屏."""
    fields = [f"p{i}" for i in range(30)]
    msg = unknown_param_message(
        unknown_keys=["zzz"], declared_fields=fields, strategy_label="m.S"
    )
    assert "..." in msg
    assert "p29" not in msg


def test_unknown_param_message_context_appears() -> None:
    """用 context 区分 param_grid 与构造期两条来源."""
    msg = unknown_param_message(
        unknown_keys=["a"],
        declared_fields=[],
        strategy_label="m.S",
        context="param_grid",
    )
    assert "param_grid" in msg


def test_legacy_init_warning_message_names_args_and_migration() -> None:
    """告警须点名参数与迁移写法."""
    msg = legacy_init_warning_message(
        strategy_name="MyStrategy", init_arg_names=["fast_period", "slow_period"]
    )
    assert "MyStrategy" in msg
    assert "fast_period, slow_period" in msg
    assert "IntParam" in msg
    assert "self.params.fast_period" in msg
    assert PARAM_DOC_ANCHOR in msg
