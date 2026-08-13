import datetime as dt

import pytest
from akquant import Strategy
from akquant.params import (
    PARAM_DOC_ANCHOR,
    BoolParam,
    ChoiceParam,
    DateRange,
    DateRangeParam,
    FloatParam,
    IntParam,
    ListParam,
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
    assert "未声明任何内联参数字段" in msg


def test_legacy_init_warning_message_partial_migration_wording_differs() -> None:
    """半迁移态(已有其他字段)不得再声称"未声明任何内联参数字段", 那与实际矛盾."""
    msg = legacy_init_warning_message(
        strategy_name="MixedStrategy",
        init_arg_names=["slow"],
        some_fields_declared=True,
    )
    assert "MixedStrategy" in msg
    assert "slow" in msg
    assert "未声明任何内联参数字段" not in msg
    assert "self.params.slow" in msg
    assert PARAM_DOC_ANCHOR in msg


def test_unknown_param_message_flags_unmigrated_init_arg_over_typo_check() -> None:
    """Key 命中该策略 __init__ 签名但未声明为字段时, 应给迁移路径而非拼写检查.

    覆盖半迁移态运行期报错: declared_fields 非空(已有 fast), 但 unknown key
    (slow) 其实是 __init__ 里忘迁的参数, 不是拼错——不应走"请检查键名拼写"分支。
    """
    msg = unknown_param_message(
        unknown_keys=["slow"],
        declared_fields=["fast"],
        strategy_label="m.Mixed",
        init_signature_names=["slow"],
    )
    assert "slow" in msg
    assert "__init__" in msg
    assert "请检查键名拼写" not in msg
    assert "self.params.slow" in msg


def test_unknown_param_message_pure_typo_still_suggests_spelling_check() -> None:
    """Key 不在 __init__ 签名里时, 仍是拼错/多传, 维持"检查拼写"分支."""
    msg = unknown_param_message(
        unknown_keys=["fsat"],
        declared_fields=["fast", "slow"],
        strategy_label="m.Mixed",
        init_signature_names=["slow"],
    )
    assert "fsat" in msg
    assert "请检查键名拼写" in msg


def test_list_param_default_and_injection() -> None:
    """ListParam 应声明出 list 字段并接受外部传值."""

    class SymbolsStrategy(Strategy):
        symbols = ListParam(["AAA"], item_type=str, title="标的集")

    assert set(SymbolsStrategy.__param_model__.model_fields) == {"symbols"}
    assert SymbolsStrategy().params.symbols == ["AAA"]
    assert SymbolsStrategy(symbols=["X", "Y"]).params.symbols == ["X", "Y"]


def test_list_param_default_not_shared_between_instances() -> None:
    """默认值必须走 default_factory, 两个实例不得共享同一列表对象."""

    class SymbolsStrategy(Strategy):
        symbols = ListParam(["AAA"], item_type=str)

    a = SymbolsStrategy()
    b = SymbolsStrategy()
    assert a.params.symbols == b.params.symbols
    assert a.params.symbols is not b.params.symbols


def test_list_param_none_default_is_empty_list() -> None:
    """default=None 视为空列表, 而非 None."""

    class EmptyStrategy(Strategy):
        tags = ListParam(item_type=str)

    assert EmptyStrategy().params.tags == []
