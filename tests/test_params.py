import datetime as dt

import pytest
from akquant.params import (
    BoolParam,
    ChoiceParam,
    DateRange,
    DateRangeParam,
    FloatParam,
    IntParam,
    ParamModel,
    ParamSpec,
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
