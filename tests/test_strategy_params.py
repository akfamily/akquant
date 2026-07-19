import pickle

import pytest
from akquant import Strategy
from akquant.params import IntParam
from pydantic import ValidationError


class ParamStrategy(Strategy):
    """内联声明两个整型参数字段的策略."""

    fast = IntParam(10, ge=2, le=200)
    slow = IntParam(30, ge=3, le=500)


class ChildStrategy(ParamStrategy):
    """覆盖父类 slow 字段、继承 fast 字段的子策略."""

    slow = IntParam(60, ge=3, le=500)  # 覆盖父类


class NoParamStrategy(Strategy):
    """未声明任何参数字段的策略."""


def test_defaults_injected() -> None:
    """未传参时应使用字段默认值."""
    s = ParamStrategy()
    assert s.params.fast == 10
    assert s.params.slow == 30


def test_override_via_kwargs() -> None:
    """构造期 kwargs 应覆盖对应字段默认值."""
    s = ParamStrategy(fast=12)
    assert s.params.fast == 12
    assert s.params.slow == 30


def test_field_attr_removed_from_class() -> None:
    """字段只能经 self.params 访问，类上不可见."""
    assert not hasattr(ParamStrategy, "fast")


def test_params_is_frozen() -> None:
    """Params 实例应只读，赋值触发校验错误."""
    s = ParamStrategy()
    with pytest.raises(ValidationError):
        setattr(s.params, "fast", 1)


def test_inheritance_merge_child_overrides() -> None:
    """子类应继承父类字段，并可覆盖同名字段."""
    s = ChildStrategy()
    assert s.params.fast == 10  # 继承自父
    assert s.params.slow == 60  # 子类覆盖


def test_unknown_kwarg_rejected() -> None:
    """未声明的字段名应被拒绝."""
    with pytest.raises(ValidationError):
        ParamStrategy(unknown=1)


def test_unknown_kwarg_rejected_alongside_valid_field() -> None:
    """声明了字段的策略类，构造时混入额外非法字段名也应被拒绝(extra="forbid").

    即便随同一个合法字段 (fast) 一起传入，未声明的 bogus 键仍应触发
    ValidationError, 而非被静默忽略——固化"声明字段的类拒绝任何未知 kwarg"
    这一比原 spec 措辞更严格(fail-fast)的实际语义。
    """
    with pytest.raises(ValidationError):
        ParamStrategy(fast=10, bogus=1)


def test_out_of_range_rejected() -> None:
    """超出约束范围的字段值应被拒绝."""
    with pytest.raises(ValidationError):
        ParamStrategy(fast=1)  # ge=2


def test_no_param_strategy_has_empty_params() -> None:
    """未声明字段的策略也应具备空的 params."""
    s = NoParamStrategy()
    assert s.params is not None
    assert s.params.model_dump() == {}


def test_params_survive_pickle() -> None:
    """Params 应随策略实例一起 pickle 往返."""
    s = ParamStrategy(fast=15)
    restored = pickle.loads(pickle.dumps(s))
    assert restored.params.fast == 15
    assert restored.params.slow == 30
