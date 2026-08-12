import pickle
import subprocess
import sys
import warnings

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


def test_legacy_init_emits_warning_at_class_definition() -> None:
    """带参 __init__ 且无内联字段 -> 类定义期即告警."""
    with pytest.warns(UserWarning, match="未声明任何内联参数字段"):

        class LegacyStrategy(Strategy):
            def __init__(self, fast_period: int = 5) -> None:
                super().__init__()
                self.fast_period = fast_period


def test_inline_param_strategy_emits_no_warning() -> None:
    """已用内联字段声明的策略不得告警."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class InlineStrategy(Strategy):
            fast = IntParam(5, ge=1)


def test_init_without_named_args_emits_no_warning() -> None:
    """只有 self/*args/**kwargs 的 __init__ 不算遗留写法."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class NoArgInitStrategy(Strategy):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)


def test_inherited_init_does_not_rewarn() -> None:
    """子类没自己定义 __init__ 时不得因继承而重复告警."""
    with pytest.warns(UserWarning):

        class ParentLegacy(Strategy):
            def __init__(self, window: int = 3) -> None:
                super().__init__()
                self.window = window

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class ChildLegacy(ParentLegacy):
            pass


def test_legacy_warning_points_at_user_class_definition() -> None:
    """Stacklevel 须让告警指向用户的类定义行, 而非框架内部."""
    with pytest.warns(UserWarning) as records:

        class StackCheckStrategy(Strategy):
            def __init__(self, window: int = 3) -> None:
                super().__init__()
                self.window = window

    assert records[0].filename == __file__


def test_akquant_own_module_is_exempt_from_legacy_warning() -> None:
    """AKQuant 包自身模块下的遗留写法类(依赖注入的内部包装器)不应告警.

    直接把 ``__module__`` 覆盖为 ``akquant.*`` 前缀来命中豁免分支——这是对
    "按 ``cls.__module__`` 是否以 akquant 开头判定" 这条规则本身的精确验证:
    在类体里赋值 ``__module__`` 会在 ``type.__new__`` 构建类、进而调用
    ``__init_subclass__`` 之前就生效, 因此 ``cls.__module__`` 在判定时已经是
    覆盖后的值, 等价于"该类物理定义在 akquant 包内"的场景, 且不依赖模块缓存/
    reload 这类脆弱手段。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class FakeInternalWrapper(Strategy):
            __module__ = "akquant.fake_internal_wrapper"

            def __init__(self, injected: int = 1) -> None:
                super().__init__()
                self.injected = injected


def test_import_akquant_emits_no_legacy_param_warning() -> None:
    """全新解释器进程 import akquant 不应产生本条告警.

    ``akquant`` 在当前测试进程里早已被 import 过(位于 sys.modules 缓存中),
    再次 import 不会重新执行类体, 无法验证真实的"用户第一次 import akquant"
    场景; 用 ``importlib.reload`` 强制重跑类体则有污染 pydantic 动态模型注册表
    等全局状态的风险。故用子进程起一个全新解释器, 在其中以
    ``warnings.simplefilter("always")`` 捕获 import 期间的全部告警, 断言其中
    不包含本条告警的特征文案——这直接复现并锁定了 concern 中报告的真实场景。
    """
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as records:\n"
        "    warnings.simplefilter('always')\n"
        "    import akquant\n"
        "    import akquant.backtest\n"
        "matches = [str(r.message) for r in records "
        "if '未声明任何内联参数字段' in str(r.message)]\n"
        "assert not matches, matches\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
