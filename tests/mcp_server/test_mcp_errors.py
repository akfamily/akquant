"""akquant.mcp.service.errors: 回测异常的阶段、用户帧与提示."""

from pathlib import Path
from typing import Any, Dict, Optional

import akquant as aq
import pytest
from akquant.mcp.service.data import synthetic_frame
from akquant.mcp.service.errors import (
    INIT_MARKERS,
    LOAD_MARKERS,
    RUN_MARKERS,
    describe_exception,
)

HEADER = "from akquant import IntParam, Strategy\n\n\n"


def _line_of(source: str, marker: str) -> int:
    return next(i for i, t in enumerate(source.splitlines(), 1) if marker in t)


def _describe(
    tmp_path: Path, source: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """在进程内跑回测, 返回 describe_exception 的结果.

    每个用例的 tmp_path 不同, 不会命中策略加载器按路径的模块缓存。
    """
    path = tmp_path / "strategy_under_test.py"
    path.write_text(source, encoding="utf-8")
    kwargs: Dict[str, Any] = {}
    if params is not None:
        kwargs["strategy_params"] = params
    with pytest.raises(Exception) as info:
        aq.run_backtest(
            data=synthetic_frame(end="2023-02-28"),
            strategy_source=str(path),
            show_progress=False,
            **kwargs,
        )
    return describe_exception(info.value, str(path))


def test_markers_exist_in_framework() -> None:
    """标记用的框架函数名必须真实存在; 框架改名时这里先红."""
    from akquant import strategy, strategy_framework_hooks, strategy_loader
    from akquant.backtest import engine

    assert LOAD_MARKERS == {"_load_python_plain"}
    assert INIT_MARKERS == {"_build_strategy_instance", "_on_start_internal"}
    assert RUN_MARKERS == {"call_user_callback"}
    assert hasattr(strategy_loader, "_load_python_plain")
    assert hasattr(engine, "_build_strategy_instance")
    assert hasattr(strategy.Strategy, "_on_start_internal")
    assert hasattr(strategy_framework_hooks, "call_user_callback")


def test_on_bar_error_is_run_phase(tmp_path: Path) -> None:
    """on_bar 抛错: phase=run, 最后一个用户帧指向 raise 行."""
    src = HEADER + (
        "class S(Strategy):\n"
        "    def on_bar(self, bar):\n"
        "        raise ValueError('boom in on_bar')\n"
    )
    out = _describe(tmp_path, src)
    assert out["phase"] == "run"
    assert out["type"] == "ValueError"
    assert out["user_frames"][-1]["line"] == _line_of(src, "raise ValueError")
    assert out["user_frames"][-1]["function"] == "on_bar"
    assert out["innermost_framework_frame"]["function"] == "call_user_callback"
    assert out["hint"] is None


def test_on_start_helper_is_init_phase(tmp_path: Path) -> None:
    """on_start 调 helper 抛错: phase=init, 用户帧 on_start -> helper."""
    src = HEADER + (
        "class S(Strategy):\n"
        "    def on_start(self):\n"
        "        self.helper()\n\n"
        "    def helper(self):\n"
        "        raise KeyError('boom in helper')\n\n"
        "    def on_bar(self, bar):\n"
        "        pass\n"
    )
    out = _describe(tmp_path, src)
    assert out["phase"] == "init"
    assert [f["function"] for f in out["user_frames"]] == ["on_start", "helper"]
    assert out["user_frames"][-1]["line"] == _line_of(src, "raise KeyError")


def test_init_error_is_init_phase(tmp_path: Path) -> None:
    """__init__ 抛错: phase=init."""
    src = HEADER + (
        "class S(Strategy):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        raise RuntimeError('boom in init')\n\n"
        "    def on_bar(self, bar):\n"
        "        pass\n"
    )
    out = _describe(tmp_path, src)
    assert out["phase"] == "init"
    assert out["user_frames"][-1]["line"] == _line_of(src, "raise RuntimeError")


def test_syntax_error_is_load_phase(tmp_path: Path) -> None:
    """语法错误: phase=load, 用户帧取自 SyntaxError 本身."""
    src = HEADER + "class S(Strategy)\n    pass\n"
    out = _describe(tmp_path, src)
    assert out["phase"] == "load"
    assert out["type"] == "SyntaxError"
    assert out["user_frames"][0]["line"] == _line_of(src, "class S")


def test_hint_unknown_param(tmp_path: Path) -> None:
    """未声明的参数: hint 指向 IntParam 声明."""
    src = HEADER + "class S(Strategy):\n    def on_bar(self, bar):\n        pass\n"
    out = _describe(tmp_path, src, params={"n": 3})
    assert out["phase"] == "init"
    assert "IntParam" in out["hint"]


def test_hint_custom_init(tmp_path: Path) -> None:
    """__init__ 带必填的非参数位置参数: hint 指出它不是策略参数."""
    src = HEADER + (
        "class S(Strategy):\n"
        "    n = IntParam(3)\n\n"
        "    def __init__(self, required_arg):\n"
        "        super().__init__()\n\n"
        "    def on_bar(self, bar):\n"
        "        pass\n"
    )
    out = _describe(tmp_path, src, params={"n": 2})
    assert "必填参数" in out["hint"]


def test_hint_ambiguous(tmp_path: Path) -> None:
    """多个 Strategy 子类: phase=load, hint 指向 strategy_attr."""
    src = HEADER + "class A(Strategy):\n    pass\n\n\nclass B(Strategy):\n    pass\n"
    out = _describe(tmp_path, src)
    assert out["phase"] == "load"
    assert "strategy_attr" in out["hint"]


def test_plain_exception_is_engine_phase(tmp_path: Path) -> None:
    """不经过任何标记函数的异常: phase=engine, 无用户帧."""
    try:
        raise ValueError("engine side")
    except ValueError as exc:
        out = describe_exception(exc, str(tmp_path / "s.py"))
    assert out["phase"] == "engine"
    assert out["user_frames"] == []
    assert out["hint"] is None


def test_syntax_error_raised_at_run_time_is_run_phase(tmp_path: Path) -> None:
    """on_bar 里 eval 出 SyntaxError: 属于运行阶段, 不是加载阶段."""
    src = HEADER + (
        "class S(Strategy):\n    def on_bar(self, bar):\n        eval('1 +')\n"
    )
    out = _describe(tmp_path, src)
    assert out["phase"] == "run"
    assert out["type"] == "SyntaxError"


def test_phase_uses_outer_traceback_not_context_chain(tmp_path: Path) -> None:
    """外层是运行阶段, __context__ 里残留 init 标记: 仍判 run."""

    def _on_start_internal() -> None:
        raise KeyError("early, swallowed")

    def call_user_callback() -> None:
        raise ValueError("late")

    try:
        try:
            _on_start_internal()
        except KeyError:
            call_user_callback()
    except ValueError as exc:
        assert isinstance(exc.__context__, KeyError)
        out = describe_exception(exc, str(tmp_path / "s.py"))
    assert out["phase"] == "run"


MULTI = "from akquant import SMA, Strategy\n\n\n"


def _describe_multi(tmp_path: Path, source: str) -> Dict[str, Any]:
    """同 _describe, 但用三个标的的数据."""
    path = tmp_path / "strategy_under_test.py"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(Exception) as info:
        aq.run_backtest(
            data=synthetic_frame(symbols=("A", "B", "C"), end="2023-02-28"),
            strategy_source=str(path),
            show_progress=False,
        )
    return describe_exception(info.value, str(path))


def test_shared_indicator_points_to_declaration(tmp_path: Path) -> None:
    """多标的共用增量指标: 错误在框架里抛出, user_frames 指回 self.I 声明行."""
    src = MULTI + (
        "class S(Strategy):\n"
        "    def on_start(self) -> None:\n"
        '        self.fast = self.I(factory=lambda: SMA(3), name="fast")\n'
        '        self.ma = self.I(SMA(5), name="ma")\n\n'
        "    def on_bar(self, bar):\n"
        "        pass\n"
    )
    out = _describe_multi(tmp_path, src)
    assert "shared instance" in out["message"]
    assert "factory=lambda" in out["hint"]
    assert len(out["user_frames"]) == 1
    frame = out["user_frames"][0]
    assert frame["line"] == _line_of(src, 'name="ma"')
    assert frame["function"] == "on_start"
    assert frame["code"].startswith("self.ma = self.I(")


@pytest.mark.parametrize("call", ['self.I("SMA", 5)', 'self.I("SMA", period=5)'])
def test_hint_indicator_call_shape(tmp_path: Path, call: str) -> None:
    """self.I 传错位置参数 / 关键字: hint 给出正确写法."""
    src = MULTI + (
        "class S(Strategy):\n"
        "    def on_start(self) -> None:\n"
        f"        self.ma = {call}\n\n"
        "    def on_bar(self, bar):\n"
        "        pass\n"
    )
    out = _describe(tmp_path, src)
    assert out["phase"] == "init"
    assert out["user_frames"][-1]["line"] == _line_of(src, "self.I(")
    assert "指标对象" in out["hint"]


def test_hint_history_depth(tmp_path: Path) -> None:
    """没设历史深度就取历史: hint 指向 set_history_depth."""
    src = MULTI + (
        "class S(Strategy):\n"
        "    def on_bar(self, bar):\n"
        "        self.get_history(3, bar.symbol)\n"
    )
    out = _describe(tmp_path, src)
    assert "History tracking is not enabled" in out["message"]
    assert "set_history_depth" in out["hint"]


def test_hint_attribute_error_suggests_close_name(tmp_path: Path) -> None:
    """AttributeError: hint 给相近的属性名; 没有相近名字时列出可用属性."""
    template = MULTI + (
        "class S(Strategy):\n"
        "    def on_bars(self, batch):\n"
        "        return batch.{attr}\n"
    )
    out = _describe(tmp_path, template.format(attr="mising"))
    assert out["phase"] == "run"
    assert out["hint"] == "BarBatch 没有属性 mising; 相近的属性: missing"
    other = tmp_path / "other"
    other.mkdir()
    out = _describe(other, template.format(attr="zzz"))
    assert out["hint"].startswith("BarBatch 没有属性 zzz; 可用属性: ")
    assert "flush_reason" in out["hint"]


def test_shared_indicator_matches_declaration_by_name(tmp_path: Path) -> None:
    """有多个没写 factory 的声明时, 只指回报错消息里点名的那一个."""
    src = MULTI + (
        "class S(Strategy):\n"
        "    def on_start(self) -> None:\n"
        '        self.a = self.I(SMA(3), name="first_ma")\n'
        '        self.b = self.I(SMA(5), name="second_ma")\n\n'
        "    def on_bar(self, bar):\n"
        "        pass\n"
    )
    out = _describe_multi(tmp_path, src)
    named = next(n for n in ("first_ma", "second_ma") if n in out["message"])
    assert [f["line"] for f in out["user_frames"]] == [_line_of(src, named)]
