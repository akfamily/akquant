"""回测异常 -> 结构化错误: phase、用户帧、框架帧、修复提示.

phase 按 traceback 里出现的框架函数名判定(2026-10-30 实测, 见
tests/mcp_server/test_mcp_errors.py 的标记存在性测试); 框架改名时
测试先红, 不会静默退化成 engine。
"""

from __future__ import annotations

import ast
import difflib
import os
import re
import traceback
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

#: strategy_loader._load_python_plain: 读取并执行策略模块
LOAD_MARKERS = frozenset({"_load_python_plain"})
#: engine._build_strategy_instance(实例化) / Strategy._on_start_internal
INIT_MARKERS = frozenset({"_build_strategy_instance", "_on_start_internal"})
#: strategy_framework_hooks.call_user_callback: on_bar 等用户回调都经过它
RUN_MARKERS = frozenset({"call_user_callback"})
MESSAGE_MAX = 2000
_CHAIN_MAX = 5
#: AttributeError 找不到相近名字时, 对象公开属性不超过这个数才整份列出
_ATTR_LIST_MAX = 30
_INDICATOR_NAME_RE = re.compile(r"[Ii]ndicator '([^']+)'")

#: (异常消息需同时包含的子串, hint); 只收实测过、确定性高的规则
HINT_RULES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (
        ("Unknown strategy param(s)",),
        "参数要在类体里用 IntParam / FloatParam / BoolParam / ChoiceParam / "
        "ListParam 声明, 读取用 self.params.<名>",
    ),
    (
        (
            "Failed to instantiate strategy with provided parameters",
            "missing",
        ),
        "__init__ 有必填参数, 但它不是声明的策略参数: 给它一个默认值, "
        "或去掉; 策略参数只用类体 IntParam 等声明, 读取用 self.params",
    ),
    (
        ("no Strategy subclass found",),
        "文件里要定义一个 Strategy 子类(一期不支持函数式策略)",
    ),
    (
        ("multiple Strategy subclasses found",),
        "本文件定义了多个 Strategy 子类(或没有定义、只导入了多个); "
        "用 strategy_attr 参数指定要用的类名",
    ),
    (
        ("registered with a shared instance",),
        '多标的策略里增量指标不能共用一个实例: 把 self.I(SMA(5), name="ma") 改成 '
        'self.I(factory=lambda: SMA(5), name="ma"), 每个标的各建一个; 取值用 '
        "ind.of(symbol)[0]",
    ),
    (
        ("Strategy.I() takes",),
        "self.I 的第一个参数是指标对象, 其余参数只能用关键字: "
        'self.I(SMA(5), name="ma"); 周期等构造参数写在指标对象里',
    ),
    (
        ("Strategy.I() got an unexpected keyword argument",),
        "self.I 的第一个参数是指标对象, 其余参数只能用关键字: "
        'self.I(SMA(5), name="ma"); 周期等构造参数写在指标对象里',
    ),
    (
        ("History tracking is not enabled",),
        "get_history 等历史数据接口要先设置历史深度: 在 on_start 里调用 "
        "self.set_history_depth(N), N 不小于要取的根数",
    ),
)


def same_file(a: str, b: str) -> bool:
    """按 normcase + realpath 比较两个路径(Windows 大小写与分隔符不一致)."""
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(
        os.path.realpath(b)
    )


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: Set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen and len(seen) < _CHAIN_MAX:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _frame_dict(frame: traceback.FrameSummary) -> Dict[str, Any]:
    return {
        "file": frame.filename,
        "line": frame.lineno,
        "function": frame.name,
        "code": frame.line,
    }


def _phase_from_names(
    names: set[str], chain: List[BaseException], check_syntax: bool
) -> Optional[str]:
    if names & INIT_MARKERS or names & RUN_MARKERS:
        # 运行/初始化阶段的框架帧在栈里: SyntaxError 也是用户代码在运行时抛的
        return "init" if names & INIT_MARKERS else "run"
    if names & LOAD_MARKERS or (
        check_syntax and any(isinstance(e, SyntaxError) for e in chain)
    ):
        return "load"
    return None


def _phase(
    chain: List[BaseException], frames_by_exc: List[traceback.StackSummary]
) -> str:
    """先看最外层异常的栈; 栈里没有任何标记时才合并整条链."""
    outer = {f.name for f in frames_by_exc[0]}
    found = _phase_from_names(outer, chain[:1], True)
    if found:
        return found
    union = {f.name for frames in frames_by_exc for f in frames}
    return _phase_from_names(union, chain, True) or "engine"


def _user_frames(
    chain: List[BaseException],
    frames_by_exc: List[traceback.StackSummary],
    strategy_path: str,
) -> List[Dict[str, Any]]:
    for frames in frames_by_exc:
        hits = [_frame_dict(f) for f in frames if same_file(f.filename, strategy_path)]
        if hits:
            return hits
    for exc in chain:
        if (
            isinstance(exc, SyntaxError)
            and exc.filename
            and same_file(exc.filename, strategy_path)
        ):
            code = (exc.text or "").strip() or None
            return [
                {
                    "file": exc.filename,
                    "line": exc.lineno,
                    "function": "<module>",
                    "code": code,
                }
            ]
    return []


def _attribute_hint(exc: BaseException) -> Optional[str]:
    """AttributeError: 给出相近的属性名(解释器自带的建议只在打印时生成)."""
    if not isinstance(exc, AttributeError):
        return None
    name, obj = getattr(exc, "name", None), getattr(exc, "obj", None)
    if not isinstance(name, str) or obj is None:
        return None
    try:
        public = sorted(n for n in dir(obj) if not n.startswith("_"))
    except Exception:
        return None
    owner = type(obj).__name__
    close = difflib.get_close_matches(name, public, n=3, cutoff=0.6)
    if close:
        return f"{owner} 没有属性 {name}; 相近的属性: {', '.join(close)}"
    if public and len(public) <= _ATTR_LIST_MAX:
        return f"{owner} 没有属性 {name}; 可用属性: {', '.join(public)}"
    return None


def _hint(chain: List[BaseException]) -> Optional[str]:
    messages = [str(e) for e in chain]
    for needles, text in HINT_RULES:
        if any(all(n in m for n in needles) for m in messages):
            return text
    for exc in chain:
        found = _attribute_hint(exc)
        if found:
            return found
    return None


def _indicator_calls(strategy_path: str) -> List[Tuple[ast.Call, str, str]]:
    """策略文件里的 ``self.I(...)`` 调用: (调用节点, 所在函数名, 源码行)."""
    try:
        with open(strategy_path, encoding="utf-8-sig") as handle:
            text = handle.read()
        tree = ast.parse(text)
    except (OSError, ValueError, SyntaxError):
        return []
    lines = text.splitlines()
    out: List[Tuple[ast.Call, str, str]] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            target = node.func if isinstance(node, ast.Call) else None
            if (
                isinstance(node, ast.Call)
                and isinstance(target, ast.Attribute)
                and target.attr == "I"
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                code = (
                    lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                )
                out.append((node, func.name, code))
    return out


def _keyword(call: ast.Call, name: str) -> Optional[ast.expr]:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _declaration_frames(
    chain: List[BaseException], strategy_path: str
) -> List[Dict[str, Any]]:
    """指标相关的错误在框架里抛出、栈上没有策略文件时, 指回 ``self.I`` 声明行.

    先按报错消息里的指标名匹配 ``name=``; 匹配不到且属于"共用实例"错误时,
    列出没写 ``factory=`` 的声明。
    """
    messages = [str(e) for e in chain]
    names = {n for m in messages for n in _INDICATOR_NAME_RE.findall(m)}
    shared = any("registered with a shared instance" in m for m in messages)
    if not names and not shared:
        return []
    calls = _indicator_calls(strategy_path)
    hits = [
        item
        for item in calls
        if isinstance(value := _keyword(item[0], "name"), ast.Constant)
        and value.value in names
    ]
    if not hits and shared:
        hits = [item for item in calls if _keyword(item[0], "factory") is None]
    return [
        {"file": strategy_path, "line": call.lineno, "function": func, "code": code}
        for call, func, code in hits
    ]


def describe_exception(exc: BaseException, strategy_path: str) -> Dict[str, Any]:
    """把回测异常转成结构化错误(见 spec §2.6 / §3.4)."""
    chain = list(_chain(exc))
    frames_by_exc = [traceback.extract_tb(e.__traceback__) for e in chain]
    outer = frames_by_exc[0]
    framework = [f for f in outer if not same_file(f.filename, strategy_path)]
    innermost = framework[-1] if framework else None
    return {
        "phase": _phase(chain, frames_by_exc),
        "type": type(exc).__name__,
        "message": str(exc)[:MESSAGE_MAX],
        "user_frames": _user_frames(chain, frames_by_exc, strategy_path)
        or _declaration_frames(chain, strategy_path),
        "innermost_framework_frame": None
        if innermost is None
        else {
            "file": innermost.filename,
            "line": innermost.lineno,
            "function": innermost.name,
        },
        "hint": _hint(chain),
    }
