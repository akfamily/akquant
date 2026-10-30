"""回测异常 -> 结构化错误: phase、用户帧、框架帧、修复提示.

phase 按 traceback 里出现的框架函数名判定(2026-10-30 实测, 见
tests/mcp_server/test_mcp_errors.py 的标记存在性测试); 框架改名时
测试先红, 不会静默退化成 engine。
"""

from __future__ import annotations

import os
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
            "unexpected keyword argument",
        ),
        "不要重写 __init__(或让它接受 **kwargs 并传给 super().__init__); "
        "状态初始化放到 on_start",
    ),
    (
        ("no Strategy subclass found",),
        "文件里要定义一个 Strategy 子类(一期不支持函数式策略)",
    ),
    (
        ("multiple Strategy subclasses found",),
        "用 strategy_attr 参数指定要用的类名",
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


def _hint(chain: List[BaseException]) -> Optional[str]:
    messages = [str(e) for e in chain]
    for needles, text in HINT_RULES:
        if any(all(n in m for n in needles) for m in messages):
            return text
    return None


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
        "user_frames": _user_frames(chain, frames_by_exc, strategy_path),
        "innermost_framework_frame": None
        if innermost is None
        else {
            "file": innermost.filename,
            "line": innermost.lineno,
            "function": innermost.name,
        },
        "hint": _hint(chain),
    }
