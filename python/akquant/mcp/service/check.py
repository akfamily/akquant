"""check_strategy: 只做 ast 静态分析, 不 import、不执行用户代码.

诊断码见 spec §2.4。``unknown-self-method`` 只是 warning: 动态属性
(setattr、mixin)静态看不出来, 不能阻断。

``unresolved-base``(warning): 类的基类无法静态确认(来自其他文件、或
``from akquant import *``)。这类类按"可能的策略"处理, 不报 no-strategy;
对它不做 unknown-self-method 检查(方法可能定义在看不到的基类里)。
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Set

from ...strategy_inspect import inspect_strategy
from . import ServiceError
from .paths import PathArg, ServiceConfig, resolve_in_root

_INIT_METHODS = frozenset({"__init__", "on_start"})
_FuncDef = (ast.FunctionDef, ast.AsyncFunctionDef)


def _diag(
    severity: str,
    code: str,
    line: int,
    col: int,
    message: str,
    hint: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "line": line,
        "col": col,
        "message": message,
        "hint": hint,
    }


@lru_cache(maxsize=1)
def _base_attrs() -> FrozenSet[str]:
    import akquant

    return frozenset(dir(akquant.Strategy))


def _strategy_signature(name: str) -> Optional[inspect.Signature]:
    import akquant

    obj = getattr(akquant.Strategy, name, None)
    if obj is None or not callable(obj):
        return None
    try:
        return inspect.signature(obj)
    except (TypeError, ValueError):
        return None


def _self_call_attr(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
    ):
        return func.attr
    return None


def _flatten(target: ast.expr) -> List[ast.expr]:
    if isinstance(target, (ast.Tuple, ast.List)):
        return [x for elt in target.elts for x in _flatten(elt)]
    return [target]


def _defined_names(
    node: ast.ClassDef, by_name: Dict[str, ast.ClassDef], seen: Set[str]
) -> Set[str]:
    """类及其同文件父类里的方法、类属性、``self.x = ...`` 属性名."""
    if node.name in seen:
        return set()
    seen.add(node.name)
    names: Set[str] = set()
    for stmt in node.body:
        if isinstance(stmt, _FuncDef):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                names.update(t.id for t in _flatten(target) if isinstance(t, ast.Name))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
    for sub in ast.walk(node):
        targets: List[ast.expr] = []
        if isinstance(sub, ast.Assign):
            targets = [t for target in sub.targets for t in _flatten(target)]
        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
            targets = [sub.target]
        for t in targets:
            if (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                names.add(t.attr)
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in by_name:
            names |= _defined_names(by_name[base.id], by_name, seen)
    return names


def _check_call(
    call: ast.Call,
    attr: str,
    method: str,
    defined: Set[str],
    cls: str,
    resolvable: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if attr == "I" and method not in _INIT_METHODS:
        out.append(
            _diag(
                "warning",
                "i-call-outside-init",
                call.lineno,
                call.col_offset,
                f"self.I(...) 写在 {method} 里",
                "self.I() 是声明式指标, 应在 on_start 里声明一次",
            )
        )
    if attr in defined:
        return out
    if attr not in _base_attrs():
        if not resolvable:
            return out
        out.append(
            _diag(
                "warning",
                "unknown-self-method",
                call.lineno,
                call.col_offset,
                f"self.{attr}(...) 在 {cls} 与 akquant.Strategy 上都找不到",
                "用 search_api 查正确的方法名",
            )
        )
        return out
    sig = _strategy_signature(attr)
    if sig is None:
        return out
    params = sig.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return out
    allowed = [p for p in sig.parameters if p != "self"]
    for kw in call.keywords:
        if kw.arg is not None and kw.arg not in sig.parameters:
            out.append(
                _diag(
                    "error",
                    "unknown-kwarg",
                    call.lineno,
                    call.col_offset,
                    f"Strategy.{attr} 没有参数 {kw.arg}",
                    "可用参数: " + ", ".join(allowed),
                )
            )
    return out


def _is_strategy_ref(base: ast.expr) -> bool:
    if isinstance(base, ast.Name):
        return base.id == "Strategy"
    return isinstance(base, ast.Attribute) and base.attr == "Strategy"


def _is_object(base: ast.expr) -> bool:
    return isinstance(base, ast.Name) and base.id == "object"


def _mentions_strategy(
    node: ast.ClassDef, by_name: Dict[str, ast.ClassDef], seen: Set[str]
) -> bool:
    """基类链上出现名为 Strategy 的基类(如 ``from akquant import *`` 引入)."""
    if node.name in seen:
        return False
    seen = seen | {node.name}
    for base in node.bases:
        if _is_strategy_ref(base):
            return True
        if isinstance(base, ast.Name) and base.id in by_name:
            if _mentions_strategy(by_name[base.id], by_name, seen):
                return True
    return False


def _fully_resolvable(
    node: ast.ClassDef, by_name: Dict[str, ast.ClassDef], seen: Set[str]
) -> bool:
    """基类链上每个基类都是 Strategy 或同文件类(递归)."""
    if node.name in seen:
        return True
    seen = seen | {node.name}
    for base in node.bases:
        if _is_strategy_ref(base) or _is_object(base):
            continue
        if isinstance(base, ast.Name) and base.id in by_name:
            if not _fully_resolvable(by_name[base.id], by_name, seen):
                return False
            continue
        return False
    return True


def _check_class(
    node: ast.ClassDef, by_name: Dict[str, ast.ClassDef]
) -> List[Dict[str, Any]]:
    diags: List[Dict[str, Any]] = []
    defined = _defined_names(node, by_name, set())
    resolvable = _fully_resolvable(node, by_name, set())
    for stmt in node.body:
        if not isinstance(stmt, _FuncDef):
            continue
        if stmt.name == "__init__" and stmt.args.kwarg is None:
            diags.append(
                _diag(
                    "warning",
                    "custom-init",
                    stmt.lineno,
                    stmt.col_offset,
                    f"{node.name} 重写了 __init__ 且不接受 **kwargs, "
                    "外部传 strategy_params 时实例化会失败",
                    "不要重写 __init__: 参数用类体 IntParam 等声明, "
                    "状态初始化放到 on_start",
                )
            )
        for sub in ast.walk(stmt):
            attr = _self_call_attr(sub)
            if attr is not None and isinstance(sub, ast.Call):
                diags.extend(
                    _check_call(sub, attr, stmt.name, defined, node.name, resolvable)
                )
    return diags


def _unresolved_warning(names: List[str]) -> Dict[str, Any]:
    return _diag(
        "warning",
        "unresolved-base",
        1,
        0,
        "基类无法静态确认, 可能的策略类: " + (", ".join(names) or "无"),
        "基类无法静态确认; 回测时如报多个 Strategy 子类, 请传 strategy_attr 指定类名",
    )


def _select(
    tree: ast.Module,
    names: List[str],
    strategy_attr: Optional[str],
    unknown: Optional[List[str]] = None,
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    unknown = unknown or []
    possible = names + unknown
    if strategy_attr:
        if strategy_attr in names:
            return strategy_attr, []
        if strategy_attr in unknown:
            return strategy_attr, [_unresolved_warning([strategy_attr])]
        tip = "可选: " + (", ".join(possible) or "无")
        msg = f"文件里没有名为 {strategy_attr} 的 Strategy 子类"
        return None, [_diag("error", "no-strategy", 1, 0, msg, tip)]
    if len(names) == 1:
        return names[0], []
    if not names and unknown:
        return None, [_unresolved_warning(unknown)]
    if not names:
        functional = any(
            isinstance(n, _FuncDef) and n.name == "on_bar" for n in tree.body
        )
        if functional:
            tip = "一期只支持类式策略: 把 on_bar 等函数改写为 Strategy 子类的方法"
        else:
            tip = "定义一个继承 akquant.Strategy 的类"
        return None, [
            _diag("error", "no-strategy", 1, 0, "文件里没有 Strategy 子类", tip)
        ]
    msg = "文件里有多个 Strategy 子类: " + ", ".join(names)
    tip = "传 strategy_attr 指定要用的类"
    return None, [_diag("error", "ambiguous-strategy", 1, 0, msg, tip)]


def check_strategy(
    cfg: ServiceConfig, path: PathArg, strategy_attr: Optional[str] = None
) -> Dict[str, Any]:
    """静态检查策略文件, 返回类信息与诊断列表."""
    file = resolve_in_root(cfg, path)
    if not file.is_file():
        raise ServiceError(f"策略文件不存在: {path}")
    try:
        text = file.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        diag = _diag(
            "error",
            "syntax-error",
            1,
            0,
            "文件不是 UTF-8 编码",
            "请把策略文件另存为 UTF-8",
        )
        return {"ok": False, "classes": [], "selected": None, "diagnostics": [diag]}
    try:
        tree = ast.parse(text, filename=str(file))
    except ValueError as exc:  # 3.10/3.11 对 NUL 字节抛 ValueError
        diag = _diag("error", "syntax-error", 1, 0, f"语法错误: {exc}")
        return {"ok": False, "classes": [], "selected": None, "diagnostics": [diag]}
    except SyntaxError as exc:
        diag = _diag(
            "error",
            "syntax-error",
            exc.lineno or 1,
            max((exc.offset or 1) - 1, 0),
            f"语法错误: {exc.msg}",
            exc.text.strip() if exc.text else None,
        )
        return {"ok": False, "classes": [], "selected": None, "diagnostics": [diag]}
    infos = inspect_strategy(file).classes
    names = [c.name for c in infos if c.kind == "strategy"]
    by_name = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    # inspect_strategy 会丢掉没有 self.I 调用的 unknown 类, 这里自己补
    known = {c.name for c in infos if c.kind != "unknown"}
    unknown = [
        n
        for n, node in by_name.items()
        if n not in known
        and n not in names
        and any(not _is_object(b) for b in node.bases)
        and (
            not _fully_resolvable(node, by_name, set())
            or _mentions_strategy(node, by_name, set())
        )
    ]
    selected, diags = _select(tree, names, strategy_attr, unknown)
    for name in [selected] if selected else names + unknown:
        node = by_name.get(name)
        if node is not None:
            diags.extend(_check_class(node, by_name))
    diags.sort(key=lambda d: (d["line"], d["col"]))
    return {
        "ok": not any(d["severity"] == "error" for d in diags),
        "classes": [asdict(c) for c in infos],
        "selected": selected,
        "diagnostics": diags,
    }
