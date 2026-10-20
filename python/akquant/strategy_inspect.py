"""策略/指标脚本的静态检查: 不导入、不执行用户代码, 只用 ``ast`` 读源码.

给平台列表用: 判断文件里每个类是指标脚本还是策略、显示名、主图/附图、画不画。
识别不了的写法(参数化的值、循环里的声明、factory)标记为 dynamic, 不猜。
"""

from __future__ import annotations

import ast
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .indicator_declaration import resolve_plot_defaults
from .indicator_script import _snake_case

__all__ = ["ClassInfo", "IndicatorInfo", "InspectResult", "inspect_strategy"]

#: 视为 AKQuant 的模块名; qmfquant 是平台使用的重新打包版本。
_AK_MODULES = frozenset({"akquant", "qmfquant"})
_BASE_KINDS = {"Indicator": "indicator", "Strategy": "strategy"}
_CLASS_KEYWORDS = ("title", "id", "overlay")
_I_FIELDS = ("name", "pane", "plot", "outputs", "label")
_LOOP_NODES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
#: 里面的调用不一定执行(或执行时机不定)。嵌套函数/lambda 何时被调用也不知道。
_CONDITIONAL_NODES = (
    ast.If,
    ast.IfExp,
    ast.Try,
    getattr(ast, "TryStar", ast.Try),  # 3.11+
    ast.Match,
    ast.With,
    ast.AsyncWith,
    ast.BoolOp,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
)
#: 运行时执行顺序: 实例化时跑 __init__, 启动时跑 on_start; 其他方法时机未知。
_METHOD_PHASE = {"__init__": 0, "on_start": 1}
_OTHER_PHASE = 2


class _NotLiteral:
    """哨兵: 表达式不是字面量, 静态取不到值."""


_NOT_LITERAL = _NotLiteral()


@dataclass
class _FileNames:
    """文件顶层的名字绑定, 用于识别基类与推导自动命名."""

    #: 绑定到 akquant/qmfquant 模块的本地名
    modules: Set[str] = field(default_factory=set)
    #: ``from akquant import X as Y``: 本地名 Y -> 原名 X
    imported: Dict[str, str] = field(default_factory=dict)
    #: 顶层 ``class`` 名
    classes: Set[str] = field(default_factory=set)
    #: 顶层 ``def`` 名
    funcs: Set[str] = field(default_factory=set)


@dataclass
class _ICall:
    """一条 ``self.I(...)`` 调用及其执行上下文."""

    call: ast.Call
    in_loop: bool
    #: 在分支/短路/嵌套函数里, 或在 __init__/on_start 以外的方法里
    conditional: bool
    #: 0=__init__, 1=on_start, 2=其他(见 _METHOD_PHASE)
    phase: int


@dataclass
class IndicatorInfo:
    """一条 ``self.I(...)`` 声明的静态信息."""

    name: Optional[str]
    pane: Optional[int]
    plot: Optional[bool]
    outputs: Optional[List[str]]
    label: Optional[str]
    dynamic: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """文件里一个策略/指标脚本类的静态信息."""

    name: str
    kind: str  # "indicator" | "strategy" | "unknown"
    title: Optional[str]
    id: Optional[str]
    overlay: Optional[bool]
    draws_indicators: bool
    indicators: List[IndicatorInfo]
    has_dynamic: bool
    dynamic: List[str] = field(default_factory=list)


@dataclass
class InspectResult:
    """:func:`inspect_strategy` 的返回值."""

    classes: List[ClassInfo]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 序列化的 dict."""
        return asdict(self)


def inspect_strategy(source: Union[str, os.PathLike[str]]) -> InspectResult:
    """静态检查一个策略/指标脚本文件或源码字符串.

    不导入、不执行用户代码。解析不了的问题通过 ``error``/``dynamic`` 表达,
    不抛异常。

    :param source: 文件路径(``PathLike`` 或已存在文件的路径字符串)或源码字符串;
        只接受明文, 加密文件由调用方先解密
    :raises FileNotFoundError: 以 ``.py`` 结尾的字符串路径不存在
    :raises TypeError: 入参不是 str/PathLike
    """
    text = _read_source(source)
    if text.startswith("\ufeff"):
        text = text[1:]  # 带 BOM 的源码字符串
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return InspectResult(
            classes=[], error=f"SyntaxError: {exc.msg} (line {exc.lineno})"
        )
    except (ValueError, RecursionError, MemoryError) as exc:
        return InspectResult(classes=[], error=f"{type(exc).__name__}: {exc}")
    names = _collect_names(tree)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    by_name = {c.name: c for c in classes}
    infos: List[ClassInfo] = []
    for node in classes:
        kind = _resolve_kind(node, by_name, names, set())
        if kind is None:
            continue
        info = _inspect_class(node, kind, by_name, names)
        if info is not None:
            infos.append(info)
    return InspectResult(classes=infos)


def _read_source(source: Union[str, os.PathLike[str]]) -> str:
    if isinstance(source, os.PathLike):
        return Path(source).read_text(encoding="utf-8-sig")
    if not isinstance(source, str):
        raise TypeError(f"source 须为路径或源码字符串, 收到 {type(source).__name__}")
    if "\n" not in source:
        path = Path(source)
        try:
            is_file = path.is_file()
        except (OSError, ValueError):
            # 超长或含非法字符的"路径", 只可能是源码
            is_file = False
        if is_file:
            return path.read_text(encoding="utf-8-sig")
        if source.endswith(".py"):
            raise FileNotFoundError(source)
    return source


def _collect_names(tree: ast.Module) -> _FileNames:
    """收集文件顶层的 akquant 导入与本地 class/def 名."""
    names = _FileNames()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in _AK_MODULES:
                    names.modules.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom) and node.module in _AK_MODULES:
            for a in node.names:
                if a.name != "*":
                    names.imported[a.asname or a.name] = a.name
        elif isinstance(node, ast.ClassDef):
            names.classes.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.funcs.add(node.name)
    return names


def _ak_export(expr: ast.expr, names: _FileNames) -> Optional[str]:
    """``expr`` 引用的是 akquant 导出时返回其原名, 否则 None."""
    if isinstance(expr, ast.Name):
        return names.imported.get(expr.id)
    if (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id in names.modules
    ):
        return expr.attr
    return None


def _base_kind(expr: ast.expr, names: _FileNames) -> Optional[str]:
    exported = _ak_export(expr, names)
    return _BASE_KINDS.get(exported) if exported is not None else None


def _resolve_kind(
    node: ast.ClassDef,
    by_name: Dict[str, ast.ClassDef],
    names: _FileNames,
    seen: Set[str],
) -> Optional[str]:
    """沿同文件继承链找 AKQuant 基类; 找不到但有基类无法解析时返回 ``"unknown"``."""
    if node.name in seen:
        return None
    seen = seen | {node.name}
    unresolved = False
    for base in node.bases:
        kind = _base_kind(base, names)
        if kind is not None:
            return kind
        if isinstance(base, ast.Name) and base.id in by_name:
            inner = _resolve_kind(by_name[base.id], by_name, names, seen)
            if inner in ("indicator", "strategy"):
                return inner
            unresolved = unresolved or inner == "unknown"
        else:
            unresolved = True
    return "unknown" if unresolved else None


def _literal(expr: ast.expr) -> Any:
    try:
        return ast.literal_eval(expr)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _NOT_LITERAL


def _class_keywords(node: ast.ClassDef) -> Dict[str, Any]:
    """类定义行的 title/id/overlay; 写成 None 视同没写, 取不到值记为哨兵."""
    out: Dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None or kw.arg not in _CLASS_KEYWORDS:
            continue
        value = _literal(kw.value)
        if value is None:
            continue
        if kw.arg == "overlay":
            ok = isinstance(value, bool)
        else:
            ok = isinstance(value, str) and bool(value.strip())
            if ok:
                value = value.strip()
        out[kw.arg] = value if ok else _NOT_LITERAL
    return out


def _inherited_overlay(
    node: ast.ClassDef, by_name: Dict[str, ast.ClassDef], seen: Set[str]
) -> Any:
    """同文件父类链上最近一次写的 overlay; 都没写为 None(即默认 False)."""
    if node.name in seen:
        return None
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in by_name:
            parent = by_name[base.id]
            kws = _class_keywords(parent)
            if "overlay" in kws:
                return kws["overlay"]
            found = _inherited_overlay(parent, by_name, seen | {node.name})
            if found is not None:
                return found
    return None


def _inspect_class(
    node: ast.ClassDef,
    kind: str,
    by_name: Dict[str, ast.ClassDef],
    names: _FileNames,
) -> Optional[ClassInfo]:
    calls = _collect_i_calls(node)
    if kind == "unknown" and not calls:
        # 基类来自别的文件且不声明指标: 多半和策略无关, 不列出
        return None
    kws = _class_keywords(node)
    dynamic = [k for k in _CLASS_KEYWORDS if kws.get(k) is _NOT_LITERAL]
    title = kws.get("title")
    declared_id = kws.get("id")
    overlay: Optional[bool] = None
    if kind == "indicator":
        raw = (
            kws["overlay"]
            if "overlay" in kws
            else _inherited_overlay(node, by_name, set())
        )
        if raw is _NOT_LITERAL:
            if "overlay" not in dynamic:
                dynamic.append("overlay")
        else:
            overlay = bool(raw)  # None(都没写) 即默认 False
    if "title" in dynamic:
        resolved_title: Optional[str] = None
    else:
        resolved_title = title or node.name
    if "id" in dynamic:
        resolved_id: Optional[str] = None
    elif declared_id:
        resolved_id = declared_id
    else:
        resolved_id = _snake_case(node.name) if kind == "indicator" else None
    is_script: Optional[bool] = {"indicator": True, "strategy": False}.get(kind)
    indicators = _build_indicator_infos(calls, is_script, overlay, names)
    return ClassInfo(
        name=node.name,
        kind=kind,
        title=resolved_title,
        id=resolved_id,
        overlay=overlay,
        draws_indicators=any(i.plot is not False for i in indicators),
        indicators=indicators,
        has_dynamic=bool(dynamic) or any(i.dynamic for i in indicators),
        dynamic=dynamic,
    )


def _is_self_i(call: ast.Call) -> bool:
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == "I"
        and isinstance(f.value, ast.Name)
        and f.value.id == "self"
    )


def _collect_i_calls(node: ast.ClassDef) -> List[_ICall]:
    """按运行时顺序返回类里的 ``self.I`` 调用.

    __init__ 里的先于 on_start 里的(与源码中方法的先后无关); 其他方法里的
    执行时机未知, 排在最后并记为 conditional。
    """
    found: List[_ICall] = []

    def visit(n: ast.AST, in_loop: bool, conditional: bool, phase: int) -> None:
        if isinstance(n, ast.Call) and _is_self_i(n):
            found.append(_ICall(n, in_loop, conditional, phase))
        loop = in_loop or isinstance(n, _LOOP_NODES)
        branch = conditional or isinstance(n, _CONDITIONAL_NODES)
        for child in ast.iter_child_nodes(n):
            visit(child, loop, branch, phase)

    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            phase = _METHOD_PHASE.get(stmt.name, _OTHER_PHASE)
            for inner in stmt.body:
                visit(inner, False, phase == _OTHER_PHASE, phase)
        else:
            visit(stmt, False, True, _OTHER_PHASE)
    found.sort(key=lambda c: (c.phase, c.call.lineno, c.call.col_offset))
    return found


def _auto_name_base(
    call: ast.Call, kw: Dict[str, ast.expr], names: _FileNames
) -> Optional[str]:
    """复刻 ``Strategy._resolve_indicator_name`` 的基名; 推不出返回 None.

    运行时取 ``type(indicator).__name__`` 或 ``factory.__name__``。静态只认
    确定知道名字的对象: akquant 导出、同文件的 class(以及 factory 的同文件
    def)。别处导入或运行期构造的可调用对象, 名字无从得知。
    """
    target = call.args[0] if call.args else kw.get("indicator")
    if target is not None:
        if not isinstance(target, ast.Call):
            return None
        exported = _ak_export(target.func, names)
        if exported is not None:
            return exported.lower()
        if isinstance(target.func, ast.Name) and target.func.id in names.classes:
            return target.func.id.lower()
        return None
    factory = kw.get("factory")
    if factory is None:
        return None
    if isinstance(factory, ast.Lambda):
        return "indicator"
    raw: Optional[str] = _ak_export(factory, names)
    if raw is None and isinstance(factory, ast.Name):
        if factory.id in names.classes or factory.id in names.funcs:
            raw = factory.id
    if raw is None:
        return None
    return raw.strip("_").lower() or "indicator"


def _field_value(fname: str, expr: ast.expr) -> Any:
    """字面量且类型合法时返回值, 否则返回哨兵."""
    value = _literal(expr)
    if value is _NOT_LITERAL or value is None:
        return value
    if fname == "pane":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif fname == "plot":
        ok = isinstance(value, bool)
    elif fname == "outputs":
        ok = isinstance(value, (list, tuple))
    else:  # name / label
        ok = isinstance(value, str)
    return value if ok else _NOT_LITERAL


def _auto_name_scope(calls: List[_ICall], names: _FileNames) -> Tuple[Set[str], bool]:
    """其他方法里的自动命名声明可能在任意时刻执行: 涉及的基名全程不可靠.

    返回 (不可靠的基名, 是否全部不可靠)。
    """
    bases: Set[str] = set()
    for item in calls:
        if item.phase != _OTHER_PHASE:
            continue
        kw = {k.arg: k.value for k in item.call.keywords if k.arg is not None}
        has_unpack = any(k.arg is None for k in item.call.keywords)
        if "name" in kw:
            value = _field_value("name", kw["name"])
            if value is _NOT_LITERAL:
                return bases, True
            if str(value or "").strip():
                continue
        elif has_unpack:
            return bases, True
        base = _auto_name_base(item.call, kw, names)
        if base is None:
            return bases, True
        bases.add(base)
    return bases, False


def _build_indicator_infos(
    calls: List[_ICall],
    is_script: Optional[bool],
    overlay: Optional[bool],
    names: _FileNames,
) -> List[IndicatorInfo]:
    counts: Dict[str, int] = {}
    # 自动命名按声明序号计数: 某次声明的基名或次数推不出, 或它不一定执行时,
    # 之后同基名(基名都推不出时则是全部)的自动名序号都不再可靠
    unsure_bases, unsure_all = _auto_name_scope(calls, names)
    out: List[IndicatorInfo] = []
    for item in calls:
        call = item.call
        kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
        # **opts 展开: 没显式写的字段都可能被它带进来
        has_unpack = any(k.arg is None for k in call.keywords)
        dyn: List[str] = []
        values: Dict[str, Any] = {}
        for fname in _I_FIELDS:
            if fname in kw:
                value = _field_value(fname, kw[fname])
            elif has_unpack:
                value = _NOT_LITERAL
            else:
                value = None
            if value is _NOT_LITERAL:
                dyn.append(fname)
                value = None
            values[fname] = value
        no_target = not call.args and "indicator" not in kw and "factory" not in kw
        if item.in_loop or "factory" in kw or (has_unpack and no_target):
            dyn.append("count")

        name: Optional[str] = None
        if "name" not in dyn:
            explicit = str(values["name"] or "").strip()
            if explicit:
                name = explicit
            else:
                base = _auto_name_base(call, kw, names)
                if base is None:
                    unsure_all = True
                    dyn.append("name")
                elif item.in_loop or unsure_all or base in unsure_bases:
                    unsure_bases.add(base)
                    dyn.append("name")
                else:
                    ordinal = counts.get(base, 0) + 1
                    counts[base] = ordinal
                    name = base if ordinal == 1 else f"{base}_{ordinal}"
                    if item.conditional:
                        # 自己的名字可推, 但它不一定执行, 之后同基名的序号不可靠
                        unsure_bases.add(base)
        else:
            # 动态名字可能为空串而落回自动命名, 自动名序号此后都不可靠
            unsure_all = True

        plot, pane = _resolve_plot_pane(
            is_script, overlay, values["plot"], values["pane"], dyn
        )
        outputs = values["outputs"]
        out.append(
            IndicatorInfo(
                name=name,
                pane=pane,
                plot=plot,
                outputs=[str(o) for o in outputs] if outputs is not None else None,
                label=values["label"],
                dynamic=dyn,
            )
        )
    return out


def _resolve_plot_pane(
    is_script: Optional[bool],
    overlay: Optional[bool],
    lit_plot: Optional[bool],
    lit_pane: Optional[int],
    dyn: List[str],
) -> Tuple[Optional[bool], Optional[int]]:
    """按 resolve_plot_defaults 解析最终值; 推不出的字段为 None 并记入 dyn."""
    plot_known = "plot" not in dyn
    pane_known = "pane" not in dyn
    if is_script is None:
        # 类别未知: 默认值无从推导, 只报写明的字面量
        if lit_plot is None and plot_known:
            dyn.append("plot")
            plot_known = False
        if lit_pane is None and pane_known:
            dyn.append("pane")
            pane_known = False
        return (lit_plot if plot_known else None, lit_pane if pane_known else None)
    if overlay is None and is_script and lit_pane is None and pane_known:
        # overlay 是动态的: 指标脚本的默认 pane 取决于它
        dyn.append("pane")
        pane_known = False
    plot, pane = resolve_plot_defaults(
        is_indicator_script=is_script,
        overlay=bool(overlay),
        plot=lit_plot,
        pane=lit_pane if pane_known else 0,
    )
    return (plot if plot_known else None, pane if pane_known else None)
