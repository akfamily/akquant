"""类成员内省: 供 api_index(``Cls.member`` 条目)与 check(回调入参的属性名)共用.

只列 akquant 自己定义的成员: 继承自 ``object``、``Enum``、pydantic
``BaseModel`` 等外部基类的成员不列。pyo3 导出类的 ``__module__`` 是
``builtins``, 按"不是真正的内置类型"识别。
"""

from __future__ import annotations

import builtins
import dataclasses
import inspect
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Iterator, Optional, Set

_IVAR_RE = re.compile(r"^:ivar\s+(\w+):", re.MULTILINE)
_FIELD_RE = re.compile(r"^:\w+", re.MULTILINE)
_MISSING = object()
VALUE_MAX = 200


@dataclass(frozen=True)
class Member:
    """类的一个公开成员."""

    kind: str  # "method" | "property" | "attribute" | "constant"
    signature: Optional[str]
    doc: str
    source_module: Optional[str]


def _is_akquant_class(klass: type) -> bool:
    module = getattr(klass, "__module__", "") or ""
    if module == "akquant" or module.startswith("akquant."):
        return True
    # pyo3 导出类: __module__ 是 builtins, 但不是真正的内置类型
    return module == "builtins" and getattr(builtins, klass.__name__, None) is not klass


def _owners(cls: type) -> Iterator[type]:
    for klass in cls.__mro__:
        if klass is cls or (klass is not object and _is_akquant_class(klass)):
            yield klass


def _ivar_docs(cls: type) -> Dict[str, str]:
    """把类 docstring 里的 ``:ivar name:`` 段落拆成 {name: 文本}."""
    doc = inspect.getdoc(cls) or ""
    out: Dict[str, str] = {}
    matches = list(_IVAR_RE.finditer(doc))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        body = doc[match.end() : end]
        cut = _FIELD_RE.search(body)
        if cut:
            body = body[: cut.start()]
        out[match.group(1)] = " ".join(body.split())
    return out


def _fields(cls: type) -> Dict[str, str]:
    """数据字段 -> 类型注解文本(dataclass 字段、pydantic 字段、类体注解)."""
    out: Dict[str, str] = {}
    for klass in _owners(cls):
        for name, annotation in getattr(klass, "__annotations__", {}).items():
            text = annotation if isinstance(annotation, str) else _type_name(annotation)
            out.setdefault(name, text)
    if dataclasses.is_dataclass(cls):
        for item in dataclasses.fields(cls):
            text = item.type if isinstance(item.type, str) else _type_name(item.type)
            out.setdefault(item.name, text)
    model_fields = getattr(cls, "model_fields", None)
    if isinstance(model_fields, dict):
        for name, info in model_fields.items():
            out.setdefault(str(name), _type_name(getattr(info, "annotation", None)))
    return out


def _type_name(annotation: Any) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _signature(obj: Any) -> Optional[str]:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _field_doc(cls: type, name: str, ivars: Dict[str, str]) -> str:
    if name in ivars:
        return ivars[name]
    model_fields = getattr(cls, "model_fields", None)
    if isinstance(model_fields, dict) and name in model_fields:
        return str(getattr(model_fields[name], "description", None) or "")
    return ""


def member_names(cls: type) -> Set[str]:
    """``obj.<name>`` 可能合法的全部公开名字(宁多勿少, 供静态检查用)."""
    names = {n for n in dir(cls) if not n.startswith("_")}
    names.update(n for n in _fields(cls) if not n.startswith("_"))
    return names


def class_members(cls: type) -> Dict[str, Member]:
    """列出 akquant 为该类定义的公开成员."""
    ivars = _ivar_docs(cls)
    fields = _fields(cls)
    names: Set[str] = set(fields)
    for klass in _owners(cls):
        names.update(vars(klass))
    out: Dict[str, Member] = {}
    for name in sorted(n for n in names if not n.startswith("_")):
        static = inspect.getattr_static(cls, name, _MISSING)
        module = getattr(cls, "__module__", None)
        if isinstance(static, cached_property):
            doc = inspect.getdoc(static.func) or ivars.get(name, "")
            out[name] = Member("property", None, doc, module)
        elif inspect.isroutine(static) or isinstance(
            static, (staticmethod, classmethod)
        ):
            obj = getattr(cls, name)
            doc = inspect.getdoc(obj) or ""
            out[name] = Member("method", _signature(obj), doc, module)
        elif isinstance(static, property) or (
            static is not _MISSING
            and name not in fields
            and inspect.isdatadescriptor(static)
        ):
            doc = inspect.getdoc(static) or ivars.get(name, "")
            out[name] = Member("property", None, doc, module)
        elif name in fields:
            doc = _field_doc(cls, name, ivars)
            out[name] = Member("attribute", f": {fields[name]}", doc, module)
        elif static is _MISSING or inspect.isclass(static):
            continue
        else:
            try:
                value = getattr(cls, name)
            except Exception:
                continue
            kind = "constant" if isinstance(value, cls) else "attribute"
            text = repr(value)
            if len(text) > VALUE_MAX:
                text = text[:VALUE_MAX] + "…"
            doc = ivars.get(name) or f"值: {text}"
            out[name] = Member(kind, None, doc, module)
    return out
