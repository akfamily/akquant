"""search_api / get_api: 运行时内省 akquant 的公开 API.

内容来自已安装版本的签名与 docstring, 与框架版本天然一致。
索引范围: ``akquant.__all__``(跳过子模块)、``Strategy`` 的公开成员、
``run_backtest`` 的每个参数。
"""

from __future__ import annotations

import difflib
import inspect
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from . import ServiceError

SUMMARY_MAX = 120
_PARAM_RE = re.compile(r"^:param\s+(\w+):", re.MULTILINE)
_FIELD_RE = re.compile(r"^:\w+", re.MULTILINE)


@dataclass(frozen=True)
class ApiEntry:
    """索引里的一条 API."""

    name: str
    kind: str
    signature: Optional[str]
    doc: str
    source_module: Optional[str]


def first_sentence(text: str) -> str:
    """取首段的第一句, 压缩空白, 最多 SUMMARY_MAX 字."""
    para = text.strip().split("\n\n", 1)[0]
    flat = " ".join(para.split())
    for sep in ("。", ". "):
        idx = flat.find(sep)
        if idx != -1:
            flat = flat[: idx + len(sep)].rstrip()
            break
    return flat[:SUMMARY_MAX]


def _param_docs(doc: str) -> Dict[str, str]:
    """把 reST ``:param name:`` 段落拆成 {name: 文本}."""
    out: Dict[str, str] = {}
    matches = list(_PARAM_RE.finditer(doc))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        body = doc[match.end() : end]
        cut = _FIELD_RE.search(body)
        if cut:
            body = body[: cut.start()]
        out[match.group(1)] = " ".join(body.split())
    return out


def _signature(obj: Any) -> Optional[str]:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _top_level_entry(name: str, obj: Any) -> Optional[ApiEntry]:
    if obj is None or inspect.ismodule(obj):
        return None
    if inspect.isclass(obj):
        kind = "class"
    elif callable(obj):
        kind = "function"
    else:
        return ApiEntry(name, "constant", None, f"值: {obj!r}", None)
    doc = inspect.getdoc(obj) or ""
    module = getattr(obj, "__module__", None)
    return ApiEntry(name, kind, _signature(obj), doc, module)


def _strategy_entries(strategy: type) -> List[ApiEntry]:
    entries: List[ApiEntry] = []
    for name in dir(strategy):
        if name.startswith("_"):
            continue
        static = inspect.getattr_static(strategy, name)
        full = f"Strategy.{name}"
        if isinstance(static, property) or inspect.isdatadescriptor(static):
            fget = getattr(static, "fget", None)
            module = getattr(fget, "__module__", None)
            doc = inspect.getdoc(static) or ""
            entries.append(ApiEntry(full, "property", None, doc, module))
        elif inspect.isroutine(static) or isinstance(
            static, (staticmethod, classmethod)
        ):
            obj = getattr(strategy, name)
            doc = inspect.getdoc(obj) or ""
            module = getattr(obj, "__module__", None)
            entries.append(ApiEntry(full, "method", _signature(obj), doc, module))
    return entries


def _run_backtest_params(func: Any) -> List[ApiEntry]:
    docs = _param_docs(inspect.getdoc(func) or "")
    entries: List[ApiEntry] = []
    for pname, param in inspect.signature(func).parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        full = f"run_backtest.{pname}"
        doc = docs.get(pname, "")
        entries.append(ApiEntry(full, "parameter", str(param), doc, func.__module__))
    return entries


@lru_cache(maxsize=1)
def _index() -> Dict[str, ApiEntry]:
    import akquant

    entries: Dict[str, ApiEntry] = {}
    for name in getattr(akquant, "__all__", []):
        entry = _top_level_entry(name, getattr(akquant, name, None))
        if entry is not None:
            entries[name] = entry
    for entry in _strategy_entries(akquant.Strategy):
        entries[entry.name] = entry
    for entry in _run_backtest_params(akquant.run_backtest):
        entries[entry.name] = entry
    return entries


def search_api(query: str, limit: int = 20) -> Dict[str, Any]:
    """按关键词搜索: 名称命中优先, 其次 docstring 首段命中."""
    q = query.strip().lower()
    if not q:
        raise ServiceError("query 不能为空")
    if not 1 <= limit <= 200:
        raise ServiceError("limit 取值范围 1~200")
    hits: List[Tuple[int, int, str, ApiEntry]] = []
    for entry in _index().values():
        if q in entry.name.lower():
            rank = 0
        elif q in entry.doc.split("\n\n", 1)[0].lower():
            rank = 1
        else:
            continue
        hits.append((rank, len(entry.name), entry.name, entry))
    hits.sort(key=lambda h: (h[0], h[1], h[2]))
    results = [
        {"name": e.name, "kind": e.kind, "summary": first_sentence(e.doc)}
        for _, _, _, e in hits[:limit]
    ]
    return {"results": results, "truncated": len(hits) > limit}


def get_api(name: str) -> Dict[str, Any]:
    """返回一个 API 的完整签名与 docstring; 找不到时给相近名称."""
    index = _index()
    key = name.strip()
    entry = index.get(key)
    if entry is None:
        close = difflib.get_close_matches(key, list(index), n=5, cutoff=0.6)
        if close:
            tip = "相近的名称: " + ", ".join(close)
        else:
            tip = "可先用 search_api 搜索"
        raise ServiceError(f"找不到 API: {name}; {tip}")
    return {
        "name": entry.name,
        "kind": entry.kind,
        "signature": entry.signature,
        "doc": entry.doc,
        "source_module": entry.source_module,
    }
