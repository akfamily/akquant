"""get_guide: 读取打包在 akquant/mcp/guides/ 下的写策略要点."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import ServiceError
from .api_index import first_sentence

GUIDE_DIR = Path(__file__).resolve().parent.parent / "guides"
GUIDE_ORDER = (
    "strategy-skeleton",
    "orders-and-positions",
    "indicators",
    "data-format",
    "market-rules",
)
_TOPIC_RE = re.compile(r"^[a-z0-9-]+$")


def _parse(text: str) -> Tuple[str, str]:
    """返回 (标题, 摘要): 标题取首个 ``# `` 行, 摘要取其后第一段正文."""
    lines = text.splitlines()
    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), "")
    paras = [p.strip() for p in text.split("\n\n")]
    body = next((p for p in paras if p and not p.startswith("#")), "")
    return title, first_sentence(body)


def _read(topic: str) -> str:
    if topic not in GUIDE_ORDER or not _TOPIC_RE.match(topic):
        raise ServiceError(f"没有这个主题: {topic}; 可选: {', '.join(GUIDE_ORDER)}")
    return (GUIDE_DIR / f"{topic}.md").read_text(encoding="utf-8")


def get_guide(topic: Optional[str] = None) -> Dict[str, Any]:
    """不传 topic 返回主题列表, 传 topic 返回该篇全文."""
    if topic is None:
        topics: List[Dict[str, str]] = []
        for name in GUIDE_ORDER:
            title, summary = _parse(_read(name))
            topics.append({"topic": name, "title": title, "summary": summary})
        return {"topics": topics}
    text = _read(topic)
    title, _ = _parse(text)
    return {"topic": topic, "title": title, "content": text}
