"""akquant.mcp guides: 主题列表、读取与代码块可运行."""

import re
from typing import Any, Dict, List

import akquant as aq
import pytest
from akquant.mcp.service import ServiceError
from akquant.mcp.service.data import synthetic_frame
from akquant.mcp.service.guides import GUIDE_DIR, GUIDE_ORDER, get_guide

_BLOCK_RE = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)
#: 这几篇必须至少含一个可回测的 Strategy 示例, 防止代码块测试恒真
_MUST_HAVE_STRATEGY = {"strategy-skeleton", "orders-and-positions", "indicators"}


def test_topics_match_files() -> None:
    """GUIDE_ORDER 与 guides 目录下的文件一一对应."""
    files = sorted(p.stem for p in GUIDE_DIR.glob("*.md"))
    assert files == sorted(GUIDE_ORDER)
    listing = get_guide()["topics"]
    assert [t["topic"] for t in listing] == list(GUIDE_ORDER)
    assert all(t["title"] and t["summary"] for t in listing)


def test_get_guide_and_errors() -> None:
    """读取单篇; 未知主题与路径穿越报错."""
    out = get_guide("indicators")
    assert out["content"].startswith("# ")
    assert out["title"]
    with pytest.raises(ServiceError, match="strategy-skeleton"):
        get_guide("nope")
    with pytest.raises(ServiceError):
        get_guide("../service/guides")


@pytest.mark.parametrize("topic", GUIDE_ORDER)
def test_python_blocks_run(topic: str) -> None:
    """每个 python 代码块可独立执行, 其中的 Strategy 子类能跑通回测."""
    blocks = _BLOCK_RE.findall(get_guide(topic)["content"])
    strategies: List[type] = []
    for i, code in enumerate(blocks):
        module = f"guide_{topic.replace('-', '_')}_{i}"
        namespace: Dict[str, Any] = {"__name__": module}
        exec(compile(code, f"<{topic}#{i}>", "exec"), namespace)
        strategies.extend(
            obj
            for obj in namespace.values()
            if isinstance(obj, type)
            and issubclass(obj, aq.Strategy)
            and obj.__module__ == module
        )
    if topic in _MUST_HAVE_STRATEGY:
        assert strategies, f"{topic} 应至少含一个 Strategy 示例"
    for cls in strategies:
        result = aq.run_backtest(
            data=synthetic_frame(end="2023-12-31"),
            strategy=cls,
            initial_cash=1_000_000,
            show_progress=False,
        )
        assert not result.metrics_df.empty
