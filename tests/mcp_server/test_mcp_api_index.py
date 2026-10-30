"""akquant.mcp.service.api_index: 运行时内省 API."""

import pytest
from akquant.mcp.service import ServiceError
from akquant.mcp.service.api_index import (
    _param_docs,
    first_sentence,
    get_api,
    search_api,
)


def _names(query: str, limit: int = 50) -> list[str]:
    return [r["name"] for r in search_api(query, limit)["results"]]


def test_search_finds_strategy_method_and_param() -> None:
    """能搜到 Strategy 方法与 run_backtest 参数, 名称命中排在前面."""
    assert "Strategy.buy" in _names("buy")
    names = _names("initial_cash")
    assert "run_backtest.initial_cash" in names
    hit = next(
        r
        for r in search_api("initial_cash")["results"]
        if r["name"] == "run_backtest.initial_cash"
    )
    assert hit["kind"] == "parameter"
    assert hit["summary"]


def test_search_truncates_and_validates() -> None:
    """截断时 truncated 为真; 空 query 与越界 limit 报错."""
    out = search_api("e", limit=1)
    assert len(out["results"]) == 1
    assert out["truncated"] is True
    with pytest.raises(ServiceError):
        search_api("  ")
    with pytest.raises(ServiceError):
        search_api("buy", limit=0)


def test_get_api_shapes() -> None:
    """函数、方法、参数三类条目都有签名与文档."""
    rb = get_api("run_backtest")
    assert rb["kind"] == "function"
    assert "strategy_source" in rb["signature"]
    buy = get_api("Strategy.buy")
    assert buy["kind"] == "method"
    assert "quantity" in buy["signature"]
    assert buy["doc"]
    param = get_api("run_backtest.initial_cash")
    assert param["kind"] == "parameter"
    assert param["signature"].startswith("initial_cash")
    assert param["doc"]
    assert get_api("Bar")["kind"] == "class"


def test_get_api_unknown_suggests_close_names() -> None:
    """拼错的名字报错, 消息里带相近候选."""
    with pytest.raises(ServiceError, match="Strategy.buy"):
        get_api("Strategy.buyy")


def test_first_sentence_and_param_docs() -> None:
    """首句截取与 :param 段落拆分."""
    assert first_sentence("简化版回测入口.\n\n后文") == "简化版回测入口."
    assert first_sentence("第一句。第二句。") == "第一句。"
    doc = ":param a: 甲\n    续行\n:type a: int\n:param b: 乙\n:return: 无"
    assert _param_docs(doc) == {"a": "甲 续行", "b": "乙"}
