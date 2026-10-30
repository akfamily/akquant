"""MCP 层: 内存 Client 调用全部工具与 resource, 加一次 stdio 冒烟."""

import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, TypeVar

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
from akquant.mcp.server import build_server  # noqa: E402
from akquant.mcp.service.paths import ServiceConfig  # noqa: E402
from mcp import Client, StdioServerParameters  # noqa: E402

T = TypeVar("T")

TOOLS = {
    "search_api",
    "get_api",
    "get_guide",
    "check_strategy",
    "prepare_data",
    "run_backtest",
    "get_run_details",
}

BUYER = """\
from akquant import Strategy


class Buyer(Strategy):
    def on_start(self) -> None:
        self.count = 0

    def on_bar(self, bar):
        self.count += 1
        if self.count == 3:
            self.buy(symbol=bar.symbol, quantity=100)
"""


def _run(fn: Callable[[], Awaitable[T]]) -> T:
    return anyio.run(fn)


def _call(cfg: ServiceConfig, name: str, args: Dict[str, Any]) -> Any:
    async def go() -> Any:
        async with Client(build_server(cfg)) as client:
            return await client.call_tool(name, args)

    return _run(go)


def test_tool_list(cfg: ServiceConfig) -> None:
    """注册的工具正好是 7 个, 每个都有描述."""

    async def go() -> Any:
        async with Client(build_server(cfg)) as client:
            return await client.list_tools()

    tools = _run(go).tools
    assert {t.name for t in tools} == TOOLS
    assert all(t.description for t in tools)


def test_structured_results_and_errors(cfg: ServiceConfig) -> None:
    """成功有 structured_content; ServiceError 变成 is_error."""
    ok = _call(cfg, "search_api", {"query": "buy"})
    assert ok.is_error is False
    assert ok.structured_content is not None
    names = [r["name"] for r in ok.structured_content["results"]]
    assert "Strategy.buy" in names
    bad = _call(cfg, "get_api", {"name": "Strategy.buyy"})
    assert bad.is_error is True
    assert bad.structured_content is None
    assert "Strategy.buy" in bad.content[0].text
    wrong_type = _call(cfg, "search_api", {"query": "buy", "limit": "many"})
    assert wrong_type.is_error is True


def test_unexpected_exception_carries_detail(
    cfg: ServiceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """非 ServiceError 异常也把类型和信息带给模型."""

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr("akquant.mcp.service.api_index.search_api", boom)
    out = _call(cfg, "search_api", {"query": "buy"})
    assert out.is_error is True
    assert "内部错误 RuntimeError: boom" in out.content[0].text


def test_unexpected_resource_error_is_readable(cfg: ServiceConfig) -> None:
    """未知 topic 的 resource 读取失败时带可读信息."""

    async def go() -> Any:
        async with Client(build_server(cfg)) as client:
            return await client.read_resource("akquant://guide/nope")

    with pytest.raises(Exception) as info:
        _run(go)
    assert "strategy-skeleton" in repr(info.value)


def test_full_loop(cfg: ServiceConfig, write_file: Callable[[str, str], Path]) -> None:
    """全链路: guide -> check -> prepare -> run -> details."""
    topics = _call(cfg, "get_guide", {}).structured_content["topics"]
    assert len(topics) == 5
    write_file("buyer.py", BUYER)
    checked = _call(cfg, "check_strategy", {"path": "buyer.py"})
    assert checked.structured_content["ok"] is True
    spec = {"source": "synthetic", "end": "2023-03-31"}
    prepared = _call(cfg, "prepare_data", {"spec": spec})
    dataset_id = prepared.structured_content["dataset_id"]
    run_args = {"path": "buyer.py", "dataset_id": dataset_id}
    run = _call(cfg, "run_backtest", run_args).structured_content
    assert run["status"] == "ok", run
    detail_args = {"run_id": run["run_id"], "section": "orders"}
    details = _call(cfg, "get_run_details", detail_args)
    assert details.structured_content["total"] >= 1
    bad_args = {**run_args, "config": {"nope": 1}}
    bad = _call(cfg, "run_backtest", bad_args)
    assert bad.is_error is True


def test_guide_resource(cfg: ServiceConfig) -> None:
    """写策略要点以模板 resource 暴露."""

    async def go() -> Any:
        async with Client(build_server(cfg)) as client:
            templates = await client.list_resource_templates()
            content = await client.read_resource("akquant://guide/indicators")
            return templates, content

    templates, content = _run(go)
    uris = [t.uri_template for t in templates.resource_templates]
    assert "akquant://guide/{topic}" in uris
    assert content.contents[0].text.startswith("# ")


def test_stdio_smoke(tmp_path: Path) -> None:
    """真实 stdio 子进程: 协议帧不被 akquant 的输出污染, 跑通一次回测."""
    (tmp_path / "buyer.py").write_text(BUYER, encoding="utf-8")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "akquant.mcp", "--root", str(tmp_path)],
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )

    async def go() -> Any:
        async with Client(params) as client:
            spec = {"source": "synthetic", "end": "2023-03-31"}
            prepared = await client.call_tool("prepare_data", {"spec": spec})
            dataset_id = prepared.structured_content["dataset_id"]
            args = {"path": "buyer.py", "dataset_id": dataset_id}
            return await client.call_tool("run_backtest", args)

    result = _run(go)
    assert result.is_error is False
    assert result.structured_content["status"] == "ok"
