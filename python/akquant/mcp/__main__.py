"""``akquant-mcp`` 命令入口: 以 stdio 方式启动 AKQuant MCP 服务."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .service import ServiceError
from .service.paths import DEFAULT_MAX_RUNS, make_config

INSTALL_HINT = '缺少 MCP SDK(需要 mcp>=2.3), 请先安装: pip install "akquant[mcp]"'


def main(argv: Optional[List[str]] = None) -> int:
    """解析参数并启动服务; 失败时向 stderr 说明原因并返回 1."""
    parser = argparse.ArgumentParser(
        prog="akquant-mcp", description="AKQuant MCP 服务(stdio)"
    )
    parser.add_argument("--root", default=None, help="项目根目录, 默认当前目录")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=DEFAULT_MAX_RUNS,
        help="保留最近多少次回测结果",
    )
    args = parser.parse_args(argv)
    try:
        from .server import build_server
    except ImportError as exc:
        is_mcp = (exc.name or "").split(".")[0] == "mcp"
        if not is_mcp and "MCPServer" not in str(exc):
            raise
        print(INSTALL_HINT, file=sys.stderr)
        return 1
    try:
        cfg = make_config(root=args.root, max_runs=args.max_runs)
    except ServiceError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    build_server(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
