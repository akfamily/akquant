"""``akquant`` 命令入口.

目前只有一个子命令 ``akquant mcp``: 以 stdio 方式启动 MCP 服务, 与
``akquant-mcp`` 完全相同. MCP 官方注册表的客户端按包名启动服务
(``uvx --with akquant[mcp]==X.Y.Z akquant mcp --root=...``), 所以需要这个命令.

stdio 方式下 stdout 只能输出协议消息, 本模块不向 stdout 打印任何东西
(``--version`` 与用法提示除外, 它们不会和服务同时运行).
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _version() -> str:
    from . import __version__

    return __version__


def main(argv: Optional[List[str]] = None) -> int:
    """解析子命令并执行; 返回进程退出码."""
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="akquant",
        description="AKQuant 命令行。子命令: mcp(以 stdio 方式启动 MCP 服务)",
    )
    parser.add_argument("--version", action="version", version=_version())
    parser.add_argument("command", choices=["mcp"], help="子命令")
    if args and args[0] == "mcp":
        # 其余参数原样交给 akquant-mcp, 由它自己解析(含 --root=... 写法)
        from .mcp import __main__ as mcp_entry

        return int(mcp_entry.main(args[1:], prog="akquant mcp"))
    parser.parse_args(args)  # 无子命令、未知子命令或 --version: 由 argparse 处理并退出
    return 2


if __name__ == "__main__":
    sys.exit(main())
