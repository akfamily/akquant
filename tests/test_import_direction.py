"""导入方向: 核心(回测、策略、执行后端)不依赖网关包.

网关依赖核心是正常方向; 反过来的话, 改网关会牵动回测, 只做回测也要加载柜台适配器。
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "python" / "akquant"

#: 允许依赖网关的子包: 网关自身, 以及实盘入口。
GATEWAY_SIDE = ("gateway", "live")

#: 核心里允许保留的网关导入 {(相对路径, 被导入模块)}。新增条目前先想能不能把
#: 被用到的东西挪到核心一侧。
ALLOWED = {
    # 下单审计日志: 信号分发与实盘下单共用同一份流水格式。
    ("signal/dispatcher.py", "akquant.gateway.order_audit"),
}


def _imported_modules(path: Path) -> set[str]:
    """返回文件里导入的 akquant 内部模块的绝对名(含函数体内的导入)."""
    relative = path.relative_to(PACKAGE_ROOT)
    package = ["akquant", *relative.parts[:-1]]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                module = ".".join([*base, *(node.module or "").split(".")]).rstrip(".")
            else:
                module = node.module or ""
            found.add(module)
            # ``from . import gateway`` 这种写法: 被导入的名字本身是子模块
            found.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def test_core_modules_do_not_import_gateway() -> None:
    """gateway/ 与 live/ 之外的模块不得导入 akquant.gateway(白名单除外)."""
    offenders: set[tuple[str, str]] = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if relative.split("/")[0] in GATEWAY_SIDE:
            continue
        for module in _imported_modules(path):
            if module == "akquant.gateway" or module.startswith("akquant.gateway."):
                # 只记到子模块一级, 忽略由 ``from x import 名字`` 拼出来的非模块名
                parts = module.split(".")
                offenders.add((relative, ".".join(parts[:3])))

    assert offenders - ALLOWED == set()
    assert ALLOWED - offenders == set(), "白名单里有已不存在的条目, 请删掉"
