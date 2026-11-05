"""akquant.mcp.service.check: 策略静态检查."""

from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest
from akquant.mcp.service import ServiceError
from akquant.mcp.service.check import check_strategy
from akquant.mcp.service.paths import ServiceConfig

WriteFile = Callable[[str, str], Path]

CLEAN = """\
import akquant as aq
from akquant import Bar, IntParam, Strategy


class Clean(Strategy):
    n = IntParam(3)

    def on_start(self) -> None:
        self.count = 0
        self.ema = self.I(aq.EMA(10), name="ema")

    def on_bar(self, bar: Bar) -> None:
        self.count += 1
        self.helper()
        if self.count == self.params.n:
            self.buy(symbol=bar.symbol, quantity=100)

    def helper(self) -> None:
        pass
"""


def _codes(result: Dict[str, Any]) -> List[str]:
    return [d["code"] for d in result["diagnostics"]]


def test_clean_strategy(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """规范写法无诊断."""
    out = check_strategy(cfg, write_file("s.py", CLEAN))
    assert out["ok"] is True
    assert out["selected"] == "Clean"
    assert out["diagnostics"] == []
    assert [c["name"] for c in out["classes"]] == ["Clean"]


def test_syntax_error(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """语法错误带行号."""
    src = "from akquant import Strategy\n\n\nclass S(Strategy)\n    pass\n"
    out = check_strategy(cfg, write_file("s.py", src))
    assert out["ok"] is False
    assert _codes(out) == ["syntax-error"]
    assert out["diagnostics"][0]["line"] == 4


def test_functional_file_is_no_strategy(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """只有函数式 on_bar: no-strategy, hint 提示改写为类式."""
    src = "def on_bar(ctx, bar):\n    pass\n"
    out = check_strategy(cfg, write_file("s.py", src))
    assert _codes(out) == ["no-strategy"]
    assert "类式" in out["diagnostics"][0]["hint"]


def test_ambiguous_and_strategy_attr(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """多个 Strategy 子类需要 strategy_attr; 指定后只检查该类."""
    src = (
        "from akquant import Strategy\n\n\n"
        "class A(Strategy):\n    pass\n\n\n"
        "class B(Strategy):\n    pass\n"
    )
    path = write_file("s.py", src)
    out = check_strategy(cfg, path)
    assert _codes(out) == ["ambiguous-strategy"]
    assert out["ok"] is False
    picked = check_strategy(cfg, path, strategy_attr="B")
    assert picked["ok"] is True
    assert picked["selected"] == "B"
    missing = check_strategy(cfg, path, strategy_attr="C")
    assert _codes(missing) == ["no-strategy"]


def test_unknown_method_and_kwarg(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """拼错的方法是 warning; Strategy 方法的错误关键字是 error."""
    src = (
        "from akquant import Strategy\n\n\n"
        "class S(Strategy):\n"
        "    def on_bar(self, bar):\n"
        "        self.biy(bar.symbol)\n"
        "        self.buy(symbol=bar.symbol, qty=100)\n"
    )
    out = check_strategy(cfg, write_file("s.py", src))
    assert _codes(out) == ["unknown-self-method", "unknown-kwarg"]
    assert out["diagnostics"][0]["severity"] == "warning"
    assert out["diagnostics"][1]["severity"] == "error"
    assert out["diagnostics"][1]["line"] == 7
    assert "quantity" in out["diagnostics"][1]["hint"]
    assert out["ok"] is False


def test_i_outside_init_and_plain_init_not_flagged(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """on_bar 里的 self.I 是 warning; 不接受 **kwargs 的 __init__ 不再报."""
    src = (
        "import akquant as aq\n"
        "from akquant import Strategy\n\n\n"
        "class S(Strategy):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n\n"
        "    def on_bar(self, bar):\n"
        "        self.I(aq.EMA(5), name='e')\n"
    )
    out = check_strategy(cfg, write_file("s.py", src))
    assert _codes(out) == ["i-call-outside-init"]
    assert out["ok"] is True


def test_kwargs_init_and_inherited_and_assigned_attrs(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """接受 **kwargs 的 __init__、同文件父类方法、self.x 赋值的可调用都不报."""
    src = (
        "from akquant import Strategy\n\n\n"
        "class Base(Strategy):\n"
        "    def shared(self):\n"
        "        pass\n\n\n"
        "class S(Base):\n"
        "    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n\n"
        "    def on_start(self):\n"
        "        self.cb = print\n\n"
        "    def on_bar(self, bar):\n"
        "        self.shared()\n"
        "        self.cb('x')\n"
    )
    out = check_strategy(cfg, write_file("s.py", src), strategy_attr="S")
    assert out["diagnostics"] == []


def test_path_errors(cfg: ServiceConfig, tmp_path: Path) -> None:
    """文件不存在、越界都报 ServiceError."""
    with pytest.raises(ServiceError, match="不存在"):
        check_strategy(cfg, "missing.py")
    with pytest.raises(ServiceError):
        check_strategy(cfg, tmp_path.parent / "x.py")


SIBLING_BASE = """\
from akquant import Strategy


class MyBase(Strategy):
    def helper(self):
        pass
"""

SIBLING_CHILD = """\
from base import MyBase


class S(MyBase):
    def on_bar(self, bar):
        self.my_helper()
"""


def test_non_utf8_file_is_syntax_error(cfg: ServiceConfig, tmp_path: Path) -> None:
    """GBK 编码文件: 不抛原始异常, 给出 UTF-8 提示."""
    path = tmp_path / "gbk.py"
    path.write_bytes("# 中文\n".encode("gbk") + b"from akquant import Strategy\n")
    out = check_strategy(cfg, path)
    assert out["ok"] is False
    assert _codes(out) == ["syntax-error"]
    assert "UTF-8" in out["diagnostics"][0]["hint"]


def test_nul_byte_file_no_raw_exception(cfg: ServiceConfig, tmp_path: Path) -> None:
    """含 NUL 字节的文件按语法错误处理."""
    path = tmp_path / "nul.py"
    path.write_bytes(b"x = 1\x00\n")
    out = check_strategy(cfg, path)
    assert out["ok"] is False
    assert "syntax-error" in _codes(out)


def test_sibling_file_base_is_unresolved_warning(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """基类来自同目录其他文件: 警告 unresolved-base, 不报 no-strategy."""
    write_file("base.py", SIBLING_BASE)
    path = write_file("s.py", SIBLING_CHILD)
    out = check_strategy(cfg, path)
    codes = _codes(out)
    assert out["ok"] is True
    assert "unresolved-base" in codes
    assert "unknown-self-method" not in codes
    assert "no-strategy" not in codes
    picked = check_strategy(cfg, path, strategy_attr="S")
    assert picked["ok"] is True
    assert picked["selected"] == "S"
    assert "unknown-self-method" not in _codes(picked)


def test_star_import_strategy_is_not_no_strategy(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """From akquant import * 后继承 Strategy: 不报 no-strategy."""
    src = "from akquant import *\n\n\nclass S(Strategy):\n    pass\n"
    out = check_strategy(cfg, write_file("star.py", src))
    assert out["ok"] is True
    assert "no-strategy" not in _codes(out)


def test_enum_only_file_is_no_strategy(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """只有 Enum 的文件不是候选策略: no-strategy, 不出 unresolved-base."""
    src = (
        "import enum\nfrom enum import IntEnum\n\n\n"
        "class Side(enum.Enum):\n    BUY = 1\n\n\n"
        "class Level(IntEnum):\n    LOW = 1\n"
    )
    out = check_strategy(cfg, write_file("enum_only.py", src))
    assert out["ok"] is False
    assert _codes(out) == ["no-strategy"]


def test_functional_plus_enum_keeps_class_hint(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """函数式 on_bar 加 Enum: 仍给出“一期只支持类式策略”的提示."""
    src = (
        "from enum import Enum\n\n\n"
        "class Side(Enum):\n    BUY = 1\n\n\n"
        "def on_bar(ctx, bar):\n    pass\n"
    )
    out = check_strategy(cfg, write_file("func_enum.py", src))
    assert _codes(out) == ["no-strategy"]
    assert "类式" in out["diagnostics"][0]["hint"]


def test_ambiguous_ignores_exception_helper(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """Ambiguous 时不扫描辅助类: class E(Exception) 里的 self 调用不报."""
    src = (
        "from akquant import Strategy\n\n\n"
        "class E(Exception):\n"
        "    def __init__(self, m):\n        self.m = m\n        self.nope()\n\n\n"
        "class A(Strategy):\n    pass\n\n\n"
        "class B(Strategy):\n    pass\n"
    )
    out = check_strategy(cfg, write_file("amb_exc.py", src))
    assert _codes(out) == ["ambiguous-strategy"]


@pytest.mark.parametrize(
    "helper",
    [
        "from pydantic import BaseModel\n\n\nclass Cfg(BaseModel):\n    n: int = 1\n",
        "from typing import NamedTuple\n\n\nclass Pt(NamedTuple):\n    x: int\n",
        "import typing\n\n\nclass Td(typing.TypedDict):\n    x: int\n",
        "from typing import Generic, TypeVar\n\nT = TypeVar('T')\n\n\n"
        "class Box(Generic[T]):\n    pass\n",
        "import abc\n\n\nclass Ab(abc.ABC):\n    pass\n",
        "class MyErr(ValueError):\n    pass\n\n\nclass Sub(MyErr):\n    pass\n",
        "class Weird(Exception):\n    pass\n\n\nclass Err2(RuntimeError, Weird):\n"
        "    pass\n",
    ],
)
def test_helper_classes_next_to_strategy_are_silent(
    cfg: ServiceConfig, write_file: WriteFile, helper: str
) -> None:
    """辅助类 + 一个正常策略: 选中策略, 没有任何诊断."""
    out = check_strategy(cfg, write_file("helper.py", helper + "\n\n" + CLEAN))
    assert out["ok"] is True
    assert out["selected"] == "Clean"
    assert out["diagnostics"] == []


def test_resolvable_strategy_plus_cross_file_candidate_warns(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """一个可解析策略加一个跨文件候选: ok 为 True, 但给 unresolved-base 警告."""
    write_file("base.py", SIBLING_BASE)
    src = "from base import MyBase\n\n\nclass Other(MyBase):\n    pass\n\n\n" + CLEAN
    out = check_strategy(cfg, write_file("two.py", src))
    assert out["ok"] is True
    assert out["selected"] == "Clean"
    warns = [d for d in out["diagnostics"] if d["code"] == "unresolved-base"]
    assert len(warns) == 1
    assert warns[0]["severity"] == "warning"
    assert "Other" in warns[0]["message"]
    assert "strategy_attr" in warns[0]["hint"]
