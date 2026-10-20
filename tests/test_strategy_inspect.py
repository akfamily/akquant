"""aq.inspect_strategy: 纯静态解析, 不执行用户代码."""

import json
from pathlib import Path
from typing import Any, Optional

import akquant as aq
import pytest
from _plot_cases import CASES
from akquant.strategy_inspect import ClassInfo

DUAL = """
import akquant as aq

class DualMA(aq.Indicator, title="双均线", overlay=True):
    def on_start(self):
        self.fast = self.I(aq.EMA(10), label="EMA10")
        self.slow = self.I(aq.EMA(30))
        self.rsi = self.I(aq.RSI(14), name="rsi", pane=1)
        self.atr = self.I(aq.ATR(14), input_mode="hlc", plot=False)

class Trend(aq.Strategy, title="趋势策略"):
    def on_start(self):
        self.fast = self.I(aq.EMA(10))
        self.macd = self.I(aq.MACD(12, 26, 9), name="macd", plot=True, pane=2,
                           outputs=("dif", "dea", "hist"))

class Helper:
    pass
"""


def _cls(result: aq.InspectResult, name: str) -> ClassInfo:
    return next(c for c in result.classes if c.name == name)


def test_indicator_script_classified_with_resolved_panes() -> None:
    """指标脚本: 类别、元数据与 pane/plot 默认值按脚本类型解析."""
    res = aq.inspect_strategy(DUAL)
    assert res.error is None
    assert [c.name for c in res.classes] == ["DualMA", "Trend"]  # Helper 不列出
    dual = _cls(res, "DualMA")
    assert (dual.kind, dual.title, dual.id, dual.overlay) == (
        "indicator",
        "双均线",
        "dual_ma",
        True,
    )
    got = [(i.name, i.pane, i.plot, i.label) for i in dual.indicators]
    assert got == [
        ("ema", 0, True, "EMA10"),
        ("ema_2", 0, True, None),
        ("rsi", 1, True, None),
        ("atr", 0, False, None),
    ]
    assert dual.draws_indicators is True
    assert dual.has_dynamic is False


def test_strategy_defaults_not_plotted() -> None:
    """策略里的 self.I 默认不画, 显式 plot=True 才画."""
    trend = _cls(aq.inspect_strategy(DUAL), "Trend")
    assert (trend.kind, trend.title, trend.id, trend.overlay) == (
        "strategy",
        "趋势策略",
        None,
        None,
    )
    assert [(i.name, i.pane, i.plot) for i in trend.indicators] == [
        ("ema", 0, False),
        ("macd", 2, True),
    ]
    assert trend.indicators[1].outputs == ["dif", "dea", "hist"]
    assert trend.draws_indicators is True


def test_strategy_without_plot_does_not_draw() -> None:
    """策略只写 pane= 不再隐含 plot=True."""
    src = (
        "import akquant as aq\n"
        "class S(aq.Strategy):\n"
        "    def on_start(self):\n"
        "        self.x = self.I(aq.SMA(5), pane=1)\n"
    )
    s = _cls(aq.inspect_strategy(src), "S")
    assert s.indicators[0].plot is False
    assert s.draws_indicators is False


def test_import_aliases_and_qmfquant() -> None:
    """识别 from-import 别名与 qmfquant 模块名, 同文件继承也能定类别."""
    src = (
        "from akquant import Indicator as Ind\n"
        "import qmfquant as qq\n"
        "class A(Ind):\n    pass\n"
        "class B(qq.Strategy):\n    pass\n"
        "class C(A, overlay=True):\n    pass\n"
    )
    res = aq.inspect_strategy(src)
    kinds = {c.name: (c.kind, c.overlay) for c in res.classes}
    assert kinds == {
        "A": ("indicator", False),
        "B": ("strategy", None),
        "C": ("indicator", True),
    }


def test_overlay_inherited_within_file_and_title_not() -> None:
    """类定义行的 overlay 沿同文件父类继承, title/id 不继承."""
    src = (
        "import akquant as aq\n"
        "class Base(aq.Indicator, title='基', overlay=True):\n    pass\n"
        "class Child(Base):\n    pass\n"
    )
    child = _cls(aq.inspect_strategy(src), "Child")
    assert (child.overlay, child.title, child.id) == (True, "Child", "child")


def test_unknown_base_from_other_file() -> None:
    """基类来自别的文件: 有 self.I 的记 unknown, 没有的不列出."""
    src = (
        "from mylib import MyBase\n"
        "class X(MyBase):\n"
        "    def on_start(self):\n        self.a = self.I(SMA(5))\n"
        "class Y(MyBase):\n    pass\n"
    )
    res = aq.inspect_strategy(src)
    assert [(c.name, c.kind) for c in res.classes] == [("X", "unknown")]
    x = res.classes[0]
    assert x.indicators[0].plot is None and "plot" in x.indicators[0].dynamic
    assert x.draws_indicators is True  # 无法确定时按"可能画"处理


def test_dynamic_values_marked_not_guessed() -> None:
    """非字面量、循环与 factory 标记为 dynamic, 不猜."""
    src = (
        "import akquant as aq\n"
        "from functools import partial\n"
        "class D(aq.Indicator, title=TITLE):\n"
        "    def on_start(self):\n"
        "        self.a = self.I(aq.SMA(self.params.n), pane=self.params.p)\n"
        "        for n in (5, 10):\n"
        "            self.I(aq.SMA(n), name=f'sma{n}')\n"
        "        self.b = self.I(factory=partial(aq.SMA, 5))\n"
    )
    d = _cls(aq.inspect_strategy(src), "D")
    assert "title" in d.dynamic and d.has_dynamic
    a, loop, fac = d.indicators
    assert a.name == "sma" and a.pane is None and "pane" in a.dynamic
    assert loop.name is None and {"name", "count"} <= set(loop.dynamic)
    assert "count" in fac.dynamic


def test_syntax_error_reported_not_raised() -> None:
    """语法错误写进 error, 不抛异常."""
    res = aq.inspect_strategy("class X(:\n")
    assert res.classes == []
    assert res.error is not None and res.error.startswith("SyntaxError")


def test_path_input_and_missing_py_path(tmp_path: Path) -> None:
    """接受 Path/路径字符串; .py 路径不存在报 FileNotFoundError."""
    f = tmp_path / "s.py"
    f.write_text(DUAL, encoding="utf-8")
    assert [c.name for c in aq.inspect_strategy(f).classes] == ["DualMA", "Trend"]
    assert [c.name for c in aq.inspect_strategy(str(f)).classes] == ["DualMA", "Trend"]
    with pytest.raises(FileNotFoundError):
        aq.inspect_strategy(str(tmp_path / "missing.py"))


def test_does_not_execute_user_code(tmp_path: Path) -> None:
    """只解析不执行: 源码里的副作用不会发生."""
    marker = tmp_path / "ran"
    src = (
        f"open({str(marker)!r}, 'w').close()\n"
        "import akquant as aq\n"
        "class X(aq.Indicator):\n"
        "    pass\n"
    )
    aq.inspect_strategy(src)
    assert not marker.exists()


def test_to_dict_is_json_serializable() -> None:
    """to_dict() 可直接 JSON 序列化, 字段集合固定."""
    payload = aq.inspect_strategy(DUAL).to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert '"kind": "indicator"' in text
    assert set(payload) == {"classes", "error"}
    assert set(payload["classes"][0]) == {
        "name",
        "kind",
        "title",
        "id",
        "overlay",
        "draws_indicators",
        "indicators",
        "has_dynamic",
        "dynamic",
    }


def test_static_matches_runtime_for_plain_literals() -> None:
    """同一段源码, 静态结果与真实实例化后的声明逐项一致(防两套规则漂移)."""
    namespace: dict[str, Any] = {}
    exec(compile(DUAL, "<dual>", "exec"), namespace)  # 测试里执行可信源码
    for cls_name in ("DualMA", "Trend"):
        inst = namespace[cls_name]()
        inst.on_start()
        runtime = [(k, d.pane, d.plot) for k, d in inst._incremental_indicators.items()]
        static = [
            (i.name, i.pane, i.plot)
            for i in _cls(aq.inspect_strategy(DUAL), cls_name).indicators
        ]
        assert static == runtime


def _case_source(
    script: bool, overlay: bool, plot: Optional[bool], pane: Optional[int]
) -> str:
    base = f"aq.Indicator, overlay={overlay!r}" if script else "aq.Strategy"
    extra = ""
    if plot is not None:
        extra += f", plot={plot!r}"
    if pane is not None:
        extra += f", pane={pane!r}"
    return (
        "import akquant as aq\n"
        f"class K({base}):\n"
        "    def on_start(self):\n"
        f"        self.x = self.I(aq.SMA(3), name='x'{extra})\n"
    )


@pytest.mark.parametrize("script,overlay,plot,pane,expected", CASES)
def test_static_follows_shared_plot_default_cases(
    script: bool,
    overlay: bool,
    plot: Optional[bool],
    pane: Optional[int],
    expected: tuple[bool, int],
) -> None:
    """与 test_plot_defaults 共用一张真值表: 静态 == 运行时 == 期望(spec §7)."""
    src = _case_source(script, overlay, plot, pane)
    (static,) = _cls(aq.inspect_strategy(src), "K").indicators
    namespace: dict[str, Any] = {}
    exec(compile(src, "<case>", "exec"), namespace)  # 测试里执行可信源码
    inst = namespace["K"]()
    inst.on_start()
    runtime = inst._incremental_indicators["x"]
    assert (static.plot, static.pane) == (runtime.plot, runtime.pane) == expected


# ---- Fix round 1 回归测试 ----


def _names(src: str, cls_name: str = "S") -> list[tuple[object, bool]]:
    c = _cls(aq.inspect_strategy(src), cls_name)
    return [(i.name, "name" in i.dynamic) for i in c.indicators]


def test_auto_name_only_from_known_callees() -> None:
    """自动名基名只从 akquant 导出或同文件类推导, 未知可调用不猜."""
    src = (
        "import akquant as aq\n"
        "from akquant import SMA as S\n"
        "from helpers import make\n"
        "class Local:\n    pass\n"
        "class S1(aq.Strategy):\n"
        "    def on_start(self):\n"
        "        self.I(Local(3))\n"
        "        self.I(S(3))\n"
        "        self.I(make(3))\n"
        "        self.I(Local(3))\n"
    )
    assert _names(src, "S1") == [
        ("local", False),
        ("sma", False),
        (None, True),
        (None, True),  # 前面基名未知: 序号不再可靠
    ]


def test_factory_name_only_from_known_defs() -> None:
    """factory= 只认同文件 def/class、akquant 导出与 lambda."""
    src = (
        "import akquant as aq\n"
        "from helpers import build\n"
        "def _make_fast():\n    return aq.SMA(3)\n"
        "class S(aq.Strategy):\n"
        "    def on_start(self):\n"
        "        self.I(factory=_make_fast)\n"
        "        self.I(factory=lambda: aq.SMA(3))\n"
        "        self.I(factory=aq.SMA)\n"
        "        self.I(factory=build)\n"
    )
    assert _names(src) == [
        ("make_fast", False),
        ("indicator", False),
        ("sma", False),
        (None, True),
    ]


def test_conditional_call_makes_later_same_base_unsure() -> None:
    """分支(if/短路)里的声明自己可推名, 之后同基名的序号不可靠."""
    src = (
        "import akquant as aq\n"
        "class S(aq.Strategy):\n"
        "    def on_start(self):\n"
        "        self.I(aq.SMA(3))\n"
        "        if FLAG:\n"
        "            self.I(aq.SMA(5))\n"
        "        self.I(aq.SMA(7))\n"
        "        self.I(aq.EMA(7))\n"
        "        x = FLAG and self.I(aq.EMA(9))\n"
        "        self.I(aq.EMA(11))\n"
    )
    assert _names(src) == [
        ("sma", False),
        ("sma_2", False),
        (None, True),
        ("ema", False),
        ("ema_2", False),
        (None, True),
    ]


def test_init_declarations_counted_before_on_start() -> None:
    """__init__ 先于 on_start 执行: 编号按运行时顺序, 不按源码顺序."""
    src = (
        "import akquant as aq\n"
        "class S(aq.Strategy):\n"
        "    def on_start(self):\n"
        "        self.I(aq.SMA(3), pane=2)\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.I(aq.SMA(5), pane=1)\n"
    )
    assert _names(src) == [("sma", False), ("sma_2", False)]
    c = _cls(aq.inspect_strategy(src), "S")
    ns: dict[str, Any] = {}
    exec(compile(src, "<s>", "exec"), ns)
    inst = ns["S"]()
    inst.on_start()
    runtime = [(k, d.pane) for k, d in inst._incremental_indicators.items()]
    assert runtime == [("sma", 1), ("sma_2", 2)]  # __init__ 里那条(pane=1)先编号
    assert [(i.name, i.pane) for i in c.indicators] == runtime


def test_helper_method_declaration_is_conditional() -> None:
    """其他方法里的声明执行时机未知: 该基名的自动名全部不可靠."""
    src = (
        "import akquant as aq\n"
        "class S(aq.Strategy):\n"
        "    def on_start(self):\n"
        "        self.I(aq.SMA(3))\n"
        "        self.I(aq.EMA(3))\n"
        "    def _more(self):\n"
        "        self.I(aq.SMA(5))\n"
    )
    assert _names(src) == [(None, True), ("ema", False), (None, True)]


def test_kwargs_unpacking_marks_unwritten_fields_dynamic() -> None:
    """**opts 可能带任何字段: 没显式写的都记为动态."""
    src = (
        "import akquant as aq\n"
        "class S(aq.Strategy):\n"
        "    def on_start(self):\n"
        "        self.I(aq.EMA(3), label='e', **opts)\n"
    )
    c = _cls(aq.inspect_strategy(src), "S")
    (i,) = c.indicators
    assert i.plot is None and i.pane is None and i.name is None
    assert {"plot", "pane", "name", "outputs"} <= set(i.dynamic)
    assert "label" not in i.dynamic and i.label == "e"
    assert c.draws_indicators is True


def test_bom_file_parses(tmp_path: Path) -> None:
    """带 BOM 的 UTF-8 文件(Windows 记事本常见)能正常解析."""
    f = tmp_path / "bom.py"
    f.write_bytes(DUAL.encode("utf-8-sig"))
    res = aq.inspect_strategy(f)
    assert res.error is None
    assert [c.name for c in res.classes] == ["DualMA", "Trend"]


def test_title_id_strip_and_overlay_none() -> None:
    """title/id 去首尾空白, 空串记为动态; overlay=None 视为没写(继承)."""
    src = (
        "import akquant as aq\n"
        "class P(aq.Indicator, overlay=True):\n    pass\n"
        "class C(P, title='  趋势 ', id='  ', overlay=None):\n    pass\n"
    )
    c = _cls(aq.inspect_strategy(src), "C")
    assert (c.title, c.id, c.overlay) == ("趋势", None, True)
    assert c.dynamic == ["id"]
