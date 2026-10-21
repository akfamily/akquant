"""self.I 的绘图默认值: 由脚本类型决定, plot= 显式覆盖, 绘图参数不再隐含 plot=True."""

from typing import Any, Optional

import akquant as aq
import pytest
from _plot_cases import CASES
from akquant.indicator_declaration import resolve_plot_defaults


@pytest.mark.parametrize("script,overlay,plot,pane,expected", CASES)
def test_resolve_plot_defaults_table(
    script: bool,
    overlay: bool,
    plot: Optional[bool],
    pane: Optional[int],
    expected: tuple[bool, int],
) -> None:
    """纯函数真值表."""
    assert (
        resolve_plot_defaults(
            is_indicator_script=script, overlay=overlay, plot=plot, pane=pane
        )
        == expected
    )


def _declare(cls: type[aq.Strategy], **kwargs: Any) -> aq.IndicatorDeclaration:
    inst = cls()
    inst.I(aq.SMA(3), name="x", **kwargs)
    return inst._incremental_indicators["x"]


class _Strat(aq.Strategy):
    pass


class _Sub(aq.Indicator):
    pass


class _Over(aq.Indicator, overlay=True):
    pass


@pytest.mark.parametrize(
    "cls,kwargs,expected",
    [
        (_Strat, {}, (False, 0)),
        (_Strat, {"color": "red"}, (False, 0)),  # 绘图参数不再隐含 plot=True
        (_Strat, {"pane": 1}, (False, 1)),
        (_Strat, {"plot": True}, (True, 0)),
        (_Sub, {}, (True, 1)),
        (_Sub, {"pane": 0}, (True, 0)),
        (_Sub, {"plot": False}, (False, 1)),
        (_Over, {}, (True, 0)),
        (_Over, {"pane": 2}, (True, 2)),
    ],
)
def test_self_i_uses_script_type_defaults(
    cls: type, kwargs: dict, expected: tuple[bool, int]
) -> None:
    """self.I 按脚本类型落地 (plot, pane)."""
    decl = _declare(cls, **kwargs)
    assert (decl.plot, decl.pane) == expected
