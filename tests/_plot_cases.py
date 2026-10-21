"""self.I 绘图默认值的共用真值表: 运行时与静态检查两边的测试都由它驱动(spec §7)."""

from typing import Optional

#: (is_indicator_script, overlay, plot, pane, (resolved_plot, resolved_pane))
CASES: list[tuple[bool, bool, Optional[bool], Optional[int], tuple[bool, int]]] = [
    (False, False, None, None, (False, 0)),
    (False, False, None, 2, (False, 2)),
    (False, False, True, None, (True, 0)),
    (False, False, True, 3, (True, 3)),
    (True, False, None, None, (True, 1)),
    (True, True, None, None, (True, 0)),
    (True, False, None, 0, (True, 0)),
    (True, True, None, 2, (True, 2)),
    (True, False, False, None, (False, 1)),
    (True, True, False, None, (False, 0)),
]
