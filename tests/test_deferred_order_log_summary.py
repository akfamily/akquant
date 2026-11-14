# -*- coding: utf-8 -*-
"""同周期订单被延后撮合的告警: 每次回测只告警首笔并在结束时汇总一条.

多标的策略一次回测里这类延后可达数千笔, 逐条 WARNING 会淹没日志。首笔仍按
WARNING 输出(带完整订单上下文), 其余降为 DEBUG, 运行结束时汇总笔数。
"""

import logging
import re
from typing import Any, List

import akquant as aq
import numpy as np
import pandas as pd
import pytest
from akquant import PrecomputedIndicator, Strategy

PER_ORDER = "Deferred same-cycle order until cross-symbol reduce-first orders"
SUMMARY = re.compile(r"Deferred (\d+) same-cycle orders in this run")


def _data(n_sym: int, n_day: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    days = pd.date_range("2022-01-03", periods=n_day, freq="B")
    frames = []
    for i in range(n_sym):
        close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n_day)))
        frames.append(
            pd.DataFrame(
                {
                    "date": days.tz_localize("Asia/Shanghai"),
                    "symbol": f"S{i:02d}",
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1e6,
                }
            )
        )
    return pd.concat(frames).sort_values(["date", "symbol"]).reset_index(drop=True)


class _Cross(Strategy):
    def on_start(self) -> None:
        self.fast = self.I(
            PrecomputedIndicator("fast", lambda df: df["close"].rolling(3).mean()),
            name="fast",
        )
        self.slow = self.I(
            PrecomputedIndicator("slow", lambda df: df["close"].rolling(8).mean()),
            name="slow",
        )

    def on_bar(self, bar: Any) -> None:
        fast, slow = self.fast[0], self.slow[0]
        if fast is None or slow is None or fast != fast or slow != slow:
            return
        position = self.get_position(bar.symbol)
        if fast > slow and position <= 0:
            self.buy(symbol=bar.symbol, quantity=100)
        elif fast < slow and position > 0:
            self.sell(symbol=bar.symbol, quantity=100)


def _run(caplog: pytest.LogCaptureFixture, level: int) -> List[logging.LogRecord]:
    with caplog.at_level(level, logger="akquant"):
        aq.run_backtest(
            strategy=_Cross,
            data=_data(12, 120),
            initial_cash=1e7,
            commission_rate=0.0,
            show_progress=False,
        )
    return [r for r in caplog.records if "same-cycle order" in r.getMessage()]


def test_one_warning_plus_summary_per_run(caplog: pytest.LogCaptureFixture) -> None:
    """WARNING 级别: 首笔一条(带订单上下文) + 汇总一条, 汇总笔数大于 1."""
    records = [r for r in _run(caplog, logging.WARNING) if r.levelno >= logging.WARNING]
    per_order = [r for r in records if PER_ORDER in r.getMessage()]
    summaries = [r for r in records if SUMMARY.search(r.getMessage())]
    assert len(per_order) == 1
    assert getattr(per_order[0], "order_id", None), "首笔告警应带订单上下文"
    assert len(summaries) == 1
    match = SUMMARY.search(summaries[0].getMessage())
    assert match is not None
    assert int(match.group(1)) > 1
    assert len(records) == 2


_DEBUG_SCRIPT = """
import logging, re, sys
sys.path.insert(0, {tests_dir!r})
logging.basicConfig(level=logging.DEBUG)
records = []
class _Collect(logging.Handler):
    def emit(self, record):
        records.append(record)
logging.getLogger("akquant").addHandler(_Collect())
import akquant as aq
import test_deferred_order_log_summary as t
# 导入 akquant 会把它的日志级别设回 INFO, 所以在导入之后再调
logging.getLogger("akquant").setLevel(logging.DEBUG)
aq.run_backtest(strategy=t._Cross, data=t._data(12, 120), initial_cash=1e7,
                commission_rate=0.0, show_progress=False)
per_order = [r for r in records if t.PER_ORDER in r.getMessage()]
summary = [t.SUMMARY.search(r.getMessage()) for r in records]
summary = [m for m in summary if m]
warn = sum(r.levelno >= logging.WARNING for r in per_order)
print("RESULT", len(per_order), warn, len(summary), summary[0].group(1))
"""


def test_debug_level_keeps_every_deferral() -> None:
    """DEBUG 级别仍能看到每一笔, 笔数与汇总一致.

    在子进程里跑: Rust 日志桥接会缓存 Python 侧的日志级别(进程内首次输出时定下),
    同一进程里先按 WARNING 跑过再调到 DEBUG 看不到明细。
    """
    import subprocess
    import sys
    from pathlib import Path

    script = _DEBUG_SCRIPT.format(tests_dir=str(Path(__file__).parent))
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, proc.stderr[-2000:]
    per_order, warnings, summaries, total = (int(v) for v in lines[-1].split()[1:])
    assert summaries == 1
    assert total > 1
    assert per_order == total
    assert warnings == 1
