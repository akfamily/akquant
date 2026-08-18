"""因异常中止的实盘会话, 收尾摘要不能自称 "Manual Stop".

``run_live`` 在策略回调抛异常时会打 ``CRITICAL`` + traceback 并停止事件处理
(设计如此: 实盘不把异常继续往上抛), 但紧随其后的收尾摘要标题此前硬编码为
``TRADING SUMMARY (Manual Stop)`` —— 一次因错误中止的会话看起来像是正常手动停止,
而那条 CRITICAL 往往已被几十行日志淹没。
"""

from typing import Any, List

import pandas as pd
import pytest
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar

SYM = "600016.SH"


def _instrument() -> Instrument:
    return Instrument(
        symbol=SYM,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
    )


def _bars() -> List[Bar]:
    return [
        Bar(
            timestamp=int(
                pd.Timestamp(f"2023-01-{day:02d} 14:00:00", tz="Asia/Shanghai").value
            ),
            open=10.0,
            high=10.2,
            low=9.8,
            close=10.0,
            volume=1_000_000.0,
            symbol=SYM,
        )
        for day in (3, 4, 5)
    ]


class _Boom(Strategy):
    """第一根 bar 就抛异常."""

    def on_bar(self, bar: Bar) -> None:
        raise RuntimeError("boom-from-on-bar")


class _Quiet(Strategy):
    """什么都不做, 正常跑完."""

    def on_bar(self, bar: Bar) -> None:
        _ = bar


def _run(strategy: Any) -> None:
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars()},
        cash=1_000_000.0,
        show_progress=False,
        duration="60s",
    )


def test_summary_marks_error_abort(caplog: pytest.LogCaptureFixture) -> None:
    """回调异常导致中止时, 摘要标题必须点出是错误中止."""
    with caplog.at_level("INFO"):
        _run(_Boom())
    assert "TRADING SUMMARY" in caplog.text, "未打印收尾摘要"
    assert "Manual Stop" not in caplog.text, (
        "因错误中止的会话仍自称 Manual Stop, 会掩盖故障"
    )


def test_normal_session_summary_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    """未出错的会话保持既有标题(本次修复不改变正常路径的输出)."""
    with caplog.at_level("INFO"):
        _run(_Quiet())
    assert "TRADING SUMMARY" in caplog.text, "未打印收尾摘要"
    assert "ERROR" not in caplog.text.split("TRADING SUMMARY")[1][:40], (
        "正常会话的摘要被误标为错误中止"
    )
