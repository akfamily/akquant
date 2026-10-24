"""实盘 BarAggregator：终点戳、到期补发与旧参数迁移."""

from __future__ import annotations

import time

import pytest
from akquant import BarAggregator, DataFeed

S = 1_000_000_000


def test_stamp_false_is_rejected_with_migration_hint() -> None:
    """旧参数传 False 报迁移提示，传 True 兼容旧写法."""
    with pytest.raises(ValueError, match="终点戳"):
        BarAggregator(DataFeed(), stamp_bar_at_interval_end=False)
    BarAggregator(DataFeed(), stamp_bar_at_interval_end=True)  # 兼容旧写法


def test_flush_due_and_late_tick_count_exposed_to_python() -> None:
    """Python 侧可调用 flush_due 与 late_tick_count."""
    agg = BarAggregator(DataFeed(), 1, volume_is_cumulative=False)
    agg.on_tick("X", 10.0, 1.0, 60 * S)
    assert agg.flush_due(122 * S) == 1
    agg.on_tick("X", 10.5, 1.0, 119 * S)
    assert agg.late_tick_count("X") == 1
    assert agg.late_tick_count("Y") == 0


class _RecordingAggregator:
    """替身：记录构造参数与 flush_due 调用次数."""

    instances: list[_RecordingAggregator] = []

    def __init__(self, feed: object, **kwargs: object) -> None:
        """记录构造参数."""
        self.kwargs = kwargs
        self.flush_calls = 0
        _RecordingAggregator.instances.append(self)

    def on_tick(self, *_args: object) -> None:
        """忽略 tick."""

    def flush_due(self, now_ns: int, grace_ns: int = 2 * S) -> int:
        """记录调用次数."""
        self.flush_calls += 1
        return 0


@pytest.mark.parametrize("emit_ticks", [False, True])
def test_ctp_gateway_never_requests_start_stamp_and_flushes(
    monkeypatch: pytest.MonkeyPatch, emit_ticks: bool
) -> None:
    """只出 bar 与双流都不再传起点戳参数，且后台线程定时调用 flush_due."""
    from akquant.gateway.brokers.ctp import native

    monkeypatch.setattr(native, "HAS_OPENCTP", True)
    monkeypatch.setattr(native, "BarAggregator", _RecordingAggregator)
    _RecordingAggregator.instances.clear()
    gw = native.CTPMarketGateway(
        feed=DataFeed(),
        symbols=["rb2401"],
        front_url="tcp://test",
        emit_ticks=emit_ticks,
        emit_bars=True,
    )
    try:
        agg = _RecordingAggregator.instances[-1]
        assert "stamp_bar_at_interval_end" not in agg.kwargs
        gw._start_bar_flush_thread()  # connect() 内调用；测试不连柜台，直接启动
        deadline = time.time() + 3
        while agg.flush_calls == 0 and time.time() < deadline:
            time.sleep(0.05)
        assert agg.flush_calls > 0
    finally:
        gw._stop_bar_flush_thread()
    # 重复停止安全，停止后线程不再调用
    gw._stop_bar_flush_thread()
    calls = agg.flush_calls
    time.sleep(1.2)
    assert agg.flush_calls == calls
