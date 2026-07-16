"""运行时多周期 Bar 聚合器.

把 on_bar 的 bar 流流式聚合成更高周期(minute/hour/day 的 N 倍)并回调;
回测/实盘同一句 update_bar; 同参数与离线 feed.resample 产出一致
(时钟对齐, label=right, closed=right)。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from .akquant import Bar

_INTERVAL_FREQ = {"minute": "min", "hour": "h", "day": "D"}


class BarGenerator:
    """策略内流式多周期聚合器(vnpy 式独立 helper)."""

    def __init__(
        self,
        on_window_bar: Callable[[Bar], None],
        window: int = 1,
        interval: str = "minute",
        *,
        session_windows: Optional[list[tuple[str, str]]] = None,
        timezone: Optional[str] = None,
    ) -> None:
        """绑定回调与目标周期.

        :param on_window_bar: 窗口闭合时回调, 收到聚合后的 Bar。
        :param window: 周期倍数(>=1)。
        :param interval: "minute" | "hour" | "day"。
        :param session_windows: 交易时段(本地时间), 给定则不跨段聚合(Task 2)。
        :param timezone: 时钟/session 对齐所在市场时区, 如 "Asia/Shanghai"。
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        if interval not in _INTERVAL_FREQ:
            raise ValueError("interval must be one of: minute, hour, day")
        self._cb = on_window_bar
        self._freq = f"{window}{_INTERVAL_FREQ[interval]}"
        self._tz = timezone
        self._session_windows = session_windows
        self._parsed_sessions = [
            (
                pd.Timestamp(f"2000-01-01 {s}").time(),
                pd.Timestamp(f"2000-01-01 {e}").time(),
            )
            for (s, e) in (session_windows or [])
        ]
        # symbol -> {"label_ns","skey","open","high","low","close","volume"}
        self._states: dict[str, dict[str, Any]] = {}

    def _bucket_label_ns(self, ts_ns: int) -> int:
        """算 bar 所属窗口的右缘标签(Unix 纳秒), 对齐 pandas ceil(freq)."""
        ts = pd.Timestamp(ts_ns, unit="ns")
        if self._tz is not None:
            ts = ts.tz_localize("UTC").tz_convert(self._tz)
        label = ts.ceil(self._freq)
        if self._tz is not None:
            label = label.tz_convert("UTC")
        return int(label.value)

    def _session_key(self, ts_ns: int) -> Any:
        """返回 bar 所属 (本地日, session 序号); 无 session_windows 返回 None."""
        if not self._session_windows:
            return None
        ts = pd.Timestamp(ts_ns, unit="ns")
        if self._tz is not None:
            ts = ts.tz_localize("UTC").tz_convert(self._tz)
        t = ts.time()
        for idx, (start, end) in enumerate(self._parsed_sessions):
            if start <= t <= end:
                return (ts.date(), idx)
        return (ts.date(), -1)  # 段外(集合竞价等), 单独归一段

    def update_bar(self, bar: Bar) -> None:
        """喂入一根 bar; 跨桶或跨 session 时闭合并回调上一窗口。假设时间戳非递减."""
        symbol = bar.symbol
        ts_ns = int(bar.timestamp)
        label_ns = self._bucket_label_ns(ts_ns)
        skey = self._session_key(ts_ns)
        st = self._states.get(symbol)
        if st is not None and (label_ns != st["label_ns"] or skey != st["skey"]):
            self._emit(symbol, st)
            st = None
        if st is None:
            self._states[symbol] = {
                "label_ns": label_ns,
                "skey": skey,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        else:
            if bar.high > st["high"]:
                st["high"] = bar.high
            if bar.low < st["low"]:
                st["low"] = bar.low
            st["close"] = bar.close
            st["volume"] += bar.volume

    def flush(self) -> None:
        """强制闭合所有 symbol 的当前未满窗口(回测收尾 / 数据结束)."""
        for symbol, st in list(self._states.items()):
            self._emit(symbol, st)
        self._states.clear()

    def current(self, symbol: str) -> Optional[Bar]:
        """返回 symbol 正在形成(未闭合)的窗口 bar 快照; 无则 None。不触发回调."""
        st = self._states.get(symbol)
        if st is None:
            return None
        return self._make_bar(symbol, st)

    def _emit(self, symbol: str, st: dict[str, Any]) -> None:
        self._cb(self._make_bar(symbol, st))

    @staticmethod
    def _make_bar(symbol: str, st: dict[str, Any]) -> Bar:
        return Bar(
            int(st["label_ns"]),
            float(st["open"]),
            float(st["high"]),
            float(st["low"]),
            float(st["close"]),
            float(st["volume"]),
            symbol,
        )
