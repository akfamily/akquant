"""C.2 流式 Parquet 数据源 (out-of-core) 端到端测试.

验证 DataFeed.from_parquet 的流式回测与等价内存回测数值一致。
"""

from pathlib import Path

import akquant
import numpy as np
import pandas as pd
from akquant import DataFeed, run_backtest


class _BuyOnUp(akquant.Strategy):
    """收盘 > 开盘则买入 100 股."""

    def on_bar(self, bar: akquant.Bar) -> None:
        """逐 bar 策略."""
        if bar.close > bar.open:
            self.buy(bar.symbol, 100)


def _final_equity(result: object) -> float:
    curve = result.equity_curve  # type: ignore[attr-defined]
    values = curve.values if hasattr(curve, "values") else list(curve)
    return round(float(np.asarray(values).ravel()[-1]), 6)


def _canonical_frame(n: int) -> pd.DataFrame:
    ts = np.arange(n, dtype=np.int64) * 60_000_000_000 + 1_600_000_000_000_000_000
    base = 10.0 + (np.arange(n) % 5)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + (np.arange(n) % 2),  # 交替涨跌
            "volume": np.full(n, 1000.0),
            "symbol": ["X"] * n,
        }
    )


def test_parquet_stream_backtest_matches_inmemory(tmp_path: Path) -> None:
    """流式 parquet 回测与等价内存回测末值一致."""
    frame = _canonical_frame(30)
    path = tmp_path / "data.parquet"
    frame.to_parquet(path, index=False)

    # 流式 (out-of-core): 小 chunk 强制多次分块读取
    feed = DataFeed.from_parquet(str(path), "X", 7)
    r_stream = run_backtest(
        data=feed, strategy=_BuyOnUp, symbols="X", initial_cash=100000.0
    )

    # 等价内存回测: 同一数据, datetime 索引
    mem = frame.copy()
    mem["date"] = pd.to_datetime(mem["timestamp"], unit="ns", utc=True)
    mem = mem.set_index("date")[["open", "high", "low", "close", "volume"]]
    r_mem = run_backtest(
        data=mem, strategy=_BuyOnUp, symbols="X", initial_cash=100000.0
    )

    assert _final_equity(r_stream) == _final_equity(r_mem)


def test_parquet_stream_visits_all_bars(tmp_path: Path) -> None:
    """流式回测应访问全部 bar (chunk 边界不丢数据)."""
    n = 25
    frame = _canonical_frame(n)
    path = tmp_path / "data.parquet"
    frame.to_parquet(path, index=False)

    seen: list[int] = []

    class _Counter(akquant.Strategy):
        def on_bar(self, bar: akquant.Bar) -> None:
            seen.append(bar.timestamp)

    feed = DataFeed.from_parquet(str(path), "X", 4)  # 4 不整除 25, 测边界
    run_backtest(data=feed, strategy=_Counter, symbols="X", initial_cash=100000.0)

    assert len(seen) == n
    assert seen == sorted(seen)  # 时间戳升序


def test_write_canonical_parquet_multi_symbol_stream(tmp_path: Path) -> None:
    """write_canonical_parquet 从任意源产出规范 parquet, 多标的流式回测闭环."""
    from akquant import write_canonical_parquet

    rows = []
    for i in range(5):
        day = f"2024-01-0{i + 1}"
        rows.append(
            {
                "date": day,
                "open": 10.0 + i,
                "high": 11.0 + i,
                "low": 9.0 + i,
                "close": 10.5 + i,
                "volume": 100.0,
                "symbol": "A",
            }
        )
        rows.append(
            {
                "date": day,
                "open": 20.0 + i,
                "high": 21.0 + i,
                "low": 19.0 + i,
                "close": 20.5 + i,
                "volume": 200.0,
                "symbol": "B",
            }
        )
    path = tmp_path / "canon.parquet"
    write_canonical_parquet(pd.DataFrame(rows), path)

    # 规范列
    import pyarrow.parquet as pq

    assert pq.read_table(path).column_names == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
    ]

    seen: dict[str, int] = {}

    class _PerSymbol(akquant.Strategy):
        def on_bar(self, bar: akquant.Bar) -> None:
            seen[bar.symbol] = seen.get(bar.symbol, 0) + 1

    feed = DataFeed.from_parquet(str(path), "UNKNOWN", 3)
    run_backtest(
        data=feed, strategy=_PerSymbol, symbols=["A", "B"], initial_cash=100000.0
    )
    assert seen == {"A": 5, "B": 5}
