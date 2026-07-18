"""out-of-core 流式回测压测验收脚本 (C.2d 配套).

目的:实测 ``DataFeed.from_parquet`` 流式回测的**峰值内存与数据总量无关**
(有界内存 = out-of-core),对比全量内存路径随行数线性增长的内存占用。

用法(在 workspace 根,`S` 记为 akquant/scripts/stress_out_of_core.py):

    uv run --package akquant python S
    uv run --package akquant python S --rows 5000000 --symbols 500 --chunk 65536
    uv run --package akquant python S --rows 2000000 --compare   # 额外跑内存路径对比

输出:
1. 生成的规范 parquet 行数 / 文件大小。
2. 流式回测:墙钟耗时 + 峰值内存(私有提交)。
3. 可选 --compare:全量内存回测的峰值内存,直观对比。

内存测量:Windows 用 psapi ``GetProcessMemoryInfo``(峰值私有提交);
Unix 用 ``resource.getrusage``。均为进程级(含 Rust 侧),故能反映真实占用。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import akquant
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from akquant import DataFeed, run_backtest


def _peak_rss_bytes() -> int:
    """返回当前进程峰值内存占用(字节),跨平台."""
    if sys.platform.startswith("win"):
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE  # 64 位下须设, 否则句柄被截断
        get_proc_mem = ctypes.windll.psapi.GetProcessMemoryInfo
        get_proc_mem.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PMC),
            wintypes.DWORD,
        ]
        get_proc_mem.restype = wintypes.BOOL

        counters = _PMC()
        counters.cb = ctypes.sizeof(counters)
        if not get_proc_mem(get_current_process(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo 调用失败")
        # 峰值私有提交(private commit)比工作集更能反映真实内存占用,
        # 且不受工作集裁剪/零页懒提交影响。
        return int(counters.PeakPagefileUsage)

    import resource

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: KB; macOS: bytes
    return int(maxrss * 1024) if sys.platform.startswith("linux") else int(maxrss)


def _mb(n: int) -> float:
    return n / (1024 * 1024)


def generate_canonical_parquet(
    path: Path, rows: int, symbols: int, batch: int = 500_000
) -> None:
    """分批生成规范 parquet(生成阶段本身也保持有界内存).

    列:timestamp(i64 ns UTC,全局升序)+ OHLCV(f64)+ symbol(str)。
    """
    writer: pq.ParquetWriter | None = None
    schema = pa.schema(
        [
            ("timestamp", pa.int64()),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
            ("symbol", pa.string()),
        ]
    )
    sym_names = [f"S{i:04d}" for i in range(symbols)]
    base_ts = 1_600_000_000_000_000_000
    step = 60_000_000_000  # 1 分钟
    written = 0
    try:
        writer = pq.ParquetWriter(path, schema, compression="zstd")
        while written < rows:
            n = min(batch, rows - written)
            idx = np.arange(written, written + n, dtype=np.int64)
            # 全局按 timestamp 升序:同一 tick 轮转各标的
            ts = base_ts + (idx // symbols) * step
            price = 10.0 + (idx % 100).astype(np.float64) * 0.01
            sym_idx = (idx % symbols).astype(np.int64)
            syms = pa.array(np.asarray(sym_names, dtype=object)[sym_idx])
            table = pa.table(
                {
                    "timestamp": pa.array(ts),
                    "open": pa.array(price),
                    "high": pa.array(price + 0.05),
                    "low": pa.array(price - 0.05),
                    "close": pa.array(price + 0.01),
                    "volume": pa.array(np.full(n, 100.0)),
                    "symbol": syms,
                },
                schema=schema,
            )
            writer.write_table(table)
            written += n
    finally:
        if writer is not None:
            writer.close()


class _CountStrategy(akquant.Strategy):
    """极简策略:仅计数,专注考察数据路径内存."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def on_bar(self, bar: akquant.Bar) -> None:
        self.n += 1


def main() -> None:
    """解析参数, 生成数据并运行流式(及可选内存)回测压测."""
    parser = argparse.ArgumentParser(description="out-of-core 流式回测压测")
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--symbols", type=int, default=200)
    parser.add_argument("--chunk", type=int, default=65_536)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="额外跑全量内存回测(pandas)对比峰值内存",
    )
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="akq_ooc_"))
    path = tmp / "market.parquet"

    print(f"[1] 生成规范 parquet: rows={args.rows:,} symbols={args.symbols} ...")
    t0 = time.perf_counter()
    generate_canonical_parquet(path, args.rows, args.symbols)
    gen_dt = time.perf_counter() - t0
    file_mb = _mb(path.stat().st_size)
    print(f"    完成: {gen_dt:.1f}s, 文件 {file_mb:.1f} MiB")

    print(f"[2] 流式回测 (DataFeed.from_parquet, chunk={args.chunk:,}) ...")
    t0 = time.perf_counter()
    feed = DataFeed.from_parquet(str(path), "UNKNOWN", args.chunk)
    result = run_backtest(
        data=feed,
        strategy=_CountStrategy,
        symbols=[f"S{i:04d}" for i in range(args.symbols)],
        initial_cash=1_000_000.0,
        show_progress=False,
    )
    stream_dt = time.perf_counter() - t0
    stream_peak = _peak_rss_bytes()
    bars = getattr(result.strategy, "n", None)
    print(f"    完成: {stream_dt:.1f}s, 处理 bar={bars:,}")
    print(f"    峰值内存(私有提交): {_mb(stream_peak):.0f} MiB")
    print(
        f"    每百万 bar 峰值内存 ≈ "
        f"{_mb(stream_peak) / max(1, args.rows / 1_000_000):.0f} MiB/M"
        "  (基本不随行数增长即为 out-of-core)"
    )

    if args.compare:
        import pandas as pd

        print("[3] 全量内存回测 (pandas) 对比 ...")
        t0 = time.perf_counter()
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
        df = df.set_index("date")
        try:
            run_backtest(
                data=df,
                strategy=_CountStrategy,
                symbols=[f"S{i:04d}" for i in range(args.symbols)],
                initial_cash=1_000_000.0,
                show_progress=False,
            )
            mem_dt = time.perf_counter() - t0
            mem_peak = _peak_rss_bytes()
            print(f"    完成: {mem_dt:.1f}s")
            print(f"    峰值内存(私有提交): {_mb(mem_peak):.0f} MiB")
            print(
                f"    内存路径较流式额外峰值: "
                f"+{_mb(mem_peak - stream_peak):.0f} MiB(即全量载入的代价)"
            )
        except MemoryError:
            print("    内存路径 OOM(印证流式 out-of-core 的必要性)")

    # 清理
    try:
        os.remove(path)
        os.rmdir(tmp)
    except OSError:
        pass


if __name__ == "__main__":
    main()
