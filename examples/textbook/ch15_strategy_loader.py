"""
第 15 章：动态策略加载（Strategy Loader）.

本章演示如何在不直接导入策略类的情况下，通过 strategy_source + strategy_loader
在运行时加载策略实现。
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import akquant as aq
from akquant import Bar, Strategy


def make_bars(symbol: str, count: int) -> list[Bar]:
    """构造示例 K 线."""
    # 北京时间交易日上午连续竞价内的 1 分钟线（时间戳打在区间终点）
    start = datetime(2024, 1, 2, 9, 31, tzinfo=timezone(timedelta(hours=8)))
    bars: list[Bar] = []
    for i in range(count):
        dt = start + timedelta(minutes=i)
        ts_ns = int(dt.timestamp() * 1_000_000_000)
        price = 50.0 + float(i)
        bars.append(
            Bar(
                timestamp=ts_ns,
                open=price,
                high=price + 0.8,
                low=price - 0.8,
                close=price + 0.3,
                volume=2000.0 + float(i),
                symbol=symbol,
            )
        )
    return bars


def chapter_plain_loader() -> None:
    """场景一：python_plain 从源码文件加载策略类."""
    bars = make_bars("CH15_PLAIN", 3)
    with TemporaryDirectory() as tmp_dir:
        strategy_path = Path(tmp_dir) / "chapter_strategy.py"
        strategy_path.write_text(
            "\n".join(
                [
                    "from akquant.strategy import Strategy",
                    "",
                    "class ChapterPlainStrategy(Strategy):",
                    "    def __init__(self):",
                    "        self.calls = 0",
                    "",
                    "    def on_bar(self, bar):",
                    "        self.calls += 1",
                ]
            ),
            encoding="utf-8",
        )
        result = aq.run_backtest(
            data=bars,
            strategy=None,
            strategy_source=str(strategy_path),
            strategy_loader="python_plain",
            strategy_loader_options={"strategy_attr": "ChapterPlainStrategy"},
            symbols="CH15_PLAIN",
            show_progress=False,
        )
    strategy = result.strategy
    calls = getattr(strategy, "calls", -1) if strategy is not None else -1
    print(f"chapter15_plain_calls={calls}")


def chapter_encrypted_loader() -> None:
    """场景二：encrypted_external 通过外部回调加载策略类."""
    bars = make_bars("CH15_ENC", 2)

    class ChapterEncryptedStrategy(Strategy):
        def __init__(self) -> None:
            self.calls = 0

        def on_bar(self, bar: Bar) -> None:
            _ = bar
            self.calls += 1

    def decrypt_and_load(source: Any, options: dict[str, Any]) -> type[Strategy]:
        _ = source
        _ = options
        return ChapterEncryptedStrategy

    result = aq.run_backtest(
        data=bars,
        strategy=None,
        strategy_source=b"chapter15_encrypted_payload",
        strategy_loader="encrypted_external",
        strategy_loader_options={"decrypt_and_load": decrypt_and_load},
        symbols="CH15_ENC",
        show_progress=False,
    )
    strategy = result.strategy
    calls = getattr(strategy, "calls", -1) if strategy is not None else -1
    print(f"chapter15_encrypted_calls={calls}")


def main() -> None:
    """运行第 15 章示例."""
    chapter_plain_loader()
    chapter_encrypted_loader()
    print("done_ch15_strategy_loader")


if __name__ == "__main__":
    main()
