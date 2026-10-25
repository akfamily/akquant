"""Dynamic strategy loading demo for strategy_source and strategy_loader."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import akquant as aq
from akquant import Bar, Strategy


def make_bars(symbol: str, count: int) -> list[Bar]:
    """Build deterministic bars for demo."""
    # 北京时间交易日上午连续竞价内的 1 分钟线(时间戳打在区间终点)
    start = datetime(2024, 1, 2, 9, 31, tzinfo=timezone(timedelta(hours=8)))
    bars: list[Bar] = []
    for i in range(count):
        dt = start + timedelta(minutes=i)
        ts_ns = int(dt.timestamp() * 1_000_000_000)
        price = 100.0 + float(i)
        bars.append(
            Bar(
                timestamp=ts_ns,
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.5,
                volume=1000.0 + float(i),
                symbol=symbol,
            )
        )
    return bars


def run_python_plain_scenario() -> None:
    """Load strategy from plain python source file."""
    bars = make_bars(symbol="PLAIN_DEMO", count=3)
    with TemporaryDirectory() as tmp_dir:
        strategy_path = Path(tmp_dir) / "plain_strategy.py"
        strategy_path.write_text(
            "\n".join(
                [
                    "from akquant.strategy import Strategy",
                    "",
                    "class DemoPlainStrategy(Strategy):",
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
            strategy_loader_options={"strategy_attr": "DemoPlainStrategy"},
            symbols="PLAIN_DEMO",
            show_progress=False,
        )
    strategy = result.strategy
    calls = getattr(strategy, "calls", -1) if strategy is not None else -1
    print(f"plain_loader_calls={calls}")


def run_encrypted_external_scenario() -> None:
    """Load strategy via encrypted_external callback hook."""
    bars = make_bars(symbol="ENC_DEMO", count=2)

    class DecryptedStrategy(Strategy):
        def __init__(self) -> None:
            self.calls = 0

        def on_bar(self, bar: Bar) -> None:
            _ = bar
            self.calls += 1

    def decrypt_and_load(source: Any, options: dict[str, Any]) -> type[Strategy]:
        _ = source
        _ = options
        return DecryptedStrategy

    result = aq.run_backtest(
        data=bars,
        strategy=None,
        strategy_source=b"encrypted_payload",
        strategy_loader="encrypted_external",
        strategy_loader_options={"decrypt_and_load": decrypt_and_load},
        symbols="ENC_DEMO",
        show_progress=False,
    )
    strategy = result.strategy
    calls = getattr(strategy, "calls", -1) if strategy is not None else -1
    print(f"encrypted_loader_calls={calls}")


def main() -> None:
    """Run all dynamic loading scenarios."""
    run_python_plain_scenario()
    run_encrypted_external_scenario()
    print("done_strategy_source_loader_demo")


if __name__ == "__main__":
    main()
