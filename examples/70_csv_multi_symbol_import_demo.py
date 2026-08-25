"""从 CSV 文件导入多品种数据的回测示例（平台对接格式）.

本示例演示的是**平台方实际在用**的多品种文件导入格式（见
`vendor/docs/多品种回测数据格式说明.md`），而不是随意的 CSV 结构：

1. 固定 7 列：`date` / `symbol` / `open` / `high` / `low` / `close` / `volume`；
2. `symbol` 是**去后缀的纯数字**代码（如 `600487`，不是 `600487.SH`）；
3. `date` 是**naive datetime**（按东八区解析后再去掉时区信息）；
4. 多标的数据先各自成表，再 `pd.concat` 成一张大表，按 `date` + `symbol` 排序；
5. 调用 `run_backtest(data=df, symbols=[...])` 时**显式传 symbols**。

为了让示例能独立跑通（不依赖外部数据源、不依赖本机某个路径下的文件），脚本会先在
临时目录生成一份符合上述格式的 CSV，再把它读回来跑回测，跑完清理临时文件。这样
同时演示了「文件应该长什么样」和「怎么从文件读」。

容易踩的坑（详见下方注释里对应位置）：

* `symbol` 去后缀是**平台口径**，不是 AKQuant 的强制要求——带后缀（如 `600487.SH`）
  同样能跑，但 CSV 里的 `symbol` 列必须和 `run_backtest(symbols=...)` 传入的列表
  完全一致（一个去了后缀、另一个没去，会导致标的对不上、静默收不到数据）。
* `date` 必须是 naive（无时区）。如果 CSV 里的时间戳字符串本身就是东八区本地时间
  （没有时区标记），直接 `pd.to_datetime` 即可保持 naive；如果源头是带时区的
  datetime 或 UTC 毫秒时间戳，要先按 `tz="Asia/Shanghai"` 本地化再 `tz_localize(None)`
  去掉时区，不能直接丢弃时区信息（否则时间会整体偏移 8 小时）。
* 多标的必须 `pd.concat` 成**一张表**再传给 `data=`；不要传 `{symbol: df}` 形式的
  dict（`01_quickstart.py` 里那种写法也能跑，但它是另一条输入路径，和本示例演示的
  「文件导入」平台格式是两回事，混用容易记错字段要求）。
* 额外一个实测踩到的坑：`symbol` 列若全是纯数字字符串（如 `002202`），
  `pd.read_csv` 默认会把它推断成 `int64`，前导 0 被吃掉变成 `2202`，导致该标的和
  `symbols=` 参数对不上、被判定为「数据里没有」而静默收不到行情。必须在
  `pd.read_csv(..., dtype={"symbol": str})` 里显式指定 dtype。
"""

import tempfile
from pathlib import Path
from typing import Any, cast

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy

# 平台格式约定：symbol 去后缀，只留纯数字代码。
SYMBOLS = ["600487", "002202"]


def _make_symbol_frame(
    symbol: str, start: str, periods: int, base_price: float
) -> pd.DataFrame:
    """构造单个标的的日线 DataFrame，列结构与平台导出格式完全一致（7 列）."""
    # naive datetime：先按东八区解析，再去掉时区信息（坑 2）。
    dates = pd.date_range(
        start=start, periods=periods, freq="B", tz="Asia/Shanghai"
    ).tz_localize(None)
    closes = [round(base_price + 0.1 * i, 2) for i in range(periods)]
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,  # 纯数字，无 .SH/.SZ 后缀（坑 1）。
            "open": [round(c - 0.05, 2) for c in closes],
            "high": [round(c + 0.2, 2) for c in closes],
            "low": [round(c - 0.2, 2) for c in closes],
            "close": closes,
            "volume": [1_000_000.0 + 1_000.0 * i for i in range(periods)],
        }
    )


def write_demo_csv(csv_path: Path) -> None:
    """在给定路径生成一份符合平台格式的多品种 CSV（自带示例数据）."""
    df_1 = _make_symbol_frame("600487", "2024-01-02", periods=40, base_price=12.0)
    df_2 = _make_symbol_frame("002202", "2024-01-02", periods=40, base_price=8.0)
    # 多标的合并规则：先各自成表，再 concat 成一张大表，按 date+symbol 排序（坑 3）。
    merged = pd.concat([df_1, df_2], ignore_index=True)
    merged = merged.sort_values(["date", "symbol"]).reset_index(drop=True)
    merged.to_csv(csv_path, index=False)


def read_demo_csv(csv_path: Path) -> pd.DataFrame:
    """从 CSV 读回 DataFrame，并显式校验/规整 date 列与排序，贴近真实导入链路.

    第四个坑（比前三个更容易被忽略）：`symbol` 列内容是纯数字字符串（如
    `002202`），如果不显式指定 `dtype={"symbol": str}`，`pd.read_csv` 会把它当数值
    列推断成 `int64`，前导 0 直接被吃掉变成 `2202`——和 `run_backtest(symbols=...)`
    里的 `"002202"` 对不上，导致该标的被判定为「数据里没有」而全程收不到行情，
    且只会在日志里留一条 WARNING，不会报错。必须显式指定 dtype 才能保留前导 0。
    """
    df = pd.read_csv(csv_path, dtype={"symbol": str})
    # CSV 里写的是不带时区标记的本地时间字符串，to_datetime 直接解析出来就是 naive。
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return cast(pd.DataFrame, df)


class SymbolCoverageStrategy(Strategy):
    """最小策略：只按标的分别计数，收尾打印，用来证明两个标的都收到了数据."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """初始化每标的计数器."""
        super().__init__()
        self.bar_count_by_symbol: dict[str, int] = {}

    def on_bar(self, bar: Bar) -> None:
        """收到一根 bar 就给对应标的计数加一；第一次见到某标的顺手买入一手."""
        symbol = bar.symbol
        self.bar_count_by_symbol[symbol] = self.bar_count_by_symbol.get(symbol, 0) + 1
        if self.get_position(symbol) == 0:
            self.order_target_percent(target_percent=0.3, symbol=symbol)

    def on_stop(self) -> None:
        """回测结束时打印每个标的收到的 bar 数，验证多标的确实都被驱动到了."""
        for symbol in SYMBOLS:
            count = self.bar_count_by_symbol.get(symbol, 0)
            print(f"symbol={symbol} bars_received={count}")


def main() -> None:
    """生成临时 CSV -> 读回 DataFrame -> 显式传 symbols 跑回测 -> 清理临时文件."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "multi_symbol_demo.csv"
        write_demo_csv(csv_path)

        print(f"csv_written={csv_path}")
        # 直接打印文件原始文本（而不是 pd.read_csv 之后的结果）：
        # 这样才能看到 symbol 列在磁盘上确实是带前导 0 的字符串 "002202"，
        # 不会被上面提到的 dtype 推断坑误导。
        with csv_path.open("r", encoding="utf-8") as fh:
            preview_lines = [next(fh) for _ in range(5)]
        print("".join(preview_lines).rstrip())

        data = read_demo_csv(csv_path)

        result = aq.run_backtest(
            data=data,
            strategy=SymbolCoverageStrategy,
            # 显式传 symbols：这是「只跑哪些标的」的白名单语义——不传的话回测无法确定
            # 应该激活哪些标的的撮合与风控；传入的代码必须和 CSV 里 symbol 列的口径
            # （这里是去后缀的纯数字）完全一致。
            symbols=SYMBOLS,
            initial_cash=1_000_000.0,
            commission_rate=0.0003,
            stamp_tax_rate=0.001,
            transfer_fee_rate=0.0,
            min_commission=5.0,
            timezone="Asia/Shanghai",
            show_progress=False,
        )

        print("metrics")
        print(
            result.metrics_df.loc[
                ["total_return_pct", "sharpe_ratio", "max_drawdown_pct"]
            ]
        )


if __name__ == "__main__":
    main()
