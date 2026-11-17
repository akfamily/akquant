"""
多股票轮动策略示例 (Multi-Stock Rotation Strategy).

==============================================

本示例展示如何同时回测多个标的，并实现基于动量的轮动策略。

教学目标 (Learning Objectives):
1.  同时获取并清洗多只股票的数据 (如 贵州茅台 vs 五粮液)。
2.  使用 `self.get_history` 高效获取多标的历史数据。
3.  实现跨标的比较与仓位轮动逻辑。

前置条件 (Prerequisites):
- 安装 akshare: `pip install akshare`
"""

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, IntParam, Strategy


def synthetic_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """生成固定种子的合成日线（几何随机游走），只用于取数失败时让示例跑通."""
    dates = pd.bdate_range(start_date, end_date)
    rng = np.random.default_rng(sum(map(ord, symbol)))
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, len(dates))))
    open_ = np.concatenate([[10.0], close[:-1]])
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
        }
    )


def fetch_daily(
    symbol: str, start_date: str, end_date: str, adjust: str = ""
) -> pd.DataFrame:
    """用 AKShare 取新浪 A 股日线（代码带市场前缀，如 sh600000）.

    取数失败（未安装 AKShare、断网、接口改名等）时改用合成数据，保证示例能跑通；
    合成数据的回测结果没有参考意义。
    """
    try:
        import akshare as ak

        df = ak.stock_zh_a_hist_sina(
            symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust
        )
        if df.empty:
            raise ValueError("返回空表")
        return pd.DataFrame(df)
    except Exception as exc:  # noqa: BLE001 - 示例容错，任何取数问题都回退
        print(f"警告: AKShare 取数失败（{type(exc).__name__}: {exc}），改用合成数据")
        return synthetic_daily(symbol, start_date, end_date)


class MomentumRotationStrategy(Strategy):
    """
    动量轮动策略 (Momentum Rotation).

    逻辑:
    - 每日收盘时计算所有标的过去 N 天的收益率 (Momentum)。
    - 持有动量最强的标的。
    - 如果当前持仓不是最强标的，则换仓。
    """

    lookback_period = IntParam(20, ge=1, le=500, title="动量回看周期")

    def on_start(self) -> None:
        """设置轮动标的列表与数据预热长度."""
        # 定义需要轮动的标的列表
        self.symbols = ["sh600519", "sz000858"]  # 茅台 vs 五粮液
        # 设置数据预热长度
        self.warmup_period = self.params.lookback_period + 1

    def on_bar(self, bar: Bar) -> None:
        """处理每个 Bar 数据."""
        # 注意: 在多标的回测中，on_bar 会被每个标的分别触发。
        # 对于轮动策略，我们通常只需要在"每天"处理一次。
        # 简单起见，我们在处理到列表最后一个标的时执行轮动逻辑。
        # 或者，更严谨的做法是使用 on_timer 定时触发。

        if bar.symbol != self.symbols[-1]:
            return

        # 1. 计算所有标的的动量
        momentums: dict[str, float] = {}
        for s in self.symbols:
            # 获取历史收盘价 (包含当前bar)
            closes = self.get_history(
                count=self.params.lookback_period, symbol=s, field="close"
            )

            # 检查数据是否足够
            if len(closes) < self.params.lookback_period:
                return

            # 计算动量: (当前收盘价 - N天前收盘价) / N天前收盘价
            p_now = closes[-1]
            p_prev = closes[0]
            mom = (p_now - p_prev) / p_prev
            momentums[s] = mom

        # 2. 选出动量最大的标的
        best_symbol = max(momentums, key=lambda k: momentums[k])
        best_mom = momentums[best_symbol]

        # 3. 交易执行
        # 如果动量为负，是否空仓？这里假设只要有正动量就持有，全负则空仓 (可选)
        if best_mom < 0:
            # 清空所有持仓
            for s in self.symbols:
                if self.get_position(s) > 0:
                    self.close_position(s)
            return

        # 持有 best_symbol
        current_pos_symbol = None
        for s in self.symbols:
            if self.get_position(s) > 0:
                current_pos_symbol = s
                break

        # 如果当前没有持仓，买入 best
        if current_pos_symbol is None:
            # 目标仓位 95%
            # 参数顺序: order_target_percent(symbol, target_percent)，与 buy/sell 一致
            self.order_target_percent(target_percent=0.95, symbol=best_symbol)
            print(f"[{bar.timestamp_iso}] BUY {best_symbol}: Momentum={best_mom:.2%}")

        # 如果持有其他标的，换仓
        elif current_pos_symbol != best_symbol:
            print(
                f"[{bar.timestamp_iso}] ROTATE: "
                f"Sell {current_pos_symbol} -> Buy {best_symbol}"
            )
            self.close_position(current_pos_symbol)
            self.order_target_percent(target_percent=0.95, symbol=best_symbol)


if __name__ == "__main__":
    # 1. 准备数据: 白酒双雄
    symbols = ["sh600519", "sz000858"]  # 贵州茅台, 五粮液
    data_map = {}

    print("Fetching data...")
    for s in symbols:
        df = fetch_daily(
            symbol=s, start_date="20220101", end_date="20231231", adjust="qfq"
        )
        if not df.empty:
            # 简单清洗: 筛选列 + 标准化
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = s
            df = df.sort_values("date").reset_index(drop=True)
            data_map[s] = df
        else:
            print(f"Skipping {s} due to no data")

    if not data_map:
        print("No data available.")
        exit(0)

    # 2. 运行回测
    print(f"\nRunning Rotation Strategy on {list(data_map.keys())}...")

    # 注意: 当传入多个标的数据时，data 参数接受一个字典 {symbol: DataFrame}
    result = aq.run_backtest(
        data=data_map,
        strategy=MomentumRotationStrategy,
        symbols=list(data_map.keys()),
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
    )

    # 3. 结果
    print("\n=== Backtest Result ===")
    print(result)
