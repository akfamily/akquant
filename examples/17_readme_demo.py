import akquant as aq
import numpy as np
import pandas as pd
from akquant import Strategy


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


# 1. 准备数据
# 使用 akshare 获取 A 股历史数据 (需安装: pip install akshare)
df = fetch_daily(symbol="sh600000", start_date="20250212", end_date="20260212")


class MyStrategy(Strategy):
    """简单的阳线买入、阴线卖出示例策略.

    本策略在每个 Bar 到来时进行判断：
    - 当收盘价大于开盘价（阳线）且当前无持仓，则买入固定数量；
    - 当收盘价小于开盘价（阴线）且当前有持仓，则平仓。
    """

    def on_bar(self, bar: aq.Bar) -> None:
        """处理每个到来的 Bar，并根据 K 线形态进行交易决策。.

        :param bar: 当前 Bar 数据，包含 open/close/high/low/volume/timestamp 等字段。
        :type bar: akquant.Bar
        :return: 无返回值
        :rtype: None
        """
        # 简单策略示例:
        # 当收盘价 > 开盘价 (阳线) -> 买入
        # 当收盘价 < 开盘价 (阴线) -> 卖出

        # 获取当前持仓
        current_pos = self.get_position(bar.symbol)

        if current_pos == 0 and bar.close > bar.open:
            self.buy(symbol=bar.symbol, quantity=100)
            print(f"[{bar.timestamp_iso}] Buy 100 at {bar.close:.2f}")

        elif current_pos > 0 and bar.close < bar.open:
            self.close_position(symbol=bar.symbol)
            print(f"[{bar.timestamp_iso}] Sell 100 at {bar.close:.2f}")


# 运行回测
events: list[aq.BacktestStreamEvent] = []
result = aq.run_backtest(
    data=df,
    strategy=MyStrategy,
    initial_cash=100000.0,
    symbols="sh600000",
    on_event=events.append,
)

# 打印回测结果
print("\n=== Backtest Result ===")
print(result)
print(f"stream_events={len(events)}")
