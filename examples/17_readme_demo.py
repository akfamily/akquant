import akquant as aq
import akshare as ak
from akquant import Strategy

# 1. 准备数据
# 使用 akshare 获取 A 股历史数据 (需安装: pip install akshare)
df = ak.stock_zh_a_hist_sina(
    symbol="sh600000", start_date="20250212", end_date="20260212"
)


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
