"""
股票 ATR 通道突破策略 (Stock ATR Breakout Strategy).

==============================================

本示例展示如何使用 AKShare 获取股票数据，并实现经典的 ATR 通道突破策略。

教学目标 (Learning Objectives):
1.  使用 `ak.stock_zh_a_daily` 获取股票数据。
2.  使用 `self.get_history` 高效计算 ATR (平均真实波幅)。
3.  实现基于波动率的通道突破策略。

前置条件 (Prerequisites):
- 安装 akshare: `pip install akshare`
"""

import akquant as aq
import akshare as ak
from akquant import Bar, FloatParam, IntParam, Strategy


class AtrBreakoutStrategy(Strategy):
    """
    ATR 通道突破策略 (ATR Channel Breakout).

    逻辑:
    - 计算 ATR (平均真实波幅)。
    - 计算上轨: Close + k * ATR
    - 计算下轨: Close - k * ATR
    - 价格突破上轨 -> 买入 (Buy)
    - 价格跌破下轨 -> 卖出 (Sell)
    """

    period = IntParam(20, ge=2, le=500, title="ATR 周期")
    k = FloatParam(2.0, ge=0.1, le=10.0, title="突破倍数")

    def on_start(self) -> None:
        """设置数据预热长度 (必须设置才能使用 get_history)."""
        self.warmup_period = self.params.period + 1

    def on_bar(self, bar: Bar) -> None:
        """处理每个 Bar 数据."""
        symbol = bar.symbol

        # 1. 获取历史数据
        # ATR 需要 High, Low, Close
        # 注意：get_history 返回的数据可能包含当前 bar，为了避免未来函数
        # 我们需要获取 N+1 个数据并剔除最后一个
        req_count = self.params.period + 1
        h_highs = self.get_history(count=req_count, symbol=symbol, field="high")
        h_lows = self.get_history(count=req_count, symbol=symbol, field="low")
        h_closes = self.get_history(count=req_count, symbol=symbol, field="close")

        # 检查数据是否足够
        if len(h_closes) < req_count:
            return

        highs = h_highs[:-1]
        lows = h_lows[:-1]
        closes = h_closes[:-1]

        # 2. 计算 ATR
        # 注意: 这里简化计算，使用过去 period 天的 TR 的平均值
        # 真实的 ATR 通常使用 EMA 平滑

        tr_sum = 0.0
        for i in range(1, len(closes)):
            h = highs[i]
            low_val = lows[i]
            pc = closes[i - 1]
            tr = max(h - low_val, abs(h - pc), abs(low_val - pc))
            tr_sum += tr

        # 加上第 0 个元素 (假设第 0 个元素的 TR 为 H-L，因为没有前一天的数据)
        tr_sum += highs[0] - lows[0]

        atr = tr_sum / len(closes)

        # 3. 计算轨道
        # 使用昨天的收盘价作为基准
        prev_close = closes[-1]
        upper_band = prev_close + self.params.k * atr
        lower_band = prev_close - self.params.k * atr

        # 4. 交易逻辑
        current_pos = self.get_position(symbol)

        # 突破上轨 -> 买入
        if bar.close > upper_band and current_pos == 0:
            # 买入 500 股
            self.buy(symbol=symbol, quantity=500)
            print(
                f"[{bar.timestamp_iso}] LONG BREAKOUT: Price {bar.close:.2f} > "
                f"Upper {upper_band:.2f} (ATR={atr:.2f})"
            )

        # 跌破下轨 -> 卖出平仓
        elif bar.close < lower_band and current_pos > 0:
            self.close_position(symbol=symbol)
            print(
                f"[{bar.timestamp_iso}] CLOSE LONG: Price {bar.close:.2f} < "
                f"Lower {lower_band:.2f} (ATR={atr:.2f})"
            )


if __name__ == "__main__":
    # 1. 准备数据: 贵州茅台 (sh600519)
    # 这种高价股适合演示资金管理，但这里简单演示策略逻辑
    symbol = "sh600519"

    df = ak.stock_zh_a_daily(
        symbol=symbol, start_date="20220101", end_date="20231231", adjust="qfq"
    )

    # 2. 运行回测
    result = aq.run_backtest(
        data=df,
        strategy=AtrBreakoutStrategy,
        symbols=symbol,
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        lot_size=100,
    )

    # 3. 结果分析
    print("\n=== Backtest Result ===")
    print(result)
