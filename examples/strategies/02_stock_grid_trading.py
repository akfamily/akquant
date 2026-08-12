"""
股票网格交易策略示例 (Stock Grid Trading Strategy).

===========================================

本示例展示如何使用 AKShare 获取股票历史数据，并运行一个经典的网格交易策略。

教学目标 (Learning Objectives):
1.  使用 `ak.stock_zh_a_daily` 获取股票数据。
2.  理解网格交易的基本逻辑：分批建仓、高抛低吸。
3.  演示如何在 `on_bar` 中管理复杂的持仓状态。

前置条件 (Prerequisites):
- 安装 akshare: `pip install akshare`
"""

import akquant as aq
import akshare as ak
from akquant import Bar, FloatParam, IntParam, Strategy


class GridTradingStrategy(Strategy):
    """
    网格交易策略 (Grid Trading).

    逻辑:
    - 设定一个基准价格 (base_price) 和网格间距 (grid_pct)。
    - 价格每下跌 grid_pct，买入一份 (Buy Dip)。
    - 价格每上涨 grid_pct，卖出一份 (Sell Rally)。
    - 这里的实现是一个简化的动态网格：
        - 初始建仓 50%。
        - 记录上次交易价格。
        - 相比上次买入价下跌 X%，加仓。
        - 相比上次卖出价上涨 X%，减仓。
    """

    grid_pct = FloatParam(0.03, ge=0.0, le=1.0, title="网格间距")
    trade_lot = IntParam(100, ge=1, title="每次交易数量(股/份)")

    def on_start(self) -> None:
        """初始化每个标的的上次交易价格记录."""
        self.last_trade_price: dict[str, float] = {}

    def on_bar(self, bar: Bar) -> None:
        """Process each Bar data."""
        symbol = bar.symbol
        close = bar.close

        # 1. 初始建仓 (如果从未交易过)
        if symbol not in self.last_trade_price:
            # 假设初始建仓 10 手
            initial_lots = 10
            self.buy(symbol=symbol, quantity=initial_lots * self.params.trade_lot)
            self.last_trade_price[symbol] = close
            print(
                f"[{bar.timestamp_iso}] Initial Position: "
                f"Buy {initial_lots * self.params.trade_lot} at {close:.3f}"
            )
            return

        last_price = self.last_trade_price[symbol]

        # 2. 计算价格变化幅度
        change_pct = (close - last_price) / last_price

        # 3. 网格逻辑

        # 情况 A: 价格下跌超过 grid_pct -> 加仓 (买入)
        if change_pct <= -self.params.grid_pct:
            self.buy(symbol=symbol, quantity=self.params.trade_lot)
            self.last_trade_price[symbol] = close
            print(
                f"[{bar.timestamp_iso}] Grid BUY: Price dropped {change_pct:.2%}, "
                f"Buy {self.params.trade_lot} at {close:.3f}"
            )

        # 情况 B: 价格上涨超过 grid_pct -> 减仓 (卖出)
        elif change_pct >= self.params.grid_pct:
            # 检查持仓是否足够
            current_pos = self.get_position(symbol)
            if current_pos >= self.params.trade_lot:
                self.sell(symbol=symbol, quantity=self.params.trade_lot)
                self.last_trade_price[symbol] = close
                print(
                    f"[{bar.timestamp_iso}] Grid SELL: Price rose {change_pct:.2%}, "
                    f"Sell {self.params.trade_lot} at {close:.3f}"
                )
            else:
                print(
                    f"[{bar.timestamp_iso}] Grid Signal: Sell triggered but "
                    f"insufficient position ({current_pos})"
                )


if __name__ == "__main__":
    # 1. 准备数据: 使用波动较大的股票，例如 宁德时代 (sz300750)
    symbol = "sz300750"

    df = ak.stock_zh_a_daily(
        symbol=symbol, start_date="20220101", end_date="20231231", adjust="qfq"
    )

    # 2. 运行回测
    result = aq.run_backtest(
        data=df,
        strategy=GridTradingStrategy,
        symbols=symbol,
        initial_cash=500_000.0,
        lot_size=100,  # 股票每手100股
        commission_rate=0.0003,  # 万三佣金
        stamp_tax_rate=0.001,  # 千一印花税
    )

    # 3. 输出结果
    print("\n=== Backtest Result ===")
    print(result)
