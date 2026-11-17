"""
股票网格交易策略示例 (Stock Grid Trading Strategy).

===========================================

本示例展示如何使用 AKShare 获取股票历史数据，并运行一个经典的网格交易策略。

教学目标 (Learning Objectives):
1.  使用 `ak.stock_zh_a_hist_sina` 获取股票数据。
2.  理解网格交易的基本逻辑：分批建仓、高抛低吸。
3.  演示如何在 `on_bar` 中管理复杂的持仓状态。

前置条件 (Prerequisites):
- 安装 akshare: `pip install akshare`
"""

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, FloatParam, IntParam, Strategy


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

    grid_pct = FloatParam(0.03, ge=0.001, le=0.5, title="网格间距")
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

    df = fetch_daily(
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
