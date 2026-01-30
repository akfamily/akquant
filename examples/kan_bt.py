from typing import Any

import backtrader as bt  # type: ignore
import numpy as np
import pandas as pd


# 1. 复制数据生成逻辑 (保持完全一致)
def generate_mock_data(
    start_date: str = "20230101", end_date: str = "20230630"
) -> pd.DataFrame:
    """Generate mock data for backtrader."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    # 随机生成价格走势
    np.random.seed(42)
    price = 10.0 + np.cumsum(np.random.randn(n) * 0.1)

    data = pd.DataFrame(
        {
            "open": price + np.random.randn(n) * 0.05,
            "high": price + 0.2,
            "low": price - 0.2,
            "close": price,
            "volume": np.random.randint(1000, 10000, n),
        },
        index=dates,
    )

    return data


# 2. 定义 Backtrader 策略
class SmaStrategy(bt.Strategy):
    """Simple Moving Average strategy for Backtrader."""

    def __init__(self) -> None:
        """Initialize the strategy."""
        pass

    def log(self, txt: str, dt: Any = None) -> None:
        """Log function for this strategy."""
        dt = dt or self.datas[0].datetime.date(0)
        # print(f'{dt.isoformat()}, {txt}')

    def next(self) -> None:
        """Define logic for each iteration."""
        # 获取当前 bar 数据
        # 注意: Backtrader 的 close[0] 是当前 bar

        # 对应 akquant: position = self.get_position(bar.symbol)
        position_size = self.position.size

        # 对应 akquant: if position == 0:
        if position_size == 0:
            # self.buy(symbol=bar.symbol, quantity=100)
            self.buy(size=100)
            # self.log(f'BUY CREATE, {self.datas[0].close[0]}')

        # 对应 akquant: elif position > 0:
        elif position_size > 0:
            # 对应 akquant: if bar.close > bar.open * 1.05:
            # Backtrader 中当前 bar 的 open/close 直接访问
            current_open = self.datas[0].open[0]
            current_close = self.datas[0].close[0]

            if current_close > current_open * 1.05:
                # self.sell(symbol=bar.symbol, quantity=100)
                self.sell(size=100)
                # self.log(f'SELL CREATE, {self.datas[0].close[0]}')


# 3. 自定义佣金模式 (模拟 A 股)
class CNCommission(bt.CommInfoBase):
    """Commission scheme for China stock market."""

    params = (
        ("stamp_tax", 0.0005),  # 印花税
        ("commission", 0.0003),  # 佣金
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size: float, price: float, pseudoexec: bool) -> float:
        if size > 0:  # 买入
            return float(size * price * self.p.commission)
        elif size < 0:  # 卖出
            # 佣金 + 印花税
            return float(-size * price * (self.p.commission + self.p.stamp_tax))
        return 0.0


def run_backtrader() -> None:
    """Run Backtrader backtest."""
    # 生成数据
    df = generate_mock_data()

    # 转换为 Backtrader 数据格式
    # 注意：Backtrader 默认 PandasData 需要 datetime 索引或列
    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaStrategy)
    cerebro.adddata(data)

    # 设置资金
    start_cash = 500_000.0
    cerebro.broker.setcash(start_cash)

    # 设置佣金
    comminfo = CNCommission(commission=0.0003, stamp_tax=0.0005)
    cerebro.broker.addcommissioninfo(comminfo)

    # 添加分析器
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        riskfreerate=0.0,
        timeframe=bt.TimeFrame.Days,
        compression=1,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()
    print(f"Final Portfolio Value: {end_value:.2f}")

    strat = results[0]

    # 计算 Total Return
    total_return = (end_value - start_cash) / start_cash
    print(f"Total Return: {total_return:.2%}")

    # 获取 Sharpe Ratio
    sharpe = strat.analyzers.sharpe.get_analysis()
    # Handle case where Sharpe might be None if not enough data or flat return
    sharpe_ratio = sharpe.get("sharperatio", 0.0)
    if sharpe_ratio is None:
        sharpe_ratio = 0.0
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # 获取 Max Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_dd = drawdown.get("max", {}).get("drawdown", 0.0)
    print(f"Max Drawdown: {max_dd:.2%}")


if __name__ == "__main__":
    run_backtrader()
