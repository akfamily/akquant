"""
第 9 章：基金与可转债策略 (Funds & Convertible Bonds)（函数式写法）.

本示例展示了 **ETF 网格交易 (Grid Trading)** 策略。
ETF (Exchange Traded Fund) 是场内交易基金，交易规则与股票类似，但免收印花税，
非常适合高频网格策略。

策略逻辑：
1. **中枢定价**：以过去 20 日均线为中枢。
2. **分档挂单**：
    - 每下跌 1%，买入一份。
    - 每上涨 1%，卖出一份。
3. **收益来源**：在震荡市中，通过反复低吸高抛赚取差价。

适用标的：
- 宽基 ETF (如 510300 沪深300 ETF)
- 行业 ETF (如 512880 证券 ETF)
- 可转债 (T+0, 无涨跌停)

本示例与 ch09_funds.py 的交易逻辑、参数、数据源完全一致，
只把策略入口从类风格换成函数式：

1. __init__ 里的状态 (grid_step / lot_size / last_buy_price) -> ctx 属性
   （在 initialize 中赋值）
2. Strategy.on_bar 方法 -> 模块级 on_bar(ctx, bar) 函数
3. strategy=ETFGridStrategy -> strategy=on_bar + initialize=initialize

跨 Bar 的状态 (last_buy_price) 直接挂在 ctx 上读写，作用与类风格的 self 属性一致。
两份示例的回测统计输出应完全一致，这说明函数式与类风格在引擎层等价。
"""

from typing import Any, Optional

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar


# 模拟数据生成 (震荡市)
def generate_etf_data(length: int = 200) -> pd.DataFrame:
    """生成 ETF 模拟数据."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=length, freq="D")

    # 构造正弦波震荡
    x = np.linspace(0, 4 * np.pi, length)
    prices = 4.0 + 0.2 * np.sin(x) + np.random.normal(0, 0.05, length)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 0.05,
            "low": prices - 0.05,
            "close": prices,
            "volume": 1000000,
            "symbol": "510300",  # 沪深300 ETF
        }
    )
    return df


def initialize(ctx: Any) -> None:
    """
    初始化 ETF 网格策略状态，等价于类风格的 __init__.

    注意 last_buy_price 的初值必须是 None 而不是 0.0：on_bar 用
    `is None` 判断「还没建过底仓」，写成 0.0 会让初始建仓分支永远进不去。

    另外类风格版没有设置 warmup_period，本示例也不设置——
    预热期会改变被跳过的 Bar 数量，凭空加上会让两版输出不再一致。
    """
    ctx.grid_step = 0.01  # 网格间距 1%
    ctx.lot_size = 1000  # 每次买卖 1000 股
    # 类风格写 `self.last_buy_price: Optional[float] = None`；函数式不能直接在
    # ctx 属性上写类型标注（mypy 只允许标注 self 属性），故先声明局部变量再赋值。
    last_buy_price: Optional[float] = None
    ctx.last_buy_price = last_buy_price


def on_bar(ctx: Any, bar: Bar) -> None:
    """收到 Bar 事件的回调，等价于类风格的 on_bar；self 全部换成 ctx."""
    symbol = bar.symbol
    price = bar.close

    # 获取当前持仓
    pos = ctx.get_position(symbol)

    # 初始化建仓 (如果空仓)
    if pos == 0 and ctx.last_buy_price is None:
        ctx.log(f"初始建仓 @ {price:.3f}")
        ctx.buy(symbol, ctx.lot_size * 5)  # 底仓 5000 股
        ctx.last_buy_price = price
        return

    if ctx.last_buy_price is None:
        return

    # 网格买入逻辑：比上次买入价跌了 step
    if price < ctx.last_buy_price * (1 - ctx.grid_step):
        ctx.log(f"网格买入: {price:.3f} < {ctx.last_buy_price:.3f} * (1-1%)")
        ctx.buy(symbol, ctx.lot_size)
        ctx.last_buy_price = price  # 更新基准价

    # 网格卖出逻辑：比上次买入价涨了 step
    elif price > ctx.last_buy_price * (1 + ctx.grid_step):
        if pos >= ctx.lot_size:
            ctx.log(f"网格卖出: {price:.3f} > {ctx.last_buy_price:.3f} * (1+1%)")
            ctx.sell(symbol, ctx.lot_size)
            # 卖出后，基准价上移？或者保持不变？
            # 简单网格通常卖出后不更新基准价，或者是基于中枢动态调整
            # 这里为了演示简单，我们假设价格回升，基准价也随之上移
            ctx.last_buy_price = price


if __name__ == "__main__":
    df = generate_etf_data()

    print("开始运行第 9 章 ETF 网格策略示例...")

    # 基金回测配置：免印花税
    result = aq.run_backtest(
        strategy=on_bar,
        initialize=initialize,
        data=df,
        initial_cash=100_000,
        commission_rate=0.0001,  # 基金佣金通常较低 (万一)
        stamp_tax_rate=0.0,  # ETF 免印花税
    )

    # 打印最终结果
    metrics = result.metrics_df
    end_value = (
        metrics.loc["end_market_value", "value"]
        if "end_market_value" in metrics.index
        else 0.0
    )
    print(f"最终权益: {float(str(end_value)):.2f}")
