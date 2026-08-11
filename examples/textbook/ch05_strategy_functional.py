"""
第 5 章：构建第一个策略（函数式写法）.

本示例详细展示了一个完整策略的结构，重点介绍：
1. **策略生命周期**：函数式下四个生命周期钩子一一对应类风格的方法——
   `__init__` -> `initialize`、`on_start` -> `on_start`、
   `on_bar` -> `on_bar`、`on_stop` -> `on_stop`（均为模块级函数，首参为 ctx）
2. **数据获取**：使用 `get_history` 获取过去 N 天的数据
3. **交易接口**：使用 `buy`, `sell` 和 `order_target_percent`
4. **日志记录**：使用 `ctx.log` 记录关键信息

本示例与 ch05_strategy.py 的交易逻辑、参数、数据源完全一致，
只把策略入口从类风格换成函数式：

1. __init__ 里的参数与状态 -> ctx 属性（在 initialize 中赋值）
2. Strategy.on_start / on_bar / on_stop 方法 -> 模块级 (ctx, ...) 函数
3. strategy=MyFirstStrategy -> strategy=on_bar + initialize=initialize

两份示例的回测统计输出应完全一致，这说明函数式与类风格在引擎层等价。

策略逻辑 (双均线改进版)：
- 计算 5日均线 (MA5) 和 20日均线 (MA20)
- 金叉 (MA5 > MA20) 且无持仓 -> 买入
- 死叉 (MA5 < MA20) 且有持仓 -> 卖出
- 增加风控：如果亏损超过 5%，强制止损
"""

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar


# 模拟数据生成 (与第3章相同，方便复现)
def generate_mock_data(length: int = 500) -> pd.DataFrame:
    """生成模拟数据."""
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=length, freq="D")
    prices = 100 + np.cumsum(np.random.randn(length))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 100000,
            "symbol": "MOCK",
        }
    )
    return df


# ------------------------------------------------------------------------------
# 1. 初始化 (Initialization)
# ------------------------------------------------------------------------------
def initialize(ctx: Any) -> None:
    """
    策略状态初始化，等价于类风格的 __init__.

    类风格通过构造器参数 (short_window / long_window / stop_loss_pct) 传参；
    本示例不做参数扫描，因此函数式直接在这里写默认值即可。

    关键差异：函数式没有类体，warmup_period 必须在这里挂到 ctx 上。
    引擎会取 ctx.warmup_period 与 run_backtest(warmup_period=...) 的较大值，
    所以在此赋值即可生效。

    注意：引擎还有一路“按指标调用自动推断 warmup”的机制，但它靠 AST 解析策略
    类体实现；函数式下被解析的是引擎内部的 FunctionalStrategy 而非本文件的
    on_bar，推断结果恒为 0。因此函数式必须像这样显式赋值，不能依赖自动推断。
    """
    # 策略参数
    ctx.short_window = 5
    ctx.long_window = 20
    ctx.stop_loss_pct = 0.05

    # 内部状态变量
    ctx.entry_price = 0.0  # 记录开仓价格

    # 设置预热期 (Warmup Period)
    # 本示例会请求 long_window + 1 根数据，并用 [:-1] 排除当前 Bar，
    # 因此预热期也需要同步加 1，避免首次回调拿到前导 NaN。
    ctx.warmup_period = ctx.long_window + 1


# ------------------------------------------------------------------------------
# 2. 启动回调 (On Start)
# ------------------------------------------------------------------------------
def on_start(ctx: Any) -> None:
    """回测开始时触发. 此时引擎已就绪，可以进行一些初始化操作."""
    ctx.log("策略启动！")
    ctx.log(
        f"参数设置: MA{ctx.short_window} vs MA{ctx.long_window}, "
        f"止损={ctx.stop_loss_pct:.1%}"
    )


# ------------------------------------------------------------------------------
# 3. Bar 数据回调 (On Bar) - 核心逻辑
# ------------------------------------------------------------------------------
def on_bar(ctx: Any, bar: Bar) -> None:
    """每根 K 线走完时触发；与类风格版逻辑一致，self 全部换成 ctx."""
    symbol = bar.symbol

    # 3.1 获取历史数据
    # count=21 表示获取过去 21 根 Bar (包含当前这根)
    closes = ctx.get_history(count=ctx.long_window + 1, symbol=symbol, field="close")

    # 再次检查数据长度 (防御性编程)
    if len(closes) < ctx.long_window + 1:
        return

    # 3.2 计算技术指标
    # 使用切片 [:-1] 排除当前 Bar，只用截止到昨天的数据计算信号 (避免未来函数)
    # 这里的逻辑假设我们在今天收盘后计算信号，明天开盘交易
    history_closes = closes[:-1]
    ma_short = history_closes[-ctx.short_window :].mean()
    ma_long = history_closes[-ctx.long_window :].mean()

    # 3.3 获取账户信息
    current_pos = ctx.get_position(symbol)

    # 3.4 交易逻辑

    # 情况 A: 持仓中 -> 检查止损或死叉
    if current_pos > 0:
        # 计算浮动盈亏比例
        pnl_pct = (bar.close - ctx.entry_price) / ctx.entry_price

        # 止损检查
        if pnl_pct < -ctx.stop_loss_pct:
            ctx.log(f"触发止损! 当前亏损: {pnl_pct:.2%}")
            ctx.close_position(symbol)  # 清仓
            return

        # 死叉卖出
        if ma_short < ma_long:
            ctx.log(
                f"死叉卖出 (MA{ctx.short_window}={ma_short:.2f} < "
                f"MA{ctx.long_window}={ma_long:.2f})"
            )
            ctx.close_position(symbol)  # 清仓

    # 情况 B: 空仓中 -> 检查金叉
    elif current_pos == 0:
        if ma_short > ma_long:
            ctx.log(
                f"金叉买入 (MA{ctx.short_window}={ma_short:.2f} > "
                f"MA{ctx.long_window}={ma_long:.2f})"
            )

            # 使用 order_target_percent 买入 95% 的资金
            ctx.order_target_percent(symbol=symbol, target_percent=0.95)

            # 记录开仓价格 (近似值，实际成交价要等订单成交后才知道，这里暂用
            # 收盘价代替)
            ctx.entry_price = bar.close


# ------------------------------------------------------------------------------
# 4. 结束回调 (On Stop)
# ------------------------------------------------------------------------------
def on_stop(ctx: Any) -> None:
    """回测结束时触发. 常用于统计结果或资源释放."""
    ctx.log("策略停止。")


if __name__ == "__main__":
    df = generate_mock_data()

    print("开始运行第 5 章示例策略...")
    result = aq.run_backtest(
        strategy=on_bar,
        initialize=initialize,
        on_start=on_start,
        on_stop=on_stop,
        data=df,
        initial_cash=100_000,
        commission_rate=0.0003,  # 万三手续费
    )

    # 打印最终资金
    metrics = result.metrics_df
    end_value = (
        metrics.loc["end_market_value", "value"]
        if "end_market_value" in metrics.index
        else 0.0
    )
    print(f"回测结束，最终权益: {float(str(end_value)):.2f}")
