"""
第 8 章：期权与衍生品策略 (Options & Derivatives)（函数式写法）.

本示例展示了期权交易的核心特性，特别是如何利用期权构建非线性损益结构。
示例策略：**备兑看涨 (Covered Call)**
这是一种保守的增强收益策略：在持有标的期货的同时，卖出看涨期权 (Short Call) 以收取
权利金。

适用场景：
- 对标的期货长期看涨，但预期短期内窄幅震荡或小幅上涨。
- 通过权利金收入降低持仓成本，提供一定的下跌保护。

交易逻辑：
1. 买入 1 手螺纹钢期货 (RB2310)。
2. 卖出 1 手虚值 (OTM) 看涨期权 (行权价 > 当前价)。
3. 到期时：
    - 若价格 < 行权价：期权归零，赚取全部权利金。
    - 若价格 > 行权价：期货被行权指派，最大收益锁定在 (行权价 - 开仓价) + 权利金。

本示例与 ch08_options.py 的交易逻辑、参数、数据源完全一致，
只把策略入口从类风格换成函数式：

1. on_start 里的状态 (future_symbol / option_symbol / has_position / bar_count)
   -> ctx 属性（在 initialize 中赋值）
2. Strategy.on_start / on_bar 方法 -> 模块级 (ctx, ...) 函数
3. strategy=CoveredCallStrategy -> strategy=on_bar + initialize=initialize

跨 Bar 的状态 (has_position / bar_count) 直接挂在 ctx 上读写，作用与类风格的
self 属性一致。两份示例的回测统计输出应完全一致，这说明函数式与类风格在引擎层等价。
"""

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, InstrumentConfig


# 模拟数据生成 (期货 + 期权)
def generate_option_data(length: int = 100) -> pd.DataFrame:
    """生成期权模拟数据."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=length, freq="D")

    # 标的期货价格 (震荡向上)
    futures_prices = np.linspace(3500, 3800, length) + np.random.normal(0, 50, length)

    # 虚值看涨期权价格 (简单模拟：随标的价格上涨而上涨，随时间衰减)
    # 假设行权价 K=3800
    K = 3800
    # 简单定价模型: Max(0, S-K) + TimeValue
    time_to_maturity = np.linspace(1.0, 0.0, length)  # 剩余时间从 1 到 0
    intrinsic_value = np.maximum(0, futures_prices - K)
    time_value = 200 * time_to_maturity * (futures_prices / K)  # 时间价值
    option_prices = intrinsic_value + time_value

    # 构造期货数据
    df_fut = pd.DataFrame(
        {
            "date": dates,
            "open": futures_prices,
            "high": futures_prices + 20,
            "low": futures_prices - 20,
            "close": futures_prices,
            "volume": 500000,
            "symbol": "RB2310",
        }
    )

    # 构造期权数据
    df_opt = pd.DataFrame(
        {
            "date": dates,
            "open": option_prices,
            "high": option_prices + 5,
            "low": option_prices - 5,
            "close": option_prices,
            "volume": 10000,
            "symbol": "RB2310-C-3800",  # 看涨期权，行权价 3800
        }
    )

    # 合并数据
    return pd.concat([df_fut, df_opt])


def initialize(ctx: Any) -> None:
    """
    初始化备兑看涨策略状态.

    类风格版把这些状态放在 on_start 里；函数式两种位置都可行
    （on_start(ctx) 里写 ctx.xxx = ... 效果相同），
    这里选 initialize 以贴合「构造期初始化」的语义。

    注意：类风格版没有设置 warmup_period，本示例也不设置——
    预热期会改变被跳过的 Bar 数量，凭空加上会让两版输出不再一致。
    """
    ctx.future_symbol = "RB2310"
    ctx.option_symbol = "RB2310-C-3800"
    ctx.has_position = False
    ctx.bar_count = 0


def on_start(ctx: Any) -> None:
    """策略启动回调，等价于类风格的 on_start."""
    ctx.log("策略启动: 备兑看涨 (Covered Call)")


def on_bar(ctx: Any, bar: Bar) -> None:
    """收到 Bar 事件的回调，等价于类风格的 on_bar；self 全部换成 ctx."""
    ctx.bar_count += 1

    # 简单逻辑：第一天开仓，一直持有到最后一天
    if not ctx.has_position:
        ctx.log(f"开仓: 买入期货 {ctx.future_symbol}, 卖出期权 {ctx.option_symbol}")

        # 1. 买入 1 手期货
        ctx.buy(ctx.future_symbol, 1)

        # 2. 卖出 1 手看涨期权 (收取权利金)
        ctx.sell(ctx.option_symbol, 1)

        ctx.has_position = True

    # 在最后一天平仓 (模拟到期结算)
    if ctx.bar_count == 99:
        ctx.log("到期平仓...")
        ctx.close_position(ctx.future_symbol)
        ctx.close_position(ctx.option_symbol)


if __name__ == "__main__":
    df = generate_option_data()

    print("开始运行商品期权策略示例...")

    # 配置合约
    rb_fut_config = InstrumentConfig(
        symbol="RB2310", asset_type="FUTURES", multiplier=10.0, margin_ratio=0.1
    )

    rb_opt_config = InstrumentConfig(
        symbol="RB2310-C-3800",
        asset_type="OPTION",  # 期权类型
        option_type="CALL",  # 看涨期权
        strike_price=3800.0,  # 行权价
        underlying_symbol="RB2310",  # 标的期货合约（期权必填）
        multiplier=10.0,  # 1张期权对应1手期货 (10吨)
        margin_ratio=0.0,  # 期权买方不收保证金，卖方收 (引擎会自动计算卖方保证金)
    )

    from akquant import (
        BacktestConfig,
        ChinaOptionsConfig,
        ChinaOptionsFeeConfig,
        StrategyConfig,
    )

    config = BacktestConfig(
        strategy_config=StrategyConfig(initial_cash=500_000),
        instruments_config=[rb_fut_config, rb_opt_config],
        china_options=ChinaOptionsConfig(
            fee_per_contract=5.0,
            fee_by_symbol_prefix=[
                ChinaOptionsFeeConfig(
                    symbol_prefix="RB",
                    commission_per_contract=8.0,
                )
            ],
        ),
    )

    result = aq.run_backtest(
        strategy=on_bar,
        initialize=initialize,
        on_start=on_start,
        data=df,
        config=config,
    )

    # 打印最终结果
    metrics = result.metrics_df
    end_value = (
        metrics.loc["end_market_value", "value"]
        if "end_market_value" in metrics.index
        else 0.0
    )
    print(f"最终权益: {float(str(end_value)):.2f}")
