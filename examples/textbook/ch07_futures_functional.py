"""
第 7 章：期货与衍生品策略 (Futures & Derivatives)（函数式写法）.

本示例展示了期货交易的核心特性：
1. **保证金 (Margin)**：只需缴纳少量资金即可控制大额合约。
2. **杠杆 (Leverage)**：放大收益与风险。
3. **做空 (Short Selling)**：使用 `short()` / `cover()` 显式表达开空与平空。
4. **合约乘数 (Multiplier)**：一手合约代表的价值。

示例场景：
- 交易品种：螺纹钢期货 (RB)
- 逻辑：简单的动量策略
    - 价格 > 均线 -> 做多
    - 价格 < 均线 -> 做空
- 演示保证金占用和盈亏计算

本示例与 ch07_futures.py 的交易逻辑、参数、数据源完全一致，
只把策略入口从类风格换成函数式：

1. __init__ 里的状态 (ma_window / warmup_period / last_signal) -> ctx 属性
   （在 initialize 中赋值）
2. Strategy.on_bar 方法 -> 模块级 on_bar(ctx, bar) 函数
3. strategy=FuturesTrendStrategy -> strategy=on_bar + initialize=initialize

跨 Bar 的状态 (last_signal) 直接挂在 ctx 上读写，作用与类风格的 self 属性一致。
两份示例的回测统计输出应完全一致，这说明函数式与类风格在引擎层等价。
"""

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar


# 模拟数据生成 (模拟螺纹钢期货)
def generate_futures_data(length: int = 100) -> pd.DataFrame:
    """生成期货模拟数据."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=length, freq="D")

    # 构造一个先涨后跌的趋势
    trend = np.concatenate(
        [
            np.linspace(3500, 4000, 50),  # 上涨趋势
            np.linspace(4000, 3200, 50),  # 下跌趋势
        ]
    )
    noise = np.random.normal(0, 20, length)
    prices = trend + noise

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 30,
            "low": prices - 30,
            "close": prices,
            "volume": 500000,
            "symbol": "RB2310",  # 螺纹钢 2310 合约
        }
    )
    return df


def initialize(ctx: Any) -> None:
    """
    初始化趋势策略状态，等价于类风格的 __init__.

    关键差异：函数式没有类体，warmup_period 必须在这里挂到 ctx 上。
    引擎会取 ctx.warmup_period 与 run_backtest(warmup_period=...) 的较大值，
    所以在此赋值即可生效。

    注意：引擎还有一路“按指标调用自动推断 warmup”的机制，但它靠 AST 解析策略
    类体实现；函数式下被解析的是引擎内部的 FunctionalStrategy 而非本文件的
    on_bar，推断结果恒为 0。因此函数式必须像这样显式赋值，不能依赖自动推断。
    """
    ctx.ma_window = 10
    # 下面用 get_history(count=ma_window + 1) 取 11 根，warmup 必须 >= 11，
    # 否则首位是 NaN，均线恒为 NaN，信号会永远锁在初值上（示例跑不出反转）。
    ctx.warmup_period = ctx.ma_window + 1

    # 记录上一次的信号，避免重复发单
    ctx.last_signal = 0


def on_bar(ctx: Any, bar: Bar) -> None:
    """收到 Bar 事件的回调，等价于类风格的 on_bar；self 全部换成 ctx."""
    symbol = bar.symbol

    # 1. 获取历史数据
    closes = ctx.get_history(count=ctx.ma_window + 1, symbol=symbol, field="close")
    if len(closes) < ctx.ma_window + 1:
        return

    # 2. 计算均线 (使用截止到昨天的数据)
    ma = closes[:-1][-ctx.ma_window :].mean()
    current_price = bar.close

    # 3. 获取持仓
    # position > 0: 多头
    # position < 0: 空头
    # position == 0: 空仓
    pos = ctx.get_position(symbol)

    # 4. 交易逻辑

    # 信号：价格 > MA -> 看多 (1)
    # 信号：价格 < MA -> 看空 (-1)
    signal = 1 if current_price > ma else -1

    if signal != ctx.last_signal:
        ctx.log(f"趋势反转! 价格={current_price:.0f}, MA={ma:.0f}, 信号={signal}")

        # 如果当前有反向持仓，先平仓
        if (signal == 1 and pos < 0) or (signal == -1 and pos > 0):
            ctx.close_position(symbol)

        # 开新仓 (做多或做空)
        # 这里的 quantity=1 表示 1 手
        if signal == 1:
            ctx.log("开多单 1 手")
            ctx.buy(symbol, 1)
        elif signal == -1:
            ctx.log("开空单 1 手")
            ctx.short(symbol, 1)

        ctx.last_signal = signal


if __name__ == "__main__":
    df = generate_futures_data()

    print("开始运行第 7 章期货策略示例...")

    # 1. 定义期货合约属性 (关键步骤)
    # 螺纹钢：乘数 10，保证金 10%
    from akquant import (
        BacktestConfig,
        ChinaFuturesConfig,
        ChinaFuturesInstrumentTemplateConfig,
        ChinaFuturesValidationConfig,
        InstrumentConfig,
        StrategyConfig,
    )

    rb_config = InstrumentConfig(
        symbol="RB2310",
        asset_type="FUTURES",  # 资产类型
        multiplier=10.0,  # 合约乘数 (1手 = 10吨)
        margin_ratio=0.1,  # 保证金比率 (10%)
    )

    # 2. 运行回测
    # 使用 BacktestConfig 配置合约和资金参数
    config = BacktestConfig(
        strategy_config=StrategyConfig(initial_cash=500_000, commission_rate=0.0001),
        instruments_config=[rb_config],
        china_futures=ChinaFuturesConfig(
            enforce_sessions=False,
            instrument_templates_by_symbol_prefix=[
                ChinaFuturesInstrumentTemplateConfig(
                    symbol_prefix="RB",
                    multiplier=10.0,
                    margin_ratio=0.1,
                    tick_size=1.0,
                    lot_size=1.0,
                    commission_rate=0.0001,
                    enforce_tick_size=False,
                    enforce_lot_size=True,
                )
            ],
            validation_by_symbol_prefix=[
                ChinaFuturesValidationConfig(
                    symbol_prefix="RB",
                    enforce_tick_size=False,
                    enforce_lot_size=True,
                )
            ],
        ),
    )

    result = aq.run_backtest(
        strategy=on_bar,
        initialize=initialize,
        data=df,
        config=config,
        # 对很多“按当根收盘价记交易”的期货策略，显式指定 fill_policy=aq.CurrentClose()
        # 会比默认的“下一根开盘成交”更贴近人工记录口径。
        fill_policy=aq.CurrentClose(),
        # 注意：推荐显式声明滑点类型。
        # 0.0002 = 2 bps；如果写成 0.2，表示 20% 滑点，不是 0.2 个点。
        slippage={"type": "percent", "value": 0.0002},
    )

    print("\n" + "=" * 40)
    print("期货账户资金分析")
    print("=" * 40)

    # 打印最后几天的权益变动
    # 注意：期货有 leverage，portfolio_value 可能波动较大
    equity = result.equity_curve.tail()
    print(equity)

    print("\n保证金占用情况 (示例):")
    # 假设最后一天持仓 1 手，价格 3200
    # 保证金 = 3200 * 10 * 1 * 0.1 = 3200 元
    # 杠杆倍数 = 3200 * 10 / 3200 = 10 倍
    # metrics_df 的期末权益键是 end_market_value（等于 equity_curve 末值）。
    # 注意不是 end_portfolio_value——那个键不存在，配上 else 兜底会静默打印 0.00。
    metrics = result.metrics_df
    end_value = (
        metrics.loc["end_market_value", "value"]
        if "end_market_value" in metrics.index
        else 0.0
    )
    print(f"最终权益: {float(str(end_value)):.2f}")
