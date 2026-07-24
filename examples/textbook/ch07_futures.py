"""
第 7 章：期货与衍生品策略 (Futures & Derivatives).

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
"""

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, Strategy


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


class FuturesTrendStrategy(Strategy):
    """期货趋势策略."""

    def __init__(self) -> None:
        """初始化策略."""
        super().__init__()
        self.ma_window = 10
        self.warmup_period = 10

        # 记录上一次的信号，避免重复发单
        self.last_signal = 0

    def on_bar(self, bar: Bar) -> None:
        """收到 Bar 事件的回调."""
        symbol = bar.symbol

        # 1. 获取历史数据
        closes = self.get_history(
            count=self.ma_window + 1, symbol=symbol, field="close"
        )
        if len(closes) < self.ma_window + 1:
            return

        # 2. 计算均线 (使用截止到昨天的数据)
        ma = closes[:-1][-self.ma_window :].mean()
        current_price = bar.close

        # 3. 获取持仓
        # position > 0: 多头
        # position < 0: 空头
        # position == 0: 空仓
        pos = self.get_position(symbol)

        # 4. 交易逻辑

        # 信号：价格 > MA -> 看多 (1)
        # 信号：价格 < MA -> 看空 (-1)
        signal = 1 if current_price > ma else -1

        if signal != self.last_signal:
            self.log(f"趋势反转! 价格={current_price:.0f}, MA={ma:.0f}, 信号={signal}")

            # 如果当前有反向持仓，先平仓
            if (signal == 1 and pos < 0) or (signal == -1 and pos > 0):
                self.close_position(symbol)

            # 开新仓 (做多或做空)
            # 这里的 quantity=1 表示 1 手
            if signal == 1:
                self.log("开多单 1 手")
                self.buy(symbol, 1)
            elif signal == -1:
                self.log("开空单 1 手")
                self.short(symbol, 1)

            self.last_signal = signal


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
        strategy=FuturesTrendStrategy,
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
    metrics = result.metrics_df
    final_val = (
        metrics.loc["end_portfolio_value", "value"]
        if "end_portfolio_value" in metrics.index
        else 0.0
    )
    print(f"最终权益: {float(str(final_val)):.2f}")
