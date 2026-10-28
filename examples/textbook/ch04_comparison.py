"""
第 4 章：回测框架对比 (Pandas vs Backtrader vs AKQuant).

本脚本通过实现同一个简单的“双均线策略” (Golden Cross)，展示三种不同风格的回测实现方式：
1. **Pandas (向量化)**：利用 DataFrame 进行全量矩阵运算，
   适合快速验证思路，但细节模拟能力有限。
2. **Backtrader (事件驱动 - Python)**：经典的 Python
   事件驱动框架，适合观察逐 Bar 状态推进。
3. **AKQuant (事件驱动 - Rust)**：在保持事件驱动建模方式的同时，
   将部分核心执行逻辑下沉到 Rust。

说明：
- 本脚本主要用于展示三种回测范式的实现差异，不作为通用性能结论。
- 实际耗时会受到策略写法、数据规模、指标计算位置、依赖版本与运行环境影响。

策略逻辑：
- 当 5 日均线 > 20 日均线 (金叉) -> 全仓买入
- 当 5 日均线 < 20 日均线 (死叉) -> 清仓卖出
"""

import time

import numpy as np
import pandas as pd


# 模拟数据生成 (为了方便演示，不依赖外部文件)
def generate_mock_data(length: int = 1000) -> pd.DataFrame:
    """生成模拟数据."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-02", periods=length, freq="B")
    prices = 100 + np.cumsum(np.random.randn(length))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 100000,
        }
    )
    df["symbol"] = "MOCK"
    return df


# ==============================================================================
# 1. Pandas 向量化回测
# ==============================================================================
def run_pandas_backtest(df: pd.DataFrame) -> None:
    """运行 Pandas 向量化回测."""
    print("\n[Pandas] 开始向量化回测...")
    start_time = time.time()
    initial_cash = 100000.0
    target_weight = 0.95

    # 1. 计算指标 (全量计算)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()

    # 2. 生成信号 (1: 持仓, 0: 空仓)
    # shift(1) 是为了避免未来函数：今天的信号只能基于昨天的收盘价计算，用于今天的交易
    df["signal"] = np.where(df["ma5"] > df["ma20"], 1.0, 0.0)
    df["position"] = df["signal"].shift(1).fillna(0.0)

    # 3. 逐笔订单复利（使用 open 成交，最后未平仓按 close 估值）
    pos_diff = df["position"].diff().fillna(df["position"])
    entry_prices = df.loc[pos_diff > 0, "open"].to_numpy(dtype=float)
    exit_prices = df.loc[pos_diff < 0, "open"].to_numpy(dtype=float)
    if entry_prices.size > exit_prices.size:
        exit_prices = np.append(exit_prices, float(df["close"].iloc[-1]))

    final_equity = initial_cash
    if entry_prices.size > 0 and exit_prices.size > 0:
        trades_count = min(entry_prices.size, exit_prices.size)
        entry_used = entry_prices[:trades_count]
        exit_used = exit_prices[:trades_count]
        factors = (1.0 - target_weight) + target_weight * (exit_used / entry_used)
        final_equity = initial_cash * float(np.prod(factors))
    cumulative_return = final_equity / initial_cash - 1.0

    print(f"[Pandas] 耗时: {time.time() - start_time:.4f}s")
    print(f"[Pandas] 累计收益: {cumulative_return:.2%}")


# ==============================================================================
# 2. Backtrader 事件驱动回测
# ==============================================================================
def run_backtrader_backtest(df: pd.DataFrame) -> None:
    """运行 Backtrader 回测."""
    try:
        import backtrader as bt  # type: ignore
    except ImportError:
        print("\n[Backtrader] 未安装 backtrader，跳过演示 (pip install backtrader)")
        return

    print("\n[Backtrader] 开始事件驱动回测...")

    class SmaCross(bt.Strategy):
        params = (
            ("pfast", 5),
            ("pslow", 20),
        )

        def __init__(self) -> None:
            self.sma1 = bt.ind.SMA(period=self.params.pfast)  # type: ignore
            self.sma2 = bt.ind.SMA(period=self.params.pslow)  # type: ignore
            self.crossover = bt.ind.CrossOver(self.sma1, self.sma2)

        def next(self) -> None:
            if not self.position:
                if self.crossover > 0:
                    self.buy()
            elif self.crossover < 0:
                self.close()

    cerebro = bt.Cerebro()

    # 转换数据格式
    data = bt.feeds.PandasData(
        dataname=df.set_index("date"),
        # Backtrader 默认不包含 symbol，这里仅演示单标的
    )
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross)
    cerebro.broker.setcash(100000.0)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    start_time = time.time()
    cerebro.run()
    end_val = cerebro.broker.getvalue()
    cumulative_return = end_val / 100000.0 - 1.0

    print(f"[Backtrader] 耗时: {time.time() - start_time:.4f}s")
    print(f"[Backtrader] 最终资金: {end_val:.2f}")
    print(f"[Backtrader] 累计收益: {cumulative_return:.2%}")


# ==============================================================================
# 3. AKQuant 事件驱动回测
# ==============================================================================
def run_akquant_backtest(df: pd.DataFrame) -> None:
    """运行 AKQuant 回测."""
    import akquant as aq
    from akquant import Bar, Strategy

    print("\n[AKQuant] 开始事件驱动回测 (Rust Engine)...")

    class AKStrategy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.ma_short = 5
            self.ma_long = 20
            self.warmup_period = 20

        def on_bar(self, bar: Bar) -> None:
            symbol = bar.symbol
            closes = self.get_history(
                count=self.ma_long + 1, symbol=symbol, field="close"
            )
            if len(closes) < self.ma_long + 1:
                return

            # 为了避免未来函数，我们使用 [:-1] 切片，仅使用截止到昨天的数据
            # 或者，如果我们在收盘后交易（日线级别通常假设次日开盘成交），可以使用当前值
            # 这里为了演示方便，直接使用当前值计算信号，但在真实交易中要注意信号产生的
            # 时机

            # 计算均线
            ma5 = closes[-self.ma_short :].mean()
            ma20 = closes[-self.ma_long :].mean()

            pos = self.get_position(symbol)

            if ma5 > ma20 and pos == 0:
                self.order_target_percent(symbol=symbol, target_percent=0.95)
            elif ma5 < ma20 and pos > 0:
                self.close_position(symbol)

    start_time = time.time()
    result = aq.run_backtest(
        strategy=AKStrategy, data=df, initial_cash=100000.0, commission_rate=0.0
    )

    metrics = result.metrics_df
    end_value = 0.0
    if "end_market_value" in metrics.index:
        val = metrics.loc["end_market_value", "value"]
        end_value = float(str(val))
    else:
        # 尝试从 result.equity_curve 获取
        equity = result.equity_curve
        if not equity.empty:
            val = equity.iloc[-1]
            end_value = float(str(val))

    print(f"[AKQuant] 耗时: {time.time() - start_time:.4f}s")
    print(f"[AKQuant] 最终资金: {end_value:.2f}")
    print(f"[AKQuant] 累计收益: {end_value / 100000.0 - 1.0:.2%}")


if __name__ == "__main__":
    # 1. 准备一份共用的数据
    df = generate_mock_data(length=3000)  # 约 12 年的数据

    # 2. 运行对比
    run_pandas_backtest(df.copy())
    run_backtrader_backtest(df.copy())
    run_akquant_backtest(df.copy())
