import akquant as aq
import akshare as ak
import pandas as pd
from akquant import Bar, Strategy
from akquant.config import BacktestConfig, RiskConfig, StrategyConfig

df = ak.stock_zh_a_daily(symbol="sh600000", start_date="20200131", end_date="20230228")


# def generate_data() -> pd.DataFrame:
#     """Generate dummy data for backtesting."""
#     n = 1000000  # 精确生成1000万根K线

#     # 1. 生成日期序列（使用periods确保数量精确）
#     dates = pd.date_range(start="2020-01-01", periods=n, freq="1min")

#     # 2. 生成合理价格序列（关键优化）
#     # - 收益率均值设为0（避免1000万次累积后价格爆炸/归零）
#     # - 标准差0.0001（约0.01%波动，符合5分钟K线特征）
#     # - 使用np.exp避免cumprod数值溢出，更稳定
#     np.random.seed(42)  # 可复现性（生产环境可移除）
#     log_returns = np.random.normal(0, 0.0001, n)
#     price = 100 * np.exp(np.cumsum(log_returns))

#     # 3. 构建DataFrame（内存优化关键）
#     df = pd.DataFrame(
#         {
#             "date": dates,
#             "open": price.astype(np.float32),  # float32节省50%内存
#             "high": (price * 1.005).astype(np.float32),  # 缩小振幅至±0.5%更合理
#             "low": (price * 0.995).astype(np.float32),
#             "close": price.astype(np.float32),
#             "volume": np.full(n, 10000, dtype=np.int32),  # int32替代默认int64
#             "symbol": pd.Categorical(["600000"] * n),  # category类型节省90%+内存
#         }
#     )

#     # 4. 强制修正：确保high >= max(open,close) 且 low <= min(open,close)
#     # （解决原逻辑中open=close导致K线形态失真的问题）
#     df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
#     df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))

#     return df


# 生成数据（注意：需8-12GB可用内存）
# df = generate_data()


class MyStrategy(Strategy):
    """
    Example strategy for testing broker execution.

    This strategy buys on the first bar and holds for 100 bars or until 10% profit.
    """

    def __init__(self) -> None:
        """Initialize strategy state."""
        self.bars_held: dict[str, int] = {}
        self.entry_prices: dict[str, float] = {}

    def on_bar(self, bar: Bar) -> None:
        """
        Handle bar data event.

        :param bar: The current bar data
        """
        symbol = bar.symbol
        pos = self.get_position(symbol)

        # 维护持仓计数
        if pos > 0:
            if symbol not in self.bars_held:
                self.bars_held[symbol] = 0
            self.bars_held[symbol] += 1
        else:
            # 如果没有持仓，清理状态
            if symbol in self.bars_held:
                del self.bars_held[symbol]
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]

        # 交易逻辑
        if pos == 0:
            # 简单示例：无条件买入
            print(
                f"Buy Signal for {symbol}: Open={bar.open}, High={bar.high}, "
                f"Low={bar.low}, Close={bar.close}"
            )
            self.order_target_percent(target_percent=1.0, symbol=symbol)
            # 初始化计数器 (虽然会在下个 bar 的 pos>0 分支中自增，但这里先占位)
            self.bars_held[symbol] = 0
            self.entry_prices[symbol] = bar.close

        elif pos > 0:
            entry_price = self.entry_prices.get(symbol, bar.close)
            current_bars_held = self.bars_held.get(symbol, 0)

            # 计算收益率
            pnl_pct = (bar.close - entry_price) / entry_price

            # 止盈条件：收益率 >= 10%
            if pnl_pct >= 0.10:
                self.sell(symbol, pos)
                print(
                    f"Take Profit Triggered for {symbol}: Entry={entry_price}, "
                    f"Current={bar.close}, PnL={pnl_pct:.2%}"
                )

            # 持仓时间条件：持有满 100 个 Bar
            elif current_bars_held >= 100:
                self.sell(symbol, pos)


# 配置风险参数：safety_margin
risk_config = RiskConfig(safety_margin=0.0001)
strategy_config = StrategyConfig(risk=risk_config)
backtest_config = BacktestConfig(strategy_config=strategy_config)

engine = aq.Engine()
engine.set_force_session_continuous(True)

result = aq.run_backtest(
    strategy=MyStrategy,
    data=df,
    initial_cash=500_000,
    commission_rate=0.0,
    stamp_tax_rate=0.0,
    transfer_fee_rate=0.0,
    min_commission=0.0,
    lot_size=1,
    execution_mode=aq.ExecutionMode.NextHighLowMid,  # type: ignore
    config=backtest_config,
)

pd.set_option("display.max_columns", None)
print(result)
print(result.cash_curve)
