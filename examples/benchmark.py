import logging
import time

import akquant as aq
import pandas as pd
from benchmark_utils import get_benchmark_data

"""
Akquant 多标的性能基准测试脚本 (Online Mode)
========================================

本脚本演示如何在 Akquant 中进行多标的 (Multi-Symbol) 回测。
它同时交易 3 只股票，展示了引擎处理并发数据流的能力。

展示的关键特性：
1.  **多标的支持**: 同时回测 3 只股票 (STOCK_A, STOCK_B, STOCK_C)。
2.  **独立指标管理**: 为每个标的维护独立的指标状态。
3.  **统一数据流**: 引擎自动合并并按时间顺序分发多标的行情数据。
4.  **资金共享**: 所有标的共享同一个账户资金池。
"""

# 0. 设置日志
# 获取 logger 并设置级别
logger = aq.get_logger()
# 如果需要输出到文件，可以手动配置 logging

file_handler = logging.FileHandler("backtest.log", mode="w")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
logger.addHandler(file_handler)
logger.info("Logging to file: backtest.log")

# 1. 生成数据
SYMBOLS = ["STOCK_A", "STOCK_B", "STOCK_C"]
DATA_SIZE = 5000  # 约 20 年的日线数据

logger.info(f"Generating data for {len(SYMBOLS)} symbols ({DATA_SIZE} bars each)...")
t0_gen = time.time()

data_dict = {}
for i, symbol in enumerate(SYMBOLS):
    # 生成数据 (使用 'B' 代表工作日频率，从 1990 年开始)
    # 固定种子以确保结果可复现
    df = get_benchmark_data(
        DATA_SIZE, symbol, freq="B", start_time="1990-01-01", seed=42 + i
    )

    # 重命名列
    df = df.rename(
        columns={
            "date": "日期",
            "open": "开盘",
            "high": "最高",
            "low": "最低",
            "close": "收盘",
            "volume": "成交量",
            "symbol": "股票代码",
        }
    )
    data_dict[symbol] = df

logger.info(f"Data generation took {time.time() - t0_gen:.4f}s")


# 2. 定义多标的策略
class MultiSymbolStrategy(aq.Strategy):
    """
    多标的移动平均线交叉策略.

    为每个标的维护独立的 SMA 指标。.
    """

    def __init__(self) -> None:
        """初始化策略."""
        self.qty = 100
        self.symbols = SYMBOLS

        # 为每个标的创建独立的指标组
        # 使用字典映射: symbol -> indicators
        self.indicators = {}
        for sym in self.symbols:
            self.indicators[sym] = {"sma5": aq.SMA(5), "sma20": aq.SMA(20)}

        # 禁用 Python 侧历史数据缓存以最大化性能
        self.set_history_depth(0)

    def on_bar(self, bar: aq.Bar) -> None:
        """
        Bar 数据事件处理函数.

        引擎会按时间顺序推送所有订阅标的的 Bar。.
        """
        symbol = bar.symbol
        close = bar.close

        # 获取该标的的指标组
        inds = self.indicators.get(symbol)
        if not inds:
            return

        # 实时更新指标
        sma5 = inds["sma5"].update(close)
        sma20 = inds["sma20"].update(close)

        # 检查指标是否就绪
        if sma5 is None or sma20 is None:
            return

        # 打印详细日志 (仅针对前 500 个 bar 的 STOCK_A，用于演示)
        if (
            symbol == "STOCK_A"
            and self.get_position(symbol) == 0
            and sma5 > sma20
            and self.qty > 0
        ):
            # 即将买入
            pass

        # 获取当前持仓
        position = self.get_position(symbol)
        # 交易逻辑 (简单的金叉/死叉)
        # 记录交易信号详情
        action = None

        if sma5 > sma20:
            if position == 0:
                # 金叉且无持仓 -> 买入
                self.buy(symbol=symbol, quantity=self.qty)
                action = "BUY_OPEN"
            elif position < 0:
                # 平空反手
                self.buy(symbol=symbol, quantity=abs(position) + self.qty)
                action = "BUY_REVERSE"

        elif sma5 < sma20:
            if position > 0:
                # 死叉且持有多单 -> 平仓
                self.close_position(symbol)
                self.cancel_all_orders()
                action = "SELL_CLOSE"

        # 如果有动作，打印当时的指标状态
        if action and symbol == "STOCK_A":
            # 为了避免日志过多，我们只打印前几笔交易的详细过程
            # 通过一个计数器来控制（这里简单用时间判断，或者假设这是演示脚本）
            # 更好的方式是看是否有成交产生，但 on_bar 是发单时刻
            current_dt = pd.to_datetime(bar.timestamp, unit="ns")
            if current_dt.year == 1990:  # 只看 1990 年的
                log_msg = (
                    f"[{current_dt}] {symbol} Signal: {action} | "
                    f"SMA5: {sma5:.2f}, SMA20: {sma20:.2f} | "
                    f"Pos: {position}"
                )
                logger.info(log_msg)
                print(log_msg)
                logger.info(log_msg)


if __name__ == "__main__":
    # 3. 运行回测
    logger.info("Running multi-symbol backtest...")

    t0_run = time.time()

    strategy = MultiSymbolStrategy()

    # run_backtest 自动处理 Dict[str, DataFrame] 输入：
    # 1. 转换所有 DataFrame 为 Bar 列表
    # 2. 合并并按时间排序
    # 3. 注册所有标的到引擎
    result = aq.run_backtest(
        data=data_dict,
        strategy=strategy,
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        execution_mode="next_open",
        lot_size=100,
        show_progress=True,
    )

    t_run = time.time() - t0_run
    logger.info(f"Engine execution took {t_run:.4f}s")

    # 4. 打印报告
    total_bars = len(SYMBOLS) * DATA_SIZE
    throughput = total_bars / t_run if t_run > 0 else 0

    print("-" * 50)
    print("Backtest Results (Akquant - Multi Symbol)")
    print("-" * 50)
    print(f"Symbols       : {SYMBOLS}")
    print(f"Total Bars    : {total_bars}")
    print(f"Execution Time: {t_run:.4f} s")
    print(f"Throughput    : {throughput:,.0f} bars/sec")
    print(f"Total Return  : {result.metrics.total_return * 100:.2f}%")
    print(f"Total Trades  : {result.trade_metrics.total_closed_trades}")
    print("-" * 50)

    # 打印详细统计
    print("-" * 50)
    print(
        f"Final Equity: {result.equity_curve[-1][1] if result.equity_curve else 'N/A'}"
    )
    print(f"Total Commission: {result.trade_metrics.total_commission:.2f}")
    print(f"Net PnL: {result.trade_metrics.net_pnl:.2f}")

    # 打印每个标的的盈亏情况 (从 closed_trades 统计)
    print("-" * 50)
    print("PnL by Symbol:")
    pnl_by_symbol: dict[str, float] = {}
    for trade in result.trades:
        pnl_by_symbol[trade.symbol] = pnl_by_symbol.get(trade.symbol, 0) + trade.pnl

    for sym in SYMBOLS:
        print(f"  {sym}: {pnl_by_symbol.get(sym, 0.0):.2f}")

    # 打印详细成交记录
    print("-" * 50)
    print("Trade Details (Last 20 trades):")
    headers = (
        f"{'Symbol':<10} {'Dir':<5} {'Entry Time':<18} {'Exit Time':<18} "
        f"{'Qty':<6} {'Entry':<10} {'Exit':<10} {'Gross':<10} "
        f"{'Comm':<8} {'Net PnL':<10}"
    )
    print(headers)

    # Sort trades by exit time
    sorted_trades = sorted(result.trades, key=lambda x: x.exit_time)

    # Print last 20 trades
    for trade in sorted_trades[-20:]:
        entry_dt = pd.to_datetime(trade.entry_time, unit="ns").strftime(
            "%Y-%m-%d %H:%M"
        )
        exit_dt = pd.to_datetime(trade.exit_time, unit="ns").strftime("%Y-%m-%d %H:%M")
        side = (
            str(trade.side).split(".")[-1]
            if "." in str(trade.side)
            else str(trade.side)
        )

        # Use new fields from updated Rust struct
        # pnl is now Gross PnL (Backtrader style)
        # net_pnl is now available directly
        gross_pnl = trade.pnl
        net_pnl = trade.net_pnl

        print(
            f"{trade.symbol:<10} {side:<5} {entry_dt:<18} {exit_dt:<18} "
            f"{int(trade.quantity):<6} {trade.entry_price:<10.2f} "
            f"{trade.exit_price:<10.2f} {gross_pnl:<10.2f} "
            f"{trade.commission:<8.2f} {net_pnl:<10.2f}"
        )

    # Verify calculation for the last trade
    if sorted_trades:
        last_trade = sorted_trades[-1]
        print("\nVerification for last trade:")
        print(f"  Gross PnL (from struct): {last_trade.pnl:.2f}")
        print(f"  Commission (from struct): {last_trade.commission:.2f}")
        print(f"  Net PnL (from struct):   {last_trade.net_pnl:.2f}")

        calc_net = last_trade.pnl - last_trade.commission
        print(f"  Calculated Net (Gross - Comm): {calc_net:.2f}")

        if abs(calc_net - last_trade.net_pnl) < 0.01:
            print("  ✅ PnL Calculation Verified (Net = Gross - Commission)")
        else:
            print(
                f"  ❌ PnL Calculation Mismatch! Diff: "
                f"{abs(calc_net - last_trade.net_pnl):.4f}"
            )

    # 5. 可视化
    print("-" * 50)
    print("Plotting results...")

    # 构建组合基准收益率 (简单平均)
    bench_returns = pd.DataFrame()
    for sym, df in data_dict.items():
        # prepare_dataframe 在 run_backtest 内部被调用，所以这里的 df 还是原始的
        # 我们需要手动处理一下时间戳以便对齐
        temp_df = df.copy()
        # 列名已被修改为中文
        if "日期" in temp_df.columns:
            temp_df = temp_df.set_index("日期")
        elif "date" in temp_df.columns:
            temp_df = temp_df.set_index("date")

        s_ret = temp_df["收盘"].pct_change().fillna(0)
        bench_returns[sym] = s_ret

    # 假设等权重持有
    combined_bench = bench_returns.mean(axis=1)

    aq.plot_result(
        result,
        show=False,
        title="Multi Symbol Backtest Result",
        filename="benchmark_multi_result.html",
    )
