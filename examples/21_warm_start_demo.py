#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Warm Start (热启动) 示例.

演示如何保存回测状态并在后续运行中恢复，实现“断点续传”。
场景模拟：
1. 阶段一：运行 2021-2022 年的数据，计算长周期均线，建立底仓。
2. 保存状态 (Snapshot)。
3. 阶段二：加载 2023 年的数据，从快照恢复，继续运行策略。

注意：Warm Start 对于实盘交易至关重要，它可以让你每天只需加载最新的数据，
而不需要每次都重跑几年的历史数据来初始化策略状态。
"""

import os
from typing import List

import akquant as aq
import akquant.indicator as ind
import akshare as ak
import numpy as np
import pandas as pd
from akquant import Bar, IntParam, Strategy


# 1. 定义一个简单的均线策略
class MovingAverageStrategy(Strategy):
    """Simple moving average strategy for testing warm start."""

    fast_window = IntParam(10, ge=2, le=200, title="快线窗口")
    slow_window = IntParam(30, ge=3, le=500, title="慢线窗口")

    def on_start(self) -> None:
        """Call when strategy starts.

        Warm Start 注意：状态变量与指标的初始化必须收在
        ``if not self.is_restored:`` 分支内——恢复路径不会重新调用
        ``__init__``，此处若无条件执行会把 buy_count/sell_count 等
        累积计数清零，破坏跨阶段的连续性。
        """
        # 注册指标 (akquant 会自动处理指标状态的序列化)
        # 使用框架提供的 self.is_restored 判断是否是从快照恢复
        if not self.is_restored:
            # 记录一些状态变量，验证它们是否被正确保存和恢复
            self.buy_count = 0
            self.sell_count = 0
            self.total_volume = 0
            self._debug_count = 0
            self.sma_fast = ind.SMA(self.params.fast_window)
            self.sma_slow = ind.SMA(self.params.slow_window)
        else:
            print("[Strategy] Resumed from snapshot. Indicators restored.")

        self.register_precomputed_indicator("sma_fast", self.sma_fast)
        self.register_precomputed_indicator("sma_slow", self.sma_slow)
        print(
            f"[Strategy] Started. Fast={self.params.fast_window}, "
            f"Slow={self.params.slow_window}"
        )

    def on_resume(self) -> None:
        """Call only when strategy state is restored from snapshot."""
        print(
            "[Strategy] on_resume called. "
            f"Existing counts: buy={self.buy_count}, sell={self.sell_count}"
        )

    def on_bar(self, bar: Bar) -> None:
        """Call on every bar."""
        # 累加状态变量
        self.total_volume += int(bar.volume)

        # 简单的双均线逻辑
        self.sma_fast.update(bar.close)
        self.sma_slow.update(bar.close)

        fast = self.sma_fast.value
        slow = self.sma_slow.value

        # Debug: 打印前几个 bar 的指标值
        if self.total_volume < 1e7:  # 仅打印少量
            if not hasattr(self, "_debug_count"):
                self._debug_count = 0
            self._debug_count += 1
            # 增加 Debug 数量以覆盖 Warm Start 启动初期
            if self._debug_count < 20:
                print(
                    f"Debug: {pd.Timestamp(bar.timestamp, unit='ns')} "
                    f"Close={bar.close:.2f} Fast={fast:.2f} Slow={slow:.2f}"
                )

        if np.isnan(fast) or np.isnan(slow):
            return

        pos = self.get_position(bar.symbol)

        # 金叉买入
        if fast > slow and pos == 0:
            self.buy(bar.symbol, 100)  # 买入 100 股
            self.buy_count += 1
            print(
                f"[{pd.Timestamp(bar.timestamp, unit='ns')}] BUY  100 "
                f"@ {bar.close:.2f} | Count: {self.buy_count}"
            )

        # 死叉卖出
        elif fast < slow and pos > 0:
            self.sell(bar.symbol, pos)
            self.sell_count += 1
            print(
                f"[{pd.Timestamp(bar.timestamp, unit='ns')}] SELL {pos} "
                f"@ {bar.close:.2f} | Count: {self.sell_count}"
            )

    def on_trade(self, trade: aq.Trade) -> None:
        """Call on trade execution."""
        print(f"[Trade] {trade.symbol} {trade.side} {trade.quantity} @ {trade.price}")

    def on_order(self, order: aq.Order) -> None:
        """Call on order status update."""
        print(f"[Order] {order.status} {order.symbol} {order.quantity}")


def get_real_data(symbol: str, start_date: str, end_date: str) -> List[Bar]:
    """获取真实 A 股数据 (akshare)."""
    print(f"Downloading data for {symbol} ({start_date} -> {end_date})...")
    # format date for akshare: YYYYMMDD
    s_date = start_date.replace("-", "")
    e_date = end_date.replace("-", "")

    df = ak.stock_zh_a_daily(symbol=symbol, start_date=s_date, end_date=e_date)
    if df.empty:
        print(f"Warning: No data found for {symbol}")
        return []

    # 打印前几行数据检查
    # print(df.head())

    # 转换为 akquant Bar 列表
    bars = []
    # 打印前 3 行数据以供调试
    print(f"Sample data for {symbol}:")
    print(df.head(3))

    for _, row in df.iterrows():
        # akshare date is already datetime.date or string
        # stock_zh_a_daily returns 'date' column as object (string) or datetime
        ts = pd.Timestamp(row["date"])
        bar = Bar(
            timestamp=int(ts.value),
            symbol=symbol,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        bars.append(bar)
    return bars


def main() -> None:
    """Run warm start demo."""
    checkpoint_file = "checkpoint_demo.pkl"
    symbol = "sz300750"  # 宁德时代 (波动大，容易触发交易)

    # 清理旧的快照文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print("=" * 50)
    print("阶段一：初始回测 (2020-01-01 -> 2021-12-31)")
    print("=" * 50)

    # 获取前半段数据
    data_phase1 = get_real_data(symbol, "2020-01-01", "2021-12-31")

    # 运行回测
    result1 = aq.run_backtest(
        data=data_phase1,
        strategy=MovingAverageStrategy,
        symbols=symbol,
        initial_cash=1_000_000.0,
        commission_rate=0.0003,  # A股佣金
    )

    print("\n阶段一结束:")
    print(f"  - 最终权益: {result1.metrics.end_market_value:.2f}")
    # 类型忽略: 动态属性访问
    print(f"  - 引擎现金: {result1.engine.portfolio.cash:.2f}")  # type: ignore[union-attr]
    print(
        f"  - 策略状态: BuyCount={result1.strategy.buy_count}, "  # type: ignore[union-attr]
        f"SellCount={result1.strategy.sell_count}"  # type: ignore[union-attr]
    )

    # 保存快照
    print(f"\n[System] 保存快照到 {checkpoint_file}...")
    # 类型忽略: run_backtest 返回的 result.engine 类型可能未被mypy完全识别为 Engine
    aq.save_checkpoint(result1.engine, result1.strategy, checkpoint_file)  # type: ignore[arg-type]

    print("\n" + "=" * 50)
    print("阶段二：热启动续跑 (2022-01-01 -> 2023-12-31)")
    print("=" * 50)

    # 获取后半段数据
    # 新版本快照会恢复 get_history() 所依赖的历史缓冲，
    # 正常 warm start 不需要再手工拼接额外的 lookback 数据。
    # lookback_days = 30
    # data_lookback = data_phase1[-lookback_days:]
    data_new = get_real_data(symbol, "2022-01-01", "2023-12-31")
    data_phase2 = data_new  # data_lookback + data_new

    print(f"[System] 准备数据: 新数据 {len(data_new)} 条")

    # 使用 run_from_checkpoint 从检查点恢复
    # 注意：
    # 1. 不需要传入 strategy 类，因为策略实例已从快照恢复
    # 2. 不需要传入 initial_cash，资金状态已恢复
    # 3. 传入拼接后的数据 data_phase2
    result2 = aq.run_from_checkpoint(
        checkpoint_path=checkpoint_file,
        data=data_phase2,
        symbols=symbol,
        commission_rate=0.0003,
    )

    print("\n阶段二结束:")
    print(f"  - 最终权益: {result2.metrics.end_market_value:.2f}")
    # 验证状态是否累加 (BuyCount 应该大于阶段一的值)
    print(
        f"  - 策略状态: BuyCount={result2.strategy.buy_count}, "  # type: ignore[union-attr]
        f"SellCount={result2.strategy.sell_count}"  # type: ignore[union-attr]
    )

    # 验证连续性
    print(f"Debug: Phase 1 End MV = {result1.metrics.end_market_value}")
    print(f"Debug: Phase 2 Init MV = {result2.metrics.initial_market_value}")

    # 允许微小误差
    assert (
        result2.metrics.initial_market_value - result1.metrics.end_market_value
    ) < 1.0
    # 类型忽略: 动态属性访问
    assert result2.strategy.buy_count >= result1.strategy.buy_count  # type: ignore[union-attr]

    print("\n" + "=" * 50)
    print("对比组：全量回测 (2020-01-01 -> 2023-12-31)")
    print("=" * 50)

    # 合并完整数据
    data_full = data_phase1 + data_new  # type: ignore[operator]

    result_full = aq.run_backtest(
        data=data_full,
        strategy=MovingAverageStrategy,
        symbols=symbol,
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
    )

    print("\n全量回测结束:")
    print(f"  - 最终权益: {result_full.metrics.end_market_value:.2f}")
    print(
        f"  - 策略状态: BuyCount={result_full.strategy.buy_count}, "  # type: ignore[union-attr]
        f"SellCount={result_full.strategy.sell_count}"  # type: ignore[union-attr]
    )

    print("\n[Result Verification]")
    print(f"Warm Start Equity: {result2.metrics.end_market_value:.2f}")
    print(f"Full Run Equity  : {result_full.metrics.end_market_value:.2f}")

    # 验证一致性 (允许误差，因为分段回测可能导致指标计算的微小浮点差异)
    diff = abs(result2.metrics.end_market_value - result_full.metrics.end_market_value)
    if diff < 10000:  # 1% 左右的误差容忍
        print(f"[Success] 结果基本一致 (diff={diff:.2f})。Warm Start 功能正常。")
    else:
        print(f"[Warning] 存在较大差异 (diff={diff:.2f})。请检查指标连续性。")

    # 清理
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


if __name__ == "__main__":
    main()
