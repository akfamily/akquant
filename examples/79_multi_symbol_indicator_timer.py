#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多标的指标在定时器与横截面回调里按标的取值：``self.ind.of(symbol)``.

一个 ETF 轮动策略，给三个标的各算两个指标：

- ``mom``：20 日动量，用 ``PrecomputedIndicator`` 对整段数据向量化计算
  （第三方 pandas-ta、自写的 numpy 函数都可以这样接入）。预计算指标自己按标的
  缓存结果，直接传实例即可，不需要 ``factory=``；
- ``ma``：20 日均线，增量指标，多标的用 ``factory=`` 给每个标的各造一个实例。

每周一 14:50 的定时器里用 ``.of(symbol)[0]`` 读各标的的值（``.of(symbol)[5]`` 是
5 根之前的值，回溯照常可用），持有动量为正、均线向上的标的里动量最高的那一只。

``on_bar`` / ``on_tick`` / 窗口回调里 ``self.ma[0]`` 取的是该回调的标的；在
``on_timer``、``on_cross_section`` 这类不属于某个标的的回调里，多标的指标必须用
``.of(symbol)`` 指定标的，不指定会直接报错（而不是静默取到别的标的的值）。
14:50 的定时器早于当日 15:00 收盘的日线，读到的是上一交易日收盘后的值，没有未来函数。

数据是本地合成的，不需要联网。

运行方式::

    python examples/79_multi_symbol_indicator_timer.py

输出：每次调仓的持仓变化与最终收益。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from akquant import SMA, PrecomputedIndicator, Strategy, run_backtest

SYMBOLS = {"510300": "沪深300ETF", "510500": "中证500ETF", "159915": "创业板ETF"}


def make_data(days: int = 250) -> Dict[str, pd.DataFrame]:
    """三条走势不同的合成日线（收盘时刻 15:00，上海时区）."""
    index = pd.date_range(
        "2024-01-02 15:00", periods=days, freq="B", tz="Asia/Shanghai"
    )
    data = {}
    for k, symbol in enumerate(SYMBOLS):
        t = np.arange(days, dtype=float)
        # 三个标的的强弱交替出现，轮动才有意义
        close = 3.0 + 0.4 * np.sin(t / 25.0 + k * 2.1) + 0.002 * t * (k - 1)
        data[symbol] = pd.DataFrame(
            {
                "date": index,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1e8,
                "symbol": symbol,
            }
        )
    return data


def momentum_20(df: pd.DataFrame) -> pd.Series:
    """20 日动量（收益率），整段一次算完."""
    return df["close"] / df["close"].shift(20) - 1.0


class EtfRotation(Strategy):
    """每周一按动量轮动，只持有一只."""

    def on_start(self) -> None:
        """声明指标与定时器."""
        self.mom = self.I(PrecomputedIndicator("mom20", momentum_20), name="mom")
        self.ma = self.I(factory=lambda: SMA(20), name="ma")
        self.schedule_weekly("14:50:00", "rebalance")
        self.holding: Optional[str] = None
        self.pending_buy: Optional[str] = None
        self.switches = 0

    def on_timer(self, payload: str) -> None:
        """按标的读指标、排序、调仓."""
        scores: Dict[str, float] = {}
        for symbol in SYMBOLS:
            mom = self.mom.of(symbol)[0]
            ma_now = self.ma.of(symbol)[0]
            ma_before = self.ma.of(symbol)[5]
            if mom is None or ma_now is None or ma_before is None or math.isnan(mom):
                continue  # 预热期还没有值
            if mom > 0 and ma_now > ma_before:
                scores[symbol] = mom
        target = max(scores, key=lambda s: scores[s]) if scores else None
        if target == self.holding:
            return
        if self.holding is not None:
            # 先卖后买: 两笔同时提交时买单可能先撮合, 卖出的钱还没回来就被拒
            self.order_target_percent(self.holding, 0.0)
            self.pending_buy = target
        elif target is not None:
            self.order_target_percent(target, 0.95)
        print(
            f"{self.now:%Y-%m-%d} 调仓: {self.holding} -> {target}  "
            f"scores={_fmt(scores)}"
        )
        self.holding = target
        self.switches += 1

    def on_trade(self, trade: Any) -> None:
        """卖单成交、资金回笼后再买新标的."""
        if self.pending_buy is not None and str(trade.side).endswith("Sell"):
            target, self.pending_buy = self.pending_buy, None
            self.order_target_percent(target, 0.95)


def _fmt(scores: Dict[str, float]) -> Dict[str, Any]:
    return {s: round(v, 4) for s, v in scores.items()}


if __name__ == "__main__":
    strategy = EtfRotation()
    result = run_backtest(
        data=make_data(),
        strategy=strategy,
        initial_cash=1_000_000,
        commission_rate=0.0001,
        min_commission=0.0,
        lot_size=100,
        show_progress=False,
    )
    rejected = [o for o in result.orders if "Rejected" in str(o.status)]
    print(
        f"调仓次数: {strategy.switches}，成交 {len(result.executions)} 笔，"
        f"拒单 {len(rejected)} 笔"
    )
    print(f"总收益率: {result.metrics.total_return_pct:.2f}%")
