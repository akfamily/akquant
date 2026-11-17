# -*- coding: utf-8 -*-
"""walk-forward 样本外收益不计入预热段的盈亏.

样本外回测从“样本外起点前 warmup 根”开始跑, 预热期内不调 ``on_bar``, 但其它回调
(如 ``on_after_trading``)照常触发。策略在这些回调里下单时, 预热段就有成交。修复前
每段样本外的收益都以 ``initial_cash`` 为基准, 预热段的盈亏被算进了样本外。
"""

from __future__ import annotations

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
import pytest
from akquant import Bar, IntParam, Strategy

SYMBOL = "X"
INITIAL_CASH = 100_000.0
TRAIN, TEST, WARMUP = 60, 20, 10


def _frame() -> pd.DataFrame:
    """120 根日线: 只在第一段样本外的预热段(第 50～59 根)从 100 跌到 90, 其余走平."""
    dates = pd.bdate_range("2021-01-04", periods=120)
    price = np.full(len(dates), 100.0)
    price[50:60] = np.linspace(99.0, 90.0, 10)
    price[60:] = 90.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1e6,
            "symbol": SYMBOL,
        }
    ).set_index("date")


class _BuyAfterClose(Strategy):
    """不在 on_bar 下单: 每天收盘后的回调里, 空仓就买 100 股(预热期内也会触发)."""

    dummy = IntParam(1)

    def on_bar(self, bar: Bar) -> None:
        pass

    def on_after_trading(self, trading_date: Any, timestamp: Any) -> None:
        if self.get_position(SYMBOL) == 0:
            self.buy(SYMBOL, 100)


class _BuyOnBar(Strategy):
    """只在 on_bar 下单(预热期内不触发)."""

    dummy = IntParam(1)

    def on_bar(self, bar: Bar) -> None:
        if self.get_position(bar.symbol) == 0:
            self.buy(bar.symbol, 100)


def _walk_forward(strategy: type[Strategy], **kwargs: Any) -> pd.DataFrame:
    return aq.run_walk_forward(
        strategy=strategy,
        param_grid={"dummy": [1]},
        data=_frame(),
        train_period=TRAIN,
        test_period=TEST,
        warmup_period=WARMUP,
        initial_cash=INITIAL_CASH,
        max_workers=1,
        show_progress=False,
        symbols=[SYMBOL],
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        **kwargs,
    )


@pytest.mark.parametrize("compounding", [False, True], ids=["additive", "compounding"])
def test_warmup_losses_are_not_counted_out_of_sample(compounding: bool) -> None:
    """预热段里买入并亏掉的钱不算样本外的: 样本外价格走平, 拼接净值应一直持平."""
    result = _walk_forward(_BuyAfterClose, compounding=compounding)
    assert len(result) == 3 * TEST
    assert result["equity"].tolist() == pytest.approx([INITIAL_CASH] * len(result))


def test_on_bar_strategy_result_is_unchanged() -> None:
    """只在 on_bar 下单的策略预热段没有成交, 基准仍是初始资金, 结果与修复前相同."""
    result = _walk_forward(_BuyOnBar)
    assert len(result) == 3 * TEST
    assert result["equity"].tolist() == pytest.approx([INITIAL_CASH] * len(result))
