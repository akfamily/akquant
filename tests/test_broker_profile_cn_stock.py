"""A 股 broker_profile 模板: 现行印花税率与 T+1."""

from typing import Any

import pandas as pd
import pytest
from akquant import Bar, CurrentClose, Strategy, run_backtest

SYMBOL = "PROFILE_T1"
PROFILES = ["cn_stock_miniqmt", "cn_stock_t1_low_fee", "cn_stock_sim_high_slippage"]


def _bars() -> list[Bar]:
    """两个交易日, 每天 10:00 与 14:00 各一根, 价格恒为 10."""
    days = pd.bdate_range("2025-01-02", periods=2, tz="Asia/Shanghai")
    return [
        Bar(
            timestamp=(day + pd.Timedelta(hours=hour)).value,
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
            volume=1e7,
            symbol=SYMBOL,
        )
        for day in days
        for hour in (10, 14)
    ]


class _BuyThenSellSameDay(Strategy):
    """第一根买 1000 股, 第二根(同日)卖出; T+1 下卖单应被拒."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def on_bar(self, bar: Bar) -> None:
        self.n += 1
        if self.n == 1:
            self.buy(bar.symbol, 1000)
        elif self.n == 2:
            self.sell(bar.symbol, 1000)


def _run(**kwargs: Any) -> Any:
    days = pd.bdate_range("2025-01-02", periods=2)
    return run_backtest(
        data=_bars(),
        strategy=_BuyThenSellSameDay,
        symbols=SYMBOL,
        fill_policy=CurrentClose(),
        trading_calendar=[d.date() for d in days],
        show_progress=False,
        **kwargs,
    )


@pytest.mark.parametrize("profile", PROFILES)
def test_cn_stock_profiles_enable_t_plus_one(profile: str) -> None:
    """三个 A 股模板都开 T+1: 当日买入当日不可卖."""
    result = _run(broker_profile=profile)
    statuses = result.orders_df["status"].tolist()
    assert statuses == ["filled", "rejected"]
    assert (
        "Insufficient available position" in result.orders_df.iloc[1]["reject_reason"]
    )


def test_explicit_t_plus_one_false_overrides_profile() -> None:
    """显式 t_plus_one=False 优先于模板."""
    result = _run(broker_profile="cn_stock_t1_low_fee", t_plus_one=False)
    assert result.orders_df["status"].tolist() == ["filled", "filled"]


def test_no_profile_keeps_t_plus_zero() -> None:
    """不传模板也不传 t_plus_one: 仍为 T+0(默认值不变)."""
    result = _run()
    assert result.orders_df["status"].tolist() == ["filled", "filled"]


class _SellNextDay(_BuyThenSellSameDay):
    """第一根买, 第三根(次日)卖."""

    def on_bar(self, bar: Bar) -> None:
        self.n += 1
        if self.n == 1:
            self.buy(bar.symbol, 1000)
        elif self.n == 3:
            self.sell(bar.symbol, 1000)


@pytest.mark.parametrize("profile", PROFILES)
def test_cn_stock_profiles_use_current_stamp_tax(profile: str) -> None:
    """卖出 1 万元的印花税按现行 0.05% 收 5 元(佣金与过户费另计)."""
    days = pd.bdate_range("2025-01-02", periods=2)
    result = run_backtest(
        data=_bars(),
        strategy=_SellNextDay,
        symbols=SYMBOL,
        fill_policy=CurrentClose(),
        trading_calendar=[d.date() for d in days],
        broker_profile=profile,
        slippage={"type": "zero"},
        show_progress=False,
    )
    executions = result.executions_df
    assert executions["side"].tolist() == ["buy", "sell"]
    buy_fee, sell_fee = executions["commission"].astype(float).tolist()
    # 买卖两边的佣金与过户费相同, 卖出多出的部分就是印花税
    assert sell_fee - buy_fee == pytest.approx(10_000.0 * 0.0005)


def test_metrics_attribute_matches_metrics_df() -> None:
    """result.metrics.<名称> 能取到 metrics_df 里的交易类字段, 取值一致."""
    result = _run(broker_profile="cn_stock_t1_low_fee", t_plus_one=False)
    frame = result.metrics_df
    assert result.metrics.total_commission == pytest.approx(
        float(frame.at["total_commission", "value"])
    )
    assert result.metrics.total_commission > 0
    assert float(result.metrics.closed_trade_count) == float(
        frame.at["closed_trade_count", "value"]
    )
    # 原有绩效字段照旧, 时间字段仍转为带时区的时间
    assert result.metrics.total_return_pct == result._raw.metrics.total_return_pct
    assert result.metrics.start_time.tzinfo is not None


def test_metrics_unknown_attribute_points_to_metrics_df() -> None:
    """不存在的字段仍抛 AttributeError, 提示去 metrics_df 查字段."""
    result = _run()
    with pytest.raises(AttributeError, match="metrics_df"):
        _ = result.metrics.no_such_metric
