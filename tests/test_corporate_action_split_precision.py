"""送转、拆股后持仓是精确数, 能一次清仓(issue #460、#461).

``CorporateAction`` 的比例是 Python 浮点数。此前转成 Decimal 时保留了浮点数的二进制
尾数(1.44 变成 1.43999999999999994...), 拆股后的持仓比应有的整数少一点点: 策略读到
20880, 引擎里是 20879.999999999999...; 按 20880 卖出时因"可用持仓不足"被拒, 仓位
永远平不掉。
"""

from __future__ import annotations

from typing import Any, List

import akquant as aq
import pandas as pd
import pytest

SYMBOL = "000001.SZ"


class _RegisterActions:
    """回测开始时把公司行动登记进引擎的分析器插件."""

    name = "register_corporate_actions"

    def __init__(self, actions: List[Any]) -> None:
        self.actions = actions

    def on_start(self, ctx: Any) -> None:
        for action in self.actions:
            ctx["engine"].add_corporate_action(action)

    def on_bar(self, ctx: Any) -> None:
        pass

    def on_trade(self, ctx: Any) -> None:
        pass

    def on_finish(self, ctx: Any) -> dict[str, Any]:
        return {}


def _bars() -> pd.DataFrame:
    prices = [10.0, 10.0, 7.0, 7.0]
    index = pd.to_datetime(
        [f"2024-01-0{day} 15:00:00" for day in (2, 3, 4, 5)]
    ).tz_localize("Asia/Shanghai")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1_000_000.0] * 4,
        },
        index=index,
    )


def _split(ratio: float) -> Any:
    return aq.CorporateAction(
        symbol=SYMBOL,
        date=pd.Timestamp("2024-01-04").date(),
        action_type=aq.akquant.CorporateActionType.Split,
        value=ratio,
    )


def _held_on_last_day(result: Any) -> float:
    positions = result.positions_df
    last_day = positions[pd.to_datetime(positions["date"]).dt.day == 5]
    return float(last_day["long_shares"].sum())


def _run(strategy: Any, ratio: float, **kwargs: Any) -> Any:
    return aq.run_backtest(
        data={SYMBOL: _bars()},
        strategy=strategy,
        symbols=SYMBOL,
        freq="1d",
        fill_policy=aq.NextOpen(),
        t_plus_one=True,
        analyzer_plugins=[_RegisterActions([_split(ratio)])],
        show_progress=False,
        **kwargs,
    )


@pytest.mark.parametrize(("ratio", "bought", "after"), [(1.44, 14500, 20880)])
def test_close_position_fills_after_split(
    ratio: float, bought: int, after: int
) -> None:
    """#460: 拆股后 close_position() 全部成交, 仓位清零."""

    class Strategy(aq.Strategy):
        def on_bar(self, bar: Any) -> None:
            day = bar.timestamp_iso[:10]
            if day == "2024-01-02":
                self.buy(SYMBOL, bought)
            elif day == "2024-01-04":
                self.close_position(SYMBOL)

    result = _run(Strategy, ratio, initial_cash=500000.0)
    orders = result.orders_df
    sell = orders[orders["side"] == "sell"].iloc[0]
    assert (sell["status"], sell["filled_quantity"]) == ("filled", float(after))
    # 清仓单在最后一根 bar 开盘成交; 持仓表只记有持仓的日子, 最后一天没有记录
    assert _held_on_last_day(result) == 0.0


def test_liquidation_sells_the_odd_lot_left_by_a_split() -> None:
    """#461: 送转出零股(1000 股 × 1.45 = 1450)后, 清仓把不足一手的 50 股一并卖出."""

    class Strategy(aq.Strategy):
        def on_bar(self, bar: Any) -> None:
            day = bar.timestamp_iso[:10]
            if day == "2024-01-02":
                self.buy(SYMBOL, 1000)
            elif day == "2024-01-04":
                self.rebalance_weights({}, liquidate_unmentioned=True)

    config = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=500000.0),
        instruments_config=[
            aq.InstrumentConfig(
                symbol=SYMBOL, asset_type="STOCK", lot_size=100, tick_size=0.01
            )
        ],
    )
    result = _run(Strategy, 1.45, config=config)
    orders = result.orders_df
    sell = orders[orders["side"] == "sell"].iloc[0]
    assert (sell["status"], sell["filled_quantity"]) == ("filled", 1450.0)
    # 清仓单在最后一根 bar 开盘成交; 持仓表只记有持仓的日子, 最后一天没有记录
    assert _held_on_last_day(result) == 0.0
