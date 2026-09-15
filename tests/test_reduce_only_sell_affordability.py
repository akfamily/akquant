"""纯减仓卖单的资金闸门 (issue #400).

A 股卖出费用从成交款净额中扣除, 卖出不需要预先持有现金。这些测试端到端覆盖
**两道独立闸门**: 提交时的 ``CashMarginRule`` 与成交时的模拟撮合器 —— 只修一处
会让拒单原地转移到另一处。
"""

from typing import Any, List

import pandas as pd
from akquant import (
    BacktestConfig,
    Bar,
    NextOpen,
    Strategy,
    StrategyConfig,
    run_backtest,
)
from akquant.config import RiskConfig

SYMBOL = "600000"

# A 股卖出费用: 佣金 0.03% + 印花税 0.1% + 过户费 0.002%, 最低佣金 5 元
A_SHARE_FEES = {
    "commission_rate": 0.0003,
    "stamp_tax_rate": 0.001,
    "transfer_fee_rate": 0.00002,
    "min_commission": 5.0,
}


def _build_bars(prices: List[float]) -> List[Bar]:
    return [
        Bar(
            timestamp=pd.Timestamp(
                f"2023-01-{day:02d} 15:00:00", tz="Asia/Shanghai"
            ).value,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1e8,
            symbol=SYMBOL,
        )
        for day, price in enumerate(prices, start=3)
    ]


class _BuyThenSell(Strategy):
    """第 0 根 bar 买入, 第 2 根起反复尝试全量卖出.

    第 1 根留给 NextOpen 成交, 第 2 根起持仓已过 T+1 可卖。重复尝试是为了
    抓住"账户冻结"——一旦卖不掉, 后续每根 bar 都会继续被拒。
    """

    buy_quantity: float = 0.0
    sell_quantity: float = 0.0

    def on_bar(self, bar: Bar) -> None:
        index = getattr(self, "_bar_index", 0)
        self._bar_index = index + 1
        if index == 0:
            self.buy(bar.symbol, self.buy_quantity)
        elif index >= 2:
            self.sell(bar.symbol, self.sell_quantity)


def _run(
    buy_quantity: float,
    sell_quantity: float,
    initial_cash: float,
    prices: List[float],
    fees: dict,
    risk_config: RiskConfig,
) -> Any:
    class _Sized(_BuyThenSell):
        pass

    _Sized.buy_quantity = buy_quantity
    _Sized.sell_quantity = sell_quantity

    return run_backtest(
        data=_build_bars(prices),
        strategy=_Sized,
        symbols=SYMBOL,
        config=BacktestConfig(
            strategy_config=StrategyConfig(
                initial_cash=initial_cash,
                risk=risk_config,
                # 末根 bar 自动平仓会凭空多出一张卖单, 干扰"卖单是否被拒"的断言
                exit_on_last_bar=False,
                **fees,
            ),
            show_progress=False,
        ),
        t_plus_one=True,
        fill_policy=NextOpen(),
        lot_size=100,
    )


def _orders(result: Any) -> pd.DataFrame:
    orders: pd.DataFrame = result.orders_df
    return orders


def _sells(result: Any) -> pd.DataFrame:
    orders = _orders(result)
    sells: pd.DataFrame = orders[orders["side"] == "sell"]
    return sells


def test_full_position_sell_fills_when_remaining_cash_is_below_fees() -> None:
    """满仓后现金不足以覆盖卖出费用时, 全量卖单仍须成交 (issue #400 主场景).

    10 元买入 10,000 股耗尽 100,100 元账户, 余现金 74 元; 全量卖出费用合计
    132 元 > 余现金, 但卖出所得 100,000 元完全足以覆盖。
    """
    result = _run(
        buy_quantity=10000,
        sell_quantity=10000,
        initial_cash=100100.0,
        prices=[10.0, 10.0, 10.0, 10.0],
        fees=A_SHARE_FEES,
        risk_config=RiskConfig(check_cash=True),
    )

    sells = _sells(result)
    assert not sells.empty, _orders(result)
    assert (sells["status"] == "filled").any(), sells[["status", "reject_reason"]]
    # 不止"没被拒": 整笔 10,000 股必须真的成交出去。
    assert float(sells["filled_quantity"].sum()) == 10000.0, sells[
        ["status", "filled_quantity", "reject_reason"]
    ]


def test_reduce_only_sell_fills_when_free_margin_is_negative() -> None:
    """可用资金为负时, 纯减仓卖单仍须成交.

    两融账户 2 倍杠杆买入后股价下跌, 可用资金转负。此时减仓是唯一的自救手段,
    且它**释放**保证金而非占用。注意所有费率取库默认值 0 —— 需求为 0 的订单
    也会撞上 ``required > free_margin``, 与手续费无关。
    """
    result = _run(
        buy_quantity=20000,
        sell_quantity=20000,
        initial_cash=100100.0,
        prices=[10.0, 10.0, 8.0, 8.0, 8.0],
        fees={},
        risk_config=RiskConfig(
            check_cash=True,
            account_mode="margin",
            initial_margin_ratio=0.5,
            allow_force_liquidation=False,
        ),
    )

    sells = _sells(result)
    assert not sells.empty, _orders(result)
    assert (sells["status"] == "filled").any(), sells[["status", "reject_reason"]]
    assert float(sells["filled_quantity"].sum()) == 20000.0, sells[
        ["status", "filled_quantity", "reject_reason"]
    ]


def test_sell_still_rejected_when_fees_exceed_proceeds() -> None:
    """卖出所得不足以覆盖费用时, 卖单仍须被拒.

    豁免的语义是"费用从成交款净额中扣除", 而不是"平仓一律放过": 所得盖不住
    费用时这单确实需要资金, 放过它等于让现金透支 —— 只有 ``check_cash=False``
    才允许透支 (#280)。
    """
    # 持 100 股 0.01 元的股票: 卖出所得 1 元, 费用 >5 元 (最低佣金主导)。
    result = _run(
        buy_quantity=100,
        sell_quantity=100,
        initial_cash=2.0,
        prices=[0.01, 0.01, 0.01, 0.01],
        fees=A_SHARE_FEES,
        risk_config=RiskConfig(check_cash=True),
    )

    sells = _sells(result)
    assert not sells.empty, _orders(result)
    assert (sells["status"] == "rejected").all(), sells[["status", "reject_reason"]]


def test_buy_still_rejected_when_cash_insufficient() -> None:
    """买入侧的验资不得被放宽 —— 买入占用保证金, 必须现金预付."""
    result = _run(
        buy_quantity=20000,  # 需 200,000 元, 账户只有 100,100 元
        sell_quantity=0,
        initial_cash=100100.0,
        prices=[10.0, 10.0, 10.0, 10.0],
        fees=A_SHARE_FEES,
        risk_config=RiskConfig(check_cash=True),
    )

    orders = _orders(result)
    buys = orders[orders["side"] == "buy"]
    assert not buys.empty, orders
    assert (buys["status"] == "rejected").all(), buys[["status", "reject_reason"]]


def test_sell_exceeding_position_never_goes_short_in_cash_account() -> None:
    """现金账户卖出超过持仓时不得裸空 —— 减仓豁免不能顺带放过反向开仓部分."""
    result = _run(
        buy_quantity=10000,
        sell_quantity=20000,  # 超出持仓一倍
        initial_cash=100100.0,
        prices=[10.0, 10.0, 10.0, 10.0],
        fees=A_SHARE_FEES,
        risk_config=RiskConfig(check_cash=True),
    )

    positions = result.positions_df
    if not positions.empty:
        assert float(positions["short_shares"].max()) == 0.0, "现金账户出现空头持仓"
        assert float(positions["long_shares"].min()) >= 0.0, "现金账户出现负持仓"
