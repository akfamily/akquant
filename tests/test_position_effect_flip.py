# -*- coding: utf-8 -*-
"""Regression for issue #361.

同一根 bar 内"先平后开"的反手（``close_position()`` 紧跟 ``buy()`` / ``sell()``）
必须拆成 Close + Open 两腿。修复前 auto 拆腿读的是**结算仓**，而结算仓不含同
周期已提交未成交的平仓单，导致第二腿被误判成平仓：反手两笔都标 Close，且该腿
在风控投影里 delta 为 0，等于零保证金过闸。

同源问题还在 ``_target_to_orders``：它同样按结算仓算 delta，先 ``close_position``
再 ``order_target_*`` 会在全平单之外再补一笔同向单造成超卖，故一并锁住。

多策略下还有一层：``ctx.positions`` 是账户级全局，故 ``ctx.active_orders`` 也必须
含**同周期内前面 slot 刚提交**的单，否则 slot 1 看不到 slot 0 的平仓意图，同样会
把开仓腿误判成平仓。
"""

import akquant as aq
import numpy as np
import pandas as pd
from akquant import (
    BacktestConfig,
    Bar,
    ChinaFuturesConfig,
    InstrumentConfig,
    Strategy,
    StrategyConfig,
)
from akquant.backtest import BacktestResult

SYMBOL = "RB2310"
PRICES = np.array([3500.0, 3510.0, 3520.0, 3530.0, 3540.0, 3550.0])


def _futures_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=len(PRICES), freq="D"),
            "open": PRICES,
            "high": PRICES + 5.0,
            "low": PRICES - 5.0,
            "close": PRICES,
            "volume": 1_000.0,
            "symbol": SYMBOL,
        }
    )


def _run(strategy: type[Strategy], initial_cash: float = 500_000.0) -> BacktestResult:
    config = BacktestConfig(
        strategy_config=StrategyConfig(initial_cash=initial_cash, commission_rate=0.0),
        instruments_config=[
            InstrumentConfig(
                symbol=SYMBOL,
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
            )
        ],
        china_futures=ChinaFuturesConfig(enforce_sessions=False),
    )
    return aq.run_backtest(
        strategy=strategy,
        data=_futures_data(),
        config=config,
        fill_policy=aq.CurrentClose(),
    )


def _effects(result: BacktestResult) -> list[str]:
    """成交流水的 ``side-position_effect`` 序列."""
    return [
        f"{str(t.side).rsplit('.', 1)[-1]}-{str(t.position_effect).rsplit('.', 1)[-1]}"
        for t in result._raw.executions
    ]


class _ShortToLong(Strategy):
    """bar0 开空 1 手；bar2 ``close_position()`` + ``buy(1)`` 反手做多."""

    def __init__(self) -> None:
        super().__init__()
        self.bar_index = 0

    def on_bar(self, bar: Bar) -> None:
        if self.bar_index == 0:
            self.short(bar.symbol, 1)
        elif self.bar_index == 2:
            self.close_position(bar.symbol)
            self.buy(bar.symbol, 1)
        self.bar_index += 1


class _LongToShort(Strategy):
    """bar0 开多 1 手；bar2 ``close_position()`` + ``sell(1)`` 反手做空."""

    def __init__(self) -> None:
        super().__init__()
        self.bar_index = 0

    def on_bar(self, bar: Bar) -> None:
        if self.bar_index == 0:
            self.buy(bar.symbol, 1)
        elif self.bar_index == 2:
            self.close_position(bar.symbol)
            self.sell(bar.symbol, 1)
        self.bar_index += 1


def test_short_to_long_flip_splits_close_then_open() -> None:
    """空头反手做多：应为 Buy-Close + Buy-Open，而非两笔 Buy-Close（#361）."""
    result = _run(_ShortToLong)

    assert _effects(result) == ["Sell-Open", "Buy-Close", "Buy-Open"]


def test_long_to_short_flip_splits_close_then_open() -> None:
    """多头反手做空：与做多方向对称，应为 Sell-Close + Sell-Open."""
    result = _run(_LongToShort)

    assert _effects(result) == ["Buy-Open", "Sell-Close", "Sell-Open"]


def test_flip_reaches_opposite_position() -> None:
    """标签修正不得改变最终净仓：-1 手反手后应为 +1 手."""
    result = _run(_ShortToLong)
    positions = result.positions_df

    assert float(positions["long_shares"].iloc[-1]) == 1.0
    assert float(positions["short_shares"].iloc[-1]) == 0.0


class _FlipOversized(Strategy):
    """空 1 手后 ``close_position()`` + ``buy(10)``：开仓腿须整体过保证金闸."""

    def __init__(self) -> None:
        super().__init__()
        self.bar_index = 0

    def on_bar(self, bar: Bar) -> None:
        if self.bar_index == 0:
            self.short(bar.symbol, 1)
        elif self.bar_index == 2:
            self.close_position(bar.symbol)
            self.buy(bar.symbol, 10)
        self.bar_index += 1


def test_flip_open_leg_is_margin_checked() -> None:
    """反手的开仓腿不得借"平仓"标签零保证金过闸（#361）.

    一手保证金 = 3520 × 10 × 0.1 = 3520；资金 34_000 够 9 手不够 10 手。修复前
    ``buy(10)`` 会拆出一条假 Close 腿（投影 delta 为 0 → 免保证金）并成交，
    只有 open-9 腿被拒。修复后整个 10 手作为开仓一体过闸，应被整单拒绝。
    """
    result = _run(_FlipOversized, initial_cash=34_000.0)
    orders = result.orders_df

    buys = orders[orders["side"] == "buy"]
    # close_position 的平空单 + buy(10) 的开仓单，共两笔买单
    assert len(buys) == 2
    closing_buy = buys.iloc[0]
    opening_buy = buys.iloc[1]

    assert float(closing_buy["quantity"]) == 1.0
    assert closing_buy["status"] == "filled"

    assert float(opening_buy["quantity"]) == 10.0
    assert opening_buy["status"] == "rejected"
    # 未再出现"假平仓腿零保证金成交"：成交总量不含被拒的开仓量
    assert float(orders[orders["status"] == "filled"]["quantity"].sum()) == 2.0


class _CloseThenTarget(Strategy):
    """bar0 建仓；bar2 ``close_position()`` 后紧跟 ``order_target()`` 同量目标.

    目标仓位按投影持仓算 delta：全平单已把投影仓打到 0，故重建到原目标应下
    一笔买单，而不是在全平单之外再补一笔卖单（后者会卖超持仓）。
    """

    def __init__(self) -> None:
        super().__init__()
        self.bar_index = 0

    def on_bar(self, bar: Bar) -> None:
        if self.bar_index == 0:
            self.buy(bar.symbol, 10)
        elif self.bar_index == 2:
            self.close_position(bar.symbol)
            self.order_target(bar.symbol, 10)
        self.bar_index += 1


def test_close_then_target_does_not_oversell() -> None:
    """``close_position()`` + ``order_target()`` 不得卖超持仓（与 #361 同源）."""
    result = _run(_CloseThenTarget)
    orders = result.orders_df

    sells = orders[orders["side"] == "sell"]
    # 只应有 close_position 的一笔全平单；不得再多一笔同向卖单
    assert len(sells) == 1
    assert float(sells.iloc[0]["quantity"]) == 10.0

    total_sold = float(sells["quantity"].sum())
    assert total_sold <= 10.0, f"卖出 {total_sold} 超过持仓 10"


class _SlotCloser(Strategy):
    """slot 0：bar0 开空 1 手，bar2 平掉（同周期提交 Buy-Close）."""

    def __init__(self) -> None:
        super().__init__()
        self.bar_index = 0

    def on_bar(self, bar: Bar) -> None:
        if self.bar_index == 0:
            self.short(bar.symbol, 1)
        elif self.bar_index == 2:
            self.close_position(bar.symbol)
        self.bar_index += 1


class _SlotBuyer(Strategy):
    """slot 1：bar2 买 1 手，须看到 slot 0 同周期的平仓单."""

    def __init__(self) -> None:
        super().__init__()
        self.bar_index = 0
        self.seen_closable: float | None = None

    def on_bar(self, bar: Bar) -> None:
        if self.bar_index == 2:
            self.seen_closable = float(self.execution.get_closable_position(bar.symbol))
            self.buy(bar.symbol, 1)
        self.bar_index += 1


def test_pending_close_is_visible_across_strategy_slots() -> None:
    """多策略同周期：slot 0 的在途平仓单必须对 slot 1 可见.

    账户仓位是全局的（两个 slot 都看到 -1），故 slot 1 的买单在 slot 0 已提交
    全平单之后应判 Open。修复前 ``active_orders`` 在 slot 循环前只快照一次，
    slot 1 看不到 slot 0 本周期的单，会把这笔误判成 Close。
    """
    config = BacktestConfig(
        strategy_config=StrategyConfig(initial_cash=500_000.0, commission_rate=0.0),
        instruments_config=[
            InstrumentConfig(
                symbol=SYMBOL,
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
            )
        ],
        china_futures=ChinaFuturesConfig(enforce_sessions=False),
    )
    result = aq.run_backtest(
        strategy=_SlotCloser,
        data=_futures_data(),
        config=config,
        strategies_by_slot={"beta": _SlotBuyer},
        fill_policy=aq.CurrentClose(),
    )

    effects = _effects(result)
    # slot 0 开空 -> slot 0 平空 -> slot 1 开多（而非第二笔 Buy-Close）
    assert effects == ["Sell-Open", "Buy-Close", "Buy-Open"]


def test_dataframes_expose_position_effect_columns() -> None:
    """``orders_df`` / ``executions_df`` 须暴露开平语义与 ISO 时间列.

    #361 争的就是 ``position_effect``，而这两张最常用的表原先都不导出它，
    使用者只能自行遍历 ``Trade`` / ``Order`` 对象才看得到。
    """
    result = _run(_ShortToLong)

    executions = result.executions_df
    assert "position_effect" in executions.columns
    assert "timestamp_iso" in executions.columns
    assert executions["position_effect"].tolist() == ["open", "close", "open"]
    # side 不带 `OrderSide.` 前缀（Rust 快路径与 Python 兜底须同口径）
    assert executions["side"].tolist() == ["sell", "buy", "buy"]
    assert all(s.endswith("Z") for s in executions["timestamp_iso"])

    orders = result.orders_df
    for column in (
        "position_effect",
        "reduce_only",
        "created_at_iso",
        "updated_at_iso",
    ):
        assert column in orders.columns
    assert orders["position_effect"].tolist() == ["open", "close", "open"]
    assert orders["reduce_only"].tolist() == [False, False, False]


class _ExplicitCloseToday(Strategy):
    """显式下 ``close_today`` 平今单，用于锁住导出词表."""

    def __init__(self) -> None:
        super().__init__()
        self.bar_index = 0

    def on_bar(self, bar: Bar) -> None:
        if self.bar_index == 0:
            self.short(bar.symbol, 1)
        elif self.bar_index == 2:
            self.submit_order(
                symbol=bar.symbol,
                side="Buy",
                quantity=1,
                position_effect="close_today",
            )
        self.bar_index += 1


def test_position_effect_uses_canonical_vocabulary() -> None:
    """导出词表须与下单入参一致：``close_today``，不是 ``closetoday``.

    多词枚举若按 ``format!("{:?}").to_lowercase()`` 导出会变成 ``closetoday``，
    而 API 接受的是 ``close_today``，使用者按入参词表筛 DataFrame 会静默匹配
    不到——与 #361 同类的静默错答。
    """
    result = _run(_ExplicitCloseToday)

    executions = result.executions_df
    assert executions["position_effect"].tolist() == ["open", "close_today"]
    assert result.orders_df["position_effect"].tolist() == ["open", "close_today"]

    # 按入参词表回筛必须命中
    matched = executions[executions["position_effect"] == "close_today"]
    assert len(matched) == 1
