from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def make_bars() -> list[Bar]:
    """Build deterministic bars for multi-strategy demo."""
    idx = pd.date_range(start="2024-02-01", periods=8, freq="D")
    prices = [100.0, 99.0, 98.0, 97.0, 96.0, 97.0, 98.0, 99.0]
    bars: list[Bar] = []
    for ts, price in zip(idx, prices):
        bars.append(
            Bar(
                timestamp=int(ts.value),
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000.0,
                symbol="MULTI",
            )
        )
    return bars


class AlphaStrategy(Strategy):
    """Primary strategy used in the multi-strategy demo."""

    def on_bar(self, bar: Bar) -> None:
        """Submit deterministic buy orders."""
        self.buy(symbol=bar.symbol, quantity=10)


class BetaStrategy(Strategy):
    """Secondary strategy used in the multi-strategy demo."""

    def on_bar(self, bar: Bar) -> None:
        """Submit small buy or close orders."""
        position = self.get_position(bar.symbol)
        if position == 0:
            self.buy(symbol=bar.symbol, quantity=2)
        else:
            self.close_position(bar.symbol)


def run_single_strategy() -> Any:
    """Run single-strategy ownership baseline."""
    config = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(
            strategy_id="alpha",
            initial_cash=100000.0,
            commission_rate=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            min_commission=0.0,
        )
    )
    return aq.run_backtest(
        data=make_bars(),
        strategy=AlphaStrategy,
        symbols="MULTI",
        config=config,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        show_progress=False,
    )


def run_multi_strategy() -> Any:
    """Run multi-slot strategy with risk actions."""
    config = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(
            strategy_id="alpha",
            strategies_by_slot={"beta": BetaStrategy},
            strategy_max_order_size={"alpha": 5.0, "beta": 20.0},
            strategy_reduce_only_after_risk={"alpha": True, "beta": False},
            strategy_risk_cooldown_bars={"alpha": 2, "beta": 0},
            initial_cash=100000.0,
            commission_rate=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            min_commission=0.0,
        )
    )
    return aq.run_backtest(
        data=make_bars(),
        strategy=AlphaStrategy,
        symbols="MULTI",
        config=config,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        show_progress=False,
    )


def main() -> None:
    """Print single-slot and multi-slot summary outputs."""
    probe = aq.Engine()
    if not hasattr(probe, "set_strategy_risk_cooldown_bars"):
        print("multi_strategy_demo_skipped=engine_missing_strategy_cooldown")
        print("done_multi_strategy_demo")
        return

    single_result = run_single_strategy()
    single_orders = single_result.orders_df
    single_owner_ids = sorted(
        {
            str(value)
            for value in single_orders.get("owner_strategy_id", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .tolist()
            if str(value)
        }
    )

    multi_result = run_multi_strategy()
    multi_orders = multi_result.orders_df
    multi_owner_ids = sorted(
        {
            str(value)
            for value in multi_orders.get("owner_strategy_id", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .tolist()
            if str(value)
        }
    )
    owner_series = multi_orders.get("owner_strategy_id", pd.Series(dtype=str)).astype(
        str
    )
    status_series = multi_orders.get("status", pd.Series(dtype=str)).astype(str)
    alpha_rejections = multi_orders[
        (owner_series == "alpha") & (status_series.str.lower() == "rejected")
    ]
    reject_reasons = alpha_rejections.get("reject_reason", pd.Series(dtype=str)).fillna(
        ""
    )
    cooldown_rejections = int(
        reject_reasons.astype(str).str.contains("cooldown", case=False).sum()
    )
    size_rejections = int(
        reject_reasons.astype(str).str.contains("order quantity", case=False).sum()
    )
    beta_orders = int((owner_series == "beta").sum())

    print(f"single_owner_ids={single_owner_ids}")
    print(f"multi_owner_ids={multi_owner_ids}")
    print(f"multi_alpha_rejections={len(alpha_rejections)}")
    print(f"multi_alpha_size_rejections={size_rejections}")
    print(f"multi_alpha_cooldown_rejections={cooldown_rejections}")
    print(f"multi_beta_orders={beta_orders}")
    print("done_multi_strategy_demo")


if __name__ == "__main__":
    main()
