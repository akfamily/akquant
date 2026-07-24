from collections import defaultdict
from typing import Any

import pandas as pd
import pytest
from akquant import (
    BacktestConfig,
    Bar,
    ChinaFuturesConfig,
    ChinaFuturesInstrumentTemplateConfig,
    CurrentClose,
    InstrumentConfig,
    Strategy,
    StrategyConfig,
    run_backtest,
)
from akquant.config import RiskConfig


class AlwaysBuyStrategy(Strategy):
    """Submit a buy order on every bar."""

    def on_bar(self, bar: Bar) -> None:
        """Submit fixed-size market buy."""
        self.buy(bar.symbol, 30)


def _build_bars(
    timestamps: list[pd.Timestamp], prices: list[float], symbol: str = "RISK"
) -> list[Bar]:
    bars: list[Bar] = []
    for ts, price in zip(timestamps, prices):
        bars.append(
            Bar(
                timestamp=ts.value,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=10000.0,
                symbol=symbol,
            )
        )
    return bars


def _reject_reasons(result: Any) -> list[str]:
    orders_df = result.orders_df
    if orders_df.empty or "reject_reason" not in orders_df.columns:
        return []
    reasons = orders_df["reject_reason"].fillna("").astype(str).tolist()
    return [r for r in reasons if r.strip()]


def test_account_max_drawdown_rule_rejects_new_orders() -> None:
    """Reject new orders after drawdown breaches threshold."""
    bars = _build_bars(
        [
            pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-01 11:00:00", tz="Asia/Shanghai"),
        ],
        [100.0, 50.0],
    )
    result = run_backtest(
        data=bars,
        strategy=AlwaysBuyStrategy,
        symbols="RISK",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
        lot_size=1,
        risk_config=RiskConfig(max_account_drawdown=0.01),
    )
    reasons = _reject_reasons(result)
    assert any("Max drawdown" in r for r in reasons), reasons


def test_account_max_daily_loss_rule_rejects_new_orders_same_day() -> None:
    """Reject new orders after intraday daily-loss breach."""
    bars = _build_bars(
        [
            pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-01 14:00:00", tz="Asia/Shanghai"),
        ],
        [100.0, 50.0],
    )
    result = run_backtest(
        data=bars,
        strategy=AlwaysBuyStrategy,
        symbols="RISK",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
        lot_size=1,
        risk_config=RiskConfig(max_daily_loss=0.01),
    )
    reasons = _reject_reasons(result)
    assert any("Daily loss" in r for r in reasons), reasons


def test_account_stop_loss_threshold_rule_rejects_new_orders() -> None:
    """Reject new orders when equity falls below stop-loss threshold."""
    bars = _build_bars(
        [
            pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-01 11:00:00", tz="Asia/Shanghai"),
        ],
        [100.0, 50.0],
    )
    result = run_backtest(
        data=bars,
        strategy=AlwaysBuyStrategy,
        symbols="RISK",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
        lot_size=1,
        risk_config=RiskConfig(stop_loss_threshold=0.99),
    )
    reasons = _reject_reasons(result)
    assert any("stop-loss threshold" in r for r in reasons), reasons


def test_risk_config_accepts_check_cash_keyword() -> None:
    """RiskConfig supports check_cash as constructor argument."""
    rc = RiskConfig(check_cash=False)
    assert rc.check_cash is False


def test_check_cash_false_can_disable_margin_rejection_for_buy() -> None:
    """check_cash from Python RiskConfig should propagate to Rust engine config."""
    from akquant.akquant import Engine
    from akquant.risk import apply_risk_config

    engine = Engine()
    assert engine.risk_manager.config.check_cash is True

    apply_risk_config(engine, RiskConfig(check_cash=False))
    assert engine.risk_manager.config.check_cash is False


def test_apply_risk_config_logs_structured_context_for_invalid_sector_rule(
    caplog: Any,
) -> None:
    """Risk config warnings should include structured phase context."""
    from akquant.akquant import Engine
    from akquant.risk import apply_risk_config

    engine = Engine()

    with caplog.at_level("WARNING", logger="akquant"):
        apply_risk_config(engine, RiskConfig(sector_concentration=0.2))

    warning_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.risk"
        and "sector_concentration must be a tuple" in record.getMessage()
    )
    assert warning_record.phase == "risk"
    assert warning_record.strategy_id is None
    assert warning_record.slot is None
    assert warning_record.symbol is None


def test_short_option_margin_is_checked_and_account_margin_updates() -> None:
    """Short option opening orders should trigger margin checks and account margin."""

    class ShortPutStrategy(Strategy):
        account_snapshots: list[dict[str, float]] = []

        def __init__(self) -> None:
            self.order_count = 0

        def on_bar(self, bar: Bar) -> None:
            if bar.symbol != "PUT_OPT":
                return
            if self.order_count == 0:
                self.sell("PUT_OPT", 1)
                self.order_count += 1
            elif self.order_count == 1:
                self.sell("PUT_OPT", 100000)
                self.order_count += 1

        def on_trade(self, trade: Any) -> None:
            self.__class__.account_snapshots.append(self.get_account())

    dates = pd.date_range("2023-12-01 09:30", "2023-12-01 09:32", freq="1min")
    data_opt = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 1.5,
            "high": 1.5,
            "low": 1.5,
            "close": 1.5,
            "volume": 100,
            "symbol": "PUT_OPT",
        }
    )
    data_ul = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 105.0,
            "high": 105.0,
            "low": 105.0,
            "close": 105.0,
            "volume": 1000,
            "symbol": "UL",
        }
    )
    ShortPutStrategy.account_snapshots = []
    result = run_backtest(
        data={"PUT_OPT": data_opt, "UL": data_ul},
        strategy=ShortPutStrategy,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=BacktestConfig(
            strategy_config=StrategyConfig(
                initial_cash=50000.0,
                commission_rate=0.0,
                risk=RiskConfig(check_cash=True, safety_margin=0.0001),
            ),
            instruments_config=[
                InstrumentConfig(
                    symbol="PUT_OPT",
                    asset_type="OPTION",
                    multiplier=100,
                    margin_ratio=0.35,
                    option_type="PUT",
                    strike_price=90,
                    expiry_date=20231201,
                    underlying_symbol="UL",
                ),
                InstrumentConfig(symbol="UL", asset_type="STOCK"),
            ],
        ),
    )

    reasons = _reject_reasons(result)
    assert any("Insufficient margin" in r for r in reasons), reasons
    assert ShortPutStrategy.account_snapshots
    assert any(s["margin"] > 0.0 for s in ShortPutStrategy.account_snapshots)
    assert not result.margin_curve.empty
    assert len(result.margin_curve) == len(result.equity_curve)
    assert float(result.margin_curve.max()) > 0.0


def test_vol_adjusted_short_option_margin_is_checked() -> None:
    """Vol-adjusted short option should consume higher margin.

    The next leg should therefore be rejected.
    """

    class VolAdjustedShortPutStrategy(Strategy):
        account_snapshots: list[dict[str, float]] = []

        def __init__(self) -> None:
            self.order_count = 0

        def on_bar(self, bar: Bar) -> None:
            if bar.symbol != "PUT_VOL":
                return
            if self.order_count < 2:
                self.sell("PUT_VOL", 1)
                self.order_count += 1

        def on_trade(self, trade: Any) -> None:
            self.__class__.account_snapshots.append(self.get_account())

    dates = pd.date_range("2023-12-01 09:30", "2023-12-01 09:32", freq="1min")
    data_opt = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 4.0,
            "high": 4.0,
            "low": 4.0,
            "close": 4.0,
            "volume": 100,
            "symbol": "PUT_VOL",
        }
    )
    data_ul = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 95.0,
            "high": 95.0,
            "low": 95.0,
            "close": 95.0,
            "volume": 1000,
            "symbol": "UL",
        }
    )
    VolAdjustedShortPutStrategy.account_snapshots = []
    result = run_backtest(
        data={"PUT_VOL": data_opt, "UL": data_ul},
        strategy=VolAdjustedShortPutStrategy,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=BacktestConfig(
            strategy_config=StrategyConfig(
                initial_cash=6000.0,
                commission_rate=0.0,
                risk=RiskConfig(check_cash=True, safety_margin=0.0001),
            ),
            instruments_config=[
                InstrumentConfig(
                    symbol="PUT_VOL",
                    asset_type="OPTION",
                    multiplier=100,
                    margin_ratio=0.2,
                    option_type="PUT",
                    strike_price=100,
                    expiry_date=20231201,
                    underlying_symbol="UL",
                    option_margin_model="US_BROKER_SINGLE_LEG_VOL_ADJUSTED",
                    implied_volatility=0.3,
                    reference_volatility=0.2,
                ),
                InstrumentConfig(symbol="UL", asset_type="STOCK"),
            ],
        ),
    )

    reasons = _reject_reasons(result)
    assert any("Insufficient margin" in r for r in reasons), reasons
    assert VolAdjustedShortPutStrategy.account_snapshots
    sell_rows = result.orders_df[result.orders_df["side"].astype(str) == "sell"]
    assert float(sell_rows["filled_quantity"].sum()) == pytest.approx(1.0, rel=1e-12)
    assert not result.margin_curve.empty
    assert float(result.margin_curve.max()) >= 5750.0


def test_check_cash_false_lets_short_option_overdraft_at_execution() -> None:
    """#280: check_cash=False must skip the execution-time margin gate too.

    Mirror of ``test_vol_adjusted_short_option_margin_is_checked`` but with
    check_cash disabled: the account owner opts into overdrafting, so both
    short-option legs fill in full and cash is allowed to go negative — no
    "Insufficient margin at execution" rejection.
    """

    class OverdraftShortPutStrategy(Strategy):
        account_snapshots: list[dict[str, float]] = []

        def __init__(self) -> None:
            self.order_count = 0

        def on_bar(self, bar: Bar) -> None:
            if bar.symbol != "PUT_VOL":
                return
            if self.order_count < 2:
                self.sell("PUT_VOL", 1)
                self.order_count += 1

        def on_trade(self, trade: Any) -> None:
            self.__class__.account_snapshots.append(self.get_account())

    dates = pd.date_range("2023-12-01 09:30", "2023-12-01 09:32", freq="1min")
    data_opt = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 4.0,
            "high": 4.0,
            "low": 4.0,
            "close": 4.0,
            "volume": 100,
            "symbol": "PUT_VOL",
        }
    )
    data_ul = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 95.0,
            "high": 95.0,
            "low": 95.0,
            "close": 95.0,
            "volume": 1000,
            "symbol": "UL",
        }
    )
    OverdraftShortPutStrategy.account_snapshots = []
    result = run_backtest(
        data={"PUT_VOL": data_opt, "UL": data_ul},
        strategy=OverdraftShortPutStrategy,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=BacktestConfig(
            strategy_config=StrategyConfig(
                initial_cash=6000.0,
                commission_rate=0.0,
                risk=RiskConfig(check_cash=False, safety_margin=0.0001),
            ),
            instruments_config=[
                InstrumentConfig(
                    symbol="PUT_VOL",
                    asset_type="OPTION",
                    multiplier=100,
                    margin_ratio=0.2,
                    option_type="PUT",
                    strike_price=100,
                    expiry_date=20231201,
                    underlying_symbol="UL",
                    option_margin_model="US_BROKER_SINGLE_LEG_VOL_ADJUSTED",
                    implied_volatility=0.3,
                    reference_volatility=0.2,
                ),
                InstrumentConfig(symbol="UL", asset_type="STOCK"),
            ],
        ),
    )

    # No execution-time margin rejection — both legs fill in full.
    reasons = _reject_reasons(result)
    assert not any("Insufficient margin" in r for r in reasons), reasons
    sell_rows = result.orders_df[result.orders_df["side"].astype(str) == "sell"]
    assert float(sell_rows["filled_quantity"].sum()) == pytest.approx(2.0, rel=1e-12)


def test_margin_account_allows_short_sell_when_enabled() -> None:
    """Margin account should allow opening short stock positions."""

    class ShortStockStrategy(Strategy):
        def __init__(self) -> None:
            self.ordered = False

        def on_bar(self, bar: Bar) -> None:
            if not self.ordered:
                self.sell(bar.symbol, 10)
                self.ordered = True

    bars = _build_bars(
        [
            pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-01 10:01:00", tz="Asia/Shanghai"),
        ],
        [10.0, 10.0],
        symbol="SHORTABLE",
    )
    result = run_backtest(
        data=bars,
        strategy=ShortStockStrategy,
        symbols="SHORTABLE",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
        lot_size=1,
        risk_config=RiskConfig(account_mode="margin", enable_short_sell=True),
    )

    reasons = _reject_reasons(result)
    assert all("Insufficient available position" not in r for r in reasons), reasons
    sell_rows = result.orders_df[result.orders_df["side"].astype(str) == "sell"]
    assert not sell_rows.empty
    assert float(sell_rows["filled_quantity"].sum()) > 0.0


def test_margin_account_stock_buy_uses_initial_margin_ratio() -> None:
    """Margin account stock buying should apply initial margin ratio for sizing."""

    class LeveragedBuyStrategy(Strategy):
        account_snapshots: list[dict[str, Any]] = []

        def __init__(self) -> None:
            self.ordered = False

        def on_bar(self, bar: Bar) -> None:
            if not self.ordered:
                self.buy(bar.symbol, 1500)
                self.ordered = True

        def on_trade(self, trade: Any) -> None:
            self.__class__.account_snapshots.append(self.get_account())

    bars = _build_bars(
        [
            pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-01 10:01:00", tz="Asia/Shanghai"),
        ],
        [100.0, 100.0],
        symbol="MARGIN_BUY",
    )
    LeveragedBuyStrategy.account_snapshots = []
    result = run_backtest(
        data=bars,
        strategy=LeveragedBuyStrategy,
        symbols="MARGIN_BUY",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
        lot_size=1,
        risk_config=RiskConfig(
            account_mode="margin",
            enable_short_sell=True,
            initial_margin_ratio=0.5,
        ),
    )

    reasons = _reject_reasons(result)
    assert not any("Insufficient margin" in r for r in reasons), reasons
    filled_qty = float(result.orders_df["filled_quantity"].sum())
    assert filled_qty >= 1500.0
    assert LeveragedBuyStrategy.account_snapshots
    snap = LeveragedBuyStrategy.account_snapshots[-1]
    assert "borrowed_cash" in snap
    assert "short_market_value" in snap
    assert "maintenance_ratio" in snap
    assert snap.get("account_mode") == "margin"


def test_margin_account_same_cycle_cover_then_reopen_short_releases_margin() -> None:
    """Same-cycle short rotation should cover before reopening another short."""

    class MarginShortRotationStrategy(Strategy):
        def __init__(self) -> None:
            self.pending: dict[int, set[str]] = defaultdict(set)
            self.step = 0

        def on_bar(self, bar: Bar) -> None:
            bucket = self.pending[bar.timestamp]
            bucket.add(bar.symbol)
            if len(bucket) < 2:
                return
            self.pending.pop(bar.timestamp, None)

            if self.step == 0:
                self.short("AAA", 200.0, tag="init-short")
            elif self.step == 1:
                self.cover("AAA", 200.0, tag="cover-short")
                self.short("BBB", 200.0, tag="reopen-short")
            self.step += 1

    timestamps = [
        pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-01 10:01:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-01 10:02:00", tz="Asia/Shanghai"),
    ]
    data = {
        "AAA": pd.DataFrame(
            {
                "date": timestamps,
                "open": [10.0, 10.0, 10.0],
                "high": [10.0, 10.0, 10.0],
                "low": [10.0, 10.0, 10.0],
                "close": [10.0, 10.0, 10.0],
                "volume": [10000.0, 10000.0, 10000.0],
                "symbol": ["AAA", "AAA", "AAA"],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "date": timestamps,
                "open": [10.0, 10.0, 10.0],
                "high": [10.0, 10.0, 10.0],
                "low": [10.0, 10.0, 10.0],
                "close": [10.0, 10.0, 10.0],
                "volume": [10000.0, 10000.0, 10000.0],
                "symbol": ["BBB", "BBB", "BBB"],
            }
        ),
    }

    result = run_backtest(
        data=data,
        strategy=MarginShortRotationStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=5000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
        lot_size=1,
        risk_config=RiskConfig(
            account_mode="margin",
            enable_short_sell=True,
            initial_margin_ratio=0.5,
        ),
    )

    reasons = _reject_reasons(result)
    assert not any("Insufficient margin" in r for r in reasons), reasons
    orders_df = result.orders_df
    cover_rows = orders_df[orders_df["tag"].astype(str) == "cover-short"]
    reopen_rows = orders_df[orders_df["tag"].astype(str) == "reopen-short"]
    assert not cover_rows.empty
    assert not reopen_rows.empty
    assert set(cover_rows["status"].astype(str).str.lower()) == {"filled"}
    assert set(reopen_rows["status"].astype(str).str.lower()) == {"filled"}
    final_positions = result.positions.iloc[-1]
    assert float(final_positions.get("AAA", 0.0)) == 0.0
    assert float(final_positions.get("BBB", 0.0)) == -200.0


def test_margin_account_daily_financing_interest_is_deducted() -> None:
    """Margin account financing interest should be deducted on day switch."""

    class FinancingInterestStrategy(Strategy):
        account_snapshots: list[dict[str, Any]] = []

        def __init__(self) -> None:
            self.ordered = False

        def on_bar(self, bar: Bar) -> None:
            if not self.ordered:
                self.buy(bar.symbol, 150)
                self.ordered = True
                return
            self.__class__.account_snapshots.append(self.get_account())

    bars = _build_bars(
        [
            pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-01 14:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        ],
        [100.0, 100.0, 100.0],
        symbol="INTEREST",
    )
    FinancingInterestStrategy.account_snapshots = []
    run_backtest(
        data=bars,
        strategy=FinancingInterestStrategy,
        symbols="INTEREST",
        initial_cash=10000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
        lot_size=1,
        risk_config=RiskConfig(
            account_mode="margin",
            initial_margin_ratio=0.5,
            financing_rate_annual=36.5,
            borrow_rate_annual=0.0,
            allow_force_liquidation=False,
        ),
    )

    assert len(FinancingInterestStrategy.account_snapshots) >= 2
    day1_snapshot = FinancingInterestStrategy.account_snapshots[0]
    day2_snapshot = FinancingInterestStrategy.account_snapshots[1]
    day1_cash = float(day1_snapshot["cash"])
    day2_cash = float(day2_snapshot["cash"])
    assert day2_cash < day1_cash
    assert float(day2_snapshot.get("accrued_interest", 0.0)) > 0.0
    assert float(day2_snapshot.get("daily_interest", 0.0)) > 0.0


def test_margin_account_force_liquidation_on_maintenance_breach() -> None:
    """Margin account should force-liquidate positions when breached."""

    class ForceLiquidationStrategy(Strategy):
        pos_snapshots: list[float] = []

        def __init__(self) -> None:
            self.ordered = False

        def on_bar(self, bar: Bar) -> None:
            if not self.ordered:
                self.buy(bar.symbol, 150)
                self.ordered = True
            self.__class__.pos_snapshots.append(float(self.get_position(bar.symbol)))

    bars = _build_bars(
        [
            pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-01 14:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        ],
        [100.0, 20.0, 20.0],
        symbol="LIQ",
    )
    ForceLiquidationStrategy.pos_snapshots = []
    result = run_backtest(
        data=bars,
        strategy=ForceLiquidationStrategy,
        symbols="LIQ",
        initial_cash=10000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
        lot_size=1,
        risk_config=RiskConfig(
            account_mode="margin",
            initial_margin_ratio=0.5,
            maintenance_margin_ratio=0.5,
            financing_rate_annual=0.0,
            borrow_rate_annual=0.0,
            allow_force_liquidation=True,
        ),
    )

    assert ForceLiquidationStrategy.pos_snapshots
    assert any(p > 0.0 for p in ForceLiquidationStrategy.pos_snapshots[:-1])
    assert ForceLiquidationStrategy.pos_snapshots[-1] == 0.0
    liquidation_df = result.liquidation_audit_df
    assert not liquidation_df.empty
    assert "liquidated_symbols" in liquidation_df.columns
    assert "priority" in liquidation_df.columns


def test_futures_margin_account_snapshot_uses_margin_accounting() -> None:
    """Futures margin account snapshots should not deduct full notional from cash."""

    class FuturesAccountProbeStrategy(Strategy):
        account_snapshots: list[dict[str, Any]] = []

        def __init__(self) -> None:
            self.step = 0

        def on_bar(self, bar: Bar) -> None:
            if self.step == 0:
                self.buy(bar.symbol, 1.0)
                self.step = 1
            elif self.step == 1 and float(self.get_position(bar.symbol)) > 0.0:
                self.sell(bar.symbol, 1.0)
                self.step = 2

        def on_trade(self, trade: Any) -> None:
            self.__class__.account_snapshots.append(self.get_account())

    bars = _build_bars(
        [
            pd.Timestamp("2023-05-16 21:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-05-16 21:30:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-05-16 22:00:00", tz="Asia/Shanghai"),
        ],
        [4410.0, 4494.0, 4494.0],
        symbol="RB2305",
    )
    FuturesAccountProbeStrategy.account_snapshots = []
    run_backtest(
        data=bars,
        strategy=FuturesAccountProbeStrategy,
        symbols="RB2305",
        show_progress=False,
        fill_policy=CurrentClose(),
        config=BacktestConfig(
            strategy_config=StrategyConfig(
                initial_cash=500000.0,
                commission_rate=0.0,
                slippage=0.0,
                min_commission=0.0,
                stamp_tax_rate=0.0,
                transfer_fee_rate=0.0,
                risk=RiskConfig(
                    account_mode="margin",
                    check_cash=True,
                    initial_margin_ratio=0.1,
                    maintenance_margin_ratio=0.3,
                    enable_short_sell=True,
                    allow_force_liquidation=True,
                ),
            ),
            china_futures=ChinaFuturesConfig(
                enforce_sessions=False,
                instrument_templates_by_symbol_prefix=[
                    ChinaFuturesInstrumentTemplateConfig(
                        symbol_prefix="RB",
                        multiplier=10.0,
                        margin_ratio=0.1,
                        commission_rate=0.0,
                        tick_size=1.0,
                        lot_size=1.0,
                    )
                ],
            ),
        ),
    )

    assert len(FuturesAccountProbeStrategy.account_snapshots) >= 2
    open_snapshot = FuturesAccountProbeStrategy.account_snapshots[0]
    close_snapshot = FuturesAccountProbeStrategy.account_snapshots[1]

    assert float(open_snapshot["cash"]) == pytest.approx(500000.0, rel=0.0, abs=1e-6)
    assert float(open_snapshot["equity"]) == pytest.approx(500000.0, rel=0.0, abs=1e-6)
    assert float(open_snapshot["margin"]) == pytest.approx(4410.0, rel=0.0, abs=1e-6)
    assert float(open_snapshot["used_margin"]) == pytest.approx(
        4410.0, rel=0.0, abs=1e-6
    )
    assert float(open_snapshot["notional_value"]) == pytest.approx(
        44100.0, rel=0.0, abs=1e-6
    )
    assert float(open_snapshot["unrealized_pnl"]) == pytest.approx(
        0.0, rel=0.0, abs=1e-6
    )
    assert float(open_snapshot["market_value"]) == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert float(open_snapshot["maintenance_ratio"]) == pytest.approx(
        500000.0 / 4410.0, rel=0.0, abs=1e-6
    )

    assert float(close_snapshot["cash"]) == pytest.approx(500840.0, rel=0.0, abs=1e-6)
    assert float(close_snapshot["equity"]) == pytest.approx(500840.0, rel=0.0, abs=1e-6)
    assert float(close_snapshot["margin"]) == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert float(close_snapshot["used_margin"]) == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert float(close_snapshot["notional_value"]) == pytest.approx(
        0.0, rel=0.0, abs=1e-6
    )


def test_futures_short_overnight_avoids_spurious_liquidation() -> None:
    """Short futures should not accrue stock-style borrow interest.

    It also should not trigger spurious liquidation.
    """

    class FuturesShortOvernightProbeStrategy(Strategy):
        account_snapshots: list[dict[str, Any]] = []

        def __init__(self) -> None:
            self.ordered = False

        def on_bar(self, bar: Bar) -> None:
            if not self.ordered:
                self.short(bar.symbol, 1.0)
                self.ordered = True
            self.__class__.account_snapshots.append(self.get_account())

    bars = _build_bars(
        [
            pd.Timestamp("2023-01-04 22:30:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-04 23:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-05 09:30:00", tz="Asia/Shanghai"),
        ],
        [4008.0, 3999.0, 3985.0],
        symbol="RB2305",
    )
    FuturesShortOvernightProbeStrategy.account_snapshots = []
    result = run_backtest(
        data=bars,
        strategy=FuturesShortOvernightProbeStrategy,
        symbols="RB2305",
        show_progress=False,
        fill_policy=CurrentClose(),
        config=BacktestConfig(
            strategy_config=StrategyConfig(
                initial_cash=500000.0,
                commission_rate=0.0,
                slippage=0.0,
                min_commission=0.0,
                stamp_tax_rate=0.0,
                transfer_fee_rate=0.0,
                risk=RiskConfig(
                    account_mode="margin",
                    check_cash=True,
                    initial_margin_ratio=0.1,
                    maintenance_margin_ratio=0.3,
                    enable_short_sell=True,
                    allow_force_liquidation=True,
                    liquidation_priority="long_first",
                ),
            ),
            china_futures=ChinaFuturesConfig(
                enforce_sessions=False,
                instrument_templates_by_symbol_prefix=[
                    ChinaFuturesInstrumentTemplateConfig(
                        symbol_prefix="RB",
                        multiplier=10.0,
                        margin_ratio=0.1,
                        commission_rate=0.0,
                        tick_size=1.0,
                        lot_size=1.0,
                    )
                ],
            ),
        ),
    )

    assert len(FuturesShortOvernightProbeStrategy.account_snapshots) >= 3
    day2_snapshot = FuturesShortOvernightProbeStrategy.account_snapshots[-1]
    assert float(day2_snapshot.get("daily_interest", 0.0)) == pytest.approx(
        0.0, rel=0.0, abs=1e-6
    )
    assert float(day2_snapshot.get("accrued_interest", 0.0)) == pytest.approx(
        0.0, rel=0.0, abs=1e-6
    )
    assert result.liquidation_audit_df.empty
    assert float(result.positions.iloc[-1].get("RB2305", 0.0)) == pytest.approx(
        -1.0, rel=0.0, abs=1e-6
    )


def test_futures_force_liquidation_emits_callbacks_and_realizes_pnl_only() -> None:
    """Forced liquidation should emit order/trade callbacks.

    It should also keep futures equity on a PnL basis.
    """

    class FuturesForcedLiquidationProbeStrategy(Strategy):
        order_events: list[dict[str, Any]] = []
        trade_events: list[dict[str, Any]] = []
        position_snapshots: list[float] = []

        def __init__(self) -> None:
            self.ordered = False

        def on_bar(self, bar: Bar) -> None:
            if not self.ordered:
                self.short(bar.symbol, 1.0)
                self.ordered = True
            self.__class__.position_snapshots.append(
                float(self.get_position(bar.symbol))
            )

        def on_order(self, order: Any) -> None:
            self.__class__.order_events.append(
                {
                    "symbol": str(order.symbol),
                    "side": str(order.side),
                    "status": str(order.status),
                    "filled_quantity": float(order.filled_quantity),
                    "tag": str(getattr(order, "tag", "")),
                }
            )

        def on_trade(self, trade: Any) -> None:
            self.__class__.trade_events.append(
                {
                    "symbol": str(trade.symbol),
                    "side": str(trade.side),
                    "quantity": float(trade.quantity),
                    "price": float(trade.price),
                }
            )

    bars = _build_bars(
        [
            pd.Timestamp("2023-01-04 22:30:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-04 23:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-05 09:30:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-01-05 10:00:00", tz="Asia/Shanghai"),
        ],
        [4000.0, 6000.0, 6000.0, 6000.0],
        symbol="RB2305",
    )
    FuturesForcedLiquidationProbeStrategy.order_events = []
    FuturesForcedLiquidationProbeStrategy.trade_events = []
    FuturesForcedLiquidationProbeStrategy.position_snapshots = []
    result = run_backtest(
        data=bars,
        strategy=FuturesForcedLiquidationProbeStrategy,
        symbols="RB2305",
        show_progress=False,
        fill_policy=CurrentClose(),
        config=BacktestConfig(
            strategy_config=StrategyConfig(
                initial_cash=5000.0,
                commission_rate=0.0,
                slippage=0.0,
                min_commission=0.0,
                stamp_tax_rate=0.0,
                transfer_fee_rate=0.0,
                risk=RiskConfig(
                    account_mode="margin",
                    check_cash=True,
                    initial_margin_ratio=0.1,
                    maintenance_margin_ratio=0.3,
                    enable_short_sell=True,
                    allow_force_liquidation=True,
                    liquidation_priority="long_first",
                ),
            ),
            china_futures=ChinaFuturesConfig(
                enforce_sessions=False,
                instrument_templates_by_symbol_prefix=[
                    ChinaFuturesInstrumentTemplateConfig(
                        symbol_prefix="RB",
                        multiplier=10.0,
                        margin_ratio=0.1,
                        commission_rate=0.0,
                        tick_size=1.0,
                        lot_size=1.0,
                    )
                ],
            ),
        ),
    )

    assert not result.liquidation_audit_df.empty
    assert (
        result.orders_df["status"].astype(str).str.lower().tolist().count("filled") == 2
    )
    assert len(result.trades_df) == 1
    assert float(result.trades_df.iloc[0]["exit_price"]) == pytest.approx(
        6000.0, rel=0.0, abs=1e-6
    )
    assert float(result.positions.iloc[-1].get("RB2305", 0.0)) == pytest.approx(
        0.0, rel=0.0, abs=1e-6
    )
    assert float(result.equity_curve.iloc[-1]) == pytest.approx(
        -15000.0, rel=0.0, abs=1e-6
    )

    assert any(
        event["tag"] == "__forced_liquidation__"
        and event["status"].lower().endswith("filled")
        for event in FuturesForcedLiquidationProbeStrategy.order_events
    )
    assert len(FuturesForcedLiquidationProbeStrategy.trade_events) == 2
    assert (
        FuturesForcedLiquidationProbeStrategy.trade_events[-1]["side"]
        .lower()
        .endswith("buy")
    )
    assert FuturesForcedLiquidationProbeStrategy.position_snapshots[
        -1
    ] == pytest.approx(0.0, rel=0.0, abs=1e-6)


def test_futures_margin_portfolio_update_can_read_account_metrics() -> None:
    """Portfolio update callback should read cached futures account metrics safely."""

    class FuturesPortfolioUpdateProbeStrategy(Strategy):
        portfolio_updates: list[dict[str, float]] = []

        def __init__(self) -> None:
            self.ordered = False

        def on_bar(self, bar: Bar) -> None:
            if not self.ordered:
                self.buy(bar.symbol, 1.0)
                self.ordered = True

        def on_portfolio_update(self, snapshot: dict[str, Any]) -> None:
            account = self.get_account()
            self.__class__.portfolio_updates.append(
                {
                    "snapshot_equity": float(snapshot["equity"]),
                    "portfolio_value": float(self.equity),
                    "account_equity": float(account["equity"]),
                    "used_margin": float(account["used_margin"]),
                    "notional_value": float(account["notional_value"]),
                }
            )

    bars = _build_bars(
        [
            pd.Timestamp("2023-05-16 21:00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2023-05-16 21:30:00", tz="Asia/Shanghai"),
        ],
        [4410.0, 4494.0],
        symbol="RB2305",
    )
    FuturesPortfolioUpdateProbeStrategy.portfolio_updates = []
    run_backtest(
        data=bars,
        strategy=FuturesPortfolioUpdateProbeStrategy,
        symbols="RB2305",
        show_progress=False,
        fill_policy=CurrentClose(),
        config=BacktestConfig(
            strategy_config=StrategyConfig(
                initial_cash=500000.0,
                commission_rate=0.0,
                slippage=0.0,
                min_commission=0.0,
                stamp_tax_rate=0.0,
                transfer_fee_rate=0.0,
                risk=RiskConfig(
                    account_mode="margin",
                    check_cash=True,
                    initial_margin_ratio=0.1,
                    maintenance_margin_ratio=0.3,
                    enable_short_sell=True,
                    allow_force_liquidation=True,
                ),
            ),
            china_futures=ChinaFuturesConfig(
                enforce_sessions=False,
                instrument_templates_by_symbol_prefix=[
                    ChinaFuturesInstrumentTemplateConfig(
                        symbol_prefix="RB",
                        multiplier=10.0,
                        margin_ratio=0.1,
                        commission_rate=0.0,
                        tick_size=1.0,
                        lot_size=1.0,
                    )
                ],
            ),
        ),
    )

    assert FuturesPortfolioUpdateProbeStrategy.portfolio_updates
    for update in FuturesPortfolioUpdateProbeStrategy.portfolio_updates:
        assert update["portfolio_value"] == pytest.approx(
            update["account_equity"], rel=0.0, abs=1e-6
        )
        assert update["snapshot_equity"] == pytest.approx(
            update["account_equity"], rel=0.0, abs=1e-6
        )

    leveraged_updates = [
        update
        for update in FuturesPortfolioUpdateProbeStrategy.portfolio_updates
        if update["used_margin"] > 0.0
    ]
    assert leveraged_updates
    assert any(
        update["used_margin"] == pytest.approx(4410.0, rel=0.0, abs=1e-6)
        for update in leveraged_updates
    )
    assert any(
        update["notional_value"] == pytest.approx(44100.0, rel=0.0, abs=1e-6)
        for update in leveraged_updates
    )
