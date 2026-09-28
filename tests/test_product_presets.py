"""品种预设与 InstrumentConfig 默认值哨兵."""

import akquant
import pandas as pd
import pytest
from akquant import Bar, InstrumentConfig, Strategy


def test_etf_option_preset_expands() -> None:
    """ETF_OPTION 展开为 OPTION 并补齐 ETF 期权缺省规则."""
    conf = InstrumentConfig(
        symbol="10007000.SH",
        asset_type="ETF_OPTION",
        option_type="CALL",
        strike_price=3.0,
        expiry_date=20991231,
        underlying_symbol="510050.SH",
    )
    assert conf.asset_type == "OPTION"
    assert conf.product_preset == "ETF_OPTION"
    assert conf.multiplier == 10000.0
    assert conf.tick_size == 0.0001
    assert conf.lot_size == 1
    assert conf.sellable_after_days == 0
    assert conf.option_margin_model == "CHINA_SINGLE_LEG"


def test_convertible_bond_preset_expands() -> None:
    """CONVERTIBLE_BOND 展开为 FUND 并补齐可转债缺省规则."""
    conf = InstrumentConfig(symbol="113050.SH", asset_type="CONVERTIBLE_BOND")
    assert conf.asset_type == "FUND"
    assert conf.product_preset == "CONVERTIBLE_BOND"
    assert conf.multiplier == 1.0
    assert conf.tick_size == 0.001
    assert conf.lot_size == 10
    assert conf.sellable_after_days == 0


def test_explicit_fields_override_preset() -> None:
    """显式传入的字段优先于预设值; 预设名大小写不敏感."""
    conf = InstrumentConfig(
        symbol="113050.SH",
        asset_type="convertible_bond",  # type: ignore[arg-type]
        lot_size=1,
        sellable_after_days=1,
        tick_size=0.01,
    )
    assert (conf.lot_size, conf.sellable_after_days, conf.tick_size) == (1, 1, 0.01)


def test_defaulted_fields_distinguish_explicit_one() -> None:
    """defaulted_fields 能区分缺省的 1.0 与显式传入的 1.0."""
    defaulted = InstrumentConfig(symbol="RB2401", asset_type="FUTURES")
    explicit = InstrumentConfig(symbol="RB2401", asset_type="FUTURES", multiplier=1.0)
    assert "multiplier" in defaulted.defaulted_fields
    assert "multiplier" not in explicit.defaulted_fields
    assert defaulted.multiplier == explicit.multiplier == 1.0


def _bars(symbol: str, n: int, price: float) -> pd.DataFrame:
    ts = pd.date_range("2024-01-02 10:00", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 100000,
            "symbol": symbol,
        }
    )


def test_futures_template_does_not_override_explicit_multiplier() -> None:
    """显式传 multiplier=1 不能再被期货模板当成"没传"而覆盖成 10."""

    class _Probe(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.multiplier: float = -1.0

        def on_bar(self, bar: Bar) -> None:
            self.multiplier = float(self.get_instrument("RB2401").multiplier)

    probe = _Probe()
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
        instruments_config=[
            InstrumentConfig(
                symbol="RB2401", asset_type="FUTURES", multiplier=1.0, margin_ratio=0.1
            )
        ],
        china_futures=akquant.ChinaFuturesConfig(
            enforce_sessions=False,
            instrument_templates_by_symbol_prefix=[
                akquant.ChinaFuturesInstrumentTemplateConfig(
                    symbol_prefix="RB", multiplier=10.0
                )
            ],
        ),
    )
    akquant.run_backtest(
        data={"RB2401": _bars("RB2401", 3, 3800.0)},
        strategy=probe,
        config=config,
        show_progress=False,
    )
    assert probe.multiplier == 1.0


def test_convertible_bond_round_trip_same_day_and_lot() -> None:
    """可转债 T+0: 当天买入当天卖出能成交; 5 张不足一手被拒."""

    class _Trader(Strategy):
        def on_bar(self, bar: Bar) -> None:
            if self._bar_count == 1:
                self.buy(symbol="113050.SH", quantity=5)
                self.buy(symbol="113050.SH", quantity=10)
            if self._bar_count == 2:
                self.sell(symbol="113050.SH", quantity=10)

    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=100_000.0),
        instruments_config=[
            InstrumentConfig(symbol="113050.SH", asset_type="CONVERTIBLE_BOND")
        ],
    )
    result = akquant.run_backtest(
        data={"113050.SH": _bars("113050.SH", 4, 120.0)},
        strategy=_Trader,
        config=config,
        t_plus_one=True,
        show_progress=False,
    )
    statuses = [(str(o.side), float(o.quantity), str(o.status)) for o in result.orders]
    assert ("OrderSide.Buy", 5.0, "OrderStatus.Rejected") in statuses
    assert ("OrderSide.Buy", 10.0, "OrderStatus.Filled") in statuses
    assert ("OrderSide.Sell", 10.0, "OrderStatus.Filled") in statuses


def test_etf_option_default_fees_end_to_end() -> None:
    """ETF_OPTION 预设 + 默认费率: 买开 1 张费用 6.6."""

    class _Buy(Strategy):
        def on_bar(self, bar: Bar) -> None:
            if bar.symbol == "OPT" and self._bar_count == 1:
                self.buy(symbol="OPT", quantity=1)

    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
        instruments_config=[
            InstrumentConfig(
                symbol="OPT",
                asset_type="ETF_OPTION",
                option_type="CALL",
                strike_price=3.0,
                expiry_date=20991231,
                underlying_symbol="UL",
            ),
            InstrumentConfig(symbol="UL", asset_type="FUND"),
        ],
    )
    result = akquant.run_backtest(
        data={"OPT": _bars("OPT", 4, 0.1234), "UL": _bars("UL", 4, 3.0)},
        strategy=_Buy,
        config=config,
        show_progress=False,
    )
    orders = result.orders_df
    filled = orders[orders["filled_quantity"] > 0]
    assert [float(c) for c in filled["commission"]] == pytest.approx([6.6])


def test_unknown_asset_type_still_rejected() -> None:
    """非底层类型也非预设名的 asset_type 仍然报错."""
    with pytest.raises(ValueError, match="Unsupported asset_type"):
        InstrumentConfig(symbol="X", asset_type="BOND")  # type: ignore[arg-type]
