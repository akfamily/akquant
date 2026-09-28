"""期权/期货到期结算: 时点、收尾补结算、缺价延后、行权结算费."""

from typing import Any

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy

DAYS = ["2023-11-29", "2023-11-30", "2023-12-01", "2023-12-04"]


def _daily(symbol: str, days: list[str], closes: list[float]) -> pd.DataFrame:
    ts = pd.to_datetime([f"{d} 15:00" for d in days])
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1000,
            "symbol": symbol,
        }
    )


def _zero_fee_options() -> akquant.ChinaOptionsConfig:
    return akquant.ChinaOptionsConfig(
        commission_per_contract=0.0,
        exchange_fee_per_contract=0.0,
        clearing_fee_per_contract=0.0,
        exercise_fee_per_contract=0.0,
    )


def _call(**overrides: Any) -> akquant.InstrumentConfig:
    fields: dict[str, Any] = dict(
        symbol="OPT",
        asset_type="OPTION",
        multiplier=100.0,
        option_type="CALL",
        strike_price=100.0,
        expiry_date=20231201,
        underlying_symbol="UL",
    )
    fields.update(overrides)
    return akquant.InstrumentConfig(**fields)


class _ExpiryRecorder(Strategy):
    """首根 OPT bar 买入 1 张, 记录每天的持仓与 on_expiry 事件."""

    def __init__(self) -> None:
        super().__init__()
        self.day_positions: dict[str, float] = {}
        self.expiries: list[dict[str, Any]] = []

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != "OPT":
            return
        day = str(bar.timestamp_iso)[:10]
        self.day_positions[day] = float(self.get_position("OPT"))
        if day == DAYS[0]:
            self.buy(symbol="OPT", quantity=1)

    def on_expiry(self, event: dict[str, Any]) -> None:
        self.expiries.append(dict(event))


def _run(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    instruments: list[akquant.InstrumentConfig],
    china_options: Any = None,
) -> Any:
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=100_000.0),
        instruments_config=instruments,
        china_options=china_options or _zero_fee_options(),
    )
    return akquant.run_backtest(
        data=data, strategy=strategy, config=config, show_progress=False
    )


def test_option_settles_with_expiry_day_close() -> None:
    """到期日 12/01 标的收 120: 按 120 结算, 且到期日当天仍持有."""
    strat = _ExpiryRecorder()
    _run(
        strat,
        {
            "OPT": _daily("OPT", DAYS, [1.0, 1.0, 1.0, 1.0]),
            "UL": _daily("UL", DAYS, [100.0, 101.0, 120.0, 120.0]),
        },
        [_call(), akquant.InstrumentConfig(symbol="UL", asset_type="STOCK")],
    )
    assert strat.day_positions["2023-12-01"] == 1.0
    assert strat.day_positions["2023-12-04"] == 0.0
    assert len(strat.expiries) == 1
    event = strat.expiries[0]
    assert event["trading_date"] == "2023-12-04"
    assert event["cash_flow"] == pytest.approx((120.0 - 100.0) * 100.0)


def test_expiry_on_last_data_day_is_settled_at_end() -> None:
    """数据在到期日结束: 收尾补结算, 期末权益包含到期现金流."""
    days = DAYS[:3]
    result = _run(
        _ExpiryRecorder(),
        {
            "OPT": _daily("OPT", days, [1.0, 1.0, 1.0]),
            "UL": _daily("UL", days, [100.0, 101.0, 120.0]),
        },
        [_call(), akquant.InstrumentConfig(symbol="UL", asset_type="STOCK")],
    )
    # 11/30 以 1.0 × 100 买入 1 张(零费率), 到期现金流 2000
    assert result.metrics.end_market_value == pytest.approx(100_000.0 - 100.0 + 2000.0)


def test_futures_expiry_uses_expiry_day_close() -> None:
    """期货到期同样在到期日之后结算, 用到期日收盘价."""

    class _FutRecorder(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.day_positions: dict[str, float] = {}

        def on_bar(self, bar: Bar) -> None:
            day = str(bar.timestamp_iso)[:10]
            self.day_positions[day] = float(self.get_position("FUT"))
            if day == DAYS[0]:
                self.buy(symbol="FUT", quantity=1)

    strat = _FutRecorder()
    _run(
        strat,
        {"FUT": _daily("FUT", DAYS, [100.0, 101.0, 120.0, 120.0])},
        [
            akquant.InstrumentConfig(
                symbol="FUT",
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
                expiry_date=20231201,
            )
        ],
    )
    assert strat.day_positions["2023-12-01"] == 1.0
    assert strat.day_positions["2023-12-04"] == 0.0


def test_missing_underlying_defers_until_price_arrives() -> None:
    """标的 12/04 才有第一根 bar: 12/04 当天仍持有, 12/05 起按 12/04 价格结算."""
    days = DAYS + ["2023-12-05"]
    strat = _ExpiryRecorder()
    _run(
        strat,
        {
            "OPT": _daily("OPT", days, [1.0] * 5),
            "UL": _daily("UL", days[3:], [130.0, 130.0]),
        },
        [_call(), akquant.InstrumentConfig(symbol="UL", asset_type="STOCK")],
    )
    assert strat.day_positions["2023-12-04"] == 1.0
    assert strat.day_positions["2023-12-05"] == 0.0
    assert strat.expiries[0]["cash_flow"] == pytest.approx(3000.0)


def test_configured_settlement_price_is_used() -> None:
    """配置了 settlement_price 就用它, 不看标的最近价."""
    strat = _ExpiryRecorder()
    _run(
        strat,
        {
            "OPT": _daily("OPT", DAYS, [1.0] * 4),
            "UL": _daily("UL", DAYS, [100.0, 101.0, 120.0, 120.0]),
        },
        [
            _call(settlement_price=110.0),
            akquant.InstrumentConfig(symbol="UL", asset_type="STOCK"),
        ],
    )
    assert strat.expiries[0]["cash_flow"] == pytest.approx(1000.0)
    assert strat.expiries[0]["settlement_price"] == pytest.approx(110.0)


def test_no_underlying_ever_keeps_position() -> None:
    """标的始终没有价格: 不按 0 作废, 持仓一直保留到回测结束."""
    strat = _ExpiryRecorder()
    _run(
        strat,
        {"OPT": _daily("OPT", DAYS, [1.0] * 4)},
        [_call()],
    )
    assert strat.day_positions["2023-12-04"] == 1.0
    assert strat.expiries == []


def test_option_without_expiry_date_never_expires() -> None:
    """未配置 expiry_date 的期权永不到期: 持仓一直保留, 不产生到期事件."""
    strat = _ExpiryRecorder()
    _run(
        strat,
        {
            "OPT": _daily("OPT", DAYS, [1.0] * 4),
            "UL": _daily("UL", DAYS, [100.0, 101.0, 120.0, 120.0]),
        },
        [
            _call(expiry_date=None),
            akquant.InstrumentConfig(symbol="UL", asset_type="STOCK"),
        ],
    )
    for day in DAYS[1:]:
        assert strat.day_positions[day] == 1.0
    assert strat.expiries == []


def test_exercise_fee_reported_and_deducted() -> None:
    """行权结算费 0.6/张: 事件里报 fee, 现金流是毛额, 期末权益扣除费用."""
    strat = _ExpiryRecorder()
    options = akquant.ChinaOptionsConfig(
        commission_per_contract=0.0,
        exchange_fee_per_contract=0.0,
        clearing_fee_per_contract=0.0,
        exercise_fee_per_contract=0.6,
    )
    result = _run(
        strat,
        {
            "OPT": _daily("OPT", DAYS, [1.0] * 4),
            "UL": _daily("UL", DAYS, [100.0, 101.0, 120.0, 120.0]),
        },
        [_call(), akquant.InstrumentConfig(symbol="UL", asset_type="STOCK")],
        china_options=options,
    )
    event = strat.expiries[0]
    assert event["cash_flow"] == pytest.approx(2000.0)
    assert event["fee"] == pytest.approx(0.6)
    assert result.metrics.end_market_value == pytest.approx(
        100_000.0 - 100.0 + 2000.0 - 0.6
    )
