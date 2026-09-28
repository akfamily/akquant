"""期权按张计费口径(沪深 ETF 期权)与期权默认走 ChinaMarket."""

from typing import Any

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy


def _bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    ts = pd.date_range("2024-01-02 10:00", periods=len(closes), freq="1min")
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


def _option_conf() -> akquant.InstrumentConfig:
    return akquant.InstrumentConfig(
        symbol="OPT",
        asset_type="OPTION",
        option_type="CALL",
        strike_price=2.0,
        expiry_date=20991231,
        underlying_symbol="UL",
        multiplier=10000.0,
        tick_size=0.0001,
        lot_size=1,
    )


class _Trader(Strategy):
    # 不用带参 __init__: 0.3.x 起那会触发"参数无法从外部传入"的弃用告警。构造后再赋值。
    actions: dict[int, tuple[str, float]]

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != "OPT":
            return
        action = self.actions.get(self._bar_count)
        if action is None:
            return
        side, qty = action
        if side == "buy":
            self.buy(symbol="OPT", quantity=qty)
        else:
            self.sell(symbol="OPT", quantity=qty)


def _commissions(
    actions: dict[int, tuple[str, float]], china_options: Any = None
) -> list[float]:
    trader = _Trader()
    trader.actions = actions
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=10_000_000.0),
        instruments_config=[
            _option_conf(),
            akquant.InstrumentConfig(symbol="UL", asset_type="FUND"),
        ],
        china_options=china_options,
    )
    result = akquant.run_backtest(
        data={"OPT": _bars("OPT", [0.1] * 8), "UL": _bars("UL", [2.0] * 8)},
        strategy=trader,
        config=config,
        show_progress=False,
    )
    return _filled_commissions(result)


def _filled_commissions(result: Any) -> list[float]:
    """按下单顺序返回已成交订单的手续费(trades_df 只含已平仓的往返交易, 不能用)."""
    orders = result.orders_df
    filled = orders[orders["filled_quantity"] > 0]
    return [float(c) for c in filled["commission"]]


def test_default_market_uses_per_contract_fees_for_options() -> None:
    """没配 china_options 时也按张计费: 买开 2 张 = 2 × 6.6."""
    assert _commissions({1: ("buy", 2)}) == pytest.approx([13.2])


def test_sell_open_only_pays_commission() -> None:
    """卖出开仓免经手费与结算费: 3 张 × 5."""
    assert _commissions({1: ("sell", 3)}) == pytest.approx([15.0])


def test_sell_crossing_zero_splits_fee() -> None:
    """持多 2 张卖 5 张: 2 × 6.6 + 3 × 5.

    position_effect="auto" 会把这笔卖单拆成平仓 2 张 + 卖开 3 张两条订单(#361),
    所以按腿核对: 平仓腿全额 13.2, 卖开腿只收佣金 15.0, 合计 28.2。
    """
    fees = _commissions({1: ("buy", 2), 3: ("sell", 5)})
    assert fees == pytest.approx([13.2, 13.2, 15.0])


@pytest.mark.parametrize(
    ("sell_open_exempt", "expected"),
    [
        # 不豁免: 3 × (2 + 1.0 + 0.5)
        (False, 10.5),
        # 豁免: 卖开只收佣金 3 × 2
        (True, 6.0),
    ],
)
def test_china_options_config_overrides_fee_fields(
    sell_open_exempt: bool, expected: float
) -> None:
    """ChinaOptionsConfig 的费率字段与 sell_open_exempt 开关都生效."""
    cfg = akquant.ChinaOptionsConfig(
        commission_per_contract=2.0,
        exchange_fee_per_contract=1.0,
        clearing_fee_per_contract=0.5,
        sell_open_exempt=sell_open_exempt,
    )
    assert _commissions({1: ("sell", 3)}, cfg) == pytest.approx([expected])


def test_prefix_fee_rule_takes_priority() -> None:
    """前缀规则优先于全局规则."""
    cfg = akquant.ChinaOptionsConfig(
        commission_per_contract=5.0,
        fee_by_symbol_prefix=[
            akquant.ChinaOptionsFeeConfig(
                symbol_prefix="OPT",
                commission_per_contract=1.0,
                exchange_fee_per_contract=0.0,
                clearing_fee_per_contract=0.0,
            )
        ],
    )
    assert _commissions({1: ("buy", 4)}, cfg) == pytest.approx([4.0])


def test_negative_fee_is_rejected() -> None:
    """费率不能为负."""
    with pytest.raises(ValueError, match="exchange_fee_per_contract must be >= 0"):
        akquant.ChinaOptionsConfig(exchange_fee_per_contract=-1.0)
