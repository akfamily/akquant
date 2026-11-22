"""期货期权到期: 实值行权(履约)按行权价转为标的期货持仓, 其余放弃.

规则依据《上海期货交易所期权交易管理办法》第三十四、三十五、四十二条(大商所、郑商所、
广期所同): 以到期日标的期货结算价判定实值; 买卖双方按行权价获得方向相反的期货持仓;
行权与履约双方都收手续费。
"""

from typing import Any, Dict, List, Literal, Tuple

import akquant
import pandas as pd
from akquant import BarBatch, Strategy

DAYS = pd.date_range("2023-12-05 15:00", periods=5, freq="B", tz="Asia/Shanghai")
EXPIRY = 20231207  # DAYS[2]
FUTURES = "m2401"
FUTURES_CLOSES = [3000.0, 3050.0, 3100.0, 3150.0, 3150.0]
# (期权, 类型, 行权价, 开仓方向与数量)
OPTIONS: List[Tuple[str, Literal["CALL", "PUT"], float, int]] = [
    ("m2401-C-3000", "CALL", 3000.0, 2),  # 实值多头: 行权得买持仓 2
    ("m2401-P-3200", "PUT", 3200.0, -1),  # 实值空头: 履约得买持仓 1
    ("m2401-P-3000", "PUT", 3000.0, 1),  # 虚值多头: 放弃
]


def _bars(symbol: str, closes: List[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": DAYS,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1000,
            "symbol": symbol,
        }
    )


class _HoldToExpiry(Strategy):
    """首日按 OPTIONS 开仓并持有到期, 记录每日持仓与到期事件; 到期后平掉期货."""

    def __init__(self) -> None:
        super().__init__()
        self.positions_by_day: List[Dict[str, float]] = []
        self.events: List[Dict[str, Any]] = []

    def on_bars(self, batch: BarBatch) -> None:
        if not self.positions_by_day:
            for symbol, _, _, qty in OPTIONS:
                if qty > 0:
                    self.buy(symbol, qty)
                else:
                    self.sell(symbol, -qty)
        symbols = [FUTURES] + [o[0] for o in OPTIONS]
        self.positions_by_day.append({s: self.get_position(s) for s in symbols})
        if len(self.positions_by_day) == 4:  # 到期后次一交易日平掉行权得到的期货
            self.sell(FUTURES, self.get_position(FUTURES))

    def on_expiry(self, event: Dict[str, Any]) -> None:
        self.events.append(event)


def _run(futures_registered: bool) -> Tuple[_HoldToExpiry, Any]:
    data = {FUTURES: _bars(FUTURES, FUTURES_CLOSES)}
    configs = []
    if futures_registered:
        configs.append(
            akquant.InstrumentConfig(
                symbol=FUTURES, asset_type="FUTURES", multiplier=10.0, margin_ratio=0.1
            )
        )
    for symbol, kind, strike, _ in OPTIONS:
        data[symbol] = _bars(symbol, [50.0] * len(DAYS))
        configs.append(
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="OPTION",
                multiplier=10.0,
                option_type=kind,
                strike_price=strike,
                expiry_date=EXPIRY,
                underlying_symbol=FUTURES,
                option_margin_model="CHINA_SINGLE_LEG",
            )
        )
    strategy = _HoldToExpiry()
    result = akquant.run_backtest(
        data=data,
        strategy=strategy,
        config=akquant.BacktestConfig(
            strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
            instruments_config=configs,
            china_options=akquant.ChinaOptionsConfig(
                commission_per_contract=0.0,
                exchange_fee_per_contract=0.0,
                clearing_fee_per_contract=0.0,
                exercise_fee_per_contract=1.0,
            ),
        ),
        freq="1d",
        show_progress=False,
    )
    return strategy, result


def test_itm_options_become_futures_positions_at_strike() -> None:
    """到期后实值期权转为期货买持仓 3 手(行权 2 + 履约 1), 期权持仓归零."""
    strategy, result = _run(futures_registered=True)

    after = strategy.positions_by_day[3]  # 到期日次一交易日
    assert after[FUTURES] == 3.0
    assert all(after[o[0]] == 0.0 for o in OPTIONS)

    reasons = {e["symbol"]: e["reason"] for e in strategy.events}
    assert reasons == {
        "m2401-C-3000": "exercise",
        "m2401-P-3200": "assignment",
        "m2401-P-3000": "abandon",
    }
    fees = {e["symbol"]: e["fee"] for e in strategy.events}
    assert fees == {"m2401-C-3000": 2.0, "m2401-P-3200": 1.0, "m2401-P-3000": 0.0}

    trades = result.trades_df
    futures_trades = trades[trades["symbol"] == FUTURES]
    # 期货持仓按行权价建立, 次日按 3150 平仓: 3000 的 2 手盈 3000, 3200 的 1 手亏 500
    lots = sorted(
        zip(
            futures_trades["entry_price"],
            futures_trades["quantity"],
            futures_trades["pnl"],
        )
    )
    assert lots == [(3000.0, 2.0, 3000.0), (3200.0, 1.0, -500.0)]


def test_futures_options_cash_settle_when_underlying_not_registered_as_futures() -> (
    None
):
    """标的未登记为期货时维持原有现金结算, 不建立期货持仓."""
    strategy, _ = _run(futures_registered=False)

    after = strategy.positions_by_day[3]
    assert after[FUTURES] == 0.0
    assert {e["reason"] for e in strategy.events} == {"expiry"}
