"""期权 Greeks 限额 ``Strategy.option_greek_limits``: 超限且增加暴露的委托作为风控拒单.

豆粕期权链(Black-76, σ=0.2, 标的期货 m2401 恒为 3000, 乘数 10): 平值认购每手 Delta
约 5。拒单经引擎订单账本回吐 ``on_reject``, 结果的委托表里状态为 Rejected。
"""

from typing import Any, Dict, List, Literal, Optional

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy
from akquant.gateway.order_submitter import BrokerOrderSubmitter
from akquant.option import black76_price

DAYS = ["2023-11-27", "2023-11-28", "2023-11-29"]
EXPIRY = 20231215
RATE = 0.02
SIGMA = 0.2
FUTURES = "m2401"
CALL = "m2401-C-3000"


def _t(day: str) -> float:
    return (pd.Timestamp(str(EXPIRY)) - pd.Timestamp(day)).days / 365.0


def _daily(symbol: str, closes: List[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([f"{d} 15:00" for d in DAYS]),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 100000,
            "symbol": symbol,
        }
    )


def _setup() -> tuple[Dict[str, pd.DataFrame], List[akquant.InstrumentConfig]]:
    data = {FUTURES: _daily(FUTURES, [3000.0] * 3)}
    configs = [
        akquant.InstrumentConfig(
            symbol=FUTURES, asset_type="FUTURES", multiplier=10.0, margin_ratio=0.1
        )
    ]
    kinds: tuple[tuple[Literal["CALL", "PUT"], str], ...] = (
        ("CALL", "C"),
        ("PUT", "P"),
    )
    for strike in (2900.0, 3000.0, 3100.0):
        for kind, flag in kinds:
            symbol = f"m2401-{flag}-{int(strike)}"
            closes = [
                float(black76_price(3000.0, strike, _t(d), RATE, SIGMA, kind))
                for d in DAYS
            ]
            data[symbol] = _daily(symbol, closes)
            configs.append(
                akquant.InstrumentConfig(
                    symbol=symbol,
                    asset_type="OPTION",
                    multiplier=10.0,
                    option_type=kind,
                    strike_price=strike,
                    expiry_date=EXPIRY,
                    underlying_symbol=FUTURES,
                )
            )
    # 价格低于内在价值(200)的期权: 解不出隐含波动率, 下单时 Greeks 无法计算
    data["m2401-C-2800"] = _daily("m2401-C-2800", [0.01] * 3)
    configs.append(
        akquant.InstrumentConfig(
            symbol="m2401-C-2800",
            asset_type="OPTION",
            multiplier=10.0,
            option_type="CALL",
            strike_price=2800.0,
            expiry_date=EXPIRY,
            underlying_symbol=FUTURES,
        )
    )
    return data, configs


class _Limited(Strategy):
    """每天全部 bar 到齐后调用 ``on_day``; 记录拒单."""

    option_rate = RATE
    n_symbols = 0

    def __init__(self) -> None:
        super().__init__()
        self._seen: Dict[str, set] = {}
        self.rejected: List[Any] = []
        self.records: Dict[str, Any] = {}

    def on_bar(self, bar: Bar) -> None:
        day = str(bar.timestamp_iso)[:10]
        seen = self._seen.setdefault(day, set())
        seen.add(bar.symbol)
        if len(seen) == self.n_symbols:
            self.on_day(DAYS.index(day))

    def on_day(self, i: int) -> None:
        """当日 bar 到齐后的回调."""

    def on_reject(self, order: Any) -> None:
        self.rejected.append(order)


def _run(strategy: _Limited) -> Any:
    data, configs = _setup()
    strategy.n_symbols = len(data)
    return akquant.run_backtest(
        data=data,
        strategy=strategy,
        config=akquant.BacktestConfig(
            strategy_config=akquant.StrategyConfig(initial_cash=10_000_000.0),
            instruments_config=configs,
            china_options=akquant.ChinaOptionsConfig(
                commission_per_contract=0.0,
                exchange_fee_per_contract=0.0,
                clearing_fee_per_contract=0.0,
                exercise_fee_per_contract=0.0,
            ),
        ),
        show_progress=False,
        freq="1d",
    )


class _OpenOverLimit(_Limited):
    """Delta 上限 12: 买 2 张平值认购(约 10)通过, 再买 1 张(含在途约 15)被拒."""

    option_greek_limits = {FUTURES: {"delta": 12.0}}

    def on_day(self, i: int) -> None:
        if i != 0:
            return
        self.records["unit_delta"] = self.get_greeks(CALL)["delta"] * 10.0
        self.buy(CALL, 2)
        self.buy(CALL, 1)
        self.buy("m2401-C-2800", 1)


def test_order_breaching_limit_is_rejected_with_projection() -> None:
    """投影含在途委托; 超限的拒单写明标的、Greek、下单后与下单前的值和上限."""
    strategy = _OpenOverLimit()
    result = _run(strategy)

    unit = strategy.records["unit_delta"]
    reasons = [o.reject_reason for o in strategy.rejected]
    assert reasons == [
        f"Order error: 期权 Greeks 限额(标的 {FUTURES}): delta 下单后 "
        f"{3 * unit:.6g}, 超过上限 12(下单前 {2 * unit:.6g}, 含在途委托)",
        reasons[1],
    ]
    assert reasons[1].startswith(
        f"Order error: 期权 Greeks 限额无法校验(标的 {FUTURES}): "
        "m2401-C-2800 无法计算 Greeks:"
    )
    assert strategy.get_position(CALL) == 2.0
    orders = result.orders_df
    assert (orders["status"].astype(str).str.lower() == "rejected").sum() == 2


class _ReduceOverLimit(_Limited):
    """无限额时持有 2 张认购(Delta 约 10), 次日上限设为 3: 减仓放行, 加仓被拒."""

    def on_day(self, i: int) -> None:
        if i == 0:
            self.buy(CALL, 2)
        elif i == 1:
            self.option_greek_limits = {FUTURES: {"delta": 3.0}}
            self.sell(CALL, 1)  # 约 10 -> 5: 减小, 放行
            self.buy(CALL, 1)  # 约 5 -> 10: 增大, 拒绝
            self.buy(FUTURES, 1)  # 约 5 -> 15: 增大, 拒绝
            self.sell(FUTURES, 1)  # 约 5.08 -> -4.92: 绝对值减小, 放行


def test_reducing_exposure_passes_even_when_over_limit() -> None:
    """超限状态下减小暴露的委托放行, 增加暴露的被拒."""
    strategy = _ReduceOverLimit()
    _run(strategy)

    assert [o.symbol for o in strategy.rejected] == [CALL, FUTURES]
    assert strategy.get_position(CALL) == 1.0
    assert strategy.get_position(FUTURES) == -1.0


class _BadConfig(_Limited):
    option_greek_limits = {FUTURES: {"rho": 1.0}}

    def on_day(self, i: int) -> None:
        if i == 0:
            self.buy(CALL, 1)


def test_invalid_limit_config_raises() -> None:
    """不支持的 Greek 名报出可选项."""
    with pytest.raises(
        ValueError, match="键 'rho' 不是 delta, gamma, vega, theta, cash_delta"
    ):
        _run(_BadConfig())


def test_broker_submitter_applies_greek_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """实盘下单器的前置风控同样校验 Greeks 限额."""
    calls: List[tuple] = []

    def fake(strategy: Any, symbol: str, side: str, quantity: float) -> Optional[str]:
        calls.append((symbol, side, quantity))
        return "期权 Greeks 限额(标的 m2401): delta 超限"

    monkeypatch.setattr("akquant.strategy_option_risk.greek_limit_violation", fake)
    submitter = BrokerOrderSubmitter.__new__(BrokerOrderSubmitter)
    submitter._strategy = object()
    submitter._strategy_limits = {}
    reason = submitter._check_risk(
        symbol=CALL, side="Buy", quantity=1.0, price=None, owner_strategy_id=""
    )
    assert reason == "期权 Greeks 限额(标的 m2401): delta 超限"
    assert calls == [(CALL, "Buy", 1.0)]
