"""买入开仓期权(权利方)按全额权利金校验可用资金, 与股票买入同一口径.

所需资金 = 权利金 × 合约乘数 × 数量 + 手续费。提交时按参考价(限价或最新价)、扣安全垫
校验, 成交时按真实成交价再校验一次; 买不起时拒单, 带缩量标记的单缩到买得起的最大张数。
期权多头的权利金计入占用(与现金账户里股票的全额市值一样), 所以持仓后可用资金已扣掉付出的
权利金, 挂着的期权买单冻结 委托价 × 乘数 × 数量。

卖出开仓(义务方)、买入平仓、卖出平仓, 以及期货、股票的结果与修复前逐笔相同(期望值取自
修复前的实跑)。
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy
from akquant.config import RiskConfig

_OPT = "CALL_OPT"
_UL = "UL"
_FUT = "RB2310"
_STOCK = "000001.SZ"
_TZ = "Asia/Shanghai"
_CASH = 1_000_000.0
_MULT = 10_000.0
_PREMIUM = 0.1  # 每张权利金 0.1 × 10000 = 1000 元
_SAFETY = 0.0001  # 风控缺省安全垫
# 期权买入开仓的默认费用(元/张): 佣金 5 + 经手费 1.3 + 结算费 0.3
_BUY_FEE = 5.0 + 1.3 + 0.3

Action = Callable[["_Script"], Any]
FillRow = Tuple[str, float, float, float]


def _daily(prices: List[float]) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-02 15:00", periods=len(prices), freq="B"
    ).tz_localize(_TZ)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1e9] * len(prices),
        },
        index=index,
    )


class _Script(Strategy):
    """第 n 根 K 线先记下账户快照, 再依次执行 script[n] 里的动作."""

    script: Dict[int, List[Action]] = {}
    target = _OPT

    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.accounts: List[Dict[str, Any]] = []

    def on_bar(self, bar: Bar) -> None:
        """按序号执行."""
        if bar.symbol != self.target:
            return
        self.accounts.append(dict(self.get_account()))
        for action in self.script.get(self.n, []):
            action(self)
        self.n += 1


def _option_instruments() -> List[Any]:
    return [
        aq.InstrumentConfig(
            symbol=_OPT,
            asset_type="OPTION",
            multiplier=_MULT,
            margin_ratio=0.35,
            option_type="CALL",
            strike_price=10,
            expiry_date=20301231,
            underlying_symbol=_UL,
        ),
    ]


def _run(
    script: Dict[int, List[Action]],
    *,
    prices: List[float],
    target: str = _OPT,
    instruments: Optional[List[Any]] = None,
    account_mode: str = "cash",
    policy: Any = None,
    check_cash: bool = True,
    active: bool = True,
    initial_cash: float = _CASH,
    commission_rate: float = 0.0,
) -> Tuple[Any, _Script]:
    cls = type("_Run", (_Script,), {"script": script, "target": target})
    strategy = cast(_Script, cls())
    risk = RiskConfig(
        account_mode=account_mode,
        check_cash=check_cash,
        active=active,
        enable_short_sell=account_mode == "margin",
        initial_margin_ratio=0.5 if account_mode == "margin" else 1.0,
    )
    config = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(
            initial_cash=initial_cash, commission_rate=commission_rate, risk=risk
        ),
        instruments_config=(
            instruments if instruments is not None else _option_instruments()
        ),
    )
    result = aq.run_backtest(
        data={target: _daily(prices)},
        strategy=strategy,
        symbols=target,
        freq="1d",
        fill_policy=policy if policy is not None else aq.NextOpen(),
        config=config,
        show_progress=False,
    )
    return result, strategy


def _orders(result: Any) -> List[Dict[str, Any]]:
    return cast(
        List[Dict[str, Any]],
        result.orders_df.sort_values("created_at").to_dict("records"),
    )


def _fills(result: Any) -> List[FillRow]:
    """(方向, 数量, 价格, 手续费) 按成交顺序."""
    rows: List[FillRow] = []
    for order in _orders(result):
        if float(order["filled_quantity"]) <= 0:
            continue
        rows.append(
            (
                str(order["side"]),
                float(order["filled_quantity"]),
                float(order["avg_price"]),
                float(order["commission"]),
            )
        )
    return rows


def _final_cash(result: Any) -> float:
    return float(result.cash_curve.iloc[-1])


def _buy(qty: float, **kwargs: Any) -> Action:
    return lambda s: s.buy(s.target, qty, **kwargs)


def _sell(qty: float, **kwargs: Any) -> Action:
    return lambda s: s.sell(s.target, qty, **kwargs)


def _target_value(value: float) -> Action:
    return lambda s: s.order_target_value(s.target, value)


def _max_affordable(available: float, unit: float) -> int:
    """独立算: 满足 n × unit <= available 的最大整数 n."""
    return int(math.floor(available / unit))


# ---------------------------------------------------------------------------
# 1. 复现: 显式数量超买 → 拒单, 现金不变
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("account_mode", ["cash", "margin"])
@pytest.mark.parametrize("policy_name", ["NextOpen", "CurrentClose"])
def test_unaffordable_option_buy_is_rejected(
    account_mode: str, policy_name: str
) -> None:
    """100 万资金买 2000 张(权利金合计 200 万)被拒, 现金不变."""
    policy = aq.NextOpen() if policy_name == "NextOpen" else aq.CurrentClose()
    result, strategy = _run(
        {0: [_buy(2000)]},
        prices=[_PREMIUM] * 4,
        account_mode=account_mode,
        policy=policy,
    )

    orders = _orders(result)
    assert len(orders) == 1
    assert str(orders[0]["status"]).lower() == "rejected"
    assert "Insufficient margin" in str(orders[0]["reject_reason"])
    assert float(orders[0]["filled_quantity"]) == 0.0
    assert _final_cash(result) == pytest.approx(_CASH)
    assert strategy.accounts[-1]["cash"] == pytest.approx(_CASH)


# ---------------------------------------------------------------------------
# 2. 买得起: 成交, 现金与可用资金都减少权利金加手续费
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("account_mode", ["cash", "margin"])
def test_affordable_option_buy_fills_and_reduces_available_cash(
    account_mode: str,
) -> None:
    """买 500 张(权利金合计 50 万): 成交; 可用资金 = 现金, 不再把权利金市值算回去."""
    qty = 500
    result, strategy = _run(
        {0: [_buy(qty)]},
        prices=[_PREMIUM] * 4,
        account_mode=account_mode,
    )

    premium_total = _PREMIUM * _MULT * qty  # 500,000
    fee = _BUY_FEE * qty  # 3,300
    expected_cash = _CASH - premium_total - fee  # 496,700
    assert _fills(result) == [
        ("buy", float(qty), pytest.approx(_PREMIUM), pytest.approx(fee))
    ]
    assert _final_cash(result) == pytest.approx(expected_cash)

    # 成交后下一根 K 线(价格不变)的账户快照
    after = strategy.accounts[2]
    assert after["cash"] == pytest.approx(expected_cash)
    assert after["equity"] == pytest.approx(expected_cash + premium_total)
    assert after["margin"] == pytest.approx(premium_total)
    assert after["frozen_cash"] == pytest.approx(0.0)
    assert after["available_cash"] == pytest.approx(expected_cash)
    assert after["available_cash"] == pytest.approx(
        strategy.accounts[0]["available_cash"] - premium_total - fee
    )


# ---------------------------------------------------------------------------
# 3. 缩量: order_target_value 带缩量标记, 缩到买得起的最大张数
# ---------------------------------------------------------------------------


def test_target_value_option_buy_is_resized_to_affordable_quantity() -> None:
    """目标市值 150 万(1500 张)只买得起 floor(100万×(1−安全垫) / (1000+6.6)) 张."""
    result, _ = _run({0: [_target_value(1_500_000.0)]}, prices=[_PREMIUM] * 4)

    unit = _PREMIUM * _MULT + _BUY_FEE  # 1006.6
    expected = _max_affordable(_CASH * (1 - _SAFETY), unit)
    assert expected == 993
    orders = _orders(result)
    assert len(orders) == 1
    assert float(orders[0]["filled_quantity"]) == float(expected)
    assert str(orders[0]["status"]).lower() == "filled"
    assert _final_cash(result) == pytest.approx(_CASH - expected * unit)
    assert _final_cash(result) >= 0.0


# ---------------------------------------------------------------------------
# 4. 成交时闸门: 次日权利金高开, 按参考价买得起、按开盘价买不起
# ---------------------------------------------------------------------------


def test_gap_up_rejects_explicit_option_buy_at_execution() -> None:
    """参考价 0.1 时 900 张买得起(90 万), 开盘 0.2 时要 180 万 → 成交时拒单."""
    result, _ = _run({0: [_buy(900)]}, prices=[_PREMIUM, 0.2, 0.2, 0.2])

    orders = _orders(result)
    assert len(orders) == 1
    assert str(orders[0]["status"]).lower() == "rejected"
    assert "Insufficient margin at execution" in str(orders[0]["reject_reason"])
    assert _final_cash(result) == pytest.approx(_CASH)


def test_gap_up_resizes_target_value_option_buy_at_execution() -> None:
    """同样的跳空, 带缩量标记的 900 张缩到按开盘价买得起的最大张数."""
    result, _ = _run({0: [_target_value(900_000.0)]}, prices=[_PREMIUM, 0.2, 0.2, 0.2])

    unit = 0.2 * _MULT + _BUY_FEE  # 2006.6
    expected = _max_affordable(_CASH, unit)  # 成交时不扣安全垫
    assert expected == 498
    orders = _orders(result)
    assert len(orders) == 1
    assert float(orders[0]["quantity"]) == float(expected)
    assert float(orders[0]["filled_quantity"]) == float(expected)
    assert float(orders[0]["avg_price"]) == pytest.approx(0.2)
    assert _final_cash(result) == pytest.approx(_CASH - expected * unit)


# ---------------------------------------------------------------------------
# 5. 在途冻结: 挂着的期权限价买单冻结权利金, 用超剩余资金的第二张被拒
# ---------------------------------------------------------------------------


def test_resting_option_buy_freezes_premium_and_blocks_second_buy() -> None:
    """限价 0.05 买 1000 张(不成交)冻结 50 万; 再挂 1100 张(55 万)超出剩余资金被拒."""
    limit = 0.05
    result, strategy = _run(
        {
            0: [_buy(1000, price=limit, time_in_force=aq.TimeInForce.GTC)],
            1: [_buy(1100, price=limit, time_in_force=aq.TimeInForce.GTC)],
        },
        prices=[_PREMIUM] * 4,
    )

    frozen = limit * _MULT * 1000  # 500,000, 不含手续费(与股票相同)
    snapshot = strategy.accounts[1]
    assert snapshot["frozen_cash"] == pytest.approx(frozen)
    assert snapshot["available_cash"] == pytest.approx(_CASH - frozen)
    assert snapshot["cash"] == pytest.approx(_CASH)

    orders = _orders(result)
    assert len(orders) == 2
    assert str(orders[0]["status"]).lower() != "rejected"
    assert float(orders[0]["filled_quantity"]) == 0.0
    assert str(orders[1]["status"]).lower() == "rejected"
    assert "Insufficient margin" in str(orders[1]["reject_reason"])


# ---------------------------------------------------------------------------
# 6. 卖出开仓、买入平仓、卖出平仓: 与修复前逐笔相同
# ---------------------------------------------------------------------------


def test_writer_open_and_buy_to_close_unchanged_when_funds_tight() -> None:
    """2 万资金卖出开仓 10 张; 权利金涨到 0.15 后账户已透支, 买入平仓 10 张照常成交."""
    result, strategy = _run(
        {0: [_sell(10)], 2: [_buy(10)]},
        prices=[_PREMIUM, _PREMIUM, 0.15, 0.15, 0.15],
        initial_cash=20_000.0,
    )

    assert _fills(result) == WRITER_FILLS
    assert _final_cash(result) == pytest.approx(WRITER_FINAL_CASH)
    assert [str(o["status"]).lower() for o in _orders(result)] == ["filled", "filled"]
    # 平仓前的可用资金为负(资金紧张), 平仓单仍不被拒
    assert strategy.accounts[2]["available_cash"] < 0.0


def test_buy_then_sell_to_close_unchanged() -> None:
    """买入开仓 5 张、权利金涨到 0.12 后卖出平仓 5 张, 逐笔与修复前相同."""
    result, _ = _run(
        {0: [_buy(5)], 2: [_sell(5)]},
        prices=[_PREMIUM, _PREMIUM, 0.12, 0.12, 0.12],
    )

    assert _fills(result) == LONG_ROUND_TRIP_FILLS
    assert _final_cash(result) == pytest.approx(LONG_ROUND_TRIP_FINAL_CASH)


# ---------------------------------------------------------------------------
# 7. 期货、股票不变性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("account_mode", ["cash", "margin"])
def test_futures_unchanged(account_mode: str) -> None:
    """期货开 3 手、超量开 100 手(被拒或缩量)、再平 2 手, 逐笔与修复前相同."""
    instruments = [
        aq.InstrumentConfig(
            symbol=_FUT, asset_type="FUTURES", multiplier=10.0, margin_ratio=0.1
        )
    ]
    result, _ = _run(
        {0: [_buy(3)], 1: [_buy(100)], 2: [_target_value(3_000_000.0)], 3: [_sell(2)]},
        prices=[3500.0, 3600.0, 3550.0, 3400.0, 3400.0],
        target=_FUT,
        instruments=instruments,
        account_mode=account_mode,
        initial_cash=100_000.0,
        commission_rate=0.0001,
    )

    observed = (
        [
            (str(o["status"]).lower(), float(o["filled_quantity"]))
            for o in _orders(result)
        ],
        _fills(result),
        _final_cash(result),
    )
    assert observed == FUTURES_EXPECTED[account_mode]


@pytest.mark.parametrize("account_mode", ["cash", "margin"])
def test_stock_unchanged(account_mode: str) -> None:
    """股票超买被拒、目标市值缩量、卖出, 逐笔与修复前相同."""
    instruments = [aq.InstrumentConfig(symbol=_STOCK, asset_type="STOCK")]
    result, _ = _run(
        {0: [_buy(20_000)], 1: [_target_value(150_000.0)], 3: [_sell(2000)]},
        prices=[10.0, 10.0, 11.0, 12.0, 12.0],
        target=_STOCK,
        instruments=instruments,
        account_mode=account_mode,
        initial_cash=100_000.0,
        commission_rate=0.0003,
    )

    observed = (
        [
            (str(o["status"]).lower(), float(o["filled_quantity"]))
            for o in _orders(result)
        ],
        _fills(result),
        _final_cash(result),
    )
    assert observed == STOCK_EXPECTED[account_mode]


# ---------------------------------------------------------------------------
# 8. check_cash=False / 风控关闭: 照旧不校验
# ---------------------------------------------------------------------------


def test_check_cash_false_still_overdraws_on_option_buy() -> None:
    """check_cash=False 时 2000 张照旧整单成交, 现金透支."""
    result, _ = _run({0: [_buy(2000)]}, prices=[_PREMIUM] * 4, check_cash=False)

    assert _fills(result) == [
        ("buy", 2000.0, pytest.approx(_PREMIUM), pytest.approx(_BUY_FEE * 2000))
    ]
    assert _final_cash(result) == pytest.approx(
        _CASH - _PREMIUM * _MULT * 2000 - _BUY_FEE * 2000
    )


# 期望值取自修复前(54aede1f)的实跑
WRITER_FILLS: List[FillRow] = [("sell", 10.0, 0.1, 50.0), ("buy", 10.0, 0.15, 66.0)]
WRITER_FINAL_CASH = 14884.0
LONG_ROUND_TRIP_FILLS: List[FillRow] = [
    ("buy", 5.0, 0.1, 33.0),
    ("sell", 5.0, 0.12, 33.0),
]
LONG_ROUND_TRIP_FINAL_CASH = 1000934.0
FUTURES_EXPECTED: Dict[str, Any] = {
    "cash": (
        [("filled", 3.0), ("rejected", 0.0), ("filled", 24.0), ("filled", 2.0)],
        [
            ("buy", 3.0, 3600.0, 10.8),
            ("buy", 24.0, 3400.0, 81.6),
            ("sell", 2.0, 3400.0, 6.8),
        ],
        -756099.2,
    ),
    "margin": (
        [("filled", 3.0), ("rejected", 0.0), ("filled", 25.0), ("filled", 2.0)],
        [
            ("buy", 3.0, 3600.0, 10.8),
            ("buy", 25.0, 3400.0, 85.0),
            ("sell", 2.0, 3400.0, 6.8),
        ],
        95897.4,
    ),
}
STOCK_EXPECTED: Dict[str, Any] = {
    "cash": (
        [("rejected", 0.0), ("filled", 9087.0), ("filled", 2000.0)],
        [("buy", 9087.0, 11.0, 29.9871), ("sell", 2000.0, 12.0, 7.2)],
        24005.8129,
    ),
    "margin": (
        [("rejected", 0.0), ("filled", 15000.0), ("filled", 2000.0)],
        [("buy", 15000.0, 11.0, 49.5), ("sell", 2000.0, 12.0, 7.2)],
        -41112.23291207613,
    ),
}
