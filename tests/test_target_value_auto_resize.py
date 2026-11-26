"""按目标金额、比例、权重买入: 资金不足时缩量成交, 而不是整单被拒 (#459).

数量由目标金额换算出来的买入(order_target_value / order_target_percent /
rebalance_weights)打开引擎已有的"按可用资金缩量"; 数量由调用方明确给出的入口
(order_target / rebalance_positions / buy(symbol, quantity))仍整单校验。
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy

_SYMBOL = "000001.SZ"
_OTHER = "600000.SH"
_CASH = 100000.0
_LOT = 100
_SAFETY = 0.0001  # 风控缺省安全垫
_COMMISSION = 0.0003
_SLIPPAGE = 0.002
_POLICIES = [aq.NextOpen, aq.CurrentClose]
_POLICY_IDS = ["next_open", "current_close"]

Action = Callable[[Strategy], Any]
OrderRow = Tuple[str, str, str, float, str]


def _daily(prices: List[float], opens: Optional[List[float]] = None) -> pd.DataFrame:
    index = pd.to_datetime(
        [f"2024-01-{day:02d} 15:00" for day in range(2, 2 + len(prices))]
    ).tz_localize("Asia/Shanghai")
    opens = opens or prices
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, prices)],
            "low": [min(o, c) for o, c in zip(opens, prices)],
            "close": prices,
            "volume": [1e9] * len(prices),
        },
        index=index,
    )


def _run(
    actions: Dict[int, Action],
    policy: Any,
    *,
    fees: bool = False,
    cash: float = _CASH,
    data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Any:
    """第 n 根 K 线(按 _SYMBOL 计)执行 actions[n]."""

    class _Run(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def on_bar(self, bar: Bar) -> None:
            if bar.symbol != _SYMBOL:
                return
            action = actions.get(self.n)
            if action is not None:
                action(self)
            self.n += 1

    data = data or {_SYMBOL: _daily([10.0] * 4)}
    cost: Dict[str, Any] = (
        {
            "commission_rate": _COMMISSION,
            "min_commission": 5.0,
            "slippage": {"type": "percent", "value": _SLIPPAGE},
        }
        if fees
        else {"commission_rate": 0.0, "min_commission": 0.0}
    )
    return aq.run_backtest(
        data=data,
        strategy=_Run,
        symbols=list(data),
        initial_cash=cash,
        freq="1d",
        fill_policy=policy,
        t_plus_one=False,
        lot_size=_LOT,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        show_progress=False,
        **cost,
    )


def _orders(result: Any) -> List[OrderRow]:
    """(品种, 方向, 状态, 成交量, 拒单原因) 按提交顺序."""
    return [
        (
            str(row["symbol"]),
            str(row["side"]),
            str(row["status"]),
            float(row["filled_quantity"]),
            str(row["reject_reason"] or ""),
        )
        for row in result.orders_df.sort_values("created_at", kind="stable").to_dict(
            "records"
        )
    ]


def _affordable(cash: float, price: float, *, fees: bool) -> float:
    """按手取整后买得起的最大数量: 可用资金扣安全垫, 每股成本含滑点与佣金."""
    unit = price * (1 + _SLIPPAGE) * (1 + _COMMISSION) if fees else price
    return float(math.floor(cash * (1 - _SAFETY) / unit / _LOT) * _LOT)


_FULL: List[Tuple[str, Action]] = [
    ("rebalance_weights", lambda s: s.rebalance_weights({_SYMBOL: 1.0})),
    ("order_target_percent", lambda s: s.order_target_percent(_SYMBOL, 1.0)),
    ("order_target_value", lambda s: s.order_target_value(_SYMBOL, _CASH)),
]


@pytest.mark.parametrize("fees", [False, True], ids=["no_fees", "fees"])
@pytest.mark.parametrize("policy", _POLICIES, ids=_POLICY_IDS)
@pytest.mark.parametrize("action", [a for _, a in _FULL], ids=[n for n, _ in _FULL])
def test_full_weight_buy_is_resized_instead_of_rejected(
    action: Action, policy: Any, fees: bool
) -> None:
    """满仓调仓: 此前整单被拒(Insufficient margin), 现在缩到买得起的最大整手数."""
    expected = _affordable(_CASH, 10.0, fees=fees)
    assert expected == 9900.0  # 99990 / 10 = 9999 股, 按手取整; 含费用时同为 9900

    assert _orders(_run({0: action}, policy(), fees=fees)) == [
        (_SYMBOL, "buy", "filled", expected, "")
    ]


def _target_percent(percent: float) -> Action:
    return lambda s: s.order_target_percent(_SYMBOL, percent)


@pytest.mark.parametrize("policy", _POLICIES, ids=_POLICY_IDS)
@pytest.mark.parametrize(("percent", "quantity"), [(0.99, 9900.0), (0.5, 5000.0)])
def test_buy_within_cash_is_unchanged(
    percent: float, quantity: float, policy: Any
) -> None:
    """资金充足时数量不变(修复前实跑: 0.99 → 9900, 0.5 → 5000)."""
    result = _run({0: _target_percent(percent)}, policy(), fees=True)

    assert _orders(result) == [(_SYMBOL, "buy", "filled", quantity, "")]


def test_gap_up_open_resizes_at_execution() -> None:
    """NextOpen 次日高开: 提交时按 10 元缩到 9900, 成交时按开盘价 10.5 再缩."""
    data = {_SYMBOL: _daily([10.0, 10.5, 10.5, 10.5])}
    expected = _affordable(_CASH, 10.5, fees=False)
    assert expected == 9500.0  # 99990 / 10.5 = 9522 股

    result = _run(
        {0: lambda s: s.order_target_percent(_SYMBOL, 1.0)}, aq.NextOpen(), data=data
    )

    assert _orders(result) == [(_SYMBOL, "buy", "filled", expected, "")]


@pytest.mark.parametrize("policy", _POLICIES, ids=_POLICY_IDS)
def test_two_symbol_rebalance_fills_both_legs_within_cash(policy: Any) -> None:
    """各 50% 的两腿都成交: 第一腿足额, 第二腿按第一腿占用后的余额缩量.

    两腿各按 50000 元换算数量(5000 股、2500 股); 含滑点与佣金后第一腿实际占用
    超过 50000, 第二腿此前整单被拒, 现在缩到余额买得起的最大整手数。
    """
    data = {_SYMBOL: _daily([10.0] * 4), _OTHER: _daily([20.0] * 4)}
    result = _run(
        # 第 1 根才调仓: 第 0 根上另一品种还没有价格
        {1: lambda s: s.rebalance_weights({_SYMBOL: 0.5, _OTHER: 0.5})},
        policy(),
        fees=True,
        data=data,
    )

    rows = _orders(result)
    assert [(row[0], row[2]) for row in rows] == [
        (_SYMBOL, "filled"),
        (_OTHER, "filled"),
    ]
    filled = {row[0]: row[3] for row in rows}
    first_cost = 5000.0 * 10.0 * (1 + _SLIPPAGE) * (1 + _COMMISSION)
    second = _affordable(_CASH - first_cost, 20.0, fees=True)
    assert second == 2400.0
    assert filled == {_SYMBOL: 5000.0, _OTHER: second}
    spent = sum(float(t["quantity"]) * float(t["price"]) for t in _trades(result))
    assert spent <= _CASH


def _trades(result: Any) -> List[Dict[str, Any]]:
    return list(result.executions_df.to_dict("records"))


@pytest.mark.parametrize("policy", _POLICIES, ids=_POLICY_IDS)
def test_rotation_buys_with_cash_released_by_the_sell(policy: Any) -> None:
    """先持有 A(一半仓位), 再调成全仓 B: A 卖出回笼的资金用于买 B, B 缩量成交."""
    data = {_SYMBOL: _daily([10.0] * 5), _OTHER: _daily([20.0] * 5)}
    result = _run(
        {
            1: lambda s: s.order_target_percent(_SYMBOL, 0.5),
            3: lambda s: s.rebalance_weights({_SYMBOL: 0.0, _OTHER: 1.0}),
        },
        policy(),
        data=data,
    )

    rows = _orders(result)
    assert [(row[0], row[1], row[2]) for row in rows] == [
        (_SYMBOL, "buy", "filled"),
        (_SYMBOL, "sell", "filled"),
        (_OTHER, "buy", "filled"),
    ]
    # 无费用、价格不变: 权益仍是 100000, B 每股 20 元 → 99990 / 20 = 4999 股, 按手 4900
    assert rows[1][3] == 5000.0
    assert rows[2][3] == _affordable(_CASH, 20.0, fees=False) == 4900.0


_EXPLICIT: List[Tuple[str, Action]] = [
    ("order_target", lambda s: s.order_target(_SYMBOL, 10000)),
    ("rebalance_positions", lambda s: s.rebalance_positions({_SYMBOL: 10000.0})),
    ("buy", lambda s: s.buy(_SYMBOL, 10000)),
    (
        "order_target_value_opt_out",
        lambda s: s.order_target_value(
            _SYMBOL, _CASH, allow_quantity_auto_resize=False
        ),
    ),
    (
        "order_target_percent_opt_out",
        lambda s: s.order_target_percent(
            _SYMBOL, 1.0, allow_quantity_auto_resize=False
        ),
    ),
    (
        "rebalance_weights_opt_out",
        lambda s: s.rebalance_weights({_SYMBOL: 1.0}, allow_quantity_auto_resize=False),
    ),
]


@pytest.mark.parametrize("policy", _POLICIES, ids=_POLICY_IDS)
@pytest.mark.parametrize(
    "action", [a for _, a in _EXPLICIT], ids=[n for n, _ in _EXPLICIT]
)
def test_explicit_quantity_or_opt_out_is_still_rejected(
    action: Action, policy: Any
) -> None:
    """数量由调用方明确给出、或显式关掉缩量: 买不起仍整单被拒."""
    rows = _orders(_run({0: action}, policy()))

    assert [row[:4] for row in rows] == [(_SYMBOL, "buy", "rejected", 0.0)]
    assert "Insufficient margin" in rows[0][4]


@pytest.mark.parametrize("policy", _POLICIES, ids=_POLICY_IDS)
def test_order_target_can_opt_in(policy: Any) -> None:
    """order_target 显式打开缩量时同样缩量."""
    result = _run(
        {0: lambda s: s.order_target(_SYMBOL, 10000, allow_quantity_auto_resize=True)},
        policy(),
    )

    assert _orders(result) == [(_SYMBOL, "buy", "filled", 9900.0, "")]


@pytest.mark.parametrize("policy", _POLICIES, ids=_POLICY_IDS)
def test_less_than_one_lot_places_no_order(policy: Any) -> None:
    """资金 500、每股 10 元、每手 100: 目标 50 股不足一手, 不下单(现状)."""
    result = _run(
        {0: lambda s: s.order_target_percent(_SYMBOL, 1.0)}, policy(), cash=500.0
    )

    assert _orders(result) == []


@pytest.mark.parametrize("policy", _POLICIES, ids=_POLICY_IDS)
def test_one_lot_unaffordable_after_resize_is_rejected(policy: Any) -> None:
    """资金 1000、每股 10 元: 目标 100 股, 扣安全垫后一手都买不起, 仍被拒(现状)."""
    result = _run(
        {0: lambda s: s.order_target_percent(_SYMBOL, 1.0)}, policy(), cash=1000.0
    )

    rows = _orders(result)
    assert [row[:4] for row in rows] == [(_SYMBOL, "buy", "rejected", 0.0)]


# ---------------------------------------------------------------------------
# 期货: 目标金额按名义市值(数量 × 价格 × 合约乘数)换算
# ---------------------------------------------------------------------------

_FUT = "RB2310"
_MULTIPLIER = 10.0


def _run_futures(action: Action) -> Any:
    class _Fut(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def on_bar(self, bar: Bar) -> None:
            if self.n == 0:
                action(self)
            self.n += 1

    config = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=_CASH, commission_rate=0.0),
        instruments_config=[
            aq.InstrumentConfig(
                symbol=_FUT,
                asset_type="FUTURES",
                multiplier=_MULTIPLIER,
                margin_ratio=0.1,
            )
        ],
        china_futures=aq.ChinaFuturesConfig(),
    )
    return aq.run_backtest(
        data={_FUT: _daily([10.0] * 4)},
        strategy=_Fut,
        symbols=_FUT,
        freq="1d",
        fill_policy=aq.NextOpen(),
        config=config,
        show_progress=False,
    )


_FUT_HALF: List[Tuple[str, Action]] = [
    ("order_target_value", lambda s: s.order_target_value(_FUT, _CASH / 2)),
    ("order_target_percent", lambda s: s.order_target_percent(_FUT, 0.5)),
    ("rebalance_weights", lambda s: s.rebalance_weights({_FUT: 0.5})),
]


@pytest.mark.parametrize(
    "action", [a for _, a in _FUT_HALF], ids=[n for n, _ in _FUT_HALF]
)
def test_futures_target_value_uses_contract_multiplier(action: Action) -> None:
    """权益 100000 的 50% = 名义市值 50000 = 500 手(每手 10 元 × 乘数 10).

    此前按 50000 / 10 = 5000 手下单, 名义市值是目标的 10 倍。
    """
    expected = _CASH / 2 / (10.0 * _MULTIPLIER)
    assert expected == 500.0

    assert _orders(_run_futures(action)) == [(_FUT, "buy", "filled", expected, "")]


def test_futures_rebalance_at_target_weight_places_no_order() -> None:
    """已持有目标权重对应的手数时再调仓: 不下单(现值按名义市值算)."""

    class _Twice(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0
            self.second: Any = "unset"

        def on_bar(self, bar: Bar) -> None:
            if self.n == 0:
                self.rebalance_weights({_FUT: 0.5})
            elif self.n == 2:
                self.second = self.rebalance_weights({_FUT: 0.5})
            self.n += 1

    config = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=_CASH, commission_rate=0.0),
        instruments_config=[
            aq.InstrumentConfig(
                symbol=_FUT,
                asset_type="FUTURES",
                multiplier=_MULTIPLIER,
                margin_ratio=0.1,
            )
        ],
        china_futures=aq.ChinaFuturesConfig(),
    )
    result = aq.run_backtest(
        data={_FUT: _daily([10.0] * 5)},
        strategy=_Twice,
        symbols=_FUT,
        freq="1d",
        fill_policy=aq.NextOpen(),
        config=config,
        show_progress=False,
    )

    strategy = result.strategy
    assert strategy is not None
    assert strategy.second == []
    assert _orders(result) == [(_FUT, "buy", "filled", 500.0, "")]


def test_futures_rebalance_tolerance_compares_notional_value() -> None:
    """价格涨 4%、容差 5%: 现值按名义市值算, 在容差内不调仓.

    此前现值少乘了合约乘数, 被当成远低于目标而去下单。
    """

    class _Drift(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0
            self.second: Any = "unset"

        def on_bar(self, bar: Bar) -> None:
            if self.n == 0:
                self.rebalance_weights({_FUT: 0.5})
            elif self.n == 3:
                self.second = self.rebalance_weights(
                    {_FUT: 0.5}, rebalance_tolerance=0.05
                )
            self.n += 1

    config = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=_CASH, commission_rate=0.0),
        instruments_config=[
            aq.InstrumentConfig(
                symbol=_FUT,
                asset_type="FUTURES",
                multiplier=_MULTIPLIER,
                margin_ratio=0.1,
            )
        ],
        china_futures=aq.ChinaFuturesConfig(),
    )
    result = aq.run_backtest(
        data={_FUT: _daily([10.0, 10.0, 10.4, 10.4, 10.4, 10.4])},
        strategy=_Drift,
        symbols=_FUT,
        freq="1d",
        fill_policy=aq.NextOpen(),
        config=config,
        show_progress=False,
    )

    strategy = result.strategy
    assert strategy is not None
    assert strategy.second == []
    assert len(_orders(result)) == 1
