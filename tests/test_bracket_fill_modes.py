"""place_bracket 的止损止盈子单在各成交模式下的提交与触发."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy

_SYMBOL = "000001.SZ"
_TZ = "Asia/Shanghai"


def _daily(prices: List[float]) -> pd.DataFrame:
    """日线, 每根开高低收相同, 2024-01-02 起的交易日."""
    idx = pd.date_range("2024-01-02 15:00", periods=len(prices), freq="B").tz_localize(
        _TZ
    )
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [100000.0] * len(prices),
        },
        index=idx,
    )


def _minutes(prices: List[float]) -> pd.DataFrame:
    """同一交易日的 1 分钟线, 每根开高低收相同."""
    idx = pd.date_range(
        "2024-01-02 09:31", periods=len(prices), freq="1min"
    ).tz_localize(_TZ)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [100000.0] * len(prices),
        },
        index=idx,
    )


class _Bracket(Strategy):
    """第一根 K 线 place_bracket(100 股, 止损 9, 止盈 13)."""

    def __init__(self) -> None:
        super().__init__()
        self.placed = False

    def on_bar(self, bar: Bar) -> None:
        """只下一次."""
        if not self.placed:
            self.placed = True
            self.place_bracket(
                bar.symbol, 100, stop_trigger_price=9.0, take_profit_price=13.0
            )


def _run(data: pd.DataFrame, fill_policy: Any, *, t_plus_one: bool, freq: str) -> Any:
    return aq.run_backtest(
        data={_SYMBOL: data},
        strategy=_Bracket,
        symbols=_SYMBOL,
        initial_cash=100000.0,
        freq=freq,
        fill_policy=fill_policy,
        t_plus_one=t_plus_one,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        show_progress=False,
    )


_ROLE = {"market": 0, "stopmarket": 1, "limit": 2}


def _orders(result: Any) -> List[Tuple[str, str, str, float]]:
    """(方向, 类型, 状态, 成交量), 按 进场、止损、止盈 排列."""
    rows = [
        (
            str(row["side"]),
            str(row["order_type"]),
            str(row["status"]),
            float(row["filled_quantity"]),
        )
        for row in result.orders_df.to_dict("records")
    ]
    return sorted(rows, key=lambda row: _ROLE[row[1]])


def _fills(result: Any) -> List[Tuple[str, float, float]]:
    return [
        (str(row["side"]), float(row["quantity"]), float(row["price"]))
        for row in result.executions_df.to_dict("records")
    ]


def _reject_reasons(result: Any) -> List[str]:
    return [
        str(row["reject_reason"])
        for row in result.orders_df.to_dict("records")
        if str(row["status"]) == "rejected"
    ]


def _position(result: Any) -> float:
    return float(result.engine.portfolio.positions.get(_SYMBOL, 0.0))


_FLAT = [10.0, 10.0, 10.0, 10.0]
_STOP = [10.0, 8.5, 8.0, 8.0]
_TAKE = [10.0, 13.5, 14.0, 14.0]

_POLICIES: List[Callable[[], Any]] = [aq.NextOpen, aq.CurrentClose]


def test_current_close_bracket_exits_are_accepted_after_entry_fill() -> None:
    """CurrentClose: 进场当根成交后两张子单被接受并挂着."""
    result = _run(_daily(_FLAT), aq.CurrentClose(), t_plus_one=False, freq="1d")

    assert _orders(result) == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "new", 0.0),
        ("sell", "limit", "new", 0.0),
    ]
    assert _reject_reasons(result) == []
    assert _fills(result) == [("buy", 100.0, 10.0)]
    assert _position(result) == 100.0


def test_current_close_bracket_stop_fills_and_cancels_take_profit() -> None:
    """CurrentClose: 次日触及止损, 止损成交、止盈被撤、持仓归零."""
    result = _run(_daily(_STOP), aq.CurrentClose(), t_plus_one=False, freq="1d")

    assert _orders(result) == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "filled", 100.0),
        ("sell", "limit", "cancelled", 0.0),
    ]
    assert _fills(result) == [("buy", 100.0, 10.0), ("sell", 100.0, 8.5)]
    assert _position(result) == 0.0


def test_current_close_bracket_take_profit_fills_and_cancels_stop() -> None:
    """CurrentClose: 次日触及止盈, 止盈成交、止损被撤、持仓归零."""
    result = _run(_daily(_TAKE), aq.CurrentClose(), t_plus_one=False, freq="1d")

    assert _orders(result) == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "cancelled", 0.0),
        ("sell", "limit", "filled", 100.0),
    ]
    # 当根成交模式下限价卖单按限价成交(普通限价单同样如此), NextOpen 下按开盘价 13.5
    assert _fills(result) == [("buy", 100.0, 10.0), ("sell", 100.0, 13.0)]
    assert _position(result) == 0.0


def test_current_close_bracket_exits_ignore_the_entry_bar_range() -> None:
    """CurrentClose: 进场按收盘价成交, 这根 K 线进场前的最低价不触发止损.

    第 1 根开 10、低 8.5、收 10; 止损 9 只能由之后的 K 线触发。
    """
    idx = pd.date_range("2024-01-02 15:00", periods=3, freq="B").tz_localize(_TZ)
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 8.6],
            "high": [10.0, 10.0, 8.6],
            "low": [8.5, 10.0, 8.6],
            "close": [10.0, 10.0, 8.6],
            "volume": [100000.0] * 3,
        },
        index=idx,
    )
    result = _run(data, aq.CurrentClose(), t_plus_one=False, freq="1d")

    assert _orders(result) == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "filled", 100.0),
        ("sell", "limit", "cancelled", 0.0),
    ]
    fills = result.executions_df.to_dict("records")
    assert [(str(r["side"]), float(r["price"])) for r in fills] == [
        ("buy", 10.0),
        ("sell", 8.6),
    ]
    assert str(fills[1]["timestamp"]) == "2024-01-04 15:00:00+08:00"


@pytest.mark.parametrize("t_plus_one", [False, True])
@pytest.mark.parametrize("prices", [_FLAT, _STOP, _TAKE], ids=["flat", "stop", "take"])
def test_bracket_exit_statuses_match_between_next_open_and_current_close(
    prices: List[float], t_plus_one: bool
) -> None:
    """同一策略在 NextOpen 与 CurrentClose 下子单的最终状态一致(日线)."""
    by_policy = [
        _orders(_run(_daily(prices), policy(), t_plus_one=t_plus_one, freq="1d"))
        for policy in _POLICIES
    ]

    assert by_policy[0] == by_policy[1]
    assert not [row for row in by_policy[1] if row[2] == "rejected"]


def test_current_close_t1_bracket_exits_wait_for_next_day() -> None:
    """T+1 日线: 当日买入的持仓当日不可卖, 子单仍被接受, 次日触及止损成交."""
    result = _run(_daily(_STOP), aq.CurrentClose(), t_plus_one=True, freq="1d")

    assert _orders(result) == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "filled", 100.0),
        ("sell", "limit", "cancelled", 0.0),
    ]
    assert _fills(result) == [("buy", 100.0, 10.0), ("sell", 100.0, 8.5)]
    assert _position(result) == 0.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_t1_bracket_stop_hit_same_day_is_rejected_at_execution(policy: Any) -> None:
    """T+1 分钟线, 当日触及止损: 两种成交模式都在成交时因当日不可卖拒掉止损.

    子单在提交时被接受(不在提交时因当日不可卖被拒); 触发时按成交时的可用持仓
    校验拒掉止损, 止盈仍挂着(这是所有卖单共用的成交时规则, 本修复不改)。
    """
    result = _run(
        _minutes([10.0, 10.0, 8.5, 8.0, 8.0]), policy(), t_plus_one=True, freq="1min"
    )

    assert _orders(result) == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "rejected", 0.0),
        ("sell", "limit", "new", 0.0),
    ]
    reasons = _reject_reasons(result)
    assert len(reasons) == 1
    assert "Insufficient available position" in reasons[0]
    assert "at execution" in reasons[0]
    assert _position(result) == 100.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_t0_bracket_stop_hit_same_day_fills(policy: Any) -> None:
    """T+0 分钟线: 两种成交模式下止损都在当日触发成交、止盈被撤."""
    result = _run(
        _minutes([10.0, 10.0, 8.5, 8.0, 8.0]), policy(), t_plus_one=False, freq="1min"
    )

    assert _orders(result) == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "filled", 100.0),
        ("sell", "limit", "cancelled", 0.0),
    ]
    assert _position(result) == 0.0


class _LargeBracket(Strategy):
    """第一根 K 线 place_bracket(4000 股 @10, 止损 9, 止盈 13)."""

    def __init__(self) -> None:
        super().__init__()
        self.placed = False

    def on_bar(self, bar: Bar) -> None:
        """只下一次."""
        if not self.placed:
            self.placed = True
            self.place_bracket(
                bar.symbol, 4000, stop_trigger_price=9.0, take_profit_price=13.0
            )


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_bracket_exits_see_the_entry_fill_in_submission_risk_checks(
    policy: Any,
) -> None:
    """子单提交时风控看得到刚成交的进场持仓: 平仓不被持仓比例上限当成开空拒掉.

    进场 4000 股占权益 40%(上限 45%); 若看不到进场持仓, 止盈 4000 股 @13 会被
    算成开空 52% 而拒单。
    """
    result = aq.run_backtest(
        data={_SYMBOL: _daily(_FLAT)},
        strategy=_LargeBracket,
        symbols=_SYMBOL,
        initial_cash=100000.0,
        freq="1d",
        fill_policy=policy(),
        t_plus_one=False,
        commission_rate=0.0,
        risk_config={"max_position_pct": 0.45},
        show_progress=False,
    )

    assert _orders(result) == [
        ("buy", "market", "filled", 4000.0),
        ("sell", "stopmarket", "new", 0.0),
        ("sell", "limit", "new", 0.0),
    ]
    assert _reject_reasons(result) == []


class _SellThenBuy(Strategy):
    """不用 bracket: 第一根买 9000 股 A, 第二根同一根 K 线里卖 A 买 B."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def on_bar(self, bar: Bar) -> None:
        """只在 A 的回调里下单."""
        if bar.symbol != _SYMBOL:
            return
        if self.n == 0:
            self.buy(_SYMBOL, 9000)
        elif self.n == 1:
            self.sell(_SYMBOL, 9000)
            self.buy("600000.SH", 4000)
        self.n += 1


def test_current_close_sell_then_buy_without_bracket_is_unchanged() -> None:
    """不变性: 同一根 K 线里先卖后买(买入靠卖出回款), 成交逐笔与修复前相同."""
    result = aq.run_backtest(
        data={
            _SYMBOL: _daily([10.0, 11.0, 11.5, 12.0]),
            "600000.SH": _daily([20.0, 22.0, 23.0, 24.0]),
        },
        strategy=_SellThenBuy,
        symbols=[_SYMBOL, "600000.SH"],
        initial_cash=100000.0,
        freq="1d",
        fill_policy=aq.CurrentClose(),
        t_plus_one=False,
        commission_rate=0.0003,
        show_progress=False,
    )

    fills = [
        (
            str(row["symbol"]),
            str(row["side"]),
            float(row["quantity"]),
            float(row["price"]),
            round(float(row["commission"]), 6),
            str(row["timestamp"]),
        )
        for row in result.executions_df.to_dict("records")
    ]
    # 修复前(0.4.0)实跑记下的值
    assert fills == [
        (_SYMBOL, "buy", 9000.0, 10.0, 27.0, "2024-01-02 15:00:00+08:00"),
        (_SYMBOL, "sell", 9000.0, 11.0, 29.7, "2024-01-03 15:00:00+08:00"),
        ("600000.SH", "buy", 4000.0, 22.0, 26.4, "2024-01-03 15:00:00+08:00"),
    ]
    assert [str(s) for s in result.orders_df["status"]] == ["filled"] * 3
    assert result.engine is not None
    assert float(result.engine.portfolio.cash) == pytest.approx(20916.9)


# ---------------------------------------------------------------------------
# 同一 OCO 组的在途量按一组算一次(取剩余量最大的一条腿)
# ---------------------------------------------------------------------------

_HOLD = [10.0] * 6


class _Scripted(Strategy):
    """按 K 线序号执行脚本; 记下动作返回值、投影持仓、买入力与冻结资金."""

    script: Dict[int, List[Any]] = {}
    probes: Dict[int, List[str]] = {}

    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.returns: List[Any] = []
        self.seen: Dict[str, float] = {}
        self.min_position = 0.0
        self.receipts: List[Any] = []

    def on_bar(self, bar: Bar) -> None:
        """按序号执行."""
        ctx = self.ctx
        assert ctx is not None
        for probe in self.probes.get(self.n, []):
            if probe == "projected":
                self.seen["projected"] = ctx.get_projected_position(_SYMBOL)
            elif probe == "closable":
                self.seen["closable"] = ctx.get_closable_position(_SYMBOL)
            elif probe == "buying_power":
                self.seen["buying_power"] = float(self.buying_power)
            elif probe == "frozen_cash":
                self.seen["frozen_cash"] = float(self.get_account()["frozen_cash"])
        for name, args, kwargs in self.script.get(self.n, []):
            if name == "oco":
                self.place_oco(self.receipts[-2], self.receipts[-1])
                continue
            value = getattr(self, name)(*args, **kwargs)
            self.returns.append(value)
            if name in ("buy", "sell"):
                self.receipts.append(value)
        self.min_position = min(self.min_position, self.get_position(_SYMBOL))
        self.n += 1


def _run_script(
    script: Dict[int, List[Any]],
    policy: Any,
    *,
    probes: Optional[Dict[int, List[str]]] = None,
    prices: Optional[List[float]] = None,
) -> Any:
    strategy_cls = type(
        "_ScriptedRun", (_Scripted,), {"script": script, "probes": probes or {}}
    )
    return aq.run_backtest(
        data={_SYMBOL: _daily(prices or _HOLD)},
        strategy=strategy_cls,
        symbols=_SYMBOL,
        initial_cash=100000.0,
        freq="1d",
        fill_policy=policy,
        t_plus_one=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        show_progress=False,
    )


def _order_rows(result: Any) -> List[Tuple[str, str, str, float, str]]:
    """(方向, 类型, 状态, 成交量, 拒单原因) 按提交顺序."""
    return [
        (
            str(row["side"]),
            str(row["order_type"]),
            str(row["status"]),
            float(row["filled_quantity"]),
            str(row["reject_reason"]),
        )
        for row in result.orders_df.sort_values("created_at", kind="stable").to_dict(
            "records"
        )
    ]


_BRACKET_100 = (
    "place_bracket",
    (_SYMBOL, 100),
    {"stop_trigger_price": 9.0, "take_profit_price": 13.0},
)


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_manual_sell_with_bracket_exits_counts_the_group_once(policy: Any) -> None:
    """挂着 bracket 子单时 sell(100): 与挂一张 100 股普通卖单时相同.

    NextOpen 提交时不校验可用持仓, 卖单成交; CurrentClose 提交时按
    可用 100、在途 100 被拒(此前在途算成 200)。持仓始终不为负。
    """
    result = _run_script(
        {0: [_BRACKET_100], 2: [("sell", (_SYMBOL, 100), {})]}, policy()
    )

    rows = _order_rows(result)
    assert [row[:4] for row in rows[:3]] == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "new", 0.0),
        ("sell", "limit", "new", 0.0),
    ]
    manual = rows[3]
    if policy is aq.NextOpen:
        assert manual[:4] == ("sell", "market", "filled", 100.0)
        assert _position(result) == 0.0
    else:
        assert manual[:4] == ("sell", "market", "rejected", 0.0)
        assert "Available: 100, Pending Sell: 100, Required: 100" in manual[4]
        assert _position(result) == 100.0
    assert result.strategy.min_position >= 0.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
@pytest.mark.parametrize(
    "action",
    [
        ("close_position", (_SYMBOL,), {}),
        ("order_target", (_SYMBOL, 0), {}),
        ("order_target_percent", (_SYMBOL, 0.0), {}),
    ],
    ids=["close_position", "order_target", "order_target_percent"],
)
def test_flatten_with_bracket_exits_does_not_buy_back(action: Any, policy: Any) -> None:
    """挂着 bracket 子单时平到 0: 投影持仓为 0, 不下单, 不再反向买入."""
    result = _run_script(
        {0: [_BRACKET_100], 2: [action]}, policy(), probes={2: ["projected"]}
    )

    assert result.strategy.seen["projected"] == 0.0
    assert result.strategy.returns[-1] is None
    assert len(result.orders_df) == 3
    assert _position(result) == 100.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_extra_position_beyond_bracket_exits_can_be_sold(policy: Any) -> None:
    """持仓 200、子单各 100: CurrentClose 下卖 100 成交(可用 200, 在途 100)."""
    script: Dict[int, List[Any]] = {
        0: [("buy", (_SYMBOL, 100), {})],
        1: [_BRACKET_100],
        3: [("sell", (_SYMBOL, 100), {})],
    }
    result = _run_script(script, policy())

    rows = _order_rows(result)
    assert rows[-1][:4] == ("sell", "market", "filled", 100.0)
    assert not [row for row in rows if row[2] == "rejected"]
    assert _position(result) == 100.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_close_position_beyond_bracket_exits_sells_the_rest(policy: Any) -> None:
    """持仓 200、子单各 100: close_position 卖出 100(投影持仓 100)."""
    script: Dict[int, List[Any]] = {
        0: [("buy", (_SYMBOL, 100), {})],
        1: [_BRACKET_100],
        3: [("close_position", (_SYMBOL,), {})],
    }
    result = _run_script(script, policy(), probes={3: ["projected"]})

    assert result.strategy.seen["projected"] == 100.0
    rows = _order_rows(result)
    assert rows[-1][:4] == ("sell", "market", "filled", 100.0)
    assert _position(result) == 100.0


def test_manual_place_oco_reduces_projection_once() -> None:
    """手工 place_oco 的两张卖单: 登记后投影持仓只减一次, 成员带上 oco 组号."""
    script: Dict[int, List[Any]] = {
        0: [("buy", (_SYMBOL, 100), {})],
        1: [
            ("sell", (_SYMBOL, 100), {"price": 13.0}),
            ("sell", (_SYMBOL, 100), {"trigger_price": 9.0}),
            ("oco", (), {}),
        ],
        2: [("close_position", (_SYMBOL,), {})],
    }
    result = _run_script(script, aq.NextOpen(), probes={2: ["projected", "closable"]})

    strategy = result.strategy
    assert strategy.seen["projected"] == 0.0
    # auto 拆腿: 第一张定为平仓 100, 第二张(可平已为 0)定为开仓 100; 可平持仓
    # 只扣平仓腿, 与不登记 OCO 时相同
    assert strategy.seen["closable"] == 0.0
    assert strategy.returns[-1] is None
    legs = {str(getattr(r, "primary", r)) for r in strategy.receipts[1:]}
    # 组号写在引擎的订单上(result.orders、ctx.active_orders 可见)
    group_ids = {order.graph_id for order in result.orders if order.id in legs}
    assert len(group_ids) == 1
    assert str(next(iter(group_ids))).startswith("oco:")
    assert _position(result) == 100.0


def test_independent_sells_are_still_counted_separately() -> None:
    """不变性: 不是 OCO 的两张在途卖单各算一次.

    持仓 200、挂两张互不相干的 60 股限价卖单: 投影持仓与可平持仓都是 80;
    之后卖 80 刚好通过提交时校验(可用 200, 在途 120)。用 200 股是为了避开
    auto 拆腿: 持仓 100 时第二张 60 会被拆成平 40、开 20 两条腿。
    """
    script: Dict[int, List[Any]] = {
        0: [("buy", (_SYMBOL, 200), {})],
        1: [
            ("sell", (_SYMBOL, 60), {"price": 13.0}),
            ("sell", (_SYMBOL, 60), {"price": 14.0}),
        ],
        2: [("sell", (_SYMBOL, 80), {})],
        3: [("sell", (_SYMBOL, 1), {})],
    }
    result = _run_script(
        script, aq.CurrentClose(), probes={2: ["projected", "closable"]}
    )

    assert result.strategy.seen["projected"] == 80.0
    assert result.strategy.seen["closable"] == 80.0
    rows = _order_rows(result)
    assert rows[3][:4] == ("sell", "market", "filled", 80.0)
    # 余下 120 股都被两张在途卖单占着: 再卖 1 股被拒, 在途按 60 + 60 计
    assert rows[4][:4] == ("sell", "market", "rejected", 0.0)
    assert "Available: 120, Pending Sell: 120, Required: 1" in rows[4][4]


def test_buying_power_with_bracket_exits_matches_a_single_resting_sell() -> None:
    """资金投影: 挂着 bracket 子单时的买入力等于只挂一张等量普通止损卖单时的值."""
    bracket = _run_script(
        {0: [_BRACKET_100]}, aq.CurrentClose(), probes={2: ["buying_power"]}
    )
    single = _run_script(
        {
            0: [("buy", (_SYMBOL, 100), {})],
            1: [("sell", (_SYMBOL, 100), {"trigger_price": 9.0})],
        },
        aq.CurrentClose(),
        probes={2: ["buying_power"]},
    )

    assert bracket.strategy.seen["buying_power"] == pytest.approx(
        single.strategy.seen["buying_power"]
    )
    assert single.strategy.seen["buying_power"] == pytest.approx(99990.0)


def test_frozen_cash_counts_a_buy_side_oco_group_once() -> None:
    """两张互斥的限价买单组成 OCO: 冻结资金取占用较大的那条腿(100×9)."""
    script: Dict[int, List[Any]] = {
        0: [
            ("buy", (_SYMBOL, 100), {"price": 9.0}),
            ("buy", (_SYMBOL, 100), {"price": 8.0}),
            ("oco", (), {}),
        ],
    }
    result = _run_script(script, aq.NextOpen(), probes={1: ["frozen_cash"]})

    assert result.strategy.seen["frozen_cash"] == pytest.approx(900.0)
