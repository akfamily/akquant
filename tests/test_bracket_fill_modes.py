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


def _position(result: Any, symbol: str = _SYMBOL) -> float:
    return float(result.engine.portfolio.positions.get(symbol, 0.0))


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


def _two_days(first: List[float], second: List[float]) -> pd.DataFrame:
    """连续两个交易日的 1 分钟线(2024-01-02、2024-01-03, 各从 09:31 起)."""
    frames = []
    for day, prices in (("2024-01-02", first), ("2024-01-03", second)):
        frame = _minutes(prices)
        frame.index = pd.date_range(
            f"{day} 09:31", periods=len(prices), freq="1min"
        ).tz_localize(_TZ)
        frames.append(frame)
    return pd.concat(frames)


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_t1_bracket_stop_hit_same_day_stays_working(policy: Any) -> None:
    """T+1 分钟线, 进场当日触及止损: 当日不可卖, 止损保持挂着、不被拒.

    此前止损在成交时以“可用持仓不足”被拒且不补挂, 持仓只剩止盈一侧。
    """
    result = _run(
        _minutes([10.0, 10.0, 8.5, 8.0, 8.0]), policy(), t_plus_one=True, freq="1min"
    )

    assert _orders(result) == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "new", 0.0),
        ("sell", "limit", "new", 0.0),
    ]
    assert _reject_reasons(result) == []
    assert _position(result) == 100.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_t1_bracket_stop_fills_next_day_when_still_triggered(policy: Any) -> None:
    """T+1: 进场当日触及的止损挂到次日, 次日可卖且价格仍在止损价下方时成交."""
    result = _run(
        _two_days([10.0, 10.0, 8.5, 8.0, 8.0], [8.2, 8.2, 8.2]),
        policy(),
        t_plus_one=True,
        freq="1min",
    )

    assert _orders(result) == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "filled", 100.0),
        ("sell", "limit", "cancelled", 0.0),
    ]
    assert _reject_reasons(result) == []
    assert _fills(result)[-1] == ("sell", 100.0, 8.2)
    assert str(result.executions_df.iloc[-1]["timestamp"]).startswith("2024-01-03")
    assert _position(result) == 0.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_t1_bracket_stop_not_refired_when_price_recovers_next_day(policy: Any) -> None:
    """T+1: 次日价格回到止损价上方, 止损不成交(按价格重新判断), 保护仍在."""
    result = _run(
        _two_days([10.0, 10.0, 8.5, 8.0, 8.0], [10.0, 10.0, 10.0]),
        policy(),
        t_plus_one=True,
        freq="1min",
    )

    assert _orders(result) == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "new", 0.0),
        ("sell", "limit", "new", 0.0),
    ]
    assert _reject_reasons(result) == []
    assert _position(result) == 100.0


class _BracketThenFlatten(_Bracket):
    """第一根 place_bracket, 第三根 close_position."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.returned: List[Any] = []

    def on_bar(self, bar: Bar) -> None:
        """第三根平仓."""
        super().on_bar(bar)
        if self.n == 2:
            self.returned.append(self.close_position(bar.symbol))
        self.n += 1


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_t1_close_position_on_entry_day_keeps_bracket_exits(policy: Any) -> None:
    """T+1 进场当日 close_position: 当日卖不掉, 不撤止损止盈、不下会被拒的平仓单.

    此前子单先被撤掉、平仓单随后因当日不可卖被拒, 持仓隔夜没有任何保护。
    """
    result = aq.run_backtest(
        data={_SYMBOL: _minutes([10.0] * 6)},
        strategy=_BracketThenFlatten,
        symbols=_SYMBOL,
        initial_cash=100000.0,
        freq="1min",
        fill_policy=policy(),
        t_plus_one=True,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        show_progress=False,
    )

    strategy = result.strategy
    assert strategy is not None
    assert strategy.returned == [None]
    assert _orders(result) == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "new", 0.0),
        ("sell", "limit", "new", 0.0),
    ]
    assert _reject_reasons(result) == []
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
    symbol = _SYMBOL

    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.returns: List[Any] = []
        self.seen: Dict[str, float] = {}
        self.min_position = 0.0
        self.receipts: List[Any] = []

    def on_bar(self, bar: Bar) -> None:
        """按序号执行(多品种时只在 symbol 的 K 线上执行)."""
        if bar.symbol != self.symbol:
            return
        ctx = self.ctx
        assert ctx is not None
        for probe in self.probes.get(self.n, []):
            if probe == "projected":
                self.seen["projected"] = ctx.get_projected_position(self.symbol)
            elif probe == "closable":
                self.seen["closable"] = ctx.get_closable_position(self.symbol)
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
        self.min_position = min(self.min_position, self.get_position(self.symbol))
        self.n += 1


def _run_script(
    script: Dict[int, List[Any]],
    policy: Any,
    *,
    probes: Optional[Dict[int, List[str]]] = None,
    prices: Optional[List[float]] = None,
    base: type = _Scripted,
    extra_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Any:
    strategy_cls = type(
        "_ScriptedRun", (base,), {"script": script, "probes": probes or {}}
    )
    data = {_SYMBOL: _daily(prices or _HOLD), **(extra_data or {})}
    return aq.run_backtest(
        data=data,
        strategy=strategy_cls,
        symbols=list(data),
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
def test_manual_sell_all_with_bracket_exits_fills_and_cancels_them(
    policy: Any,
) -> None:
    """挂着 bracket 子单时 sell(100): 两种成交模式都成交, 持仓归零后子单自动撤销.

    子单只减仓, 不占用可卖数量。此前 CurrentClose 下卖单在提交时被拒
    (最早在途卖出算成 200, 后来算成 100)。持仓始终不为负。
    """
    result = _run_script(
        {0: [_BRACKET_100], 2: [("sell", (_SYMBOL, 100), {})]}, policy()
    )

    rows = _order_rows(result)
    assert [row[:4] for row in rows] == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "cancelled", 0.0),
        ("sell", "limit", "cancelled", 0.0),
        ("sell", "market", "filled", 100.0),
    ]
    assert _position(result) == 0.0
    assert result.strategy.min_position >= 0.0


def test_plain_resting_sell_still_reserves_available_position() -> None:
    """不变性: 普通在途卖单照旧占用可卖数量(CurrentClose 下再卖同样数量被拒)."""
    script: Dict[int, List[Any]] = {
        0: [("buy", (_SYMBOL, 100), {})],
        1: [("sell", (_SYMBOL, 100), {"price": 13.0})],
        2: [("sell", (_SYMBOL, 100), {})],
    }
    result = _run_script(script, aq.CurrentClose())

    last = _order_rows(result)[-1]
    assert last[:4] == ("sell", "market", "rejected", 0.0)
    assert "Available: 100, Pending Sell: 100, Required: 100" in last[4]


_FLATTEN_ACTIONS: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = [
    ("close_position", (_SYMBOL,), {}),
    ("order_target", (_SYMBOL, 0), {}),
    ("order_target_value", (_SYMBOL, 0), {}),
    ("order_target_percent", (_SYMBOL, 0.0), {}),
]
_FLATTEN_IDS = [
    "close_position",
    "order_target",
    "order_target_value",
    "order_target_percent",
]


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
@pytest.mark.parametrize("action", _FLATTEN_ACTIONS, ids=_FLATTEN_IDS)
def test_flatten_with_bracket_exits_cancels_them_and_sells(
    action: Any, policy: Any
) -> None:
    """挂着 bracket 子单时平到 0: 先撤两张子单, 再卖出 100, 期末持仓 0.

    投影持仓(调用前)仍是 0(子单按一组算一次); 撤单在平仓单过提交时的风控
    之前生效, CurrentClose 下平仓单不因“在途卖出 100”被拒。
    """
    result = _run_script(
        {0: [_BRACKET_100], 2: [action]}, policy(), probes={2: ["projected"]}
    )

    assert result.strategy.seen["projected"] == 0.0
    assert result.strategy.returns[-1] is not None
    rows = _order_rows(result)
    assert [row[:4] for row in rows] == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "cancelled", 0.0),
        ("sell", "limit", "cancelled", 0.0),
        ("sell", "market", "filled", 100.0),
    ]
    assert not [row for row in rows if row[2] == "rejected"]
    assert _position(result) == 0.0
    assert result.strategy.min_position >= 0.0


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
def test_close_position_beyond_bracket_exits_sells_everything(policy: Any) -> None:
    """持仓 200(其中 100 来自 bracket): close_position 撤掉子单, 卖出 200."""
    script: Dict[int, List[Any]] = {
        0: [("buy", (_SYMBOL, 100), {})],
        1: [_BRACKET_100],
        3: [("close_position", (_SYMBOL,), {})],
    }
    result = _run_script(script, policy(), probes={3: ["projected"]})

    # 调用前的投影持仓仍按子单在途计: 200 - 100
    assert result.strategy.seen["projected"] == 100.0
    rows = _order_rows(result)
    assert [row[:3] for row in rows[2:4]] == [
        ("sell", "stopmarket", "cancelled"),
        ("sell", "limit", "cancelled"),
    ]
    assert rows[-1][:4] == ("sell", "market", "filled", 200.0)
    assert not [row for row in rows if row[2] == "rejected"]
    assert _position(result) == 0.0


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


# ---------------------------------------------------------------------------
# 多策略槽位: 各自 place_oco 的组互不合并、互不误撤
# ---------------------------------------------------------------------------


class _SlotOco(Strategy):
    """第 1 根买 100; 第 2 根挂限价卖 + 止损卖并 place_oco; 第 3 根可选再卖 100."""

    limit_price = 13.0
    stop_price = 9.0
    group_id: Optional[str] = None
    sell_again = False

    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.legs: List[str] = []
        self.seen: Dict[str, float] = {}

    def on_bar(self, bar: Bar) -> None:
        """按序号下单."""
        if self.n == 0:
            self.buy(_SYMBOL, 100)
        elif self.n == 1:
            first = self.sell(_SYMBOL, 100, price=self.limit_price)
            second = self.sell(_SYMBOL, 100, trigger_price=self.stop_price)
            self.legs = [str(getattr(r, "primary", r)) for r in (first, second)]
            self.place_oco(first, second, group_id=self.group_id)
        elif self.n == 2:
            ctx = self.ctx
            assert ctx is not None
            self.seen["projected"] = ctx.get_projected_position(_SYMBOL)
            if self.sell_again:
                self.sell(_SYMBOL, 100)
        self.n += 1


def _run_slots(
    first: _SlotOco, second: _SlotOco, prices: List[float], policy: Any
) -> Any:
    return aq.run_backtest(
        data={_SYMBOL: _daily(prices)},
        strategy=first,
        strategies_by_slot={"slot_b": second},
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


def _status_by_id(result: Any) -> Dict[str, Tuple[str, str]]:
    return {
        str(row["id"]): (str(row["status"]), str(row["reject_reason"]))
        for row in result.orders_df.to_dict("records")
    }


@pytest.mark.parametrize("group_id", [None, "g"], ids=["default_id", "same_id"])
def test_slot_oco_groups_are_counted_separately(group_id: Optional[str]) -> None:
    """两个槽位各挂一组 OCO(默认组号都是 oco-1, 或都传 "g"): 按两组计算.

    持仓 200、两组各在途 100: 投影持仓 0; CurrentClose 下主槽位再卖 100 被拒,
    在途卖出 200。
    """
    a = type("_A", (_SlotOco,), {"group_id": group_id, "sell_again": True})()
    b = type("_B", (_SlotOco,), {"group_id": group_id})()

    result = _run_slots(a, b, [10.0] * 5, aq.CurrentClose())

    assert a.seen["projected"] == 0.0
    assert b.seen["projected"] == 0.0
    manual = [
        row
        for row in result.orders_df.to_dict("records")
        if str(row["order_type"]) == "market" and str(row["side"]) == "sell"
    ]
    assert len(manual) == 1
    assert str(manual[0]["status"]) == "rejected"
    assert "Available: 200, Pending Sell: 200, Required: 100" in str(
        manual[0]["reject_reason"]
    )
    assert _position(result) == 200.0


@pytest.mark.parametrize("group_id", [None, "g"], ids=["default_id", "same_id"])
def test_slot_oco_fill_cancels_only_its_own_peer(group_id: Optional[str]) -> None:
    """主槽位的限价腿成交: 只撤它自己的止损腿, slot_b 的两张单照旧挂着."""
    a = type("_A", (_SlotOco,), {"group_id": group_id})()
    b = type("_B", (_SlotOco,), {"group_id": group_id, "limit_price": 15.0})()

    result = _run_slots(a, b, [10.0, 10.0, 10.0, 13.5, 13.5], aq.NextOpen())

    status = _status_by_id(result)
    assert status[a.legs[0]][0] == "filled"
    assert status[a.legs[1]][0] == "cancelled"
    assert status[b.legs[0]][0] == "new"
    assert status[b.legs[1]][0] == "new"
    assert _position(result) == 100.0


def test_single_strategy_oco_graph_id_is_unchanged() -> None:
    """单策略(所属 _default)下组员的 graph_id 仍是 oco:<组号>."""
    a = _SlotOco()
    result = aq.run_backtest(
        data={_SYMBOL: _daily([10.0] * 4)},
        strategy=a,
        symbols=_SYMBOL,
        initial_cash=100000.0,
        freq="1d",
        fill_policy=aq.NextOpen(),
        t_plus_one=False,
        commission_rate=0.0,
        show_progress=False,
    )

    graph = {order.id: order.graph_id for order in result.orders}
    assert [graph[leg] for leg in a.legs] == ["oco:oco-1", "oco:oco-1"]


# ---------------------------------------------------------------------------
# bracket 止损止盈只减仓: 持仓了结后自动撤销, 成交时不超出持仓
# ---------------------------------------------------------------------------

_FUT = "RB2310"
_POSITION_CLOSED = "position closed"


class _FuturesScripted(_Scripted):
    """同 _Scripted, 品种为期货 RB2310."""

    symbol = _FUT


class _Recording(_Scripted):
    """同 _Scripted, 另按到达顺序记下 on_order / on_trade 回调."""

    def __init__(self) -> None:
        super().__init__()
        self.events: List[Tuple[str, str, str]] = []

    def on_order(self, order: Any) -> None:
        """记 (order, 类型, 状态)."""
        self.events.append(
            (
                "order",
                str(order.order_type).rsplit(".", 1)[-1],
                str(order.status).rsplit(".", 1)[-1],
            )
        )

    def on_trade(self, trade: Any) -> None:
        """记 (trade, 方向, 数量)."""
        self.events.append(
            ("trade", str(trade.side).rsplit(".", 1)[-1], str(float(trade.quantity)))
        )


def _bracket(
    symbol: str = _SYMBOL, quantity: float = 100, stop: float = 9.0, take: float = 13.0
) -> Tuple[str, Tuple[Any, ...], Dict[str, Any]]:
    return (
        "place_bracket",
        (symbol, quantity),
        {"stop_trigger_price": stop, "take_profit_price": take},
    )


def _run_futures_script(
    script: Dict[int, List[Any]],
    policy: Any,
    *,
    probes: Optional[Dict[int, List[str]]] = None,
    prices: Optional[List[float]] = None,
) -> Any:
    """期货合约配置同 test_position_effect_flip(乘数 10、保证金 10%)."""
    strategy_cls = type(
        "_FuturesRun", (_FuturesScripted,), {"script": script, "probes": probes or {}}
    )
    config = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=100000.0, commission_rate=0.0),
        instruments_config=[
            aq.InstrumentConfig(
                symbol=_FUT, asset_type="FUTURES", multiplier=10.0, margin_ratio=0.1
            )
        ],
        china_futures=aq.ChinaFuturesConfig(),
    )
    return aq.run_backtest(
        data={_FUT: _daily(prices or _HOLD)},
        strategy=strategy_cls,
        symbols=_FUT,
        freq="1d",
        fill_policy=policy,
        config=config,
        show_progress=False,
    )


def _fill_rows(result: Any) -> List[Tuple[str, float, float]]:
    return [
        (str(row["side"]), float(row["quantity"]), float(row["price"]))
        for row in result.executions_df.to_dict("records")
    ]


def _bracket_exit_rows(result: Any) -> List[Tuple[str, str, float, str]]:
    """Bracket 子单 (类型, 状态, 成交量, 原因), 按提交顺序."""
    return [
        (row[1], row[2], row[3], row[4])
        for row in _order_rows(result)
        if row[1] in ("stopmarket", "limit")
    ]


# 第 0 根进场, 第 2 根手动卖出, 第 4 根起价格跌破止损价 9
_STOP_LATER = [10.0, 10.0, 10.0, 10.0, 8.5, 8.0, 8.0]


def test_manual_sell_all_cancels_bracket_exits() -> None:
    """A1 股票 NextOpen: 手动卖完, 两张子单撤销; 之后触及止损价没有成交、没有拒单."""
    result = _run_script(
        {0: [_BRACKET_100], 2: [("sell", (_SYMBOL, 100), {})]},
        aq.NextOpen(),
        prices=_STOP_LATER,
        base=_Recording,
    )

    exits = _bracket_exit_rows(result)
    assert [row[:3] for row in exits] == [
        ("stopmarket", "cancelled", 0.0),
        ("limit", "cancelled", 0.0),
    ]
    assert all(_POSITION_CLOSED in row[3] for row in exits)
    assert _fill_rows(result) == [("buy", 100.0, 10.0), ("sell", 100.0, 10.0)]
    assert not [row for row in _order_rows(result) if row[2] == "rejected"]
    assert _position(result) == 0.0
    # 回调顺序: 撤单与卖出成交在同一批回调里派发(下一根 K 线的 on_bar 之前);
    # 订单账本先按挂单顺序派发 on_order(两张子单在卖单之前挂出), 再派发 on_trade
    assert result.strategy.events[-5:] == [
        ("order", "Market", "New"),
        ("order", "StopMarket", "Cancelled"),
        ("order", "Limit", "Cancelled"),
        ("order", "Market", "Filled"),
        ("trade", "Sell", "100.0"),
    ]


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_futures_manual_close_cancels_bracket_exits(policy: Any) -> None:
    """A1 期货: 进场 100 手、手动平仓, 子单撤销; 之后触及止损价不开空(起点为 -100)."""
    result = _run_futures_script(
        {0: [_bracket(_FUT)], 2: [("sell", (_FUT, 100), {})]},
        policy(),
        prices=_STOP_LATER,
    )

    assert [row[:2] for row in _bracket_exit_rows(result)] == [
        ("stopmarket", "cancelled"),
        ("limit", "cancelled"),
    ]
    assert _fill_rows(result) == [("buy", 100.0, 10.0), ("sell", 100.0, 10.0)]
    assert _position(result, _FUT) == 0.0
    assert result.strategy.min_position >= 0.0


def test_rebuy_after_sell_all_is_not_hit_by_old_stop() -> None:
    """A1 卖完后再买入(不带 bracket): 旧止损价被触及时新持仓不受影响, 之后能全部卖出.

    起点上旧止损单还挂着, 新持仓的投影持仓为 0, 第 6 根按旧止损把新持仓卖掉。
    """
    script: Dict[int, List[Any]] = {
        0: [_BRACKET_100],
        2: [("sell", (_SYMBOL, 100), {})],
        3: [("buy", (_SYMBOL, 100), {})],
        7: [("sell", (_SYMBOL, 100), {})],
    }
    prices = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 8.5, 8.0, 8.0]
    result = _run_script(
        script, aq.NextOpen(), probes={5: ["projected"]}, prices=prices
    )

    assert result.strategy.seen["projected"] == 100.0
    assert _fill_rows(result) == [
        ("buy", 100.0, 10.0),
        ("sell", 100.0, 10.0),
        ("buy", 100.0, 10.0),
        ("sell", 100.0, 8.0),
    ]
    assert not [row for row in _order_rows(result) if row[2] == "rejected"]
    assert _position(result) == 0.0


def test_stock_partial_reduce_then_stop_fills_the_rest() -> None:
    """A2 股票 NextOpen: 持仓 100、手动卖 50, 止损触发后按持仓成交 50, 止盈撤销.

    起点上止损单整单被拒(可用 50 < 100), 余下 50 没有保护。
    """
    result = _run_script(
        {0: [_BRACKET_100], 2: [("sell", (_SYMBOL, 50), {})]},
        aq.NextOpen(),
        prices=_STOP_LATER,
    )

    assert [row[:3] for row in _bracket_exit_rows(result)] == [
        ("stopmarket", "filled", 50.0),
        ("limit", "cancelled", 0.0),
    ]
    assert _fill_rows(result) == [
        ("buy", 100.0, 10.0),
        ("sell", 50.0, 10.0),
        ("sell", 50.0, 8.5),
    ]
    assert not [row for row in _order_rows(result) if row[2] == "rejected"]
    assert _position(result) == 0.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_futures_partial_reduce_then_stop_fills_the_rest(policy: Any) -> None:
    """A2 期货: 持仓 100、手动平 50, 止损触发后成交 50, 不开空(起点为 -50)."""
    result = _run_futures_script(
        {0: [_bracket(_FUT)], 2: [("sell", (_FUT, 50), {})]},
        policy(),
        prices=_STOP_LATER,
    )

    assert [row[:3] for row in _bracket_exit_rows(result)] == [
        ("stopmarket", "filled", 50.0),
        ("limit", "cancelled", 0.0),
    ]
    assert _fill_rows(result) == [
        ("buy", 100.0, 10.0),
        ("sell", 50.0, 10.0),
        ("sell", 50.0, 8.5),
    ]
    assert _position(result, _FUT) == 0.0
    assert result.strategy.min_position >= 0.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_two_brackets_reduce_does_not_cancel_the_other_group(policy: Any) -> None:
    """A1 两组 bracket 各 100: 第一组止损成交后第二组照常有效, 全部了结后无残留."""
    script: Dict[int, List[Any]] = {
        0: [_bracket(stop=9.0, take=13.0)],
        1: [_bracket(stop=8.0, take=15.0)],
    }
    prices = [10.0, 10.0, 10.0, 8.5, 8.5, 15.5, 15.5]
    result = _run_script(script, policy(), prices=prices)

    assert [row[:3] for row in _bracket_exit_rows(result)] == [
        ("stopmarket", "filled", 100.0),
        ("limit", "cancelled", 0.0),
        ("stopmarket", "cancelled", 0.0),
        ("limit", "filled", 100.0),
    ]
    # 第二组的止损不是因持仓了结撤销, 而是被它自己的止盈按 OCO 撤销
    assert not [row for row in _bracket_exit_rows(result) if _POSITION_CLOSED in row[3]]
    assert not [row for row in _order_rows(result) if row[2] in ("new", "rejected")]
    assert _position(result) == 0.0


@pytest.mark.parametrize(
    ("futures", "policy"),
    [(False, aq.NextOpen), (True, aq.NextOpen), (True, aq.CurrentClose)],
    ids=["stock-next_open", "futures-next_open", "futures-current_close"],
)
def test_second_stop_in_same_match_finds_no_position_and_is_cancelled(
    futures: bool, policy: Any
) -> None:
    """A2 防线: 两组 bracket 各 100、手动卖 100 后两张止损在同一根 K 线触发.

    第一张成交 100 后持仓为 0, 第二张不成交、撤销(股票起点上被拒, 期货起点上
    开出 -100), 它的止盈随后由 A1 撤销。
    """
    symbol = _FUT if futures else _SYMBOL
    script: Dict[int, List[Any]] = {
        0: [_bracket(symbol)],
        1: [_bracket(symbol)],
        3: [("sell", (symbol, 100), {})],
    }
    prices = [10.0, 10.0, 10.0, 10.0, 10.0, 8.5, 8.0]
    if futures:
        result = _run_futures_script(script, policy(), prices=prices)
    else:
        result = _run_script(script, policy(), prices=prices)

    exits = _bracket_exit_rows(result)
    assert [row[:3] for row in exits] == [
        ("stopmarket", "filled", 100.0),
        ("limit", "cancelled", 0.0),
        ("stopmarket", "cancelled", 0.0),
        ("limit", "cancelled", 0.0),
    ]
    assert [_POSITION_CLOSED in row[3] for row in exits] == [False, False, True, True]
    assert _fill_rows(result)[-1] == ("sell", 100.0, 8.5)
    assert not [row for row in _order_rows(result) if row[2] == "rejected"]
    assert _position(result, symbol) == 0.0
    assert result.strategy.min_position >= 0.0


def test_slot_b_sell_flattening_the_account_cancels_slot_a_exits() -> None:
    """多槽位: 持仓按账户合计; slot_b 卖出使品种持仓归零, slot_a 的子单一并撤销."""

    class _A(_SlotOco):
        def on_bar(self, bar: Bar) -> None:
            if self.n == 0:
                self.place_bracket(
                    _SYMBOL, 100, stop_trigger_price=9.0, take_profit_price=13.0
                )
            self.n += 1

    class _B(_SlotOco):
        def on_bar(self, bar: Bar) -> None:
            if self.n == 2:
                self.sell(_SYMBOL, 100)
            self.n += 1

    result = _run_slots(_A(), _B(), _STOP_LATER, aq.NextOpen())

    assert [row[:2] for row in _bracket_exit_rows(result)] == [
        ("stopmarket", "cancelled"),
        ("limit", "cancelled"),
    ]
    assert _position(result) == 0.0


# ---------------------------------------------------------------------------
# 目标仓位为 0 的调用先撤该品种的 bracket 止损止盈
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_order_target_nonzero_ignores_bracket_exits(policy: Any) -> None:
    """目标不为 0: 子单不撤、也不计作“将要卖出”, 持仓 100、目标 50 → 卖出 50.

    此前按含子单的投影持仓(0)算差额, 买入 50, 持仓变成 150。之后止损触发时
    按剩下的 50 成交, 止盈撤销。
    """
    result = _run_script(
        {0: [_BRACKET_100], 2: [("order_target", (_SYMBOL, 50), {})]},
        policy(),
        prices=_STOP_LATER,
    )

    assert [row[:4] for row in _order_rows(result)] == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "filled", 50.0),
        ("sell", "limit", "cancelled", 0.0),
        ("sell", "market", "filled", 50.0),
    ]
    assert not [row for row in _order_rows(result) if row[2] == "rejected"]
    assert _position(result) == 0.0
    assert result.strategy.min_position >= 0.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_order_target_above_position_buys_only_the_difference(policy: Any) -> None:
    """持仓 100、子单各 100、目标 150: 买入 50(此前按投影持仓 0 买入 150)."""
    result = _run_script(
        {0: [_BRACKET_100], 2: [("order_target", (_SYMBOL, 150), {})]}, policy()
    )

    assert [row[:4] for row in _order_rows(result)] == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "new", 0.0),
        ("sell", "limit", "new", 0.0),
        ("buy", "market", "filled", 50.0),
    ]
    assert _position(result) == 150.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_order_target_same_as_position_places_no_order(policy: Any) -> None:
    """持仓 100、子单各 100、目标 100: 不下单(此前买入 100)."""
    result = _run_script(
        {0: [_BRACKET_100], 2: [("order_target", (_SYMBOL, 100), {})]}, policy()
    )

    assert result.strategy.returns[-1] is None
    assert len(_order_rows(result)) == 3
    assert _position(result) == 100.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_close_position_keeps_plain_resting_sell(policy: Any) -> None:
    """普通在途卖单(非 bracket)挂着时 close_position: 不撤它, 投影持仓为 0 不下单."""
    script: Dict[int, List[Any]] = {
        0: [("buy", (_SYMBOL, 100), {})],
        1: [("sell", (_SYMBOL, 100), {"price": 13.0})],
        2: [("close_position", (_SYMBOL,), {})],
    }
    result = _run_script(script, policy())

    assert result.strategy.returns[-1] is None
    assert [row[:4] for row in _order_rows(result)] == [
        ("buy", "market", "filled", 100.0),
        ("sell", "limit", "new", 0.0),
    ]
    assert _position(result) == 100.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_close_position_leaves_other_symbol_bracket_exits(policy: Any) -> None:
    """多品种: 对 A 调 close_position, B 上的 bracket 子单不受影响."""
    other = "600000.SH"

    class _TwoSymbols(_Scripted):
        def on_bar(self, bar: Bar) -> None:
            if bar.symbol == other and self.n == 1:
                self.place_bracket(
                    other, 100, stop_trigger_price=9.0, take_profit_price=13.0
                )
            super().on_bar(bar)

    script: Dict[int, List[Any]] = {
        0: [_BRACKET_100],
        2: [("close_position", (_SYMBOL,), {})],
    }
    result = _run_script(
        script, policy(), base=_TwoSymbols, extra_data={other: _daily(_HOLD)}
    )

    by_symbol: Dict[str, List[Tuple[str, str]]] = {}
    for row in result.orders_df.sort_values("created_at", kind="stable").to_dict(
        "records"
    ):
        by_symbol.setdefault(str(row["symbol"]), []).append(
            (str(row["order_type"]), str(row["status"]))
        )
    assert by_symbol[_SYMBOL] == [
        ("market", "filled"),
        ("stopmarket", "cancelled"),
        ("limit", "cancelled"),
        ("market", "filled"),
    ]
    assert by_symbol[other] == [
        ("market", "filled"),
        ("stopmarket", "new"),
        ("limit", "new"),
    ]
    assert _position(result) == 0.0
    assert _position(result, other) == 100.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_futures_close_position_cancels_exits_and_never_opens_short(
    policy: Any,
) -> None:
    """期货 close_position: 平仓后持仓 0, 之后触及止损价不开反向仓."""
    result = _run_futures_script(
        {0: [_bracket(_FUT)], 2: [("close_position", (_FUT,), {})]},
        policy(),
        prices=_STOP_LATER,
    )

    assert [row[:2] for row in _bracket_exit_rows(result)] == [
        ("stopmarket", "cancelled"),
        ("limit", "cancelled"),
    ]
    assert _fill_rows(result) == [("buy", 100.0, 10.0), ("sell", 100.0, 10.0)]
    assert _position(result, _FUT) == 0.0
    assert result.strategy.min_position >= 0.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
@pytest.mark.parametrize(
    "action",
    [
        ("rebalance_positions", ({_SYMBOL: 0.0},), {}),
        ("rebalance_weights", ({_SYMBOL: 0.0},), {}),
    ],
    ids=["rebalance_positions", "rebalance_weights"],
)
def test_rebalance_to_zero_cancels_bracket_exits(action: Any, policy: Any) -> None:
    """调仓入口里目标为 0 的品种与 order_target(S, 0) 同一路径: 先撤子单再卖出."""
    result = _run_script({0: [_BRACKET_100], 2: [action]}, policy())

    assert [row[:4] for row in _order_rows(result)] == [
        ("buy", "market", "filled", 100.0),
        ("sell", "stopmarket", "cancelled", 0.0),
        ("sell", "limit", "cancelled", 0.0),
        ("sell", "market", "filled", 100.0),
    ]
    assert len(result.strategy.returns[-1]) == 1
    assert _position(result) == 0.0


def test_current_close_rebuy_after_close_position_sells_in_full() -> None:
    """CurrentClose: close_position 了结后再买入, 新持仓能全部卖出(在途卖出为 0)."""
    script: Dict[int, List[Any]] = {
        0: [_BRACKET_100],
        2: [("close_position", (_SYMBOL,), {})],
        3: [("buy", (_SYMBOL, 100), {})],
        4: [("sell", (_SYMBOL, 100), {})],
    }
    result = _run_script(script, aq.CurrentClose())

    rows = _order_rows(result)
    assert rows[-1][:4] == ("sell", "market", "filled", 100.0)
    assert not [row for row in rows if row[2] == "rejected"]
    assert _position(result) == 0.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_slot_close_position_cancels_only_its_own_bracket_exits(policy: Any) -> None:
    """多槽位: slot_a close_position 只撤自己的子单, slot_b 的子单照旧挂着."""

    class _Slot(_SlotOco):
        flatten = False

        def on_bar(self, bar: Bar) -> None:
            if self.n == 0:
                self.place_bracket(
                    _SYMBOL, 100, stop_trigger_price=9.0, take_profit_price=13.0
                )
            elif self.n == 2 and self.flatten:
                self.legs = [str(self.close_position(_SYMBOL))]
            self.n += 1

    a = type("_A", (_Slot,), {"flatten": True})()
    b = _Slot()
    result = _run_slots(a, b, [10.0] * 5, policy())

    owners: Dict[str, List[Tuple[str, str]]] = {}
    for row in result.orders_df.sort_values("created_at", kind="stable").to_dict(
        "records"
    ):
        owners.setdefault(str(row["owner_strategy_id"]), []).append(
            (str(row["order_type"]), str(row["status"]))
        )
    assert owners["_default"] == [
        ("market", "filled"),
        ("stopmarket", "cancelled"),
        ("limit", "cancelled"),
        ("market", "filled"),
    ]
    assert owners["slot_b"] == [
        ("market", "filled"),
        ("stopmarket", "new"),
        ("limit", "new"),
    ]
    assert _position(result) == 100.0


@pytest.mark.parametrize("policy", _POLICIES, ids=["next_open", "current_close"])
def test_bracket_entry_that_only_covers_a_short_leaves_no_exits(policy: Any) -> None:
    """A1 子单生成时已无持仓可平: 先开空 100, 再用 bracket 买入 100 把持仓平掉.

    两张子单随即撤销; 之后再买入 100, 旧止损价被触及时新持仓不受影响
    (此前子单一直挂着, 按旧止损价把新持仓卖掉)。
    """
    script: Dict[int, List[Any]] = {
        0: [("sell", (_FUT, 100), {})],
        2: [_bracket(_FUT)],
        4: [("buy", (_FUT, 100), {})],
    }
    prices = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 8.5, 8.0, 8.0]
    result = _run_futures_script(script, policy(), prices=prices)

    exits = _bracket_exit_rows(result)
    assert [row[:3] for row in exits] == [
        ("stopmarket", "cancelled", 0.0),
        ("limit", "cancelled", 0.0),
    ]
    assert all(_POSITION_CLOSED in row[3] for row in exits)
    assert _fill_rows(result) == [
        ("sell", 100.0, 10.0),
        ("buy", 100.0, 10.0),
        ("buy", 100.0, 10.0),
    ]
    assert _position(result, _FUT) == 100.0


# ---------------------------------------------------------------------------
# paper 实盘(run_live 模拟成交)与回测同一口径
# ---------------------------------------------------------------------------


def _run_paper(script: Dict[int, List[Any]], prices: List[float]) -> Any:
    """Replay 行情 + paper 成交, 期货 RB2310(不受 T+1 影响); 返回策略实例."""
    index = pd.date_range(
        "2024-01-02 09:31:00", periods=len(prices), freq="1min", tz="Asia/Shanghai"
    )
    bars = [Bar(int(ts.value), p, p, p, p, 1e6, _FUT) for ts, p in zip(index, prices)]
    strategy = type(
        "_PaperRun", (_FuturesScripted,), {"script": script, "probes": {}}
    )()
    strategy.seen_positions = []
    on_bar = strategy.on_bar

    def recording_on_bar(bar: Bar) -> None:
        strategy.seen_positions.append(strategy.get_position(_FUT))
        on_bar(bar)

    strategy.on_bar = recording_on_bar  # type: ignore[method-assign]
    aq.run_live(
        strategy_cls=strategy,
        instruments=[
            aq.Instrument(
                symbol=_FUT,
                asset_type=aq.AssetType.Futures,
                multiplier=10.0,
                margin_ratio=0.1,
                tick_size=0.01,
                lot_size=1,
                option_type=None,
                strike_price=None,
                expiry_date=None,
            )
        ],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars, "freq": "1min"},
        cash=1_000_000.0,
        show_progress=False,
        duration="60s",
    )
    return strategy


_PAPER_PRICES = [10.0, 10.0, 10.0, 10.0, 10.0, 8.5, 8.0, 8.0, 8.0]


def test_paper_manual_close_does_not_open_short_on_old_stop() -> None:
    """paper: 进场 2 手、手动平仓后触及止损价, 不开空仓(此前 -2)."""
    strategy = _run_paper(
        {0: [_bracket(_FUT, 2)], 3: [("sell", (_FUT, 2), {})]}, _PAPER_PRICES
    )

    assert max(strategy.seen_positions) == 2.0
    assert strategy.seen_positions[-1] == 0.0
    assert min(strategy.seen_positions) >= 0.0
    assert strategy.get_open_orders(_FUT) == []


def test_paper_close_position_cancels_bracket_exits() -> None:
    """paper: 挂着子单时 close_position 平掉持仓(此前买入 2 手, 持仓变 4)."""
    strategy = _run_paper(
        {0: [_bracket(_FUT, 2)], 3: [("close_position", (_FUT,), {})]},
        _PAPER_PRICES,
    )

    assert max(strategy.seen_positions) == 2.0
    assert strategy.seen_positions[-1] == 0.0
    assert min(strategy.seen_positions) >= 0.0
    assert strategy.get_open_orders(_FUT) == []


def test_place_oco_then_close_position_in_same_callback_does_not_buy_back() -> None:
    """同一回调里先 place_oco 再 close_position: 两张卖单按一组算, 不反向买入.

    此前组要等回调返回后才登记, 投影持仓算成 100 - 200 = -100, close_position
    下出买入 100, 持仓变成 200。手工 OCO 组不会被平仓调用撤掉, 所以这里不下单。
    """
    script: Dict[int, List[Any]] = {
        0: [("buy", (_SYMBOL, 100), {})],
        2: [
            ("sell", (_SYMBOL, 100), {"price": 13.0}),
            ("sell", (_SYMBOL, 100), {"trigger_price": 9.0}),
            ("oco", (), {}),
            ("close_position", (_SYMBOL,), {}),
        ],
    }
    result = _run_script(script, aq.NextOpen(), probes={3: ["projected"]})

    assert result.strategy.returns[-1] is None
    assert [row[:3] for row in _order_rows(result)] == [
        ("buy", "market", "filled"),
        ("sell", "limit", "new"),
        ("sell", "stopmarket", "new"),
    ]
    assert result.strategy.seen["projected"] == 0.0
    assert _position(result) == 100.0
