"""place_bracket 的止损止盈子单在各成交模式下的提交与触发."""

from typing import Any, Callable, List, Tuple

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
