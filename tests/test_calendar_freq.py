"""run_backtest(freq=) 的日历周期：1w / 1mon / 1q / 1y.

日历周期的 K 线区间是 ``(同品种上一根的时间戳, 本根的时间戳]``，该品种第一根取
``时间戳 − 名义长度``（周 7 天、月 31 天、季 92 天、年 366 天）。区间让引擎知道
这根 K 线覆盖了连续竞价：不传 ``freq`` 时 15:00 打戳的 K 线被判成收盘集合竞价。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import akquant as aq
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from akquant import Bar, IntParam, Strategy, Tick, run_backtest
from akquant.backtest.engine import _parse_bar_period
from akquant.checkpoint import save_checkpoint
from akquant.data import ParquetDataCatalog

TZ = "Asia/Shanghai"
A = "600000.SH"
B = "000001.SZ"
DAY = 86_400_000_000_000
WEEK = 7 * DAY

# 2025 年的周线，打在每周最后一个交易日 15:00。4/4（周五）清明节休市，那一周
# 在周四 4/3 收盘。
WEEK_ENDS = [
    "2025-03-07",
    "2025-03-14",
    "2025-03-21",
    "2025-03-28",
    "2025-04-03",
    "2025-04-11",
    "2025-04-18",
]
# 品种 B 缺 3/21 那一周（整周停牌）：3/28 那根的区间覆盖两周。
B_WEEK_ENDS = [d for d in WEEK_ENDS if d != "2025-03-21"]


def _ns(text: str) -> int:
    return int(pd.Timestamp(text, tz=TZ).value)


def _close_ns(day: str) -> int:
    return _ns(f"{day} 15:00")


def _ohlc(n: int, base: float = 10.0) -> Dict[str, List[float]]:
    """第 i 根：开 base+i，高 开+3，低 开-3，收 开+1."""
    opens = [base + i for i in range(n)]
    return {
        "open": opens,
        "high": [o + 3 for o in opens],
        "low": [o - 3 for o in opens],
        "close": [o + 1 for o in opens],
        "volume": [1e6] * n,
    }


def _frame(days: List[str], *, time: Optional[str] = "15:00") -> pd.DataFrame:
    stamps = [f"{d} {time}" if time else d for d in days]
    idx = pd.DatetimeIndex(pd.to_datetime(stamps)).tz_localize(TZ)
    return pd.DataFrame(_ohlc(len(days)), index=idx)


def _long(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for sym, df in frames.items():
        part = df.copy()
        part["symbol"] = sym
        part.index.name = "timestamp"
        parts.append(part.reset_index())
    return pd.concat(parts, ignore_index=True)


def _bars(frames: Dict[str, pd.DataFrame]) -> List[Bar]:
    out: List[Bar] = []
    for sym, df in frames.items():
        for ts, row in zip(pd.DatetimeIndex(df.index), df.to_dict("records")):
            out.append(
                Bar(
                    timestamp=int(ts.value),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    symbol=sym,
                )
            )
    return out


class _Probe(Strategy):
    """记录每根 bar 的 (品种, 时间戳, 区间起点, 时段, 是否开市, self.freq)."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: List[Tuple[str, int, Any, Any, bool, Any]] = []

    def on_bar(self, bar: Bar) -> None:
        self.seen.append(
            (
                str(bar.symbol),
                int(bar.timestamp),
                bar.interval_start,
                str(self.session_of()).split(".")[-1],
                self.is_market_open(),
                self.freq,
            )
        )


def _run(data: Any, strategy: Any = _Probe, **kwargs: Any) -> Any:
    kwargs.setdefault("symbols", [A, B])
    kwargs.setdefault("initial_cash", 1e6)
    kwargs.setdefault("commission_rate", 0.0)
    kwargs.setdefault("stamp_tax_rate", 0.0)
    kwargs.setdefault("transfer_fee_rate", 0.0)
    kwargs.setdefault("min_commission", 0.0)
    return run_backtest(data=data, strategy=strategy, show_progress=False, **kwargs)


def _expected_starts(days: List[str], first_len: int) -> List[int]:
    stamps = [_close_ns(d) for d in days]
    return [stamps[0] - first_len, *stamps[:-1]]


def _starts_by_symbol(seen: List[Tuple[Any, ...]]) -> Dict[str, List[Tuple[int, Any]]]:
    out: Dict[str, List[Tuple[int, Any]]] = {}
    for sym, ts, start, *_ in seen:
        out.setdefault(sym, []).append((ts, start))
    return out


# --------------------------------------------------------------------------- #
# 写法
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("freq", "nominal"),
    [
        ("1w", WEEK),
        ("W", WEEK),
        ("2w", 2 * WEEK),
        (" 1 Week ", WEEK),
        ("1mon", 31 * DAY),
        ("1MON", 31 * DAY),
        ("month", 31 * DAY),
        ("3mon", 93 * DAY),
        ("1q", 92 * DAY),
        ("1quarter", 92 * DAY),
        ("1y", 366 * DAY),
        ("1Year", 366 * DAY),
    ],
)
def test_calendar_freq_forms(freq: str, nominal: int) -> None:
    """周、月、季、年：大小写不敏感，倍数可省略，全称与 d/day 同例接受."""
    period = _parse_bar_period(freq)
    assert period is not None
    assert period.calendar
    assert period.ns == nominal


@pytest.mark.parametrize("freq", ["1min", "5min", "1h", "1d", "D", "2day"])
def test_fixed_freq_forms_unchanged(freq: str) -> None:
    """分钟、小时、日仍是固定长度."""
    period = _parse_bar_period(freq)
    assert period is not None
    assert not period.calendar


@pytest.mark.parametrize("bad", ["1m", "M", "1M", "0w", "1wk", "30s", "abc"])
def test_calendar_freq_rejects_month_as_m(bad: str) -> None:
    """月不用 m/M（与分钟混淆）；报错里列出新写法，不再让周/月先 resample."""
    with pytest.raises(ValueError, match="bar 周期") as info:
        _parse_bar_period(bad)
    message = str(info.value)
    for form in ("1w", "1mon", "1q", "1y"):
        assert form in message
    assert "周/月周期请先" not in message


# --------------------------------------------------------------------------- #
# 区间：按品种取上一根
# --------------------------------------------------------------------------- #


def _two_symbol_frames(time: Optional[str] = "15:00") -> Dict[str, pd.DataFrame]:
    return {A: _frame(WEEK_ENDS, time=time), B: _frame(B_WEEK_ENDS, time=time)}


class _Adapter:
    def __init__(self, frames: Dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def load(self, request: Any) -> pd.DataFrame:
        return self.frames[request.symbol]


def _data_forms(tmp_path: Path) -> Dict[str, Dict[str, Any]]:
    frames = _two_symbol_frames()
    long = _long(frames)
    catalog_root = tmp_path / "catalog"
    catalog = ParquetDataCatalog(root_path=str(catalog_root))
    for sym, df in frames.items():
        catalog.write(sym, df)
    import polars as pl

    return {
        "dict": {"data": frames},
        "long": {"data": long},
        # 乱序的长表：照现有路径按时间排序后再取上一根
        "long_shuffled": {"data": long.sample(frac=1.0, random_state=3)},
        "bars": {"data": _bars(frames)},
        "polars": {"data": pl.from_pandas(long)},
        "pyarrow": {"data": pa.Table.from_pandas(long, preserve_index=False)},
        "catalog": {"data": None, "catalog_path": str(catalog_root)},
        "adapter": {"data": _Adapter(frames)},
    }


@pytest.mark.parametrize(
    "form",
    [
        "dict",
        "long",
        "long_shuffled",
        "bars",
        "polars",
        "pyarrow",
        "catalog",
        "adapter",
    ],
)
def test_interval_start_is_previous_bar_of_same_symbol(
    form: str, tmp_path: Path
) -> None:
    """每种输入：区间起点 = 同品种上一根时间戳；缺一周的品种覆盖两周."""
    kwargs = _data_forms(tmp_path)[form]
    seen = _run(freq="1w", **kwargs).strategy.seen
    by_symbol = _starts_by_symbol(seen)
    assert [s for _, s in by_symbol[A]] == _expected_starts(WEEK_ENDS, WEEK)
    assert [s for _, s in by_symbol[B]] == _expected_starts(B_WEEK_ENDS, WEEK)
    # 节假日周（周四收盘）后的那根从周四 15:00 起
    assert dict(by_symbol[A])[_close_ns("2025-04-11")] == _close_ns("2025-04-03")
    # B 缺 3/21：3/28 那根从 3/14 起，覆盖两周
    assert dict(by_symbol[B])[_close_ns("2025-03-28")] == _close_ns("2025-03-14")
    assert all(row[3] == "Continuous" for row in seen)
    assert all(row[4] for row in seen)
    assert all(row[5] == "1w" for row in seen)


def test_single_symbol_frame_interval_start() -> None:
    """单标的 DataFrame（无 symbol 列）."""
    seen = _run(_frame(WEEK_ENDS), symbols=A, freq="1w").strategy.seen
    assert [row[2] for row in seen] == _expected_starts(WEEK_ENDS, WEEK)


def test_duplicate_timestamp_rows_share_previous_bar() -> None:
    """同品种同一时间戳的重复行照旧都投递；两行的区间都从上一个不同的时间戳起."""
    df = _frame(WEEK_ENDS[:3])
    dup = pd.concat([df.iloc[:2], df.iloc[1:2], df.iloc[2:]])
    seen = _run({A: dup}, symbols=A, freq="1w").strategy.seen
    starts = [row[2] for row in seen]
    first = _close_ns(WEEK_ENDS[0])
    second = _close_ns(WEEK_ENDS[1])
    assert starts == [first - WEEK, first, first, second]


def test_date_only_weekly_rewritten_then_previous_bar() -> None:
    """只有日期的周线：先改写到 15:00，再按上一根取区间（不再是 24 小时）."""
    frames = _two_symbol_frames(time=None)
    seen = _run(frames, freq="1w").strategy.seen
    by_symbol = _starts_by_symbol(seen)
    assert [ts for ts, _ in by_symbol[A]] == [_close_ns(d) for d in WEEK_ENDS]
    assert [s for _, s in by_symbol[A]] == _expected_starts(WEEK_ENDS, WEEK)
    assert [s for _, s in by_symbol[B]] == _expected_starts(B_WEEK_ENDS, WEEK)
    assert all(row[4] for row in seen)


def test_date_only_bar_list_weekly() -> None:
    """只有日期的 Bar 列表同样先改写再取区间."""
    seen = _run(_bars(_two_symbol_frames(time=None)), freq="1w").strategy.seen
    by_symbol = _starts_by_symbol(seen)
    assert [s for _, s in by_symbol[B]] == _expected_starts(B_WEEK_ENDS, WEEK)


def test_bar_list_own_interval_start_is_kept() -> None:
    """Bar 自带的 interval_start 原样保留，其余按上一根取."""
    bars = _bars({A: _frame(WEEK_ENDS[:3])})
    own = _close_ns(WEEK_ENDS[1]) - DAY
    bars[2] = Bar(
        timestamp=bars[2].timestamp,
        open=bars[2].open,
        high=bars[2].high,
        low=bars[2].low,
        close=bars[2].close,
        volume=bars[2].volume,
        symbol=A,
        interval_start=own,
    )
    seen = _run(bars, symbols=A, freq="1w").strategy.seen
    first = _close_ns(WEEK_ENDS[0])
    assert [row[2] for row in seen] == [first - WEEK, first, own]


def test_without_freq_weekly_bar_is_closing_auction() -> None:
    """对照：不传 freq 时 15:00 打戳的周线被判成收盘集合竞价."""
    seen = _run(_two_symbol_frames()).strategy.seen
    assert all(row[2] is None for row in seen)
    assert all(row[3] == "ClosingAuction" for row in seen)
    assert not any(row[4] for row in seen)
    assert all(row[5] is None for row in seen)


# --------------------------------------------------------------------------- #
# 撮合：与不传 freq 对照
# --------------------------------------------------------------------------- #


class _Orders(Strategy):
    """按参数在第 i 根 bar 下单；记录成交."""

    plan: List[Tuple[int, str, Dict[str, Any]]] = []

    def __init__(self) -> None:
        super().__init__()
        self.index = 0

    def on_bar(self, bar: Bar) -> None:
        for at, side, kwargs in self.plan:
            if at == self.index:
                getattr(self, side)(bar.symbol, 100, **kwargs)
        self.index += 1


def _orders_strategy(plan: List[Tuple[int, str, Dict[str, Any]]]) -> Any:
    return type("_Planned", (_Orders,), {"plan": plan})


def _fills(result: Any) -> List[Tuple[str, int, float]]:
    return [
        (str(row["side"]), int(pd.Timestamp(row["timestamp"]).value), row["price"])
        for row in result.executions_df.to_dict("records")
    ]


_WEEKLY_A = {A: _frame(WEEK_ENDS)}
# 第 i 根：开 10+i，高 13+i，低 7+i，收 11+i


@pytest.mark.parametrize("freq", ["1w", None])
def test_next_open_market_order(freq: Optional[str]) -> None:
    """NextOpen 市价单：传 1w 按下一根开盘价成交，不传按收盘价."""
    strategy = _orders_strategy([(0, "buy", {})])
    result = _run(_WEEKLY_A, strategy, symbols=A, freq=freq, fill_policy=aq.NextOpen())
    expected_price = 11.0 if freq else 12.0
    assert _fills(result) == [("buy", _close_ns(WEEK_ENDS[1]), expected_price)]


@pytest.mark.parametrize("freq", ["1w", None])
def test_limit_buy_touched_by_low(freq: Optional[str]) -> None:
    """限价买 9：下一根低 8、收 12。传 1w 时最低价触及即成交，不传只比收盘价."""
    strategy = _orders_strategy([(0, "buy", {"price": 9.0})])
    result = _run(_WEEKLY_A, strategy, symbols=A, freq=freq, fill_policy=aq.NextOpen())
    if freq:
        assert _fills(result) == [("buy", _close_ns(WEEK_ENDS[1]), 9.0)]
    else:
        assert _fills(result) == []


@pytest.mark.parametrize("freq", ["1w", None])
def test_stop_sell_touched_by_low(freq: Optional[str]) -> None:
    """止损卖（触发 11）：第 2 根开 12、低 9。传 1w 时盘中触发，不传只比收盘价 13."""
    strategy = _orders_strategy([(0, "buy", {}), (1, "sell", {"trigger_price": 11.0})])
    result = _run(_WEEKLY_A, strategy, symbols=A, freq=freq, fill_policy=aq.NextOpen())
    sells = [f for f in _fills(result) if f[0] == "sell"]
    if freq:
        assert sells == [("sell", _close_ns(WEEK_ENDS[2]), 11.0)]
    else:
        assert sells == []


def test_current_close_market_order() -> None:
    """CurrentClose：当根收盘价成交."""
    strategy = _orders_strategy([(0, "buy", {})])
    result = _run(
        _WEEKLY_A, strategy, symbols=A, freq="1w", fill_policy=aq.CurrentClose()
    )
    assert _fills(result) == [("buy", _close_ns(WEEK_ENDS[0]), 11.0)]


class _TickStop(Strategy):
    """首个模拟 tick 市价买；持仓后在价格不低于 12 的 tick 上挂止损卖 9；记录 tick.

    挂单时价格已低于触发价的止损会在挂单的那个 tick 上当场触发，故等到高位再挂。
    """

    def __init__(self) -> None:
        super().__init__()
        self.ticks: List[Tuple[int, float]] = []
        self.bought = False
        self.stopped = False

    def on_bar(self, bar: Bar) -> None:
        pass

    def on_tick(self, tick: Tick) -> None:
        self.ticks.append((int(tick.timestamp), float(tick.price)))
        if not self.bought:
            self.bought = True
            self.buy(tick.symbol, 100)
        elif (
            not self.stopped
            and self.get_position(tick.symbol) > 0
            and tick.price >= 12.0
        ):
            self.stopped = True
            self.sell(tick.symbol, 100, trigger_price=9.0)


@pytest.mark.parametrize("freq", ["1w", None])
def test_simulate_ticks_on_weekly_bars(freq: Optional[str]) -> None:
    """simulate_ticks：每根周线 4 个模拟 tick（E-4s～E-1s），on_tick 里的止损能触发."""
    result = _run(_WEEKLY_A, _TickStop, symbols=A, freq=freq, simulate_ticks=True)
    ticks = result.strategy.ticks
    if not freq:
        assert ticks == []
        assert _fills(result) == []
        return
    second = 1_000_000_000
    expected: List[Tuple[int, float]] = []
    for i, day in enumerate(WEEK_ENDS):
        end = _close_ns(day)
        prices = [10.0 + i, 7.0 + i, 13.0 + i, 11.0 + i]
        expected += [(end - back * second, p) for back, p in zip([4, 3, 2, 1], prices)]
    assert ticks == expected
    # 第一根开盘 tick 买入 10；第二根的最低价 tick（8）触发止损 9，按 tick 价成交
    assert _fills(result) == [
        ("buy", _close_ns(WEEK_ENDS[0]) - 4 * second, 10.0),
        ("sell", _close_ns(WEEK_ENDS[1]) - 3 * second, 8.0),
    ]


# --------------------------------------------------------------------------- #
# 月、季、年
# --------------------------------------------------------------------------- #


_CALENDAR_CASES = {
    # 月末最后一个交易日；2025 年 2 月 28 天，1 月最后交易日是 1/27（春节）
    "1mon": (["2025-01-27", "2025-02-28", "2025-03-31", "2025-04-30"], 31 * DAY),
    "1q": (["2024-03-29", "2024-06-28", "2024-09-30", "2024-12-31"], 92 * DAY),
    "1y": (["2021-12-31", "2022-12-30", "2023-12-29", "2024-12-31"], 366 * DAY),
}


class _ProbeAndBuy(_Probe):
    def on_bar(self, bar: Bar) -> None:
        super().on_bar(bar)
        if len(self.seen) == 1:
            self.buy(bar.symbol, 100)


@pytest.mark.parametrize("freq", sorted(_CALENDAR_CASES))
def test_month_quarter_year(freq: str) -> None:
    """月、季、年线：市价单按下一根开盘价成交，is_market_open() 为 True."""
    days, nominal = _CALENDAR_CASES[freq]
    result = _run(
        {A: _frame(days)},
        _ProbeAndBuy,
        symbols=A,
        freq=freq,
        fill_policy=aq.NextOpen(),
    )
    seen = result.strategy.seen
    assert [row[2] for row in seen] == _expected_starts(days, nominal)
    assert all(row[4] for row in seen)
    assert all(row[5] == freq for row in seen)
    assert _fills(result) == [("buy", _close_ns(days[1]), 11.0)]


# --------------------------------------------------------------------------- #
# 多周期窗口、续跑、网格
# --------------------------------------------------------------------------- #


class _Windowed(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars("1d")

    def on_bar(self, bar: Bar) -> None:
        pass


def test_window_subscription_on_calendar_base_is_rejected() -> None:
    """基础数据是日历周期时再声明窗口周期：报清楚的错，不静默聚合."""
    with pytest.raises(ValueError, match="日历周期") as info:
        _run(_WEEKLY_A, _Windowed, symbols=A, freq="1w")
    assert "1w" in str(info.value)
    assert "subscribe_bars" in str(info.value)


def test_resume_from_checkpoint_keeps_previous_bar(tmp_path: Path) -> None:
    """续跑：续接数据带上存档前的最后一根时按它取区间；只有新数据时退回名义长度."""
    head, tail = WEEK_ENDS[:4], WEEK_ENDS[4:]
    first = _run({A: _frame(WEEK_ENDS).iloc[:4]}, symbols=A, freq="1w")
    ckpt = tmp_path / "ckpt.pkl"
    save_checkpoint(first.engine, first.strategy, str(ckpt), config=first)

    only_new = aq.run_from_checkpoint(
        str(ckpt), data={A: _frame(WEEK_ENDS).iloc[4:]}, symbols=A
    )
    new_seen = cast(Any, only_new.strategy).seen[len(head) :]
    assert [row[1] for row in new_seen] == [_close_ns(d) for d in tail]
    tail_stamps = [_close_ns(d) for d in tail]
    assert [row[2] for row in new_seen] == [tail_stamps[0] - WEEK, *tail_stamps[:-1]]
    assert all(row[5] == "1w" for row in new_seen)
    assert all(row[4] for row in new_seen)

    with_overlap = aq.run_from_checkpoint(
        str(ckpt), data={A: _frame(WEEK_ENDS).iloc[3:]}, symbols=A
    )
    seen = cast(Any, with_overlap.strategy).seen[len(head) :]
    assert [row[1] for row in seen] == tail_stamps
    assert [row[2] for row in seen] == [_close_ns(head[-1]), *tail_stamps[:-1]]


class _BuyIfWeekly(Strategy):
    qty = IntParam(100)

    def on_bar(self, bar: Bar) -> None:
        if self.freq == "1w" and self.is_market_open() and not self.get_position():
            self.buy(bar.symbol, self.params.qty)


def test_grid_search_subprocess_carries_freq() -> None:
    """网格搜索的子进程里 self.freq 与区间同样生效."""
    results = cast(
        pd.DataFrame,
        aq.run_grid_search(
            strategy=_BuyIfWeekly,
            param_grid={"qty": [100, 200]},
            data={A: _frame(WEEK_ENDS)},
            symbols=A,
            initial_cash=1e6,
            freq="1w",
            max_workers=2,
            show_progress=False,
            return_df=True,
        ),
    )
    assert len(results) == 2
    # 子进程里 self.freq == "1w" 且 is_market_open() 为真才会下单
    assert (np.asarray(results["execution_count"], dtype=float) == 1).all()
    pnl = dict(zip(results["qty"], results["total_pnl"]))
    assert pnl[100] > 0
    assert pnl[200] == pytest.approx(2 * pnl[100])


# --------------------------------------------------------------------------- #
# 不变性：日线逐笔
# --------------------------------------------------------------------------- #


_DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]


@pytest.mark.parametrize("freq", ["1d", None])
def test_daily_fills_unchanged(freq: Optional[str]) -> None:
    """日线：传 1d 区间起点仍是时间戳前 24h，逐笔成交与此前相同."""
    strategy = _orders_strategy(
        [
            (0, "buy", {}),
            (1, "buy", {"price": 9.0}),
            (2, "sell", {"trigger_price": 11.0}),
        ]
    )
    result = _run(
        {A: _frame(_DAYS)}, strategy, symbols=A, freq=freq, fill_policy=aq.NextOpen()
    )
    fills = _fills(result)
    if freq:
        assert fills == [
            ("buy", _close_ns(_DAYS[1]), 11.0),
            ("buy", _close_ns(_DAYS[2]), 9.0),
            ("sell", _close_ns(_DAYS[3]), 11.0),
        ]
    else:
        assert fills == [("buy", _close_ns(_DAYS[1]), 12.0)]
    probe = _run({A: _frame(_DAYS)}, symbols=A, freq=freq).strategy.seen
    expected = [_close_ns(d) - DAY for d in _DAYS] if freq else [None] * len(_DAYS)
    assert [row[2] for row in probe] == expected
