"""BarGenerator: 流式多周期聚合, 与 pandas resample(label=right,closed=right) 一致."""

import pandas as pd
from akquant import BarGenerator
from akquant.akquant import Bar


def _bars_1min(n: int, symbol: str = "X", start="2024-01-02 09:30:00"):
    """造 n 根 1min bar(时间戳=右缘, 递增), OHLCV 可预期."""
    idx = pd.date_range(start, periods=n, freq="1min")
    bars = []
    for i, ts in enumerate(idx):
        base = 10.0 + i
        bars.append(
            Bar(
                int(ts.value),
                base,
                base + 0.5,
                base - 0.5,
                base + 0.1,
                100.0 + i,
                symbol,
            )
        )
    return bars


def _expected_resample(bars, freq):
    """用 pandas resample 造期望窗口 bar(label=right, closed=right)."""
    df = pd.DataFrame(
        {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        },
        index=pd.to_datetime([b.timestamp for b in bars]),
    )
    out = df.resample(freq, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open"])


def _collect(freq_window_interval, bars):
    window, interval, freq = freq_window_interval
    got = []
    bg = BarGenerator(lambda b: got.append(b), window=window, interval=interval)
    for b in bars:
        bg.update_bar(b)
    bg.flush()
    return got


def test_matches_pandas_resample_5min() -> None:
    """5min 聚合逐桶应与 pandas resample(label=right, closed=right) 一致."""
    bars = _bars_1min(17)  # 跨多个 5min 桶 + 尾部半截
    got = _collect((5, "minute", "5min"), bars)
    exp = _expected_resample(bars, "5min")
    assert len(got) == len(exp)
    for gbar, (ts, row) in zip(got, exp.iterrows()):
        assert gbar.timestamp == int(pd.Timestamp(ts).value)
        assert gbar.open == row["open"]
        assert gbar.high == row["high"]
        assert gbar.low == row["low"]
        assert gbar.close == row["close"]
        assert gbar.volume == row["volume"]


def test_matches_pandas_resample_15min_and_hour() -> None:
    """15min 与 1h 聚合逐桶应与 pandas resample 一致."""
    bars = _bars_1min(140)
    for window, interval, freq in [(15, "minute", "15min"), (1, "hour", "1h")]:
        got = _collect((window, interval, freq), bars)
        exp = _expected_resample(bars, freq)
        assert len(got) == len(exp)
        assert [b.close for b in got] == list(exp["close"])
        assert [b.volume for b in got] == list(exp["volume"])


def test_daily_aggregation() -> None:
    """日线聚合: 3 个交易日各出 1 根日 bar."""
    # 3 天各 2 根 bar
    bars = (
        _bars_1min(2, start="2024-01-02 09:30:00")
        + _bars_1min(2, start="2024-01-03 09:30:00")
        + _bars_1min(2, start="2024-01-04 09:30:00")
    )
    got = _collect((1, "day", "1D"), bars)
    assert len(got) == 3


def test_close_timing_and_flush() -> None:
    """窗口在下一桶首根到来时闭合; 未满窗口靠 flush 闭合."""
    # start 从 09:31 起(而非 09:30), 避免首根恰落在 5min 网格边界上
    # (ceil 边界归自身桶, 09:30 本身即边界, 会导致第1根单独提前闭合)
    # 5min: 第1桶(1..5)在第6根到来时闭合, 第2桶(6,7)靠 flush
    bars = _bars_1min(7, start="2024-01-02 09:31:00")
    got = []
    bg = BarGenerator(lambda b: got.append(b), window=5, interval="minute")
    for b in bars[:5]:
        bg.update_bar(b)
    assert got == []  # 第5根后尚未跨桶, 不闭合
    bg.update_bar(bars[5])  # 第6根跨入新桶 -> 闭合第1窗口
    assert len(got) == 1
    bg.flush()
    assert len(got) == 2


def test_gap_bucket_not_filled() -> None:
    """跳空的桶不产出空 bar."""
    b1 = _bars_1min(1, start="2024-01-02 09:30:00")
    b2 = _bars_1min(1, start="2024-01-02 10:30:00")  # 跳过中间多个 5min 桶
    got = _collect((5, "minute", "5min"), b1 + b2)
    assert len(got) == 2  # 仅两个有成交的桶, 中间空桶不补


def test_multi_symbol_independent() -> None:
    """多标的独立聚合, 互不干扰."""
    a = _bars_1min(6, symbol="A")
    b = _bars_1min(6, symbol="B")
    interleaved = [x for pair in zip(a, b) for x in pair]
    got = []
    bg = BarGenerator(lambda bar: got.append(bar), window=5, interval="minute")
    for x in interleaved:
        bg.update_bar(x)
    bg.flush()
    syms = sorted({g.symbol for g in got})
    assert syms == ["A", "B"]
    assert len([g for g in got if g.symbol == "A"]) == 2
    assert len([g for g in got if g.symbol == "B"]) == 2


def test_ohlcv_correctness() -> None:
    """OHLCV 聚合正确性: open=首/high=最大/low=最小/close=尾/volume=和."""
    # start 从 09:31 起, 避免首根恰落在 5min 网格边界上(理由同上)
    bars = _bars_1min(5, start="2024-01-02 09:31:00")  # 恰一个 5min 桶
    got = _collect((5, "minute", "5min"), bars)
    assert len(got) == 1
    wb = got[0]
    assert wb.open == bars[0].open
    assert wb.high == max(b.high for b in bars)
    assert wb.low == min(b.low for b in bars)
    assert wb.close == bars[-1].close
    assert wb.volume == sum(b.volume for b in bars)


def test_current_snapshot_no_callback() -> None:
    """current() 返回未闭合窗口快照且不触发回调."""
    # start 从 09:31 起, 避免首根恰落在 5min 网格边界上(理由同上)
    bars = _bars_1min(3, start="2024-01-02 09:31:00")
    got = []
    bg = BarGenerator(lambda b: got.append(b), window=5, interval="minute")
    for b in bars:
        bg.update_bar(b)
    snap = bg.current("X")
    assert snap is not None and snap.close == bars[-1].close
    assert got == []  # current 不触发回调


def test_timezone_day_boundary() -> None:
    """按时区(tz)本地日界对齐: UTC 22:00(北京次日 06:00) 与 UTC 02:00 属不同北京日."""
    s1 = _bars_1min(1, start="2024-01-01 22:00:00")  # 北京 01-02 06:00
    s2 = _bars_1min(1, start="2024-01-02 02:00:00")  # 北京 01-02 10:00 (同一北京日)
    got = []
    bg = BarGenerator(
        lambda b: got.append(b), window=1, interval="day", timezone="Asia/Shanghai"
    )
    for b in s1 + s2:
        bg.update_bar(b)
    bg.flush()
    assert len(got) == 1  # 同属北京 01-02, 聚成 1 根日线


def test_invalid_params() -> None:
    """非法 window/interval 应抛 ValueError."""
    import pytest

    with pytest.raises(ValueError):
        BarGenerator(lambda b: None, window=0)
    with pytest.raises(ValueError):
        BarGenerator(lambda b: None, interval="weekly")
