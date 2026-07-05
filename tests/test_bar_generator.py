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


def test_session_windows_no_cross_lunch() -> None:
    """配 session: 11:30 前最后一窗段末闭合, 13:00 起新窗; 不并入同一 5min 桶.

    afternoon 段起点用 13:01/13:02(非 13:00 整点), 理由同 test_close_timing_and_flush
    等的 09:31 起点注释: 09:30/13:00 恰落在 5min 时钟网格边界上, ceil(freq) 下边界点
    自成单独一桶(见 Task 1 report Deviation 1), 若从 13:00 整点起会在 13:00/13:01 间
    产生一次与 session 无关的额外网格切分, 使 got 变 3 根(300/100/100)而非 2 根
    (300/200), 与本用例断言不符。故此处如其余用例一样避开整点。
    """
    # 11:28,11:29,11:30(上午段末) + 13:01,13:02(下午段首, 避开 13:00 整点网格边界)
    ts = [
        "2024-01-02 11:28:00",
        "2024-01-02 11:29:00",
        "2024-01-02 11:30:00",
        "2024-01-02 13:01:00",
        "2024-01-02 13:02:00",
    ]
    bars = []
    for i, t in enumerate(ts):
        base = 10.0 + i
        bars.append(Bar(int(pd.Timestamp(t).value), base, base, base, base, 100.0, "X"))
    got = []
    bg = BarGenerator(
        lambda b: got.append(b),
        window=5,
        interval="minute",
        session_windows=[("09:30", "11:30"), ("13:00", "15:00")],
    )
    for b in bars:
        bg.update_bar(b)
    bg.flush()
    # 上午段(11:26-11:30]窗 与 下午段窗 各自独立, 不合并
    assert len(got) == 2
    # 第一根窗口的成交量只含上午段的 3 根
    assert got[0].volume == 300.0
    assert got[1].volume == 200.0


def test_no_session_windows_merges_across_gap() -> None:
    """不配 session: 纯时钟对齐(对照组, 与上一个测试区分行为).

    起点用 13:01(非 13:00 整点), 理由同上: 13:00 恰落在 5min 网格边界上会自成一桶,
    使 3 根合并变 2 根(100/200), 与"同一 5min 桶归并"的本意不符。
    """
    # 同一 5min 时钟桶归并规则下 3 根应归并为 1 窗(仅验证不配 session 时不额外切分)
    ts = ["2024-01-02 13:01:00", "2024-01-02 13:02:00", "2024-01-02 13:03:00"]
    bars = [
        Bar(int(pd.Timestamp(t).value), 10.0, 10.0, 10.0, 10.0, 100.0, "X") for t in ts
    ]
    got = []
    bg = BarGenerator(lambda b: got.append(b), window=5, interval="minute")
    for b in bars:
        bg.update_bar(b)
    bg.flush()
    assert len(got) == 1  # 13:01-13:03 同一 (13:00,13:05] 桶


def test_session_boundary_inside_clock_bucket_discriminates() -> None:
    """判别性测试: session 边界落在同一时钟桶内 → 无 session 会合并, 有 session 才切分.

    window=10min, 五根 10:01..10:05 全部 ceil 到同一 (10:00,10:10] 桶:
    - 不配 session: 合并成 1 窗(vol 500)。
    - 配 session 段界 10:03/10:04: 上午段(10:01-10:03)与下午段(10:04-10:05)切开
      → 2 窗(vol 300 / 200)。二者标签同为 10:10 但 session_key 不同, 靠段变闭合。
    这个用例在 session 逻辑缺失时会失败(得 1 窗), 故真正覆盖该特性。
    """
    ts = [
        "2024-01-02 10:01:00",
        "2024-01-02 10:02:00",
        "2024-01-02 10:03:00",
        "2024-01-02 10:04:00",
        "2024-01-02 10:05:00",
    ]
    bars = [
        Bar(int(pd.Timestamp(t).value), 10.0, 10.0, 10.0, 10.0, 100.0, "X") for t in ts
    ]

    # 对照: 不配 session → 同一 10min 桶合并为 1 窗
    plain: list = []
    bg_plain = BarGenerator(lambda b: plain.append(b), window=10, interval="minute")
    for b in bars:
        bg_plain.update_bar(b)
    bg_plain.flush()
    assert len(plain) == 1
    assert plain[0].volume == 500.0

    # 配 session: 段界落在桶内 → 切成 2 窗
    got: list = []
    bg = BarGenerator(
        lambda b: got.append(b),
        window=10,
        interval="minute",
        session_windows=[("10:00", "10:03"), ("10:04", "10:10")],
    )
    for b in bars:
        bg.update_bar(b)
    bg.flush()
    assert len(got) == 2
    assert got[0].volume == 300.0  # 上午段 10:01-10:03
    assert got[1].volume == 200.0  # 下午段 10:04-10:05
