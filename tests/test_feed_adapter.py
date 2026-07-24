import akquant
import pandas as pd
import pytest
from akquant.feed_adapter import BasePandasFeedAdapter


class InMemoryFeedAdapter(BasePandasFeedAdapter):
    """In-memory adapter for feed adapter tests."""

    name = "memory"

    def __init__(self, frame: pd.DataFrame) -> None:
        """Store source frame."""
        self.frame = frame.copy()

    def load(self, request: akquant.FeedSlice) -> pd.DataFrame:
        """Return sliced data by symbol and time range."""
        frame = self.frame[
            self.frame["symbol"].astype(str) == str(request.symbol)
        ].copy()
        normalized = self.normalize(frame, str(request.symbol))
        return self._clip_time_range(
            normalized, request.start_time, request.end_time, request.timezone
        )


class OneShotBuyStrategy(akquant.Strategy):
    """Submit a single buy order on first bar."""

    def __init__(self) -> None:
        """Initialize state."""
        self._submitted = False

    def on_bar(self, bar: akquant.Bar) -> None:
        """Submit order once."""
        if self._submitted:
            return
        self.buy(symbol=bar.symbol, quantity=1)
        self._submitted = True


def _make_minute_frame(symbol: str = "TEST") -> pd.DataFrame:
    """Build deterministic minute-level frame."""
    times = pd.date_range("2024-01-01 09:31:00", periods=6, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": times,
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "high": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
            "close": [10.2, 11.2, 12.2, 13.2, 14.2, 15.2],
            "volume": [100, 110, 120, 130, 140, 150],
            "symbol": [symbol] * 6,
        }
    )


def test_feed_resample_aggregates_ohlcv() -> None:
    """Resample should aggregate OHLCV with default rules."""
    adapter = InMemoryFeedAdapter(_make_minute_frame("RES"))
    resampled = adapter.resample(freq="5min", emit_partial=False)
    result = resampled.load(akquant.FeedSlice(symbol="RES"))

    assert len(result) == 1
    row = result.iloc[0]
    assert float(row["open"]) == 10.0
    assert float(row["high"]) == 14.5
    assert float(row["low"]) == 9.5
    assert float(row["close"]) == 14.2
    assert float(row["volume"]) == 600.0
    assert str(row["symbol"]) == "RES"


def test_feed_replay_defaults_drop_partial_tail() -> None:
    """Replay should default to non-partial output."""
    adapter = InMemoryFeedAdapter(_make_minute_frame("REP"))
    replayed = adapter.replay(freq="5min")
    result = replayed.load(akquant.FeedSlice(symbol="REP"))

    assert len(result) == 1
    assert float(result.iloc[0]["close"]) == 14.2


def test_feed_replay_session_align_drops_partial_each_day() -> None:
    """Replay session alignment should drop partial tail for each session."""
    day1 = _make_minute_frame("S1")
    day2 = _make_minute_frame("S1").copy()
    day2["timestamp"] = day2["timestamp"] + pd.Timedelta(days=1)
    data = pd.concat([day1, day2], axis=0, ignore_index=True)

    adapter = InMemoryFeedAdapter(data)
    replayed = adapter.replay(freq="5min", align="session", emit_partial=False)
    result = replayed.load(akquant.FeedSlice(symbol="S1"))

    assert len(result) == 2
    assert float(result.iloc[0]["volume"]) == 600.0
    assert float(result.iloc[1]["volume"]) == 600.0


def test_feed_replay_session_windows_split_midday() -> None:
    """Replay session windows should split intra-day aggregation buckets."""
    morning_times = pd.date_range(
        "2024-01-01 11:10:00",
        periods=20,
        freq="min",
        tz="Asia/Shanghai",
    )
    afternoon_times = pd.date_range(
        "2024-01-01 13:00:00",
        periods=10,
        freq="min",
        tz="Asia/Shanghai",
    )
    timestamps = list(morning_times) + list(afternoon_times)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [10.0 + i * 0.1 for i in range(len(timestamps))],
            "high": [10.2 + i * 0.1 for i in range(len(timestamps))],
            "low": [9.8 + i * 0.1 for i in range(len(timestamps))],
            "close": [10.1 + i * 0.1 for i in range(len(timestamps))],
            "volume": [100.0] * len(timestamps),
            "symbol": ["SH"] * len(timestamps),
        }
    )
    adapter = InMemoryFeedAdapter(frame)

    replay_default = adapter.replay(freq="15min", align="session", emit_partial=False)
    default_result = replay_default.load(
        akquant.FeedSlice(symbol="SH", timezone="Asia/Shanghai")
    )

    replay_split = adapter.replay(
        freq="15min",
        align="session",
        emit_partial=False,
        session_windows=[("09:30", "11:30"), ("13:00", "15:00")],
    )
    split_result = replay_split.load(
        akquant.FeedSlice(symbol="SH", timezone="Asia/Shanghai")
    )

    assert len(default_result) == 3
    assert len(split_result) == 2


def test_feed_replay_session_windows_naive_data_defaults_to_shanghai() -> None:
    """Naive replay input should follow Shanghai like `load_bar_from_df`."""
    morning_times = pd.date_range("2024-01-01 11:10:00", periods=20, freq="min")
    afternoon_times = pd.date_range("2024-01-01 13:00:00", periods=10, freq="min")
    timestamps = list(morning_times) + list(afternoon_times)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [10.0 + i * 0.1 for i in range(len(timestamps))],
            "high": [10.2 + i * 0.1 for i in range(len(timestamps))],
            "low": [9.8 + i * 0.1 for i in range(len(timestamps))],
            "close": [10.1 + i * 0.1 for i in range(len(timestamps))],
            "volume": [100.0] * len(timestamps),
            "symbol": ["SH_NAIVE"] * len(timestamps),
        }
    )
    adapter = InMemoryFeedAdapter(frame)

    replayed = adapter.replay(
        freq="15min",
        align="session",
        emit_partial=False,
        session_windows=[("09:30", "11:30"), ("13:00", "15:00")],
    )
    result = replayed.load(
        akquant.FeedSlice(symbol="SH_NAIVE", timezone="Asia/Shanghai")
    )

    assert len(result) == 2


def test_feed_slice_time_range_respects_request_timezone_on_naive_input() -> None:
    """Naive source timestamps should honor `FeedSlice` request timezone."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02 20:00:00", "2024-01-02 21:00:00", "2024-01-02 22:00:00"]
            ),
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.2, 11.2, 12.2],
            "volume": [100.0, 100.0, 100.0],
            "symbol": ["UTC_BOUNDARY"] * 3,
        }
    )
    adapter = InMemoryFeedAdapter(frame)

    result = adapter.load(
        akquant.FeedSlice(
            symbol="UTC_BOUNDARY",
            start_time=pd.Timestamp("2024-01-02 13:00:00", tz="UTC"),
            timezone="UTC",
        )
    )

    assert len(result) == 2
    assert list(result.index) == [
        pd.Timestamp("2024-01-02 21:00:00"),
        pd.Timestamp("2024-01-02 22:00:00"),
    ]


def test_feed_replay_global_align_keeps_cross_session_bins() -> None:
    """Global align should aggregate across intraday session boundaries."""
    morning_times = pd.date_range(
        "2024-01-01 11:10:00",
        periods=20,
        freq="min",
        tz="Asia/Shanghai",
    )
    afternoon_times = pd.date_range(
        "2024-01-01 13:00:00",
        periods=10,
        freq="min",
        tz="Asia/Shanghai",
    )
    timestamps = list(morning_times) + list(afternoon_times)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [10.0 + i * 0.1 for i in range(len(timestamps))],
            "high": [10.2 + i * 0.1 for i in range(len(timestamps))],
            "low": [9.8 + i * 0.1 for i in range(len(timestamps))],
            "close": [10.1 + i * 0.1 for i in range(len(timestamps))],
            "volume": [100.0] * len(timestamps),
            "symbol": ["GLB"] * len(timestamps),
        }
    )
    adapter = InMemoryFeedAdapter(frame)
    replayed = adapter.replay(freq="15min", align="global", emit_partial=False)
    result = replayed.load(akquant.FeedSlice(symbol="GLB", timezone="Asia/Shanghai"))

    assert len(result) == 3


def test_feed_replay_rejects_session_windows_when_align_not_session() -> None:
    """Replay should reject session windows unless align is session."""
    adapter = InMemoryFeedAdapter(_make_minute_frame("BAD"))
    with pytest.raises(ValueError, match="session_windows"):
        adapter.replay(
            freq="5min",
            align="global",
            session_windows=[("09:30", "11:30")],
        )


def test_feed_replay_day_mode_calendar_vs_trading() -> None:
    """Day mode should support calendar-day and trading-day partitioning."""
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 15:55:00", tz="UTC"),
            pd.Timestamp("2024-01-01 16:05:00", tz="UTC"),
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [10.0, 11.0],
            "high": [10.2, 11.2],
            "low": [9.8, 10.8],
            "close": [10.1, 11.1],
            "volume": [100.0, 120.0],
            "symbol": ["DAY"] * 2,
        }
    )
    adapter = InMemoryFeedAdapter(frame)

    trading_result = adapter.replay(
        freq="1D",
        align="day",
        day_mode="trading",
        emit_partial=True,
    ).load(akquant.FeedSlice(symbol="DAY", timezone="Asia/Shanghai"))
    calendar_result = adapter.replay(
        freq="1D",
        align="day",
        day_mode="calendar",
        emit_partial=True,
    ).load(akquant.FeedSlice(symbol="DAY", timezone="Asia/Shanghai"))

    assert len(trading_result) == 2
    assert len(calendar_result) == 1


def test_feed_replay_rejects_day_mode_when_align_not_day() -> None:
    """Replay should reject non-default day_mode unless align is day."""
    adapter = InMemoryFeedAdapter(_make_minute_frame("BAD_DAY"))
    with pytest.raises(ValueError, match="day_mode"):
        adapter.replay(
            freq="5min",
            align="global",
            day_mode="calendar",
        )


def test_run_backtest_accepts_resampled_adapter() -> None:
    """run_backtest should run with resampled adapter input."""
    adapter = InMemoryFeedAdapter(_make_minute_frame("RBK")).resample(
        freq="5min",
        emit_partial=False,
    )
    result = akquant.run_backtest(
        data=adapter,
        strategy=OneShotBuyStrategy,
        symbols="RBK",
        fill_policy=akquant.CurrentClose(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    assert not result.orders_df.empty
    assert set(result.orders_df["symbol"].astype(str)) == {"RBK"}
