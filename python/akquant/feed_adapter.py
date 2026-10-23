from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, cast

import pandas as pd
from pandas.tseries.frequencies import to_offset

from .akquant import Bar
from .utils import load_bar_from_df


@dataclass(frozen=True)
class FeedSlice:
    """Data request window for a single symbol."""

    symbol: str
    start_time: pd.Timestamp | None = None
    end_time: pd.Timestamp | None = None
    timezone: str | None = None


class DataFeedAdapter(Protocol):
    """Protocol for external feed adapters."""

    name: str

    def load(self, request: FeedSlice) -> pd.DataFrame:
        """Load market data for one request window."""
        ...


LabelKind = Literal["right", "left"]
SessionWindow = tuple[str, str]
AlignKind = Literal["session", "day", "global"]
DayModeKind = Literal["trading", "calendar"]
DEFAULT_INPUT_TIMEZONE = "Asia/Shanghai"


class BasePandasFeedAdapter:
    """Base adapter that normalizes pandas-based sources."""

    name = "base"
    required_columns = ("open", "high", "low", "close", "volume")

    def normalize(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize frame into canonical OHLCV schema."""
        data = frame.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            for candidate in ("date", "timestamp", "datetime", "time"):
                if candidate in data.columns:
                    data = data.set_index(candidate)
                    break
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        if "symbol" not in data.columns:
            data["symbol"] = symbol
        missing = [c for c in self.required_columns if c not in data.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        return cast(pd.DataFrame, data)

    def to_bars(self, frame: pd.DataFrame, symbol: str | None = None) -> list[Bar]:
        """Convert normalized frame into Bar objects."""
        return load_bar_from_df(frame, symbol)

    def _clip_time_range(
        self,
        frame: pd.DataFrame,
        start_time: pd.Timestamp | None,
        end_time: pd.Timestamp | None,
        timezone: str | None = None,
    ) -> pd.DataFrame:
        data = frame
        if data.empty or not isinstance(data.index, pd.DatetimeIndex):
            return data
        compare_index = _localize_input_index(data.index, timezone)
        start_bound = _normalize_request_boundary(start_time, timezone)
        end_bound = _normalize_request_boundary(end_time, timezone)
        mask = pd.Series(True, index=data.index)
        if start_bound is not None:
            mask &= compare_index >= start_bound
        if end_bound is not None:
            mask &= compare_index <= end_bound
        data = data.loc[mask.to_numpy()]
        return cast(pd.DataFrame, data)

    def resample(
        self,
        freq: str,
        agg: Mapping[str, str] | None = None,
        label: LabelKind = "right",
        closed: LabelKind = "right",
        emit_partial: bool = True,
    ) -> "ResampledFeedAdapter":
        """Create a resampled adapter view."""
        return ResampledFeedAdapter(
            source=self,
            freq=freq,
            agg=agg,
            label=label,
            closed=closed,
            emit_partial=emit_partial,
        )

    def replay(
        self,
        freq: str,
        align: AlignKind = "session",
        day_mode: DayModeKind = "trading",
        emit_partial: bool = False,
        agg: Mapping[str, str] | None = None,
        label: LabelKind = "right",
        closed: LabelKind = "right",
        session_windows: list[SessionWindow] | None = None,
    ) -> "ReplayFeedAdapter":
        """Create a replay adapter view."""
        return ReplayFeedAdapter(
            source=self,
            freq=freq,
            align=align,
            day_mode=day_mode,
            emit_partial=emit_partial,
            agg=agg,
            label=label,
            closed=closed,
            session_windows=session_windows,
        )


def _default_agg_mapping() -> dict[str, str]:
    return {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }


def _drop_partial_tail(
    frame: pd.DataFrame,
    last_timestamp: pd.Timestamp,
    freq: str,
    label: LabelKind,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    if label == "right":
        return cast(pd.DataFrame, frame[frame.index <= last_timestamp])
    offset = to_offset(freq)
    end_index = pd.DatetimeIndex(frame.index) + offset
    mask = end_index <= last_timestamp
    return cast(pd.DataFrame, frame.loc[mask])


def _resample_ohlcv_frame(
    frame: pd.DataFrame,
    freq: str,
    agg: Mapping[str, str] | None = None,
    label: LabelKind = "right",
    closed: LabelKind = "right",
    emit_partial: bool = True,
) -> pd.DataFrame:
    if frame.empty:
        return cast(pd.DataFrame, frame.copy())

    agg_map = dict(_default_agg_mapping())
    if agg:
        agg_map.update(dict(agg))

    if "symbol" in frame.columns:
        grouped = frame.groupby(frame["symbol"].astype(str), sort=False)
        chunks: list[pd.DataFrame] = []
        for sym, group in grouped:
            source = group.drop(columns=["symbol"])
            used_agg = {k: v for k, v in agg_map.items() if k in source.columns}
            resampled = source.resample(freq, label=label, closed=closed).agg(
                cast(Any, used_agg)
            )
            if not emit_partial:
                resampled = _drop_partial_tail(
                    resampled,
                    last_timestamp=source.index.max(),
                    freq=freq,
                    label=label,
                )
            marker_cols = [
                c for c in ("open", "high", "low", "close") if c in resampled
            ]
            if marker_cols:
                resampled = resampled[resampled[marker_cols].notna().any(axis=1)]
            resampled = resampled.dropna(how="all")
            if resampled.empty:
                continue
            resampled["symbol"] = str(sym)
            chunks.append(resampled)
        if not chunks:
            return cast(pd.DataFrame, frame.iloc[0:0].copy())
        result = pd.concat(chunks, axis=0)
        result.sort_index(inplace=True)
        return result

    used_agg = {k: v for k, v in agg_map.items() if k in frame.columns}
    resampled = frame.resample(freq, label=label, closed=closed).agg(
        cast(Any, used_agg)
    )
    if not emit_partial:
        resampled = _drop_partial_tail(
            resampled,
            last_timestamp=frame.index.max(),
            freq=freq,
            label=label,
        )
    marker_cols = [c for c in ("open", "high", "low", "close") if c in resampled]
    if marker_cols:
        resampled = resampled[resampled[marker_cols].notna().any(axis=1)]
    return cast(pd.DataFrame, resampled.dropna(how="all"))


def _localize_input_index(
    index: pd.DatetimeIndex, timezone: str | None = None
) -> pd.DatetimeIndex:
    """Normalize input index to a timezone-aware series.

    Naive market data is interpreted in ``timezone`` (the run's timezone),
    defaulting to Asia/Shanghai, matching the main DataFrame -> Bar path.
    """
    if index.tz is None:
        return index.tz_localize(timezone or DEFAULT_INPUT_TIMEZONE)
    return index


def _normalize_request_boundary(
    boundary: pd.Timestamp | None,
    timezone: str | None,
) -> pd.Timestamp | None:
    if boundary is None:
        return None
    if boundary.tzinfo is None:
        return cast(
            pd.Timestamp, boundary.tz_localize(timezone or DEFAULT_INPUT_TIMEZONE)
        )
    return boundary


def _session_partition_keys(
    index: pd.DatetimeIndex,
    timezone: str | None,
) -> pd.DatetimeIndex:
    localized = _localize_input_index(index, timezone)
    if timezone:
        localized = localized.tz_convert(timezone)
    return localized.normalize()


def _calendar_partition_keys(
    index: pd.DatetimeIndex, timezone: str | None = None
) -> pd.DatetimeIndex:
    localized = _localize_input_index(index, timezone).tz_convert("UTC")
    return localized.normalize()


def _parse_hhmm_to_minutes(value: str) -> int:
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid session time format: {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"invalid session time value: {value!r}")
    return hour * 60 + minute


def _session_window_partition_keys(
    index: pd.DatetimeIndex,
    timezone: str | None,
    session_windows: list[SessionWindow],
) -> tuple[pd.DatetimeIndex, list[bool]]:
    localized = _localize_input_index(index, timezone)
    if timezone:
        localized = localized.tz_convert(timezone)

    parsed_windows = [
        (_parse_hhmm_to_minutes(start), _parse_hhmm_to_minutes(end))
        for start, end in session_windows
    ]
    key_values: list[pd.Timestamp] = []
    keep_mask: list[bool] = []
    for ts in localized:
        minute_of_day = ts.hour * 60 + ts.minute
        matched_key: pd.Timestamp | None = None
        for start, end in parsed_windows:
            if start <= minute_of_day < end:
                matched_key = ts.normalize() + pd.Timedelta(minutes=start)
                break
        if matched_key is None:
            keep_mask.append(False)
            key_values.append(pd.Timestamp("1970-01-01", tz=localized.tz))
        else:
            keep_mask.append(True)
            key_values.append(matched_key)
    return pd.DatetimeIndex(key_values), keep_mask


def _replay_ohlcv_frame(
    frame: pd.DataFrame,
    freq: str,
    align: AlignKind,
    agg: Mapping[str, str] | None = None,
    label: LabelKind = "right",
    closed: LabelKind = "right",
    emit_partial: bool = False,
    timezone: str | None = None,
    session_windows: list[SessionWindow] | None = None,
    day_mode: DayModeKind = "trading",
) -> pd.DataFrame:
    if align == "global":
        return _resample_ohlcv_frame(
            frame=frame,
            freq=freq,
            agg=agg,
            label=label,
            closed=closed,
            emit_partial=emit_partial,
        )
    if frame.empty:
        return cast(pd.DataFrame, frame.copy())

    agg_map = dict(_default_agg_mapping())
    if agg:
        agg_map.update(dict(agg))

    if "symbol" in frame.columns:
        by_symbol = frame.groupby(frame["symbol"].astype(str), sort=False)
    else:
        by_symbol = [("", frame)]  # type: ignore

    chunks: list[pd.DataFrame] = []
    for sym, group in by_symbol:
        source = group.drop(columns=["symbol"]) if "symbol" in group.columns else group
        source = source.sort_index()
        used_agg = {k: v for k, v in agg_map.items() if k in source.columns}
        if align == "day":
            if day_mode == "calendar":
                keys = _calendar_partition_keys(
                    pd.DatetimeIndex(source.index), timezone
                )
            else:
                keys = _session_partition_keys(pd.DatetimeIndex(source.index), timezone)
        elif session_windows:
            keys, keep_mask = _session_window_partition_keys(
                pd.DatetimeIndex(source.index),
                timezone,
                session_windows,
            )
            source = source.loc[keep_mask]
            keys = keys[keep_mask]
        else:
            keys = _session_partition_keys(pd.DatetimeIndex(source.index), timezone)
        if source.empty:
            continue

        grouped_sessions = source.groupby(keys, sort=True)
        for _, session_source in grouped_sessions:
            resampled = session_source.resample(freq, label=label, closed=closed).agg(
                cast(Any, used_agg)
            )
            if not emit_partial:
                resampled = _drop_partial_tail(
                    frame=resampled,
                    last_timestamp=session_source.index.max(),
                    freq=freq,
                    label=label,
                )
            marker_cols = [
                c for c in ("open", "high", "low", "close") if c in resampled
            ]
            if marker_cols:
                resampled = resampled[resampled[marker_cols].notna().any(axis=1)]
            resampled = resampled.dropna(how="all")
            if resampled.empty:
                continue
            if "symbol" in frame.columns:
                resampled["symbol"] = str(sym)
            chunks.append(resampled)

    if not chunks:
        return cast(pd.DataFrame, frame.iloc[0:0].copy())
    result = pd.concat(chunks, axis=0)
    result.sort_index(inplace=True)
    return result


class ResampledFeedAdapter(BasePandasFeedAdapter):
    """Adapter view that aggregates source feed by target frequency."""

    name = "resampled"

    def __init__(
        self,
        source: Any,
        freq: str,
        agg: Mapping[str, str] | None = None,
        label: LabelKind = "right",
        closed: LabelKind = "right",
        emit_partial: bool = True,
    ) -> None:
        """Initialize resample view."""
        self.source = source
        self.freq = freq
        self.agg = dict(agg or {})
        self.label = label
        self.closed = closed
        self.emit_partial = emit_partial

    def load(self, request: FeedSlice) -> pd.DataFrame:
        """Load source data and aggregate to target frequency."""
        frame = self.source.load(request)
        normalized = self.normalize(frame, request.symbol)
        return _resample_ohlcv_frame(
            normalized,
            freq=self.freq,
            agg=self.agg,
            label=self.label,
            closed=self.closed,
            emit_partial=self.emit_partial,
        )


class ReplayFeedAdapter(ResampledFeedAdapter):
    """Adapter view that replays high-frequency data on lower-frequency clock."""

    name = "replay"

    def __init__(
        self,
        source: Any,
        freq: str,
        align: AlignKind = "session",
        day_mode: DayModeKind = "trading",
        emit_partial: bool = False,
        agg: Mapping[str, str] | None = None,
        label: LabelKind = "right",
        closed: LabelKind = "right",
        session_windows: list[SessionWindow] | None = None,
    ) -> None:
        """Initialize replay view."""
        if align not in ("session", "day", "global"):
            raise ValueError("align must be one of: session/day/global")
        if day_mode not in ("trading", "calendar"):
            raise ValueError("day_mode must be one of: trading/calendar")
        if align != "day" and day_mode != "trading":
            raise ValueError("day_mode is only effective when align='day'")
        if session_windows and align != "session":
            raise ValueError("session_windows is only supported when align='session'")
        self.align = align
        self.day_mode = day_mode
        self.session_windows = list(session_windows or [])
        super().__init__(
            source=source,
            freq=freq,
            agg=agg,
            label=label,
            closed=closed,
            emit_partial=emit_partial,
        )

    def load(self, request: FeedSlice) -> pd.DataFrame:
        """Load source data and replay by configured alignment."""
        frame = self.source.load(request)
        normalized = self.normalize(frame, request.symbol)
        return _replay_ohlcv_frame(
            frame=normalized,
            freq=self.freq,
            align=self.align,
            agg=self.agg,
            label=self.label,
            closed=self.closed,
            emit_partial=self.emit_partial,
            timezone=request.timezone,
            session_windows=self.session_windows,
            day_mode=self.day_mode,
        )


class CSVFeedAdapter(BasePandasFeedAdapter):
    """CSV-backed feed adapter draft."""

    name = "csv"

    def __init__(
        self,
        path_template: str,
        read_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize CSV adapter with a symbol path template."""
        self.path_template = path_template
        self.read_kwargs = dict(read_kwargs or {})

    def load(self, request: FeedSlice) -> pd.DataFrame:
        """Load and normalize CSV data for one symbol window."""
        path = Path(self.path_template.format(symbol=request.symbol))
        frame = pd.read_csv(path, **self.read_kwargs)
        data = self.normalize(frame, request.symbol)
        return self._clip_time_range(
            data, request.start_time, request.end_time, request.timezone
        )


class ParquetFeedAdapter(BasePandasFeedAdapter):
    """Parquet-backed feed adapter draft."""

    name = "parquet"

    def __init__(
        self,
        path_template: str,
        read_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize Parquet adapter with a symbol path template."""
        self.path_template = path_template
        self.read_kwargs = dict(read_kwargs or {})

    def load(self, request: FeedSlice) -> pd.DataFrame:
        """Load and normalize Parquet data for one symbol window."""
        path = Path(self.path_template.format(symbol=request.symbol))
        frame = pd.read_parquet(path, **self.read_kwargs)
        data = self.normalize(frame, request.symbol)
        return self._clip_time_range(
            data, request.start_time, request.end_time, request.timezone
        )
