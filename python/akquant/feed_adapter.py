from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol, Union, cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .akquant import AssetType, Bar, SessionResolver, WindowAggregator, from_arrays
from .utils import load_bar_from_df


@dataclass(frozen=True)
class FeedSlice:
    """Data request window for a single symbol."""

    symbol: str
    start_time: pd.Timestamp | None = None
    end_time: pd.Timestamp | None = None
    # Timezone for the run: naive start/end bounds AND naive source index values
    # are both interpreted in it (default Asia/Shanghai when None).
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
AssetTypeArg = Union[AssetType, str, Mapping[str, Union[AssetType, str]], None]
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
        sessions: Mapping[str, str] | None = None,
        trading_calendar: Iterable[Any] | None = None,
        asset_type: AssetTypeArg = None,
        base_interval_min: int | None = None,
    ) -> "ReplayFeedAdapter":
        """Create a replay adapter view.

        ``align="session"``（未给 ``session_windows``）与 ``align="day"`` +
        ``day_mode="trading"`` 用引擎 ``subscribe_bars`` 同一个窗口聚合器
        (:class:`akquant.WindowAggregator`) 切分, 对同一数据、同一时段表产出相同的
        窗口: 日内窗口按(交易日, 交易块)切分、标签为本地时钟 ceil 到周期但不超过块
        结束(A 股 1h: 10:00、11:00、11:30、14:00、15:00); 日线窗口按交易日切分(夜盘
        归下一交易日)、标签为交易日收盘时刻。

        交易块截断只对默认 ``label="right", closed="right"`` 且 ``freq`` 为
        ``Nmin`` / ``Nh`` / ``1D`` 时生效; 其余情况(含 ``align="global"``、
        ``day_mode="calendar"``、手写 ``session_windows``)沿用 pandas ``resample``
        按本地日期(或手写时段)分区的旧实现。

        :param sessions: 代码前缀 → 时段模板名, 同 ``run_backtest`` 的 ``sessions``。
        :param trading_calendar: 交易日列表, 同 ``run_backtest`` 的
            ``trading_calendar``。
        :param asset_type: 资产类型(决定默认时段表), 同 ``run_backtest``, 默认股票;
            也可传 ``{symbol: asset_type}``, 未列出的品种按股票。
        :param base_interval_min: 源数据 bar 周期(分钟); 不传时按相邻时间戳间隔的
            众数推断。
        :param emit_partial: ``False`` 时丢弃数据未到达标签的窗口中, 位于该交易日
            最后的那一个(即每个交易日的尾部未闭合窗口与数据末尾的窗口)。
        """
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
            sessions=sessions,
            trading_calendar=trading_calendar,
            asset_type=asset_type,
            base_interval_min=base_interval_min,
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
            # 终点早于起点表示跨午夜(如 21:00～次日 02:30): 午夜后的部分归前一天的段
            if end < start and minute_of_day >= start:
                matched_key = ts.normalize() + pd.Timedelta(minutes=start)
                break
            if end < start and minute_of_day < end:
                matched_key = (
                    ts.normalize() - pd.Timedelta(days=1) + pd.Timedelta(minutes=start)
                )
                break
        if matched_key is None:
            keep_mask.append(False)
            key_values.append(pd.Timestamp("1970-01-01", tz=localized.tz))
        else:
            keep_mask.append(True)
            key_values.append(matched_key)
    return pd.DatetimeIndex(key_values), keep_mask


def _engine_window_freq(freq: str) -> str | None:
    """把 pandas 频率转成窗口聚合器的写法(Nmin/Nh/1d); 不支持的返回 None."""
    try:
        offset = to_offset(freq)
    except (TypeError, ValueError):
        return None
    n = int(getattr(offset, "n", 0))
    if n <= 0:
        return None
    if isinstance(offset, pd.offsets.Minute):
        return f"{n}min"
    if isinstance(offset, pd.offsets.Hour):
        return f"{n}h"
    if isinstance(offset, pd.offsets.Day) and n == 1:
        return "1d"
    return None


def _resolve_asset_type(asset_type: AssetTypeArg, symbol: str) -> AssetType:
    from .bar_generator import _to_asset_type

    if asset_type is None:
        return AssetType.Stock
    if isinstance(asset_type, Mapping):
        value = asset_type.get(symbol)
        return AssetType.Stock if value is None else _to_asset_type(value)
    return _to_asset_type(asset_type)


def _infer_base_interval_min(ts_ns: np.ndarray) -> int | None:
    """相邻时间戳间隔的众数(分钟); 不是整分钟或数据不足时返回 None."""
    if ts_ns.size < 2:
        return None
    diffs = np.diff(ts_ns)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    values, counts = np.unique(diffs, return_counts=True)
    mode = int(values[int(np.argmax(counts))])
    minute_ns = 60 * 1_000_000_000
    if mode % minute_ns != 0:
        return None
    return mode // minute_ns


def _engine_windows_for_symbol(
    source: pd.DataFrame,
    symbol: str,
    window_freq: str,
    used_agg: Mapping[str, str],
    emit_partial: bool,
    timezone: str,
    resolver: SessionResolver,
    asset_type: AssetType,
    base_interval_min: int | None,
) -> pd.DataFrame:
    """用引擎窗口聚合器把单品种基础 bar 聚合成窗口(与 subscribe_bars 同口径).

    OHLCV 取聚合器的结果; ``agg`` 里的其他列(或改写了默认聚合方式的列)按聚合器
    给每行分配的窗口用 pandas 分组计算。
    """
    price_cols = ["open", "high", "low", "close"]
    source = source.loc[source[price_cols].notna().all(axis=1).to_numpy()]
    if source.empty:
        return source.iloc[0:0]
    source_index = pd.DatetimeIndex(source.index)
    local_index = _localize_input_index(source_index, timezone)
    utc_naive = local_index.tz_convert("UTC").tz_localize(None).as_unit("ns")
    ts_ns = utc_naive.to_numpy().view(np.int64)
    volumes = (
        source["volume"].fillna(0.0).astype(float).to_numpy()
        if "volume" in source.columns
        else np.zeros(len(source))
    )
    bars = from_arrays(
        ts_ns,
        source["open"].astype(float).to_numpy(),
        source["high"].astype(float).to_numpy(),
        source["low"].astype(float).to_numpy(),
        source["close"].astype(float).to_numpy(),
        volumes,
        symbol,
        None,
        None,
    )
    base = (
        base_interval_min
        if base_interval_min is not None
        else _infer_base_interval_min(ts_ns)
    )
    aggregator = WindowAggregator(
        window_freq,
        base_interval_min=base,
        resolver=resolver,
        timezone=timezone,
        asset_types={symbol: asset_type},
    )

    # 逐行喂入, 同时记录每行归属的窗口序号(窗口按闭合顺序编号)。单周期、单品种下
    # update 至多返回两根: 被新 bar 挤出的旧窗口(不含本行)与本行即时闭合的窗口。
    windows: list[Bar] = []
    row_window = np.empty(len(bars), dtype=np.int64)
    for i, bar in enumerate(bars):
        closed_now = aggregator.update(bar)
        count = len(closed_now)
        if count == 0:
            row_window[i] = len(windows)
        elif count == 2:
            row_window[i] = len(windows) + 1
        elif aggregator.current(symbol) is None:
            row_window[i] = len(windows)
        else:
            row_window[i] = len(windows) + 1
        windows.extend(closed_now)
    windows.extend(aggregator.flush())
    if not windows:
        return source.iloc[0:0]

    labels = np.fromiter((b.timestamp for b in windows), np.int64, len(windows))
    keep = np.ones(len(windows), dtype=bool)
    if not emit_partial:
        last_ts = np.full(len(windows), np.iinfo(np.int64).min, dtype=np.int64)
        first_ts = np.full(len(windows), np.iinfo(np.int64).max, dtype=np.int64)
        np.maximum.at(last_ts, row_window, ts_ns)
        np.minimum.at(first_ts, row_window, ts_ns)

        def trading_day(ts: int) -> Any:
            try:
                return resolver.resolve_instant(symbol, ts - 1, asset_type)[0]
            except Exception:  # noqa: BLE001 - 解析失败时按本地日期
                return pd.Timestamp(ts, tz="UTC").tz_convert(timezone).date()

        # 数据未到达标签的窗口, 若是该交易日的最后一个窗口(或数据末尾)即为未闭合尾部
        for k in np.flatnonzero(last_ts < labels):
            if k == len(windows) - 1 or trading_day(int(first_ts[k])) != trading_day(
                int(first_ts[k + 1])
            ):
                keep[k] = False

    default_agg = _default_agg_mapping()
    columns: dict[str, Any] = {}
    extra_agg: dict[str, str] = {}
    for col, how in used_agg.items():
        if default_agg.get(col) == how:
            columns[col] = [float(getattr(b, col)) for b in windows]
        else:
            extra_agg[col] = how
    if extra_agg:
        grouped = (
            source[list(extra_agg)]
            .groupby(row_window, sort=True)
            .agg(cast(Any, extra_agg))
            .reindex(range(len(windows)))
        )
        for col in extra_agg:
            columns[col] = grouped[col].to_numpy()

    index = pd.DatetimeIndex(labels, tz="UTC")
    if source_index.tz is not None:
        index = index.tz_convert(source_index.tz)
    else:
        index = index.tz_convert(timezone).tz_localize(None)
    index.name = source_index.name
    out = pd.DataFrame(
        {col: columns[col] for col in used_agg if col in columns}, index=index
    )
    return cast(pd.DataFrame, out.loc[keep])


def _replay_engine_frame(
    frame: pd.DataFrame,
    window_freq: str,
    agg: Mapping[str, str] | None,
    emit_partial: bool,
    timezone: str | None,
    sessions: Mapping[str, str] | None,
    trading_calendar: Iterable[Any] | None,
    asset_type: AssetTypeArg,
    base_interval_min: int | None,
    symbol: str,
) -> pd.DataFrame:
    from .backtest.engine import _normalize_trading_calendar

    tz = timezone or DEFAULT_INPUT_TIMEZONE
    resolver = SessionResolver(
        sessions=dict(sessions) if sessions is not None else None,
        trading_calendar=_normalize_trading_calendar(trading_calendar),
        timezone=tz,
    )
    agg_map = dict(_default_agg_mapping())
    if agg:
        agg_map.update(dict(agg))

    if "symbol" in frame.columns:
        by_symbol: Iterable[tuple[Any, pd.DataFrame]] = frame.groupby(
            frame["symbol"].astype(str), sort=False
        )
    else:
        by_symbol = [(symbol, frame)]

    chunks: list[pd.DataFrame] = []
    for sym, group in by_symbol:
        source = group.drop(columns=["symbol"]) if "symbol" in group.columns else group
        source = source.sort_index(kind="stable")
        used_agg = {k: v for k, v in agg_map.items() if k in source.columns}
        sym_name = str(sym)
        out = _engine_windows_for_symbol(
            source,
            sym_name,
            window_freq,
            used_agg,
            emit_partial,
            tz,
            resolver,
            _resolve_asset_type(asset_type, sym_name),
            base_interval_min,
        )
        if out.empty:
            continue
        if "symbol" in frame.columns:
            out["symbol"] = sym_name
        chunks.append(out)

    if not chunks:
        return cast(pd.DataFrame, frame.iloc[0:0].copy())
    result = pd.concat(chunks, axis=0)
    result.sort_index(inplace=True, kind="stable")
    return result


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
    sessions: Mapping[str, str] | None = None,
    trading_calendar: Iterable[Any] | None = None,
    asset_type: AssetTypeArg = None,
    base_interval_min: int | None = None,
    symbol: str = "",
) -> pd.DataFrame:
    window_freq = _engine_window_freq(freq)
    engine_path = (
        window_freq is not None
        and label == "right"
        and closed == "right"
        and (
            (align == "session" and not session_windows)
            or (align == "day" and day_mode == "trading" and window_freq == "1d")
        )
    )
    if engine_path and not frame.empty:
        return _replay_engine_frame(
            frame,
            cast(str, window_freq),
            agg,
            emit_partial,
            timezone,
            sessions,
            trading_calendar,
            asset_type,
            base_interval_min,
            symbol,
        )
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
        sessions: Mapping[str, str] | None = None,
        trading_calendar: Iterable[Any] | None = None,
        asset_type: AssetTypeArg = None,
        base_interval_min: int | None = None,
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
        if base_interval_min is not None and base_interval_min <= 0:
            raise ValueError(
                "base_interval_min 必须为正整数（源数据 bar 的分钟数），不知道时不传"
            )
        self.sessions = dict(sessions) if sessions is not None else None
        self.trading_calendar = (
            list(trading_calendar) if trading_calendar is not None else None
        )
        self.asset_type = asset_type
        self.base_interval_min = base_interval_min
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
            sessions=self.sessions,
            trading_calendar=self.trading_calendar,
            asset_type=self.asset_type,
            base_interval_min=self.base_interval_min,
            symbol=str(request.symbol),
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
