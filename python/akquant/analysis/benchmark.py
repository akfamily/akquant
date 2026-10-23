"""Benchmark analysis helpers shared by reports, exports, and APIs."""

import datetime as _dt
from typing import Any, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd

TradingDayEnds = Sequence[Tuple[_dt.date, int]]
"""引擎实际开过的交易日及其收盘时刻 ``[(交易日, 收盘纳秒)]``."""

_DEFAULT_TIMEZONE = "Asia/Shanghai"


def _utc_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """带时区索引转为 UTC 纳秒整数数组（NaT 为 int64 最小值）."""
    naive_utc = index.tz_convert("UTC").tz_localize(None)
    return np.asarray(naive_utc.as_unit("ns").to_numpy(dtype="datetime64[ns]")).view(
        np.int64
    )


def _trading_day_labels(
    index: pd.DatetimeIndex, ends: Optional[TradingDayEnds], tz: Optional[str]
) -> pd.DatetimeIndex:
    """把时间索引映射成所属交易日的本地 0 点.

    交易日 T 覆盖 ``(上一交易日收盘, T 收盘]``：取第一个收盘时刻不早于该时刻的
    已开交易日（夜盘点归下一交易日），口径与 Rust
    ``BacktestResult::trading_day_key`` 一致。``ends`` 为空，或时刻晚于全部
    收盘时刻时，退回本地自然日（``index.normalize()``）。

    带时区的索引返回 ``tz`` 下的带时区标签；不带时区的索引按 ``tz`` 本地时间
    解释，返回不带时区的标签。

    :param index: 待映射的时间索引。
    :param ends: 交易日及收盘时刻（纳秒），来自结果对象的 ``trading_day_ends``。
    :param tz: 结果时区；为空时取索引自身时区，再退回 ``Asia/Shanghai``。
    :return: 与 ``index`` 等长的交易日标签索引。
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if not ends or len(index) == 0:
        return index.normalize()
    naive = index.tz is None
    zone = tz or (str(index.tz) if not naive else _DEFAULT_TIMEZONE)
    if naive:
        local = index.tz_localize(
            zone,
            ambiguous=np.ones(len(index), dtype=bool),
            nonexistent="shift_forward",
        )
    else:
        local = index.tz_convert(zone)

    end_ns = np.asarray([int(end) for _, end in ends], dtype=np.int64)
    order = np.argsort(end_ns, kind="stable")
    end_ns = end_ns[order]
    days = pd.DatetimeIndex(
        pd.to_datetime([pd.Timestamp(ends[int(i)][0]) for i in order])
    )
    day_labels = days.tz_localize(
        zone,
        ambiguous=np.ones(len(days), dtype=bool),
        nonexistent="shift_forward",
    )
    day_labels_ns = _utc_ns(day_labels)

    ts = _utc_ns(local)
    pos = np.searchsorted(end_ns, ts, side="left")
    hit = (pos < len(end_ns)) & ~np.asarray(local.isna())
    fallback = _utc_ns(local.normalize())
    out = np.where(hit, day_labels_ns[np.minimum(pos, len(end_ns) - 1)], fallback)
    labels = pd.DatetimeIndex(out.view("M8[ns]")).tz_localize("UTC").tz_convert(zone)
    if naive:
        labels = labels.tz_localize(None)
    return labels.rename(index.name)


def daily_last_by_trading_day(
    series: pd.Series,
    ends: Optional[TradingDayEnds] = None,
    tz: Optional[str] = None,
) -> pd.Series:
    """按交易日取每日最后一个值（只保留有数据的交易日）.

    ``ends`` 为空时与 ``series.resample("D").last().dropna()`` 完全一致。
    """
    if series.empty:
        return pd.Series(dtype=float)
    if not ends:
        return cast(pd.Series, series.resample("D").last().dropna())
    labels = _trading_day_labels(cast(pd.DatetimeIndex, series.index), ends, tz)
    return cast(pd.Series, series.groupby(labels).last().dropna())


def sum_by_trading_day(
    series: pd.Series,
    ends: Optional[TradingDayEnds] = None,
    tz: Optional[str] = None,
) -> pd.Series:
    """按交易日求和，日期轴连续（无数据的日期为 0），与 ``resample("D").sum()`` 同形."""
    if series.empty or not ends:
        return cast(pd.Series, series.resample("D").sum())
    labels = _trading_day_labels(cast(pd.DatetimeIndex, series.index), ends, tz)
    relabeled = series.set_axis(labels)
    return cast(pd.Series, relabeled.resample("D").sum())


def result_trading_day_ends(result: Any) -> Tuple[list[Tuple[_dt.date, int]], str]:
    """取结果对象的 ``trading_day_ends`` 与时区（缺失时为空列表与默认时区）."""
    ends = getattr(result, "trading_day_ends", None)
    tz = getattr(result, "_timezone", None) or _DEFAULT_TIMEZONE
    return (list(ends) if ends else []), str(tz)


def normalize_curve_freq(curve_freq: str) -> str:
    """Normalize curve frequency option."""
    value = str(curve_freq).strip()
    if value.lower() == "raw":
        return "raw"
    if value.upper() == "D":
        return "D"
    raise ValueError("curve_freq must be 'raw' or 'D'")


def resolve_equity_curve(result: Any, curve_freq: str) -> pd.Series:
    """Resolve equity curve for analysis rendering."""
    normalized_curve_freq = normalize_curve_freq(curve_freq)
    if normalized_curve_freq == "D" and hasattr(result, "equity_curve_daily"):
        series = cast(pd.Series, result.equity_curve_daily)
        if not series.empty:
            return series
    return cast(pd.Series, result.equity_curve)


def build_daily_returns_from_equity(
    equity_curve: pd.Series,
    trading_day_ends: Optional[TradingDayEnds] = None,
    timezone: Optional[str] = None,
) -> pd.Series:
    """Build daily returns from an equity curve.

    口径对齐 Rust ``src/analysis/result.rs::calculate``: 只保留真实存在行情的
    交易日 (``dropna`` 而非 ``ffill``, 后者会给非交易日补出 0.0 伪收益), 且首日
    不产生收益点。给出 ``trading_day_ends``（结果对象的同名属性）时按交易日分组
    （夜盘点归下一交易日），否则按本地自然日。
    """
    if equity_curve.empty:
        return pd.Series(dtype=float)
    daily_equity = daily_last_by_trading_day(equity_curve, trading_day_ends, timezone)
    returns = daily_equity.pct_change().dropna()
    return cast(pd.Series, returns)


def normalize_returns_series_with_reason(
    series: pd.Series, series_label: str = "收益序列"
) -> tuple[pd.Series, Optional[str]]:
    """Normalize returns to a daily index and report validation errors."""
    cleaned = series.copy()
    cleaned = pd.to_numeric(cleaned, errors="coerce")
    cleaned = cast(pd.Series, cleaned.dropna())
    if cleaned.empty:
        return cast(pd.Series, cleaned), f"{series_label}为空"

    raw_index = cleaned.index
    if isinstance(raw_index, pd.RangeIndex):
        return (
            pd.Series(dtype=float),
            f"{series_label}索引必须为日期索引，当前为 RangeIndex",
        )
    if not isinstance(raw_index, pd.DatetimeIndex) and pd.api.types.is_numeric_dtype(
        raw_index.dtype
    ):
        return (
            pd.Series(dtype=float),
            f"{series_label}索引必须为 DatetimeIndex 或可解析的日期索引",
        )

    if isinstance(raw_index, pd.DatetimeIndex):
        dt_index = raw_index
    else:
        dt_index = pd.DatetimeIndex(pd.to_datetime(raw_index, errors="coerce"))

    valid_mask = ~pd.isna(dt_index)
    if not bool(valid_mask.any()):
        return pd.Series(dtype=float), f"{series_label}索引无法解析为日期"

    cleaned = cleaned.loc[valid_mask].copy()
    dt_index = dt_index[valid_mask]
    if dt_index.empty:
        return pd.Series(dtype=float), f"{series_label}索引无法解析为日期"

    if dt_index.tz is not None:
        # Preserve the local trading day before dropping timezone info.
        dt_index = dt_index.tz_localize(None)
    cleaned.index = dt_index.normalize()
    cleaned = cast(pd.Series, cleaned.groupby(cleaned.index).last())
    cleaned = cast(pd.Series, cleaned.sort_index().astype(float))
    return cleaned, None


def normalize_returns_series(series: pd.Series) -> pd.Series:
    """Normalize return series index to a timezone-naive daily index."""
    normalized, _ = normalize_returns_series_with_reason(series)
    return normalized


def resolve_benchmark_returns(
    benchmark: Optional[Union[str, pd.Series]], strategy_returns: pd.Series
) -> tuple[Optional[pd.Series], str]:
    """Resolve benchmark input into an aligned return series and label."""
    if benchmark is None:
        return None, "未提供基准"
    if isinstance(benchmark, str):
        return None, f"暂不支持自动拉取基准: {benchmark}"
    if not isinstance(benchmark, pd.Series):
        return None, "基准类型错误，需为 pd.Series 或 str"
    benchmark_label = (
        str(benchmark.name)
        if benchmark.name is not None and str(benchmark.name).strip()
        else "Benchmark"
    )
    benchmark_series, benchmark_reason = normalize_returns_series_with_reason(
        benchmark, f"基准序列 {benchmark_label}"
    )
    if benchmark_reason is not None:
        return None, benchmark_reason
    quantile_95 = float(benchmark_series.abs().quantile(0.95))
    if quantile_95 > 2.0:
        benchmark_series = cast(pd.Series, benchmark_series.pct_change().fillna(0.0))
    strategy_series, strategy_reason = normalize_returns_series_with_reason(
        strategy_returns, "策略收益序列"
    )
    if strategy_reason is not None:
        return None, f"{strategy_reason}，无法对齐基准: {benchmark_label}"
    if strategy_series.index.intersection(benchmark_series.index).empty:
        return None, f"策略与基准无重叠区间: {benchmark_label}"
    return benchmark_series, benchmark_label


def _json_value(value: float) -> Optional[float]:
    """Convert pandas/NaN values to JSON-friendly numbers."""
    if pd.isna(value):
        return None
    return float(value)


def _empty_benchmark_analysis(reason: str, benchmark_label: str) -> dict[str, Any]:
    """Build an empty structured benchmark analysis payload."""
    return {
        "schema_version": "1.0",
        "available": False,
        "reason": reason,
        "benchmark": {"label": benchmark_label},
        "summary": {
            "total_excess": None,
            "annual_excess": None,
            "tracking_error": None,
            "information_ratio": None,
            "beta": None,
            "alpha": None,
        },
        "series": [],
        "meta": {
            "aligned_points": 0,
            "start_date": None,
            "end_date": None,
            "annual_factor": 252.0,
        },
    }


def build_benchmark_analysis(
    strategy_returns: pd.Series,
    benchmark: Optional[Union[str, pd.Series]],
    annual_factor: float = 252.0,
) -> dict[str, Any]:
    """Build a structured benchmark analysis payload."""
    benchmark_returns, benchmark_label = resolve_benchmark_returns(
        benchmark, strategy_returns
    )
    if benchmark_returns is None:
        reason = (
            "未提供可用基准数据，已跳过相对收益分析。"
            if benchmark is None
            else f"未生成基准对比: {benchmark_label}"
        )
        return _empty_benchmark_analysis(reason, benchmark_label)

    strategy_series = normalize_returns_series(strategy_returns)
    aligned = pd.concat(
        [strategy_series.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return _empty_benchmark_analysis(
            "策略与基准收益率无可用重叠样本，已跳过对比。",
            benchmark_label,
        )

    strategy_aligned = cast(pd.Series, aligned["strategy"].astype(float))
    benchmark_aligned = cast(pd.Series, aligned["benchmark"].astype(float))
    excess = cast(pd.Series, strategy_aligned - benchmark_aligned)

    annual_excess = float(excess.mean() * annual_factor)
    tracking_error = float(excess.std(ddof=0) * (annual_factor**0.5))
    info_ratio = annual_excess / tracking_error if tracking_error > 0 else float("nan")

    strategy_total = float((1.0 + strategy_aligned).to_numpy(dtype=float).prod())
    benchmark_total = float((1.0 + benchmark_aligned).to_numpy(dtype=float).prod())
    total_excess = (
        strategy_total / benchmark_total - 1.0 if benchmark_total > 0 else float("nan")
    )

    mean_strategy = float(strategy_aligned.mean())
    mean_benchmark = float(benchmark_aligned.mean())
    variance_benchmark = float(benchmark_aligned.var(ddof=0))
    beta = float("nan")
    alpha = float("nan")
    if variance_benchmark > 0:
        covariance = float(
            (
                (strategy_aligned - mean_strategy)
                * (benchmark_aligned - mean_benchmark)
            ).mean()
        )
        beta = covariance / variance_benchmark
        alpha = (mean_strategy - beta * mean_benchmark) * annual_factor

    cumulative_strategy = cast(pd.Series, (1.0 + strategy_aligned).cumprod() - 1.0)
    cumulative_benchmark = cast(pd.Series, (1.0 + benchmark_aligned).cumprod() - 1.0)
    cumulative_excess = cast(pd.Series, cumulative_strategy - cumulative_benchmark)

    series_records = []
    series_frame = pd.DataFrame(
        {
            "strategy_return": strategy_aligned,
            "benchmark_return": benchmark_aligned,
            "excess_return": excess,
            "strategy_cum_return": cumulative_strategy,
            "benchmark_cum_return": cumulative_benchmark,
            "excess_cum_return": cumulative_excess,
        }
    )
    for dt, row in series_frame.iterrows():
        series_records.append(
            {
                "date": cast(pd.Timestamp, dt).strftime("%Y-%m-%d"),
                "strategy_return": float(row["strategy_return"]),
                "benchmark_return": float(row["benchmark_return"]),
                "excess_return": float(row["excess_return"]),
                "strategy_cum_return": float(row["strategy_cum_return"]),
                "benchmark_cum_return": float(row["benchmark_cum_return"]),
                "excess_cum_return": float(row["excess_cum_return"]),
            }
        )

    return {
        "schema_version": "1.0",
        "available": True,
        "reason": None,
        "benchmark": {"label": benchmark_label},
        "summary": {
            "total_excess": _json_value(total_excess),
            "annual_excess": _json_value(annual_excess),
            "tracking_error": _json_value(tracking_error),
            "information_ratio": _json_value(info_ratio),
            "beta": _json_value(beta),
            "alpha": _json_value(alpha),
        },
        "series": series_records,
        "meta": {
            "aligned_points": int(len(aligned)),
            "start_date": cast(pd.Timestamp, aligned.index[0]).strftime("%Y-%m-%d"),
            "end_date": cast(pd.Timestamp, aligned.index[-1]).strftime("%Y-%m-%d"),
            "annual_factor": float(annual_factor),
        },
    }


def benchmark_analysis_from_result(
    result: Any,
    benchmark: Optional[Union[str, pd.Series]],
    curve_freq: str = "raw",
    annual_factor: float = 252.0,
) -> dict[str, Any]:
    """Build structured benchmark analysis directly from a backtest-like result."""
    equity_curve = resolve_equity_curve(result, curve_freq)
    ends, tz = result_trading_day_ends(result)
    strategy_returns = build_daily_returns_from_equity(equity_curve, ends, tz)
    return build_benchmark_analysis(
        strategy_returns=strategy_returns,
        benchmark=benchmark,
        annual_factor=annual_factor,
    )
