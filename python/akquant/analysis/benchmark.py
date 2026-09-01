"""Benchmark analysis helpers shared by reports, exports, and APIs."""

from typing import Any, Optional, Union, cast

import pandas as pd


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


def build_daily_returns_from_equity(equity_curve: pd.Series) -> pd.Series:
    """Build daily returns from an equity curve.

    口径对齐 Rust ``src/analysis/result.rs::calculate``: 只保留真实存在行情的
    交易日 (``dropna`` 而非 ``ffill``, 后者会给非交易日补出 0.0 伪收益), 且首日
    不产生收益点。
    """
    if equity_curve.empty:
        return pd.Series(dtype=float)
    daily_equity = equity_curve.resample("D").last().dropna()
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
    strategy_returns = build_daily_returns_from_equity(equity_curve)
    return build_benchmark_analysis(
        strategy_returns=strategy_returns,
        benchmark=benchmark,
        annual_factor=annual_factor,
    )
