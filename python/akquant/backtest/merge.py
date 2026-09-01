"""分阶段回测结果合并 (issue #282).

`run_from_checkpoint` 支持把一条长回测切成多段续跑，但每段返回独立的
``BacktestResult``（曲线只覆盖本段）。``merge_results`` 把多段按时间顺序拼成一个
``MergedResult``，提供与 ``BacktestResult`` 一致的只读 pandas 视图，并从合并后的
权益曲线 + 交易明细重算一组**核心指标**。

设计约束：``BacktestResult`` 包装 Rust ``_raw`` 对象，无法在 raw 层合并；且全部
60 个指标均由 Rust 引擎在完整回测中计算，没有「从权益曲线算指标」的可复用入口。
因此 ``MergedResult`` 用纯 pandas 支撑，指标只覆盖能从曲线/交易无歧义推导的子集，
口径对齐 Rust ``src/analysis/result.rs``。
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import pandas as pd

if TYPE_CHECKING:
    from .result import BacktestResult

__all__ = ["MergedResult", "merge_results"]

_DEFAULT_DAYS_PER_YEAR = 252.0


def _concat_curve(curves: List[pd.Series], *, dedupe_boundary: bool) -> pd.Series:
    """Concat curve Series across segments; same-ts keeps latest (upsert)."""
    non_empty = [c for c in curves if c is not None and not c.empty]
    if not non_empty:
        return pd.Series(dtype=float)
    merged = pd.concat(non_empty)
    merged = merged.sort_index(kind="stable")
    if dedupe_boundary:
        merged = merged[~merged.index.duplicated(keep="last")]
    return cast(pd.Series, merged)


def _concat_frame(
    frames: List[pd.DataFrame],
    *,
    sort_col: Optional[str],
    dedupe_col: Optional[str],
) -> pd.DataFrame:
    """Concat DataFrames across segments, sort by time, optional key dedupe."""
    non_empty = [f for f in frames if f is not None and not f.empty]
    if not non_empty:
        return pd.DataFrame()
    merged = pd.concat(non_empty, ignore_index=True)
    if sort_col is not None and sort_col in merged.columns:
        merged = merged.sort_values(sort_col, kind="stable").reset_index(drop=True)
    if dedupe_col is not None and dedupe_col in merged.columns:
        merged = merged.drop_duplicates(subset=[dedupe_col], keep="last").reset_index(
            drop=True
        )
    return merged


def _compute_core_metrics(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    initial_cash: float,
    *,
    days_per_year: float = _DEFAULT_DAYS_PER_YEAR,
) -> Dict[str, Any]:
    """Recompute core metrics from merged equity curve + trades.

    口径对齐 Rust ``src/analysis/result.rs``：
    - total_return = end / initial - 1
    - max_drawdown = max peak-to-trough on equity (cummax)
    - sharpe = mean(daily_ret) * dpy / (std(daily_ret) * sqrt(dpy))
    - sortino = mean(daily_ret) * dpy / (downside_std * sqrt(dpy))
    - calmar = annualized_return / max_drawdown
    仅覆盖能无歧义推导的字段；依赖引擎内部态的指标不在此列。
    """
    metrics: Dict[str, Any] = {}
    if equity_curve.empty:
        return metrics

    equity = equity_curve.astype(float)
    start_equity = float(equity.iloc[0])
    end_equity = float(equity.iloc[-1])
    base = float(initial_cash) if initial_cash > 0 else start_equity

    metrics["initial_market_value"] = base
    metrics["end_market_value"] = end_equity
    metrics["total_pnl"] = end_equity - base
    metrics["total_return_pct"] = (end_equity / base - 1.0) * 100.0 if base else 0.0
    metrics["start_time"] = equity.index[0]
    metrics["end_time"] = equity.index[-1]

    # 最大回撤 (峰值回撤)
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak.where(
        running_peak != 0.0, other=1.0
    )
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    metrics["max_drawdown"] = abs(max_dd)
    metrics["max_drawdown_pct"] = abs(max_dd) * 100.0

    # 年化收益 / 波动 / sharpe / sortino / calmar：基于日频权益
    daily_equity = equity.resample("D").last().dropna()
    daily_ret = daily_equity.pct_change().dropna()
    span_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    years = span_days / 365.25 if span_days > 0 else 0.0
    if years > 0 and base:
        annualized_return = (end_equity / base) ** (1.0 / years) - 1.0
    else:
        annualized_return = 0.0
    metrics["annualized_return"] = annualized_return * 100.0

    if len(daily_ret) >= 2:
        std_dev = float(daily_ret.std(ddof=1))
        mean_ret = float(daily_ret.mean())
        ann_vol = std_dev * math.sqrt(days_per_year)
        metrics["volatility"] = ann_vol * 100.0
        ann_mean = mean_ret * days_per_year
        metrics["sharpe_ratio"] = ann_mean / ann_vol if ann_vol != 0.0 else 0.0
        downside = daily_ret[daily_ret < 0.0]
        if len(downside) >= 1:
            downside_std = float(downside.std(ddof=1)) if len(downside) >= 2 else 0.0
            ann_downside = downside_std * math.sqrt(days_per_year)
            metrics["sortino_ratio"] = (
                ann_mean / ann_downside if ann_downside != 0.0 else 0.0
            )
        else:
            metrics["sortino_ratio"] = 0.0
    else:
        metrics["volatility"] = 0.0
        metrics["sharpe_ratio"] = 0.0
        metrics["sortino_ratio"] = 0.0

    max_dd_abs = abs(max_dd)
    metrics["calmar_ratio"] = (
        annualized_return / max_dd_abs if max_dd_abs != 0.0 else 0.0
    )

    # 交易层指标
    if not trades_df.empty and "net_pnl" in trades_df.columns:
        pnl = pd.to_numeric(trades_df["net_pnl"], errors="coerce").dropna()
        closed = int(len(pnl))
        wins = pnl[pnl > 0.0]
        losses = pnl[pnl < 0.0]
        metrics["closed_trade_count"] = float(closed)
        metrics["winning_trades"] = float(len(wins))
        metrics["losing_trades"] = float(len(losses))
        metrics["win_rate"] = (len(wins) / closed * 100.0) if closed else 0.0
        metrics["loss_rate"] = (len(losses) / closed * 100.0) if closed else 0.0
        gross_profit = float(wins.sum())
        gross_loss = abs(float(losses.sum()))
        metrics["profit_factor"] = (
            gross_profit / gross_loss if gross_loss != 0.0 else 0.0
        )
        if "commission" in trades_df.columns:
            metrics["total_commission"] = float(
                pd.to_numeric(trades_df["commission"], errors="coerce")
                .fillna(0.0)
                .sum()
            )
    else:
        metrics["closed_trade_count"] = 0.0
        metrics["winning_trades"] = 0.0
        metrics["losing_trades"] = 0.0
        metrics["win_rate"] = 0.0
        metrics["loss_rate"] = 0.0
        metrics["profit_factor"] = 0.0

    return metrics


class _MergedMetrics:
    """Attribute-style access to merged core metrics; unknown -> AttributeError."""

    def __init__(self, values: Dict[str, Any]) -> None:
        """Wrap the recomputed core-metric values dict."""
        self._values = values

    def __getattr__(self, name: str) -> Any:
        """Return a core metric, or raise AttributeError for out-of-scope fields."""
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(
            f"MergedResult.metrics 不提供 '{name}'——合并结果仅重算核心指标子集 "
            "(见 merge_results 文档)。完整 60 项指标需在单段完整回测的 "
            "BacktestResult 上读取。"
        )

    def __dir__(self) -> List[str]:
        """List available core-metric names."""
        return list(self._values.keys())


class MergedResult:
    """Merged multi-segment result mirroring ``BacktestResult`` read-only views.

    仅暴露曲线/交易/订单/执行/持仓视图与核心指标；不包装 Rust ``_raw``，
    因此 ``plot`` / ``report_quantstats`` 等依赖 Rust 对象的方法不可用。
    """

    def __init__(
        self,
        *,
        equity_curve: pd.Series,
        cash_curve: pd.Series,
        margin_curve: pd.Series,
        orders_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        executions_df: pd.DataFrame,
        positions_df: pd.DataFrame,
        indicator_outputs: Dict[str, List[Dict[str, Any]]],
        initial_cash: float,
        timezone: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Store pre-merged pandas frames and recomputed core metrics."""
        self._equity_curve = equity_curve
        self._cash_curve = cash_curve
        self._margin_curve = margin_curve
        self._orders_df = orders_df
        self._trades_df = trades_df
        self._executions_df = executions_df
        self._positions_df = positions_df
        self.indicator_outputs = indicator_outputs
        self.initial_cash = initial_cash
        self._timezone = timezone
        self._metrics = metrics

    @staticmethod
    def _to_daily_curve(series: pd.Series) -> pd.Series:
        """Resample a curve to daily end-of-day values."""
        if series.empty:
            return pd.Series(dtype=float)
        return cast(pd.Series, series.resample("D").last().dropna())

    @property
    def equity_curve(self) -> pd.Series:
        """Merged equity curve across all segments."""
        return self._equity_curve

    @property
    def cash_curve(self) -> pd.Series:
        """Merged cash curve across all segments."""
        return self._cash_curve

    @property
    def margin_curve(self) -> pd.Series:
        """Merged margin curve across all segments."""
        return self._margin_curve

    @property
    def equity_curve_daily(self) -> pd.Series:
        """Daily end-of-day equity curve."""
        return self._to_daily_curve(self._equity_curve)

    @property
    def daily_returns(self) -> pd.Series:
        """Daily returns derived from the merged equity curve.

        口径对齐 Rust ``src/analysis/result.rs::calculate`` 与同文件
        ``_recompute_metrics``: 只保留真实存在行情的交易日, 首日不产生收益点。
        """
        equity = self._equity_curve
        if equity.empty:
            return pd.Series(dtype=float)
        daily_equity = equity.resample("D").last().dropna()
        return cast(pd.Series, daily_equity.pct_change().dropna())

    @property
    def orders_df(self) -> pd.DataFrame:
        """Merged orders across all segments."""
        return self._orders_df

    @property
    def trades_df(self) -> pd.DataFrame:
        """Merged closed trades across all segments."""
        return self._trades_df

    @property
    def executions_df(self) -> pd.DataFrame:
        """Merged executions across all segments."""
        return self._executions_df

    @property
    def positions_df(self) -> pd.DataFrame:
        """Merged position snapshots (expired instruments optionally dropped)."""
        return self._positions_df

    @property
    def metrics(self) -> _MergedMetrics:
        """Recomputed core metrics (attribute-style access)."""
        return _MergedMetrics(self._metrics)

    @property
    def metrics_df(self) -> pd.DataFrame:
        """Recomputed core metrics as a value-indexed DataFrame."""
        if not self._metrics:
            return pd.DataFrame(columns=["value"])
        return pd.DataFrame({"value": self._metrics}, index=list(self._metrics.keys()))

    def to_quantstats(self) -> pd.Series:
        """Return merged daily returns (timezone-naive) for quantstats."""
        returns = self.daily_returns.copy()
        if returns.empty:
            return returns
        idx = returns.index
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            returns.index = idx.tz_localize(None)
        return returns

    def __repr__(self) -> str:
        """Compact summary of the merged result."""
        n = len(self._equity_curve)
        return f"<MergedResult equity_points={n} trades={len(self._trades_df)}>"


def _collect_expired_symbols(
    results: List["BacktestResult"],
) -> Dict[str, pd.Timestamp]:
    """Collect expiry_date per symbol from each segment's instrument snapshots."""
    expiry: Dict[str, pd.Timestamp] = {}
    for result in results:
        strategy = getattr(result, "strategy", None)
        snapshots = getattr(strategy, "_instrument_snapshots", None)
        if not isinstance(snapshots, dict):
            continue
        for symbol, snap in snapshots.items():
            raw_expiry = getattr(snap, "expiry_date", None)
            if raw_expiry in (None, 0):
                continue
            try:
                # expiry_date 可能是 int(YYYYMMDD) 或 date
                if isinstance(raw_expiry, int):
                    ts = pd.Timestamp(str(raw_expiry))
                elif isinstance(raw_expiry, (str, date, datetime)):
                    ts = pd.Timestamp(raw_expiry)
                else:
                    continue
            except Exception:
                continue
            expiry[str(symbol)] = ts
    return expiry


def merge_results(
    *results: "BacktestResult",
    drop_expired_instruments: bool = True,
    dedupe_boundary: bool = True,
) -> MergedResult:
    """按时间顺序合并多段分阶段回测结果 (issue #282).

    - 曲线/订单/交易/执行/持仓按时间戳拼接，``dedupe_boundary`` 时去除相邻段
      重叠的边界时间戳（同戳保留后一段）；
    - metrics 由合并后的权益曲线 + 交易明细**重算核心子集**（total_return /
      max_drawdown / sharpe / sortino / calmar / win_rate / profit_factor 等），
      非各段简单相加；
    - ``drop_expired_instruments=True`` 时清理已过期合约的持仓行，防资产爆炸；
    - 要求各段时间**递增**、允许 gap，**重叠**（后段起点 < 前段终点）抛 ValueError。

    :param results: 一段或多段 BacktestResult，按时间先后传入。
    :return: MergedResult（只读 pandas 视图 + 核心指标）。
    """
    if not results:
        raise ValueError("merge_results requires at least one result")

    result_list = list(results)

    equity_segments = [r.equity_curve for r in result_list]

    # 校验时间递增、无重叠
    prev_end: Optional[pd.Timestamp] = None
    for idx, curve in enumerate(equity_segments):
        if curve.empty:
            continue
        seg_start = curve.index[0]
        seg_end = curve.index[-1]
        if prev_end is not None and seg_start < prev_end:
            raise ValueError(
                f"merge_results: segment #{idx} starts at {seg_start} which overlaps "
                f"the previous segment ending at {prev_end}; overlapping segments are "
                "not supported (segments must be time-ordered and non-overlapping)"
            )
        prev_end = seg_end

    initial_cash = float(getattr(result_list[0], "initial_cash", 0.0) or 0.0)
    timezone = getattr(result_list[0], "_timezone", "Asia/Shanghai")

    equity_curve = _concat_curve(equity_segments, dedupe_boundary=dedupe_boundary)
    cash_curve = _concat_curve(
        [r.cash_curve for r in result_list], dedupe_boundary=dedupe_boundary
    )
    margin_curve = _concat_curve(
        [r.margin_curve for r in result_list], dedupe_boundary=dedupe_boundary
    )

    orders_df = _concat_frame(
        [r.orders_df for r in result_list],
        sort_col="created_at",
        dedupe_col="id",
    )
    trades_df = _concat_frame(
        [r.trades_df for r in result_list],
        sort_col="exit_time",
        dedupe_col=None,
    )
    executions_df = _concat_frame(
        [r.executions_df for r in result_list],
        sort_col="timestamp",
        dedupe_col="id",
    )
    positions_df = _concat_frame(
        [r.positions_df for r in result_list],
        sort_col="date",
        dedupe_col=None,
    )

    if drop_expired_instruments and not positions_df.empty:
        expiry = _collect_expired_symbols(result_list)
        if (
            expiry
            and "symbol" in positions_df.columns
            and "date" in positions_df.columns
        ):
            date_col = pd.to_datetime(positions_df["date"], utc=False, errors="coerce")
            keep_mask = pd.Series(True, index=positions_df.index)
            for symbol, exp_ts in expiry.items():
                exp_cmp = exp_ts
                # 对齐 tz：positions date 可能带 tz
                if (
                    getattr(date_col.dtype, "tz", None) is not None
                    and exp_ts.tz is None
                ):
                    exp_cmp = exp_ts.tz_localize(date_col.dt.tz)
                sym_mask = positions_df["symbol"].astype(str) == symbol
                expired_mask = sym_mask & (date_col > exp_cmp)
                keep_mask &= ~expired_mask
            positions_df = positions_df[keep_mask].reset_index(drop=True)

    indicator_outputs = _merge_indicator_outputs(result_list)

    metrics = _compute_core_metrics(equity_curve, trades_df, initial_cash)

    return MergedResult(
        equity_curve=equity_curve,
        cash_curve=cash_curve,
        margin_curve=margin_curve,
        orders_df=orders_df,
        trades_df=trades_df,
        executions_df=executions_df,
        positions_df=positions_df,
        indicator_outputs=indicator_outputs,
        initial_cash=initial_cash,
        timezone=timezone,
        metrics=metrics,
    )


def _merge_indicator_outputs(
    results: List["BacktestResult"],
) -> Dict[str, List[Dict[str, Any]]]:
    """Merge indicator_outputs (definitions/instances/points) across segments."""
    merged: Dict[str, List[Dict[str, Any]]] = {
        "definitions": [],
        "instances": [],
        "points": [],
    }
    seen_def: set = set()
    seen_inst: set = set()
    for result in results:
        outputs = getattr(result, "indicator_outputs", None)
        if not isinstance(outputs, dict):
            continue
        for d in outputs.get("definitions", []):
            key = d.get("name") if isinstance(d, dict) else None
            if key is not None and key in seen_def:
                continue
            if key is not None:
                seen_def.add(key)
            merged["definitions"].append(d)
        for inst in outputs.get("instances", []):
            key = inst.get("id") if isinstance(inst, dict) else None
            if key is not None and key in seen_inst:
                continue
            if key is not None:
                seen_inst.add(key)
            merged["instances"].append(inst)
        merged["points"].extend(outputs.get("points", []))
    return merged
