"""Trade analysis plotting module."""

from typing import Optional

import numpy as np
import pandas as pd

from .utils import check_plotly, get_color, go, make_subplots


def plot_trades_distribution(
    trades_df: pd.DataFrame, theme: str = "light"
) -> Optional["go.Figure"]:
    """Plot PnL distribution of trades."""
    if not check_plotly():
        return None

    if trades_df.empty:
        print("No trades to plot.")
        return None

    fig = go.Figure()

    # PnL Histogram
    # Convert to list for serialization
    pnl_data = trades_df["pnl"].fillna(0.0).tolist()

    fig.add_trace(
        go.Histogram(
            x=pnl_data,
            name="盈亏分布",
            marker_color="blue",
            opacity=0.7,
        )
    )

    fig.update_layout(
        title="交易盈亏分布 (Trade PnL Distribution)",
        xaxis_title="盈亏 (PnL)",
        yaxis_title="频次 (Count)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def _coerce_duration_to_timedelta(column: pd.Series) -> Optional[pd.Series]:
    """Best-effort conversion of a ``duration`` column to Timedelta.

    ``BacktestResult.trades_df`` hands out a real ``timedelta64`` column, but
    the same frame reaches this function after round-trips that erase the
    dtype -- ``to_csv`` writes Timedelta as ``"22 days"`` and reads it back as
    ``object``, and raw engine payloads arrive as integer nanoseconds. All of
    those still describe a duration, so coerce instead of giving up.

    :param column: Raw ``duration`` column of any dtype.
    :return: Timedelta Series, or ``None`` when nothing could be parsed.
    """
    if pd.api.types.is_timedelta64_dtype(column):
        converted = column
    else:
        try:
            if pd.api.types.is_numeric_dtype(column):
                # Bare numbers are nanoseconds -- the engine's own unit for
                # this field (``ClosedTrade.duration``).
                converted = pd.to_timedelta(column, unit="ns", errors="coerce")
            else:
                # Strings like "22 days" / "0 days 02:00:00" (what ``to_csv``
                # leaves behind) carry their own unit; passing one would raise.
                converted = pd.to_timedelta(column, errors="coerce")
        except (ValueError, TypeError):
            return None

    if converted.isna().all():
        return None
    return converted


def _select_duration_unit(durations: pd.Series) -> tuple[pd.Series, str]:
    """Pick a human-readable unit for a Timedelta column.

    :param durations: Timedelta Series with at least one non-null entry.
    :return: ``(values_in_unit, unit_label)``.
    """
    seconds = durations.dt.total_seconds()
    # ``mean`` skips NaT, so a partially-filled column still picks a sensible
    # unit instead of falling through on a single missing value.
    average = seconds.mean()

    if average < 3600:
        return seconds / 60, "分钟"
    if average < 24 * 3600:
        return seconds / 3600, "小时"
    return seconds / (24 * 3600), "天"


def plot_pnl_vs_duration(
    trades_df: pd.DataFrame, theme: str = "light"
) -> Optional["go.Figure"]:
    """Plot PnL vs Duration scatter plot.

    :param trades_df: Closed-trade frame; needs ``pnl``/``symbol`` plus either
                      ``duration`` or ``duration_bars``.
    :param theme: ``"light"`` or ``"dark"``.
    :return: Scatter figure, or ``None`` when plotly is missing / no trades.
    """
    if not check_plotly():
        return None

    if trades_df.empty:
        return None

    durations = (
        _coerce_duration_to_timedelta(trades_df["duration"])
        if "duration" in trades_df.columns
        else None
    )

    if durations is not None:
        duration_values, unit_label = _select_duration_unit(durations)
    elif "duration_bars" in trades_df.columns:
        # Bar counts are integers and never lose fidelity, so they beat
        # guessing at the unit of an unparseable ``duration``. Labelled as
        # bars rather than days because the conversion depends on the data
        # frequency, which is not available here.
        duration_values = pd.to_numeric(trades_df["duration_bars"], errors="coerce")
        unit_label = "根K线"
    else:
        return None

    duration_list = duration_values.fillna(0.0).tolist()
    pnl_values = trades_df["pnl"].fillna(0.0).tolist()
    symbols = trades_df["symbol"].astype(str).tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=duration_list,
            y=pnl_values,
            mode="markers",
            marker=dict(
                size=8,
                color=pnl_values,
                colorscale="RdBu",
                showscale=True,
                colorbar=dict(title="盈亏 (PnL)"),
            ),
            text=symbols,
            hovertemplate=(
                f"<b>%{{text}}</b><br>持仓: %{{x:.1f}} {unit_label}<br>"
                "盈亏: %{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="盈亏 vs 持仓时间 (PnL vs Duration)",
        xaxis_title=f"持仓时间 ({unit_label})",
        yaxis_title="盈亏 (PnL)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def plot_rolling_metrics(
    returns: pd.Series, window: int = 126, theme: str = "light"
) -> Optional["go.Figure"]:
    """
    Plot rolling metrics (Sharpe, Volatility).

    :param returns: Daily returns series.
    :param window: Rolling window size (default 126 days ~ 6 months).
    :param theme: Plot theme.
    """
    if not check_plotly():
        return None

    if returns.empty:
        return None

    # Calculate Rolling Stats
    # Annualized Volatility
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    # Annualized Sharpe (assuming risk-free rate = 0 for simplicity)
    rolling_sharpe = (
        returns.rolling(window).mean() / returns.rolling(window).std()
    ) * np.sqrt(252)

    # Create Subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"滚动夏普比率 (Rolling Sharpe, {window} 天)",
            f"滚动波动率 (Rolling Volatility, {window} 天)",
        ),
        vertical_spacing=0.1,
    )

    # 1. Rolling Sharpe
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.fillna(0).tolist(),
            name="夏普比率",
            line=dict(color=get_color(theme, "up_color")),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="夏普比率", row=1, col=1)

    # 2. Rolling Volatility
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.fillna(0).tolist(),
            name="波动率",
            line=dict(color="#ffa726"),  # Orange
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="波动率", tickformat=".2%", row=2, col=1)
    fig.update_xaxes(title_text="日期", tickformat="%Y-%m-%d", row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_white" if theme == "light" else "plotly_dark",
        showlegend=False,
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def plot_returns_distribution(
    returns: pd.Series, theme: str = "light"
) -> Optional["go.Figure"]:
    """Plot distribution of daily returns."""
    if not check_plotly() or returns.empty:
        return None

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=returns.tolist(),
            name="收益率",
            histnorm="probability density",
            marker_color=get_color(theme, "up_color"),
            opacity=0.7,
        )
    )

    # Normal Distribution Fit
    mu, std = returns.mean(), returns.std()
    if std > 0:
        x = np.linspace(returns.min(), returns.max(), 100)
        p = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mu) / std) ** 2)

        fig.add_trace(
            go.Scatter(
                x=x.tolist(),
                y=p.tolist(),
                mode="lines",
                name="正态分布",
                line=dict(color="gray", dash="dash"),
            )
        )

    fig.update_layout(
        title="日收益率分布 (Daily Returns Distribution)",
        xaxis_title="收益率 (Returns)",
        yaxis_title="密度 (Density)",
        xaxis_tickformat=".2%",
        template="plotly_white" if theme == "light" else "plotly_dark",
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def plot_yearly_returns(
    returns: pd.Series, theme: str = "light"
) -> Optional["go.Figure"]:
    """Plot yearly returns bar chart."""
    if not check_plotly() or returns.empty:
        return None

    # Calculate Yearly Returns
    # 'YE' is new alias for Year End in pandas, 'A' or 'Y' might be deprecated or old.
    # Check pandas version compatibility. 'Y' is standard. 'YE' is newer (pandas 2.2+).
    # Let's use 'Y' for broader compatibility or check.
    try:
        yearly_ret = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    except ValueError:
        # Fallback for older pandas
        yearly_ret = returns.resample("Y").apply(lambda x: (1 + x).prod() - 1)

    if len(yearly_ret) > 0:
        if hasattr(yearly_ret.index, "year"):
            years = yearly_ret.index.year
        else:
            years = yearly_ret.index
        # Convert Series to simple list or array for plotting
        values = yearly_ret.values.tolist()

        # Color based on value
        # Ensure v is comparable to int (0)
        colors = [
            get_color(theme, "up_color")
            if float(v) > 0  # type: ignore
            else get_color(theme, "down_color")
            for v in values
        ]
    else:
        years, values, colors = [], [], []

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=years,
            y=values,
            marker_color=colors,
            text=[f"{v:.2%}" for v in values],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="年度收益 (Yearly Returns)",
        xaxis_title="年份 (Year)",
        yaxis_title="收益率 (Return)",
        yaxis_tickformat=".2%",
        template="plotly_white" if theme == "light" else "plotly_dark",
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig
