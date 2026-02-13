"""Dashboard plotting module."""

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from .utils import check_plotly, get_color, go, make_subplots

if TYPE_CHECKING:
    from ..backtest import BacktestResult


def plot_dashboard(
    result: "BacktestResult",
    title: str = "策略仪表盘 (Strategy Dashboard)",
    theme: str = "light",
    show: bool = True,
    filename: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    Plot comprehensive dashboard for backtest result.

    Includes:
    1. Equity Curve
    2. Drawdown
    3. Monthly Heatmap (if enough data)
    """
    if not check_plotly():
        return None

    equity_series = result.equity_curve
    if equity_series.empty:
        print("No equity curve data available.")
        return None

    equity_df = equity_series.to_frame(name="equity")

    # Calculate Drawdown
    equity_df["max_equity"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["equity"] - equity_df["max_equity"]) / equity_df[
        "max_equity"
    ]

    # Calculate Daily Returns for Heatmap
    # Handle single data point or empty resample
    if len(equity_df) < 2:
        daily_returns = pd.Series(dtype=float)
    else:
        # Resample to daily, forward fill equity (mark-to-market),
        # then calculate returns
        daily_returns = equity_df["equity"].resample("D").last().ffill().pct_change()

    # Create Layout
    # Row 1: Equity
    # Row 2: Drawdown
    # Row 3: Monthly Heatmap (if > 2 months)

    rows = 3
    row_heights = [0.5, 0.25, 0.25]

    # Check if intraday to adjust vertical spacing
    # Intraday labels are rotated and take more vertical space
    is_intraday = False
    if len(equity_df) > 1:
        time_diff = equity_df.index[1] - equity_df.index[0]
        if time_diff.total_seconds() < 86400:  # Less than 1 day
            is_intraday = True

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.25 if is_intraday else 0.15,
        row_heights=row_heights,
        subplot_titles=(
            "权益曲线 (Equity)",
            "最大回撤 (Drawdown)",
            "月度收益 (Monthly Returns)",
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"type": "heatmap"}],
        ],
    )

    # Determine Trace Type (Scatter or Scattergl for large data)
    # Threshold for WebGL: 10,000 points
    TraceType = go.Scattergl if len(equity_df) > 10000 else go.Scatter

    # 1. Equity
    # Explicitly convert to list to ensure serialization
    x_data = equity_df.index
    y_data = equity_df["equity"].fillna(0.0).tolist()

    fig.add_trace(
        TraceType(
            x=x_data,
            y=y_data,
            mode="lines",
            name="权益",
            line=dict(color=get_color(theme, "up_color"), width=2),
        ),
        row=1,
        col=1,
    )
    # Format Y-axis for Equity (Currency) and X-axis (Date)
    fig.update_yaxes(tickformat=".2f", row=1, col=1)

    # Determine X-axis format based on data frequency
    # If intraday (time diff < 1 day), use minute precision
    # is_intraday is already calculated above

    xaxis_format = "%Y-%m-%d %H:%M" if is_intraday else "%Y-%m-%d"

    # Update X-axes with smart formatting to avoid clutter
    fig.update_xaxes(
        tickformat=xaxis_format,
        tickangle=-45 if is_intraday else 0,  # Slant labels for long timestamps
        nticks=20 if is_intraday else None,
        row=1,
        col=1,
    )

    # 2. Drawdown
    drawdown_data = equity_df["drawdown"].fillna(0.0).tolist()
    fig.add_trace(
        TraceType(
            x=x_data,
            y=drawdown_data,
            mode="lines",
            name="回撤",
            fill="tozeroy",
            line=dict(color=get_color(theme, "down_color"), width=1),
        ),
        row=2,
        col=1,
    )
    # Format Y-axis for Drawdown (Percentage) and X-axis (Date)
    fig.update_yaxes(tickformat=".2%", row=2, col=1)

    # Apply same X-axis formatting to Drawdown plot
    fig.update_xaxes(
        tickformat=xaxis_format,
        tickangle=-45 if is_intraday else 0,
        nticks=20 if is_intraday else None,
        row=2,
        col=1,
    )

    # Link X-axes of Equity and Drawdown
    fig.update_xaxes(matches="x", row=2, col=1)
    # fig.update_xaxes(showticklabels=False, row=1, col=1)
    # Keep labels for clarity due to spacing

    # 3. Monthly Heatmap
    # Resample to monthly
    monthly_returns = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    if len(monthly_returns) > 0:
        if hasattr(monthly_returns.index, "year"):
            years = monthly_returns.index.year.unique()
        else:
            years = monthly_returns.index.unique()
        months = [
            "1月",
            "2月",
            "3月",
            "4月",
            "5月",
            "6月",
            "7月",
            "8月",
            "9月",
            "10月",
            "11月",
            "12月",
        ]

        z = np.zeros((len(years), 12))
        z[:] = np.nan

        text = np.empty((len(years), 12), dtype=object)
        text[:] = ""

        for i, year in enumerate(years):
            if hasattr(monthly_returns.index, "year"):
                year_data = monthly_returns[monthly_returns.index.year == year]
            else:
                year_data = monthly_returns
            for date, ret in year_data.items():
                # date might be Timestamp or just index label.
                # If we used resample("ME"), it should be Timestamp
                if hasattr(date, "month"):
                    month_idx = date.month - 1
                    z[i, month_idx] = ret * 100
                    text[i, month_idx] = f"{ret * 100:.2f}%"

        # Convert to list for serialization
        z_list = z.tolist()
        text_list = text.tolist()

        fig.add_trace(
            go.Heatmap(
                z=z_list,
                x=months,
                y=years,
                text=text_list,
                texttemplate="%{text}",
                colorscale=[
                    [0, "#2e7d32"],
                    [0.5, "#ffffff"],
                    [1, "#d32f2f"],
                ],  # Green to Red via White
                zmid=0,
                showscale=True,
                colorbar=dict(title="收益率 %", len=0.25, y=0.1),
            ),
            row=3,
            col=1,
        )
        # Heatmap usually doesn't share x-axis with time series
        fig.update_xaxes(showticklabels=True, row=3, col=1)

    # Update Layout
    bg_color = get_color(theme, "bg_color")
    text_color = get_color(theme, "text_color")
    grid_color = get_color(theme, "grid_color")

    fig.update_layout(
        title=dict(text=title, y=0.98),  # Adjust main title position
        height=800,
        template="plotly_white" if theme == "light" else "plotly_dark",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        margin=dict(t=80, b=50, l=50, r=50),  # Increase top margin for titles
    )

    # Manually adjust subplot titles by adding annotations padding if needed,
    # but Plotly handles this via layout.margin mostly.
    # To increase space between subplot title and subplot content,
    # we can't easily do it via make_subplots directly
    # without complex annotations. However, increasing vertical_spacing helps (done).
    # We can also add some <br> to titles or use annotations.
    # A cleaner way is to ensure the layout margin is sufficient.

    # Let's try to update annotations (subplot titles) to add padding
    # 'pad' is not a valid property for annotations. Use 'yshift' to move them up.
    fig.update_annotations(yshift=20)  # Move subplot titles up by 20px

    fig.update_xaxes(gridcolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color)

    if filename:
        if filename.endswith(".html"):
            fig.write_html(filename)
        else:
            fig.write_image(filename)

    if show:
        fig.show()

    return fig
