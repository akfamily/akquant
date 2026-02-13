"""Strategy detailed plotting module."""

from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

from .utils import check_plotly, get_color, go, make_subplots

if TYPE_CHECKING:
    from ..backtest import BacktestResult


def plot_strategy(
    result: "BacktestResult",
    symbol: str,
    data: pd.DataFrame,
    indicators: Optional[Dict[str, pd.Series]] = None,
    title: Optional[str] = None,
    theme: str = "light",
    show: bool = True,
    filename: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    Plot detailed strategy execution for a specific symbol.

    Args:
        result: BacktestResult object.
        symbol: Symbol to plot.
        data: OHLCV DataFrame for the symbol (must have index as datetime).
        indicators: Dictionary of indicator series to overlay or plot in subplots.
        title: Chart title.
        theme: "light" or "dark".
        show: Whether to show the plot.
        filename: File path to save the plot.
    """
    if not check_plotly():
        return None

    if data.empty:
        print(f"No data provided for symbol {symbol}.")
        return None

    # Filter trades for this symbol
    trades_df = result.trades_df
    symbol_trades = pd.DataFrame()
    if not trades_df.empty:
        symbol_trades = trades_df[trades_df["symbol"] == symbol]

    # Layout:
    # Row 1: Candlestick + Overlays (Main) + Trade Markers
    # Row 2: Volume
    # Row 3+: Additional Indicators (if any, optional)

    rows = 2
    row_heights = [0.7, 0.3]

    # Check for separate subplots required by indicators?
    # For simplicity, we overlay all indicators on main chart unless
    # specified otherwise?
    # Let's assume indicators are overlays for now.

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # 2. Indicators (Overlay)
    if indicators:
        colors = ["orange", "purple", "blue", "brown", "pink"]
        for i, (name, series) in enumerate(indicators.items()):
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1),
                ),
                row=1,
                col=1,
            )

    # 3. Trade Markers
    if not symbol_trades.empty:
        # Buy Markers (Entries)
        buys = symbol_trades[symbol_trades["side"] == "LONG"]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["entry_time"],
                    y=buys["entry_price"],
                    mode="markers",
                    name="Buy (Long)",
                    marker=dict(symbol="triangle-up", size=10, color="green"),
                    text=buys.apply(
                        lambda row: f"Size: {row['quantity']}<br>PnL: {row['pnl']:.2f}",
                        axis=1,
                    ),
                    hovertemplate="<b>Buy</b><br>Price: %{y}<br>%{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Sell Markers (Exits for Long, Entries for Short)
        # For simplicity, we just mark entry and exit of closed trades
        # Exits
        fig.add_trace(
            go.Scatter(
                x=symbol_trades["exit_time"],
                y=symbol_trades["exit_price"],
                mode="markers",
                name="Exit",
                marker=dict(
                    symbol="triangle-down",
                    size=10,
                    color="red" if not buys.empty else "orange",
                ),  # Red for long exit
                text=symbol_trades.apply(
                    lambda row: (
                        f"PnL: {row['pnl']:.2f}<br>Ret: {row['return_pct']:.2%}"
                    ),
                    axis=1,
                ),
                hovertemplate="<b>Exit</b><br>Price: %{y}<br>%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 4. Volume
    if "volume" in data.columns:
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["volume"],
                name="Volume",
                marker_color="rgba(100, 100, 100, 0.5)",
            ),
            row=2,
            col=1,
        )

    # Update Layout
    bg_color = get_color(theme, "bg_color")
    text_color = get_color(theme, "text_color")
    grid_color = get_color(theme, "grid_color")

    title = title or f"Strategy Analysis: {symbol}"

    fig.update_layout(
        title=title,
        height=800,
        template="plotly_white" if theme == "light" else "plotly_dark",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis_rangeslider_visible=False,
    )

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
