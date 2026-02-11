"""Visualization module for AKQuant using Plotly."""

from typing import TYPE_CHECKING, Optional

import pandas as pd

try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False
    go = None
    make_subplots = None

if TYPE_CHECKING:
    from .backtest import BacktestResult


def plot_result(
    result: "BacktestResult",
    symbol: Optional[str] = None,
    show: bool = True,
    title: str = "Backtest Result",
    filename: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    Plot backtest result using Plotly.

    Args:
        result: The BacktestResult object.
        symbol: The symbol to plot. If None, uses the first symbol found.
        show: Whether to show the plot immediately.
        title: Chart title.
        filename: File path to save the plot. If ends with .html,
                 saves as interactive HTML.
                 If ends with .png/.jpg/.jpeg/.webp/.svg/.pdf,
                 saves as static image (requires kaleido).

    Returns:
        go.Figure: The plotly figure object.
    """
    if not _HAS_PLOTLY:
        print(
            "Plotly is not installed. Please install it using `pip install plotly` "
            "or `pip install akquant[plot]`."
        )
        return None

    # 1. Prepare Data
    # -------------------------------------------------------------------------
    # Equity Curve
    equity_curve = result.equity_curve  # List[(ts, equity)]
    if not equity_curve:
        print("No equity curve data available.")
        return go.Figure()

    equity_df = pd.DataFrame(equity_curve, columns=["time", "equity"])
    equity_df["time"] = pd.to_datetime(
        equity_df["time"], unit="ns", utc=True
    ).dt.tz_convert(result._timezone)
    equity_df.set_index("time", inplace=True)

    # Calculate Drawdown
    equity_df["max_equity"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["equity"] - equity_df["max_equity"]) / equity_df[
        "max_equity"
    ]

    # Determine Symbol
    trades_df = result.trades_df
    positions_df = result.positions_df

    if symbol is None:
        if not trades_df.empty:
            symbol = trades_df["symbol"].iloc[0]
        elif not positions_df.empty:
            symbol = positions_df.columns[0]
        else:
            # Try to get from strategy history if possible?
            # For now, if no symbol, we can't plot price chart easily without
            # accessing data feed. But we can plot equity curve.
            pass

    # Price Data (Need access to historical data)
    # Since BacktestResult doesn't store full price history, we rely on the user
    # to have the data or we extract it from the strategy if possible.
    # Ideally, the strategy should record price history or we pass data separately.
    # For this version, we will try to reconstruct price from trades or skip if
    # not available.
    # WAIT: The BacktestResult doesn't contain the OHLC data.
    # We should probably pass the 'data' or 'strategy' to this function, OR
    # the user calls this method on the result object, and maybe we can't easily
    # access the original data.

    # However, for a good plot, we NEED the price data.
    # Let's assume for now we only plot Equity/Drawdown/Positions if no price data
    # is provided.
    # BUT, to make it useful like Backtrader, we really want OHLC.

    # Let's check if we can get price data from somewhere.
    # If not, we just plot Equity and Drawdown.

    # 2. Create Subplots
    # -------------------------------------------------------------------------
    # Rows:
    # 1. Equity Curve
    # 2. Drawdown
    # 3. Daily Returns (Optional)
    #
    # If we have symbol data:
    # 1. Price (Candlestick) + Trades
    # 2. Volume
    # 3. Equity
    # 4. Drawdown

    # Let's start with a layout for Equity Analysis first, as that's always available.

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        subplot_titles=("Equity Curve", "Drawdown", "Daily Positions"),
    )

    # Plot Equity
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # Plot Drawdown
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["drawdown"],
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="red", width=1),
        ),
        row=2,
        col=1,
    )

    # Plot Positions (if available)
    if not positions_df.empty:
        # If symbol is specified, plot that symbol's position
        if symbol and symbol in positions_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=positions_df.index,
                    y=positions_df[symbol],
                    mode="lines",
                    name=f"Position ({symbol})",
                    line=dict(color="green", width=1, shape="hv"),
                ),
                row=3,
                col=1,
            )
        else:
            # Plot total exposure or top symbols?
            # Let's plot the sum of absolute positions as "Gross Exposure"
            # Or just plot the first column
            col = positions_df.columns[0]
            fig.add_trace(
                go.Scatter(
                    x=positions_df.index,
                    y=positions_df[col],
                    mode="lines",
                    name=f"Position ({col})",
                    line=dict(color="green", width=1, shape="hv"),
                ),
                row=3,
                col=1,
            )

    # 3. Add Trade Markers (on Equity Curve for now, or Price if we had it)
    # We can add annotations for trades on the Equity Curve or separate markers
    # For now, let's keep it simple.

    # Layout Updates
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        template="plotly_white",
        height=800,
        hovermode="x unified",
    )

    # Y-Axes Labels
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
    fig.update_yaxes(title_text="Position", row=3, col=1)

    if filename:
        try:
            if filename.endswith(".html"):
                fig.write_html(filename)
                print(f"Plot saved to {filename}")
            else:
                # Try saving as static image
                fig.write_image(filename)
                print(f"Plot saved to {filename}")
        except ImportError:
            print(
                "Error saving static image. Please install kaleido: "
                "`pip install -U kaleido`. "
                "Or use .html extension for interactive plot."
            )
        except Exception as e:
            print(f"Error saving plot to {filename}: {e}")

    if show:
        fig.show()

    return fig
