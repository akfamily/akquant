"""Indicator plotting module."""

from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from .utils import check_plotly, get_color, go, make_subplots

if TYPE_CHECKING:
    from ..backtest import BacktestResult


def _pane_title(pane: int) -> str:
    """Render a human-readable pane title from its integer index."""
    return "Main" if int(pane) == 0 else f"Sub {int(pane)}"


def _build_indicator_trace(
    *,
    render_type: str,
    x_data: "pd.Series[Any]",
    y_data: "pd.Series[Any]",
    trace_name: str,
    color_value: Any,
    use_webgl: bool,
) -> Any:
    """Build a Plotly trace for one indicator series by canonical render type.

    Every value in ``RENDER_TYPE_CANONICAL`` has an explicit rendering here so no
    render type silently degrades to a line.
    """
    if render_type in {"bar", "column", "histogram"}:
        return go.Bar(
            x=x_data,
            y=y_data,
            name=trace_name,
            marker=dict(color=color_value) if color_value else None,
        )

    scatter_type = go.Scattergl if use_webgl else go.Scatter
    if render_type == "scatter":
        return scatter_type(
            x=x_data,
            y=y_data,
            mode="markers",
            name=trace_name,
            marker=dict(color=color_value) if color_value else None,
        )
    if render_type == "signal":
        return scatter_type(
            x=x_data,
            y=y_data,
            mode="markers",
            name=trace_name,
            marker=dict(
                symbol="triangle-up",
                size=10,
                color=color_value if color_value else None,
            ),
        )
    if render_type == "area":
        return scatter_type(
            x=x_data,
            y=y_data,
            mode="lines",
            name=trace_name,
            fill="tozeroy",
            line=dict(color=color_value, width=2) if color_value else dict(width=2),
        )
    # render_type == "line" (canonical default)
    return scatter_type(
        x=x_data,
        y=y_data,
        mode="lines",
        name=trace_name,
        line=dict(color=color_value, width=2) if color_value else dict(width=2),
    )


def _build_trace_name(
    *,
    display_name: str,
    owner_strategy_id: str,
    symbol: str,
    include_owner: bool,
    include_symbol: bool,
) -> str:
    """Build a readable trace name from normalized indicator metadata."""
    parts = [display_name]
    if include_symbol and symbol:
        parts.append(f"[{symbol}]")
    if include_owner and owner_strategy_id:
        parts.append(f"<{owner_strategy_id}>")
    return " ".join(parts)


def plot_indicators(
    result: "BacktestResult",
    name: Optional[str] = None,
    symbol: Optional[str] = None,
    include_warmup: bool = True,
    title: str = "Indicator History",
    theme: str = "light",
    show: bool = True,
    filename: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    Plot recorded indicator history as one lightweight multi-pane figure.

    :param result: Backtest result that contains indicator outputs.
    :param name: Optional indicator key filter.
    :param symbol: Optional symbol filter.
    :param include_warmup: Whether to keep warmup points.
    :param title: Figure title.
    :param theme: Plot theme key.
    :param show: Whether to display the figure immediately.
    :param filename: Optional HTML output path.
    :return: Plotly Figure or ``None`` when no indicator data is available.
    """
    if not check_plotly():
        return None

    frame = result.indicator_df(name=name, symbol=symbol).copy()
    if frame.empty:
        print("No indicator data available.")
        return None
    if not include_warmup:
        frame = frame.loc[~frame["warmup"]]
    if frame.empty:
        print("No indicator data available after filtering.")
        return None

    definitions = result.indicator_definitions.copy()
    if definitions.empty:
        definitions = pd.DataFrame(
            columns=["indicator_key", "display_name", "pane", "render_type", "color"]
        )

    merged = frame.merge(
        definitions,
        on="indicator_key",
        how="left",
        suffixes=("", "_definition"),
    )
    merged["display_name"] = merged["display_name"].fillna(merged["indicator_key"])
    merged["pane"] = (
        pd.to_numeric(merged["pane"], errors="coerce").fillna(0).astype(int)
    )
    merged["render_type"] = merged["render_type"].fillna("line").replace("", "line")
    merged["owner_strategy_id"] = merged["owner_strategy_id"].fillna("").astype(str)
    merged["symbol"] = merged["symbol"].fillna("").astype(str)
    merged = merged.dropna(subset=["datetime", "value"]).sort_values(
        ["pane", "indicator_key", "symbol", "timestamp"]
    )
    if merged.empty:
        print("No indicator data available after filtering invalid rows.")
        return None

    panes = sorted(merged["pane"].drop_duplicates().tolist())
    subplot_titles = [_pane_title(pane) for pane in panes]
    row_count = max(len(panes), 1)

    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=tuple(subplot_titles),
    )

    include_owner = merged["owner_strategy_id"].nunique(dropna=True) > 1
    include_symbol = merged["symbol"].replace("", pd.NA).nunique(dropna=True) > 1
    use_webgl = len(merged) > 10000

    for row_index, pane_name in enumerate(panes, start=1):
        pane_frame = merged.loc[merged["pane"] == pane_name]
        for _, series_frame in pane_frame.groupby(
            [
                "indicator_key",
                "display_name",
                "owner_strategy_id",
                "symbol",
                "render_type",
            ],
            dropna=False,
            sort=False,
        ):
            display_name = str(series_frame["display_name"].iloc[0])
            owner_strategy_id = str(series_frame["owner_strategy_id"].iloc[0])
            symbol_text = str(series_frame["symbol"].iloc[0])
            render_type = str(series_frame["render_type"].iloc[0]).strip().lower()
            color_value = (
                series_frame["color"].iloc[0] if "color" in series_frame else None
            )
            trace_name = _build_trace_name(
                display_name=display_name,
                owner_strategy_id=owner_strategy_id,
                symbol=symbol_text,
                include_owner=include_owner,
                include_symbol=include_symbol,
            )
            x_data = series_frame["datetime"]
            y_data = series_frame["value"].astype(float)

            fig.add_trace(
                _build_indicator_trace(
                    render_type=render_type,
                    x_data=x_data,
                    y_data=y_data,
                    trace_name=trace_name,
                    color_value=color_value,
                    use_webgl=use_webgl,
                ),
                row=row_index,
                col=1,
            )

            fig.update_yaxes(title_text=_pane_title(pane_name), row=row_index, col=1)

        # 参考线:对该 pane 下每个带 reference_lines 的指标画静态横线。
        definitions_df = result.indicator_definitions
        pane_defs = definitions_df[definitions_df["pane"] == pane_name]
        for _, def_row in pane_defs.iterrows():
            ref_lines = def_row.get("reference_lines") or []
            for line in ref_lines:
                fig.add_hline(
                    y=line["value"],
                    line_dash="dash",
                    line_color=line.get("color") or None,
                    annotation_text=line.get("label") or None,
                    row=row_index,
                    col=1,
                )
        # 刻度分组:在 y 轴标题追加语义组名提示。
        groups = [
            g for g in pane_defs.get("scale_group", pd.Series(dtype=str)).tolist() if g
        ]
        if groups:
            base_title = _pane_title(pane_name)
            fig.update_yaxes(
                title_text=f"{base_title} · {groups[0]}", row=row_index, col=1
            )

    xaxis_format = "%Y-%m-%d"
    if len(merged) > 1:
        datetimes = merged["datetime"].sort_values()
        if (
            len(datetimes) > 1
            and (datetimes.iloc[1] - datetimes.iloc[0]).total_seconds() < 86400
        ):
            xaxis_format = "%Y-%m-%d %H:%M"

    fig.update_xaxes(tickformat=xaxis_format)
    fig.update_layout(
        title=title,
        template="plotly_dark" if theme == "dark" else "plotly_white",
        height=max(360, 280 * row_count),
        hovermode="x unified",
        paper_bgcolor=get_color(theme, "bg_color"),
        plot_bgcolor=get_color(theme, "bg_color"),
        font=dict(color=get_color(theme, "text_color")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=70, b=50),
    )

    if filename is not None:
        fig.write_html(filename)
    if show:
        fig.show()
    return fig
