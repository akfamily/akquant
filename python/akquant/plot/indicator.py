"""Indicator plotting module."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from ..chart._normalize import CANDLE_ROLES
from ..indicator_identity import (
    DEFAULT_OWNER,
    candle_group_of,
    chart_series_keys,
    display_label,
    group_display_label,
    identity_of,
)
from .utils import check_plotly, get_color, go, make_subplots

if TYPE_CHECKING:
    from ..backtest import BacktestResult


#: candle 指标空心(下跌)K 线的填充色
CANDLE_HOLLOW = "rgba(0,0,0,0)"


def _color_or_none(value: Any) -> Optional[str]:
    """把定义里的颜色归一: ``None`` / ``NaN`` / 空白串视为未指定."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    return text or None


def _candle_fallback_color(theme: str, index: int) -> str:
    """未指定颜色的 candle 指标取指标色板(与 LWC 复盘同一组), 绝不用主题涨跌色."""
    from ..lwc._template import INDICATOR_PALETTE

    palette = INDICATOR_PALETTE.get(theme, INDICATOR_PALETTE["light"])
    return palette[index % len(palette)]


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
    definition: Mapping[str, Any],
    series_key: str,
    symbol: str,
    include_symbol: bool,
) -> str:
    """Build a readable trace name from normalized indicator metadata.

    基名走 :func:`display_label`(未写 label 时就是 series 键, 冲突时带
    ``owner:`` 前缀); 同名键冲突且用户写了 label 时, label 遮住了前缀, 追加
    ``<owner>`` 保证 trace 名两两不同。
    """
    label = display_label(definition, series_key)
    parts = [label]
    if include_symbol and symbol:
        parts.append(f"[{symbol}]")
    owner, key = identity_of(definition)
    if series_key != key and label != series_key:
        parts.append(f"<{owner}>")
    return " ".join(parts)


def _build_candle_trace(
    *,
    series_frame: pd.DataFrame,
    trace_name: str,
    color_value: str,
) -> Optional[Any]:
    """把一个 candle 组(同 owner/group/symbol)的四个分量拼成 ``go.Candlestick``.

    按 ``datetime`` 把 open/high/low/close 对齐, 缺任一分量的时刻跳过; 一个完整
    时刻都没有时返回 ``None``。配色不用主题涨跌色(否则与价格 K 线分不清):
    单色 ``color_value``, 上涨实心、下跌空心(TradingView hollow candles 惯例)。
    """
    wide = series_frame.pivot_table(
        index="datetime", columns="role", values="value", aggfunc="last"
    )
    if any(role not in wide.columns for role in CANDLE_ROLES):
        return None
    wide = wide.dropna(subset=list(CANDLE_ROLES)).sort_index()
    if wide.empty:
        return None
    return go.Candlestick(
        x=wide.index,
        open=wide["open"].astype(float),
        high=wide["high"].astype(float),
        low=wide["low"].astype(float),
        close=wide["close"].astype(float),
        name=trace_name,
        increasing=dict(line=dict(color=color_value), fillcolor=color_value),
        decreasing=dict(line=dict(color=color_value), fillcolor=CANDLE_HOLLOW),
    )


def _build_candle_trace_name(
    *,
    definition: Mapping[str, Any],
    series_key: str,
    symbol: str,
    include_symbol: bool,
) -> str:
    """生成 candle 组的 trace 名; 规则同 :func:`_build_trace_name`, 以组名代替键."""
    label = group_display_label(definition, series_key)
    parts = [label]
    if include_symbol and symbol:
        parts.append(f"[{symbol}]")
    owner = identity_of(definition)[0]
    if series_key != candle_group_of(definition) and label != series_key:
        parts.append(f"<{owner}>")
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
            columns=[
                "owner_strategy_id",
                "indicator_key",
                "display_name",
                "pane",
                "render_type",
                "color",
            ]
        )
    # 指标身份是 (owner, key): 主策略与指标脚本的同名键各自关联各自的定义。
    # 缺失 owner 与 indicator_identity 同口径按 "_default" 处理。
    for table in (frame, definitions):
        if "owner_strategy_id" not in table:
            table["owner_strategy_id"] = DEFAULT_OWNER
        owners = table["owner_strategy_id"].fillna("").astype(str).str.strip()
        table["owner_strategy_id"] = owners.replace("", DEFAULT_OWNER)

    merged = frame.merge(
        definitions,
        on=["owner_strategy_id", "indicator_key"],
        how="left",
        suffixes=("", "_definition"),
    )
    merged["display_name"] = merged["display_name"].fillna(merged["indicator_key"])
    merged["pane"] = (
        pd.to_numeric(merged["pane"], errors="coerce").fillna(0).astype(int)
    )
    merged["render_type"] = merged["render_type"].fillna("line").replace("", "line")
    merged["symbol"] = merged["symbol"].fillna("").astype(str)
    for column in ("group", "role"):
        if column not in merged:
            merged[column] = ""
        merged[column] = merged[column].fillna("").astype(str).str.strip()
    merged["role"] = merged["role"].str.lower()
    # candle 分量(render_type="candle" 且带 group)按 (owner, group) 合成一条
    # Candlestick; 其余行的组名为 ""
    merged["_candle_group"] = [
        candle_group_of({"render_type": render_type, "group": group})
        for render_type, group in zip(merged["render_type"], merged["group"])
    ]
    # series 键按"定义 ∪ 实际点位"的身份计算, 无定义的点位也有键可取;
    # candle 组与普通线同一命名空间(见 chart_series_keys)
    identity_columns = ["owner_strategy_id", "indicator_key", "render_type", "group"]
    keys = chart_series_keys(
        {str(column): value for column, value in row.items()}
        for table in (definitions, merged)
        for row in table.reindex(columns=identity_columns).to_dict("records")
    )
    merged["series_key"] = [
        keys.groups[(identity_of({"owner_strategy_id": owner})[0], group)]
        if group
        else keys.lines[identity_of({"owner_strategy_id": owner, "indicator_key": key})]
        for owner, key, group in zip(
            merged["owner_strategy_id"],
            merged["indicator_key"],
            merged["_candle_group"],
        )
    ]
    merged = merged.dropna(subset=["datetime", "value"]).sort_values(
        ["pane", "series_key", "symbol", "timestamp"]
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

    include_symbol = merged["symbol"].replace("", pd.NA).nunique(dropna=True) > 1
    use_webgl = len(merged) > 10000
    has_candle = False
    candle_count = 0

    for row_index, pane_name in enumerate(panes, start=1):
        pane_frame = merged.loc[merged["pane"] == pane_name]
        candle_rows = pane_frame["_candle_group"] != ""
        for _, series_frame in pane_frame.loc[candle_rows].groupby(
            ["owner_strategy_id", "_candle_group", "symbol"],
            dropna=False,
            sort=False,
        ):
            first = series_frame.iloc[0]
            candle_trace = _build_candle_trace(
                series_frame=series_frame,
                trace_name=_build_candle_trace_name(
                    definition=first.to_dict(),
                    series_key=str(first["series_key"]),
                    symbol=str(first["symbol"]),
                    include_symbol=include_symbol,
                ),
                color_value=_color_or_none(first.get("color"))
                or _candle_fallback_color(theme, candle_count),
            )
            if candle_trace is None:
                continue
            has_candle = True
            candle_count += 1
            fig.add_trace(candle_trace, row=row_index, col=1)
            fig.update_yaxes(title_text=_pane_title(pane_name), row=row_index, col=1)
        pane_frame = pane_frame.loc[~candle_rows]
        for _, series_frame in pane_frame.groupby(
            ["owner_strategy_id", "indicator_key", "symbol"],
            dropna=False,
            sort=False,
        ):
            first = series_frame.iloc[0]
            symbol_text = str(first["symbol"])
            render_type = str(first["render_type"]).strip().lower()
            color_value = _color_or_none(first.get("color"))
            trace_name = _build_trace_name(
                definition=first.to_dict(),
                series_key=str(first["series_key"]),
                symbol=symbol_text,
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
    if has_candle:
        # Candlestick 默认给 x 轴挂 rangeslider, 多 pane 时会挤压版面
        fig.update_xaxes(rangeslider_visible=False)
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
