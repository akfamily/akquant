"""Consolidated report generation module."""

import base64
import datetime
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import pandas as pd

from ..analysis.benchmark import (
    build_benchmark_analysis as _build_benchmark_analysis,
)
from ..analysis.benchmark import (
    build_daily_returns_from_equity as _build_daily_returns_from_equity,
)
from ..analysis.benchmark import (
    normalize_curve_freq as _normalize_curve_freq,
)
from ..analysis.benchmark import (
    resolve_equity_curve as _resolve_equity_curve,
)
from ..utils import format_metric_value
from .analysis import (
    plot_pnl_vs_duration,
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_trades_distribution,
    plot_yearly_returns,
)
from .dashboard import plot_dashboard
from .indicator import plot_indicators
from .strategy import plot_strategy
from .utils import check_plotly

if TYPE_CHECKING:
    from ..backtest import BacktestResult

# Embedded SVG Icon
AKQUANT_ICON_SVG = """<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate(17, 25)">
    <rect x="0" y="40" width="15" height="30" rx="4" fill="#CE412B" />
    <rect x="25" y="20" width="15" height="50" rx="4" fill="#BE5C60" />
    <rect x="50" y="0" width="15" height="70" rx="4" fill="#3776AB" />
    <circle cx="7" cy="30" r="2" fill="#CE412B" />
    <circle cx="32" cy="10" r="2" fill="#BE5C60" />
    <circle cx="57" cy="-10" r="2" fill="#3776AB" />
  </g>
</svg>"""

# Embedded SVG Logo
AKQUANT_LOGO_SVG = """<svg width="400" height="120" viewBox="0 0 400 120" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="textGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#CE412B;stop-opacity:1" />
      <!-- Rust Orange -->
      <stop offset="100%" style="stop-color:#3776AB;stop-opacity:1" />
      <!-- Python Blue -->
    </linearGradient>
  </defs>

  <!-- Graphic: Quant Signal Bars (Ascending Data Spectrum) -->
  <g transform="translate(30, 25)">
    <!-- Bar 1: Low (Foundation/Data - Rust Color) -->
    <rect x="0" y="40" width="15" height="30" rx="4" fill="#CE412B" />

    <!-- Bar 2: Mid (Processing/Strategy - Blend) -->
    <rect x="25" y="20" width="15" height="50" rx="4" fill="#BE5C60" />

    <!-- Bar 3: High (Alpha/Profit - Python Color) -->
    <rect x="50" y="0" width="15" height="70" rx="4" fill="#3776AB" />

    <!-- Abstract Data Dots -->
    <circle cx="7" cy="30" r="2" fill="#CE412B" />
    <circle cx="32" cy="10" r="2" fill="#BE5C60" />
    <circle cx="57" cy="-10" r="2" fill="#3776AB" />
  </g>

  <!-- Text -->
  <text x="110" y="75"
        font-family="'Segoe UI', Helvetica, Arial, sans-serif"
        font-size="60"
        font-weight="bold"
        fill="url(#textGradient)">AKQuant</text>

  <!-- Subtitle -->
  <text x="115" y="100"
        font-family="'Segoe UI', Helvetica, Arial, sans-serif"
        font-size="14"
        fill="#666">High-Performance Quant Framework</text>
</svg>"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <link rel="icon" href="{favicon_uri}">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --accent-color: #3498db;
            --bg-color: #f5f7fa;
            --card-bg: #ffffff;
            --text-color: #333333;
            --text-secondary: #7f8c8d;
            --border-color: #e1e4e8;
            --success-color: #27ae60;
            --danger-color: #c0392b;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB",
                "Microsoft YaHei", sans-serif;
            margin: 0;
            padding: 20px;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: var(--card-bg);
            padding: 40px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-radius: 12px;
        }}

        header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
        }}

        .header-logo {{
            margin-bottom: 10px;
        }}

        .header-logo svg {{
            height: 80px;
            width: auto;
        }}

        header h1 {{
            margin: 0;
            color: var(--primary-color);
            font-size: 28px;
            font-weight: 700;
        }}

        header p {{
            color: var(--text-secondary);
            margin: 10px 0 0;
            font-size: 14px;
        }}

        .section-title {{
            font-size: 20px;
            font-weight: 600;
            color: var(--primary-color);
            margin: 40px 0 20px;
            padding-left: 12px;
            border-left: 4px solid var(--accent-color);
            display: flex;
            align-items: center;
        }}

        /* Summary Box */
        .summary-box {{
            display: flex;
            justify-content: space-between;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
            flex-wrap: wrap;
            gap: 20px;
        }}

        .summary-item {{
            flex: 1;
            min-width: 200px;
        }}

        .summary-label {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }}

        .summary-value {{
            font-size: 18px;
            font-weight: 600;
            color: var(--primary-color);
        }}

        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: #ffffff;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid var(--border-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }}

        .metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 5px;
        }}

        .metric-value.positive {{
            color: var(--success-color);
        }}

        .metric-value.negative {{
            color: var(--danger-color);
        }}

        .metric-label {{
            font-size: 14px;
            color: var(--text-secondary);
        }}

        /* Charts Grid Layout */
        .row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
        }}

        .col {{
            background: white;
            border-radius: 8px;
            min-width: 0; /* Prevent grid blowout */
        }}

        /* Ensure chart containers fill the column and handle overflow */
        .chart-container {{
            width: 100%;
            height: 100%;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 5px; /* Reduced padding to give more space to chart */
            box-sizing: border-box;
            background: white;
            overflow-x: auto;
            overflow-y: hidden;
        }}

        .table-scroll {{
            width: 100%;
            overflow-x: auto;
            overflow-y: hidden;
        }}

        .table {{
            width: max-content;
            min-width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 14px;
        }}

        .table th, .table td {{
            white-space: nowrap;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
            text-align: left;
        }}

        .table th {{
            position: sticky;
            top: 0;
            background: #f8f9fa;
            color: var(--primary-color);
            font-weight: 600;
            z-index: 1;
        }}

        .analysis-overview-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 14px;
            margin-bottom: 20px;
        }}

        .analysis-card {{
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px 14px;
        }}

        .analysis-card-label {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 6px;
            line-height: 1.35;
        }}

        .analysis-card-value {{
            font-size: 18px;
            font-weight: 700;
            color: var(--primary-color);
            line-height: 1.2;
        }}

        .details-block {{
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: #ffffff;
            margin-top: 14px;
        }}

        .details-block > summary {{
            list-style: none;
            cursor: pointer;
            padding: 12px 14px;
            font-size: 15px;
            font-weight: 600;
            color: var(--primary-color);
            border-bottom: 1px solid transparent;
            user-select: none;
        }}

        .details-block[open] > summary {{
            border-bottom-color: var(--border-color);
            background: #f8f9fa;
        }}

        .details-content {{
            padding: 10px 12px 12px 12px;
        }}

        .empty-panel {{
            border: 1px dashed var(--border-color);
            border-radius: 8px;
            background: #fafbfc;
            padding: 14px 16px;
            color: var(--text-secondary);
            line-height: 1.6;
        }}

        footer {{
            text-align: center;
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 12px;
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 20px; }}
            .row {{ grid-template-columns: 1fr; }} /* Stack on mobile */
            .summary-box {{ flex-direction: column; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-logo">{icon_svg}</div>
            <h1>{title}</h1>
            <p>生成时间: {date}</p>
        </header>

        <!-- Summary Section -->
        <div class="summary-box">
            <div class="summary-item">
                <div class="summary-label">回测区间</div>
                <div class="summary-value">{start_date} ~ {end_date}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">回测时长</div>
                <div class="summary-value">{duration_str}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">初始资金</div>
                <div class="summary-value">{initial_cash}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">最终权益</div>
                <div class="summary-value">{final_equity}</div>
            </div>
        </div>

        <div class="section-title">核心指标 (Key Metrics)</div>
        <div class="metrics-grid">
            {metrics_html}
        </div>

        <div class="section-title">权益与回撤 (Equity & Drawdown)</div>
        <div class="chart-container">
            {dashboard_html}
        </div>

        {indicator_section_html}

        <div class="section-title">收益分析 (Return Analysis)</div>
        <div class="row">
            <div class="col">
                <div class="chart-container">
                    {yearly_returns_html}
                </div>
            </div>
            <div class="col">
                <div class="chart-container">
                    {returns_dist_html}
                </div>
            </div>
        </div>
        <div class="chart-container" style="margin-top: 20px;">
            {rolling_metrics_html}
        </div>

        <div class="section-title">基准对比 (Benchmark Comparison)</div>
        <div class="metrics-grid">
            {benchmark_metrics_html}
        </div>
        <div class="chart-container">
            {benchmark_chart_html}
        </div>

        <div class="section-title">交易分析 (Trade Analysis)</div>
        <div class="row">
            <div class="col">
                <div class="chart-container">
                    {trades_dist_html}
                </div>
            </div>
            <div class="col">
                <div class="chart-container">
                    {pnl_duration_html}
                </div>
            </div>
        </div>
        <div class="section-title">交易复盘 (K线买卖点)</div>
        <div class="chart-container">
            {strategy_kline_html}
        </div>

        <div class="section-title">组合归因与容量分析 (Attribution & Capacity)</div>
        <div class="analysis-overview-grid">
            {analysis_overview_html}
        </div>
        <details class="details-block">
            <summary>暴露摘要明细 (Exposure Summary)</summary>
            <div class="details-content">{exposure_summary_html}</div>
        </details>
        <details class="details-block">
            <summary>容量摘要明细 (Capacity Summary)</summary>
            <div class="details-content">{capacity_summary_html}</div>
        </details>
        <details class="details-block" open>
            <summary>归因明细 (Attribution Details)</summary>
            <div class="details-content">{attribution_summary_html}</div>
        </details>
        <div class="section-title">策略归属聚合 (Strategy Ownership Aggregation)</div>
        <div class="row">
            <div class="col">
                <div class="chart-container">
                    {orders_by_strategy_html}
                </div>
            </div>
            <div class="col">
                <div class="chart-container">
                    {executions_by_strategy_html}
                </div>
            </div>
        </div>
        <details class="details-block" style="margin-top: 20px;">
            <summary>策略风控拒单明细 (Risk Rejections by Strategy)</summary>
            <div class="details-content">{risk_by_strategy_html}</div>
        </details>
        <details class="details-block">
            <summary>强平审计明细 (Forced Liquidation Audit)</summary>
            <div class="details-content">{liquidation_audit_html}</div>
        </details>
        {risk_charts_html}

        <footer>
            AKQuant Report | Powered by Plotly & AKQuant
        </footer>
    </div>
    <script>
        // Force Plotly resize on page load to ensure correct dimensions
        // in Grid/Flex layout
        window.addEventListener('load', function() {{
            setTimeout(function() {{
                var plots = document.getElementsByClassName('js-plotly-plot');
                for (var i = 0; i < plots.length; i++) {{
                    Plotly.Plots.resize(plots[i]);
                }}
            }}, 100); // Small delay to allow CSS layout to stabilize
        }});

        // Also trigger on resize to be safe
        // (though responsive: true handles most cases)
        window.addEventListener('resize', function() {{
            var plots = document.getElementsByClassName('js-plotly-plot');
            for (var i = 0; i < plots.length; i++) {{
                Plotly.Plots.resize(plots[i]);
            }}
        }});
    </script>
</body>
</html>
"""


def _format_currency(value: float) -> str:
    """Format large numbers nicely."""
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{sign}{abs_value / 1_000_000_000:.2f}B"
    elif abs_value >= 1_000_000:
        return f"{sign}{abs_value / 1_000_000:.2f}M"
    elif abs_value >= 1_000:
        return f"{sign}{abs_value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"


def _format_table(
    df: pd.DataFrame,
    max_rows: int = 10,
    percentage_columns: set[str] | None = None,
    compact_currency_columns: set[str] | None = None,
    compact_currency: bool = True,
) -> str:
    """Render a compact HTML table from a dataframe."""
    if df.empty:
        return "<div>暂无数据</div>"
    table = df.head(max_rows).copy()
    pct_cols = percentage_columns or set()
    money_cols = compact_currency_columns or set()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            if col in pct_cols:
                table[col] = table[col].map(lambda x: f"{x * 100:,.6f}%")
            elif compact_currency and col in money_cols:
                table[col] = table[col].map(_format_currency)
            else:
                table[col] = table[col].map(lambda x: f"{x:,.6f}")
    table_html = table.to_html(index=False, border=0, classes="table")
    return f'<div class="table-scroll">{table_html}</div>'


def _rename_table_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename technical columns to user-friendly labels."""
    renamed = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    return cast(pd.DataFrame, renamed)


def _build_benchmark_sections(
    strategy_returns: pd.Series,
    benchmark: Optional[Union[str, pd.Series]],
    config: dict[str, Any],
) -> dict[str, str]:
    """Build benchmark comparison metrics and chart sections."""
    payload = _build_benchmark_analysis(
        strategy_returns=strategy_returns, benchmark=benchmark
    )
    if not bool(payload["available"]):
        empty_html = f'<div class="empty-panel">{payload["reason"]}</div>'
        return {
            "benchmark_metrics_html": empty_html,
            "benchmark_chart_html": empty_html,
        }
    benchmark_label = str(payload["benchmark"]["label"])
    summary = cast(dict[str, Optional[float]], payload["summary"])
    series_frame = pd.DataFrame(payload["series"])

    def metric_color(value: Optional[float]) -> str:
        if value is None or pd.isna(value):
            return ""
        if value > 0:
            return "positive"
        if value < 0:
            return "negative"
        return ""

    def metric_fmt(value: Optional[float], kind: str) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        if kind == "pct":
            return f"{value * 100:.2f}%"
        return f"{value:.4f}"

    benchmark_metrics = [
        ("基准名称 (Benchmark)", benchmark_label, ""),
        (
            "累计超额收益 (Total Excess)",
            metric_fmt(summary["total_excess"], "pct"),
            metric_color(summary["total_excess"]),
        ),
        (
            "年化超额收益 (Annual Excess)",
            metric_fmt(summary["annual_excess"], "pct"),
            metric_color(summary["annual_excess"]),
        ),
        (
            "跟踪误差 (Tracking Error)",
            metric_fmt(summary["tracking_error"], "pct"),
            "",
        ),
        (
            "信息比率 (Information Ratio)",
            metric_fmt(summary["information_ratio"], "ratio"),
            metric_color(summary["information_ratio"]),
        ),
        (
            "Beta",
            metric_fmt(summary["beta"], "ratio"),
            metric_color(
                None if summary["beta"] is None else cast(float, summary["beta"]) - 1.0
            ),
        ),
        (
            "Alpha (Annual)",
            metric_fmt(summary["alpha"], "pct"),
            metric_color(summary["alpha"]),
        ),
    ]
    metrics_html = ""
    for label, value, color_cls in benchmark_metrics:
        metrics_html += (
            f'<div class="metric-card"><div class="metric-value {color_cls}">{value}'
            f'</div><div class="metric-label">{label}</div></div>'
        )
    go = __import__("plotly.graph_objects", fromlist=["Figure"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series_frame["date"],
            y=series_frame["strategy_cum_return"],
            mode="lines",
            name="策略累计收益",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series_frame["date"],
            y=series_frame["benchmark_cum_return"],
            mode="lines",
            name=f"基准累计收益 ({benchmark_label})",
            line=dict(color="#7f8c8d", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series_frame["date"],
            y=series_frame["excess_cum_return"],
            mode="lines",
            name="累计超额收益",
            line=dict(color="#c0392b", width=2),
        )
    )
    fig.update_layout(
        title="策略与基准累计收益对比 (Cumulative Return Comparison)",
        template="plotly_white",
        yaxis=dict(tickformat=".2%"),
        height=420,
        margin=dict(l=20, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False, config=config)
    return {
        "benchmark_metrics_html": metrics_html,
        "benchmark_chart_html": chart_html,
    }


def _build_summary_context(result: Any, curve_freq: str = "D") -> dict[str, str]:
    """Build summary text values for report header."""
    equity_curve = _resolve_equity_curve(result, curve_freq)
    start_date = "N/A"
    end_date = "N/A"
    duration_str = "N/A"
    final_equity_str = "N/A"
    initial_cash_str = (
        f"{result.initial_cash:,.2f}" if hasattr(result, "initial_cash") else "N/A"
    )

    if not equity_curve.empty:
        start_ts = equity_curve.index[0]
        end_ts = equity_curve.index[-1]
        start_date = start_ts.strftime("%Y-%m-%d")
        end_date = end_ts.strftime("%Y-%m-%d")
        duration_str = f"{(end_ts - start_ts).days} 天"
        final_equity_str = f"{equity_curve.iloc[-1]:,.2f}"

    return {
        "start_date": start_date,
        "end_date": end_date,
        "duration_str": duration_str,
        "initial_cash": initial_cash_str,
        "final_equity": final_equity_str,
    }


def _get_metric_value(
    result: Any, metrics: Any, name: str, default: float = 0.0
) -> float:
    """Read metric value from object or metrics_df."""
    if hasattr(metrics, name):
        val = getattr(metrics, name)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    try:
        return float(cast(Any, result.metrics_df.loc[name, "value"]))
    except Exception:
        return default


def _build_metrics_html(result: Any) -> str:
    """Build key-metrics HTML cards."""
    metrics = result.metrics

    def get_color_class(val: float) -> str:
        if val > 0:
            return "positive"
        if val < 0:
            return "negative"
        return ""

    metric_data = [
        (
            "累计收益率 (Total Return)",
            metrics.total_return_pct,
            format_metric_value("total_return_pct", metrics.total_return_pct),
            get_color_class(metrics.total_return_pct),
        ),
        (
            "年化收益率 (CAGR)",
            metrics.annualized_return,
            format_metric_value("annualized_return", metrics.annualized_return),
            get_color_class(metrics.annualized_return),
        ),
        (
            "平均盈亏 (Avg PnL)",
            _get_metric_value(result, metrics, "avg_pnl"),
            f"{_get_metric_value(result, metrics, 'avg_pnl'):.2f}",
            get_color_class(_get_metric_value(result, metrics, "avg_pnl")),
        ),
        (
            "夏普比率 (Sharpe)",
            metrics.sharpe_ratio,
            f"{metrics.sharpe_ratio:.2f}",
            get_color_class(metrics.sharpe_ratio),
        ),
        (
            "索提诺比率 (Sortino)",
            _get_metric_value(result, metrics, "sortino_ratio"),
            f"{_get_metric_value(result, metrics, 'sortino_ratio'):.2f}",
            get_color_class(_get_metric_value(result, metrics, "sortino_ratio")),
        ),
        (
            "卡玛比率 (Calmar)",
            _get_metric_value(result, metrics, "calmar_ratio"),
            f"{_get_metric_value(result, metrics, 'calmar_ratio'):.2f}",
            get_color_class(_get_metric_value(result, metrics, "calmar_ratio")),
        ),
        (
            "最大回撤 (Max DD)",
            metrics.max_drawdown_pct,
            format_metric_value("max_drawdown_pct", metrics.max_drawdown_pct),
            "negative",
        ),
        (
            "最大回撤金额 (Max DD Amount)",
            _get_metric_value(result, metrics, "max_drawdown_value"),
            _format_currency(_get_metric_value(result, metrics, "max_drawdown_value")),
            "negative",
        ),
        (
            "波动率 (Volatility)",
            metrics.volatility,
            format_metric_value("volatility", metrics.volatility),
            "",
        ),
        (
            "胜率 (Win Rate)",
            metrics.win_rate,
            format_metric_value("win_rate", metrics.win_rate),
            "",
        ),
        (
            "盈亏比 (Profit Factor)",
            _get_metric_value(result, metrics, "profit_factor"),
            f"{_get_metric_value(result, metrics, 'profit_factor'):.2f}",
            "",
        ),
        (
            "凯利公式 (Kelly)",
            _get_metric_value(result, metrics, "kelly_criterion"),
            format_metric_value(
                "kelly_criterion",
                _get_metric_value(result, metrics, "kelly_criterion"),
            ),
            "",
        ),
        ("交易次数 (Trades)", len(result.trades_df), f"{len(result.trades_df)}", ""),
    ]

    metrics_html = ""
    for label, _raw_val, fmt_val, color_cls in metric_data:
        metrics_html += f"""
        <div class="metric-card">
            <div class="metric-value {color_cls}">{fmt_val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """
    return metrics_html


def _select_plot_symbol(
    result: Any,
    market_data: Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]],
    plot_symbol: Optional[str],
) -> Optional[str]:
    if plot_symbol:
        return str(plot_symbol)
    trades_df = getattr(result, "trades_df", pd.DataFrame())
    if (
        isinstance(trades_df, pd.DataFrame)
        and not trades_df.empty
        and "symbol" in trades_df.columns
    ):
        counts = trades_df["symbol"].dropna().astype(str).value_counts()
        if not counts.empty:
            return str(counts.index[0])
    if isinstance(market_data, dict):
        keys = [str(k) for k in market_data.keys()]
        if keys:
            return keys[0]
    if isinstance(market_data, pd.DataFrame):
        symbol_col = _resolve_market_data_column(
            market_data, ["symbol", "code", "ticker", "ts_code", "股票代码"]
        )
        if symbol_col is not None:
            symbols = market_data[symbol_col].dropna().astype(str).str.strip()
            if not symbols.empty:
                return str(symbols.iloc[0])
    return None


def _resolve_market_data_column(
    data: pd.DataFrame, candidates: list[str]
) -> Optional[str]:
    """Resolve a market-data column name with case-insensitive matching."""
    columns = list(data.columns)
    lowered = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        resolved = lowered.get(candidate.lower())
        if resolved is not None:
            return resolved
    return None


def _normalize_market_data_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize report market data into a canonical OHLCV schema."""
    normalized = data.copy()
    resolved_columns = {
        "timestamp": _resolve_market_data_column(
            normalized,
            ["date", "timestamp", "datetime", "time", "trade_date", "日期", "时间"],
        ),
        "open": _resolve_market_data_column(normalized, ["open", "open_price", "开盘"]),
        "high": _resolve_market_data_column(normalized, ["high", "high_price", "最高"]),
        "low": _resolve_market_data_column(normalized, ["low", "low_price", "最低"]),
        "close": _resolve_market_data_column(
            normalized, ["close", "close_price", "收盘"]
        ),
        "volume": _resolve_market_data_column(normalized, ["volume", "vol", "成交量"]),
        "symbol": _resolve_market_data_column(
            normalized, ["symbol", "code", "ticker", "ts_code", "股票代码"]
        ),
    }
    rename_map = {
        source: target
        for target, source in resolved_columns.items()
        if source is not None and source != target
    }
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    if "symbol" in normalized.columns:
        normalized["symbol"] = normalized["symbol"].astype(str).str.strip()

    if not isinstance(normalized.index, pd.DatetimeIndex):
        if "timestamp" in normalized.columns:
            normalized = normalized.set_index("timestamp")
        else:
            normalized.index = pd.to_datetime(normalized.index, errors="coerce")

    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    valid_index_mask = ~normalized.index.to_series().isna()
    normalized = normalized.loc[valid_index_mask].copy()
    if normalized.empty:
        return cast(pd.DataFrame, normalized)

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(set(normalized.columns)):
        return pd.DataFrame()

    normalized = normalized.dropna(subset=list(required_cols), how="any")
    if normalized.empty:
        return cast(pd.DataFrame, normalized)

    normalized = normalized.sort_index()
    return cast(pd.DataFrame, normalized)


def _extract_symbol_market_data(
    market_data: Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]], symbol: str
) -> pd.DataFrame:
    if market_data is None:
        return pd.DataFrame()
    if isinstance(market_data, dict):
        matched_key = None
        target_symbol = str(symbol).strip()
        for key in market_data.keys():
            if str(key).strip() == target_symbol:
                matched_key = key
                break
        if matched_key is None:
            return pd.DataFrame()
        data = market_data.get(matched_key, pd.DataFrame()).copy()
    elif isinstance(market_data, pd.DataFrame):
        data = market_data.copy()
    else:
        return pd.DataFrame()

    data = _normalize_market_data_frame(data)
    if data.empty:
        return cast(pd.DataFrame, data)
    if "symbol" in data.columns:
        target_symbol = str(symbol).strip()
        symbol_mask = data["symbol"].astype(str).str.strip() == target_symbol
        data = data[symbol_mask].copy()
    return cast(pd.DataFrame, data)


def _build_chart_html_sections(
    result: Any,
    market_data: Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]] = None,
    plot_symbol: Optional[str] = None,
    include_trade_kline: bool = True,
    include_indicators: bool = False,
    indicator_name: Optional[str] = None,
    indicator_symbol: Optional[str] = None,
    indicator_include_warmup: bool = True,
    benchmark: Optional[Union[str, pd.Series]] = None,
    curve_freq: str = "D",
) -> dict[str, str]:
    """Build chart HTML sections from plot figures."""
    config = {"responsive": True}
    strategy_config = {
        "responsive": True,
        "scrollZoom": True,
        "displayModeBar": True,
        "doubleClick": "reset",
    }

    equity_curve = _resolve_equity_curve(result, curve_freq)
    fig_dashboard = plot_dashboard(
        result, show=False, theme="light", equity_series=equity_curve
    )
    dashboard_html = (
        fig_dashboard.to_html(full_html=False, include_plotlyjs=False, config=config)
        if fig_dashboard
        else "<div>暂无数据</div>"
    )
    indicator_section_html = ""
    if include_indicators:
        indicator_df = result.indicator_df(name=indicator_name, symbol=indicator_symbol)
        if not indicator_include_warmup and not indicator_df.empty:
            indicator_df = indicator_df.loc[~indicator_df["warmup"]]
        indicator_defs = result.indicator_definitions.copy()
        if indicator_name is not None and not indicator_defs.empty:
            indicator_defs = indicator_defs.loc[
                indicator_defs["indicator_key"] == str(indicator_name)
            ]
        indicator_symbol_count = (
            indicator_df["symbol"].dropna().astype(str).nunique()
            if not indicator_df.empty
            else 0
        )
        overview_html = (
            '<div class="analysis-overview-grid">'
            f'<div class="analysis-card"><div class="analysis-card-label">'
            "指标定义数 (Indicator Definitions)</div>"
            f'<div class="analysis-card-value">{len(indicator_defs)}</div></div>'
            f'<div class="analysis-card"><div class="analysis-card-label">'
            "指标点位数 (Indicator Points)</div>"
            f'<div class="analysis-card-value">{len(indicator_df)}</div></div>'
            f'<div class="analysis-card"><div class="analysis-card-label">'
            "标的数 (Symbols)</div>"
            f'<div class="analysis-card-value">'
            f"{indicator_symbol_count}"
            "</div></div>"
            "</div>"
        )
        filter_parts: list[str] = []
        if indicator_name is not None:
            filter_parts.append(f"指标={indicator_name}")
        if indicator_symbol is not None:
            filter_parts.append(f"标的={indicator_symbol}")
        if not indicator_include_warmup:
            filter_parts.append("已排除预热点")
        filter_text = " | ".join(filter_parts)
        filter_hint = (
            '<div class="empty-panel" style="margin-bottom: 16px;">'
            f"过滤条件: {filter_text}</div>"
            if filter_parts
            else ""
        )
        if indicator_df.empty:
            indicator_chart_html = "<div class='empty-panel'>暂无指标数据</div>"
        else:
            fig_indicator = plot_indicators(
                result=result,
                name=indicator_name,
                symbol=indicator_symbol,
                include_warmup=indicator_include_warmup,
                show=False,
                title="报告内指标预览 (Indicator Preview)",
                theme="light",
            )
            indicator_chart_html = (
                fig_indicator.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config=config,
                )
                if fig_indicator
                else "<div class='empty-panel'>暂无指标图表</div>"
            )
        if indicator_defs.empty:
            indicator_defs_html = "<div>暂无指标定义数据</div>"
        else:
            indicator_defs_view = _rename_table_columns(
                indicator_defs,
                {
                    "indicator_key": "指标键 (Indicator Key)",
                    "display_name": "展示名 (Display Name)",
                    "pane": "面板 (Pane)",
                    "render_type": "绘制方式 (Render Type)",
                    "unit": "单位 (Unit)",
                    "precision": "精度 (Precision)",
                    "color": "颜色 (Color)",
                },
            )
            indicator_defs_html = _format_table(
                indicator_defs_view,
                max_rows=20,
                compact_currency=False,
            )
        indicator_section_html = (
            '<div class="section-title">自定义指标 (Custom Indicators)</div>'
            f"{filter_hint}"
            f"{overview_html}"
            '<div class="chart-container">'
            f"{indicator_chart_html}"
            "</div>"
            '<details class="details-block" style="margin-top: 20px;">'
            "<summary>指标定义明细 (Indicator Definitions)</summary>"
            f'<div class="details-content">{indicator_defs_html}</div>'
            "</details>"
        )

    returns_series = _build_daily_returns_from_equity(equity_curve)
    fig_rolling = plot_rolling_metrics(returns_series, theme="light")
    rolling_metrics_html = (
        fig_rolling.to_html(full_html=False, include_plotlyjs=False, config=config)
        if fig_rolling
        else "<div>暂无数据</div>"
    )
    fig_dist_ret = plot_returns_distribution(returns_series, theme="light")
    returns_dist_html = (
        fig_dist_ret.to_html(full_html=False, include_plotlyjs=False, config=config)
        if fig_dist_ret
        else "<div>暂无数据</div>"
    )
    fig_yearly = plot_yearly_returns(returns_series, theme="light")
    yearly_returns_html = (
        fig_yearly.to_html(full_html=False, include_plotlyjs=False, config=config)
        if fig_yearly
        else "<div>暂无数据</div>"
    )
    benchmark_sections = _build_benchmark_sections(
        strategy_returns=returns_series,
        benchmark=benchmark,
        config=config,
    )

    fig_dist = plot_trades_distribution(result.trades_df)
    trades_dist_html = (
        fig_dist.to_html(full_html=False, include_plotlyjs=False, config=config)
        if fig_dist
        else "<div>无交易数据</div>"
    )
    fig_duration = plot_pnl_vs_duration(result.trades_df)
    pnl_duration_html = (
        fig_duration.to_html(full_html=False, include_plotlyjs=False, config=config)
        if fig_duration
        else "<div>无交易数据</div>"
    )
    strategy_kline_html = "<div>未提供行情数据，已跳过 K 线复盘图</div>"
    if not include_trade_kline:
        strategy_kline_html = "<div>已关闭 K 线复盘图</div>"
    elif market_data is not None:
        symbol = _select_plot_symbol(result, market_data, plot_symbol)
        if symbol:
            symbol_data = _extract_symbol_market_data(market_data, symbol)
            if not symbol_data.empty:
                fig_strategy = plot_strategy(
                    result=result,
                    symbol=symbol,
                    data=symbol_data,
                    theme="light",
                    show=False,
                )
                if fig_strategy:
                    strategy_kline_html = fig_strategy.to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        config=strategy_config,
                    )
                else:
                    strategy_kline_html = "<div>未能生成 K 线复盘图</div>"
            else:
                strategy_kline_html = "<div>行情数据不完整，无法绘制 K 线复盘图</div>"
    risk_reject_ratio_html = "<div>暂无策略级风控拒单占比图</div>"
    risk_reason_ratio_html = "<div>暂无策略级拒单原因占比图</div>"
    risk_reject_trend_html = "<div>暂无按日风控拒单趋势图</div>"
    risk_reject_trend_by_strategy_html = "<div>暂无按策略风控拒单趋势图</div>"
    risk_reason_trend_html = "<div>暂无按日拒单原因趋势图</div>"
    risk_other_reason_table_html = ""
    has_risk_ratio_chart = False
    has_risk_reason_ratio_chart = False
    has_risk_trend_chart = False
    has_risk_strategy_trend_chart = False
    has_risk_reason_trend_chart = False
    has_liquidation_count_chart = False
    has_liquidation_interest_chart = False
    total_reject_count = 0.0
    risk_df = (
        result.risk_rejections_by_strategy()
        if hasattr(result, "risk_rejections_by_strategy")
        else pd.DataFrame()
    )
    top_reasons_df = (
        result.top_reject_reason_types(top_n=8)
        if hasattr(result, "top_reject_reason_types")
        else pd.DataFrame()
    )
    if not top_reasons_df.empty:
        top_reasons_view = _rename_table_columns(
            top_reasons_df,
            {
                "reject_reason_type": "拒单类型 (Reject Type)",
                "sample_reject_reason": "示例详情 (Sample Detail)",
                "count": "拒单数 (Count)",
                "ratio": "占比 (Ratio)",
            },
        )
        top_reasons_table_html = _format_table(
            top_reasons_view,
            max_rows=8,
            percentage_columns={"占比 (Ratio)"},
            compact_currency=False,
        )
        risk_other_reason_table_html = (
            '<div class="chart-container" style="margin-top: 20px;">'
            "<h3 style='margin: 0 0 10px 0;'>"
            "拒单原因 Top 8 明细 (Top Reject Reasons)"
            "</h3>"
            f"{top_reasons_table_html}"
            "</div>"
        )
    if not risk_df.empty and "risk_reject_count" in risk_df.columns:
        risk_base_df = risk_df.copy()
        if "owner_strategy_id" not in risk_base_df.columns:
            risk_base_df["owner_strategy_id"] = "_default"
        risk_base_df["owner_strategy_id"] = (
            risk_base_df["owner_strategy_id"].fillna("_default").astype(str)
        )
        reject_count = pd.to_numeric(
            risk_base_df["risk_reject_count"], errors="coerce"
        ).fillna(0.0)
        total_reject_count = float(reject_count.sum())
        if total_reject_count > 0.0:
            px = __import__("plotly.express", fromlist=["bar"])

            ratio_df = pd.DataFrame(
                {
                    "owner_strategy_id": risk_base_df["owner_strategy_id"],
                    "reject_ratio": reject_count / total_reject_count,
                    "risk_reject_count": reject_count,
                }
            ).sort_values("reject_ratio", ascending=False)
            fig_risk_ratio = px.bar(
                ratio_df,
                x="owner_strategy_id",
                y="reject_ratio",
                title="策略级风控拒单占比 (Risk Reject Ratio by Strategy)",
                text=ratio_df["reject_ratio"].map(lambda v: f"{v:.1%}"),
                labels={
                    "owner_strategy_id": "策略ID (Strategy ID)",
                    "reject_ratio": "拒单占比 (Reject Ratio)",
                },
            )
            fig_risk_ratio.update_traces(
                hovertemplate=(
                    "策略ID=%{x}<br>拒单占比=%{y:.1%}<br>"
                    "拒单数=%{customdata[0]:.0f}<extra></extra>"
                ),
                customdata=ratio_df[["risk_reject_count"]].to_numpy(),
            )
            fig_risk_ratio.update_yaxes(tickformat=".0%")
            fig_risk_ratio.update_layout(
                height=320, margin=dict(l=20, r=20, t=60, b=20)
            )
            risk_reject_ratio_html = fig_risk_ratio.to_html(
                full_html=False, include_plotlyjs=False, config=config
            )
            has_risk_ratio_chart = True
        reason_columns = [
            ("daily_loss_reject_count", "Daily Loss"),
            ("drawdown_reject_count", "Drawdown"),
            ("reduce_only_reject_count", "Reduce-Only"),
            ("position_limit_reject_count", "Position Limit"),
            ("order_size_limit_reject_count", "Order Size Limit"),
            ("order_value_limit_reject_count", "Order Value Limit"),
            ("strategy_risk_budget_reject_count", "Strategy Risk Budget"),
            ("portfolio_risk_budget_reject_count", "Portfolio Risk Budget"),
            ("other_risk_reject_count", "Other"),
        ]
        available_reason_columns = [
            (column_name, label)
            for column_name, label in reason_columns
            if column_name in risk_base_df.columns
        ]
        if available_reason_columns:
            stacked_df = pd.DataFrame(
                {"owner_strategy_id": risk_base_df["owner_strategy_id"]}
            )
            for column_name, label in available_reason_columns:
                values = pd.to_numeric(
                    risk_base_df[column_name], errors="coerce"
                ).fillna(0.0)
                stacked_df[label] = values
            totals = stacked_df.drop(columns=["owner_strategy_id"]).sum(axis=1)
            non_zero_totals = totals > 0
            if bool(non_zero_totals.any()):
                stacked_df = stacked_df.loc[non_zero_totals].reset_index(drop=True)
                totals = totals.loc[non_zero_totals].reset_index(drop=True)
                value_columns = [
                    col for col in stacked_df.columns if col != "owner_strategy_id"
                ]
                ratio_stacked = stacked_df.copy()
                ratio_stacked[value_columns] = ratio_stacked[value_columns].div(
                    totals, axis=0
                )
                px = __import__("plotly.express", fromlist=["bar"])
                long_df = ratio_stacked.melt(
                    id_vars=["owner_strategy_id"],
                    value_vars=value_columns,
                    var_name="risk_reason",
                    value_name="reject_ratio",
                )
                fig_reason_ratio = px.bar(
                    long_df,
                    x="owner_strategy_id",
                    y="reject_ratio",
                    color="risk_reason",
                    title="策略级拒单原因占比 (Risk Reason Ratio by Strategy)",
                    labels={
                        "owner_strategy_id": "策略ID (Strategy ID)",
                        "reject_ratio": "拒单原因占比 (Reason Ratio)",
                        "risk_reason": "拒单原因 (Reason)",
                    },
                )
                fig_reason_ratio.update_layout(
                    barmode="stack",
                    height=360,
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                fig_reason_ratio.update_yaxes(tickformat=".0%")
                fig_reason_ratio.update_traces(
                    hovertemplate=(
                        "策略ID=%{x}<br>原因=%{fullData.name}<br>"
                        "占比=%{y:.1%}<extra></extra>"
                    )
                )
                risk_reason_ratio_html = fig_reason_ratio.to_html(
                    full_html=False, include_plotlyjs=False, config=config
                )
                has_risk_reason_ratio_chart = True
    risk_trend_df = (
        result.risk_rejections_trend(freq="D")
        if hasattr(result, "risk_rejections_trend")
        else pd.DataFrame()
    )
    if not risk_trend_df.empty:
        trend_df = risk_trend_df.copy()
        trend_df["date"] = pd.to_datetime(trend_df["date"], errors="coerce")
        trend_df = trend_df.dropna(subset=["date"]).sort_values("date")
        if not trend_df.empty and "risk_reject_count" in trend_df.columns:
            px = __import__("plotly.express", fromlist=["line"])
            trend_df["risk_reject_count"] = pd.to_numeric(
                trend_df["risk_reject_count"], errors="coerce"
            ).fillna(0.0)
            fig_risk_trend = px.line(
                trend_df,
                x="date",
                y="risk_reject_count",
                markers=True,
                title="按日风控拒单趋势 (Daily Risk Reject Trend)",
                labels={
                    "date": "日期 (Date)",
                    "risk_reject_count": "风控拒单数 (Risk Reject Count)",
                },
            )
            fig_risk_trend.update_layout(
                height=320, margin=dict(l=20, r=20, t=60, b=20)
            )
            fig_risk_trend.update_traces(
                hovertemplate="日期=%{x}<br>拒单数=%{y:.0f}<extra></extra>"
            )
            risk_reject_trend_html = fig_risk_trend.to_html(
                full_html=False, include_plotlyjs=False, config=config
            )
            has_risk_trend_chart = True
            reason_columns = [
                ("daily_loss_reject_count", "Daily Loss"),
                ("drawdown_reject_count", "Drawdown"),
                ("reduce_only_reject_count", "Reduce-Only"),
                ("position_limit_reject_count", "Position Limit"),
                ("order_size_limit_reject_count", "Order Size Limit"),
                ("order_value_limit_reject_count", "Order Value Limit"),
                ("strategy_risk_budget_reject_count", "Strategy Risk Budget"),
                ("portfolio_risk_budget_reject_count", "Portfolio Risk Budget"),
                ("other_risk_reject_count", "Other"),
            ]
            available_reason_columns = [
                (column_name, label)
                for column_name, label in reason_columns
                if column_name in trend_df.columns
            ]
            if available_reason_columns:
                reason_trend_df = pd.DataFrame({"date": trend_df["date"]})
                for column_name, label in available_reason_columns:
                    values = pd.to_numeric(
                        trend_df[column_name], errors="coerce"
                    ).fillna(0.0)
                    reason_trend_df[label] = values
                long_reason_df = reason_trend_df.melt(
                    id_vars=["date"],
                    value_vars=[
                        col for col in reason_trend_df.columns if col != "date"
                    ],
                    var_name="risk_reason",
                    value_name="reject_count",
                )
                fig_reason_trend = px.area(
                    long_reason_df,
                    x="date",
                    y="reject_count",
                    color="risk_reason",
                    title="按日拒单原因趋势 (Daily Risk Reason Trend)",
                    labels={
                        "date": "日期 (Date)",
                        "reject_count": "拒单数 (Reject Count)",
                        "risk_reason": "拒单原因 (Reason)",
                    },
                )
                fig_reason_trend.update_layout(
                    height=340, margin=dict(l=20, r=20, t=60, b=20)
                )
                fig_reason_trend.update_traces(
                    hovertemplate=(
                        "日期=%{x}<br>原因=%{fullData.name}<br>"
                        "拒单数=%{y:.0f}<extra></extra>"
                    )
                )
                risk_reason_trend_html = fig_reason_trend.to_html(
                    full_html=False, include_plotlyjs=False, config=config
                )
                has_risk_reason_trend_chart = True
    trend_by_strategy_df = (
        result.risk_rejections_trend_by_strategy(freq="D")
        if hasattr(result, "risk_rejections_trend_by_strategy")
        else pd.DataFrame()
    )
    if not trend_by_strategy_df.empty:
        strategy_trend_df = trend_by_strategy_df.copy()
        strategy_trend_df["date"] = pd.to_datetime(
            strategy_trend_df["date"], errors="coerce"
        )
        strategy_trend_df = strategy_trend_df.dropna(subset=["date"]).sort_values(
            ["date", "owner_strategy_id"]
        )
        if (
            not strategy_trend_df.empty
            and "risk_reject_count" in strategy_trend_df.columns
        ):
            px = __import__("plotly.express", fromlist=["line"])
            strategy_trend_df["owner_strategy_id"] = (
                strategy_trend_df["owner_strategy_id"].fillna("_default").astype(str)
            )
            strategy_trend_df["risk_reject_count"] = pd.to_numeric(
                strategy_trend_df["risk_reject_count"], errors="coerce"
            ).fillna(0.0)
            fig_risk_strategy_trend = px.line(
                strategy_trend_df,
                x="date",
                y="risk_reject_count",
                color="owner_strategy_id",
                markers=True,
                title="按策略风控拒单趋势 (Risk Reject Trend by Strategy)",
                labels={
                    "date": "日期 (Date)",
                    "risk_reject_count": "风控拒单数 (Risk Reject Count)",
                    "owner_strategy_id": "策略ID (Strategy ID)",
                },
            )
            fig_risk_strategy_trend.update_layout(
                height=340, margin=dict(l=20, r=20, t=60, b=20)
            )
            fig_risk_strategy_trend.update_traces(
                hovertemplate=(
                    "日期=%{x}<br>策略ID=%{fullData.name}<br>"
                    "拒单数=%{y:.0f}<extra></extra>"
                )
            )
            risk_reject_trend_by_strategy_html = fig_risk_strategy_trend.to_html(
                full_html=False, include_plotlyjs=False, config=config
            )
            has_risk_strategy_trend_chart = True

    liquidation_audit_df = (
        result.liquidation_audit_df
        if hasattr(result, "liquidation_audit_df")
        else pd.DataFrame()
    )
    liquidation_count_chart_html = ""
    liquidation_interest_chart_html = ""
    if not liquidation_audit_df.empty:
        liq_df = liquidation_audit_df.copy()
        if "timestamp" in liq_df.columns:
            liq_df["timestamp"] = pd.to_datetime(liq_df["timestamp"], errors="coerce")
        elif "date" in liq_df.columns:
            liq_df["timestamp"] = pd.to_datetime(liq_df["date"], errors="coerce")
        else:
            liq_df["timestamp"] = pd.NaT
        liq_df = liq_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        if not liq_df.empty and "liquidated_count" in liq_df.columns:
            liq_df["liquidated_count"] = pd.to_numeric(
                liq_df["liquidated_count"], errors="coerce"
            ).fillna(0.0)
            daily_liq = (
                liq_df.set_index("timestamp")["liquidated_count"].resample("D").sum()
            ).reset_index()
            if not daily_liq.empty and float(daily_liq["liquidated_count"].sum()) > 0.0:
                px = __import__("plotly.express", fromlist=["line"])
                fig_liq_count = px.line(
                    daily_liq,
                    x="timestamp",
                    y="liquidated_count",
                    markers=True,
                    title="按日强平标的数趋势 (Daily Liquidated Symbols Trend)",
                    labels={
                        "timestamp": "日期 (Date)",
                        "liquidated_count": "强平标的数 (Liquidated Symbols)",
                    },
                )
                fig_liq_count.update_layout(
                    height=320, margin=dict(l=20, r=20, t=60, b=20)
                )
                fig_liq_count.update_traces(
                    hovertemplate="日期=%{x}<br>强平标的数=%{y:.0f}<extra></extra>"
                )
                liquidation_count_chart_html = fig_liq_count.to_html(
                    full_html=False, include_plotlyjs=False, config=config
                )
                has_liquidation_count_chart = True

        if not liq_df.empty and "daily_interest" in liq_df.columns:
            liq_df["daily_interest"] = pd.to_numeric(
                liq_df["daily_interest"], errors="coerce"
            ).fillna(0.0)
            daily_interest = (
                liq_df.set_index("timestamp")["daily_interest"].resample("D").sum()
            ).reset_index()
            has_positive_interest = bool((daily_interest["daily_interest"] > 0.0).any())
            if not daily_interest.empty and has_positive_interest:
                px = __import__("plotly.express", fromlist=["bar"])
                fig_liq_interest = px.bar(
                    daily_interest,
                    x="timestamp",
                    y="daily_interest",
                    title="按日强平计息 (Daily Liquidation Interest)",
                    labels={
                        "timestamp": "日期 (Date)",
                        "daily_interest": "当日利息 (Daily Interest)",
                    },
                )
                fig_liq_interest.update_layout(
                    height=320, margin=dict(l=20, r=20, t=60, b=20)
                )
                fig_liq_interest.update_traces(
                    hovertemplate="日期=%{x}<br>当日利息=%{y:.4f}<extra></extra>"
                )
                liquidation_interest_chart_html = fig_liq_interest.to_html(
                    full_html=False, include_plotlyjs=False, config=config
                )
                has_liquidation_interest_chart = True

    risk_chart_blocks: list[str] = []

    def append_risk_chart_block(chart_html: str) -> None:
        risk_chart_blocks.append(
            (
                '<div class="chart-container" style="margin-top: 20px;">'
                f"{chart_html}"
                "</div>"
            )
        )

    if has_risk_ratio_chart:
        append_risk_chart_block(risk_reject_ratio_html)
    if has_risk_reason_ratio_chart:
        append_risk_chart_block(risk_reason_ratio_html)
    if has_risk_trend_chart:
        append_risk_chart_block(risk_reject_trend_html)
    if has_risk_strategy_trend_chart:
        append_risk_chart_block(risk_reject_trend_by_strategy_html)
    if has_risk_reason_trend_chart:
        append_risk_chart_block(risk_reason_trend_html)
    if has_liquidation_count_chart:
        append_risk_chart_block(liquidation_count_chart_html)
    if has_liquidation_interest_chart:
        append_risk_chart_block(liquidation_interest_chart_html)
    if risk_other_reason_table_html:
        risk_chart_blocks.append(risk_other_reason_table_html)

    if risk_chart_blocks:
        risk_charts_html = "".join(risk_chart_blocks)
    else:
        if risk_df.empty:
            reason_text = "本次回测未产生策略级风控拒单统计数据。"
        elif total_reject_count <= 0.0:
            reason_text = "本次回测风控拒单总数为 0，未触发拒单。"
        else:
            reason_text = "本次回测未形成可绘制的风控拒单图表。"
        risk_charts_html = (
            '<div class="chart-container" style="margin-top: 20px;">'
            '<div class="empty-panel">'
            f"{reason_text}<br>"
            "建议：可降低风险阈值或增加高波动样本，观察风控拒单分布与趋势。"
            "</div></div>"
        )

    return {
        "dashboard_html": dashboard_html,
        "indicator_section_html": indicator_section_html,
        "yearly_returns_html": yearly_returns_html,
        "returns_dist_html": returns_dist_html,
        "rolling_metrics_html": rolling_metrics_html,
        "benchmark_metrics_html": benchmark_sections["benchmark_metrics_html"],
        "benchmark_chart_html": benchmark_sections["benchmark_chart_html"],
        "trades_dist_html": trades_dist_html,
        "pnl_duration_html": pnl_duration_html,
        "strategy_kline_html": strategy_kline_html,
        "risk_reject_ratio_html": risk_reject_ratio_html,
        "risk_reason_ratio_html": risk_reason_ratio_html,
        "risk_reject_trend_html": risk_reject_trend_html,
        "risk_reject_trend_by_strategy_html": risk_reject_trend_by_strategy_html,
        "risk_reason_trend_html": risk_reason_trend_html,
        "risk_charts_html": risk_charts_html,
    }


def _build_analysis_table_sections(
    result: Any, compact_currency: bool = True
) -> dict[str, str]:
    """Build attribution/exposure/capacity HTML tables."""
    overview_cards: list[tuple[str, str]] = []

    def add_overview_card(label: str, value: str) -> None:
        overview_cards.append((label, value))

    exposure_df = (
        result.exposure_df() if hasattr(result, "exposure_df") else pd.DataFrame()
    )
    if not exposure_df.empty:
        latest_net_exposure_pct = float(exposure_df["net_exposure_pct"].iloc[-1])
        latest_gross_exposure_pct = float(exposure_df["gross_exposure_pct"].iloc[-1])
        max_leverage = float(exposure_df["leverage"].max())
        add_overview_card("最新净暴露比", f"{latest_net_exposure_pct * 100:.2f}%")
        add_overview_card("最新总暴露比", f"{latest_gross_exposure_pct * 100:.2f}%")
        add_overview_card("最大杠杆", f"{max_leverage:.4f}")
        exposure_view = pd.DataFrame(
            [
                {
                    "latest_net_exposure_pct": latest_net_exposure_pct,
                    "latest_gross_exposure_pct": latest_gross_exposure_pct,
                    "max_leverage": max_leverage,
                }
            ]
        )
        exposure_view = _rename_table_columns(
            exposure_view,
            {
                "latest_net_exposure_pct": "最新净暴露比 (Latest Net Exposure %)",
                "latest_gross_exposure_pct": "最新总暴露比 (Latest Gross Exposure %)",
                "max_leverage": "最大杠杆 (Max Leverage)",
            },
        )
        exposure_summary_html = _format_table(
            exposure_view,
            max_rows=1,
            percentage_columns={
                "最新净暴露比 (Latest Net Exposure %)",
                "最新总暴露比 (Latest Gross Exposure %)",
            },
            compact_currency=compact_currency,
        )
    else:
        exposure_summary_html = "<div>暂无暴露数据</div>"

    capacity_df = (
        result.capacity_df() if hasattr(result, "capacity_df") else pd.DataFrame()
    )
    if not capacity_df.empty:
        total_order_count = float(capacity_df["order_count"].sum())
        total_filled_value = float(capacity_df["filled_value"].sum())
        avg_fill_rate_qty = float(capacity_df["fill_rate_qty"].mean())
        avg_turnover = float(capacity_df["turnover"].mean())
        add_overview_card("总订单数", f"{total_order_count:,.0f}")
        add_overview_card("总成交额", _format_currency(total_filled_value))
        add_overview_card("平均成交率", f"{avg_fill_rate_qty * 100:.2f}%")
        add_overview_card("平均换手率", f"{avg_turnover * 100:.2f}%")
        capacity_view = pd.DataFrame(
            [
                {
                    "total_order_count": total_order_count,
                    "total_filled_value": total_filled_value,
                    "avg_fill_rate_qty": avg_fill_rate_qty,
                    "avg_turnover": avg_turnover,
                }
            ]
        )
        capacity_view = _rename_table_columns(
            capacity_view,
            {
                "total_order_count": "总订单数 (Total Orders)",
                "total_filled_value": "总成交额 (Total Filled Value)",
                "avg_fill_rate_qty": "平均成交率 (Avg Fill Rate Qty)",
                "avg_turnover": "平均换手率 (Avg Turnover)",
            },
        )
        capacity_summary_html = _format_table(
            capacity_view,
            max_rows=1,
            percentage_columns={
                "平均成交率 (Avg Fill Rate Qty)",
                "平均换手率 (Avg Turnover)",
            },
            compact_currency_columns={"总成交额 (Total Filled Value)"},
            compact_currency=compact_currency,
        )
    else:
        capacity_summary_html = "<div>暂无容量数据</div>"

    attribution_df = (
        result.attribution_df(by="symbol")
        if hasattr(result, "attribution_df")
        else pd.DataFrame()
    )
    if not attribution_df.empty:
        total_pnl = float(attribution_df["total_pnl"].sum())
        total_commission = float(attribution_df["total_commission"].sum())
        total_trade_count = float(attribution_df["trade_count"].sum())
        add_overview_card("归因总盈亏", _format_currency(total_pnl))
        add_overview_card("归因总手续费", _format_currency(total_commission))
        add_overview_card("归因交易次数", f"{total_trade_count:,.0f}")
        cols = [
            "group",
            "trade_count",
            "total_pnl",
            "contribution_pct",
            "total_commission",
        ]
        cols = [c for c in cols if c in attribution_df.columns]
        attribution_view = _rename_table_columns(
            attribution_df[cols],
            {
                "group": "分组 (Group)",
                "trade_count": "交易次数 (Trade Count)",
                "total_pnl": "总盈亏 (Total PnL)",
                "contribution_pct": "贡献占比 (Contribution %)",
                "total_commission": "总手续费 (Total Commission)",
            },
        )
        attribution_summary_html = _format_table(
            attribution_view,
            max_rows=10,
            percentage_columns={"贡献占比 (Contribution %)"},
            compact_currency_columns={
                "总盈亏 (Total PnL)",
                "总手续费 (Total Commission)",
            },
            compact_currency=compact_currency,
        )
    else:
        attribution_summary_html = "<div>暂无归因数据</div>"

    orders_by_strategy_df = (
        result.orders_by_strategy()
        if hasattr(result, "orders_by_strategy")
        else pd.DataFrame()
    )
    if not orders_by_strategy_df.empty:
        cols = [
            "owner_strategy_id",
            "order_count",
            "filled_order_count",
            "filled_quantity",
            "filled_value",
            "fill_rate_qty",
        ]
        cols = [c for c in cols if c in orders_by_strategy_df.columns]
        orders_by_strategy_view = _rename_table_columns(
            orders_by_strategy_df[cols],
            {
                "owner_strategy_id": "策略ID (Strategy ID)",
                "order_count": "订单数 (Orders)",
                "filled_order_count": "已成交订单数 (Filled Orders)",
                "filled_quantity": "成交数量 (Filled Qty)",
                "filled_value": "成交额 (Filled Value)",
                "fill_rate_qty": "数量成交率 (Fill Rate Qty)",
            },
        )
        orders_by_strategy_html = _format_table(
            orders_by_strategy_view,
            max_rows=20,
            percentage_columns={"数量成交率 (Fill Rate Qty)"},
            compact_currency_columns={"成交额 (Filled Value)"},
            compact_currency=compact_currency,
        )
    else:
        orders_by_strategy_html = "<div>暂无策略归属订单聚合数据</div>"

    executions_by_strategy_df = (
        result.executions_by_strategy()
        if hasattr(result, "executions_by_strategy")
        else pd.DataFrame()
    )
    if not executions_by_strategy_df.empty:
        cols = [
            "owner_strategy_id",
            "execution_count",
            "total_quantity",
            "total_notional",
            "total_commission",
            "avg_fill_price",
        ]
        cols = [c for c in cols if c in executions_by_strategy_df.columns]
        executions_by_strategy_view = _rename_table_columns(
            executions_by_strategy_df[cols],
            {
                "owner_strategy_id": "策略ID (Strategy ID)",
                "execution_count": "成交笔数 (Executions)",
                "total_quantity": "总成交数量 (Total Qty)",
                "total_notional": "总成交额 (Total Notional)",
                "total_commission": "总手续费 (Total Commission)",
                "avg_fill_price": "平均成交价 (Avg Fill Price)",
            },
        )
        executions_by_strategy_html = _format_table(
            executions_by_strategy_view,
            max_rows=20,
            compact_currency_columns={
                "总成交额 (Total Notional)",
                "总手续费 (Total Commission)",
            },
            compact_currency=compact_currency,
        )
    else:
        executions_by_strategy_html = "<div>暂无策略归属成交聚合数据</div>"

    risk_by_strategy_df = (
        result.risk_rejections_by_strategy()
        if hasattr(result, "risk_rejections_by_strategy")
        else pd.DataFrame()
    )
    if not risk_by_strategy_df.empty:
        cols = [
            "owner_strategy_id",
            "risk_reject_count",
            "daily_loss_reject_count",
            "drawdown_reject_count",
            "reduce_only_reject_count",
            "strategy_risk_budget_reject_count",
            "portfolio_risk_budget_reject_count",
            "other_risk_reject_count",
        ]
        cols = [c for c in cols if c in risk_by_strategy_df.columns]
        risk_by_strategy_view = _rename_table_columns(
            risk_by_strategy_df[cols],
            {
                "owner_strategy_id": "策略ID (Strategy ID)",
                "risk_reject_count": "风险拒单总数 (Risk Rejects)",
                "daily_loss_reject_count": "日损拒单数 (Daily Loss Rejects)",
                "drawdown_reject_count": "回撤拒单数 (Drawdown Rejects)",
                "reduce_only_reject_count": "仅平仓拒单数 (Reduce-Only Rejects)",
                "strategy_risk_budget_reject_count": (
                    "策略预算拒单数 (Strategy Budget Rejects)"
                ),
                "portfolio_risk_budget_reject_count": (
                    "组合预算拒单数 (Portfolio Budget Rejects)"
                ),
                "other_risk_reject_count": "其他拒单数 (Other Rejects)",
            },
        )
        risk_by_strategy_html = _format_table(
            risk_by_strategy_view,
            max_rows=20,
        )
    else:
        risk_by_strategy_html = "<div>暂无策略归属风控拒单聚合数据</div>"

    liquidation_audit_df = (
        result.liquidation_audit_df
        if hasattr(result, "liquidation_audit_df")
        else pd.DataFrame()
    )
    if not liquidation_audit_df.empty:
        liq_df = liquidation_audit_df.copy()
        cols = [
            "timestamp",
            "date",
            "daily_interest",
            "liquidated_count",
            "liquidated_symbols",
            "priority",
        ]
        cols = [c for c in cols if c in liq_df.columns]
        if "priority" in liq_df.columns:
            liq_df["priority"] = (
                liq_df["priority"]
                .astype(str)
                .replace({"short_first": "先平空头", "long_first": "先平多头"})
            )
        if "liquidated_symbols" in liq_df.columns:
            liq_df["liquidated_symbols"] = (
                liq_df["liquidated_symbols"].astype(str).str.replace(",", ", ")
            )
        liquidation_view = _rename_table_columns(
            liq_df[cols],
            {
                "timestamp": "时间戳 (Timestamp)",
                "date": "日期 (Date)",
                "daily_interest": "当日利息 (Daily Interest)",
                "liquidated_count": "强平标的数 (Liquidated Count)",
                "liquidated_symbols": "强平标的 (Liquidated Symbols)",
                "priority": "强平顺序 (Priority)",
            },
        )
        liquidation_audit_html = _format_table(
            liquidation_view,
            max_rows=20,
            compact_currency_columns={"当日利息 (Daily Interest)"},
            compact_currency=compact_currency,
        )
    else:
        liquidation_audit_html = "<div>暂无强平审计数据</div>"

    if overview_cards:
        analysis_overview_html = "".join(
            [
                (
                    '<div class="analysis-card">'
                    f'<div class="analysis-card-label">{label}</div>'
                    f'<div class="analysis-card-value">{value}</div>'
                    "</div>"
                )
                for label, value in overview_cards
            ]
        )
    else:
        analysis_overview_html = "<div>暂无归因与容量摘要数据</div>"

    return {
        "analysis_overview_html": analysis_overview_html,
        "exposure_summary_html": exposure_summary_html,
        "capacity_summary_html": capacity_summary_html,
        "attribution_summary_html": attribution_summary_html,
        "orders_by_strategy_html": orders_by_strategy_html,
        "executions_by_strategy_html": executions_by_strategy_html,
        "risk_by_strategy_html": risk_by_strategy_html,
        "liquidation_audit_html": liquidation_audit_html,
    }


def plot_report(
    result: "BacktestResult",
    title: str = "AKQuant 策略回测报告",
    filename: str = "akquant_report.html",
    show: bool = False,
    compact_currency: bool = True,
    market_data: Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]] = None,
    plot_symbol: Optional[str] = None,
    include_trade_kline: bool = True,
    include_indicators: bool = False,
    indicator_name: Optional[str] = None,
    indicator_symbol: Optional[str] = None,
    indicator_include_warmup: bool = True,
    benchmark: Optional[Union[str, pd.Series]] = None,
    curve_freq: str = "D",
) -> None:
    """
    生成类似 QuantStats 的整合版 HTML 报告 (中文优化版).

    内容包括:
    1. 核心指标概览 (Key Metrics)
    2. 权益曲线、回撤、月度热力图 (Dashboard)
    3. 交易分布与持仓时间分析 (Trade Analysis)

    :param compact_currency: 是否将金额列按 K/M/B 紧凑显示
    :param include_indicators: 是否在报告中包含自定义指标预览区块
    :param indicator_name: 可选指标键过滤，仅展示指定指标
    :param indicator_symbol: 可选标的过滤，仅展示指定标的指标
    :param indicator_include_warmup: 是否在指标报告区块中保留预热点
    :param benchmark: 基准收益序列 (pd.Series) 或基准标识字符串
    :param curve_freq: 曲线频率，默认 "D" 为日频末值，"raw" 为原始频率
    """
    if not check_plotly():
        return
    normalized_curve_freq = _normalize_curve_freq(curve_freq)

    # Prepare Icon
    icon_b64 = base64.b64encode(AKQUANT_ICON_SVG.encode("utf-8")).decode("utf-8")
    favicon_uri = f"data:image/svg+xml;base64,{icon_b64}"

    summary_context = _build_summary_context(result, curve_freq=normalized_curve_freq)
    metrics_html = _build_metrics_html(result)
    chart_sections = _build_chart_html_sections(
        result=result,
        market_data=market_data,
        plot_symbol=plot_symbol,
        include_trade_kline=include_trade_kline,
        include_indicators=include_indicators,
        indicator_name=indicator_name,
        indicator_symbol=indicator_symbol,
        indicator_include_warmup=indicator_include_warmup,
        benchmark=benchmark,
        curve_freq=normalized_curve_freq,
    )
    analysis_sections = _build_analysis_table_sections(
        result, compact_currency=compact_currency
    )

    # 4. Assemble HTML
    html_content = HTML_TEMPLATE.format(
        title=title,
        favicon_uri=favicon_uri,
        icon_svg=AKQUANT_LOGO_SVG,
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        start_date=summary_context["start_date"],
        end_date=summary_context["end_date"],
        duration_str=summary_context["duration_str"],
        initial_cash=summary_context["initial_cash"],
        final_equity=summary_context["final_equity"],
        metrics_html=metrics_html,
        dashboard_html=chart_sections["dashboard_html"],
        indicator_section_html=chart_sections["indicator_section_html"],
        yearly_returns_html=chart_sections["yearly_returns_html"],
        returns_dist_html=chart_sections["returns_dist_html"],
        rolling_metrics_html=chart_sections["rolling_metrics_html"],
        benchmark_metrics_html=chart_sections["benchmark_metrics_html"],
        benchmark_chart_html=chart_sections["benchmark_chart_html"],
        trades_dist_html=chart_sections["trades_dist_html"],
        pnl_duration_html=chart_sections["pnl_duration_html"],
        strategy_kline_html=chart_sections["strategy_kline_html"],
        risk_reject_ratio_html=chart_sections["risk_reject_ratio_html"],
        risk_reason_ratio_html=chart_sections["risk_reason_ratio_html"],
        risk_reject_trend_html=chart_sections["risk_reject_trend_html"],
        risk_reject_trend_by_strategy_html=chart_sections[
            "risk_reject_trend_by_strategy_html"
        ],
        risk_reason_trend_html=chart_sections["risk_reason_trend_html"],
        analysis_overview_html=analysis_sections["analysis_overview_html"],
        exposure_summary_html=analysis_sections["exposure_summary_html"],
        capacity_summary_html=analysis_sections["capacity_summary_html"],
        attribution_summary_html=analysis_sections["attribution_summary_html"],
        orders_by_strategy_html=analysis_sections["orders_by_strategy_html"],
        executions_by_strategy_html=analysis_sections["executions_by_strategy_html"],
        risk_by_strategy_html=analysis_sections["risk_by_strategy_html"],
        liquidation_audit_html=analysis_sections["liquidation_audit_html"],
        risk_charts_html=chart_sections["risk_charts_html"],
    )

    # 5. Save File
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Report saved to: {filename}")

        if show:
            import os
            import webbrowser

            webbrowser.open(f"file://{os.path.abspath(filename)}")

    except Exception as e:
        print(f"Error saving report: {e}")
