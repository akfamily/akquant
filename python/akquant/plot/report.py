"""Consolidated report generation module."""

import datetime
from typing import TYPE_CHECKING

from .analysis import (
    plot_pnl_vs_duration,
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_trades_distribution,
    plot_yearly_returns,
)
from .dashboard import plot_dashboard
from .utils import check_plotly

if TYPE_CHECKING:
    from ..backtest import BacktestResult

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
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
            overflow: hidden; /* Critical for Plotly resizing */
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
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"


def plot_report(
    result: "BacktestResult",
    title: str = "AKQuant 策略回测报告",
    filename: str = "akquant_report.html",
    show: bool = False,
) -> None:
    """
    生成类似 QuantStats 的整合版 HTML 报告 (中文优化版).

    内容包括:
    1. 核心指标概览 (Key Metrics)
    2. 权益曲线、回撤、月度热力图 (Dashboard)
    3. 交易分布与持仓时间分析 (Trade Analysis)
    """
    if not check_plotly():
        return

    # 1. Prepare Summary Data
    equity_curve = result.equity_curve
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

        duration = end_ts - start_ts
        duration_days = duration.days
        duration_str = f"{duration_days} 天"

        final_equity = equity_curve.iloc[-1]
        final_equity_str = f"{final_equity:,.2f}"

    # 2. Generate Metrics HTML
    metrics = result.metrics

    # Helper for coloring
    def get_color_class(val: float) -> str:
        if val > 0:
            return "positive"
        if val < 0:
            return "negative"
        return ""

    # Define Metrics to display
    # (Label, Value, Formatted Value, Color Class)
    # Using getattr for robust access
    def get_metric(name: str, default: float = 0.0) -> float:
        if hasattr(metrics, name):
            val = getattr(metrics, name)
            # Ensure it's a float
            try:
                return float(val)  # type: ignore
            except (ValueError, TypeError):
                return default
        # Try finding in metrics_df if not in object properties
        try:
            return float(result.metrics_df.loc[name, "value"])  # type: ignore
        except Exception:
            return default

    metric_data = [
        # Returns
        (
            "累计收益率 (Total Return)",
            metrics.total_return_pct,
            f"{metrics.total_return_pct:.2f}%",  # Fixed: removed double percentage
            get_color_class(metrics.total_return_pct),
        ),
        (
            "年化收益率 (CAGR)",
            metrics.annualized_return,
            f"{metrics.annualized_return:.2f}%",
            get_color_class(metrics.annualized_return),
        ),
        (
            "平均盈亏 (Avg PnL)",
            get_metric("avg_pnl"),
            f"{get_metric('avg_pnl'):.2f}",
            get_color_class(get_metric("avg_pnl")),
        ),
        # Risk
        (
            "夏普比率 (Sharpe)",
            metrics.sharpe_ratio,
            f"{metrics.sharpe_ratio:.2f}",
            get_color_class(metrics.sharpe_ratio),
        ),
        (
            "索提诺比率 (Sortino)",
            get_metric("sortino_ratio"),
            f"{get_metric('sortino_ratio'):.2f}",
            get_color_class(get_metric("sortino_ratio")),
        ),
        (
            "卡玛比率 (Calmar)",
            get_metric("calmar_ratio"),
            f"{get_metric('calmar_ratio'):.2f}",
            get_color_class(get_metric("calmar_ratio")),
        ),
        (
            "最大回撤 (Max DD)",
            metrics.max_drawdown_pct,
            f"{metrics.max_drawdown_pct:.2f}%",  # max_drawdown_pct is scaled (0-100+)
            "negative",
        ),
        (
            "波动率 (Volatility)",
            metrics.volatility,
            f"{metrics.volatility:.2%}",
            "",
        ),  # volatility is ratio (0-1)
        # Trading
        (
            "胜率 (Win Rate)",
            metrics.win_rate,
            f"{metrics.win_rate:.2f}%",
            "",
        ),  # win_rate is scaled (0-100)
        (
            "盈亏比 (Profit Factor)",
            get_metric("profit_factor"),
            f"{get_metric('profit_factor'):.2f}",
            "",
        ),
        (
            "凯利公式 (Kelly)",
            get_metric("kelly_criterion"),
            f"{get_metric('kelly_criterion'):.2%}",  # Kelly is typically a ratio (0-1)
            "",
        ),
        ("交易次数 (Trades)", len(result.trades_df), f"{len(result.trades_df)}", ""),
    ]

    metrics_html = ""
    for label, raw_val, fmt_val, color_cls in metric_data:
        metrics_html += f"""
        <div class="metric-card">
            <div class="metric-value {color_cls}">{fmt_val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """

    # 3. Generate Plots
    # Dashboard (Equity, Drawdown, Heatmap)
    # Add responsive config
    config = {"responsive": True}
    fig_dashboard = plot_dashboard(result, show=False, theme="light")
    dashboard_html = (
        fig_dashboard.to_html(full_html=False, include_plotlyjs=False, config=config)
        if fig_dashboard
        else "<div>暂无数据</div>"
    )

    # Return Analysis
    returns_series = result.daily_returns

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

    # Trade Analysis
    # Add responsive config to ensure charts resize with container
    config = {"responsive": True}

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

    # 4. Assemble HTML
    html_content = HTML_TEMPLATE.format(
        title=title,
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        start_date=start_date,
        end_date=end_date,
        duration_str=duration_str,
        initial_cash=initial_cash_str,
        final_equity=final_equity_str,
        metrics_html=metrics_html,
        dashboard_html=dashboard_html,
        yearly_returns_html=yearly_returns_html,
        returns_dist_html=returns_dist_html,
        rolling_metrics_html=rolling_metrics_html,
        trades_dist_html=trades_dist_html,
        pnl_duration_html=pnl_duration_html,
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
