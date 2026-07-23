import importlib.util
import inspect
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest
from akquant.backtest._viz import VizNamespace
from akquant.config import RiskConfig
from akquant.plot import (
    plot_dashboard,
    plot_indicators,
    plot_pnl_vs_duration,
    plot_trades_distribution,
)
from akquant.plot.analysis import plot_returns_distribution, plot_rolling_metrics


class RoundTripStrategy(Strategy):
    """Generate one round-trip trade for report and plot checks."""

    def __init__(self) -> None:
        """Initialize internal step counter."""
        super().__init__()
        self.step = 0

    def on_bar(self, bar: Bar) -> None:
        """Buy once and sell once."""
        self.step += 1
        if self.step == 1:
            self.buy(symbol=bar.symbol, quantity=100, tag="entry")
        elif self.step == 3:
            self.sell(symbol=bar.symbol, quantity=100, tag="exit")


class NoTradeStrategy(Strategy):
    """Produce no trades for empty-trade report branch."""

    def on_bar(self, bar: Bar) -> None:
        """Do nothing on each bar."""
        _ = bar


class IndicatorPlotStrategy(Strategy):
    """Record a couple of indicators for lightweight plotting checks."""

    def on_bar(self, bar: Bar) -> None:
        """Emit one main-pane line and one sub-pane bar series."""
        self.record_indicator(
            name="close_echo",
            value=bar.close,
            display_name="Close Echo",
            pane=0,
            render_type="line",
        )
        self.record_indicator(
            name="distance_from_ten",
            value=bar.close - 10.0,
            display_name="Distance From Ten",
            pane=1,
            render_type="bar",
        )


class RenderTypeStrategy(Strategy):
    """Record indicators covering the area, scatter, and signal render types."""

    def on_bar(self, bar: Bar) -> None:
        """Emit one series per non-line render type on the main pane."""
        self.record_indicator(
            name="area_series", value=bar.close, pane=0, render_type="area"
        )
        self.record_indicator(
            name="scatter_series", value=bar.high, pane=0, render_type="scatter"
        )
        self.record_indicator(
            name="signal_series", value=bar.low, pane=0, render_type="signal"
        )


class MarginLiquidationStrategy(Strategy):
    """Open a leveraged long to trigger forced liquidation on drawdown."""

    def __init__(self) -> None:
        """Initialize one-shot order flag."""
        super().__init__()
        self.ordered = False

    def on_bar(self, bar: Bar) -> None:
        """Buy once with leverage-like sizing."""
        if not self.ordered:
            self.buy(symbol=bar.symbol, quantity=150)
            self.ordered = True


def _build_data(symbol: str = "TEST", n: int = 5) -> list[Bar]:
    """Build deterministic daily bars."""
    data: list[Bar] = []
    for i in range(n):
        close = 10.0 + i
        data.append(
            Bar(
                timestamp=pd.Timestamp(f"2023-01-{i + 1:02d} 10:00:00").value,
                open=close,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=10000.0,
                symbol=symbol,
            )
        )
    return data


def _build_intraday_data(symbol: str = "TEST") -> list[Bar]:
    data: list[Bar] = []
    timestamps = [
        pd.Timestamp("2023-01-01 10:00:00"),
        pd.Timestamp("2023-01-01 14:00:00"),
        pd.Timestamp("2023-01-02 10:00:00"),
    ]
    closes = [10.0, 10.5, 11.0]
    for ts, close in zip(timestamps, closes):
        data.append(
            Bar(
                timestamp=ts.value,
                open=close,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=10000.0,
                symbol=symbol,
            )
        )
    return data


def _build_cross_utc_boundary_same_local_day_data(symbol: str = "TEST") -> list[Bar]:
    """Build bars that span two UTC dates but remain in one Asia/Shanghai local day."""
    data: list[Bar] = []
    timestamps = [
        pd.Timestamp("2024-01-02 01:00:00", tz="Asia/Shanghai").tz_convert("UTC"),
        pd.Timestamp("2024-01-02 23:00:00", tz="Asia/Shanghai").tz_convert("UTC"),
    ]
    closes = [10.0, 10.5]
    for ts, close in zip(timestamps, closes):
        data.append(
            Bar(
                timestamp=ts.value,
                open=close,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=10000.0,
                symbol=symbol,
            )
        )
    return data


def _build_market_df(symbol: str = "TEST", n: int = 5) -> pd.DataFrame:
    rows = []
    for bar in _build_data(symbol=symbol, n=n):
        rows.append(
            {
                "timestamp": pd.Timestamp(bar.timestamp),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "symbol": bar.symbol,
            }
        )
    df = pd.DataFrame(rows)
    df = df.set_index("timestamp")
    return cast(pd.DataFrame, df)


def _build_market_df_uppercase(symbol: str = "TEST", n: int = 5) -> pd.DataFrame:
    rows = []
    for bar in _build_data(symbol=symbol, n=n):
        rows.append(
            {
                "Timestamp": pd.Timestamp(bar.timestamp),
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
                "Symbol": bar.symbol,
            }
        )
    return cast(pd.DataFrame, pd.DataFrame(rows))


def _build_market_df_aliases(symbol: str = "TEST", n: int = 5) -> pd.DataFrame:
    rows = []
    for bar in _build_data(symbol=symbol, n=n):
        rows.append(
            {
                "trade_date": pd.Timestamp(bar.timestamp),
                "open_price": bar.open,
                "high_price": bar.high,
                "low_price": bar.low,
                "close_price": bar.close,
                "vol": bar.volume,
                "code": bar.symbol,
            }
        )
    return cast(pd.DataFrame, pd.DataFrame(rows))


def _skip_if_no_plotly() -> None:
    """Skip tests if plotly is unavailable."""
    if importlib.util.find_spec("plotly") is None:
        pytest.skip("plotly is required for report/plot tests")


def test_report_contains_new_analysis_sections(tmp_path: Path) -> None:
    """Report HTML should include attribution and capacity sections."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=RoundTripStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    report_path = tmp_path / "report_with_analysis.html"
    result.viz.report(filename=str(report_path), show=False)
    html = report_path.read_text(encoding="utf-8")
    assert "组合归因与容量分析 (Attribution & Capacity)" in html
    assert "最新净暴露比 (Latest Net Exposure %)" in html
    assert "平均换手率 (Avg Turnover)" in html
    assert "策略风控拒单明细 (Risk Rejections by Strategy)" in html
    assert "暂无策略归属风控拒单聚合数据" in html
    assert "未提供行情数据，已跳过 K 线复盘图" in html
    assert "自定义指标 (Custom Indicators)" not in html


def test_report_includes_trade_kline_with_market_data(tmp_path: Path) -> None:
    """Report HTML should embed K-line trade replay when market data is passed."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=RoundTripStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    report_path = tmp_path / "report_with_trade_kline.html"
    result.viz.report(
        filename=str(report_path),
        show=False,
        market_data=_build_market_df(symbol="TEST"),
        plot_symbol="TEST",
    )
    html = report_path.read_text(encoding="utf-8")
    assert "交易复盘 (K线买卖点)" in html
    assert "Strategy Analysis: TEST" in html
    assert "entry" in html
    assert "exit" in html


def test_report_accepts_uppercase_market_data_columns(tmp_path: Path) -> None:
    """Report should accept DuckDB-like uppercase OHLCV column names."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=RoundTripStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    report_path = tmp_path / "report_with_uppercase_market_data.html"
    result.viz.report(
        filename=str(report_path),
        show=False,
        market_data=_build_market_df_uppercase(symbol="TEST"),
        plot_symbol="TEST",
    )
    html = report_path.read_text(encoding="utf-8")
    assert "Strategy Analysis: TEST" in html
    assert "行情数据不完整，无法绘制 K 线复盘图" not in html


def test_report_accepts_market_data_alias_columns(tmp_path: Path) -> None:
    """Report should accept alias columns such as trade_date/code/open_price."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=RoundTripStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    report_path = tmp_path / "report_with_alias_market_data.html"
    result.viz.report(
        filename=str(report_path),
        show=False,
        market_data=_build_market_df_aliases(symbol="TEST"),
        plot_symbol="TEST",
    )
    html = report_path.read_text(encoding="utf-8")
    assert "Strategy Analysis: TEST" in html
    assert "行情数据不完整，无法绘制 K 线复盘图" not in html


def test_report_includes_benchmark_comparison_sections(tmp_path: Path) -> None:
    """Report HTML should include benchmark metrics and cumulative comparison chart."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=RoundTripStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )
    benchmark_idx = pd.date_range("2023-01-01", periods=5, freq="D", tz="Asia/Shanghai")
    benchmark_returns = pd.Series(
        [0.0, 0.001, -0.0005, 0.0008, 0.0],
        index=benchmark_idx,
        name="CSI300",
    )
    report_path = tmp_path / "report_with_benchmark.html"
    result.viz.report(
        filename=str(report_path),
        show=False,
        benchmark=benchmark_returns,
    )
    html = report_path.read_text(encoding="utf-8")
    assert "基准对比 (Benchmark Comparison)" in html
    assert "累计超额收益 (Total Excess)" in html
    assert "信息比率 (Information Ratio)" in html
    assert "Cumulative Return Comparison" in html
    assert "CSI300" in html


def test_report_aligns_naive_benchmark_dates_without_notice(tmp_path: Path) -> None:
    """Benchmark comparison should work with naive daily date indexes."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=RoundTripStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )
    benchmark_returns = pd.Series(
        [0.0, 0.001, -0.0005, 0.0008, 0.0],
        index=pd.date_range("2023-01-01", periods=5, freq="D"),
        name="NAIVE_BENCH",
    )
    report_path = tmp_path / "report_with_naive_benchmark.html"
    result.viz.report(
        filename=str(report_path),
        show=False,
        benchmark=benchmark_returns,
    )
    html = report_path.read_text(encoding="utf-8")
    assert "NAIVE_BENCH" in html
    assert "策略与基准无重叠区间" not in html
    assert "基准序列 NAIVE_BENCH 索引必须为日期索引" not in html


def test_backtest_result_exposes_structured_benchmark_analysis() -> None:
    """BacktestResult should expose benchmark analysis for non-HTML consumers."""
    result = run_backtest(
        data=_build_data(),
        strategy=RoundTripStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )
    benchmark_returns = pd.Series(
        [0.0, 0.001, -0.0005, 0.0008, 0.0],
        index=pd.date_range("2023-01-01", periods=5, freq="D"),
        name="CSI300",
    )

    payload = result.benchmark_analysis(benchmark=benchmark_returns, curve_freq="D")

    assert payload["schema_version"] == "1.0"
    assert payload["available"] is True
    assert payload["benchmark"]["label"] == "CSI300"
    assert payload["summary"]["information_ratio"] is not None
    assert payload["meta"]["aligned_points"] > 0
    assert payload["series"][0]["date"] == "2023-01-01"


def test_export_benchmark_analysis_keeps_cjk_label_readable(tmp_path: Path) -> None:
    """Exported benchmark JSON keeps CJK labels readable, like export_indicators."""
    result = run_backtest(
        data=_build_data(),
        strategy=RoundTripStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )
    benchmark_returns = pd.Series(
        [0.0, 0.001, -0.0005, 0.0008, 0.0],
        index=pd.date_range("2023-01-01", periods=5, freq="D"),
        name="沪深300指数",
    )

    export_path = tmp_path / "benchmark.json"
    result.export_benchmark_analysis(
        str(export_path), benchmark=benchmark_returns, curve_freq="D"
    )
    raw_text = export_path.read_text(encoding="utf-8")
    assert "沪深300指数" in raw_text
    assert "\\u" not in raw_text


def test_report_shows_notice_for_range_index_benchmark(tmp_path: Path) -> None:
    """RangeIndex benchmark input should render a clear validation notice."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=NoTradeStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )
    benchmark_returns = pd.Series(
        [0.0, 0.001, -0.0005, 0.0008, 0.0], name="RANGE_BENCH"
    )
    report_path = tmp_path / "report_with_range_index_benchmark.html"
    result.viz.report(
        filename=str(report_path),
        show=False,
        benchmark=benchmark_returns,
    )
    html = report_path.read_text(encoding="utf-8")
    assert "基准对比 (Benchmark Comparison)" in html
    assert "RangeIndex" in html
    assert "日期索引" in html


def test_report_handles_string_benchmark_with_notice(tmp_path: Path) -> None:
    """Report should render benchmark section notice when benchmark is ticker string."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=NoTradeStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )
    report_path = tmp_path / "report_with_benchmark_string.html"
    result.viz.report(
        filename=str(report_path),
        show=False,
        benchmark="000300.SH",
    )
    html = report_path.read_text(encoding="utf-8")
    assert "基准对比 (Benchmark Comparison)" in html
    assert "未生成基准对比: 暂不支持自动拉取基准: 000300.SH" in html


def test_report_handles_empty_trade_analysis_blocks(tmp_path: Path) -> None:
    """Report should still render when there are no trades."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=NoTradeStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    report_path = tmp_path / "report_empty_trades.html"
    result.viz.report(filename=str(report_path), show=False)
    html = report_path.read_text(encoding="utf-8")
    assert "暂无归因数据" in html


def test_report_optionally_includes_indicator_panel(tmp_path: Path) -> None:
    """Report should include indicator section only when explicitly enabled."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=IndicatorPlotStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    report_path = tmp_path / "report_with_indicators.html"
    result.viz.report(
        filename=str(report_path),
        show=False,
        include_indicators=True,
        indicator_name="distance_from_ten",
    )
    html = report_path.read_text(encoding="utf-8")
    assert "自定义指标 (Custom Indicators)" in html
    assert "指标定义明细 (Indicator Definitions)" in html
    assert "过滤条件: 指标=distance_from_ten" in html
    assert "Distance From Ten" in html
    assert "Close Echo" not in html
    assert "js-plotly-plot" in html


def test_report_indicator_panel_handles_empty_indicator_outputs(tmp_path: Path) -> None:
    """Indicator report section should render an empty-state panel for legacy runs."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=NoTradeStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    report_path = tmp_path / "report_indicator_panel_empty.html"
    result.viz.report(filename=str(report_path), show=False, include_indicators=True)
    html = report_path.read_text(encoding="utf-8")
    assert "自定义指标 (Custom Indicators)" in html
    assert "暂无指标数据" in html
    assert "暂无指标定义数据" in html


def test_plot_functions_return_figures_for_non_empty_result() -> None:
    """Core plot functions should return figures for non-empty inputs."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=RoundTripStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    fig_dashboard = plot_dashboard(result, show=False)
    assert fig_dashboard is not None
    fig_trades = plot_trades_distribution(result.trades_df)
    assert fig_trades is not None
    fig_duration = plot_pnl_vs_duration(result.trades_df)
    assert fig_duration is not None
    fig_rolling = plot_rolling_metrics(result.daily_returns)
    assert fig_rolling is not None
    fig_returns = plot_returns_distribution(result.daily_returns)
    assert fig_returns is not None


def test_indicator_plot_functions_return_multi_pane_figures() -> None:
    """Indicator plot helpers should return a lightweight multi-pane figure."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=IndicatorPlotStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    fig = plot_indicators(result, show=False)
    assert fig is not None
    assert len(fig.data) == 2
    assert fig.layout.title.text == "Indicator History"
    assert {trace.name for trace in fig.data} == {
        "Close Echo",
        "Distance From Ten",
    }
    assert [annotation.text for annotation in fig.layout.annotations] == [
        "Main",
        "Sub 1",
    ]

    filtered_fig = result.viz.indicators(name="distance_from_ten", show=False)
    assert filtered_fig is not None
    assert len(filtered_fig.data) == 1
    assert filtered_fig.data[0].name == "Distance From Ten"
    assert filtered_fig.data[0].type == "bar"


def test_indicator_plot_renders_each_render_type() -> None:
    """area/scatter/signal render types map to distinct, non-line traces."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=RenderTypeStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    fig = plot_indicators(result, show=False)
    assert fig is not None
    traces = {trace.name: trace for trace in fig.data}
    assert set(traces) == {"area_series", "scatter_series", "signal_series"}
    # area is a filled line
    assert traces["area_series"].mode == "lines"
    assert traces["area_series"].fill == "tozeroy"
    # scatter and signal are marker series
    assert traces["scatter_series"].mode == "markers"
    assert traces["signal_series"].mode == "markers"
    assert traces["signal_series"].marker.symbol == "triangle-up"


def test_indicator_plot_returns_none_without_indicator_data() -> None:
    """Indicator plotting should stay lightweight for legacy empty outputs."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_data(),
        strategy=NoTradeStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    assert result.viz.indicators(show=False) is None


def test_daily_curve_properties_reduce_intraday_points() -> None:
    """Daily curve properties should downsample intraday points to day-end."""
    result = run_backtest(
        data=_build_intraday_data(),
        strategy=NoTradeStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    assert len(result.equity_curve) == 3
    assert len(result.cash_curve) == 3
    assert len(result.margin_curve) == 3
    assert len(result.equity_curve_daily) == 2
    assert len(result.cash_curve_daily) == 2
    assert len(result.margin_curve_daily) == 2


def test_daily_curve_properties_follow_local_timezone_day() -> None:
    """Daily curve aggregation should follow the configured local day, not UTC day."""
    result = run_backtest(
        data=_build_cross_utc_boundary_same_local_day_data(),
        strategy=NoTradeStrategy,
        symbols="TEST",
        timezone="Asia/Shanghai",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )

    assert len(result.equity_curve) == 2
    assert len(result.equity_curve_daily) == 1
    assert len(result.daily_returns) == 1
    assert result.equity_curve_daily.index[0] == pd.Timestamp(
        "2024-01-02 00:00:00", tz="Asia/Shanghai"
    )
    assert result.daily_returns.index[0] == pd.Timestamp(
        "2024-01-02 00:00:00", tz="Asia/Shanghai"
    )


def test_report_accepts_curve_freq_daily(tmp_path: Path) -> None:
    """Report generation should support daily curve frequency mode."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_intraday_data(),
        strategy=NoTradeStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )
    report_path = tmp_path / "report_curve_freq_daily.html"
    result.viz.report(filename=str(report_path), show=False, curve_freq="D")
    assert report_path.exists()


def test_report_defaults_curve_freq_to_daily() -> None:
    """Report API should default to daily curve frequency for lighter HTML."""
    default_value = inspect.signature(VizNamespace.report)
    assert default_value.parameters["curve_freq"].default == "D"


def test_report_rejects_invalid_curve_freq(tmp_path: Path) -> None:
    """Report generation should reject unsupported curve frequency values."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_intraday_data(),
        strategy=NoTradeStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )
    report_path = tmp_path / "report_curve_freq_invalid.html"
    with pytest.raises(ValueError):
        result.viz.report(filename=str(report_path), show=False, curve_freq="W")


def test_report_displays_normalized_reject_reason_types(tmp_path: Path) -> None:
    """Report HTML should show reject types and sample details separately."""
    _skip_if_no_plotly()
    result = run_backtest(
        data=_build_intraday_data(),
        strategy=NoTradeStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
    )
    result.orders_df = pd.DataFrame(
        {
            "reject_reason": [
                (
                    "Risk: Insufficient margin at execution. "
                    "Required: 35, Available: -330445.517600"
                ),
                (
                    "Risk: Insufficient margin at execution. "
                    "Required: 35, Available: -310706.591600"
                ),
            ],
            "status": ["Rejected", "Rejected"],
        }
    )
    report_path = tmp_path / "report_reject_reason_types.html"
    result.viz.report(filename=str(report_path), show=False)
    html = report_path.read_text(encoding="utf-8")
    assert "拒单类型 (Reject Type)" in html
    assert "示例详情 (Sample Detail)" in html
    assert "Insufficient Margin" in html


def test_report_contains_forced_liquidation_audit_section(tmp_path: Path) -> None:
    """Report HTML should include forced liquidation audit section and entries."""
    _skip_if_no_plotly()
    bars = [
        Bar(
            timestamp=pd.Timestamp("2023-01-01 10:00:00").value,
            open=100.0,
            high=100.2,
            low=99.8,
            close=100.0,
            volume=10000.0,
            symbol="LIQ",
        ),
        Bar(
            timestamp=pd.Timestamp("2023-01-01 14:00:00").value,
            open=20.0,
            high=20.2,
            low=19.8,
            close=20.0,
            volume=10000.0,
            symbol="LIQ",
        ),
        Bar(
            timestamp=pd.Timestamp("2023-01-02 10:00:00").value,
            open=20.0,
            high=20.2,
            low=19.8,
            close=20.0,
            volume=10000.0,
            symbol="LIQ",
        ),
    ]
    result = run_backtest(
        data=bars,
        strategy=MarginLiquidationStrategy,
        symbols="LIQ",
        initial_cash=10000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        lot_size=1,
        show_progress=False,
        risk_config=RiskConfig(
            account_mode="margin",
            initial_margin_ratio=0.5,
            maintenance_margin_ratio=0.5,
            financing_rate_annual=0.0,
            borrow_rate_annual=0.0,
            allow_force_liquidation=True,
            liquidation_priority="short_first",
        ),
    )
    report_path = tmp_path / "report_with_liquidation_audit.html"
    result.viz.report(filename=str(report_path), show=False)
    html = report_path.read_text(encoding="utf-8")
    assert "强平审计明细 (Forced Liquidation Audit)" in html
    assert "强平标的 (Liquidated Symbols)" in html
    assert "LIQ" in html
