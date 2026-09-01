import pandas as pd
from akquant.analysis.benchmark import (
    benchmark_analysis_from_result,
    build_benchmark_analysis,
    normalize_returns_series,
    resolve_benchmark_returns,
)
from akquant.plot.report import (
    _build_analysis_table_sections,
    _build_metrics_html,
    _build_summary_context,
)
from akquant.utils import format_metric_value


class DummyMetrics:
    """Minimal metrics object for report helper tests."""

    total_return_pct = 10.0
    annualized_return = 0.021184
    sharpe_ratio = 1.2
    max_drawdown_pct = 3.0
    volatility = 0.2
    win_rate = 55.0


class DummyResult:
    """Minimal backtest result-like object for helper function tests."""

    def __init__(self, with_data: bool) -> None:
        """Initialize fixture with or without analysis data."""
        self.initial_cash = 100000.0
        idx = pd.to_datetime(
            ["2023-01-01 10:00:00", "2023-01-02 10:00:00"], utc=True
        ).tz_convert("Asia/Shanghai")
        self.equity_curve = (
            pd.Series([100000.0, 101000.0], index=idx)
            if with_data
            else pd.Series(dtype=float)
        )
        self.metrics = DummyMetrics()
        self._trades_df = pd.DataFrame({"pnl": [1.0], "symbol": ["TEST"]})

    @property
    def trades_df(self) -> pd.DataFrame:
        """Return synthetic trades dataframe."""
        return self._trades_df

    @property
    def metrics_df(self) -> pd.DataFrame:
        """Return synthetic metrics dataframe fallback."""
        return pd.DataFrame({"value": [1.5, 1.0]}, index=["profit_factor", "avg_pnl"])

    def exposure_df(self) -> pd.DataFrame:
        """Return synthetic exposure decomposition data."""
        return pd.DataFrame(
            {
                "net_exposure_pct": [0.1],
                "gross_exposure_pct": [0.2],
                "leverage": [0.3],
            }
        )

    def capacity_df(self) -> pd.DataFrame:
        """Return synthetic capacity data."""
        return pd.DataFrame(
            {
                "order_count": [2.0],
                "filled_value": [1000.0],
                "fill_rate_qty": [1.0],
                "turnover": [0.01],
            }
        )

    def attribution_df(self, by: str = "symbol") -> pd.DataFrame:
        """Return synthetic attribution data."""
        _ = by
        return pd.DataFrame(
            {
                "group": ["TEST"],
                "trade_count": [1],
                "total_pnl": [100.0],
                "contribution_pct": [1.0],
                "total_commission": [0.0],
            }
        )

    def orders_by_strategy(self) -> pd.DataFrame:
        """Return synthetic strategy-level order summary."""
        return pd.DataFrame(
            {
                "owner_strategy_id": ["alpha"],
                "order_count": [2],
                "filled_order_count": [2],
                "filled_quantity": [100.0],
                "filled_value": [1000.0],
                "fill_rate_qty": [1.0],
            }
        )

    def executions_by_strategy(self) -> pd.DataFrame:
        """Return synthetic strategy-level execution summary."""
        return pd.DataFrame(
            {
                "owner_strategy_id": ["alpha"],
                "execution_count": [2],
                "total_quantity": [100.0],
                "total_notional": [1000.0],
                "total_commission": [0.0],
                "avg_fill_price": [10.0],
            }
        )

    def risk_rejections_by_strategy(self) -> pd.DataFrame:
        """Return synthetic strategy-level risk rejection summary."""
        return pd.DataFrame(
            {
                "owner_strategy_id": ["alpha"],
                "risk_reject_count": [2],
                "daily_loss_reject_count": [1],
                "drawdown_reject_count": [1],
                "reduce_only_reject_count": [0],
                "strategy_risk_budget_reject_count": [1],
                "portfolio_risk_budget_reject_count": [0],
                "other_risk_reject_count": [0],
            }
        )


def test_build_summary_context_with_equity_data() -> None:
    """Summary context should render dates and final equity when equity exists."""
    context = _build_summary_context(DummyResult(with_data=True))
    assert context["start_date"] == "2023-01-01"
    assert context["end_date"] == "2023-01-02"
    assert context["final_equity"] == "101,000.00"


def test_build_analysis_table_sections_with_and_without_data() -> None:
    """Analysis section builder should render data and empty fallbacks."""
    result = DummyResult(with_data=True)
    sections = _build_analysis_table_sections(result)
    assert "最新净暴露比 (Latest Net Exposure %)" in sections["exposure_summary_html"]
    assert "平均换手率 (Avg Turnover)" in sections["capacity_summary_html"]
    assert "分组 (Group)" in sections["attribution_summary_html"]
    assert "TEST" in sections["attribution_summary_html"]
    assert "策略ID (Strategy ID)" in sections["orders_by_strategy_html"]
    assert "策略ID (Strategy ID)" in sections["executions_by_strategy_html"]
    assert "风险拒单总数 (Risk Rejects)" in sections["risk_by_strategy_html"]
    assert (
        "策略预算拒单数 (Strategy Budget Rejects)" in sections["risk_by_strategy_html"]
    )
    assert "10.000000%" in sections["exposure_summary_html"]
    assert "1.000000%" in sections["capacity_summary_html"]
    assert "100.000000%" in sections["attribution_summary_html"]
    assert "1.00K" in sections["capacity_summary_html"]
    assert "1.00K" in sections["orders_by_strategy_html"]
    assert "1.00K" in sections["executions_by_strategy_html"]

    sections_raw = _build_analysis_table_sections(result, compact_currency=False)
    assert "1,000.000000" in sections_raw["capacity_summary_html"]

    class EmptyResult(DummyResult):
        """Override to simulate empty analysis outputs."""

        def exposure_df(self) -> pd.DataFrame:
            return pd.DataFrame()

        def capacity_df(self) -> pd.DataFrame:
            return pd.DataFrame()

        def attribution_df(self, by: str = "symbol") -> pd.DataFrame:
            _ = by
            return pd.DataFrame()

        def orders_by_strategy(self) -> pd.DataFrame:
            return pd.DataFrame()

        def executions_by_strategy(self) -> pd.DataFrame:
            return pd.DataFrame()

        def risk_rejections_by_strategy(self) -> pd.DataFrame:
            return pd.DataFrame()

    empty_sections = _build_analysis_table_sections(EmptyResult(with_data=False))
    empty_orders_by_strategy_html = "<div>暂无策略归属订单聚合数据</div>"
    empty_executions_by_strategy_html = "<div>暂无策略归属成交聚合数据</div>"
    empty_risk_by_strategy_html = "<div>暂无策略归属风控拒单聚合数据</div>"
    assert empty_sections["exposure_summary_html"] == "<div>暂无暴露数据</div>"
    assert empty_sections["capacity_summary_html"] == "<div>暂无容量数据</div>"
    assert empty_sections["attribution_summary_html"] == "<div>暂无归因数据</div>"
    assert empty_sections["orders_by_strategy_html"] == empty_orders_by_strategy_html
    assert (
        empty_sections["executions_by_strategy_html"]
        == empty_executions_by_strategy_html
    )
    assert empty_sections["risk_by_strategy_html"] == empty_risk_by_strategy_html


def test_build_metrics_html_contains_key_labels() -> None:
    """Metrics HTML should contain expected labels and formatted values."""
    html = _build_metrics_html(DummyResult(with_data=True))
    assert "累计收益率 (Total Return)" in html
    assert "年化收益率 (CAGR)" in html
    assert "2.12%" in html
    assert "已完成交易数 (Closed Trades)" in html
    assert "成交笔数 (Executions)" in html
    assert "未平仓标的数 (Open Positions)" in html


def test_format_metric_value_uses_metric_unit_mapping() -> None:
    """Metric formatter should distinguish ratio and pct-value metrics."""
    assert format_metric_value("annualized_return", 0.021184) == "2.12%"
    assert format_metric_value("max_drawdown_pct", 1.07) == "1.07%"
    assert format_metric_value("calmar_ratio", 22.473305) == "22.47"


def test_normalize_returns_series_preserves_local_calendar_day() -> None:
    """Timezone-aware daily indexes should keep their local calendar day."""
    idx = pd.date_range("2023-01-01", periods=3, freq="D", tz="Asia/Shanghai")
    series = pd.Series([0.0, 0.01, -0.02], index=idx)

    normalized = normalize_returns_series(series)

    expected_idx = pd.date_range("2023-01-01", periods=3, freq="D")
    pd.testing.assert_index_equal(normalized.index, expected_idx)


def test_resolve_benchmark_returns_rejects_range_index() -> None:
    """Benchmark series must use date-like indexes instead of a RangeIndex."""
    strategy_idx = pd.date_range("2023-01-01", periods=3, freq="D", tz="Asia/Shanghai")
    strategy_returns = pd.Series([0.0, 0.01, -0.02], index=strategy_idx)
    benchmark_returns = pd.Series([0.0, 0.005, -0.001], name="RANGE_BENCH")

    resolved, reason = resolve_benchmark_returns(benchmark_returns, strategy_returns)

    assert resolved is None
    assert "RangeIndex" in reason
    assert "日期索引" in reason


def test_build_benchmark_analysis_returns_structured_payload() -> None:
    """Structured benchmark analysis should expose summary and aligned series."""
    strategy_idx = pd.date_range("2023-01-01", periods=3, freq="D", tz="Asia/Shanghai")
    strategy_returns = pd.Series([0.0, 0.01, -0.02], index=strategy_idx)
    benchmark_returns = pd.Series(
        [0.0, 0.005, -0.01],
        index=pd.date_range("2023-01-01", periods=3, freq="D"),
        name="CSI300",
    )

    payload = build_benchmark_analysis(strategy_returns, benchmark_returns)

    assert payload["schema_version"] == "1.0"
    assert payload["available"] is True
    assert payload["benchmark"]["label"] == "CSI300"
    assert payload["meta"]["aligned_points"] == 3
    assert payload["meta"]["start_date"] == "2023-01-01"
    assert payload["meta"]["end_date"] == "2023-01-03"
    assert payload["summary"]["tracking_error"] is not None
    assert payload["series"][0]["date"] == "2023-01-01"
    assert payload["series"][-1]["excess_cum_return"] is not None


def test_benchmark_analysis_from_result_uses_equity_curve() -> None:
    """Benchmark analysis from result should derive returns from the equity curve."""
    result = DummyResult(with_data=True)
    benchmark_returns = pd.Series(
        [0.0, 0.005],
        index=pd.date_range("2023-01-01", periods=2, freq="D"),
        name="CSI300",
    )

    payload = benchmark_analysis_from_result(result, benchmark_returns)

    assert payload["available"] is True
    assert payload["benchmark"]["label"] == "CSI300"
    # 权益曲线共 2 日, 首日不产生收益点 (口径对齐 Rust), 故仅 01-02 一点可与基准对齐
    assert payload["meta"]["aligned_points"] == 1
