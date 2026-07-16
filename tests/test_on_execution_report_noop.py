"""Strategy 基类应有 on_execution_report no-op（broker_live 需要，回测无害）."""

from akquant.strategy import Strategy


def test_on_execution_report_noop_exists() -> None:
    """Strategy 应提供 on_execution_report no-op，供 broker_live 安全调用."""
    s = Strategy.__new__(Strategy)
    assert hasattr(Strategy, "on_execution_report")
    s.on_execution_report(object())
