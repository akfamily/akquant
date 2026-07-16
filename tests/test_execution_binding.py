"""策略构造即急切绑定 SimExecution，capabilities 经 execution."""

from akquant.execution.sim import SimExecution
from akquant.strategy import Strategy


def test_new_strategy_has_sim_execution_bound() -> None:
    """Strategy.__new__ 应急切绑定 execution 为 SimExecution 实例."""
    s = Strategy.__new__(Strategy)
    assert isinstance(s.execution, SimExecution)


def test_capabilities_from_execution_when_no_injection() -> None:
    """未注入 get_execution_capabilities 时应回退到 execution.capabilities()."""
    from akquant.strategy_trading_api import get_execution_capabilities

    s = Strategy.__new__(Strategy)
    assert get_execution_capabilities(s).get("broker_live") is False
