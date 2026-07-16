"""get_positions() → positions 属性（硬改）；execution 层保留."""

from akquant.execution.sim import SimExecution
from akquant.strategy import Strategy


def test_positions_property_and_method_removed() -> None:
    """get_positions() 方法应已移除，positions 应为只读属性."""
    assert not hasattr(Strategy, "get_positions")
    assert isinstance(getattr(Strategy, "positions"), property)


def test_execution_get_positions_unchanged() -> None:
    """ExecutionBackend 协议方法名不变."""
    assert hasattr(SimExecution, "get_positions")
