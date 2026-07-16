"""公共读自由函数经 strategy.execution 取值（可被 fake execution 拦截）."""

from akquant import strategy_trading_api as api


class _FakeExec:
    def get_position(self, symbol: str | None = None) -> float:
        return 42.0

    def get_cash(self) -> float:
        return 999.0


class _S:
    execution = _FakeExec()
    ctx = None


def test_get_position_delegates_to_execution() -> None:
    """get_position 应经 strategy.execution 取值，而非直读 ctx."""
    assert api.get_position(_S()) == 42.0


def test_get_cash_delegates_to_execution() -> None:
    """get_cash 应经 strategy.execution 取值，而非直读 ctx."""
    assert api.get_cash(_S()) == 999.0
