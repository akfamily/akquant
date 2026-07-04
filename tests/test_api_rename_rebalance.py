"""order_target_positions/weights → rebalance_positions/weights（硬改）."""

from akquant.strategy import Strategy


def test_rebalance_renamed() -> None:
    """旧名应已移除，新名 rebalance_positions/weights 应存在."""
    for old in ("order_target_positions", "order_target_weights"):
        assert not hasattr(Strategy, old)
    for new in ("rebalance_positions", "rebalance_weights"):
        assert hasattr(Strategy, new)
