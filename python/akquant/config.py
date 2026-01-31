from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RiskConfig:
    """Configuration for Risk Management."""

    active: bool = True
    max_order_size: Optional[float] = None
    max_order_value: Optional[float] = None
    max_position_size: Optional[float] = None
    restricted_list: Optional[List[str]] = None


@dataclass
class StrategyConfig:
    """
    Global configuration for strategies and backtesting.

    Inspired by PyBroker's configuration system.
    """

    # Capital Management
    initial_cash: float = 100000.0

    # Fees & Commission
    fee_mode: str = "per_order"  # 'per_order', 'per_share', 'percent'
    fee_amount: float = 0.0  # Fixed amount or percentage

    # Execution
    enable_fractional_shares: bool = False
    round_fill_price: bool = True

    # Position Sizing Constraints
    max_long_positions: Optional[int] = None
    max_short_positions: Optional[int] = None

    # Bootstrap Metrics
    bootstrap_samples: int = 1000
    bootstrap_sample_size: Optional[int] = None

    # Other
    exit_on_last_bar: bool = True

    # Risk Config
    risk: Optional[RiskConfig] = None


@dataclass
class BacktestConfig:
    """Configuration specifically for running backtests."""

    strategy_config: StrategyConfig
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    instruments: Optional[List[str]] = None
    benchmark: Optional[str] = None
    timezone: str = "Asia/Shanghai"
    show_progress: bool = True
    history_depth: int = 0


# Global instance
strategy_config = StrategyConfig()
