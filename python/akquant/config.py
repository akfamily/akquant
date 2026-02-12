from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class InstrumentConfig:
    """
    Configuration for a specific instrument.

    Allows defining asset type, multiplier, margin, etc. per symbol.
    """

    symbol: str
    asset_type: str = "STOCK"  # STOCK, FUND, FUTURES, OPTION
    multiplier: float = 1.0
    margin_ratio: float = 1.0
    tick_size: float = 0.01
    lot_size: int = 1
    # Option specific
    option_type: Optional[str] = None  # CALL, PUT
    strike_price: Optional[float] = None
    expiry_date: Optional[str] = None


@dataclass
class RiskConfig:
    """Configuration for Risk Management."""

    active: bool = True
    safety_margin: float = 0.0001
    max_order_size: Optional[float] = None
    max_order_value: Optional[float] = None
    max_position_size: Optional[float] = None
    restricted_list: Optional[List[str]] = None


@dataclass
class StrategyConfig:
    """Configuration for strategy execution."""

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
    instruments_config: Optional[
        Union[List[InstrumentConfig], Dict[str, InstrumentConfig]]
    ] = None
    benchmark: Optional[str] = None
    timezone: str = "Asia/Shanghai"
    show_progress: bool = True
    history_depth: int = 0


# Global instance
strategy_config = StrategyConfig()
