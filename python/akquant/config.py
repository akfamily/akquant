from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, cast

"""
AKQuant Configuration System.

This module defines the configuration hierarchy for backtests.
It provides a structured way to define simulation parameters, asset properties,
execution rules, and risk constraints.

**Configuration Hierarchy:**

1.  **BacktestConfig** (Top Level): Defines the **Simulation Scenario**.
    *   **"When?"**: `start_time`, `end_time`
    *   **"What assets?"**: `instruments`, `instruments_config`
    *   **"How?"**: `strategy_config` (Strategy & Account settings)
    *   **"Analysis?"**: `bootstrap_samples`, `analysis_config`

2.  **InstrumentConfig** (Asset Level): Defines **Asset Properties**.
    *   **"What is it?"**: `symbol`, `asset_type`, `multiplier`
    *   **"How much leverage?"**: `margin_ratio`
    *   **"What costs?"**: `commission_rate` (overrides global), `slippage`

3.  **StrategyConfig** (Account Level): Defines **Account & Execution**.
    *   **"How much money?"**: `initial_cash`
    *   **"How to execute?"**: `slippage`, `volume_limit_pct`
    *   **"What constraints?"**: `max_long_positions`, `risk`

4.  **RiskConfig** (Risk Level): Defines **Safety Constraints**.
    *   **"How large?"**: `max_position_pct`, `max_order_size`
    *   **"When to stop?"**: `max_account_drawdown`, `stop_loss_threshold`
    *   **"What is forbidden?"**: `restricted_list`

**Usage Example:**

```python
# 1. Define Risk Rules
risk = RiskConfig(max_position_pct=0.1, stop_loss_threshold=0.8)

# 2. Configure Strategy & Account
strategy_conf = StrategyConfig(
    initial_cash=1_000_000,
    commission_rate=0.0003,
    slippage=0.0002,  # 2 bps
    risk=risk
)

# 3. Configure Specific Instruments (Optional)
rb_conf = InstrumentConfig(symbol="RB", asset_type="FUTURES", multiplier=10)

# 4. Create Backtest Configuration
config = BacktestConfig(
    start_time="2023-01-01",
    end_time="2023-12-31",
    strategy_config=strategy_conf,
    instruments=["AAPL", "GOOG"],
    instruments_config={"RB": rb_conf}
)
```
"""


InstrumentSettlementType = Literal["cash", "settlement_price", "force_close"]
InstrumentAssetType = Literal["STOCK", "FUTURES", "FUND", "OPTION"]
InstrumentOptionType = Literal["CALL", "PUT"]
InstrumentOptionMarginModel = Literal[
    "RATIO",
    "CHINA_SINGLE_LEG",
    "US_BROKER_SINGLE_LEG",
    "US_BROKER_SINGLE_LEG_VOL_ADJUSTED",
]


class InstrumentAssetTypeEnum(str, Enum):
    """Instrument asset type enum."""

    STOCK = "STOCK"
    FUTURES = "FUTURES"
    FUND = "FUND"
    OPTION = "OPTION"


class InstrumentOptionTypeEnum(str, Enum):
    """Instrument option type enum."""

    CALL = "CALL"
    PUT = "PUT"


class InstrumentSettlementTypeEnum(str, Enum):
    """Instrument settlement type enum."""

    CASH = "cash"
    SETTLEMENT_PRICE = "settlement_price"
    FORCE_CLOSE = "force_close"


class InstrumentOptionMarginModelEnum(str, Enum):
    """Instrument option margin model enum."""

    RATIO = "RATIO"
    CHINA_SINGLE_LEG = "CHINA_SINGLE_LEG"
    US_BROKER_SINGLE_LEG = "US_BROKER_SINGLE_LEG"
    US_BROKER_SINGLE_LEG_VOL_ADJUSTED = "US_BROKER_SINGLE_LEG_VOL_ADJUSTED"


InstrumentAssetTypeInput = Union[InstrumentAssetType, InstrumentAssetTypeEnum]
InstrumentOptionTypeInput = Union[InstrumentOptionType, InstrumentOptionTypeEnum]
InstrumentOptionMarginModelInput = Union[
    InstrumentOptionMarginModel, InstrumentOptionMarginModelEnum
]
InstrumentSettlementTypeInput = Union[
    InstrumentSettlementType, InstrumentSettlementTypeEnum
]


@dataclass
class InstrumentConfig:
    """
    [Asset Level] Configuration for a specific instrument.

    Defines **Asset Properties**.

    *   **"What is it?"**: `symbol`, `asset_type`, `multiplier`
    *   **"How much leverage?"**: `margin_ratio` / `option_margin_model`
    *   **"What costs?"**: `commission_rate` (overrides global), `slippage`

    **Core Properties:**
    :param symbol: Instrument symbol (e.g., "AAPL", "RB2305").
    :param asset_type: Asset type ("STOCK", "FUTURES", "FUND", "OPTION").
                       Default "STOCK".
    :param multiplier: Contract multiplier. Default 1.0.
    :param margin_ratio: Margin ratio (e.g., 0.1 for 10% margin).
                         主要用于期货/线性资产；期权仅在 `RATIO` 模式下使用。
    :param tick_size: Minimum price movement. Default 0.01.
    :param lot_size: Minimum trading unit (round lot). Default None.

    **Cost & Execution Overrides:**
    These fields override the global settings in `StrategyConfig` for
    this specific asset.
    :param commission_rate: Commission rate (e.g., 0.0003).
    :param min_commission: Minimum commission per order.
    :param stamp_tax_rate: Stamp tax rate (sell side only).
    :param transfer_fee_rate: Transfer fee rate.
    :param slippage: Asset-specific slippage policy. 推荐显式写为
                     {"type": "percent", "value": 0.0002}、
                     {"type": "fixed", "value": 0.2} 或
                     {"type": "ticks", "value": 1}。

    **Option Specific:**
    :param option_type: "CALL" or "PUT".
    :param strike_price: Strike price.
    :param expiry_date: Expiry date. Supports int (YYYYMMDD), date/datetime.
    :param underlying_symbol: Underlying asset symbol.
    :param option_margin_model: 期权保证金模型，默认 "CHINA_SINGLE_LEG"。
    :param implied_volatility: 当前隐含波动率，用于波动率调整模式。
    :param reference_volatility: 参考波动率，用于波动率调整模式。
    :param settlement_type: Settlement mode for expiry handling.
    :param settlement_price: Settlement price for expiry settlement mode.
    """

    symbol: str
    asset_type: InstrumentAssetTypeInput = InstrumentAssetTypeEnum.STOCK
    multiplier: float = 1.0
    margin_ratio: float = 1.0
    tick_size: float = 0.01
    lot_size: Optional[int] = None

    # Costs & Execution (Asset Specific)
    commission_rate: Optional[float] = None
    min_commission: Optional[float] = None
    stamp_tax_rate: Optional[float] = None
    transfer_fee_rate: Optional[float] = None
    slippage: Optional[Union[float, Dict[str, Any]]] = None

    # Option specific
    option_type: Optional[InstrumentOptionTypeInput] = None
    strike_price: Optional[float] = None
    expiry_date: Optional[Union[int, date, datetime]] = None
    underlying_symbol: Optional[str] = None
    option_margin_model: Optional[InstrumentOptionMarginModelInput] = None
    implied_volatility: Optional[float] = None
    reference_volatility: Optional[float] = None
    settlement_type: Optional[InstrumentSettlementTypeInput] = None
    settlement_price: Optional[float] = None
    sellable_after_days: Optional[int] = None
    static_attrs: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize instrument config."""
        asset_raw = (
            self.asset_type.value
            if isinstance(self.asset_type, Enum)
            else self.asset_type
        )
        self.asset_type = cast(InstrumentAssetType, str(asset_raw).strip().upper())
        if self.asset_type not in {"STOCK", "FUTURES", "FUND", "OPTION"}:
            raise ValueError(f"Unsupported asset_type: {self.asset_type}")
        if self.option_type is not None:
            option_raw = (
                self.option_type.value
                if isinstance(self.option_type, Enum)
                else self.option_type
            )
            self.option_type = cast(
                InstrumentOptionType, str(option_raw).strip().upper()
            )
            if self.option_type not in {"CALL", "PUT"}:
                raise ValueError(f"Unsupported option_type: {self.option_type}")
        if self.option_margin_model is not None:
            margin_model_raw = (
                self.option_margin_model.value
                if isinstance(self.option_margin_model, Enum)
                else self.option_margin_model
            )
            self.option_margin_model = cast(
                InstrumentOptionMarginModel, str(margin_model_raw).strip().upper()
            )
            if self.option_margin_model not in {
                "RATIO",
                "CHINA_SINGLE_LEG",
                "US_BROKER_SINGLE_LEG",
                "US_BROKER_SINGLE_LEG_VOL_ADJUSTED",
            }:
                raise ValueError(
                    f"Unsupported option_margin_model: {self.option_margin_model}"
                )
        if self.settlement_type is not None:
            settlement_raw = (
                self.settlement_type.value
                if isinstance(self.settlement_type, Enum)
                else self.settlement_type
            )
            self.settlement_type = cast(
                InstrumentSettlementType, str(settlement_raw).strip().lower()
            )
            if self.settlement_type not in {"cash", "settlement_price", "force_close"}:
                raise ValueError(f"Unsupported settlement_type: {self.settlement_type}")
        if self.lot_size is not None and self.lot_size <= 0:
            raise ValueError("lot_size must be > 0")
        if self.reference_volatility is not None and self.reference_volatility <= 0:
            raise ValueError("reference_volatility must be > 0")
        if self.sellable_after_days is not None and self.sellable_after_days not in (
            0,
            1,
        ):
            raise ValueError(
                "InstrumentConfig.sellable_after_days must be 0 (T+0) or 1 (T+1); "
                f"got {self.sellable_after_days} (T+2+ not yet supported)"
            )


@dataclass
class ChinaFuturesFeeConfig:
    """中国期货费率配置."""

    symbol_prefix: str
    commission_rate: float

    def __post_init__(self) -> None:
        """Validate and normalize fee config."""
        self.symbol_prefix = self.symbol_prefix.strip().upper()
        if not self.symbol_prefix:
            raise ValueError("symbol_prefix must not be empty")
        if self.commission_rate < 0:
            raise ValueError("commission_rate must be >= 0")


@dataclass
class ChinaFuturesSessionConfig:
    """中国期货交易时段配置."""

    start: str
    end: str
    session: str = "continuous"


@dataclass
class ChinaFuturesValidationConfig:
    """中国期货校验开关配置."""

    symbol_prefix: str
    enforce_tick_size: Optional[bool] = None
    enforce_lot_size: Optional[bool] = None

    def __post_init__(self) -> None:
        """Validate and normalize validation switch config."""
        self.symbol_prefix = self.symbol_prefix.strip().upper()
        if not self.symbol_prefix:
            raise ValueError("symbol_prefix must not be empty")
        if self.enforce_tick_size is None and self.enforce_lot_size is None:
            raise ValueError(
                "must set enforce_tick_size or enforce_lot_size "
                "in ChinaFuturesValidationConfig"
            )


@dataclass
class ChinaFuturesInstrumentTemplateConfig:
    """中国期货品种模板配置."""

    symbol_prefix: str
    multiplier: Optional[float] = None
    margin_ratio: Optional[float] = None
    tick_size: Optional[float] = None
    lot_size: Optional[float] = None
    commission_rate: Optional[float] = None
    enforce_tick_size: Optional[bool] = None
    enforce_lot_size: Optional[bool] = None

    def __post_init__(self) -> None:
        """Validate and normalize template config."""
        self.symbol_prefix = self.symbol_prefix.strip().upper()
        if not self.symbol_prefix:
            raise ValueError("symbol_prefix must not be empty")
        if self.multiplier is not None and self.multiplier <= 0:
            raise ValueError("multiplier must be > 0")
        if self.margin_ratio is not None and self.margin_ratio <= 0:
            raise ValueError("margin_ratio must be > 0")
        if self.tick_size is not None and self.tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        if self.lot_size is not None and self.lot_size <= 0:
            raise ValueError("lot_size must be > 0")
        if self.commission_rate is not None and self.commission_rate < 0:
            raise ValueError("commission_rate must be >= 0")


@dataclass
class ChinaFuturesConfig:
    """中国期货增强配置."""

    enforce_sessions: bool = True
    use_china_futures_market: bool = True
    session_profile: str = "CN_FUTURES_DAY"
    enforce_tick_size: bool = True
    enforce_lot_size: bool = True
    fee_by_symbol_prefix: Optional[List[ChinaFuturesFeeConfig]] = None
    validation_by_symbol_prefix: Optional[List[ChinaFuturesValidationConfig]] = None
    instrument_templates_by_symbol_prefix: Optional[
        List[ChinaFuturesInstrumentTemplateConfig]
    ] = None
    sessions: Optional[List[ChinaFuturesSessionConfig]] = None

    def __post_init__(self) -> None:
        """Validate duplicate prefixes across config lists."""
        valid_session_profiles = {
            "CN_FUTURES_DAY",
            "CN_FUTURES_COMMODITY_DAY",
            "CN_FUTURES_CFFEX_STOCK_INDEX_DAY",
            "CN_FUTURES_CFFEX_BOND_DAY",
            "CN_FUTURES_NIGHT_23",
            "CN_FUTURES_NIGHT_01",
            "CN_FUTURES_NIGHT_0230",
        }
        self.session_profile = self.session_profile.strip().upper()
        if self.session_profile not in valid_session_profiles:
            raise ValueError(
                "session_profile must be one of "
                "CN_FUTURES_DAY/CN_FUTURES_COMMODITY_DAY/"
                "CN_FUTURES_CFFEX_STOCK_INDEX_DAY/CN_FUTURES_CFFEX_BOND_DAY/"
                "CN_FUTURES_NIGHT_23/CN_FUTURES_NIGHT_01/CN_FUTURES_NIGHT_0230"
            )
        if self.fee_by_symbol_prefix:
            seen_fee: Dict[str, int] = {}
            for idx, fee in enumerate(self.fee_by_symbol_prefix):
                if fee.symbol_prefix in seen_fee:
                    prev_idx = seen_fee[fee.symbol_prefix]
                    raise ValueError(
                        "fee_by_symbol_prefix"
                        f"[{idx}] duplicates symbol_prefix '{fee.symbol_prefix}' "
                        f"already used at fee_by_symbol_prefix[{prev_idx}]"
                    )
                seen_fee[fee.symbol_prefix] = idx
        if self.validation_by_symbol_prefix:
            seen_validation: Dict[str, int] = {}
            for idx, rule in enumerate(self.validation_by_symbol_prefix):
                if rule.symbol_prefix in seen_validation:
                    prev_idx = seen_validation[rule.symbol_prefix]
                    raise ValueError(
                        "validation_by_symbol_prefix"
                        f"[{idx}] duplicates symbol_prefix '{rule.symbol_prefix}' "
                        "already used at "
                        f"validation_by_symbol_prefix[{prev_idx}]"
                    )
                seen_validation[rule.symbol_prefix] = idx
        if self.instrument_templates_by_symbol_prefix:
            seen_template: Dict[str, int] = {}
            for idx, template in enumerate(self.instrument_templates_by_symbol_prefix):
                if template.symbol_prefix in seen_template:
                    prev_idx = seen_template[template.symbol_prefix]
                    raise ValueError(
                        "instrument_templates_by_symbol_prefix"
                        f"[{idx}] duplicates symbol_prefix "
                        f"'{template.symbol_prefix}' already used at "
                        f"instrument_templates_by_symbol_prefix[{prev_idx}]"
                    )
                seen_template[template.symbol_prefix] = idx


@dataclass
class ChinaOptionsFeeConfig:
    """中国期权费率配置."""

    symbol_prefix: str
    commission_per_contract: float

    def __post_init__(self) -> None:
        """Validate and normalize option fee config."""
        self.symbol_prefix = self.symbol_prefix.strip().upper()
        if not self.symbol_prefix:
            raise ValueError("symbol_prefix must not be empty")
        if self.commission_per_contract < 0:
            raise ValueError("commission_per_contract must be >= 0")


@dataclass
class ChinaOptionsSessionConfig:
    """中国期权交易时段配置."""

    start: str
    end: str
    session: str = "continuous"


@dataclass
class ChinaOptionsConfig:
    """中国期权增强配置."""

    use_china_market: bool = True
    fee_per_contract: Optional[float] = None
    fee_by_symbol_prefix: Optional[List[ChinaOptionsFeeConfig]] = None
    sessions: Optional[List[ChinaOptionsSessionConfig]] = None

    def __post_init__(self) -> None:
        """Validate china options config fields."""
        if self.fee_per_contract is not None and self.fee_per_contract < 0:
            raise ValueError("fee_per_contract must be >= 0")
        if self.fee_by_symbol_prefix:
            seen_fee: Dict[str, int] = {}
            for idx, fee in enumerate(self.fee_by_symbol_prefix):
                if fee.symbol_prefix in seen_fee:
                    prev_idx = seen_fee[fee.symbol_prefix]
                    raise ValueError(
                        "fee_by_symbol_prefix"
                        f"[{idx}] duplicates symbol_prefix '{fee.symbol_prefix}' "
                        f"already used at fee_by_symbol_prefix[{prev_idx}]"
                    )
                seen_fee[fee.symbol_prefix] = idx


@dataclass
class ChinaStockConfig:
    """中国股票/基金撮合增强配置.

    与 ChinaFuturesConfig 的 enforce_tick_size 对称。缺省开启:
    A 股股票 0.01、基金/债券 0.001 的规则是确定的, 且 Instrument 缺省 tick
    已按 asset_type 分流, 不会出现"元数据缺失就默认 0.01"导致的误拒。
    """

    enforce_tick_size: bool = True


@dataclass
class RiskConfig:
    """
    [Risk Level] Configuration for Risk Management.

    Defines **Safety Constraints**.

    *   **"How large?"**: `max_position_pct`, `max_order_size`
    *   **"When to stop?"**: `max_account_drawdown`, `stop_loss_threshold`
    *   **"What is forbidden?"**: `restricted_list`

    **Order & Position Limits:**
    :param active: Master switch to enable/disable risk checks. Default True.
    :param check_cash: Enable cash/margin sufficiency checks at order time.
                       Default True. This gates BOTH the submission-time and the
                       execution-time (fill) affordability checks, for stock,
                       futures and option orders alike. When set to False the
                       account may overdraft (cash/margin can go negative) so the
                       backtest runs to completion. Note it only affects order-time
                       checks: it does NOT disable the day-end forced liquidation
                       of a margin account, which is governed independently by
                       ``allow_force_liquidation``. To let a margin account run
                       fully underwater, also set ``allow_force_liquidation=False``.
    :param safety_margin: Cash buffer to reserve (e.g., 0.0001 to avoid precision
                          issues).
    :param max_order_size: Max quantity per order.
    :param max_order_value: Max value per order.
    :param max_position_size: Max quantity per position.
    :param max_position_pct: Max position value as a percentage of total equity
                             (e.g., 0.1 for 10%).
    :param restricted_list: List of symbols forbidden to trade.
    :param sector_concentration: Tuple (limit, sector_map) to limit sector exposure.

    **Account Level Protections:**
    :param max_account_drawdown: Max allowed drawdown percentage (e.g., 0.2 for 20%).
                                 If breached, may trigger liquidations or stop trading.
    :param max_daily_loss: Max allowed daily loss percentage.
    :param stop_loss_threshold: Net value threshold (e.g., 0.8). If equity drops below
                                initial_cash * threshold, trading is stopped.
    :param account_mode: Account mode. "cash" (default) or "margin".
    :param enable_short_sell: Whether margin account allows opening short stock
                              positions.
    :param initial_margin_ratio: Initial margin ratio for stock/fund in margin mode.
    :param maintenance_margin_ratio: Maintenance margin ratio reference for
                                     margin account.
    :param financing_rate_annual: Annual financing rate for margin buying.
    :param borrow_rate_annual: Annual borrow rate for short selling.
    :param allow_force_liquidation: Whether to force-liquidate positions when
                                    maintenance ratio is breached.
    :param liquidation_priority: Forced-liquidation order, "short_first" (default)
                                 or "long_first".
    """

    active: bool = True
    check_cash: bool = True
    safety_margin: float = 0.0001
    max_order_size: Optional[float] = None
    max_order_value: Optional[float] = None
    max_position_size: Optional[float] = None
    restricted_list: Optional[List[str]] = None
    max_position_pct: Optional[float] = None
    sector_concentration: Optional[Union[float, tuple]] = None

    # Account Level Risk
    max_account_drawdown: Optional[float] = None
    max_daily_loss: Optional[float] = None
    stop_loss_threshold: Optional[float] = (
        None  # e.g., 0.8 means stop if equity < 0.8 * initial
    )
    account_mode: str = "cash"
    enable_short_sell: bool = False
    initial_margin_ratio: float = 1.0
    maintenance_margin_ratio: float = 0.3
    financing_rate_annual: float = 0.08
    borrow_rate_annual: float = 0.10
    allow_force_liquidation: bool = True
    liquidation_priority: str = "short_first"


@dataclass
class StrategyConfig:
    """
    [Account Level] Configuration for strategy execution environment.

    Defines **Account & Execution**.

    *   **"How much money?"**: `initial_cash`
    *   **"How to execute?"**: `slippage`, `volume_limit_pct`
    *   **"What constraints?"**: `max_long_positions`, `risk`

    **Capital & Costs:**
    :param initial_cash: Initial capital for the backtest. Default 100,000.0.
    :param commission_rate: Default commission rate (e.g., 0.0003).
    :param commission_policy: Optional default commission policy dict.
                              Supported types: `percent`, `fixed`, `per_unit`.
                              When provided, it takes precedence over
                              `commission_rate`.
    :param stamp_tax_rate: Default stamp tax rate (sell side).
    :param transfer_fee_rate: Default transfer fee rate.
    :param min_commission: Default minimum commission.

    **Execution Behavior:**
    :param enable_fractional_shares: Allow fractional share trading. Default False.
    :param round_fill_price: Round execution price to tick size. Default True.
    :param slippage: Global slippage policy. 推荐显式写为
                     {"type": "percent", "value": 0.0002}、
                     {"type": "fixed", "value": 0.2} 或
                     {"type": "ticks", "value": 1}。
                     裸 float 仍兼容，但按 percent 语义解析。
    :param volume_limit_pct: Max participation rate. 0.25 means order size is
                             capped at 25% of the bar's volume. Default 0.25.
    :param exit_on_last_bar: Auto-close all positions at the end of backtest.
                             Default True.
    :param indicator_mode: Indicator execution mode. "incremental" updates indicator
                           state on each bar; "precompute" prepares full series before
                           run.

    **Constraints & Risk:**
    :param max_long_positions: Max number of simultaneous long positions.
    :param max_short_positions: Max number of simultaneous short positions.
    :param risk: `RiskConfig` object containing detailed risk rules.
    """

    # Capital Management
    initial_cash: float = 100000.0

    # Fees & Commission (Default / Fallback)
    commission_rate: float = 0.0  # Commission rate (e.g. 0.0003 for 0.03%)
    commission_policy: Optional[Dict[str, Any]] = None
    stamp_tax_rate: float = 0.0  # Stamp tax rate (e.g. 0.001, sell only)
    transfer_fee_rate: float = 0.0  # Transfer fee rate
    min_commission: float = 0.0  # Minimum commission per order (e.g. 5.0)

    # Execution
    enable_fractional_shares: bool = False
    round_fill_price: bool = True
    slippage: Union[float, Dict[str, Any], None] = 0.0
    volume_limit_pct: float = 0.25  # Max participation rate (e.g., 25% of bar volume)

    # Position Sizing Constraints
    max_long_positions: Optional[int] = None
    max_short_positions: Optional[int] = None

    # Other
    exit_on_last_bar: bool = True
    indicator_mode: str = "precompute"

    # Risk Config
    risk: Optional[RiskConfig] = None

    # Multi-Strategy Topology & Risk Controls
    strategy_id: Optional[str] = None
    strategies_by_slot: Optional[Dict[str, Any]] = None
    strategy_source: Optional[str] = None
    strategy_loader: Optional[str] = None
    strategy_loader_options: Optional[Dict[str, Any]] = None
    strategy_max_order_value: Optional[Dict[str, float]] = None
    strategy_max_order_size: Optional[Dict[str, float]] = None
    strategy_max_position_size: Optional[Dict[str, float]] = None
    strategy_max_daily_loss: Optional[Dict[str, float]] = None
    strategy_max_drawdown: Optional[Dict[str, float]] = None
    strategy_reduce_only_after_risk: Optional[Dict[str, bool]] = None
    strategy_risk_cooldown_bars: Optional[Dict[str, int]] = None
    strategy_priority: Optional[Dict[str, int]] = None
    strategy_risk_budget: Optional[Dict[str, float]] = None
    strategy_fill_policy: Optional[Dict[str, Dict[str, Any]]] = None
    strategy_slippage: Optional[Dict[str, Dict[str, Any]]] = None
    strategy_commission: Optional[Dict[str, Dict[str, Any]]] = None
    portfolio_risk_budget: Optional[float] = None


@dataclass
class BacktestConfig:
    """
    [Top Level] Configuration for the entire Backtest Simulation.

    Defines the **SIMULATION SCENARIO**.

    *   **"When?"**: `start_time`, `end_time`
    *   **"What assets?"**: `instruments`, `instruments_config`
    *   **"How?"**: `strategy_config` (Strategy & Account settings)
    *   **"Analysis?"**: `bootstrap_samples`, `analysis_config`

    **Time & Scope:**
    :param start_time: Backtest start time (e.g., "2020-01-01").
    :param end_time: Backtest end time.
    :param strategy_config: Configuration for the strategy/account.

    **Asset Selection:**
    :param instruments: Quick list of symbols to trade (using default properties).
                        Example: `["AAPL", "MSFT"]`.
    :param instruments_config: Detailed configuration for specific assets.
                               List of `InstrumentConfig` or Dict
                               `{symbol: InstrumentConfig}`.
                               Use this for Futures, Options, or non-standard Stocks.

    **Environment:**
    :param benchmark: Benchmark symbol for performance comparison.
    :param timezone: Exchange timezone. Default "Asia/Shanghai".
    :param days_per_year: Annualization factor for risk metrics (Sharpe/Sortino/
                          volatility). Traditional markets use 252; crypto 24/7
                          markets use 365. Default 252.0.
    :param risk_free_rate: Annualized risk-free rate subtracted from annualized
                           return in Sharpe/Sortino/UPI. Default 0.0.
    :param show_progress: Show progress bar. Default True.
    :param history_depth: Auto-load N bars of history before strategy starts.

    **Analysis:**
    :param bootstrap_samples: Number of bootstrap samples for statistical significance.
    :param bootstrap_sample_size: Size of each bootstrap sample.
    :param analysis_config: Dictionary for extra analysis settings (e.g., plotting).
    """

    strategy_config: StrategyConfig
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    # Asset Selection
    instruments: Optional[List[str]] = None  # Quick list of symbols (Default props)
    instruments_config: Optional[
        Union[List[InstrumentConfig], Dict[str, InstrumentConfig]]
    ] = None  # Detailed props (Overrides defaults)
    china_futures: Optional[ChinaFuturesConfig] = None
    china_options: Optional[ChinaOptionsConfig] = None
    china_stock: Optional[ChinaStockConfig] = None

    benchmark: Optional[str] = None
    timezone: str = "Asia/Shanghai"
    days_per_year: float = 252.0
    risk_free_rate: float = 0.0
    show_progress: bool = True
    history_depth: int = 0

    # Analysis & Bootstrap
    bootstrap_samples: int = 1000
    bootstrap_sample_size: Optional[int] = None
    analysis_config: Optional[Dict[str, Any]] = None


# Global instance
strategy_config = StrategyConfig()
