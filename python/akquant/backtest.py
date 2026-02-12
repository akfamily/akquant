import os
import sys
from functools import cached_property
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    cast,
)

import pandas as pd

from .akquant import (
    AssetType,
    Bar,
    ClosedTrade,
    DataFeed,
    Engine,
    ExecutionMode,
    Instrument,
    Order,
    PerformanceMetrics,
)
from .akquant import (
    BacktestResult as RustBacktestResult,
)
from .config import BacktestConfig, InstrumentConfig
from .data import ParquetDataCatalog
from .log import get_logger, register_logger
from .risk import apply_risk_config
from .strategy import Strategy
from .utils import df_to_arrays, prepare_dataframe
from .utils.inspector import infer_warmup_period


class BacktestResult:
    """
    Backtest Result Wrapper.

    Wraps the underlying Rust BacktestResult to provide Python-friendly properties
    like DataFrames.
    """

    def __init__(self, raw_result: RustBacktestResult, timezone: str = "Asia/Shanghai"):
        """
        Initialize the BacktestResult wrapper.

        :param raw_result: The raw Rust BacktestResult object.
        :param timezone: The timezone string for datetime conversion.
        """
        self._raw = raw_result
        self._timezone = timezone

    @property
    def trades(self) -> List[ClosedTrade]:
        """
        Get closed trades as a list of raw objects (Raw Access).

        These are the raw Rust objects, useful for iteration and accessing complex
        fields. For statistical analysis, use `trades_df`.
        """
        return cast(List[ClosedTrade], self._raw.trades)

    @property
    def orders(self) -> List[Order]:
        """
        Get orders as a list of raw objects (Raw Access).

        These are the raw Rust objects, useful for iteration and debugging.
        For statistical analysis, use `orders_df`.
        """
        if hasattr(self._raw, "orders"):
            return cast(List[Order], self._raw.orders)
        return []

    @property
    def metrics(self) -> PerformanceMetrics:
        """
        Get performance metrics as a raw object (Raw Access).

        This is the raw Rust object containing all metrics fields.
        For a DataFrame view, use `metrics_df`.
        """
        return self._raw.metrics

    @property
    def positions(self) -> pd.DataFrame:
        """
        Get positions history as a Pandas DataFrame.

        Index: Datetime (Timezone-aware)
        Columns: Symbols
        Values: Quantity.
        """
        if not self._raw.snapshots:
            return pd.DataFrame()

        # Extract data from snapshots
        data = []
        timestamps = []

        for ts, snapshots in self._raw.snapshots:
            timestamps.append(ts)
            # Create a dict for this timestamp: {symbol: quantity}
            row = {s.symbol: s.quantity for s in snapshots}
            data.append(row)

        df = pd.DataFrame(data, index=timestamps)

        # Convert nanosecond timestamp to datetime with timezone
        df.index = pd.to_datetime(df.index, unit="ns", utc=True).tz_convert(
            self._timezone
        )

        # Sort index just in case
        df = df.sort_index()

        # Fill missing values with 0.0
        df = df.fillna(0.0)

        return cast(pd.DataFrame, df)

    @property
    def positions_df(self) -> pd.DataFrame:
        """
        Get detailed positions history as a Pandas DataFrame (PyBroker style).

        Columns:
            - date (datetime): Snapshot time.
            - symbol (str): Trading symbol.
            - long_shares (float): Long position quantity.
            - short_shares (float): Short position quantity.
            - close (float): Closing price.
            - equity (float): Total account equity.
            - market_value (float): Market value of positions.
            - margin (float): Margin used.
            - unrealized_pnl (float): Floating PnL.
        """
        # Try to use the Rust optimized getter if available
        if hasattr(self._raw, "get_positions_dict"):
            data = self._raw.get_positions_dict()
            if not data or not data["symbol"]:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Convert date to datetime
            df["date"] = pd.to_datetime(df["date"], unit="ns", utc=True).dt.tz_convert(
                self._timezone
            )

            # Reorder columns
            cols = [
                "long_shares",
                "short_shares",
                "close",
                "equity",
                "market_value",
                "margin",
                "unrealized_pnl",
                "symbol",
                "date",
            ]
            # Ensure all columns exist
            # (in case of empty or mismatch, though Rust guarantees keys)
            existing_cols = [c for c in cols if c in df.columns]
            df = df[existing_cols]

            # Sort
            df = df.sort_values(by=["symbol", "date"])

            return cast(pd.DataFrame, df)

        return pd.DataFrame()

    @property
    def metrics_df(self) -> pd.DataFrame:
        """
        Get performance metrics as a Pandas DataFrame.

        Returns a DataFrame indexed by metric name with a single 'value' column,
        matching PyBroker's format.
        """
        df = cast(pd.DataFrame, self._raw.metrics_df)

        # Convert time fields to the configured timezone
        time_fields = ["start_time", "end_time"]
        for field in time_fields:
            if field in df.index:
                val = df.at[field, "value"]
                if val is not None:
                    try:
                        # Convert to pandas Timestamp for easy tz handling
                        ts = pd.Timestamp(cast(Any, val))
                        if ts.tz is not None:
                            df.at[field, "value"] = ts.tz_convert(self._timezone)
                    except Exception:
                        pass

        return df

    @cached_property
    def orders_df(self) -> pd.DataFrame:
        """
        Get orders history as a Pandas DataFrame.

        Columns:
            - id (str): Order ID.
            - symbol (str): Trading symbol.
            - side (str): 'buy' or 'sell'.
            - order_type (str): 'market', 'limit', 'stop'.
            - quantity (float): Order quantity.
            - filled_quantity (float): Executed quantity.
            - limit_price (float): Price for limit orders.
            - stop_price (float): Trigger price for stop orders.
            - avg_price (float): Average execution price.
            - commission (float): Commission paid.
            - status (str): 'filled', 'cancelled', 'rejected', etc.
            - time_in_force (str): 'gtc', 'day', 'ioc', etc.
            - created_at (datetime): Creation time.
        """
        if not hasattr(self._raw, "orders_df"):
            return pd.DataFrame()

        df = cast(pd.DataFrame, self._raw.orders_df.copy())

        if df.empty:
            return df

        if "created_at" in df.columns:
            # Rust returns int64 timestamp (ns since epoch)
            if pd.api.types.is_numeric_dtype(df["created_at"]):
                df["created_at"] = pd.to_datetime(
                    df["created_at"], unit="ns", utc=True
                ).dt.tz_convert(self._timezone)
            elif hasattr(df["created_at"], "dt"):
                if df["created_at"].dt.tz is None:
                    df["created_at"] = (
                        df["created_at"]
                        .dt.tz_localize("UTC")
                        .dt.tz_convert(self._timezone)
                    )
                else:
                    df["created_at"] = df["created_at"].dt.tz_convert(self._timezone)

        if "updated_at" in df.columns:
            # Rust returns int64 timestamp (ns since epoch)
            if pd.api.types.is_numeric_dtype(df["updated_at"]):
                df["updated_at"] = pd.to_datetime(
                    df["updated_at"], unit="ns", utc=True
                ).dt.tz_convert(self._timezone)
            elif hasattr(df["updated_at"], "dt"):
                if df["updated_at"].dt.tz is None:
                    df["updated_at"] = (
                        df["updated_at"]
                        .dt.tz_localize("UTC")
                        .dt.tz_convert(self._timezone)
                    )
                else:
                    df["updated_at"] = df["updated_at"].dt.tz_convert(self._timezone)

        # Calculate derivative columns
        if "filled_quantity" in df.columns and "avg_price" in df.columns:
            # Calculate filled value (成交金额)
            df["filled_value"] = df["filled_quantity"] * df["avg_price"].fillna(0.0)

        if "created_at" in df.columns and "updated_at" in df.columns:
            # Calculate duration (存续时长)
            df["duration"] = df["updated_at"] - df["created_at"]

        # Sort by creation time for better readability
        if "created_at" in df.columns:
            df.sort_values(by="created_at", inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    @cached_property
    def trades_df(self) -> pd.DataFrame:
        """
        Get closed trades as a Pandas DataFrame.

        Columns:
            - symbol (str): Trading symbol.
            - entry_time (datetime): Time of entry.
            - exit_time (datetime): Time of exit.
            - entry_price (float): Average entry price.
            - exit_price (float): Average exit price.
            - quantity (float): Traded quantity.
            - side (str): 'long' or 'short'.
            - pnl (float): Gross PnL.
            - net_pnl (float): Net PnL (after commission).
            - return_pct (float): Trade return (decimal).
            - commission (float): Commission paid.
            - duration_bars (int): Number of bars held.
            - duration (timedelta): Duration of trade.
        """
        if not self._raw.trades:
            return pd.DataFrame()

        # Try to use the optimized get_trades_dict method (from Rust)
        # Fallback to loop if not available (old binary)
        if hasattr(self._raw, "get_trades_dict"):
            data_dict = self._raw.get_trades_dict()
            df = pd.DataFrame(data_dict)
        else:
            data_list = []
            for t in self._raw.trades:
                data_list.append(
                    {
                        "symbol": t.symbol,
                        "entry_time": t.entry_time,
                        "exit_time": t.exit_time,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "quantity": t.quantity,
                        "side": t.side,
                        "pnl": t.pnl,
                        "net_pnl": t.net_pnl,
                        "return_pct": t.return_pct,
                        "commission": t.commission,
                        "duration_bars": t.duration_bars,
                        "duration": t.duration,
                    }
                )
            df = pd.DataFrame(data_list)

        # Convert timestamps
        df["entry_time"] = pd.to_datetime(
            df["entry_time"], unit="ns", utc=True
        ).dt.tz_convert(self._timezone)
        df["exit_time"] = pd.to_datetime(
            df["exit_time"], unit="ns", utc=True
        ).dt.tz_convert(self._timezone)

        # Convert duration to Timedelta
        if "duration" in df.columns:
            df["duration"] = pd.to_timedelta(df["duration"], unit="ns")

        return df

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the raw result."""
        return getattr(self._raw, name)

    def __repr__(self) -> str:
        """Return the string representation of the result (Vertical Metrics)."""
        metrics = self.metrics_df
        metrics.columns = ["Value"]
        return f"BacktestResult:\n{metrics.to_string()}"

    def __dir__(self) -> List[str]:
        """Return the list of attributes including raw result attributes."""
        return list(set(dir(self._raw) + list(self.__dict__.keys()) + ["positions"]))

    def plot(
        self,
        symbol: Optional[str] = None,
        show: bool = True,
        title: str = "Backtest Result",
    ) -> Any:
        """
        Plot the backtest results using Plotly.

        :param symbol: The symbol to highlight positions for.
        :param show: Whether to display the plot immediately.
        :param title: Title of the plot.
        :return: Plotly Figure object.
        """
        try:
            from .plot import plot_result
        except ImportError:
            print(
                "Plotly is not installed. Please install it using `pip install plotly` "
                "or `pip install akquant[plot]`."
            )
            return None

        return plot_result(self, symbol=symbol, show=show, title=title)


class FunctionalStrategy(Strategy):
    """内部策略包装器，用于支持函数式 API (Zipline 风格)."""

    def __init__(
        self,
        initialize: Optional[Callable[[Any], None]],
        on_bar: Optional[Callable[[Any, Bar], None]],
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the FunctionalStrategy."""
        super().__init__()
        self._initialize = initialize
        self._on_bar_func = on_bar
        self._context = context or {}

        # 将 context 注入到 self 中，模拟 Zipline 的 context 对象
        # 用户可以通过 self.xxx 访问 context 属性
        for k, v in self._context.items():
            setattr(self, k, v)

        # 调用初始化函数
        if self._initialize is not None:
            self._initialize(self)

    def on_bar(self, bar: Bar) -> None:
        """Delegate on_bar event to the user-provided function."""
        if self._on_bar_func is not None:
            self._on_bar_func(self, bar)


def run_backtest(
    data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame], List[Bar]]] = None,
    strategy: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None] = None,
    symbol: Union[str, List[str]] = "BENCHMARK",
    initial_cash: Optional[float] = None,
    commission_rate: Optional[float] = None,
    stamp_tax_rate: float = 0.0,
    transfer_fee_rate: float = 0.0,
    min_commission: float = 0.0,
    execution_mode: Union[ExecutionMode, str] = ExecutionMode.NextOpen,
    timezone: Optional[str] = None,
    initialize: Optional[Callable[[Any], None]] = None,
    context: Optional[Dict[str, Any]] = None,
    history_depth: Optional[int] = None,
    warmup_period: int = 0,
    lot_size: Union[int, Dict[str, int], None] = None,
    show_progress: Optional[bool] = None,
    config: Optional[BacktestConfig] = None,
    instruments_config: Optional[
        Union[List[InstrumentConfig], Dict[str, InstrumentConfig]]
    ] = None,
    **kwargs: Any,
) -> BacktestResult:
    """
    简化版回测入口函数.

    :param data: 回测数据，可以是 Pandas DataFrame 或 Bar 列表.
                 可选(如果配置了config或策略订阅).
    :param strategy: 策略类、策略实例或 on_bar 回调函数
    :param symbol: 标的代码
    :param initial_cash: 初始资金 (默认 1,000,000.0)
    :param commission_rate: 佣金率 (默认 0.0)
    :param stamp_tax_rate: 印花税率 (仅卖出, 默认 0.0)
    :param transfer_fee_rate: 过户费率 (默认 0.0)
    :param min_commission: 最低佣金 (默认 0.0)
    :param execution_mode: 执行模式 (ExecutionMode.NextOpen 或 "next_open")
    :param timezone: 时区名称 (默认 "Asia/Shanghai")
    :param initialize: 初始化回调函数 (仅当 strategy 为函数时使用)
    :param context: 初始上下文数据 (仅当 strategy 为函数时使用)
    :param history_depth: 自动维护历史数据的长度 (0 表示禁用)
    :param warmup_period: 策略预热期 (等同于 history_depth，取最大值)
    :param lot_size: 最小交易单位。如果是 int，则应用于所有标的；
                     如果是 Dict[str, int]，则按代码匹配；如果不传(None)，默认为 1。
    :param show_progress: 是否显示进度条 (默认 True)
    :param config: BacktestConfig 配置对象 (可选)
    :param instruments_config: 标的配置列表或字典 (可选)
    :return: 回测结果 Result 对象
    """
    # 0. 设置默认值 (如果未传入且未在 Config 中设置)
    # 优先级: 参数 > Config > 默认值

    # Defaults
    DEFAULT_INITIAL_CASH = 1_000_000.0
    DEFAULT_COMMISSION_RATE = 0.0
    DEFAULT_TIMEZONE = "Asia/Shanghai"
    DEFAULT_SHOW_PROGRESS = True
    DEFAULT_HISTORY_DEPTH = 0

    # Resolve Initial Cash
    if initial_cash is None:
        if config and config.strategy_config:
            initial_cash = config.strategy_config.initial_cash
        else:
            initial_cash = DEFAULT_INITIAL_CASH

    # Resolve Commission Rate
    if commission_rate is None:
        if config and config.strategy_config:
            commission_rate = config.strategy_config.fee_amount
        else:
            commission_rate = DEFAULT_COMMISSION_RATE

    # Resolve Timezone
    if timezone is None:
        if config and config.timezone:
            timezone = config.timezone
        else:
            timezone = DEFAULT_TIMEZONE

    # Resolve Show Progress
    if show_progress is None:
        if config and config.show_progress is not None:
            show_progress = config.show_progress
        else:
            show_progress = DEFAULT_SHOW_PROGRESS

    # Resolve History Depth
    if history_depth is None:
        if config and config.history_depth is not None:
            history_depth = config.history_depth
        else:
            history_depth = DEFAULT_HISTORY_DEPTH

    # 1. 确保日志已初始化
    logger = get_logger()
    if not logger.handlers:
        register_logger(console=True, level="INFO")
        logger = get_logger()

    # 1.2 检查 PyCharm 环境下的进度条可见性
    if show_progress and "PYCHARM_HOSTED" in os.environ:
        # PyCharm Console 或 Run 窗口未开启模拟终端时，isatty 通常为 False
        if not sys.stderr.isatty():
            logger.warning(
                "Progress bar might be invisible in PyCharm. "
                "Solution: Enable 'Emulate terminal in output console' "
                "in Run Configuration."
            )

    # 1.5 处理 Config 覆盖 (剩余部分)
    if config:
        if config.start_date:
            kwargs["start_date"] = config.start_date
        if config.end_date:
            kwargs["end_date"] = config.end_date

        # 注意: initial_cash, commission_rate, timezone, show_progress, history_depth
        # 已经在上方通过优先级逻辑处理过了，这里不需要再覆盖

        # Risk Config injection handled later

    # Handle strategy_params explicitly
    if "strategy_params" in kwargs:
        s_params = kwargs.pop("strategy_params")
        if isinstance(s_params, dict):
            kwargs.update(s_params)

    # 2. 实例化策略 (提前实例化以获取订阅信息)
    strategy_instance = None

    if isinstance(strategy, type) and issubclass(strategy, Strategy):
        try:
            strategy_instance = strategy(**kwargs)
        except TypeError as e:
            # Try fallback only if explicit kwargs failed, but log warning
            # However, if kwargs contained extra unused params, this failure is
            # expected for strict init.
            # But we should try to match params if possible, or just let it fail
            # if user provided params that don't match?
            # The original behavior was silent fallback. We should preserve it
            # but try to be smarter?
            # Or at least warn if strategy_params were provided but ignored.

            # For now, keep the fallback but maybe inspect if it was due to
            # strategy_params
            logger.warning(
                f"Failed to instantiate strategy with provided parameters: {e}. "
                "Falling back to default constructor (no arguments)."
            )
            strategy_instance = strategy()
    elif isinstance(strategy, Strategy):
        strategy_instance = strategy
    elif callable(strategy):
        strategy_instance = FunctionalStrategy(
            initialize, cast(Callable[[Any, Bar], None], strategy), context
        )
    elif strategy is None:
        raise ValueError("Strategy must be provided.")
    else:
        raise ValueError("Invalid strategy type")

    # 注入 context
    if context and hasattr(strategy_instance, "_context"):
        pass
    elif context and strategy_instance:
        for k, v in context.items():
            setattr(strategy_instance, k, v)

    # 注入 Config 中的 Risk Config
    if config and config.strategy_config and config.strategy_config.risk:
        # 如果策略支持 set_risk_config (假设我们添加它，或者直接注入属性)
        if hasattr(strategy_instance, "risk_config"):
            strategy_instance.risk_config = config.strategy_config.risk  # type: ignore

    # 注入费率配置到 Strategy 实例
    if hasattr(strategy_instance, "commission_rate"):
        strategy_instance.commission_rate = commission_rate
    if hasattr(strategy_instance, "min_commission"):
        strategy_instance.min_commission = min_commission
    if hasattr(strategy_instance, "stamp_tax_rate"):
        strategy_instance.stamp_tax_rate = stamp_tax_rate
    if hasattr(strategy_instance, "transfer_fee_rate"):
        strategy_instance.transfer_fee_rate = transfer_fee_rate

    # 注入 lot_size
    # lot_size 参数可能是 int 或 dict。
    # 如果是 dict，则 Strategy._calculate_max_buy_qty 会自动处理
    if lot_size is not None and hasattr(strategy_instance, "lot_size"):
        strategy_instance.lot_size = lot_size
    elif lot_size is None and hasattr(strategy_instance, "lot_size"):
        # 默认值已经在 Strategy.__new__ 中设置为 1
        pass

    # 调用 on_start 获取订阅
    if hasattr(strategy_instance, "on_start"):
        strategy_instance.on_start()

    # 3. 准备数据源和 Symbol
    feed = DataFeed()
    symbols = []
    data_map_for_indicators = {}

    # Normalize symbol arg to list
    if isinstance(symbol, str):
        symbols = [symbol]
    elif isinstance(symbol, list):
        symbols = symbol
    else:
        symbols = ["BENCHMARK"]

    # Merge with Config instruments
    if config and config.instruments:
        for s in config.instruments:
            if s not in symbols:
                symbols.append(s)

    # Merge with Strategy subscriptions
    if hasattr(strategy_instance, "_subscriptions"):
        for s in strategy_instance._subscriptions:
            if s not in symbols:
                symbols.append(s)

    # Determine Data Loading Strategy
    if data is not None:
        # Use provided data
        if isinstance(data, pd.DataFrame):
            # Try to infer symbol from DataFrame if not explicitly provided or default
            if (not symbols or symbols == ["BENCHMARK"]) and "symbol" in data.columns:
                unique_symbols = data["symbol"].unique()
                if len(unique_symbols) == 1:
                    inferred = unique_symbols[0]
                    if symbols == ["BENCHMARK"]:
                        symbols = [inferred]
                    else:
                        if inferred not in symbols:
                            symbols.append(inferred)

            target_symbol = symbols[0] if symbols else "BENCHMARK"
            df = prepare_dataframe(data)
            data_map_for_indicators[target_symbol] = df
            arrays = df_to_arrays(df, symbol=target_symbol)
            feed.add_arrays(*arrays)  # type: ignore
            feed.sort()
            if target_symbol not in symbols:
                symbols = [target_symbol]
        elif isinstance(data, dict):
            # If explicit symbols are provided (i.e., not just the default "BENCHMARK"),
            # we filter the data dictionary to only include requested symbols.
            filter_symbols = "BENCHMARK" not in symbols

            for sym, df in data.items():
                if filter_symbols and sym not in symbols:
                    continue

                df_prep = prepare_dataframe(df)
                data_map_for_indicators[sym] = df_prep
                arrays = df_to_arrays(df_prep, symbol=sym)
                feed.add_arrays(*arrays)  # type: ignore
                if sym not in symbols:
                    symbols.append(sym)
            feed.sort()
        elif isinstance(data, list):
            if data:
                data.sort(key=lambda b: b.timestamp)
                feed.add_bars(data)
    else:
        # Load from Catalog / Akshare
        if not symbols:
            logger.warning("No symbols specified and no data provided.")

        catalog = ParquetDataCatalog()
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")

        loaded_count = 0
        for sym in symbols:
            # Try Catalog
            df = catalog.read(sym, start_date=start_date, end_date=end_date)
            if df.empty:
                logger.warning(f"Data not found in catalog for {sym}")
                continue

            if not df.empty:
                df = prepare_dataframe(df)
                data_map_for_indicators[sym] = df
                arrays = df_to_arrays(df, symbol=sym)
                feed.add_arrays(*arrays)  # type: ignore
                loaded_count += 1

        if loaded_count > 0:
            feed.sort()
        else:
            if symbols:
                logger.warning("Failed to load data for all requested symbols.")

    # Inject timezone to strategy
    strategy_instance.timezone = timezone

    # 3.5 Pre-calculate indicators
    # Inject data into indicators so they can be accessed in on_bar via get_value()
    if hasattr(strategy_instance, "_indicators") and data_map_for_indicators:
        for symbol_key, df_val in data_map_for_indicators.items():
            for ind in strategy_instance._indicators:
                try:
                    ind(df_val, symbol_key)
                except Exception as e:
                    logger.error(
                        f"Failed to calculate indicator {ind.name} "
                        f"for {symbol_key}: {e}"
                    )

    # 4. 配置引擎
    engine = Engine()
    # engine.set_timezone_name(timezone)
    offset_delta = pd.Timestamp.now(tz=timezone).utcoffset()
    if offset_delta is None:
        raise ValueError(f"Invalid timezone: {timezone}")
    offset = int(offset_delta.total_seconds())
    engine.set_timezone(offset)
    engine.set_cash(initial_cash)
    if history_depth > 0:
        engine.set_history_depth(history_depth)

    # ... (ExecutionMode logic)
    if isinstance(execution_mode, str):
        mode_map = {
            "next_open": ExecutionMode.NextOpen,
            "current_close": ExecutionMode.CurrentClose,
        }
        mode = mode_map.get(execution_mode.lower())
        if not mode:
            logger.warning(
                f"Unknown execution mode '{execution_mode}', defaulting to NextOpen"
            )
            mode = ExecutionMode.NextOpen
        engine.set_execution_mode(mode)
    else:
        engine.set_execution_mode(execution_mode)

    engine.set_t_plus_one(False)  # 默认 T+0，可配置
    engine.set_force_session_continuous(True)
    engine.set_stock_fee_rules(
        commission_rate, stamp_tax_rate, transfer_fee_rate, min_commission
    )

    # Configure other asset fees if provided
    if "fund_commission" in kwargs:
        engine.set_fund_fee_rules(
            kwargs["fund_commission"],
            kwargs.get("fund_transfer_fee", 0.0),
            kwargs.get("fund_min_commission", 0.0),
        )

    if "option_commission" in kwargs:
        engine.set_option_fee_rules(kwargs["option_commission"])

    # Apply Risk Config
    if config and config.strategy_config:
        apply_risk_config(engine, config.strategy_config.risk)

    # 5. 添加标的
    # 解析 Instrument Config
    inst_conf_map = {}

    # From arguments
    if instruments_config:
        if isinstance(instruments_config, list):
            for c in instruments_config:
                inst_conf_map[c.symbol] = c
        elif isinstance(instruments_config, dict):
            inst_conf_map.update(instruments_config)

    # From BacktestConfig
    if config and config.instruments_config:
        if isinstance(config.instruments_config, list):
            for c in config.instruments_config:
                if c.symbol not in inst_conf_map:
                    inst_conf_map[c.symbol] = c
        elif isinstance(config.instruments_config, dict):
            for k, v in config.instruments_config.items():
                if k not in inst_conf_map:
                    inst_conf_map[k] = v

    # Default values from kwargs
    default_multiplier = kwargs.get("multiplier", 1.0)
    default_margin_ratio = kwargs.get("margin_ratio", 1.0)
    default_tick_size = kwargs.get("tick_size", 0.01)
    default_asset_type = kwargs.get("asset_type", AssetType.Stock)

    # Option specific fields
    default_option_type = kwargs.get("option_type", None)
    default_strike_price = kwargs.get("strike_price", None)
    default_expiry_date = kwargs.get("expiry_date", None)

    def _parse_asset_type(val: Union[str, AssetType]) -> AssetType:
        if isinstance(val, AssetType):
            return val
        if isinstance(val, str):
            v_lower = val.lower()
            if "stock" in v_lower:
                return AssetType.Stock
            if "future" in v_lower:
                return AssetType.Futures
            if "fund" in v_lower:
                return AssetType.Fund
            if "option" in v_lower:
                return AssetType.Option
        return AssetType.Stock

    def _parse_option_type(val: Any) -> Any:
        # OptionType might not be available in current binary
        try:
            from .akquant import OptionType  # type: ignore

            if isinstance(val, str):
                if val.lower() == "call":
                    return OptionType.Call
                if val.lower() == "put":
                    return OptionType.Put
        except ImportError:
            pass
        return val

    def _parse_expiry(val: Any) -> Optional[int]:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            try:
                # Convert string date to nanosecond timestamp
                return int(pd.Timestamp(val).value)
            except Exception:
                pass
        return None

    for sym in symbols:
        # Determine lot_size for this symbol
        current_lot_size = None
        if isinstance(lot_size, int):
            current_lot_size = lot_size
        elif isinstance(lot_size, dict):
            current_lot_size = lot_size.get(sym)

        # Check specific config
        i_conf = inst_conf_map.get(sym)

        if i_conf:
            p_asset_type = _parse_asset_type(i_conf.asset_type)
            p_multiplier = i_conf.multiplier
            p_margin = i_conf.margin_ratio
            p_tick = i_conf.tick_size
            # If config has lot_size, use it, otherwise use global setting
            p_lot = i_conf.lot_size if i_conf.lot_size != 1 else (current_lot_size or 1)

            p_opt_type = _parse_option_type(i_conf.option_type)
            p_strike = i_conf.strike_price
            p_expiry = _parse_expiry(i_conf.expiry_date)
        else:
            p_asset_type = default_asset_type
            p_multiplier = default_multiplier
            p_margin = default_margin_ratio
            p_tick = default_tick_size
            p_lot = current_lot_size or 1

            p_opt_type = default_option_type
            p_strike = default_strike_price
            p_expiry = _parse_expiry(default_expiry_date)

        instr = Instrument(
            sym,
            p_asset_type,
            p_multiplier,
            p_margin,
            p_tick,
            p_opt_type,
            p_strike,
            p_expiry,
            p_lot,
        )
        engine.add_instrument(instr)

    # 6. 添加数据
    engine.add_data(feed)

    # 7. 运行回测
    logger.info("Running backtest via run_backtest()...")

    # 设置自动历史数据维护
    # Logic: effective_depth = max(strategy.warmup_period, inferred_warmup,
    #                              run_backtest(history_depth))
    strategy_warmup = getattr(strategy_instance, "warmup_period", 0)

    # Auto-infer from AST
    inferred_warmup = 0
    try:
        inferred_warmup = infer_warmup_period(type(strategy_instance))
        if inferred_warmup > 0:
            logger.info(f"Auto-inferred warmup period: {inferred_warmup}")
    except Exception as e:
        logger.debug(f"Failed to infer warmup period: {e}")

    effective_depth = max(
        strategy_warmup, inferred_warmup, history_depth, warmup_period
    )

    if effective_depth > 0:
        strategy_instance.set_history_depth(effective_depth)

    # 7.5 Prepare Indicators (Vectorized Pre-calculation)
    if hasattr(strategy_instance, "_prepare_indicators") and data_map_for_indicators:
        strategy_instance._prepare_indicators(data_map_for_indicators)

    try:
        engine.run(strategy_instance, show_progress)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise e
    finally:
        if hasattr(strategy_instance, "on_stop"):
            try:
                strategy_instance.on_stop()
            except Exception as e:
                logger.error(f"Error in on_stop: {e}")

    return BacktestResult(engine.get_results(), timezone=timezone)


def plot_result(
    result: Any,
    show: bool = True,
    filename: Optional[str] = None,
    benchmark: Optional[pd.Series] = None,
) -> None:
    """
    绘制回测结果 (权益曲线、回撤、日收益率).

    :param result: BacktestResult 对象
    :param show: 是否调用 plt.show()
    :param filename: 保存图片的文件名
    :param benchmark: 基准收益率序列 (可选, Series with DatetimeIndex)
    """
    try:
        from datetime import datetime

        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print(
            "Error: matplotlib is required for plotting. "
            "Please install it via 'pip install matplotlib'."
        )
        return

    # Extract data
    equity_curve = result.equity_curve  # List[Tuple[int, float]]

    if not equity_curve:
        print("No equity curve data to plot.")
        return

    # Check if timestamp is in nanoseconds (e.g. > 1e11)
    # 1e11 seconds is roughly year 5138, so valid seconds are < 1e11
    # 1e18 nanoseconds is roughly year 2001
    first_ts = equity_curve[0][0]
    scale = 1.0
    if first_ts > 1e11:
        scale = 1e-9

    from datetime import timezone

    # Use UTC to avoid local timezone issues and align with benchmark data
    times = [
        datetime.fromtimestamp(t * scale, tz=timezone.utc).replace(tzinfo=None)
        for t, _ in equity_curve
    ]
    equity = [e for _, e in equity_curve]

    # Convert to DataFrame for easier calculation
    df = pd.DataFrame({"equity": equity}, index=times)
    df.index.name = "Date"
    df["returns"] = df["equity"].pct_change().fillna(0)

    # Calculate Drawdown
    rolling_max = df["equity"].cummax()
    drawdown = (df["equity"] - rolling_max) / rolling_max

    # Create figure with GridSpec
    fig = plt.figure(figsize=(14, 10))
    # 3 rows: Equity (3), Drawdown (1), Daily Returns (1)
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)

    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df["equity"], label="Strategy", color="#1f77b4", linewidth=1.5)

    if benchmark is not None:
        # Align benchmark to strategy dates
        try:
            # Ensure benchmark has DatetimeIndex
            if not isinstance(benchmark.index, pd.DatetimeIndex):
                benchmark.index = pd.to_datetime(benchmark.index)

            # Normalize timezones: ensure benchmark is tz-naive UTC
            if benchmark.index.tz is not None:
                benchmark.index = benchmark.index.tz_convert("UTC").tz_localize(None)

            # Reindex benchmark to match strategy dates (forward fill for missing days)
            # Normalize dates to start of day for alignment if needed
            # For simplicity, we just plot what overlaps

            # Calculate cumulative return of benchmark
            bench_cum = (1 + benchmark).cumprod()

            # Rebase benchmark to match initial strategy equity
            initial_equity = df["equity"].iloc[0]
            if not bench_cum.empty:
                # Align start
                # Find the closest date in benchmark to start date
                start_date = df.index[0]
                if start_date in bench_cum.index:
                    base_val = bench_cum.loc[start_date]
                else:
                    # Fallback: use first available
                    base_val = bench_cum.iloc[0]

                bench_scaled = (bench_cum / base_val) * initial_equity

                # Filter to strategy range
                bench_plot = bench_scaled[df.index[0] : df.index[-1]]  # type: ignore
                ax1.plot(
                    bench_plot.index,
                    bench_plot,
                    label="Benchmark",
                    color="gray",
                    linestyle="--",
                    alpha=0.7,
                )
        except Exception as e:
            print(f"Warning: Failed to plot benchmark: {e}")

    ax1.set_title("Strategy Performance Analysis", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Equity", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.8)

    # Add Metrics Text Box
    metrics = result.metrics
    trade_metrics = result.trade_metrics

    metrics_text = [
        f"Total Return: {metrics.total_return_pct:>8.2f}%",
        f"Annualized:   {metrics.annualized_return:>8.2%}",
        f"Sharpe Ratio: {metrics.sharpe_ratio:>8.2f}",
        f"Max Drawdown: {metrics.max_drawdown_pct:>8.2f}%",
        f"Win Rate:     {metrics.win_rate:>8.2%}",
    ]

    if hasattr(trade_metrics, "total_closed_trades"):
        metrics_text.append(f"Trades:       {trade_metrics.total_closed_trades:>8d}")

    text_str = "\n".join(metrics_text)

    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgray")
    ax1.text(
        0.02,
        0.05,
        text_str,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=props,
    )

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(
        df.index, drawdown, 0, color="#d62728", alpha=0.3, label="Drawdown"
    )
    ax2.plot(df.index, drawdown, color="#d62728", linewidth=0.8, alpha=0.8)
    ax2.set_ylabel("Drawdown", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)
    # ax2.legend(loc='lower right', fontsize=8)

    # 3. Daily Returns
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.bar(
        df.index,
        df["returns"],
        color="gray",
        alpha=0.5,
        label="Daily Returns",
        width=1.0 if len(df) < 100 else 0.8,
    )
    # Highlight extreme returns? No, keep simple.
    ax3.set_ylabel("Returns", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.3)

    # Format X axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.xticks(rotation=0)

    # Adjust margins
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.95)

    if filename:
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        print(f"Plot saved to {filename}")

    if show:
        plt.show()
