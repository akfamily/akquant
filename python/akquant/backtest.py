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
    DataFeed,
    Engine,
    ExecutionMode,
    Instrument,
)
from .akquant import (
    BacktestResult as RustBacktestResult,
)
from .log import get_logger, register_logger
from .strategy import Strategy
from .utils import df_to_arrays, prepare_dataframe


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
    def daily_positions_df(self) -> pd.DataFrame:
        """
        Get daily positions as a Pandas DataFrame.

        Index: Datetime (Timezone-aware)
        Columns: Symbols
        Values: Quantity.
        """
        if not self._raw.daily_positions:
            return pd.DataFrame()

        # Unzip the list of tuples [(ts, {sym: qty}), ...]
        timestamps, positions = zip(*self._raw.daily_positions)

        df = pd.DataFrame(list(positions), index=timestamps)

        # Convert nanosecond timestamp to datetime with timezone
        df.index = pd.to_datetime(df.index, unit="ns", utc=True).tz_convert(
            self._timezone
        )

        # Sort index just in case
        df = df.sort_index()

        # Fill missing values with 0 (assuming 0 position if not present in map)
        df = df.fillna(0.0)

        return df

    @property
    def metrics_df(self) -> pd.DataFrame:
        """Get performance metrics as a Pandas DataFrame."""
        metrics = self._raw.metrics

        # Manually construct dictionary from known fields since PyO3 objects
        # might not expose __dict__ directly in a clean way or might have extra fields.
        # We use the fields defined in PerformanceMetrics (see akquant.pyi)
        data = {
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "max_drawdown": metrics.max_drawdown,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "volatility": metrics.volatility,
            "ulcer_index": metrics.ulcer_index,
            "upi": metrics.upi,
            "equity_r2": metrics.equity_r2,
            "std_error": metrics.std_error,
            "win_rate": metrics.win_rate,
            "initial_market_value": metrics.initial_market_value,
            "end_market_value": metrics.end_market_value,
            "total_return_pct": metrics.total_return_pct,
        }

        # Return as a DataFrame with one row
        return pd.DataFrame([data], index=["Backtest"])

    @cached_property
    def trades_df(self) -> pd.DataFrame:
        """Get closed trades as a Pandas DataFrame."""
        if not self._raw.trades:
            return pd.DataFrame()

        data = []
        for t in self._raw.trades:
            data.append(
                {
                    "symbol": t.symbol,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "direction": t.direction,
                    "pnl": t.pnl,
                    "net_pnl": t.net_pnl,
                    "return_pct": t.return_pct,
                    "commission": t.commission,
                    "duration_bars": t.duration_bars,
                }
            )

        df = pd.DataFrame(data)

        # Convert timestamps
        df["entry_time"] = pd.to_datetime(
            df["entry_time"], unit="ns", utc=True
        ).dt.tz_convert(self._timezone)
        df["exit_time"] = pd.to_datetime(
            df["exit_time"], unit="ns", utc=True
        ).dt.tz_convert(self._timezone)

        return df

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the raw result."""
        return getattr(self._raw, name)

    def __repr__(self) -> str:
        """Return the string representation of the raw result."""
        return repr(self._raw)

    def __dir__(self) -> List[str]:
        """Return the list of attributes including raw result attributes."""
        return list(
            set(dir(self._raw) + list(self.__dict__.keys()) + ["daily_positions_df"])
        )


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
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame], List[Bar]],
    strategy: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]],
    symbol: Union[str, List[str]] = "BENCHMARK",
    cash: float = 1_000_000.0,
    commission: float = 0.0003,
    stamp_tax: float = 0.0005,
    transfer_fee: float = 0.00001,
    min_commission: float = 5.0,
    execution_mode: Union[ExecutionMode, str] = ExecutionMode.NextOpen,
    timezone: str = "Asia/Shanghai",
    initialize: Optional[Callable[[Any], None]] = None,
    context: Optional[Dict[str, Any]] = None,
    history_depth: int = 0,
    lot_size: Union[int, Dict[str, int], None] = None,
    show_progress: bool = True,
    **kwargs: Any,
) -> BacktestResult:
    """
    简化版回测入口函数.

    :param data: 回测数据，可以是 Pandas DataFrame 或 Bar 列表
    :param strategy: 策略类、策略实例或 on_bar 回调函数
    :param symbol: 标的代码
    :param cash: 初始资金
    :param commission: 佣金率
    :param stamp_tax: 印花税率 (仅卖出)
    :param transfer_fee: 过户费率
    :param min_commission: 最低佣金
    :param execution_mode: 执行模式 (ExecutionMode.NextOpen 或 "next_open")
    :param timezone: 时区名称
    :param initialize: 初始化回调函数 (仅当 strategy 为函数时使用)
    :param context: 初始上下文数据 (仅当 strategy 为函数时使用)
    :param history_depth: 自动维护历史数据的长度 (0 表示禁用)
    :param lot_size: 最小交易单位。如果是 int，则应用于所有标的；
                     如果是 Dict[str, int]，则按代码匹配；如果不传(None)，默认为 1。
    :param show_progress: 是否显示进度条 (默认 True)
    :return: 回测结果 Result 对象
    """
    # 1. 确保日志已初始化
    # 如果用户没有配置过日志，这里会提供一个默认配置
    logger = get_logger()
    if not logger.handlers:
        register_logger(console=True, level="INFO")
        logger = get_logger()

    # 2. 准备数据
    feed = DataFeed()
    symbols = []

    # Normalize symbol to list
    if isinstance(symbol, str):
        symbols = [symbol]
    elif isinstance(symbol, list):
        symbols = symbol
    else:
        # If symbol not provided, try to infer from Dict keys or use default
        symbols = ["BENCHMARK"]

    if isinstance(data, pd.DataFrame):
        # Single DataFrame -> Single Symbol (use first symbol)
        target_symbol = symbols[0] if symbols else "BENCHMARK"
        df = prepare_dataframe(data)
        # Fast Path: Avoid creating Bar objects in Python
        arrays = df_to_arrays(df, symbol=target_symbol)
        feed.add_arrays(*arrays)  # type: ignore
        feed.sort()

        if target_symbol not in symbols:
            symbols = [target_symbol]

    elif isinstance(data, dict):
        # Dict[str, DataFrame] -> Multi Symbol
        symbols = list(data.keys())
        for sym, df in data.items():
            df_prep = prepare_dataframe(df)
            # Fast Path
            arrays = df_to_arrays(df_prep, symbol=sym)
            feed.add_arrays(*arrays)  # type: ignore
        feed.sort()

    elif isinstance(data, list):
        # List[Bar]
        if data:
            data.sort(key=lambda b: b.timestamp)
            feed.add_bars(data)
    else:
        raise ValueError("data must be a DataFrame, Dict[str, DataFrame], or List[Bar]")

    # 3. 设置引擎
    engine = Engine()
    engine.set_timezone_name(timezone)
    engine.set_cash(cash)

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
    engine.set_stock_fee_rules(commission, stamp_tax, transfer_fee, min_commission)

    # Configure other asset fees if provided
    if "fund_commission" in kwargs:
        engine.set_fund_fee_rules(
            kwargs["fund_commission"],
            kwargs.get("fund_transfer_fee", 0.0),
            kwargs.get("fund_min_commission", 0.0),
        )

    if "option_commission" in kwargs:
        engine.set_option_fee_rules(kwargs["option_commission"])

    # 4. 添加标的
    multiplier = kwargs.get("multiplier", 1.0)
    margin_ratio = kwargs.get("margin_ratio", 1.0)
    tick_size = kwargs.get("tick_size", 0.01)
    asset_type = kwargs.get("asset_type", AssetType.Stock)

    # Option specific fields
    option_type = kwargs.get("option_type", None)
    strike_price = kwargs.get("strike_price", None)
    expiry_date = kwargs.get("expiry_date", None)
    # lot_size is handled separately via argument

    for sym in symbols:
        # Determine lot_size for this symbol
        current_lot_size = None
        if isinstance(lot_size, int):
            current_lot_size = lot_size
        elif isinstance(lot_size, dict):
            current_lot_size = lot_size.get(sym)

        instr = Instrument(
            sym,
            asset_type,
            multiplier,
            margin_ratio,
            tick_size,
            option_type,
            strike_price,
            expiry_date,
            current_lot_size,
        )
        engine.add_instrument(instr)

    # 5. 添加数据
    engine.add_data(feed)

    # ... (Rest is same)

    # 6. 准备策略实例
    strategy_instance = None

    if isinstance(strategy, type) and issubclass(strategy, Strategy):
        # 如果是策略类，实例化它
        # 尝试传递 kwargs 给构造函数，如果失败则无参数构造
        try:
            strategy_instance = strategy(**kwargs)
        except TypeError:
            strategy_instance = strategy()
    elif isinstance(strategy, Strategy):
        # 如果已经是实例
        strategy_instance = strategy
    elif callable(strategy):
        # 如果是函数，假设是 on_bar 回调 (Zipline 风格)
        # 需要配合 initialize 使用
        strategy_instance = FunctionalStrategy(
            initialize, cast(Callable[[Any, Bar], None], strategy), context
        )
    else:
        raise ValueError("Invalid strategy type")

    # 7. 运行回测
    logger.info("Running backtest via run_backtest()...")

    # 注入 context 到策略实例
    if context and hasattr(strategy_instance, "_context"):
        # 如果是 FunctionalStrategy
        # 已经在 __init__ 中注入了
        pass
    elif context and strategy_instance:
        # 如果是普通 Strategy，尝试注入属性
        for k, v in context.items():
            setattr(strategy_instance, k, v)

    # 设置自动历史数据维护
    if history_depth > 0:
        strategy_instance.set_history_depth(history_depth)

    engine.run(strategy_instance, show_progress)

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
