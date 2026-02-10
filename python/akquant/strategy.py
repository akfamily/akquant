from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from .akquant import (
    Bar,
    ExecutionMode,
    Order,
    OrderStatus,
    StrategyContext,
    Tick,
    TimeInForce,
)
from .sizer import FixedSize, Sizer
from .utils import parse_duration_to_bars

if TYPE_CHECKING:
    from .indicator import Indicator
    from .ml.model import QuantModel


class Position:
    """
    持仓信息辅助类 (Position Helper).

    允许通过属性访问特定标的的持仓信息.
    """

    def __init__(self, ctx: StrategyContext, symbol: str) -> None:
        """
        初始化持仓辅助对象.

        :param ctx: 策略上下文
        :param symbol: 标的代码
        """
        self._ctx = ctx
        self._symbol = symbol

    @property
    def size(self) -> float:
        """持仓数量."""
        return self._ctx.get_position(self._symbol)

    @property
    def available(self) -> float:
        """可用持仓数量."""
        return self._ctx.get_available_position(self._symbol)

    def __repr__(self) -> str:
        """返回持仓信息的字符串表示."""
        return f"Position(symbol={self._symbol}, size={self.size})"


class Strategy:
    """
    策略基类 (Base Strategy Class).

    采用事件驱动设计
    """

    ctx: Optional[StrategyContext]
    execution_mode: Optional[ExecutionMode]
    sizer: Sizer
    current_bar: Optional[Bar]
    current_tick: Optional[Tick]
    _history_depth: int
    _bars_history: "defaultdict[str, deque[Bar]]"
    _indicators: List["Indicator"]
    _subscriptions: List[str]
    _last_prices: Dict[str, float]
    _rolling_train_window: int
    _rolling_step: int
    _bar_count: int
    _model_configured: bool
    model: Optional["QuantModel"]
    _known_orders: Dict[str, Order]
    timezone: str = "Asia/Shanghai"
    warmup_period: int = 0
    _last_event_type: str = ""  # "bar" or "tick"

    def __new__(cls, *args: Any, **kwargs: Any) -> "Strategy":
        """Create a new Strategy instance."""
        instance = super().__new__(cls)
        instance.ctx = None
        instance.execution_mode = None
        instance.sizer = FixedSize(100)
        instance.current_bar = None
        instance.current_tick = None
        instance._indicators = []
        instance._subscriptions = []
        instance._last_prices = {}
        instance._known_orders = {}
        instance.timezone = getattr(instance, "timezone", "Asia/Shanghai")
        instance._last_event_type = ""

        # 历史数据配置
        instance._history_depth = 0
        instance.warmup_period = getattr(instance, "warmup_period", 0)

        # 滚动训练配置
        instance._rolling_train_window = 0
        instance._rolling_step = 0
        instance._bar_count = 0
        instance._model_configured = False

        # 初始化通常在 __init__ 中的属性，允许子类省略 super().__init__()
        instance.model = None

        return instance

    def __init__(self) -> None:
        """初始化."""
        pass

    def on_start(self) -> None:
        """
        策略启动时调用.

        在此处订阅数据 (self.subscribe) 或注册指标.
        """
        pass

    def on_stop(self) -> None:
        """
        策略停止时调用.

        在此处进行资源清理或结果统计.
        """
        pass

    def to_local_time(self, timestamp: int) -> pd.Timestamp:
        """
        将 UTC 纳秒时间戳转换为本地时间 (Timestamp).

        :param timestamp: UTC 纳秒时间戳 (int64)
        :return: 本地时间 (pd.Timestamp)
        """
        ts_utc = pd.to_datetime(timestamp, unit="ns", utc=True)
        return ts_utc.tz_convert(self.timezone)

    def format_time(self, timestamp: int, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        将 UTC 纳秒时间戳格式化为本地时间字符串.

        :param timestamp: UTC 纳秒时间戳 (int64)
        :param fmt: 时间格式字符串
        :return: 格式化后的时间字符串
        """
        return self.to_local_time(timestamp).strftime(fmt)

    @property
    def now(self) -> Optional[pd.Timestamp]:
        """
        获取当前回测时间的本地时间表示.

        如果当前没有 Bar 或 Tick，则返回 None.
        """
        ts = None
        if self.current_bar:
            ts = self.current_bar.timestamp
        elif self.current_tick:
            ts = self.current_tick.timestamp

        if ts is not None:
            return self.to_local_time(ts)
        return None

    def set_history_depth(self, depth: int) -> None:
        """
        设置历史数据回溯长度.

        :param depth: 保留的 Bar 数量 (0 表示不保留)
        """
        self._history_depth = depth

    def set_rolling_window(self, train_window: int, step: int) -> None:
        """
        设置滚动训练窗口参数.

        :param train_window: 训练数据长度 (Bars)
        :param step: 滚动步长 (每隔多少个 Bar 触发一次训练)
        """
        self._rolling_train_window = train_window
        self._rolling_step = step
        # 自动调整 history_depth 以满足训练窗口需求
        if self._history_depth < train_window:
            self._history_depth = train_window

    def get_history(
        self, count: int, symbol: Optional[str] = None, field: str = "close"
    ) -> np.ndarray:
        """
        获取历史数据 (类似 Zipline data.history).

        :param count: 获取的数据长度 (必须 <= history_depth)
        :param symbol: 标的代码 (默认当前 Bar 的 symbol)
        :param field: 字段名 (open, high, low, close, volume)
        :return: Numpy 数组
        """
        if self._history_depth == 0:
            raise RuntimeError(
                "History tracking is not enabled. Call set_history_depth() first."
            )

        if self.ctx is None:
            raise RuntimeError("Context not ready")

        symbol = self._resolve_symbol(symbol)

        # Call Rust implementation
        arr = self.ctx.history(symbol, field.lower(), count)

        if arr is None:
            return cast(np.ndarray, np.full(count, np.nan))

        if len(arr) < count:
            # Pad with NaN at the beginning
            padding = np.full(count - len(arr), np.nan)
            return cast(np.ndarray, np.concatenate((padding, arr)))

        return cast(np.ndarray, arr)

    def get_history_df(self, count: int, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        获取历史数据 DataFrame (Open, High, Low, Close, Volume).

        :param count: 数据长度
        :param symbol: 标的代码
        :return: pd.DataFrame
        """
        symbol = self._resolve_symbol(symbol)

        data = {
            "open": self.get_history(count, symbol, "open"),
            "high": self.get_history(count, symbol, "high"),
            "low": self.get_history(count, symbol, "low"),
            "close": self.get_history(count, symbol, "close"),
            "volume": self.get_history(count, symbol, "volume"),
        }
        return pd.DataFrame(data)

    def get_rolling_data(
        self, length: Optional[int] = None, symbol: Optional[str] = None
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        获取滚动训练数据.

        :param length: 数据长度 (默认使用 set_rolling_window 设置的 train_window)
        :param symbol: 标的代码
        :return: (X, y) 默认为 (DataFrame, None)
        """
        if length is None:
            length = self._rolling_train_window

        if length <= 0:
            raise ValueError("Invalid rolling window length")

        df = self.get_history_df(length, symbol)

        # 默认返回 raw DataFrame 作为 X，y 为 None
        # 用户可以在策略中重写此方法或自行处理数据
        return df, None

    def on_train_signal(self, context: Any) -> None:
        """
        滚动训练信号回调.

        默认实现：如果配置了 self.model，则自动执行数据准备和训练.

        :param context: 策略上下文 (通常是 self)
        """
        if self.model:
            try:
                X_df, _ = self.get_rolling_data()

                if (
                    self.model.validation_config
                    and self.model.validation_config.verbose
                ):
                    ts_str = ""
                    if self.current_bar:
                        ts_str = self.format_time(self.current_bar.timestamp)
                    print(
                        f"[{ts_str}] Auto-training triggered | Train Size: {len(X_df)}"
                    )

                X, y = self.prepare_features(X_df, mode="training")
                self.model.fit(X, y)
            except NotImplementedError:
                # User didn't implement prepare_features, assuming manual handling
                pass
            except Exception as e:
                print(f"Auto-training failed at bar {self._bar_count}: {e}")

    def prepare_features(
        self, df: pd.DataFrame, mode: str = "training"
    ) -> Tuple[Any, Any]:
        """
        Prepare features and labels for ML model.

        Must be implemented by user if using auto-training.

        :param df: Raw dataframe from get_rolling_data
        :param mode: "training" or "inference".
                     If "training", return (X, y).
                     If "inference", return (X_last_row, None) or just X.
        :return: (X, y)
        """
        raise NotImplementedError(
            "You must implement prepare_features(self, df, mode) for auto-training"
        )

    def _auto_configure_model(self) -> None:
        """Apply model validation configuration if present."""
        if self._model_configured:
            return

        if self.model and self.model.validation_config:
            cfg = self.model.validation_config

            try:
                train_window = parse_duration_to_bars(cfg.train_window, cfg.frequency)
                step = parse_duration_to_bars(cfg.rolling_step, cfg.frequency)

                # Update settings
                self.set_rolling_window(train_window, step)
            except Exception as e:
                print(f"Failed to configure model validation: {e}")

        self._model_configured = True

    def set_sizer(self, sizer: Sizer) -> None:
        """设置仓位管理器."""
        self.sizer = sizer

    def register_indicator(self, name: str, indicator: "Indicator") -> None:
        """
        Register an indicator.

        This allows accessing the indicator via self.name and ensures it is
        calculated before the backtest starts.
        """
        self._indicators.append(indicator)
        setattr(self, name, indicator)

    def subscribe(self, instrument_id: str) -> None:
        """
        Subscribe to market data for an instrument.

        :param instrument_id: The instrument identifier (e.g., '600000').
        """
        if instrument_id not in self._subscriptions:
            self._subscriptions.append(instrument_id)

    def _prepare_indicators(self, data: Dict[str, pd.DataFrame]) -> None:
        """Pre-calculate indicators."""
        if not self._indicators:
            return

        for ind in self._indicators:
            for sym, df in data.items():
                # Calculate and cache inside indicator
                ind(df, sym)

    def on_order(self, order: Any) -> None:
        """
        订单状态更新回调.

        Args:
            order: 订单对象
        """
        pass

    def on_trade(self, trade: Any) -> None:
        """
        成交回调.

        Args:
            trade: 成交对象
        """
        pass

    def _check_order_events(self) -> None:
        """检查订单和成交事件并触发回调."""
        if self.ctx is None:
            return

        # 1. Process New Trades (from Rust Engine)
        if hasattr(self.ctx, "recent_trades"):
            for trade in self.ctx.recent_trades:
                self.on_trade(trade)

                # Update known order status if we have it
                if trade.order_id in self._known_orders:
                    pass

        # 2. Process Canceled Orders (from Rust Engine)
        if hasattr(self.ctx, "canceled_order_ids"):
            for oid in self.ctx.canceled_order_ids:
                if oid in self._known_orders:
                    order = self._known_orders[oid]
                    try:
                        order.status = OrderStatus.Cancelled
                    except Exception:
                        pass

                    self.on_order(order)
                    del self._known_orders[oid]

        # 3. Process Active Orders (New & Status Changes)
        current_active_ids = set()
        if hasattr(self.ctx, "active_orders"):
            for order in self.ctx.active_orders:
                current_active_ids.add(order.id)
                oid = order.id

                # New Order or Status Change
                if oid not in self._known_orders:
                    self._known_orders[oid] = order
                    self.on_order(order)
                else:
                    known = self._known_orders[oid]
                    status_changed = known.status != order.status
                    qty_changed = known.filled_quantity != order.filled_quantity
                    if status_changed or qty_changed:
                        self._known_orders[oid] = order
                        self.on_order(order)

        # 4. Cleanup Disappeared Orders (Filled?)
        recent_trade_order_ids = set()
        if hasattr(self.ctx, "recent_trades"):
            for t in self.ctx.recent_trades:
                recent_trade_order_ids.add(t.order_id)

        for oid in list(self._known_orders.keys()):
            if oid not in current_active_ids:
                # It disappeared. Was it canceled? (Handled in step 2)
                # Is it Filled?
                if oid in recent_trade_order_ids:
                    order = self._known_orders[oid]
                    try:
                        order.status = OrderStatus.Filled
                    except Exception:
                        pass
                    self.on_order(order)
                    del self._known_orders[oid]
                else:
                    # Disappeared but not canceled and no trade?
                    del self._known_orders[oid]

        # 5. Process Trades
        if hasattr(self.ctx, "recent_trades"):
            for t in self.ctx.recent_trades:
                self.on_trade(t)

    def _on_bar_event(self, bar: Bar, ctx: StrategyContext) -> None:
        """引擎调用的 Bar 回调 (Internal)."""
        self.ctx = ctx
        self._last_event_type = "bar"

        self._check_order_events()

        # Lazy configuration
        if not self._model_configured:
            self._auto_configure_model()

        self.current_bar = bar
        self._last_prices[bar.symbol] = bar.close

        # 检查滚动训练信号
        if self._rolling_step > 0:
            self._bar_count += 1
            if self._bar_count % self._rolling_step == 0:
                # 触发训练信号，传入 self 作为 context
                self.on_train_signal(self)

        self.on_bar(bar)

    def _on_tick_event(self, tick: Tick, ctx: StrategyContext) -> None:
        """引擎调用的 Tick 回调 (Internal)."""
        self.ctx = ctx
        self._last_event_type = "tick"
        self.current_tick = tick
        self._last_prices[tick.symbol] = tick.price
        self.on_tick(tick)

    def _on_timer_event(self, payload: str, ctx: StrategyContext) -> None:
        """引擎调用的 Timer 回调 (Internal)."""
        self.ctx = ctx
        self.on_timer(payload)

    def on_bar(self, bar: Bar) -> None:
        """
        策略逻辑入口 (Bar 数据).

        用户应重写此方法.
        """
        pass

    @property
    def position(self) -> Position:
        """
        获取当前处理标的的持仓对象.

        支持常见的策略编写语法:
        if self.position.size == 0:
            ...
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")

        symbol = self._resolve_symbol(None)
        return Position(self.ctx, symbol)

    def on_tick(self, tick: Tick) -> None:
        """
        策略逻辑入口 (Tick 数据).

        用户应重写此方法.
        """
        pass

    def on_timer(self, payload: str) -> None:
        """
        策略逻辑入口 (Timer 事件).

        用户应重写此方法.
        """
        pass

    def _resolve_symbol(self, symbol: Optional[str] = None) -> str:
        if symbol is None:
            # Prioritize based on the last event type
            if self._last_event_type == "tick" and self.current_tick:
                symbol = self.current_tick.symbol
            elif self._last_event_type == "bar" and self.current_bar:
                symbol = self.current_bar.symbol
            # Fallbacks
            elif self.current_bar:
                symbol = self.current_bar.symbol
            elif self.current_tick:
                symbol = self.current_tick.symbol
            else:
                raise ValueError("Symbol must be provided")
        return symbol

    def get_position(self, symbol: Optional[str] = None) -> float:
        """
        获取指定标的的持仓数量.

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)

        Returns:
            float: 持仓数量 (正数为多头, 负数为空头)
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")

        symbol = self._resolve_symbol(symbol)
        return self.ctx.get_position(symbol)

    def get_positions(self) -> Dict[str, float]:
        """
        获取所有持仓信息.

        Returns:
            Dict[str, float]: 持仓字典 {symbol: quantity}
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")
        return self.ctx.positions

    def get_open_orders(self, symbol: Optional[str] = None) -> list[Any]:
        """
        获取当前未完成的订单.

        Args:
            symbol: 标的代码 (如果为 None，返回所有标的订单)

        Returns:
            List[Order]: 订单列表
        """
        if self.ctx is None:
            return []

        orders = [
            o
            for o in self.ctx.active_orders
            if o.status in (OrderStatus.New, OrderStatus.Submitted)
        ]
        if symbol:
            return [o for o in orders if o.symbol == symbol]
        return orders

    def buy(
        self,
        symbol: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        trigger_price: Optional[float] = None,
    ) -> str:
        """
        买入下单.

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填, 使用 Sizer 计算)
            price: 限价 (None 为市价)
            time_in_force: 订单有效期
            trigger_price: 触发价 (止损/止盈)

        Returns:
            str: 订单 ID
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")

        # 1. Determine Symbol
        symbol = self._resolve_symbol(symbol)

        # 2. Determine Reference Price for Sizing
        ref_price = price
        if ref_price is None:
            ref_price = self._last_prices.get(symbol, 0.0)

        # 3. Determine Quantity via Sizer
        if quantity is None:
            quantity = self.sizer.get_size(ref_price, self.ctx.cash, self.ctx, symbol)

        # 4. Execute Buy
        if quantity > 0:
            return self.ctx.buy(symbol, quantity, price, time_in_force, trigger_price)
        return ""

    def sell(
        self,
        symbol: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        trigger_price: Optional[float] = None,
    ) -> str:
        """
        卖出下单.

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填, 默认卖出当前标的所有持仓)
            price: 限价 (None 为市价)
            time_in_force: 订单有效期
            trigger_price: 触发价 (止损/止盈)

        Returns:
            str: 订单 ID
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")

        # 1. Determine Symbol
        symbol = self._resolve_symbol(symbol)

        # 2. Determine Quantity (Default to Close Position if None)
        if quantity is None:
            # Default to closing the entire position for this symbol
            pos = self.ctx.get_position(symbol)
            if pos > 0:
                quantity = pos
            else:
                # If no position, maybe use Sizer?
                # For now, if no position and no quantity, we can't sell.
                return ""

        # 3. Execute Sell
        if quantity > 0:
            return self.ctx.sell(symbol, quantity, price, time_in_force, trigger_price)
        return ""

    def stop_buy(
        self,
        symbol: Optional[str] = None,
        trigger_price: float = 0.0,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
    ) -> None:
        """
        发送止损买入单 (Stop Buy Order).

        当市价上涨突破 trigger_price 时触发买入.
        - 如果 price 为 None, 触发后转为市价单 (Stop Market).
        - 如果 price 不为 None, 触发后转为限价单 (Stop Limit).
        """
        self.buy(symbol, quantity, price, time_in_force, trigger_price=trigger_price)

    def stop_sell(
        self,
        symbol: Optional[str] = None,
        trigger_price: float = 0.0,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
    ) -> None:
        """
        发送止损卖出单 (Stop Sell Order).

        当市价下跌跌破 trigger_price 时触发卖出.
        - 如果 price 为 None, 触发后转为市价单 (Stop Market).
        - 如果 price 不为 None, 触发后转为限价单 (Stop Limit).
        """
        self.sell(symbol, quantity, price, time_in_force, trigger_price=trigger_price)

    def get_portfolio_value(self) -> float:
        """计算当前投资组合总价值 (现金 + 持仓市值)."""
        if self.ctx is None:
            return 0.0

        total_value = float(self.ctx.cash)

        for symbol, qty in self.ctx.positions.items():
            if qty == 0:
                continue

            # 使用最新价格计算市值
            price = self._last_prices.get(symbol, 0.0)
            # 如果没有最新价格，尝试使用当前 bar/tick
            if price == 0.0:
                if self.current_bar and self.current_bar.symbol == symbol:
                    price = self.current_bar.close
                elif self.current_tick and self.current_tick.symbol == symbol:
                    price = self.current_tick.price

            total_value += float(qty) * price

        return total_value

    def order_target(
        self,
        target: float,
        symbol: Optional[str] = None,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        调整仓位到目标数量.

        :param target: 目标持仓数量 (例如 100, -100)
        :param symbol: 标的代码
        :param price: 限价 (可选)
        :param kwargs: 其他下单参数
        """
        symbol = self._resolve_symbol(symbol)

        current_qty = 0.0
        if self.ctx:
            current_qty = float(self.ctx.get_position(symbol))

        delta_qty = target - current_qty

        if delta_qty > 0:
            self.buy(symbol, delta_qty, price, **kwargs)
        elif delta_qty < 0:
            self.sell(symbol, abs(delta_qty), price, **kwargs)

    def order_target_value(
        self,
        target_value: float,
        symbol: Optional[str] = None,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        调整仓位到目标价值.

        :param target_value: 目标持仓市值
        :param symbol: 标的代码
        :param price: 限价 (可选)
        :param kwargs: 其他下单参数
        """
        symbol = self._resolve_symbol(symbol)

        # 获取当前价格
        current_price = self._last_prices.get(symbol, 0.0)
        if current_price == 0.0:
            if self.current_bar and self.current_bar.symbol == symbol:
                current_price = self.current_bar.close
            elif self.current_tick and self.current_tick.symbol == symbol:
                current_price = self.current_tick.price
            else:
                # 无法获取价格，无法计算数量
                print(
                    f"Warning: Cannot determine price for {symbol}, "
                    "skipping order_target_value"
                )
                return

        # 获取当前持仓
        current_qty = 0.0
        if self.ctx:
            current_qty = float(self.ctx.get_position(symbol))

        # 计算目标数量
        target_qty = target_value / current_price
        delta_qty = target_qty - current_qty

        # 下单
        if delta_qty > 0:
            self.buy(symbol, delta_qty, price, **kwargs)
        elif delta_qty < 0:
            self.sell(symbol, abs(delta_qty), price, **kwargs)

    def order_target_percent(
        self,
        target_percent: float,
        symbol: Optional[str] = None,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        调整仓位到目标百分比.

        :param target_percent: 目标持仓比例 (0.5 = 50%)
        :param symbol: 标的代码
        :param price: 限价 (可选)
        :param kwargs: 其他下单参数
        """
        portfolio_value = self.get_portfolio_value()
        target_value = portfolio_value * target_percent
        self.order_target_value(target_value, symbol, price, **kwargs)

    def cancel_order(self, order_or_id: Any) -> None:
        """
        取消订单.

        Args:
            order_or_id: 订单对象或订单 ID
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")

        order_id = order_or_id
        if hasattr(order_or_id, "id"):
            order_id = order_or_id.id

        self.ctx.cancel_order(order_id)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> None:
        """
        取消所有未完成订单.

        Args:
            symbol: 标的代码 (如果为 None, 取消所有标的订单)
        """
        for order in self.get_open_orders(symbol):
            self.cancel_order(order)

    def buy_all(self, symbol: Optional[str] = None) -> None:
        """
        全仓买入 (Buy All).

        使用当前所有可用资金买入.

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")

        symbol = self._resolve_symbol(symbol)

        # 获取参考价格
        price = 0.0
        if self.current_bar and self.current_bar.symbol == symbol:
            price = self.current_bar.close
        elif self.current_tick and self.current_tick.symbol == symbol:
            price = self.current_tick.price

        if price <= 0:
            # 无法获取价格，无法计算数量
            # 这里可以选择记录日志或抛出警告，暂时直接返回
            return

        cash = self.ctx.cash
        # 计算最大可买数量 (向下取整)
        # 注意：这里未扣除预估手续费，如果资金刚好卡在边界，可能会因为手续费导致拒单
        # 建议引擎层或用户预留 buffer，或者在这里 * 0.99
        quantity = int(cash / price)

        if quantity > 0:
            self.buy(symbol=symbol, quantity=quantity)

    def close_position(self, symbol: Optional[str] = None) -> None:
        """
        平仓 (Close Position).

        卖出/买入以抵消当前持仓.

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)
        """
        symbol = self._resolve_symbol(symbol)
        position = self.get_position(symbol)

        if position > 0:
            self.sell(symbol=symbol, quantity=position)
        elif position < 0:
            self.buy(symbol=symbol, quantity=abs(position))

    def short(
        self,
        symbol: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        trigger_price: Optional[float] = None,
    ) -> None:
        """
        卖出开空 (Short Sell).

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填, 使用 Sizer 计算)
            price: 限价 (None 为市价)
            time_in_force: 订单有效期
            trigger_price: 触发价 (止损/止盈)
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")

        # 1. Determine Symbol
        symbol = self._resolve_symbol(symbol)

        # 2. Determine Reference Price for Sizing
        ref_price = price
        if ref_price is None:
            if self.current_bar:
                ref_price = self.current_bar.close
            elif self.current_tick:
                ref_price = self.current_tick.price
            else:
                ref_price = 0.0

        # 3. Determine Quantity via Sizer
        if quantity is None:
            quantity = self.sizer.get_size(ref_price, self.ctx.cash, self.ctx, symbol)

        # 4. Execute Sell (Short)
        if quantity > 0:
            self.ctx.sell(symbol, quantity, price, time_in_force, trigger_price)

    def cover(
        self,
        symbol: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        trigger_price: Optional[float] = None,
    ) -> None:
        """
        买入平空 (Buy to Cover).

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填, 默认平掉当前标的所有空头持仓)
            price: 限价 (None 为市价)
            time_in_force: 订单有效期
            trigger_price: 触发价 (止损/止盈)
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")

        # 1. Determine Symbol
        symbol = self._resolve_symbol(symbol)

        # 2. Determine Quantity (Default to Close Short Position if None)
        if quantity is None:
            pos = self.ctx.get_position(symbol)
            if pos < 0:
                quantity = abs(pos)
            else:
                # No short position to cover
                return

        # 3. Execute Buy (Cover)
        if quantity > 0:
            self.ctx.buy(symbol, quantity, price, time_in_force, trigger_price)

    def schedule(self, timestamp: int, payload: str) -> None:
        """
        注册定时事件.

        Args:
            timestamp: 触发时间戳 (Unix 纳秒)
            payload: 事件携带的数据
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")
        self.ctx.schedule(timestamp, payload)

    def get_cash(self) -> float:
        """获取现金."""
        if self.ctx is None:
            return 0.0
        return self.ctx.cash


class VectorizedStrategy(Strategy):
    """
    向量化策略基类 (Vectorized Strategy Base Class).

    支持预计算指标的高速回测模式.
    用户应在回测前使用 Pandas/Numpy 计算好所有指标,
    然后通过本类提供的高速游标访问机制在 on_bar 中读取.
    """

    def __init__(self, precalculated_data: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Initialize VectorizedStrategy.

        :param precalculated_data: 预计算数据字典
                                  Structure: {symbol: {indicator_name: numpy_array}}
        """
        super().__init__()
        self.precalc = precalculated_data
        # 游标管理: {symbol: index}
        self.cursors: defaultdict[str, int] = defaultdict(int)

        # 默认禁用 Python 侧历史数据缓存以提升性能
        self.set_history_depth(0)

    def _on_bar_event(self, bar: Bar, ctx: StrategyContext) -> None:
        """Wrap the user on_bar handler internally."""
        # 1. Call standard setup (ctx, current_bar, history)
        # Note: We copy logic from Strategy._on_bar_event to avoid double calling on_bar
        # if we just called super()._on_bar_event(bar, ctx).
        # Actually Strategy._on_bar_event calls self.on_bar(bar).

        self.ctx = ctx
        self.current_bar = bar

        # 2. Call User Strategy
        self.on_bar(bar)

        # 3. Increment Cursor
        self.cursors[bar.symbol] += 1

    def get_value(self, name: str, symbol: Optional[str] = None) -> float:
        """
        获取当前 Bar 对应的预计算指标值.

        Args:
            name: 指标名称
            symbol: 标的代码 (如果不填, 默认使用当前 Bar 的 symbol)

        Returns:
            指标值 (float). 如果不存在或越界，返回 nan.
        """
        symbol = self._resolve_symbol(symbol)
        idx = self.cursors[symbol]

        try:
            return float(self.precalc[symbol][name][idx])
        except (KeyError, IndexError):
            return float("nan")
