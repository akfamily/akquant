from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from .akquant import Bar, OrderStatus, StrategyContext, Tick, TimeInForce
from .sizer import FixedSize, Sizer

if TYPE_CHECKING:
    from .indicator import Indicator


class Strategy:
    """
    策略基类 (Base Strategy Class).

    采用类似 NautilusTrader 的事件驱动设计
    """

    ctx: Optional[StrategyContext]
    sizer: Sizer
    current_bar: Optional[Bar]
    current_tick: Optional[Tick]
    _history_depth: int
    _bars_history: "defaultdict[str, deque[Bar]]"
    _indicators: List["Indicator"]
    _subscriptions: List[str]

    def __new__(cls, *args: Any, **kwargs: Any) -> "Strategy":
        """Create a new Strategy instance."""
        instance = super().__new__(cls)
        instance.ctx = None
        instance.sizer = FixedSize(100)
        instance.current_bar = None
        instance.current_tick = None
        instance._indicators = []
        instance._subscriptions = []

        # 历史数据存储
        instance._history_depth = 0
        instance._bars_history = defaultdict(
            lambda: deque(maxlen=max(1, instance._history_depth))
        )
        return instance

    def __init__(self) -> None:
        """Initialize the strategy."""
        pass

    def set_history_depth(self, depth: int) -> None:
        """
        设置历史数据回溯长度.

        :param depth: 保留的 Bar 数量 (0 表示不保留)
        """
        self._history_depth = depth
        if depth > 0:
            # 如果已有数据，需要调整 maxlen (通过重新创建 deque)
            # 注意: 这会清空现有历史，通常只在初始化时调用
            self._bars_history: defaultdict[str, deque[Bar]] = defaultdict(
                lambda: deque(maxlen=depth)
            )
        else:
            self._bars_history.clear()

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

        symbol = self._resolve_symbol(symbol)
        history = self._bars_history[symbol]

        if len(history) < count:
            # 数据不足时返回 NaN 填充的数组
            return cast(np.ndarray, np.full(count, np.nan))

        # 获取最近的 count 个 Bar
        # Optimization: Avoid copying the entire deque to a list
        hist_len = len(history)
        start_idx = hist_len - count
        bars = [history[i] for i in range(start_idx, hist_len)]

        # 提取字段
        return cast(np.ndarray, np.array([getattr(b, field) for b in bars]))

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

    def on_start(self) -> None:
        """
        Start the strategy.

        Use this to subscribe to data or initialize resources.
        """
        pass

    def _prepare_indicators(self, data: Dict[str, pd.DataFrame]) -> None:
        """Pre-calculate indicators."""
        if not self._indicators:
            return

        for ind in self._indicators:
            for sym, df in data.items():
                # Calculate and cache inside indicator
                ind(df, sym)

    def _on_bar_event(self, bar: Bar, ctx: StrategyContext) -> None:
        """引擎调用的 Bar 回调 (Internal)."""
        self.ctx = ctx
        self.current_bar = bar

        # 自动维护历史数据
        if self._history_depth > 0:
            self._bars_history[bar.symbol].append(bar)

        self.on_bar(bar)

    def _on_tick_event(self, tick: Tick, ctx: StrategyContext) -> None:
        """引擎调用的 Tick 回调 (Internal)."""
        self.ctx = ctx
        self.current_tick = tick
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

    def on_tick(self, tick: Tick) -> None:
        """
        策略逻辑入口 (Tick 数据).

        用户应重写此方法.
        """
        pass

    def on_timer(self, payload: str) -> None:
        """
        策略逻辑入口 (Timer 事件).

        Args:
            payload: 定时器携带的数据
        """
        pass

    def _resolve_symbol(self, symbol: Optional[str] = None) -> str:
        if symbol is None:
            if self.current_bar:
                symbol = self.current_bar.symbol
            elif self.current_tick:
                symbol = self.current_tick.symbol
            else:
                raise ValueError("Symbol must be provided")
        return symbol

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
            symbol: 标的代码 (如果为 None，取消所有标的订单)
        """
        for order in self.get_open_orders(symbol):
            self.cancel_order(order)

    def buy_all(self, symbol: Optional[str] = None) -> None:
        """
        全仓买入 (Buy All).

        使用当前所有可用资金买入.

        Args:
            symbol: 标的代码 (如果不填，默认使用当前 Bar/Tick 的 symbol)
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
            symbol: 标的代码 (如果不填，默认使用当前 Bar/Tick 的 symbol)
        """
        symbol = self._resolve_symbol(symbol)
        position = self.get_position(symbol)

        if position > 0:
            self.sell(symbol=symbol, quantity=position)
        elif position < 0:
            self.buy(symbol=symbol, quantity=abs(position))

    def buy(
        self,
        symbol: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        trigger_price: Optional[float] = None,
    ) -> None:
        """
        买入下单.

        Args:
            symbol: 标的代码 (如果不填，默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填，使用 Sizer 计算)
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

        # 4. Execute Buy
        if quantity > 0:
            self.ctx.buy(symbol, quantity, price, time_in_force, trigger_price)

    def sell(
        self,
        symbol: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        trigger_price: Optional[float] = None,
    ) -> None:
        """
        卖出下单.

        Args:
            symbol: 标的代码 (如果不填，默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填，默认卖出当前标的所有持仓)
            price: 限价 (None 为市价)
            time_in_force: 订单有效期
            trigger_price: 触发价 (止损/止盈)
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")

        # 1. Determine Symbol
        if symbol is None:
            if self.current_bar:
                symbol = self.current_bar.symbol
            elif self.current_tick:
                symbol = self.current_tick.symbol
            else:
                raise ValueError("Symbol must be provided")

        # 2. Determine Quantity (Default to Close Position if None)
        if quantity is None:
            # Default to closing the entire position for this symbol
            pos = self.ctx.get_position(symbol)
            if pos > 0:
                quantity = pos
            else:
                # If no position, maybe use Sizer?
                # For now, if no position and no quantity, we can't sell.
                return

        # 3. Execute Sell
        if quantity > 0:
            self.ctx.sell(symbol, quantity, price, time_in_force, trigger_price)

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
            symbol: 标的代码 (如果不填，默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填，使用 Sizer 计算)
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
            symbol: 标的代码 (如果不填，默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填，默认平掉当前标的所有空头持仓)
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

        当市价上涨突破 trigger_price 时触发买入。
        - 如果 price 为 None，触发后转为市价单 (Stop Market)。
        - 如果 price 不为 None，触发后转为限价单 (Stop Limit)。
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

        当市价下跌跌破 trigger_price 时触发卖出。
        - 如果 price 为 None，触发后转为市价单 (Stop Market)。
        - 如果 price 不为 None，触发后转为限价单 (Stop Limit)。
        """
        self.sell(symbol, quantity, price, time_in_force, trigger_price=trigger_price)

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

    def get_position(self, symbol: Optional[str] = None) -> float:
        """获取当前持仓数量."""
        if self.ctx is None:
            return 0.0

        if symbol is None:
            if self.current_bar:
                symbol = self.current_bar.symbol
            elif self.current_tick:
                symbol = self.current_tick.symbol
            else:
                return 0.0
        return self.ctx.get_position(symbol)

    def get_cash(self) -> float:
        """获取现金."""
        if self.ctx is None:
            return 0.0
        return self.ctx.cash


class VectorizedStrategy(Strategy):
    """
    向量化策略基类 (Vectorized Strategy Base Class).

    支持预计算指标的高速回测模式。
    用户应在回测前使用 Pandas/Numpy 计算好所有指标，
    然后通过本类提供的高速游标访问机制在 on_bar 中读取。
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

        # Skip history maintenance if depth is 0 (default for VectorizedStrategy)
        if self._history_depth > 0:
            self._bars_history[bar.symbol].append(bar)

        # 2. Call User Strategy
        self.on_bar(bar)

        # 3. Increment Cursor
        self.cursors[bar.symbol] += 1

    def get_value(self, indicator_name: str, symbol: Optional[str] = None) -> float:
        """
        获取当前 Bar 对应的预计算指标值 (O(1) Access).

        :param indicator_name: 指标名称 (key in precalculated_data)
        :param symbol: 标的代码 (默认当前 Bar symbol)
        :return: 指标值 (float) 或 NaN
        """
        if symbol is None:
            if self.current_bar is None:
                return np.nan
            sym = self.current_bar.symbol
        else:
            sym = symbol

        idx = self.cursors[sym]

        try:
            return float(self.precalc[sym][indicator_name][idx])
        except (KeyError, IndexError):
            # 越界或键不存在
            return np.nan
