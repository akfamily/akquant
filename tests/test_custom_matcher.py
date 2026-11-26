"""Test custom matcher functionality."""

import logging
from typing import Any, List, Optional, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import (
    AssetType,
    Bar,
    Instrument,
    Order,
    OrderSide,
    OrderStatus,
    Strategy,
    Trade,
)


class CustomStockMatcher:
    """Custom stock matcher for testing."""

    def match(
        self,
        order: Order,
        event: Bar,
        _instrument: Instrument,
        bar_index: int,
    ) -> Optional[Trade]:
        """Match order at the event close price and return a Trade or None."""
        if not hasattr(event, "close"):
            return None

        # Simple match logic: fill at close price
        fill_price = event.close

        # Update order status
        order.status = OrderStatus.Filled
        order.filled_quantity = order.quantity
        order.average_filled_price = fill_price

        # Create trade
        trade = Trade(
            id=f"trade_{order.id}",
            order_id=order.id,
            symbol=order.symbol,
            price=fill_price,
            quantity=order.quantity,
            side=order.side,
            timestamp=event.timestamp,
            commission=0.0,
            bar_index=bar_index,
            owner_strategy_id=None,
        )
        print(
            f"[CustomMatcher] Filled order {order.id} at {fill_price}, "
            f"bar_index={bar_index}"
        )
        return trade


class RaisingStockMatcher:
    """Matcher that raises to test Rust->Python execution warning bridging."""

    def match(
        self,
        order: Order,
        _event: Bar,
        _instrument: Instrument,
        bar_index: int,
    ) -> Optional[Trade]:
        """Raise a matcher error."""
        raise RuntimeError(f"matcher boom for {order.symbol} at {bar_index}")


class InvalidResultStockMatcher:
    """Matcher that returns an invalid payload for logging regression tests."""

    def match(
        self,
        order: Order,
        _event: Bar,
        _instrument: Instrument,
        bar_index: int,
    ) -> object:
        """Return a non-Trade object to trigger the Rust warning path."""
        return {"symbol": order.symbol, "bar_index": bar_index}


def _make_data() -> pd.DataFrame:
    """Build deterministic stock bars for matcher testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "volume": 1000,
            "symbol": "TEST",
        },
        index=dates,
    )


# 3. Setup Strategy
class MatcherStrategy(Strategy):
    """Minimal strategy to trigger buy/sell for matcher testing."""

    def on_start(self) -> None:
        """Initialize internal counter."""
        self.count = 0

    def on_bar(self, bar: Bar) -> None:
        """Place orders on specific bars to exercise the matcher."""
        # Buy on first bar, Sell on second bar (bar index 0 and 1)
        # Note: on_bar is called for each bar.
        # Bar 0: Buy. Match on Bar 1.
        # Bar 1: Position is 100. Sell. Match on Bar 2.

        if self.count == 0:
            print(f"Sending Buy Order at {bar.timestamp}")
            self.buy(self.symbol, 100)
        elif self.count == 2:
            print(f"Sending Sell Order at {bar.timestamp}")
            self.sell(self.symbol, 100)

        self.count += 1


def test_custom_matcher_generates_closed_trade() -> None:
    """Custom matcher should fill orders and produce a closed trade."""
    result = aq.run_backtest(
        strategy=MatcherStrategy,
        data=_make_data(),
        symbols="TEST",
        initial_cash=100000,
        custom_matchers={AssetType.Stock: CustomStockMatcher()},
        show_progress=False,
    )

    assert not result.trades_df.empty
    trade = result.trades_df.iloc[0]
    assert trade["symbol"] == "TEST"
    assert trade["quantity"] == 100.0
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 100.0


def test_custom_matcher_exception_bridges_rust_warning(caplog: Any) -> None:
    """Matcher exceptions should enter the structured execution logging pipeline."""
    aq.configure_logging(aq.LogConfig(console=False, level="WARNING"))

    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = aq.run_backtest(
            strategy=MatcherStrategy,
            data=_make_data(),
            symbols="TEST",
            initial_cash=100000,
            custom_matchers={AssetType.Stock: RaisingStockMatcher()},
            show_progress=False,
        )

    assert result is not None
    matching_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.execution.python"
        and "Custom Python matcher raised an exception: RuntimeError: matcher boom"
        in record.getMessage()
    )
    assert matching_record.phase == "execution"
    assert matching_record.symbol == "TEST"
    assert getattr(matching_record, "order_id", None)
    # 只有日期的 bar 改写到收盘：第二根 bar 为 2024-01-02 15:00（UTC 07:00）
    assert matching_record.event_time_iso.startswith("2024-01-02")

    aq.register_logger(console=False, level="INFO")


def test_custom_matcher_invalid_result_bridges_rust_warning(caplog: Any) -> None:
    """Matcher invalid return values should enter structured execution logging."""
    aq.configure_logging(aq.LogConfig(console=False, level="WARNING"))

    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = aq.run_backtest(
            strategy=MatcherStrategy,
            data=_make_data(),
            symbols="TEST",
            initial_cash=100000,
            custom_matchers={AssetType.Stock: InvalidResultStockMatcher()},
            show_progress=False,
        )

    assert result is not None
    matching_record = next(
        record
        for record in caplog.records
        if record.name == "akquant.execution.python"
        and "Ignored custom Python matcher result because it is not a Trade"
        in record.getMessage()
    )
    assert matching_record.phase == "execution"
    assert matching_record.symbol == "TEST"
    assert getattr(matching_record, "order_id", None)
    # 只有日期的 bar 改写到收盘：第二根 bar 为 2024-01-02 15:00（UTC 07:00）
    assert matching_record.event_time_iso.startswith("2024-01-02")

    aq.register_logger(console=False, level="INFO")


# ---------------------------------------------------------------------------
# #458: 撮合器只返回 Trade、不改订单时, 由引擎按成交了结订单
# ---------------------------------------------------------------------------

_SYMBOL = "000001.SZ"
_PRICES = [10.0, 11.0, 12.0, 12.5]


def _issue_458_data() -> pd.DataFrame:
    """四根日线, 开高低收相同: 2024-01-02..05 的 10/11/12/12.5."""
    idx = pd.to_datetime(
        [
            "2024-01-02 15:00",
            "2024-01-03 15:00",
            "2024-01-04 15:00",
            "2024-01-05 15:00",
        ]
    ).tz_localize("Asia/Shanghai")
    return pd.DataFrame(
        {
            "open": _PRICES,
            "high": _PRICES,
            "low": _PRICES,
            "close": _PRICES,
            "volume": [1000.0] * 4,
        },
        index=idx,
    )


class _CountingMatcher:
    """按开盘价成交并记下调用次数; 子类决定成交量与是否改订单."""

    def __init__(self) -> None:
        self.calls = 0

    def fill_quantity(self, order: Order) -> float:
        """本次成交量: 缺省为订单全部委托量."""
        return float(order.quantity)

    def make_trade(self, order: Order, event: Bar, bar_index: int) -> Trade:
        """按开盘价构造成交, 手续费字段故意填 5.0."""
        return Trade(
            f"t{self.calls}",
            str(order.id),
            str(order.symbol),
            order.side,
            self.fill_quantity(order),
            float(event.open),
            5.0,
            int(event.timestamp),
            int(bar_index),
        )

    def match(
        self, order: Order, event: Bar, _instrument: Instrument, bar_index: int
    ) -> Optional[Trade]:
        """每次被调用都成交, 不改订单."""
        self.calls += 1
        return self.make_trade(order, event, bar_index)


class _HalfMatcher(_CountingMatcher):
    """每次只成交委托量的一半."""

    def fill_quantity(self, order: Order) -> float:
        """一半委托量."""
        return float(order.quantity) / 2


class _MutatingMatcher(_CountingMatcher):
    """旧写法: 自己就地把订单改成全部成交, 再返回 Trade."""

    def match(
        self, order: Order, event: Bar, _instrument: Instrument, bar_index: int
    ) -> Optional[Trade]:
        """就地改订单后返回成交."""
        self.calls += 1
        order.status = OrderStatus.Filled
        order.filled_quantity = order.quantity
        order.average_filled_price = event.open
        return self.make_trade(order, event, bar_index)


class _NoneMatcher(_CountingMatcher):
    """从不成交."""

    def match(
        self, order: Order, event: Bar, _instrument: Instrument, bar_index: int
    ) -> Optional[Trade]:
        """返回 None."""
        self.calls += 1
        return None


class _BadTradeMatcher(_CountingMatcher):
    """返回不合法的成交: 数量超过剩余、数量为 0、订单号、品种或方向不符."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def match(
        self, order: Order, event: Bar, _instrument: Instrument, bar_index: int
    ) -> Optional[Trade]:
        """按 kind 构造不合法的成交."""
        self.calls += 1
        quantity = {"too_much": float(order.quantity) + 100, "zero": 0.0}.get(
            self.kind, float(order.quantity)
        )
        order_id = "not-this-order" if self.kind == "order_id" else str(order.id)
        symbol = "600000.SH" if self.kind == "symbol" else str(order.symbol)
        side = order.side
        if self.kind == "side":
            side = OrderSide.Sell if order.side == OrderSide.Buy else OrderSide.Buy
        return Trade(
            f"t{self.calls}",
            order_id,
            symbol,
            side,
            quantity,
            float(event.open),
            0.0,
            int(event.timestamp),
            int(bar_index),
        )


class _BuyOnFirstBar(Strategy):
    """第一根 K 线买 100 股, 记下每次订单回调的状态与已成交量."""

    def __init__(self) -> None:
        super().__init__()
        self.statuses: List[Tuple[str, float]] = []

    def on_bar(self, bar: Bar) -> None:
        """只在 2024-01-02 下单."""
        if bar.timestamp_iso.startswith("2024-01-02"):
            self.buy(bar.symbol, 100)

    def on_order(self, order: Any) -> None:
        """记下订单回调."""
        self.statuses.append((str(order.status), float(order.filled_quantity)))


def _run_issue_458(matcher: Any) -> Any:
    return aq.run_backtest(
        data={_SYMBOL: _issue_458_data()},
        strategy=_BuyOnFirstBar,
        symbols=_SYMBOL,
        initial_cash=100000.0,
        freq="1d",
        fill_policy=aq.NextOpen(),
        commission_rate=0.001,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        custom_matchers={AssetType.Stock: matcher},
        show_progress=False,
    )


def _final_position(result: Any) -> float:
    return float(result.engine.portfolio.positions.get(_SYMBOL, 0.0))


def _assert_single_full_fill(result: Any, matcher: _CountingMatcher) -> None:
    orders = result.orders_df
    assert len(orders) == 1
    order = orders.iloc[0]
    assert order["status"] == "filled"
    assert float(order["filled_quantity"]) == 100.0
    assert float(order["avg_price"]) == pytest.approx(11.0)
    assert matcher.calls == 1
    assert _final_position(result) == 100.0
    executions = result.executions_df
    assert len(executions) == 1
    assert float(executions.iloc[0]["quantity"]) == 100.0
    assert float(executions.iloc[0]["price"]) == 11.0
    # 撮合器返回的 commission(5.0)不被采用: 入账时引擎按费率重算 100*11*0.001
    assert float(executions.iloc[0]["commission"]) == pytest.approx(1.1)
    assert float(result.engine.portfolio.cash) == pytest.approx(
        100000.0 - 100 * 11.0 - 1.1
    )


def test_matcher_returning_trade_only_closes_the_order() -> None:
    """#458: 只返回 Trade 的撮合器, 订单由引擎记成成交, 不再被反复撮合."""
    matcher = _CountingMatcher()
    result = _run_issue_458(matcher)
    _assert_single_full_fill(result, matcher)


def test_matcher_partial_fills_keep_order_working_until_filled() -> None:
    """每次成交一半: 第一次后部分成交、留在队列, 第二次后成交完."""
    matcher = _HalfMatcher()
    result = _run_issue_458(matcher)

    fills = [status for status in result.strategy.statuses if status[1] > 0]
    assert fills == [
        ("OrderStatus.PartiallyFilled", 50.0),
        ("OrderStatus.Filled", 100.0),
    ]
    assert matcher.calls == 2
    order = result.orders_df.iloc[0]
    assert order["status"] == "filled"
    assert float(order["filled_quantity"]) == 100.0
    # 第一次 50 股 @11, 第二次 50 股 @12
    assert float(order["avg_price"]) == pytest.approx(11.5)
    assert _final_position(result) == 100.0
    assert [float(p) for p in result.executions_df["price"]] == [11.0, 12.0]
    assert [float(q) for q in result.executions_df["quantity"]] == [50.0, 50.0]


def test_matcher_that_updates_order_itself_is_not_double_counted() -> None:
    """撮合器就地把订单改成成交又返回 Trade: 结果与只返回 Trade 相同."""
    matcher = _MutatingMatcher()
    result = _run_issue_458(matcher)
    _assert_single_full_fill(result, matcher)


@pytest.mark.parametrize("kind", ["too_much", "zero", "order_id", "symbol", "side"])
def test_matcher_invalid_trade_books_nothing(kind: str, caplog: Any) -> None:
    """不合法的成交不入账、订单保持未成交, 并记错误日志."""
    aq.configure_logging(aq.LogConfig(console=False, level="WARNING"))
    matcher = _BadTradeMatcher(kind)

    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = _run_issue_458(matcher)

    order = result.orders_df.iloc[0]
    assert order["status"] == "new"
    assert float(order["filled_quantity"]) == 0.0
    assert result.executions_df.empty
    assert _final_position(result) == 0.0
    assert float(result.engine.portfolio.cash) == pytest.approx(100000.0)
    # 每次都被当作未成交: 订单留在队列里, 后三根 K 线各撮合一次
    assert matcher.calls == 3
    errors = [
        record
        for record in caplog.records
        if record.name == "akquant.execution.python"
        and record.levelno == logging.ERROR
        and "Ignored invalid trade from custom Python matcher" in record.getMessage()
    ]
    assert len(errors) == 3
    assert getattr(errors[0], "order_id", None)

    aq.register_logger(console=False, level="INFO")


def test_matcher_returning_none_leaves_order_working() -> None:
    """返回 None: 本次不成交, 订单保持在途."""
    matcher = _NoneMatcher()
    result = _run_issue_458(matcher)

    order = result.orders_df.iloc[0]
    assert order["status"] == "new"
    assert float(order["filled_quantity"]) == 0.0
    assert result.executions_df.empty
    assert _final_position(result) == 0.0
    assert matcher.calls == 3
