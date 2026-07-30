"""终态订单必须仍可通过 get_order() 查到.

此前 check_order_events 在订单进入终态(成交/撤单/拒单)时把它从
``_known_orders`` 删除, 而它同时也已不在 ``ctx.active_orders`` 里,
于是 get_order() 只查这两处必然返回 None——订单一旦终态就永久查不到。
"""

from pathlib import Path
from typing import Any

import pandas as pd
from akquant import Strategy, run_backtest, save_checkpoint
from akquant.akquant import Bar, OrderStatus
from akquant.backtest.fill_mode import CurrentClose

SYMBOL = "FINAL_ORDER_DEMO"


def _bars(days: tuple[int, ...] = (3, 4, 5), close: float = 10.0) -> list[Bar]:
    return [
        Bar(
            timestamp=pd.Timestamp(
                f"2023-01-{day:02d} 09:30:00", tz="Asia/Shanghai"
            ).value,
            open=close,
            high=close + 0.2,
            low=close - 0.2,
            close=close,
            volume=1_000_000.0,
            symbol=SYMBOL,
        )
        for day in days
    ]


def _run(strategy: Strategy) -> None:
    run_backtest(
        data=_bars(),
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )


def test_filled_order_is_still_queryable() -> None:
    """成交后的订单仍应能按 id 查到, 且状态为 Filled."""

    class FilledStrategy(Strategy):
        def __init__(self) -> None:
            self.order_id: str | None = None
            self.seen_trade = False
            self.queried: list[Any] = []

        def on_start(self) -> None:
            self.subscribe(SYMBOL)

        def on_bar(self, bar: Bar) -> None:
            if self.order_id is None:
                self.order_id = str(
                    self.submit_order(symbol=SYMBOL, side="Buy", quantity=100.0)
                )
            else:
                self.queried.append(self.get_order(self.order_id))

        def on_trade(self, trade: Any) -> None:
            self.seen_trade = True

    strategy = FilledStrategy()
    _run(strategy)

    assert strategy.seen_trade, "订单未成交, 测试前提不成立"
    assert strategy.queried, "未执行查询, 测试前提不成立"
    for order in strategy.queried:
        assert order is not None, "成交后的订单查不到"
        assert order.status == OrderStatus.Filled


def test_cancelled_order_is_still_queryable() -> None:
    """撤单后的订单仍应能按 id 查到, 且状态为 Cancelled."""

    class CancelStrategy(Strategy):
        def __init__(self) -> None:
            self.order_id: str | None = None
            self.bar_count = 0
            self.queried: list[Any] = []

        def on_start(self) -> None:
            self.subscribe(SYMBOL)

        def on_bar(self, bar: Bar) -> None:
            self.bar_count += 1
            if self.bar_count == 1:
                # 远离市价的限价单, 不会成交。
                self.order_id = str(
                    self.submit_order(
                        symbol=SYMBOL, side="Buy", quantity=100.0, price=1.0
                    )
                )
            elif self.bar_count == 2:
                assert self.order_id is not None
                self.cancel_order(self.order_id)
            else:
                self.queried.append(self.get_order(self.order_id or ""))

    strategy = CancelStrategy()
    _run(strategy)

    assert strategy.queried, "未执行查询, 测试前提不成立"
    for order in strategy.queried:
        assert order is not None, "撤单后的订单查不到"
        assert order.status == OrderStatus.Cancelled


def test_in_flight_order_query_still_works() -> None:
    """回归: 在途订单查询不得被留档改动破坏."""

    class InFlightStrategy(Strategy):
        def __init__(self) -> None:
            self.order_id: str | None = None
            self.in_flight: list[Any] = []

        def on_start(self) -> None:
            self.subscribe(SYMBOL)

        def on_bar(self, bar: Bar) -> None:
            if self.order_id is None:
                self.order_id = str(
                    self.submit_order(
                        symbol=SYMBOL, side="Buy", quantity=100.0, price=1.0
                    )
                )
            else:
                self.in_flight.append(self.get_order(self.order_id))

    strategy = InFlightStrategy()
    _run(strategy)

    assert strategy.in_flight, "未执行查询, 测试前提不成立"
    for order in strategy.in_flight:
        assert order is not None, "在途订单查不到"
        assert order.status != OrderStatus.Filled


def test_finalized_order_cache_is_bounded() -> None:
    """终态订单留档必须有上限, 按 FIFO 淘汰最旧的, 避免实盘长跑内存无界."""

    class BoundedStrategy(Strategy):
        def __init__(self) -> None:
            self.finalized_order_cache_size = 2
            self.order_ids: list[str] = []

        def on_start(self) -> None:
            self.subscribe(SYMBOL)

        def on_bar(self, bar: Bar) -> None:
            self.order_ids.append(
                str(self.submit_order(symbol=SYMBOL, side="Buy", quantity=10.0))
            )

    strategy = BoundedStrategy()
    _run(strategy)

    assert len(strategy.order_ids) >= 3, "下单次数不足, 测试前提不成立"
    # 上限为 2: 最新的两笔仍可查, 最早的一笔已被淘汰。
    assert strategy.get_order(strategy.order_ids[-1]) is not None
    assert strategy.get_order(strategy.order_ids[0]) is None


class PickleStrategy(Strategy):
    """模块级策略类, 供 pickle 往返测试使用(局部类不可 pickle)."""

    def __init__(self) -> None:
        """初始化订单 id 槽位."""
        self.order_id: str | None = None

    def on_start(self) -> None:
        """订阅测试标的."""
        self.subscribe(SYMBOL)

    def on_bar(self, bar: Bar) -> None:
        """首根 bar 下一笔市价单."""
        if self.order_id is None:
            self.order_id = str(
                self.submit_order(symbol=SYMBOL, side="Buy", quantity=100.0)
            )


def test_finalized_order_cache_is_not_checkpointed(tmp_path: Path) -> None:
    """留档是纯运行期缓存, 不得进入 checkpoint.

    它持有 Rust 侧的 Order(不可 pickle), 若随快照落盘, save_checkpoint 会直接
    抛 ``TypeError: cannot pickle 'builtins.Order' object``。
    """
    strategy = PickleStrategy()
    result = run_backtest(
        data=_bars(),
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    restored: Any = result.strategy
    assert restored._finalized_orders, "留档为空, 测试前提不成立"

    state: dict[str, Any] = restored.__getstate__()
    assert "_finalized_orders" not in state
    assert "_finalized_order_ids" not in state

    # 走真实 checkpoint 路径: 留档若泄漏进快照, 这里就会抛。
    save_checkpoint(result.engine, restored, str(tmp_path / "ckpt.pkl"))  # type: ignore[arg-type]
