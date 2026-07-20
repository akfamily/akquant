# -*- coding: utf-8 -*-
"""
LiveRunner 函数式策略入口示例.

说明:
- 演示如何用函数式回调驱动 LiveRunner（而非继承 Strategy 类）。
- 可用于 paper / broker_live 模式。
- 默认参数为占位示例，请替换为你自己的网关地址与账户信息。
"""

from typing import Any

from akquant import AssetType, Instrument
from akquant.live import LiveRunner


def initialize(ctx: Any) -> None:
    """初始化函数式策略上下文."""
    ctx.bar_count = 0
    ctx.events = []
    ctx.timer_registered = False


def on_bar(ctx: Any, bar: Any) -> None:
    """主回调: 接收聚合后的 bar/tick-bar 并执行交易逻辑."""
    ctx.bar_count += 1
    ctx.events.append(f"bar:{bar.symbol}:{ctx.bar_count}")

    if not ctx.timer_registered:
        ctx.schedule_daily("14:50:00", "rebalance")
        ctx.timer_registered = True

    pos = ctx.get_position(bar.symbol)
    if ctx.bar_count % 2 == 1 and pos == 0:
        ctx.buy(bar.symbol, 1)
    elif ctx.bar_count % 2 == 0 and pos > 0:
        ctx.sell(bar.symbol, 1)


def on_order(ctx: Any, order: Any) -> None:
    """订单状态回调."""
    _ = ctx
    print(
        f"[on_order] symbol={order.symbol} status={order.status} "
        f"filled={order.filled_quantity}"
    )


def on_trade(ctx: Any, trade: Any) -> None:
    """成交回调."""
    _ = ctx
    print(
        f"[on_trade] symbol={trade.symbol} side={trade.side} "
        f"price={trade.price} qty={trade.quantity}"
    )


def on_timer(ctx: Any, payload: str) -> None:
    """定时器回调."""
    ctx.events.append(f"timer:{payload}")
    print(f"[on_timer] payload={payload}")


def main() -> None:
    """运行函数式 LiveRunner 示例."""
    instruments = [
        Instrument(
            symbol="IF2506",
            asset_type=AssetType.Futures,
            multiplier=300.0,
            margin_ratio=0.12,
            tick_size=0.2,
            lot_size=1,
            option_type=None,
            strike_price=None,
            expiry_date=None,
        )
    ]

    runner = LiveRunner(
        strategy_cls=on_bar,
        initialize=initialize,
        on_order=on_order,
        on_trade=on_trade,
        on_timer=on_timer,
        context={"strategy_name": "live_functional_demo"},
        instruments=instruments,
        broker="ctp",
        trading_mode="paper",
        md_front="tcp://127.0.0.1:12345",
        use_aggregator=True,
    )
    runner.run(cash=1_000_000, show_progress=False, duration="30s")


if __name__ == "__main__":
    main()
