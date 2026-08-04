# -*- coding: utf-8 -*-
"""
run_live 函数式策略入口示例.

说明:
- 演示如何用函数式回调驱动 run_live（而非继承 Strategy 类）。
- 使用内置 broker="replay" 离线回放固定行情, 无需柜台即可实跑。
- 真实交易请替换 broker 与账户配置。
"""

from typing import Any

import pandas as pd
from akquant import AssetType, Instrument, run_live
from akquant.akquant import Bar


def initialize(ctx: Any) -> None:
    """初始化函数式策略上下文."""
    ctx.bar_count = 0
    ctx.events = []


def on_bar(ctx: Any, bar: Any) -> None:
    """主回调: 接收 bar 并执行交易逻辑."""
    ctx.bar_count += 1
    ctx.events.append(f"bar:{bar.symbol}:{ctx.bar_count}")

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


def main() -> None:
    """运行函数式 run_live 示例（用内置 replay broker 离线回放）."""
    symbol = "DEMO_A"
    instruments = [
        Instrument(
            symbol=symbol,
            asset_type=AssetType.Stock,
            multiplier=1.0,
            margin_ratio=1.0,
            tick_size=0.01,
            lot_size=1,
            option_type=None,
            strike_price=None,
            expiry_date=None,
        )
    ]

    # broker="replay": 内置的确定性回放行情源, 无需柜台与可选依赖即可跑通
    # 实盘数据通路。真实交易请换成 ctp / qmf / middleware 等 broker。
    bars = [
        Bar(
            timestamp=int(pd.Timestamp(text, tz="Asia/Shanghai").value),
            open=close,
            high=close + 0.5,
            low=close - 0.5,
            close=close,
            volume=1000.0,
            symbol=symbol,
        )
        for text, close in [
            ("2023-01-03 09:30:00", 10.0),
            ("2023-01-03 10:00:00", 10.3),
            ("2023-01-03 13:30:00", 10.1),
            ("2023-01-03 14:50:00", 10.6),
        ]
    ]

    run_live(
        strategy_cls=on_bar,
        initialize=initialize,
        on_order=on_order,
        on_trade=on_trade,
        context={"strategy_name": "live_functional_demo"},
        instruments=instruments,
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars},
        cash=1_000_000,
        show_progress=False,
        # 安全网: 回放正常会在 4 根 bar 后自行结束, 不会等到这里。
        duration="60s",
    )


if __name__ == "__main__":
    main()
