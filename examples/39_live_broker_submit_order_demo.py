# -*- coding: utf-8 -*-
"""
run_live broker_live + 函数式 submit_order 最小闭环示例.

说明:
- 该示例展示函数式策略在 broker_live 模式下使用 submit_order。
- submit_order 由 run_live 在连接交易网关后自动注入到策略上下文。
- 默认地址与账户参数为占位，请替换为实际网关配置。
"""

from typing import Any

from akquant import AssetType, Instrument, run_live


def initialize(ctx: Any) -> None:
    """初始化上下文."""
    ctx.sent = False
    ctx.client_seq = 0


def _next_client_order_id(ctx: Any) -> str:
    ctx.client_seq += 1
    return f"demo-live-{ctx.client_seq}"


def on_bar(ctx: Any, bar: Any) -> None:
    """收到行情后提交一笔最小市场单."""
    print(f"[on_bar] symbol={bar.symbol} close={bar.close}")
    if ctx.sent:
        return
    if not getattr(ctx, "broker_ready", False):
        print("[on_bar] broker 未就绪，跳过下单")
        return
    client_order_id = _next_client_order_id(ctx)
    broker_order_id = ctx.submit_order(
        symbol=bar.symbol,
        side="Buy",
        quantity=1.0,
        client_order_id=client_order_id,
        order_type="Market",
    )
    print(
        f"[on_bar] submitted client_order_id={client_order_id} "
        f"broker_order_id={broker_order_id}"
    )
    ctx.sent = True


def on_order(ctx: Any, order: Any) -> None:
    """订单回调."""
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


def on_reject(ctx: Any, order: Any) -> None:
    """拒单回调."""
    _ = ctx
    print(
        f"[on_reject] {getattr(order, 'symbol', '?')} "
        f"reason={getattr(order, 'reject_reason', '')}"
    )


def main() -> None:
    """运行 broker_live 函数式下单示例."""
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

    run_live(
        strategy_cls=on_bar,
        initialize=initialize,
        on_order=on_order,
        on_trade=on_trade,
        on_reject=on_reject,
        instruments=instruments,
        broker="ctp",
        trading_mode="broker_live",
        md_front="tcp://127.0.0.1:12345",
        td_front="tcp://127.0.0.1:12346",
        broker_id="9999",
        user_id="demo_user",
        password="demo_password",
        app_id="simnow_client_test",
        auth_code="0000000000000000",
        use_aggregator=True,
        cash=1_000_000,
        show_progress=False,
        duration="30s",
        # 本示例用占位柜台地址, 连不上真实柜台。默认 broker_ready_required=True
        # 会在就绪超时后直接中止启动(实盘就该如此), 这里显式放宽以便离线演示——
        # on_bar 里已用 ctx.broker_ready 守卫, 未就绪时不会下单。
        # 真实部署请保留默认值 True。
        broker_ready_required=False,
    )


if __name__ == "__main__":
    main()
