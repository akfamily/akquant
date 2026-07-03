# -*- coding: utf-8 -*-
"""
LiveRunner broker 事件审计示例.

演示目标:
- 使用 on_broker_event 统一观察 order/trade/report 事件
- 输出 owner_strategy_id 便于多 slot 归因
"""

from typing import Any

from akquant import AssetType, Instrument
from akquant.live import LiveRunner


def initialize(ctx: Any) -> None:
    """初始化上下文."""
    ctx.sent = False
    ctx.seq = 0


def on_bar(ctx: Any, bar: Any) -> None:
    """Bar 到达后发送一次最小市价单."""
    if ctx.sent:
        return
    if not getattr(ctx, "broker_ready", False):
        return
    ctx.seq += 1
    client_order_id = f"audit-{ctx.seq}"
    _ = ctx.submit_order(
        symbol=bar.symbol,
        side="Buy",
        quantity=1.0,
        client_order_id=client_order_id,
        order_type="Market",
    )
    ctx.sent = True


def on_broker_event(event: dict[str, Any]) -> None:
    """统一审计 broker 事件."""
    event_type = str(event.get("event_type", ""))
    owner = str(event.get("owner_strategy_id", ""))
    payload = event.get("payload", {})
    print(f"[audit] type={event_type} owner={owner} payload={payload}")


def main() -> None:
    """运行 broker 事件审计示例."""
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
        strategy_id="alpha",
        initialize=initialize,
        on_broker_event=on_broker_event,
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
    )
    runner.run(cash=1_000_000, show_progress=False, duration="30s")


if __name__ == "__main__":
    main()
