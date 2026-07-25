# -*- coding: utf-8 -*-
"""
函数式 + 多策略 slot + 风控限制 端到端示例.

演示目标:
- 主策略与副策略都使用函数式 on_bar 回调
- 通过 strategies_by_slot 开启多策略槽位
- 通过策略级风险限制让 alpha 被拒单、beta 正常下单
"""

from datetime import datetime, timezone
from typing import Any, List

import akquant as aq


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


def make_bars(symbol: str = "FUNC_SLOT_DEMO") -> List[aq.Bar]:
    """构造用于多策略风控验证的三根 bar."""
    t1 = _ns(datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc))
    t2 = _ns(datetime(2023, 1, 3, 15, 0, tzinfo=timezone.utc))
    t3 = _ns(datetime(2023, 1, 4, 15, 0, tzinfo=timezone.utc))
    return [
        aq.Bar(t1, 10.0, 10.0, 10.0, 10.0, 1000.0, symbol),
        aq.Bar(t2, 12.0, 12.0, 12.0, 12.0, 1000.0, symbol),
        aq.Bar(t3, 11.0, 11.0, 11.0, 11.0, 1000.0, symbol),
    ]


def alpha_on_bar(ctx: Any, bar: aq.Bar) -> None:
    """Alpha 槽位下单逻辑."""
    if getattr(ctx, "_submitted_once", False):
        return
    ctx.buy(symbol=bar.symbol, quantity=10)
    ctx._submitted_once = True


def beta_on_bar(ctx: Any, bar: aq.Bar) -> None:
    """Beta 槽位下单逻辑."""
    if getattr(ctx, "_submitted_once", False):
        return
    ctx.buy(symbol=bar.symbol, quantity=10)
    ctx._submitted_once = True


def main() -> None:
    """运行函数式多策略风控示例."""
    events: List[aq.BacktestStreamEvent] = []
    result = aq.run_backtest(
        data=make_bars(),
        strategy=alpha_on_bar,
        symbols="FUNC_SLOT_DEMO",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": beta_on_bar},
        strategy_max_order_value={"alpha": 50.0, "beta": 200.0},
        on_event=events.append,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=1,
        stream_max_buffer=256,
    )

    orders_df = result.orders_df
    alpha_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = orders_df[orders_df["owner_strategy_id"].astype(str) == "beta"]
    alpha_reject = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    beta_reject = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    risk_owner_ids = {
        str(event.get("payload", {}).get("owner_strategy_id", ""))
        for event in events
        if event.get("event_type") == "risk"
    }

    print(f"orders_total={len(orders_df)}")
    print(f"alpha_rejected={any('exceeds strategy limit' in x for x in alpha_reject)}")
    print(f"beta_rejected={any('exceeds strategy limit' in x for x in beta_reject)}")
    print(f"risk_owner_ids={sorted(x for x in risk_owner_ids if x)}")
    print("done_functional_multi_slot_risk_demo")


if __name__ == "__main__":
    main()
