# -*- coding: utf-8 -*-
"""buying_power 对"刚下单的新标的"必须把该笔单计入投影.

Strategy.buying_power -> ctx.buying_power -> Rust get_buying_power
(src/context.rs:718). 其中 pending 包含 self.orders, 即本次回调中刚提交、
尚未进入 active_orders 的单. 若价格表查不到该 symbol, projection 会跳过这笔单,
buying_power 随之偏大.

本测试是特征化测试: 先在改动前锁住当前值, 再据此约束 last_prices 的重构.
"""

from datetime import datetime, timezone
from typing import List, Optional

import akquant as aq

HELD = "AAA"
FRESH = "BBB"


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


_DAYS = [
    datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 3, 15, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 4, 15, 0, tzinfo=timezone.utc),
]


def _bars() -> List[aq.Bar]:
    bars: List[aq.Bar] = []
    for day_index, day in enumerate(_DAYS):
        for symbol, base in ((HELD, 100.0), (FRESH, 50.0)):
            price = base + day_index
            bars.append(aq.Bar(_ns(day), price, price, price, price, 1e6, symbol))
    return bars


class _Probe(aq.Strategy):
    """在同一回调内下新标的的单, 前后各读一次 buying_power.

    刻意跳过第一次 HELD 回调:同一时间戳下 bar 按输入顺序逐条送达,
    last_prices 是逐条即时更新的, 而非按时间戳整批预置。若在第一根 bar
    (整场回测最早的 HELD bar) 上就下单, FRESH 自己的 bar 还没被送达过,
    last_prices 里根本没有 FRESH 的价格——那是"价格表还没见过这个 symbol"
    的构造问题, 不是本测试要锁的"未持仓/无挂单"语义。等到第二次 HELD 回调,
    FRESH 在上一交易日的 bar 已经先一步送达并写入 last_prices, 此时 FRESH
    依旧无持仓、无挂单, 才是该否决方案要处理的真实场景。
    """

    def __init__(self) -> None:
        super().__init__()
        self.before: Optional[float] = None
        self.after: Optional[float] = None
        self._held_bar_count = 0

    def on_bar(self, bar: aq.Bar) -> None:
        if bar.symbol != HELD:
            return
        self._held_bar_count += 1
        if self._held_bar_count != 2:
            return
        # FRESH 既不在持仓, 也无既有挂单
        assert self.get_position(FRESH) == 0
        self.before = float(self.buying_power)
        self.buy(symbol=FRESH, quantity=100)
        self.after = float(self.buying_power)


def test_fresh_symbol_order_is_projected_into_buying_power() -> None:
    """刚提交的新标的买单必须压低 buying_power.

    若 last_prices 的重构让 FRESH 的价格查不到, projection 会跳过这笔单,
    after 将等于 before —— 这正是本测试要拦住的静默语义变化.
    """
    probe = _Probe()
    _ = aq.run_backtest(
        data=_bars(),
        strategy=probe,
        symbols=[HELD, FRESH],
        initial_cash=1_000_000.0,
        show_progress=False,
    )

    assert probe.before is not None
    assert probe.after is not None
    # 这笔单被计入投影, 故 after 必须严格小于 before
    assert probe.after < probe.before, (
        f"新标的买单未被计入 buying_power 投影: "
        f"before={probe.before} after={probe.after}"
    )
    # 以下两个精确值是 last_prices 迁移前记录的基线(特征化测试的核心断言):
    # before = 1_000_000 * (1 - 0.0001) = 999900.0
    # after  = 995000 * 0.9999 = 994900.5 (100 股 * 50.0 买入, 现金 -5000 后的投影)
    # 若这两个值发生变化, 说明投影算术本身被改动了, 必须去排查原因,
    # 而不是直接把新值重新灌进来当基线。
    assert probe.before == 999900.0
    assert probe.after == 994900.5
