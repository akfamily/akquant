"""0.3.71 旧存档回归夹具：策略类与生成脚本.

``tests/fixtures/checkpoint_0371_minute.pkl`` 由 AKQuant 0.3.71 发布版
（qmfquant-0.3.71 wheel 内置的 akquant）在仓库外的临时 venv 里生成：在仓库外的
目录下用该 venv 的解释器以 ``python -I -c`` 执行::

    import sys
    sys.path.insert(0, "<repo>/tests")
    import _checkpoint_0371 as m
    m.generate("<repo>/tests/fixtures/checkpoint_0371_minute.pkl")

不能直接 ``python tests/_checkpoint_0371.py`` 运行：那样类路径会变成 ``__main__``。
存档里 pickle 的策略类路径是 ``_checkpoint_0371.HoldThenSell``，因此测试与生成
脚本都必须以顶层模块名 ``_checkpoint_0371`` 导入本文件（pytest 会把 tests/
加入 sys.path）。本模块只能用两个版本都有的公开 API。
"""

from __future__ import annotations

from typing import Any, List

import pandas as pd
from akquant import Bar, Strategy

TZ = "Asia/Shanghai"
SYMBOL = "600000.SH"


def minute_bars(day: str, prices: List[float]) -> pd.DataFrame:
    """``day`` 09:31 起的连续分钟线，开高低收都取 ``prices``."""
    idx = pd.date_range(f"{day} 09:31", periods=len(prices), freq="1min", tz=TZ)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": SYMBOL,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 1e6,
        }
    )


PHASE1 = ("2024-01-02", [10.0, 10.1, 10.2, 10.3, 10.4])
PHASE2 = ("2024-01-03", [11.0, 11.1, 11.2, 11.3, 11.4])


class HoldThenSell(Strategy):
    """第一段首根 bar 买 100 股；续跑段（次日）首根 bar 卖出全部持仓."""

    def __init__(self) -> None:
        super().__init__()
        self.bought = False
        self.sold = False
        self.position_at_resume: Any = None

    def on_bar(self, bar: Bar) -> None:
        if not self.bought:
            self.bought = True
            self.buy(bar.symbol, 100)
            return
        day = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ).date()
        if not self.sold and str(day) == PHASE2[0]:
            self.sold = True
            self.position_at_resume = self.get_position(bar.symbol)
            self.sell(bar.symbol, 100)


def generate(path: str) -> None:
    """用当前安装的 akquant 跑第一段并保存存档（供 0.3.71 venv 调用）."""
    from akquant import run_backtest, save_checkpoint

    result = run_backtest(
        strategy=HoldThenSell,
        data=minute_bars(*PHASE1),
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
    )
    assert result.engine is not None
    save_checkpoint(result.engine, result.strategy, path)
    print("cash", result.engine.portfolio.cash)
    print(result.orders_df[["symbol", "side", "filled_quantity", "avg_price"]])
