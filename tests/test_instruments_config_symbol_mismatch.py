"""instruments_config 配了却在数据里找不到的标的, 必须告警而非静默丢弃.

回测的合约快照是按**数据里实际出现的** symbol 建的; ``instruments_config`` 里
symbol 与数据不一致的条目会被**完全静默丢弃**, 该标的回退到默认合约参数。
实测: 数据 symbol 为 ``600487``(去后缀写法)而配置写 ``600487.SH`` 时,
快照 key 只有 ``600487``, 且 ``lot_size`` 从配置的 100 变成默认 **1.0** ——
``tick_size`` 恰好与股票默认值(0.01)相同, 所以唯一露出马脚的是 lot_size。

A 股下 ``lot_size`` 回退成 1 会让下单数量不再整百(参见
``test_order_target_lot_size_alignment``), 而用户完全以为自己配好了。
既有的「``symbols`` 里零数据的标的会告警」不覆盖 ``instruments_config`` 这一侧。
"""

from typing import Any, List

import pandas as pd
import pytest
from akquant import (
    BacktestConfig,
    InstrumentConfig,
    Strategy,
    StrategyConfig,
    run_backtest,
)

BARE = "600487"  # 数据里的写法(平台的去后缀习惯)
SUFFIXED = "600487.SH"  # 配置里的写法


class _Noop(Strategy):
    def on_bar(self, bar: Any) -> None:
        _ = bar


def _df() -> pd.DataFrame:
    rows: List[dict] = []
    for minute in (31, 32, 33):
        rows.append(
            {
                "date": pd.Timestamp(f"2026-08-03 09:{minute}:00"),
                "symbol": BARE,
                "open": 12.34,
                "high": 12.56,
                "low": 12.11,
                "close": 12.45,
                "volume": 1_234_567.0,
            }
        )
    return pd.DataFrame(rows)


def _run(instrument_symbol: str) -> None:
    run_backtest(
        strategy=_Noop(),
        data=_df(),
        config=BacktestConfig(
            strategy_config=StrategyConfig(initial_cash=1_000_000.0),
            instruments_config=[
                InstrumentConfig(
                    symbol=instrument_symbol,
                    asset_type="STOCK",
                    tick_size=0.01,
                    lot_size=100,
                )
            ],
        ),
        show_progress=False,
    )


def test_warns_when_configured_symbol_absent_from_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """配置的标的在数据里不存在时必须点名告警."""
    with caplog.at_level("WARNING"):
        _run(SUFFIXED)
    assert SUFFIXED in caplog.text, (
        f"未点名被丢弃的配置标的 {SUFFIXED}: {caplog.text[:600]}"
    )


def test_no_warning_when_configured_symbol_matches_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """配置与数据一致时不能告警(正常用法不该被刷屏)."""
    with caplog.at_level("WARNING"):
        _run(BARE)
    assert "instruments_config" not in caplog.text, (
        f"配置与数据一致却被误告警: {caplog.text[:600]}"
    )
