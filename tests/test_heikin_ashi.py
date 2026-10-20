"""内置 Heikin-Ashi (平均 K 线) 指标: 数值、clone_indicator、self.I 四分量."""

from typing import Any, List, Tuple

import akquant as aq
import numpy as np
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest

# (open, high, low, close), 手算样例见断言注释
_OHLC = [
    (10.0, 14.0, 8.0, 12.0),
    (12.0, 16.0, 11.0, 15.0),
    (15.0, 15.5, 9.0, 10.0),
]
# 手算期望 (ha_open, ha_high, ha_low, ha_close)
_EXPECTED = [
    (11.0, 14.0, 8.0, 11.0),  # open=(10+12)/2, close=(10+14+8+12)/4
    (11.0, 16.0, 11.0, 13.5),  # open=(11+11)/2, close=54/4
    (12.25, 15.5, 9.0, 12.375),  # open=(11+13.5)/2, close=49.5/4
]


def _assert_row(actual: Any, expected: Tuple[float, ...]) -> None:
    assert actual == pytest.approx(expected, abs=1e-12)


def test_update_matches_hand_computed_values() -> None:
    """第一根即有值, 连续三根与手算一致, value getter 同步."""
    ha = aq.HeikinAshi()
    assert ha.value is None
    for bar, exp in zip(_OHLC, _EXPECTED):
        out = ha.update(*bar)
        _assert_row(out, exp)
        _assert_row(ha.value, exp)


def test_update_many_ohlc_matches_incremental() -> None:
    """批量接口与逐根结果一致, 且长度不等时报错."""
    cols = list(zip(*_OHLC))
    o, h, low, c = aq.HeikinAshi().update_many_ohlc(*[list(x) for x in cols])
    got = np.column_stack([o, h, low, c])
    np.testing.assert_allclose(got, np.array(_EXPECTED), atol=1e-12)
    with pytest.raises(ValueError):
        aq.HeikinAshi().update_many_ohlc([1.0], [1.0, 2.0], [1.0], [1.0])


def test_clone_indicator_isolated_state() -> None:
    """克隆体继承状态, 且喂副本不污染原对象 (intrabar 试算依赖这点)."""
    ha = aq.HeikinAshi()
    ha.update(*_OHLC[0])
    peek = aq.clone_indicator(ha)
    assert isinstance(peek, aq.HeikinAshi)
    _assert_row(peek.value, _EXPECTED[0])
    peek.update(*_OHLC[1])
    _assert_row(ha.value, _EXPECTED[0])
    _assert_row(ha.update(*_OHLC[1]), _EXPECTED[1])


def test_self_I_ohlc_expands_to_four_components() -> None:
    """通过 self.I(input_mode="ohlc", outputs=...) 在回测里得到四个分量."""
    seen: List[Tuple[Any, ...]] = []

    class Probe(Strategy):
        def on_start(self) -> None:
            self.ha = self.I(
                aq.HeikinAshi(),
                input_mode="ohlc",
                outputs=("open", "high", "low", "close"),
            )

        def on_bar(self, bar: Bar) -> None:
            seen.append(
                (self.ha.open[0], self.ha.high[0], self.ha.low[0], self.ha.close[0])
            )

    base = 1_672_707_000_000_000_000
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(
                    base + i * 60_000_000_000, unit="ns", tz="UTC"
                ),
                "symbol": "HA",
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": 100.0,
            }
            for i, (o, h, low, c) in enumerate(_OHLC)
        ]
    )
    run_backtest(
        strategy=Probe,
        data=df,
        symbols=["HA"],
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
    )
    assert len(seen) == 3
    for row, exp in zip(seen, _EXPECTED):
        _assert_row(row, exp)
