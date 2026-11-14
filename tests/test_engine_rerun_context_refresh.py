# -*- coding: utf-8 -*-
"""同一个 Engine 跑完一次后改配置再跑: 策略上下文要跟上新的风控配置与合约表.

策略上下文在第一次回调时创建并一直复用, 其中的风控配置与合约表是创建时的快照。
此前再次 ``run()`` 不刷新: 第二次运行里 ``ctx.risk_config``、可用买入力等仍按第一次
运行时的配置算。只影响手工复用 ``Engine`` 的用法(``run_backtest`` 每次新建引擎)。
"""

from typing import Any, List, Tuple

import akquant
import pandas as pd


def _bars(symbol: str, day0: int, n: int) -> List[Any]:
    out = []
    for i in range(n):
        ts = pd.Timestamp("2024-01-02 10:00:00", tz="UTC") + pd.Timedelta(days=day0 + i)
        out.append(
            akquant.Bar(
                timestamp=int(ts.value),
                open=10.0,
                high=10.5,
                low=9.5,
                close=10.0,
                volume=1000.0,
                symbol=symbol,
            )
        )
    return out


def _instrument(symbol: str, multiplier: float) -> Any:
    return akquant.Instrument(
        symbol=symbol,
        asset_type=akquant.AssetType.Stock,
        multiplier=multiplier,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1.0,
    )


def _risk(safety_margin: float) -> Any:
    config = akquant.RiskConfig()
    config.safety_margin = safety_margin
    return config


def _engine() -> Any:
    engine = akquant.Engine()
    engine.use_simple_market(0.0)
    engine.configure_sessions(sessions={"default": "ALWAYS_OPEN"})
    engine.set_cash(100000.0)
    return engine


class _Probe(akquant.Strategy):
    """在 BBB 的 bar 上记录上下文读到的配置; 首根挂一笔不会成交的限价买单."""

    seen: List[Tuple[float, float]]

    def on_start(self) -> None:
        self.seen = []

    def on_bar(self, bar: Any) -> None:
        if bar.symbol != "BBB":
            return
        ctx = self.ctx
        assert ctx is not None
        self.seen.append((ctx.risk_config.safety_margin, ctx.buying_power))
        if len(self.seen) == 1:
            self.buy(symbol="BBB", quantity=100, price=1.0)


def _fresh_run() -> List[Tuple[float, float]]:
    """对照: 新引擎一开始就是最终配置."""
    engine = _engine()
    engine.add_instrument(_instrument("AAA", 1.0))
    engine.add_instrument(_instrument("BBB", 5.0))
    engine.set_risk_config(_risk(0.5))
    engine.add_bars(_bars("BBB", 5, 3))
    strategy = _Probe()
    engine.run(strategy, show_progress=False)
    return strategy.seen


def test_rerun_sees_updated_risk_config_and_instruments() -> None:
    """改风控配置、加合约后再跑, 上下文读到的与新引擎一致."""
    engine = _engine()
    engine.add_instrument(_instrument("AAA", 1.0))
    engine.set_risk_config(_risk(0.1))
    engine.add_bars(_bars("AAA", 0, 2))
    strategy = _Probe()
    engine.run(strategy, show_progress=False)
    assert strategy.seen == []

    engine.set_risk_config(_risk(0.5))
    engine.add_instrument(_instrument("BBB", 5.0))
    engine.add_bars(_bars("BBB", 5, 3))
    engine.run(strategy, show_progress=False)

    expected = _fresh_run()
    assert len(expected) == 3
    # 对照自身要有区分度: 配置是新的; 挂单后可用买入力因合约乘数而下降
    assert [row[0] for row in expected] == [0.5, 0.5, 0.5]
    assert expected[1][1] < expected[0][1]
    assert strategy.seen == expected
