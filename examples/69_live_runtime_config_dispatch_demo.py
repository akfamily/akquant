# -*- coding: utf-8 -*-
"""
run_live 运行时配置下发示例.

演示目标:
- 用 run_live(strategy_runtime_config=...) 从入口切换策略运行时行为, 不改策略代码。
- 同一份策略在回测用 error_mode="raise" 严格暴露问题, 实盘用 "continue" 容错不中断交易。
- 配置会下发到主策略**与每个槽位子策略**, 与 run_backtest 行为对称。
- runtime_config_override=False 时保留策略自设值(只告警, 不覆盖)。

用 broker="replay" 驱动: 行情由本文件内的合成 bar 提供, 无需连接任何柜台。
"""

import pandas as pd
from akquant import (
    AssetType,
    Bar,
    Instrument,
    Strategy,
    StrategyRuntimeConfig,
    run_backtest,
    run_live,
)

SYMBOL = "DEMO_RC"


class BrokenOnBarStrategy(Strategy):
    """on_bar 里故意抛异常的策略, 用来观察 error_mode 的实际效果."""

    def __init__(self) -> None:
        """初始化计数器."""
        super().__init__()
        self.bars_seen = 0

    def on_start(self) -> None:
        """打印 on_start 时刻已生效的配置 —— 下发必须早于这里."""
        print(
            f"on_start error_mode={self.runtime_config.error_mode} "
            f"indicator_mode={self.runtime_config.indicator_mode}"
        )

    def on_bar(self, bar: Bar) -> None:
        """每根 bar 抛一次异常."""
        self.bars_seen += 1
        raise RuntimeError(f"deliberate failure on bar {self.bars_seen}")


class SlotProbeStrategy(Strategy):
    """槽位子策略: 只报告自己看到的配置, 用来验证 slot 也拿到了下发."""

    def on_start(self) -> None:
        """打印槽位策略看到的配置."""
        print(f"slot on_start error_mode={self.runtime_config.error_mode}")


def _instrument() -> Instrument:
    """构造一个股票标的."""
    return Instrument(
        symbol=SYMBOL,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bars(count: int = 3) -> list:
    """构造若干根连续 bar."""
    base = pd.Timestamp("2023-01-03 09:30:00", tz="Asia/Shanghai")
    out = []
    for i in range(count):
        close = 10.0 + i * 0.1
        out.append(
            Bar(
                timestamp=int((base + pd.Timedelta(minutes=i)).value),
                open=close,
                high=close + 0.05,
                low=close - 0.05,
                close=close,
                volume=1000.0,
                symbol=SYMBOL,
            )
        )
    return out


def _bar_frame(count: int = 3) -> pd.DataFrame:
    """把同样的 bar 序列做成回测用的 DataFrame."""
    rows = [
        {
            "datetime": pd.Timestamp(b.timestamp, tz="Asia/Shanghai"),
            "symbol": SYMBOL,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in _bars(count)
    ]
    return pd.DataFrame(rows)


def scenario1_live_continue() -> None:
    """实盘用 error_mode="continue": 回调异常被吞, 会话跑完全部 bar."""
    print("--- scenario1: live error_mode=continue ---")
    strategy = BrokenOnBarStrategy()
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars()},
        cash=100_000.0,
        show_progress=False,
        duration="60s",
        strategy_runtime_config={"error_mode": "continue"},
    )
    print(f"scenario1_done bars_seen={strategy.bars_seen}")


def scenario2_backtest_raise() -> None:
    """同一策略在回测用 error_mode="raise": 第一根 bar 就把异常抛出来."""
    print("--- scenario2: backtest error_mode=raise ---")
    try:
        run_backtest(
            data=_bar_frame(),
            strategy=BrokenOnBarStrategy,
            symbols=SYMBOL,
            initial_cash=100_000.0,
            show_progress=False,
            strategy_runtime_config=StrategyRuntimeConfig(error_mode="raise"),
        )
    except Exception as exc:  # noqa: BLE001  (示例刻意展示异常被抛出)
        print(f"scenario2_exception={type(exc).__name__}: {exc}")


def scenario3_slot_dispatch() -> None:
    """配置同时下发到主策略与槽位子策略."""
    print("--- scenario3: dispatch reaches slot strategies ---")
    strategy = BrokenOnBarStrategy()
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument()],
        strategy_id="alpha",
        strategies_by_slot={"beta": SlotProbeStrategy},
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars(1)},
        cash=100_000.0,
        show_progress=False,
        duration="60s",
        strategy_runtime_config={"error_mode": "continue"},
    )
    print("scenario3_done")


def scenario4_override_false() -> None:
    """runtime_config_override=False: 策略自设值保留, 入口值只告警不生效."""
    print("--- scenario4: runtime_config_override=False ---")
    strategy = BrokenOnBarStrategy()
    strategy.runtime_config = StrategyRuntimeConfig(error_mode="continue")
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars(1)},
        cash=100_000.0,
        show_progress=False,
        duration="60s",
        strategy_runtime_config={"error_mode": "raise"},
        runtime_config_override=False,
    )
    print(f"scenario4_kept={strategy.runtime_config.error_mode}")


def main() -> None:
    """依次运行四个场景."""
    scenario1_live_continue()
    scenario2_backtest_raise()
    scenario3_slot_dispatch()
    scenario4_override_false()


if __name__ == "__main__":
    main()
