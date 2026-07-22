# -*- coding: utf-8 -*-
"""
run_live 多策略 slot 编排示例.

演示目标:
- 在 run_live 中配置主策略与副策略 slot。
- 主策略使用函数式 on_bar，副策略使用类风格 Strategy。
- 使用 paper 模式进行编排验证。
"""

from typing import Any

from akquant import AssetType, Bar, Instrument, Strategy, run_live


def primary_initialize(ctx: Any) -> None:
    """初始化主策略上下文."""
    ctx.primary_submitted = False


def primary_on_bar(ctx: Any, bar: Any) -> None:
    """主策略函数式 on_bar."""
    if ctx.primary_submitted:
        return
    ctx.buy(symbol=bar.symbol, quantity=1)
    ctx.primary_submitted = True


class SecondarySlotStrategy(Strategy):
    """副策略 slot 示例."""

    def __init__(self) -> None:
        """初始化副策略状态."""
        super().__init__()
        self._submitted = False

    def on_bar(self, bar: Bar) -> None:
        """副策略 on_bar 逻辑."""
        if self._submitted:
            return
        self.buy(symbol=bar.symbol, quantity=1)
        self._submitted = True


def main() -> None:
    """运行 run_live 多 slot 编排示例."""
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
        strategy_cls=primary_on_bar,
        instruments=instruments,
        strategy_id="alpha",
        strategies_by_slot={"beta": SecondarySlotStrategy},
        initialize=primary_initialize,
        broker="ctp",
        trading_mode="paper",
        md_front="tcp://127.0.0.1:12345",
        use_aggregator=True,
        cash=1_000_000,
        show_progress=False,
        duration="30s",
    )


if __name__ == "__main__":
    main()
