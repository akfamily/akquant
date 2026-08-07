# -*- coding: utf-8 -*-
"""行情源与交易源分开指定示例（market_broker + trader_broker）.

说明:
- 一个 broker 不必同时提供行情与交易两条通道: `GatewayBundle` 的
  `market_gateway` 与 `trader_gateway` 是两个独立可选字段。
- 本例注册一个**只有交易通道**的 broker（`market_gateway=None`，形如某些券商/
  柜台插件）。它单独使用时策略收不到任何行情——`on_bar` 不触发、
  `current_tick` 恒为 None。
- 用 `market_broker="replay"` 借用内置回放行情源补上这一侧，无需柜台即可实跑。
- 两侧必须**同时写明**: 只给一侧会报错。这样 `broker` 不必兼任另一侧，避免
  "读 broker='qmf', market_broker='replay' 得先知道 qmf 只有交易通道" 的歧义。
  `broker` 仅用于"单个 broker 供两侧"的场景。
"""

from typing import Any, Callable, Sequence

import pandas as pd
from akquant import AssetType, DataFeed, Instrument, Strategy, run_live
from akquant.akquant import Bar
from akquant.gateway import GatewayBundle, register_broker, unregister_broker
from akquant.gateway.broker_models import (
    BrokerCapability,
    UnifiedAccount,
    UnifiedExecutionReport,
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedPosition,
    UnifiedTrade,
)

SYMBOL = "MIXED_DEMO"
TRADE_ONLY_BROKER = "demo_trade_only"


class _TradeOnlyGateway:
    """只有交易通道的网关（无行情）——形如只做交易的券商插件."""

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def place_order(self, req: UnifiedOrderRequest) -> str:
        return f"demo-{req.client_order_id}"

    def get_capabilities(self) -> BrokerCapability:
        return BrokerCapability(broker_name=TRADE_ONLY_BROKER)

    def cancel_order(self, broker_order_id: str) -> None:
        _ = broker_order_id

    def query_order(self, broker_order_id: str) -> UnifiedOrderSnapshot | None:
        _ = broker_order_id
        return None

    def query_trades(self, since: int | None = None) -> list[UnifiedTrade]:
        _ = since
        return []

    def query_account(self) -> UnifiedAccount | None:
        return None

    def query_positions(self) -> list[UnifiedPosition]:
        return []

    def on_order(self, callback: Callable[[UnifiedOrderSnapshot], None]) -> None:
        _ = callback

    def on_trade(self, callback: Callable[[UnifiedTrade], None]) -> None:
        _ = callback

    def on_execution_report(
        self, callback: Callable[[UnifiedExecutionReport], None]
    ) -> None:
        _ = callback

    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        return []

    def sync_today_trades(self) -> list[UnifiedTrade]:
        return []

    def heartbeat(self) -> bool:
        return True

    def start(self) -> None:
        return None


def _trade_only_builder(
    feed: DataFeed,
    symbols: Sequence[str],
    use_aggregator: bool,
    **kwargs: Any,
) -> GatewayBundle:
    """构建只有交易通道的 bundle（market_gateway 显式为 None）."""
    _ = (feed, symbols, use_aggregator, kwargs)
    return GatewayBundle(
        market_gateway=None,  # 关键: 这个 broker 不提供行情
        trader_gateway=_TradeOnlyGateway(),
        trader_capabilities=BrokerCapability(broker_name=TRADE_ONLY_BROKER),
        metadata={"broker": TRADE_ONLY_BROKER},
    )


class MixedBrokerStrategy(Strategy):
    """记录收到的 bar，用于验证行情确实来自 market_broker 指定的源."""

    def on_start(self) -> None:
        """初始化计数器."""
        self.received: list[str] = []

    def on_bar(self, bar: Bar) -> None:
        """每根 bar 打印一行，证明行情通路已接通."""
        ts = pd.Timestamp(bar.timestamp, unit="ns", tz="Asia/Shanghai")
        self.received.append(str(bar.symbol))
        print(f"[on_bar] {ts:%Y-%m-%d %H:%M:%S} {bar.symbol} close={bar.close}")


def _make_bars() -> list[Bar]:
    """构造 4 根确定性 bar 供回放."""
    return [
        Bar(
            timestamp=int(pd.Timestamp(text, tz="Asia/Shanghai").value),
            open=close,
            high=close + 0.5,
            low=close - 0.5,
            close=close,
            volume=1000.0,
            symbol=SYMBOL,
        )
        for text, close in [
            ("2023-01-03 09:30:00", 10.0),
            ("2023-01-03 10:00:00", 10.3),
            ("2023-01-03 13:30:00", 10.1),
            ("2023-01-03 14:50:00", 10.6),
        ]
    ]


def main() -> None:
    """用「只有交易的 broker + replay 行情」跑一次 paper 会话."""
    register_broker(TRADE_ONLY_BROKER, _trade_only_builder)
    try:
        instruments = [
            Instrument(
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
        ]

        strategy = MixedBrokerStrategy()
        print(
            f"market_broker=replay（行情）+ trader_broker={TRADE_ONLY_BROKER}（交易）"
        )
        run_live(
            strategy_cls=strategy,
            instruments=instruments,
            # 行情源: 借用内置回放行情补上这个 broker 缺失的那一侧。
            market_broker="replay",
            # 交易源: 只有交易通道的 broker。单独使用时收不到任何行情。
            trader_broker=TRADE_ONLY_BROKER,
            trading_mode="paper",
            gateway_options={"bars": _make_bars()},
            cash=1_000_000,
            show_progress=False,
            # 安全网: 回放会在 4 根 bar 后自行结束（replay 通过 metadata 声明
            # bounded_event_total，混搭时该声明同样生效），不会等到这里。
            duration="60s",
        )

        print(f"\n收到 {len(strategy.received)} 根 bar")
        if not strategy.received:
            raise SystemExit("行情通路未接通: 未收到任何 bar")
        print("行情来自 market_broker、交易来自 trader_broker —— 分开指定生效。")
        print("注意: 两侧必须同时写明, 只给一侧会报错。")
    finally:
        unregister_broker(TRADE_ONLY_BROKER)


if __name__ == "__main__":
    main()
