"""外部量化信号平台接入: HTTP webhook 驱动下单.

演示 ``akquant.signal``: 平台把指令 POST 到本机端点, AKQuant 校验鉴权后经风控下单
—— 全程不写策略逻辑, 因为信号已经是指令。

本例用 paper 模式 + 回放行情, 无需柜台即可完整跑通。实盘只需把
``trading_mode`` 改成 ``"broker_live"`` 并配好 ``broker`` / ``trader_broker``。

**两种模式的风控覆盖面不同**(引擎架构决定, 非配置项):
paper 走完整风控; broker_live 只有策略级三项限额前置生效。详见
docs/zh/meta/signal-ingestion-rfc.md。
"""

import json
import os
import threading
import time
import urllib.request
from typing import Any, List

import akquant as aq
import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar
from akquant.gateway import register_broker, unregister_broker
from akquant.signal import HttpSignalSource

SYMBOL = "SIGDEMO"
CLOSE = 20.0
# 生产务必从环境变量/密钥管理读取, 不要硬编码。
TOKEN = os.environ.get("AKQUANT_SIGNAL_TOKEN", "demo-token-please-change")


def build_bars(count: int = 6) -> List[Bar]:
    """造当前墙钟时间戳起的连续 bar(实时引擎按墙钟判定时序)."""
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    return [
        Bar(
            timestamp=int((now + pd.Timedelta(seconds=i)).value),
            open=CLOSE,
            high=CLOSE + 0.5,
            low=CLOSE - 0.5,
            close=CLOSE,
            volume=100_000.0,
            symbol=SYMBOL,
        )
        for i in range(count)
    ]


class SignalDrivenStrategy(Strategy):
    """自身不产生信号, 只观测成交 —— 订单全部来自外部平台."""

    def __init__(self) -> None:
        """初始化观测容器与握手闸门."""
        self.trades: List[Any] = []
        self.bars_seen = 0
        self.ready = threading.Event()
        self.settled = threading.Event()

    def on_bar(self, bar: Bar) -> None:
        """首根 bar 后放行投递, 第二根等其落地(确保成交发生在会话内)."""
        self.bars_seen += 1
        if self.bars_seen == 1:
            self.ready.set()
        elif self.bars_seen == 2:
            self.settled.wait(timeout=10.0)

    def on_trade(self, trade: Any) -> None:
        """打印并记录成交."""
        self.trades.append(trade)
        print(f"[成交] {trade.symbol} {trade.side} {trade.quantity} @ {trade.price}")


def post_signal(url: str, payload: dict) -> dict:
    """把一条信号 POST 给 AKQuant(标准库实现, 无需额外依赖)."""
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return dict(json.loads(response.read().decode()))


def main() -> None:
    """起端点 → 跑会话 → 模拟平台推送 → 校验成交."""
    print(f"AKQuant {aq.__version__} — 外部信号平台接入演示\n")

    strategy = SignalDrivenStrategy()
    # port=0 让系统分配空闲端口, 便于演示; 生产请固定端口。
    source = HttpSignalSource(token=TOKEN, port=0)

    def platform() -> None:
        """模拟信号平台: 等会话就绪后推两条指令(第二条重复, 演示幂等)."""
        if not strategy.ready.wait(timeout=15.0):
            strategy.settled.set()
            return
        try:
            url = f"http://127.0.0.1:{source.bound_port}/signal"
            payload = {
                "signal_id": "demo-0001",
                "symbol": SYMBOL,
                "action": "buy",
                "quantity": 200,
                "price": CLOSE,
                "strategy_id": "_default",
            }
            print(f"[平台] 推送信号 → {url}")
            print(f"[AKQuant] 回执: {post_signal(url, payload)}")
            print("[平台] 重推同一 signal_id(演示幂等)")
            print(f"[AKQuant] 回执: {post_signal(url, payload)}")
            time.sleep(0.4)
        finally:
            strategy.settled.set()

    started = threading.Event()

    def runner() -> None:
        started.set()
        platform()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    # 必须确认线程已就绪: 主线程随后进入引擎循环并长期持有 GIL。
    started.wait(timeout=5.0)

    register_broker(
        "signal-demo",
        lambda **kw: _demo_bundle(kw["feed"], kw["symbols"], build_bars()),
    )
    try:
        run_live(
            strategy_cls=strategy,
            instruments=[
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
            ],
            broker="signal-demo",
            trading_mode="paper",
            cash=1_000_000.0,
            show_progress=False,
            duration="20s",
            signal_source=source,
            # 策略级限额同样约束外部信号: 200 * 20 = 4000, 未超 10000
            strategy_max_order_value={"_default": 10_000.0},
        )
    finally:
        unregister_broker("signal-demo")

    accepted = [r for r in source.results if r.status.value == "accepted"]
    duplicates = [r for r in source.results if r.status.value == "duplicate"]
    print(f"\n回执统计: 接受 {len(accepted)} 条, 判重 {len(duplicates)} 条")
    print(f"实际成交: {len(strategy.trades)} 笔")

    assert len(accepted) == 1, "应只接受 1 条信号"
    assert len(duplicates) == 1, "重推应被幂等判重"
    assert strategy.trades, "信号单应成交"
    print("\n演示完成: 外部信号经鉴权与风控后成交, 重复推送被幂等丢弃。")


def _demo_bundle(feed: Any, symbols: Any, bars: List[Bar]) -> Any:
    """造只含行情侧的 GatewayBundle, 按节奏推 bar."""
    from akquant.gateway.protocols import GatewayBundle

    class _Paced:
        def __init__(self) -> None:
            self.stop = threading.Event()

        def connect(self) -> None: ...
        def disconnect(self) -> None: ...
        def subscribe(self, s: Any) -> None: ...
        def unsubscribe(self, s: Any) -> None: ...
        def on_tick(self, cb: Any) -> None: ...
        def on_bar(self, cb: Any) -> None: ...

        def start(self) -> None:
            for bar in bars:
                if self.stop.is_set():
                    return
                feed.add_bar(bar)
                if self.stop.wait(timeout=0.25):
                    return

    return GatewayBundle(
        market_gateway=_Paced(),
        trader_gateway=None,
        trader_capabilities=None,
        metadata={"broker": "signal-demo", "bounded_event_total": len(bars)},
    )


if __name__ == "__main__":
    main()
