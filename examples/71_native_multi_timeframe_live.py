"""实盘多周期 Demo: replay 行情源 + subscribe_bars, 离线可跑.

行情网关声明 ``gateway_options={"freq": "1min"}`` 后 ``self.freq == "1min"``,
5 分钟窗口在 xx:x5 那根 1 分钟 bar 的 on_bar 之后**同一步**闭合(零延迟)。
同一份策略代码不改一行即可用于 run_backtest。
"""

from typing import List

import pandas as pd
from akquant import AssetType, Bar, Instrument, Strategy, run_live

SYMBOL = "000001.SZ"


def build_bars(days: int = 1) -> List[Bar]:
    """生成 A 股交易时段的 1 分钟 bar.

    午盘故意在 14:59 收尾(而非 15:00 整): ``broker='replay'`` 靠
    ``bounded_event_total`` 数满事件数后用 ``KeyboardInterrupt`` 结束会话;
    若数据的最后一根 bar 恰好同时是某个窗口的闭合边界, 该窗口会在同一步内
    ``on_bar`` 之后才闭合, 但会话已经中断, 来不及派发(也不会被尾部 flush
    补上, 因为闭合时状态已从"在形成"移除)——这是有界会话终止时机与窗口立即
    闭合重合的已知边界效应, 持续运行的实盘不会精确停在闭合边界上, 不会触发。
    错开一分钟让最后一根 5 分钟窗口改走"会话结束 flush 尾部未满窗口"这条
    已有行为(见 docs 的「限制」一节), 得到确定的 48 根窗口。
    """
    bars: List[Bar] = []
    day = pd.Timestamp("2024-01-02", tz="Asia/Shanghai")
    price = 10.0
    for _ in range(days):
        for start, end in (("09:31", "11:30"), ("13:01", "14:59")):
            rng = pd.date_range(
                f"{day.date()} {start}",
                f"{day.date()} {end}",
                freq="1min",
                tz="Asia/Shanghai",
            )
            for ts in rng:
                price += 0.01
                bars.append(
                    Bar(
                        int(ts.value),
                        price,
                        price + 0.02,
                        price - 0.02,
                        price,
                        100.0,
                        SYMBOL,
                    )
                )
        day += pd.Timedelta(days=1)
    return bars


class FiveMinuteTrend(Strategy):
    """按 5 分钟窗口打印近 3 根收盘价, 用来展示 subscribe_bars 的实盘闭合时机."""

    def __init__(self) -> None:
        """订阅 5 分钟窗口(按 A 股早/午盘会话切分)."""
        super().__init__()
        self.subscribe_bars(
            "5min", session_windows=[("09:30", "11:30"), ("13:00", "15:00")]
        )
        self.count = 0

    def on_start(self) -> None:
        """打印基础周期, 验证行情网关声明的 freq 已注入策略."""
        self.set_history_depth(50)
        print(f"基础周期 self.freq = {self.freq!r}")

    def on_window_bar(self, bar: Bar) -> None:
        """5 分钟窗口闭合回调: 打印窗口收盘价与最近 3 根窗口历史."""
        self.count += 1
        closes = self.get_history(3, bar.symbol, "close")  # 回调内自动定档到 5min
        print(
            f"[5min] {self.format_time(bar.timestamp)} "
            f"C={bar.close:.2f} 近3根={closes.round(2).tolist()}"
        )

    def on_stop(self) -> None:
        """会话结束: 汇报共收到多少根 5 分钟窗口 bar."""
        print(f"共收到 {self.count} 根 5 分钟窗口 bar")


def main() -> None:
    """构造策略与 replay 行情源, 跑一段离线可复现的实盘会话."""
    strategy = FiveMinuteTrend()
    run_live(
        strategy_cls=strategy,
        instruments=[
            Instrument(
                symbol=SYMBOL,
                asset_type=AssetType.Stock,
                multiplier=1.0,
                margin_ratio=1.0,
                tick_size=0.01,
                lot_size=100,
                option_type=None,
                strike_price=None,
                expiry_date=None,
            )
        ],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": build_bars(), "freq": "1min"},
        duration="60s",
    )


if __name__ == "__main__":
    main()
