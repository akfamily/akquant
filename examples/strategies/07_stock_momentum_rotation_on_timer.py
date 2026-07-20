"""多股票轮动策略示例（on_timer 固定时点版本）."""

from typing import Any

import akquant as aq
import pandas as pd
from akquant import Strategy


def _build_symbol_df(
    symbol: str, timestamps: list[pd.Timestamp], closes: list[float]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ts, close in zip(timestamps, closes):
        rows.append(
            {
                "date": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 10000.0,
                "symbol": symbol,
            }
        )
    return pd.DataFrame(rows)


def make_data() -> dict[str, pd.DataFrame]:
    """构造可稳定触发轮动的示例数据."""

    def aaa_close(index: int) -> float:
        if index < 80:
            return 10.0 + index * 0.03
        if index < 160:
            return 10.0 + (160 - index) * 0.03
        return 10.0 + (index - 160) * 0.02

    def bbb_close(index: int) -> float:
        if index < 120:
            return 10.0 + (120 - index) * 0.02
        return 10.0 + (index - 120) * 0.035

    timestamps = list(
        pd.date_range("2022-01-04 10:00:00", periods=240, freq="B", tz="Asia/Shanghai")
    )
    return {
        "AAA": _build_symbol_df("AAA", timestamps, [aaa_close(i) for i in range(240)]),
        "BBB": _build_symbol_df("BBB", timestamps, [bbb_close(i) for i in range(240)]),
    }


class OnTimerMomentumRotationStrategy(Strategy):
    """使用 on_timer 在固定时点执行横截面轮动."""

    def __init__(self, lookback_period: int = 5, **kwargs: Any) -> None:
        """初始化策略参数."""
        _ = kwargs
        super().__init__()
        self.lookback_period = lookback_period
        self.symbols = ["AAA", "BBB"]
        self.warmup_period = lookback_period + 1

    def on_start(self) -> None:
        """策略启动时注册固定时点调仓定时器."""
        for symbol in self.symbols:
            self.subscribe(symbol)
        self.schedule_daily("10:00:00", "rebalance")
        self.log(
            "on_start "
            f"subscribe={self.symbols} "
            "timer=10:00:00 "
            f"lookback={self.lookback_period}"
        )

    def on_timer(self, payload: str) -> None:
        """固定时点触发调仓."""
        if payload != "rebalance":
            return
        history_map = self.get_history_map(
            count=self.lookback_period,
            symbols=self.symbols,
            field="close",
        )
        scores: dict[str, float] = {}
        for symbol, closes in history_map.items():
            if len(closes) < self.lookback_period:
                continue
            start = float(closes[0])
            end = float(closes[-1])
            if start <= 0:
                continue
            scores[symbol] = (end - start) / start
        if not scores:
            return
        selected = self.rebalance_to_topn(
            scores=scores,
            top_n=1,
            weight_mode="score",
            long_only=False,
            liquidate_unmentioned=True,
        )
        self.log(f"on_timer action=rebalance selected={selected}")


if __name__ == "__main__":
    data_map = make_data()
    result = aq.run_backtest(
        data=data_map,
        strategy=OnTimerMomentumRotationStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
    )
    print(result)
