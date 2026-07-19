"""多股票轮动策略示例（on_before_trading 版本）."""

import math
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
        "AAA": _build_symbol_df(
            "AAA",
            timestamps,
            [aaa_close(i) for i in range(240)],
        ),
        "BBB": _build_symbol_df(
            "BBB",
            timestamps,
            [bbb_close(i) for i in range(240)],
        ),
    }


def rebalance_to_best_symbol(
    strategy: Strategy, ranking: list[tuple[str, float]], symbols: list[str]
) -> str:
    """调仓到排名第一的标的，其余标的清仓."""
    best_symbol = ranking[0][0]
    for symbol in symbols:
        target_percent = 1.0 if symbol == best_symbol else 0.0
        strategy.order_target_percent(target_percent=target_percent, symbol=symbol)
    return best_symbol


class DailyBoundaryMomentumRotationStrategy(Strategy):
    """使用 on_before_trading 执行前一快照语义的横截面轮动."""

    def __init__(self, lookback_period: int = 5, **kwargs: Any) -> None:
        """初始化策略参数."""
        _ = kwargs
        super().__init__()
        self.lookback_period = lookback_period
        self.symbols = ["AAA", "BBB"]
        self.warmup_period = lookback_period + 1

    def on_start(self) -> None:
        """策略启动时订阅轮动标的."""
        for symbol in self.symbols:
            self.subscribe(symbol)
        self.log(f"on_start subscribe={self.symbols} lookback={self.lookback_period}")

    def on_before_trading(self, trading_date: Any, timestamp: int) -> None:
        """交易日盘前调仓回调（前一快照可见）."""
        _ = timestamp
        self.log(f"on_before_trading date={trading_date}")
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
            if start <= 0 or not math.isfinite(start) or not math.isfinite(end):
                continue
            scores[symbol] = (end - start) / start

        if not scores:
            return

        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        self.log(
            "rebalance ranking="
            + ", ".join(f"{symbol}:{score:.2%}" for symbol, score in ranking)
        )
        best_symbol = rebalance_to_best_symbol(self, ranking, self.symbols)
        self.log(f"action=rebalance selected={best_symbol}")


if __name__ == "__main__":
    symbols = ["AAA", "BBB"]
    data_map = make_data()

    result = aq.run_backtest(
        data=data_map,
        strategy=DailyBoundaryMomentumRotationStrategy,
        symbols=symbols,
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
    )
    print(result)
