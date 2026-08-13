"""多股票轮动策略示例（收齐同 timestamp 后执行）."""

from collections import defaultdict

import akquant as aq
import akshare as ak
import pandas as pd
from akquant import Bar, IntParam, Strategy


class BucketMomentumRotationStrategy(Strategy):
    """收齐同一时间片所有标的后执行横截面轮动."""

    lookback_period = IntParam(20, ge=1, le=500, title="动量回看周期")

    def on_start(self) -> None:
        """设置轮动标的列表、数据预热长度与同 timestamp 收齐缓存."""
        self.symbols = ["sh600519", "sz000858", "sh601318"]
        self.warmup_period = self.params.lookback_period + 1
        self._pending_by_ts: dict[int, set[str]] = defaultdict(set)

    def on_bar(self, bar: Bar) -> None:
        """Bar 回调，收齐该 timestamp 后再调仓."""
        current_set = self._pending_by_ts[bar.timestamp]
        current_set.add(bar.symbol)
        if len(current_set) < len(self.symbols):
            return
        self._pending_by_ts.pop(bar.timestamp, None)
        self._rebalance()

    def _rebalance(self) -> None:
        scores: dict[str, float] = {}
        for symbol in self.symbols:
            closes = self.get_history(
                count=self.params.lookback_period, symbol=symbol, field="close"
            )
            if len(closes) < self.params.lookback_period:
                return
            start = float(closes[0])
            end = float(closes[-1])
            if start <= 0:
                continue
            scores[symbol] = (end - start) / start

        if not scores:
            return

        best_symbol = max(scores, key=lambda symbol: scores[symbol])
        best_score = scores[best_symbol]

        if best_score < 0:
            for symbol in self.symbols:
                if self.get_position(symbol) > 0:
                    self.close_position(symbol)
            return

        current_holding: str | None = None
        for symbol in self.symbols:
            if self.get_position(symbol) > 0:
                current_holding = symbol
                break

        if current_holding is None:
            self.order_target_percent(target_percent=0.95, symbol=best_symbol)
            return

        if current_holding != best_symbol:
            self.close_position(current_holding)
            self.order_target_percent(target_percent=0.95, symbol=best_symbol)


if __name__ == "__main__":
    symbols = ["sh600519", "sz000858", "sh601318"]
    data_map: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        df = ak.stock_zh_a_daily(
            symbol=symbol, start_date="20220101", end_date="20231231", adjust="qfq"
        )
        if df.empty:
            continue
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        df = df.sort_values("date").reset_index(drop=True)
        data_map[symbol] = df

    if not data_map:
        raise SystemExit(0)

    result = aq.run_backtest(
        data=data_map,
        strategy=BucketMomentumRotationStrategy,
        symbols=list(data_map.keys()),
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
    )
    print(result)
