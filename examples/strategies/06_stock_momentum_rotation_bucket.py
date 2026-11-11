"""多股票轮动策略示例（on_bars：每个时间片的 bar 到齐后执行一次）.

不要在 on_bar 里自己数“攒满 N 根”再执行：有标的停牌或缺 bar 时那一片凑不齐，
整片都不会触发。on_bars 由框架判断时间片何时结束，缺的标的列在 batch.missing。
"""

import math

import akquant as aq
import akshare as ak
import pandas as pd
from akquant import BarBatch, IntParam, Strategy


class BucketMomentumRotationStrategy(Strategy):
    """每个时间片的 bar 到齐后执行一次横截面轮动."""

    lookback_period = IntParam(20, ge=1, le=500, title="动量回看周期")

    def on_start(self) -> None:
        """设置轮动标的列表与数据预热长度."""
        self.symbols = ["sh600519", "sz000858", "sh601318"]
        self.warmup_period = self.params.lookback_period + 1

    def on_bars(self, batch: BarBatch) -> None:
        """批量回调：本时间片的 bar 到齐后调仓一次，只对本片有 bar 的标的打分."""
        scores: dict[str, float] = {}
        for bar in batch:
            symbol = bar.symbol
            closes = self.get_history(
                count=self.params.lookback_period, symbol=symbol, field="close"
            )
            start = float(closes[0])
            end = float(closes[-1])
            # 历史不足时前面补 NaN
            if math.isnan(start) or start <= 0:
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
