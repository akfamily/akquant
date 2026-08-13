from collections import defaultdict

import akquant as aq
import pandas as pd
from akquant import Bar, ListParam, Strategy


def _build_symbol_df(
    symbol: str, timestamps: list[pd.Timestamp], closes: list[float]
) -> pd.DataFrame:
    rows = []
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
    """Build deterministic multi-symbol demo data."""
    timestamps = list(
        pd.date_range("2024-01-02 10:00:00", periods=6, freq="D", tz="Asia/Shanghai")
    )
    return {
        "AAA": _build_symbol_df(
            "AAA", timestamps, [10.0, 10.2, 10.4, 10.5, 10.4, 10.3]
        ),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 9.9, 9.8, 9.9, 10.0, 10.1]),
        "CCC": _build_symbol_df("CCC", timestamps, [10.0, 10.1, 10.0, 9.9, 9.8, 9.7]),
    }


class TargetWeightsRebalanceStrategy(Strategy):
    """TopN 动态权重调仓示例策略."""

    symbols = ListParam([], item_type=str, title="调仓标的集")

    def on_start(self) -> None:
        """初始化横截面计算所需状态."""
        self.pending: dict[int, set[str]] = defaultdict(set)
        self.lookback = 3
        self.top_n = 2
        self.target_total_weight = 0.9
        self.selected_history: list[tuple[int, list[str], dict[str, float]]] = []

    def on_bar(self, bar: Bar) -> None:
        """收齐同一时间切片后，按动量选 TopN 并调仓."""
        # on_bar 会按 symbol 逐个触发，这里先缓存，确保每个时间点只调仓一次
        bucket = self.pending[bar.timestamp]
        bucket.add(bar.symbol)
        if len(bucket) < len(self.params.symbols):
            return
        self.pending.pop(bar.timestamp, None)

        # 1) 计算每个标的的简单动量：最新收盘 / 窗口首收盘 - 1
        scores: dict[str, float] = {}
        for symbol in self.params.symbols:
            closes = self.get_history(count=self.lookback, symbol=symbol, field="close")
            if len(closes) < self.lookback:
                return
            first_close = float(closes[0])
            last_close = float(closes[-1])
            if first_close <= 0:
                return
            scores[symbol] = last_close / first_close - 1.0

        # 2) 选出 TopN 强势标的
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        selected = [symbol for symbol, _ in ranked[: self.top_n]]
        if not selected:
            return

        # 3) 生成目标权重：TopN 等权，总仓位上限由 target_total_weight 控制
        each_weight = self.target_total_weight / float(len(selected))
        target_weights = {symbol: each_weight for symbol in selected}

        # 4) 一次调用完成组合调仓
        #    liquidate_unmentioned=True 会把不在 target_weights 的持仓清到 0
        self.rebalance_weights(
            target_weights=target_weights,
            liquidate_unmentioned=True,
            rebalance_tolerance=0.01,
        )
        self.selected_history.append((bar.timestamp, selected, target_weights))


def main() -> None:
    """运行示例并打印调仓轨迹与最终状态."""
    symbols = ["AAA", "BBB", "CCC"]
    result = aq.run_backtest(
        data=make_data(),
        strategy=TargetWeightsRebalanceStrategy,
        symbols=symbols,
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy=aq.CurrentClose(),
        # get_history 依赖历史缓存深度，这里设置为 lookback 对应长度
        history_depth=3,
        show_progress=False,
    )

    strategy = result.strategy
    if strategy is not None:
        print("selected_history")
        for ts, selected, weights in strategy.selected_history:
            local_ts = pd.Timestamp(ts, tz="UTC").tz_convert("Asia/Shanghai")
            print(f"{local_ts} selected={selected} weights={weights}")

    print("final_positions")
    print(result.positions.iloc[-1])
    print("final_equity")
    print(float(result.equity_curve.iloc[-1]))


if __name__ == "__main__":
    main()
