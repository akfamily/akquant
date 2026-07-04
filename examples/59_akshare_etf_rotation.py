"""AKShare + ETF 轮动最小示例.

本示例演示 AKQuant 推荐的多标写法：
1. 用 AKShare 拉取多只 ETF 日线数据；
2. 拼接成单个 DataFrame；
3. 在 `on_daily_rebalance` 中做横截面动量轮动。

运行前请确保：
- 已安装 `akshare`
- 网络可访问 AKShare 对应数据源
"""

from typing import Any, cast

import akquant as aq
import akshare as ak
import pandas as pd
from akquant import Strategy

ETF_SYMBOLS = ["510300", "510500", "159915"]
ETF_NAMES = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
}


def fetch_etf_history(
    symbol: str,
    start_date: str = "20200101",
    end_date: str = "20251231",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """获取单只 ETF 历史行情并标准化为 AKQuant 可直接回测的列结构."""
    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    if df.empty:
        raise ValueError(f"No ETF data returned for {symbol}")

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }
    df = df.rename(columns=rename_map)
    required_columns = ["date", "open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"ETF data for {symbol} is missing columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required_columns].copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df["symbol"] = symbol
    return cast(pd.DataFrame, df.sort_values("date").reset_index(drop=True))


def build_etf_universe_dataframe(
    symbols: list[str],
    start_date: str = "20200101",
    end_date: str = "20251231",
) -> pd.DataFrame:
    """抓取多只 ETF 并拼接为单个 DataFrame（推荐模式）."""
    frames = [
        fetch_etf_history(symbol=symbol, start_date=start_date, end_date=end_date)
        for symbol in symbols
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    return cast(pd.DataFrame, combined)


class ETFMomentumRotationStrategy(Strategy):
    """每日选出动量最强 ETF 并满仓持有."""

    def __init__(
        self, lookback_period: int = 20, top_n: int = 1, **kwargs: Any
    ) -> None:
        """初始化动量窗口、选股数量和订阅标的列表."""
        _ = kwargs
        super().__init__()
        self.lookback_period = lookback_period
        self.top_n = top_n
        self.symbols = ETF_SYMBOLS.copy()
        self.warmup_period = lookback_period + 1

    def on_start(self) -> None:
        """订阅轮动标的."""
        for symbol in self.symbols:
            self.subscribe(symbol)
        self.log(
            "rotation universe="
            + ", ".join(
                f"{symbol}:{ETF_NAMES.get(symbol, symbol)}" for symbol in self.symbols
            )
        )

    def on_daily_rebalance(self, trading_date: Any, timestamp: int) -> None:
        """交易日级横截面轮动."""
        _ = timestamp
        history_map = self.get_history_map(
            count=self.lookback_period,
            symbols=self.symbols,
            field="close",
        )
        scores: dict[str, float] = {}
        for symbol, closes in history_map.items():
            if len(closes) < self.lookback_period:
                continue
            first_close = float(closes[0])
            last_close = float(closes[-1])
            if first_close <= 0:
                continue
            scores[symbol] = last_close / first_close - 1.0

        if not scores:
            self.log(f"skip rebalance on {trading_date}: insufficient history")
            return

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        self.log(
            "ranking=" + ", ".join(f"{symbol}:{score:.2%}" for symbol, score in ranked)
        )

        positive_scores = {
            symbol: score for symbol, score in scores.items() if score > 0
        }
        if not positive_scores:
            self.rebalance_weights(target_weights={}, liquidate_unmentioned=True)
            self.log("all ETF momentums are <= 0, stay in cash")
            return

        selected_symbols = [
            symbol
            for symbol, _ in sorted(
                positive_scores.items(), key=lambda item: item[1], reverse=True
            )[: self.top_n]
        ]
        each_weight = 0.95 / float(len(selected_symbols))
        target_weights = {symbol: each_weight for symbol in selected_symbols}
        self.rebalance_weights(
            target_weights=target_weights,
            liquidate_unmentioned=True,
            rebalance_tolerance=0.01,
        )
        self.log(f"rebalance on {trading_date}: {target_weights}")


def main() -> None:
    """运行 ETF 轮动回测."""
    data = build_etf_universe_dataframe(
        symbols=ETF_SYMBOLS,
        start_date="20200101",
        end_date="20251231",
    )

    result = aq.run_backtest(
        data=data,
        strategy=ETFMomentumRotationStrategy,
        symbols=ETF_SYMBOLS,
        initial_cash=1_000_000.0,
        commission_rate=0.0001,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        timezone="Asia/Shanghai",
        show_progress=False,
    )

    print("loaded_symbols")
    for symbol in ETF_SYMBOLS:
        print(f"{symbol} {ETF_NAMES.get(symbol, symbol)}")

    print("metrics")
    print(
        result.metrics_df.loc[
            [
                "total_return_pct",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown_pct",
            ]
        ]
    )

    if not result.positions.empty:
        print("final_positions")
        print(result.positions.iloc[-1])


if __name__ == "__main__":
    main()
