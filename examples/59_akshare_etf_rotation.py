"""AKShare + ETF 轮动最小示例.

本示例演示 AKQuant 推荐的多标写法：
1. 用 AKShare 拉取多只 ETF 日线数据；
2. 拼接成单个 DataFrame；
3. 在 `on_before_trading` 中做横截面动量轮动。

运行前请确保：
- 已安装 `akshare`
- 网络可访问 AKShare 对应数据源
"""

from typing import Any, cast

import akquant as aq
import akshare as ak
import pandas as pd
from akquant import IntParam, Strategy

ETF_SYMBOLS = ["510300", "510500", "159915"]
ETF_NAMES = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
}


def to_sina_etf_symbol(symbol: str) -> str:
    """把 6 位 ETF 代码转换为新浪接口需要的带市场前缀代码.

    上交所 ETF 以 5 开头（如 510300），深交所 ETF 以 1 开头（如 159915）。
    """
    return f"sh{symbol}" if symbol.startswith("5") else f"sz{symbol}"


def fetch_etf_history(
    symbol: str,
    start_date: str = "20200101",
    end_date: str = "20251231",
) -> pd.DataFrame:
    """获取单只 ETF 历史行情并标准化为 AKQuant 可直接回测的列结构.

    使用新浪数据源 ``fund_etf_hist_sina``（东方财富接口 ``fund_etf_hist_em``
    近期常被断开连接）。新浪接口一次返回全部历史、不支持日期区间与复权参数，
    这里在本地按日期截取。价格为不复权价格，除息日会有小幅跳空；
    正式研究请自行复权。
    """
    df = ak.fund_etf_hist_sina(symbol=to_sina_etf_symbol(symbol))
    if df.empty:
        raise ValueError(f"No ETF data returned for {symbol}")

    required_columns = ["date", "open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"ETF data for {symbol} is missing columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required_columns].copy()
    df["date"] = pd.to_datetime(df["date"])
    in_range = df["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    df = df.loc[in_range].copy()
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

    lookback_period = IntParam(20, ge=1, le=500, title="动量回看周期")
    top_n = IntParam(1, ge=1, title="持有标的数")

    def on_start(self) -> None:
        """初始化轮动标的列表与预热窗口，并订阅所有标的."""
        self.symbols = ETF_SYMBOLS.copy()
        self.warmup_period = self.params.lookback_period + 1
        for symbol in self.symbols:
            self.subscribe(symbol)
        self.log(
            "rotation universe="
            + ", ".join(
                f"{symbol}:{ETF_NAMES.get(symbol, symbol)}" for symbol in self.symbols
            )
        )

    def on_before_trading(self, trading_date: Any, timestamp: int) -> None:
        """交易日级横截面轮动."""
        _ = timestamp
        history_map = self.get_history_map(
            count=self.params.lookback_period,
            symbols=self.symbols,
            field="close",
        )
        scores: dict[str, float] = {}
        for symbol, closes in history_map.items():
            if len(closes) < self.params.lookback_period:
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
            )[: self.params.top_n]
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
        # 传入真实沪深交易日历: 不传时引擎按工作日推算, 节假日会被当成交易日
        trading_calendar=ak.tool_trade_date_hist_sina()["trade_date"],
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
