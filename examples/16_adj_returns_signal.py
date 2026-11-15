import akshare as ak
import pandas as pd
from akquant import Bar, Strategy, run_backtest


class AdjSignal(Strategy):
    """
    使用前复权收盘价作为信号、三轴默认策略（open + bar_offset=1）的示例策略.

    :ivar warmup_period: 暖启动所需的 Bar 数；在此期间仅更新历史，不执行交易
    """

    warmup_period = 5

    def on_bar(self, bar: Bar) -> None:
        """
        每根 Bar 到来时触发，基于最近两根前复权收盘价计算信号.

        :param bar: 当前的行情 Bar（估值使用真实收盘价 close；撮合采用三轴默认策略）
        :type bar: Bar
        """
        try:
            # 从历史中取最近 2 根“前复权收盘价”序列（不包含当前 Bar，避免前视偏差）
            adj_val = bar.extra.get("adj_close", None)
            self.log(f"{bar.timestamp_iso}, {bar.close}, {adj_val}")
            x = self.get_history(2, bar.symbol, "adj_close")
        except Exception:
            # 如果数据字段不存在或不足，直接跳过
            return

        if x is None or len(x) < 2:
            return

        # 以“前复权收盘价”计算最新一天的简单收益率
        # r_t = adj_close_t / adj_close_{t-1} - 1
        r = x[-1] / x[-2] - 1.0

        # 查询当前持仓
        pos = self.get_position(bar.symbol)

        # 简单信号：r > 0 且空仓 -> 开多；r < 0 且有多头 -> 平仓
        if pos == 0 and r > 0:
            self.buy(bar.symbol, 100)
        elif pos > 0 and r < 0:
            self.close_position(bar.symbol)


# ----------------------- 数据准备：AKShare -----------------------

# 1) 未复权日线（真实 close），用于撮合和估值
df_raw = ak.stock_zh_a_hist_sina(symbol="sz000001", start_date="20200101")
if "date" not in df_raw.columns:
    # 部分接口会把日期放在索引上，这里统一还原为列
    df_raw = df_raw.reset_index().rename(columns={"index": "date"})
# 列名统一为小写，便于后续标准化
df_raw.columns = [c.lower() for c in df_raw.columns]
# 若存在 time 列但无 date 列，统一改名为 date
if "time" in df_raw.columns and "date" not in df_raw.columns:
    df_raw = df_raw.rename(columns={"time": "date"})
# 时区本地化为 Asia/Shanghai（AKQuant 规范）
df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.tz_localize("Asia/Shanghai")
# 必须包含 symbol 列（即使只有单标的）
df_raw["symbol"] = "000001"
# 只保留 AKQuant 需要的标准列
df_raw = df_raw[["date", "open", "high", "low", "close", "volume", "symbol"]]

# 2) 前复权日线（用于信号），重命名为 adj_close
df_adj = ak.stock_zh_a_hist_sina(symbol="sz000001", adjust="qfq", start_date="20200101")
if "date" not in df_adj.columns:
    df_adj = df_adj.reset_index().rename(columns={"index": "date"})
df_adj.columns = [c.lower() for c in df_adj.columns]
if "time" in df_adj.columns and "date" not in df_adj.columns:
    df_adj = df_adj.rename(columns={"time": "date"})
df_adj["date"] = pd.to_datetime(df_adj["date"]).dt.tz_localize("Asia/Shanghai")
df_adj = df_adj[["date", "close"]].rename(columns={"close": "adj_close"})

# 3) 合并两者：既包含真实 close（撮合/估值），又包含 adj_close（信号计算）
df = (
    pd.merge(df_raw, df_adj, on="date", how="inner")
    .sort_values("date")
    .reset_index(drop=True)
)

# ----------------------- 运行回测 -----------------------
result = run_backtest(
    data=df,
    strategy=AdjSignal,  # 传类或实例均可
    lot_size=100,  # A 股常见 1 手 = 100 股
)
print(result)
