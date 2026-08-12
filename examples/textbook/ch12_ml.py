"""
第 12 章：机器学习在量化中的应用 (Machine Learning).

本示例展示了如何将机器学习 (ML) 融入到 AKQuant 策略中：
1. **特征工程 (Feature Engineering)**：构造滞后收益率、均线偏离度等因子。
2. **滚动训练 (Rolling Window)**：使用过去 N 天的数据训练模型。
3. **实时预测 (Real-time Prediction)**：使用训练好的模型对当前 Bar 进行预测。

示例模型：
- 使用 scikit-learn 的 LogisticRegression (逻辑回归) 预测次日涨跌。
- 目标变量 (Label)：次日收益率 > 0 (1: 涨, 0: 跌/平)。
- 特征 (Features)：
    - returns_1: 过去 1 天的收益率
    - returns_5: 过去 5 天的收益率
    - ma_dist_20: 当前价格相对于 20 日均线的偏离度

注意：由于 ML 模型训练较慢，本示例为了演示仅使用简单的线性模型。
实际生产中推荐使用 LightGBM/XGBoost，并配合 AKQuant 的 `run_walk_forward` 进行滚动回测。
"""

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, IntParam, Strategy

# 尝试导入 sklearn，如果未安装则跳过
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# 模拟数据生成
def generate_mock_data(length: int = 1000) -> pd.DataFrame:
    """生成模拟数据."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=length, freq="D")

    # 构造一些有规律的信号 (动量效应)
    # 如果前一天涨，今天大概率涨
    returns = np.random.randn(length) * 0.01
    for i in range(1, length):
        if returns[i - 1] > 0:
            returns[i] += 0.005  # 增加正向动量
        else:
            returns[i] -= 0.005

    prices = 100 * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": 100000,
            "symbol": "MOCK_ML",
        }
    )
    return df


class MLStrategy(Strategy):
    """机器学习演示策略."""

    train_window = IntParam(200, ge=10, title="训练窗口")

    def on_start(self) -> None:
        """回测开始时触发. 设置预热期并初始化模型、计数等内部状态."""
        # 预热期需要比训练窗口稍长，确保特征计算无空值
        self.warmup_period = self.params.train_window + 20

        self.model: Any = None
        self.scaler: Any = None

        # 记录最近一次训练的时间
        self.last_train_time = None
        self._bar_count = 0

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程函数：计算技术指标作为特征."""
        df = df.copy()

        # 1. 计算特征 (X)
        df["returns_1"] = df["close"].pct_change(1)
        df["returns_5"] = df["close"].pct_change(5)

        ma20 = df["close"].rolling(20).mean()
        df["ma_dist_20"] = (df["close"] - ma20) / ma20

        # 2. 计算目标变量 (y)
        # 预测目标：次日收益率是否 > 0
        # shift(-1) 是将未来的收益率前移到今天，作为今天的 label
        df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)

        # dropna 会删除包含 NaN 的行。但对于最后一行(当前Bar)，
        # 虽然 target 可能不准确(因为不知道明天价格)，但特征是完整的，
        # 我们需要保留它用于实时预测 (Real-time Prediction)。
        # 只有在训练时才需要丢弃 target 无效的行。
        # 这里为了演示简单，我们仅丢弃特征计算产生的 NaN (前几行)。
        return df.dropna(subset=["returns_1", "returns_5", "ma_dist_20"])  # type: ignore

    def train_model(self, symbol: str) -> None:
        """在线训练模型."""
        if not HAS_SKLEARN:
            return

        # 获取历史数据
        # 我们需要 train_window + 额外一些 buffer 来计算指标
        df = self.get_history_df(count=self.params.train_window + 50, symbol=symbol)

        if len(df) < self.params.train_window:
            return

        # 准备数据
        data = self.calculate_features(df)

        # 使用最近 train_window 条数据进行训练
        train_data = data.iloc[-self.params.train_window :]

        feature_cols = ["returns_1", "returns_5", "ma_dist_20"]
        X = train_data[feature_cols]
        y = train_data["target"]

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 训练逻辑回归模型
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_scaled, y)

        # 打印训练集准确率
        score = self.model.score(X_scaled, y)
        self.log(f"模型重训练完成 (样本数={len(train_data)}), 准确率={score:.2%}")

    def on_bar(self, bar: Bar) -> None:
        """收到 Bar 事件的回调."""
        if not HAS_SKLEARN:
            dt = pd.to_datetime(bar.timestamp)
            if dt.day == 1:  # 每月提醒一次
                self.log("未安装 scikit-learn，无法运行 ML 策略")
            return

        symbol = bar.symbol

        # 1. 定期重训练 (例如每月初)
        # 这里简化为：每隔 20 个交易日训练一次
        # 注意：在实盘中通常在盘后训练，这里为了演示放在盘中
        self._bar_count += 1

        if self.model is None or self._bar_count % 20 == 0:
            self.train_model(symbol)

        if self.model is None:
            return

        # 2. 实时预测
        # 获取最新的特征数据
        # 我们需要最近的一小段历史来计算当天的因子
        recent_df = self.get_history_df(count=30, symbol=symbol)
        if len(recent_df) < 30:
            return

        # 计算当天的特征
        # 注意：calculate_features 内部会有 dropna，所以要确保输入足够长
        features_df = self.calculate_features(recent_df)

        if features_df.empty:
            return

        # 取最后一行 (即当前 Bar 的特征)
        current_features = features_df.iloc[[-1]][
            ["returns_1", "returns_5", "ma_dist_20"]
        ]

        # 标准化
        X_curr = self.scaler.transform(current_features)

        # 预测概率
        # proba[0][1] 是预测为 1 (涨) 的概率
        prob_up = self.model.predict_proba(X_curr)[0][1]

        # 3. 交易逻辑
        pos = self.get_position(symbol)

        # 阈值设置：预测概率 > 0.55 才买入，< 0.45 卖出
        if prob_up > 0.55 and pos == 0:
            self.log(f"预测上涨概率 {prob_up:.2%} > 55%，买入")
            self.order_target_percent(symbol=symbol, target_percent=0.95)

        elif prob_up < 0.45 and pos > 0:
            self.log(f"预测上涨概率 {prob_up:.2%} < 45%，卖出")
            self.close_position(symbol)


if __name__ == "__main__":
    if not HAS_SKLEARN:
        print("请先安装 scikit-learn: pip install scikit-learn")
    else:
        df = generate_mock_data()

        print("开始运行第 12 章 ML 策略示例...")
        result = aq.run_backtest(
            strategy=MLStrategy, data=df, initial_cash=100_000, commission_rate=0.0003
        )

        # 打印最终结果
        metrics = result.metrics_df
        end_value = (
            metrics.loc["end_market_value", "value"]
            if "end_market_value" in metrics.index
            else 0.0
        )
        print(f"回测结束，最终权益: {float(str(end_value)):.2f}")
