from typing import Any, Tuple

import numpy as np
import pandas as pd
from akquant import Bar, CurrentClose
from akquant.backtest import run_backtest
from akquant.ml import SklearnAdapter
from akquant.strategy import Strategy
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class WalkForwardStrategy(Strategy):
    """演示策略：使用逻辑回归预测涨跌 (集成 Pipeline 预处理)."""

    def __init__(self) -> None:
        """初始化策略."""
        # 1. 初始化模型 (使用 Pipeline 封装预处理和模型)
        # StandardScaler: 确保使用训练集统计量进行标准化，防止数据泄露
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression())]
        )

        self.model = SklearnAdapter(pipeline)

        # 2. 配置 Walk-forward Validation
        # 框架会自动接管数据的切割、模型的重训
        self.model.set_validation(
            method="walk_forward",
            train_window=50,  # 使用过去 50 个 bar 训练
            rolling_step=10,  # 每 10 个 bar 重训一次
            frequency="1m",  # 数据频率
            verbose=True,  # 打印训练日志
        )

        # 确保历史数据长度足够 (训练窗口 + 特征计算所需窗口)
        self.set_history_depth(60)

    def prepare_features(
        self, df: pd.DataFrame, mode: str = "training"
    ) -> Tuple[pd.DataFrame, Any]:
        """
        [必须实现] 特征工程逻辑.

        该函数会被用于训练阶段（生成 X, y）和预测阶段（生成 X）
        """
        X = pd.DataFrame()
        # 特征 1: 1周期收益率
        X["ret1"] = df["close"].pct_change()
        # 特征 2: 2周期收益率
        X["ret2"] = df["close"].pct_change(2)
        # X = X.fillna(0)  # REMOVED: fillna(0) pollutes training data

        if mode == "inference":
            # 推理模式：只返回最后一行特征，不需要 y
            # 注意：inference 时传入的 df 是最近 history_depth 的数据
            # 最后一行是最新的 bar，我们需要它的特征
            # 但是 pct_change 会导致前几行是 NaN，这没关系，只要最后一行有效即可
            return X.iloc[-1:], None

        # 训练模式：构造标签 y (预测下一期的涨跌)
        # shift(-1) 把未来的收益挪到当前行作为 label
        future_ret = df["close"].pct_change().shift(-1)

        # Combine into one DataFrame to align drops
        data = pd.concat([X, future_ret.rename("future_ret")], axis=1)

        # Drop rows with NaN features (e.g. from history padding or initial pct_change)
        data = data.dropna(subset=["ret1", "ret2"])

        # For training, we must have a valid future return
        data = data.dropna(subset=["future_ret"])

        # Calculate y on valid data
        y = (data["future_ret"] > 0).astype(int)
        X_clean = data[["ret1", "ret2"]]

        return X_clean, y

    def on_bar(self, bar: Bar) -> None:
        """
        Bar 数据回调.

        :param bar: Bar 对象
        """
        # 3. 实时预测与交易

        # 获取最近的数据进行特征提取
        # 注意：需要足够的历史长度来计算特征 (例如 pct_change(2) 需要至少3根bar)
        hist_df = self.get_history_df(10)

        # 复用特征计算逻辑！
        # 直接调用 prepare_features 获取当前特征
        X_curr, _ = self.prepare_features(hist_df, mode="inference")

        try:
            # 获取预测信号 (概率)
            # SklearnAdapter 对于二分类返回 Class 1 的概率
            if self.model:
                signal = self.model.predict(X_curr)[0]

                # 打印信号方便观察
                # print(f"Time: {bar.timestamp}, Signal: {signal:.4f}")

                # 结合风控规则下单
                if signal > 0.55:
                    self.buy(bar.symbol, 100)
                elif signal < 0.45:
                    self.sell(bar.symbol, 100)

        except Exception:
            # 模型可能尚未初始化或训练失败
            pass


if __name__ == "__main__":
    # 1. 生成合成数据
    print("生成测试数据...")
    dates = pd.date_range(start="2023-01-01", periods=500, freq="1min")
    # 随机漫步价格
    price = 100 + np.cumsum(np.random.randn(500))
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 1000,
            "symbol": "TEST",
        }
    )

    # 2. 运行回测
    print("开始运行机器学习回测...")
    result = run_backtest(
        data=df,
        strategy=WalkForwardStrategy,
        symbols="TEST",
        lot_size=1,
        fill_policy=CurrentClose(),
        history_depth=60,
        warmup_period=50,
    )
    print("回测结束。")

    # 3. 打印结果
    print(result)
