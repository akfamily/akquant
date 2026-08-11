"""
第 12 章：机器学习在量化中的应用 (Machine Learning)（函数式写法）.

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

本示例与 ch12_ml.py 的交易逻辑、参数、数据源完全一致，
只把策略入口从类风格换成函数式：

1. __init__ 里的状态 (train_window / warmup_period / model / scaler /
   last_train_time / _bar_count) -> ctx 属性（在 initialize 中赋值）
2. Strategy.calculate_features 方法 -> 模块级 calculate_features(df) 纯函数
   （它不读写策略状态，因此不需要 ctx 参数）
3. Strategy.train_model 方法 -> 模块级 train_model(ctx, symbol) 函数
   （它要读写模型状态，因此显式接收 ctx）
4. Strategy.on_bar 方法 -> 模块级 on_bar(ctx, bar) 函数
5. strategy=MLStrategy -> strategy=on_bar + initialize=initialize

跨 Bar 的状态 (model / scaler / _bar_count) 直接挂在 ctx 上读写，
作用与类风格的 self 属性一致。注意计数器必须沿用 `_bar_count` 这个名字，
不能「顺手」改成 ctx.bar_count —— 详见 initialize 的 docstring。
本示例同样保留 `np.random.seed(42)` 与 `LogisticRegression(random_state=42)`，
两份示例的回测统计输出应完全一致，这说明函数式与类风格在引擎层等价。
"""

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar

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


def initialize(ctx: Any) -> None:
    """
    初始化 ML 策略状态，等价于类风格的 __init__.

    关键差异：函数式没有类体，warmup_period 必须在这里挂到 ctx 上。
    引擎会取 ctx.warmup_period 与 run_backtest(warmup_period=...) 的较大值，
    所以在此赋值即可生效。

    注意：引擎还有一路“按指标调用自动推断 warmup”的机制，但它靠 AST 解析策略
    类体实现；函数式下被解析的是引擎内部的 FunctionalStrategy 而非本文件的
    on_bar，推断结果恒为 0。因此函数式必须像这样显式赋值，不能依赖自动推断。

    另一个坑：计数器必须沿用类风格版的 `_bar_count` 名字。它看着像「私有状态」，
    实际是**引擎自己维护的字段**——引擎在每根 Bar 派发 on_bar 之前就会把它 +1
    （见 akquant/strategy_events.py 里 `strategy._bar_count += 1`），策略里再
    `+= 1` 相当于一根 Bar 累加两次，且预热期内引擎已先累加过一截。
    类风格版的 `% 20 == 0` 重训练节奏正建立在这个叠加后的计数上，
    改名成 ctx.bar_count 会得到一个「干净」的计数器，重训练次数会从 40 次掉到
    1 次，两版输出随之分叉。孪生版以复现类风格版行为为准，故如实沿用原名。
    """
    ctx.train_window = 200  # 训练窗口长度 (例如使用过去 200 天训练)
    # 预热期需要比训练窗口稍长，确保特征计算无空值
    ctx.warmup_period = ctx.train_window + 20

    ctx.model = None
    ctx.scaler = None

    # 记录最近一次训练的时间
    ctx.last_train_time = None
    ctx._bar_count = 0


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程函数：计算技术指标作为特征（纯函数，不需要 ctx）."""
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


def train_model(ctx: Any, symbol: str) -> None:
    """在线训练模型；类风格的 self.train_model(symbol) 在此显式接收 ctx."""
    if not HAS_SKLEARN:
        return

    # 获取历史数据
    # 我们需要 train_window + 额外一些 buffer 来计算指标
    df = ctx.get_history_df(count=ctx.train_window + 50, symbol=symbol)

    if len(df) < ctx.train_window:
        return

    # 准备数据
    data = calculate_features(df)

    # 使用最近 train_window 条数据进行训练
    train_data = data.iloc[-ctx.train_window :]

    feature_cols = ["returns_1", "returns_5", "ma_dist_20"]
    X = train_data[feature_cols]
    y = train_data["target"]

    # 标准化
    ctx.scaler = StandardScaler()
    X_scaled = ctx.scaler.fit_transform(X)

    # 训练逻辑回归模型
    ctx.model = LogisticRegression(random_state=42)
    ctx.model.fit(X_scaled, y)

    # 打印训练集准确率
    score = ctx.model.score(X_scaled, y)
    ctx.log(f"模型重训练完成 (样本数={len(train_data)}), 准确率={score:.2%}")


def on_bar(ctx: Any, bar: Bar) -> None:
    """收到 Bar 事件的回调，等价于类风格的 on_bar；self 全部换成 ctx."""
    if not HAS_SKLEARN:
        dt = pd.to_datetime(bar.timestamp)
        if dt.day == 1:  # 每月提醒一次
            ctx.log("未安装 scikit-learn，无法运行 ML 策略")
        return

    symbol = bar.symbol

    # 1. 定期重训练 (例如每月初)
    # 这里简化为：每隔 20 个交易日训练一次
    # 注意：在实盘中通常在盘后训练，这里为了演示放在盘中
    # 注意 _bar_count 是引擎共用字段，不要改名（见 initialize 的 docstring）
    ctx._bar_count += 1

    if ctx.model is None or ctx._bar_count % 20 == 0:
        train_model(ctx, symbol)

    if ctx.model is None:
        return

    # 2. 实时预测
    # 获取最新的特征数据
    # 我们需要最近的一小段历史来计算当天的因子
    recent_df = ctx.get_history_df(count=30, symbol=symbol)
    if len(recent_df) < 30:
        return

    # 计算当天的特征
    # 注意：calculate_features 内部会有 dropna，所以要确保输入足够长
    features_df = calculate_features(recent_df)

    if features_df.empty:
        return

    # 取最后一行 (即当前 Bar 的特征)
    current_features = features_df.iloc[[-1]][["returns_1", "returns_5", "ma_dist_20"]]

    # 标准化
    X_curr = ctx.scaler.transform(current_features)

    # 预测概率
    # proba[0][1] 是预测为 1 (涨) 的概率
    prob_up = ctx.model.predict_proba(X_curr)[0][1]

    # 3. 交易逻辑
    pos = ctx.get_position(symbol)

    # 阈值设置：预测概率 > 0.55 才买入，< 0.45 卖出
    if prob_up > 0.55 and pos == 0:
        ctx.log(f"预测上涨概率 {prob_up:.2%} > 55%，买入")
        ctx.order_target_percent(symbol=symbol, target_percent=0.95)

    elif prob_up < 0.45 and pos > 0:
        ctx.log(f"预测上涨概率 {prob_up:.2%} < 45%，卖出")
        ctx.close_position(symbol)


if __name__ == "__main__":
    if not HAS_SKLEARN:
        print("请先安装 scikit-learn: pip install scikit-learn")
    else:
        df = generate_mock_data()

        print("开始运行第 12 章 ML 策略示例...")
        result = aq.run_backtest(
            strategy=on_bar,
            initialize=initialize,
            data=df,
            initial_cash=100_000,
            commission_rate=0.0003,
        )

        # 打印最终结果
        metrics = result.metrics_df
        end_value = (
            metrics.loc["end_market_value", "value"]
            if "end_market_value" in metrics.index
            else 0.0
        )
        print(f"回测结束，最终权益: {float(str(end_value)):.2f}")
