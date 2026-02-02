# 机器学习与滚动训练指南 (ML Guide)

AKQuant 内置了一个高性能的机器学习训练框架，专为量化交易设计。它解决了传统框架中常见的“未来函数”泄露问题，并提供了开箱即用的 Walk-forward Validation（滚动/前向验证）支持。

## 核心设计理念

### 1. Signal (信号) 与 Action (决策) 分离

很多初学者容易犯的错误是让模型直接输出“买/卖”指令。在 AKQuant 中，我们将这一过程解耦：

*   **Model (模型层)**: 只负责根据历史数据预测未来的概率或数值 (Signal)。它不知道账户有多少钱，也不知道现在的市场风险如何。
*   **Strategy (策略层)**: 接收模型的 Signal，结合风控规则、资金管理、市场状态，最终做出买卖决策 (Action)。

### 2. Adapter 模式 (适配器)

为了统一 Scikit-learn（传统机器学习）和 PyTorch（深度学习）截然不同的编程范式，我们引入了适配器层：

*   **SklearnAdapter**: 适配 XGBoost, LightGBM, RandomForest 等。
*   **PyTorchAdapter**: 适配 LSTM, Transformer 等深度网络，自动处理 DataLoader 和训练循环。

用户只需要面对统一的 `QuantModel` 接口。

### 3. Walk-forward Validation (滚动验证)

在时间序列数据上，随机的 K-Fold 交叉验证是错误的，因为它会利用未来数据预测过去。正确的做法是 Walk-forward：

1.  **Window 1**: 用 2020 年数据训练，预测 2021 Q1。
2.  **Window 2**: 用 2020 Q2 - 2021 Q1 数据训练，预测 2021 Q2。
3.  ... 像滚轮一样不断向前推进。

---

## 快速开始

### 1. 定义策略

你需要继承 `Strategy` 并实现 `prepare_features` 方法。

```python
from akquant.strategy import Strategy
from akquant.ml import SklearnAdapter
from sklearn.linear_model import LogisticRegression
import pandas as pd

class MyMLStrategy(Strategy):
    def __init__(self):
        super().__init__()

        # 1. 初始化模型
        # 这里可以使用任何 sklearn 兼容的模型，如 RandomForest, XGBoost
        self.model = SklearnAdapter(LogisticRegression())

        # 2. 配置 Walk-forward Validation
        # 框架会自动接管数据的切割、模型的重训和参数冻结
        self.model.set_validation(
            method='walk_forward',
            train_window='1y',   # 每次使用过去 1 年的数据训练
            rolling_step='3m',   # 每 3 个月滚动重训一次
            frequency='1d'       # 数据频率 (用于解析时间字符串)
        )

        # 确保历史数据长度足够
        self.set_history_depth(300)

    def prepare_features(self, df: pd.DataFrame):
        """
        [必须实现] 特征工程逻辑

        :param df: 原始 OHLCV 数据 (DataFrame)
        :return: (X, y) 元组
        """
        # 简单示例: 使用收益率作为特征
        X = pd.DataFrame()
        X['ret1'] = df['close'].pct_change()
        X['ret2'] = df['close'].pct_change(2)
        X = X.fillna(0)

        # 构造标签 y (预测下一期的涨跌)
        # shift(-1) 把未来的收益挪到当前行作为 label
        future_ret = df['close'].pct_change().shift(-1)
        y = (future_ret > 0).astype(int)

        # 注意: 框架只负责按时间窗口切割原始数据 df，不负责特征和标签的对齐
        # 由于 shift(-1) 会导致最后一行 y 为 NaN，必须手动去除
        # 否则 sklearn/pytorch 训练会报错
        return X.iloc[:-1], y.iloc[:-1]

    def on_bar(self, bar):
        # 3. 实时预测与交易
        # 检查模型是否已就绪 (前 1 年的数据用于冷启动训练)
        if self._bar_count < 250:
            return

        # 获取最近的数据进行特征提取
        hist_df = self.get_history_df(5)

        # 这里为了演示，手动构造当期特征 (与 prepare_features 逻辑一致)
        curr_ret1 = (bar.close - hist_df['close'].iloc[-2]) / hist_df['close'].iloc[-2]
        curr_ret2 = (bar.close - hist_df['close'].iloc[-3]) / hist_df['close'].iloc[-3]

        X_curr = pd.DataFrame([[curr_ret1, curr_ret2]], columns=['ret1', 'ret2'])
        X_curr = X_curr.fillna(0)

        try:
            # 获取预测信号 (概率)
            # 对于二分类，返回的是 Class 1 (涨) 的概率
            signal = self.model.predict(X_curr)[0]

            # 结合风控规则下单
            if signal > 0.6:
                self.buy(bar.symbol, 100)
            elif signal < 0.4:
                self.sell(bar.symbol, 100)

        except Exception:
            # 模型可能尚未完成首次训练
            pass
```

### 2. 运行回测

```python
from akquant.backtest import run_backtest

run_backtest(
    strategy=MyMLStrategy,
    symbol="600000",
    start_date="20200101",
    end_date="20231231",
    cash=100000.0
)
```

## API 参考

### `model.set_validation`

配置模型的验证和训练方式。

```python
def set_validation(
    self,
    method: str = 'walk_forward',
    train_window: str | int = '1y',
    test_window: str | int = '3m',
    rolling_step: str | int = '3m',
    frequency: str = '1d'
)
```

*   `method`: 目前仅支持 `'walk_forward'`。
*   `train_window`: 训练窗口长度。支持 `'1y'` (1年), `'6m'` (6个月), `'50d'` (50天) 或整数 (Bar数量)。
*   `rolling_step`: 滚动步长，即每隔多久重训一次模型。
*   `frequency`: 数据的频率，用于将时间字符串正确转换为 Bar 数量 (例如 '1d' 下 1y=252 bars)。

### `strategy.prepare_features`

用户必须实现的回调函数，用于特征工程。

```python
def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]
```

*   **输入**: `df` 是框架根据 `train_window` 自动获取的历史数据。
*   **输出**: `X` (特征矩阵) 和 `y` (标签向量)。
*   **注意**: 这是一个纯函数，不应依赖外部状态。它会被用于训练和（可选的）预测阶段。

## 深度学习支持 (PyTorch)

使用 `PyTorchAdapter` 可以轻松集成深度学习模型：

```python
from akquant.ml import PyTorchAdapter
import torch.nn as nn
import torch.optim as optim

# 定义网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# 在策略中使用
self.model = PyTorchAdapter(
    network=SimpleNet(),
    criterion=nn.BCELoss(),
    optimizer_cls=optim.Adam,
    lr=0.001,
    epochs=20,
    batch_size=64,
    device='cuda'  # 支持 GPU 加速
)
```
