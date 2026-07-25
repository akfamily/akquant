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

### 4. 防止未来函数 (Look-ahead Bias)

在量化 ML 中，最危险的错误是使用未来数据。AKQuant 建议遵循以下原则：

*   **特征 (X)**: 只能使用当前时刻 $t$ 及之前的数据。
*   **标签 (y)**: 描述 $t+1$ 时刻的状态（如未来的涨跌），但在 $t$ 时刻训练时，我们实际上是拿 $t$ 时刻的 $X$ 去拟合 $t+1$ 时刻的 $y$。
*   **代码实现**: 构造 $y$ 时通常需要 `shift(-1)`，这会导致最后一行数据没有标签（因为没有未来），必须在训练前剔除。

### 5. 防止数据泄露：使用 Pipeline

特征预处理（如标准化、归一化）也可能引入 Look-ahead Bias。例如，如果在全量数据上使用 `StandardScaler`，那么训练集就隐含了未来测试集的均值和方差信息。

**解决方案**：将预处理步骤封装在 `sklearn.pipeline.Pipeline` 中。

*   **封装**: Pipeline 将 Scaler 和 Model 视为一个整体。
*   **隔离**: 在 Walk-forward 训练时，Pipeline 只会在当前的训练窗口数据上调用 `fit`（计算均值/方差），然后应用到验证集上。
*   **一致性**: 在推理阶段，Pipeline 会自动应用训练好的统计量，无需用户手动维护。

### 6. 当前兼容模式下的模型生命周期

当前版本的 Walk-forward 采用“兼容式生命周期管理”：

*   **训练窗口**: 在当前 bar 完成后，框架使用最近 `train_window` 根数据训练一个新模型副本。
*   **延迟生效**: 新模型不会在当前 bar 立即参与预测，而是从下一根 bar 开始生效。
*   **生效区间**: `test_window` 用于定义该模型计划覆盖的样本外区间。
*   **滚动更新**: `rolling_step` 决定下一次重训触发点；若设置为 `0`，框架会回退使用 `test_window`。
*   **显式状态**: 在 `on_bar` 中建议通过 `self.is_model_ready()` 和 `self.current_validation_window()` 判断模型是否已可用于推理，并读取当前窗口元数据。
*   **模型克隆**: 框架会调用 `QuantModel.clone()` 为每个训练窗口创建待训练模型副本；如果自定义模型不适合 `deepcopy`，请重写该方法。

---

## 完整可运行示例

以下代码展示了如何结合 **Pipeline** 和 **Walk-forward Validation** 构建一个稳健的策略。

```python
import numpy as np
import pandas as pd
from typing import Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from akquant import Strategy, run_backtest, CurrentClose
from akquant.ml import SklearnAdapter

class WalkForwardStrategy(Strategy):
    """
    演示策略：使用逻辑回归预测涨跌 (集成 Pipeline 预处理)
    """

    def __init__(self):
        # 1. 初始化模型 (使用 Pipeline 封装预处理和模型)
        # StandardScaler: 确保使用训练集统计量进行标准化，防止数据泄露
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ])

        self.model = SklearnAdapter(pipeline)

        # 2. 配置 Walk-forward Validation
        # 框架会自动接管数据的切割、模型的重训
        self.model.set_validation(
            method='walk_forward',
            train_window=50,   # 使用过去 50 个 bar 训练
            test_window=20,    # 每个模型默认覆盖 20 个样本外 bar
            rolling_step=10,   # 每 10 个 bar 重训一次
            frequency='1m',    # 数据频率
            incremental=False, # 是否增量训练 (Sklearn 支持 partial_fit)
            verbose=True       # 打印训练日志
        )

        # 确保历史数据长度足够 (训练窗口 + 特征计算所需窗口)
        # 也可以使用 self.warmup_period = 60
        self.set_history_depth(60)
        self._last_logged_window_index = 0
        self._last_logged_pending_activation = 0

    def prepare_features(self, df: pd.DataFrame, mode: str = "training") -> Tuple[Any, Any]:
        """
        [必须实现] 特征工程逻辑
        该函数会被用于训练阶段（生成 X, y）和预测阶段（生成 X）
        """
        X = pd.DataFrame()
        # 特征 1: 1周期收益率
        X['ret1'] = df['close'].pct_change()
        # 特征 2: 2周期收益率
        X['ret2'] = df['close'].pct_change(2)

        if mode == 'inference':
            # 推理模式：只返回最后一行特征，不需要 y
            # 注意：inference 时传入的 df 是最近 history_depth 的数据
            # 最后一行是最新的 bar，我们需要它的特征
            return X.iloc[-1:]

        # 训练模式：构造标签 y (预测下一期的涨跌)
        # shift(-1) 把未来的收益挪到当前行作为 label
        future_ret = df['close'].pct_change().shift(-1)

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

    def on_bar(self, bar):
        # 3. 实时预测与交易
        validation_window = self.current_validation_window()
        if validation_window is None:
            return

        pending_activation = validation_window["pending_activation_bar"]
        if (
            not self.is_model_ready()
            and pending_activation is not None
            and pending_activation != self._last_logged_pending_activation
        ):
            print(
                f"Bar {bar.timestamp}: "
                f"Pending Window={validation_window['pending_window_index']} "
                f"Activation Bar={pending_activation}"
            )
            self._last_logged_pending_activation = int(pending_activation)
            return

        if not self.is_model_ready():
            return

        # 获取最近的数据进行特征提取
        # 注意：需要足够的历史长度来计算特征 (例如 pct_change(2) 需要至少3根bar)
        hist_df = self.get_history_df(10)

        # 如果数据不足，直接返回
        if len(hist_df) < 5:
            return

        # 复用特征计算逻辑！
        # 直接调用 prepare_features 获取当前特征
        X_curr = self.prepare_features(hist_df, mode='inference')

        try:
            # 获取预测信号 (概率)
            # SklearnAdapter 对于二分类返回 Class 1 的概率
            signal = self.model.predict(X_curr)[0]
            window_index = int(validation_window["window_index"])
            active_start_bar = validation_window["active_start_bar"]
            active_end_bar = validation_window["active_end_bar"]

            if window_index != self._last_logged_window_index:
                print(
                    f"Bar {bar.timestamp}: "
                    f"Activated Window={window_index} "
                    f"ActiveRange=[{active_start_bar}, {active_end_bar}]"
                )
                self._last_logged_window_index = window_index

            # 打印信号方便观察
            print(
                f"Bar {bar.timestamp}: "
                f"Window={window_index} "
                f"ActiveRange=[{active_start_bar}, {active_end_bar}] "
                f"Signal={signal:.4f}"
            )

            # 结合风控规则下单
            # 使用 self.get_position(symbol) 获取持仓
            pos = self.get_position(bar.symbol)

            if signal > 0.55 and pos == 0:
                self.buy(bar.symbol, 100)
            elif signal < 0.45 and pos > 0:
                self.sell(bar.symbol, pos)

        except Exception:
            # 示例中仍然保留保护，避免推理异常中断回测
            pass

if __name__ == "__main__":
    # 1. 生成合成数据
    print("生成测试数据...")
    dates = pd.date_range(start="2023-01-01", periods=500, freq="1min")
    # 随机漫步价格
    price = 100 + np.cumsum(np.random.randn(500))
    df = pd.DataFrame({
        "timestamp": dates,
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": 1000,
        "symbol": "TEST"
    })

    # 2. 运行回测
    print("开始运行机器学习回测...")
    result = run_backtest(
        data=df,
        strategy=WalkForwardStrategy,
        symbols="TEST",
        lot_size=1,
        fill_policy=CurrentClose(), # 在当根 bar 结束时撮合
        history_depth=60,
        warmup_period=50,
    )
    print("回测结束。")

    # 3. 打印结果
    print(result)
```

### 运行结果示例

上述代码运行后，你将看到类似如下的输出（包含详细的绩效指标）：

```text
生成测试数据...
开始运行机器学习回测...
2026-02-09 15:58:29 | INFO | Running backtest via run_backtest()...
[########################################] 500/500 (0s)
回测结束。
BacktestResult:
                                            Value
name
start_time              2023-01-01 00:00:00+08:00
end_time                2023-01-01 08:19:00+08:00
duration                          0 days, 8:19:00
total_bars                                    500
closed_trade_count                                  12.0
execution_count                                     24.0
open_position_count                                  0.0
initial_market_value                     100000.0
end_market_value                        100120.50
total_pnl                                  120.50
total_return_pct                         0.120500
annualized_return                        0.127450
max_drawdown                                50.00
max_drawdown_pct                         0.049900
win_rate                                58.333333
loss_rate                               41.666667
```

## 进阶指南

### 1. 特征工程建议

优秀的特征是 ML 成功的关键。除了简单的收益率，你可以考虑：

*   **技术指标**: RSI, MACD, Bollinger Bands (推荐使用 `talib` 或 `pandas_ta`)。
*   **波动率特征**: 历史波动率, ATR。
*   **市场微观结构**: 买卖压力, 量价关系。
*   **时间特征**: 小时, 星期几 (注意这是类别特征，可能需要 One-hot 编码)。

### 2. 模型持久化 (Save/Load)

训练好的模型可以保存下来，用于实盘或后续分析。

```python
# 保存
strategy.model.save("my_model.pkl")

# 加载 (在 __init__ 中)
self.model.load("my_model.pkl")
```

### 3. 深度学习支持 (PyTorch)

使用 `PyTorchAdapter` 可以轻松集成深度学习模型。你需要定义一个标准的 `nn.Module`。

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
    frequency: str = '1d',
    incremental: bool = False,
    verbose: bool = False
)
```

*   `method`: 目前仅支持 `'walk_forward'`。
*   `train_window`: 训练窗口长度。支持 `'1y'` (1年), `'6m'` (6个月), `'50d'` (50天) 或整数 (Bar数量)。
*   `test_window`: 模型计划生效的样本外窗口长度。在兼容模式下，新模型会从下一根 bar 开始生效，并默认覆盖该窗口长度。
*   `rolling_step`: 滚动步长，即每隔多久重训一次模型。若为 `0`，框架会回退使用 `test_window`。
*   `frequency`: 数据的频率，用于将时间字符串正确转换为 Bar 数量 (例如 '1d' 下 1y=252 bars)。
*   `incremental`: 是否使用增量学习（在上次活动模型基础上继续训练）还是从头重训。默认为 `False`。
*   `verbose`: 是否打印训练日志，默认为 `False`。

### `model.clone`

为新的训练窗口创建模型副本。

```python
def clone(self) -> QuantModel
```

*   默认实现使用 `copy.deepcopy`。
*   如果你的模型包含 GPU 句柄、线程锁、外部连接或不可拷贝状态，建议显式重写该方法。
*   框架会在当前 bar 训练待生效模型，并在下一根 bar 激活它，因此 `clone()` 是隔离窗口生命周期的重要组成部分。

### `strategy.prepare_features`

用户必须实现的回调函数，用于特征工程。

```python
def prepare_features(self, df: pd.DataFrame, mode: str = "training") -> Tuple[Any, Any]
```

*   **输入**:
    *   `df`: 历史数据 DataFrame。
    *   `mode`: `"training"` (训练模式) 或 `"inference"` (推理模式)。
*   **输出**:
    *   `mode="training"`: 返回 `(X, y)`。
    *   `mode="inference"`: 返回 `X` (通常是最后一行)。
*   **注意**: 这是一个纯函数，不应依赖外部状态。

### `strategy.is_model_ready`

判断当前是否已有可用于推理的活动模型。

```python
def is_model_ready(self) -> bool
```

*   返回 `True` 表示当前 bar 可以安全调用 `self.model.predict(...)`。
*   在首个训练窗口完成前，通常会返回 `False`。

### `strategy.current_validation_window`

返回当前 Walk-forward 生命周期状态。

```python
def current_validation_window(self) -> dict[str, Any] | None
```

返回值可能包含：

*   `window_index`: 当前活动窗口编号
*   `active_start_bar` / `active_end_bar`: 当前活动模型的计划生效区间
*   `pending_activation_bar`: 待生效模型将在第几根 bar 激活
*   `pending_window_index`: 待生效窗口编号
*   `next_train_bar`: 下一次计划重训的 bar 编号
