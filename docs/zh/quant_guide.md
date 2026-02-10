# 量化交易新手指南 (Zero to Hero)

欢迎来到量化交易的世界！本指南专为零基础用户设计，旨在帮助你从理论概念到代码实战，系统性地掌握使用 AKQuant 框架进行量化策略开发的全过程。

---

## 0. 为什么选择量化？(量化 vs 主观)

投资的世界主要分为两派：**主观投资 (Discretionary)** 和 **量化投资 (Quantitative)**。

- **主观投资**像是一门**艺术**。交易员依靠个人的经验、直觉和对市场的定性分析（如新闻、宏观政策）来做决策。巴菲特、索罗斯是这方面的宗师。
- **量化投资**则是一门**科学**。它通过数学模型和计算机程序，从海量历史数据中寻找概率优势，并严格执行。西蒙斯（大奖章基金）是量化的代表。

### 核心对比

| 维度 | 主观交易 (Discretionary) | 量化交易 (Quantitative) |
| :--- | :--- | :--- |
| **决策依据** | 经验、直觉、消息、定性分析 | **数据、统计模型、代码逻辑** |
| **执行力** | 易受情绪（贪婪/恐惧）干扰，知行合一难 | **机器自动执行，冷酷无情，100% 纪律** |
| **覆盖范围** | 人的精力有限，只能关注几十只股票 | **计算机可同时监控几千只股票、几百个市场** |
| **可验证性** | 难以验证（"感觉"无法回测） | **高度可验证（通过回测检验历史表现）** |
| **缺点** | 难以复制，状态不稳定，容易犯错 | 存在模型失效风险，对突发事件（黑天鹅）反应可能滞后 |

**选择量化的理由**：
如果你不想每天盯着盘面心惊肉跳，如果你相信数据胜过直觉，如果你希望通过代码构建一个能长期为你工作的"赚钱机器"，那么量化投资就是为你准备的。

---

## 1. 量化交易基础概念：阿K的觉醒之路

为了让你更直观地理解量化交易，我们来讲一个名为“阿K”的交易员的故事。通过他的经历，我们将解锁量化交易的核心概念。

### 第一幕：情绪的奴隶 vs 机器的纪律
**故事**：
阿K 曾经是一个典型的手动交易员。每天盯着屏幕，心情随着红绿柱子起伏。
- 上午10点，看到股价猛涨：“快追！不然踏空了！” —— 结果高位接盘。
- 下午2点，股价跳水：“完了！要跌停了，快跑！” —— 结果低位割肉。
一天下来，阿K 筋疲力尽，账户还亏了钱。他发现自己最大的敌人不是市场，而是**贪婪与恐惧**。

**量化视点**：
量化交易的第一大优势是**纪律性 (Discipline)**。
我们将交易逻辑写成代码（即**策略**），让计算机自动执行。机器没有情绪，它不会因为暴涨而兴奋，也不会因为暴跌而恐慌。它只会冷酷地执行：“如果 A 发生，我就做 B”。

### 第二幕：神秘的“时光机” (回测)
**故事**：
阿K 总结了一个规律：“我发现每次连续跌三天，第四天大概率会反弹。”
他想用真金白银去赌这个规律，但心里没底：“这个规律在2018年熊市管用吗？在2020年牛市又如何？”
他希望能有一台时光机，回到过去，用这个规则模拟交易一遍，看看能赚多少钱。

**量化视点**：
这台时光机就是**回测 (Backtesting)**。
回测是指使用**历史数据**来验证交易策略在过去表现的过程。
- **输入**：你的策略逻辑（如“连跌三天买入”）。
- **数据**：过去几年的 K 线数据（OHLCV - 开高低收量）。
- **输出**：如果按此操作，你的最终收益、最大亏损是多少。

### 第三幕：关于“活下去”的教训 (风控)
**故事**：
阿K 曾经运气爆棚，重仓押注一只股票，一个月翻倍。他觉得自己是股神，于是借钱加杠杆继续满仓。
结果市场突发“黑天鹅”事件，股价腰斩。阿K 不仅利润回吐，本金也亏光了。
量化前辈告诉他：“在这个市场，活得久比赚得快更重要。”

**量化视点**：
这就是**风险控制 (Risk Management)**。在量化报告中，我们不仅看赚了多少（收益率），更看冒了多大风险。
- **最大回撤 (Max Drawdown)**：策略从历史最高点跌下来的最大幅度。如果最大回撤是 -50%，意味着你的资产可能腰斩。
- **夏普比率 (Sharpe Ratio)**：衡量性价比的指标。每承担一份风险，能换来多少超额收益。

### 第四幕：刻舟求剑的陷阱 (过拟合)
**故事**：
阿K 学会了编程，他试图找到一条完美的曲线。他不断修改参数，终于凑出了一个在过去一年收益率 500% 的策略！
他兴奋地开启实盘，结果一周就亏了 10%。
原来，他的策略代码里隐含着：“如果在2023年1月5日，就买入”。这就像背下了去年考试的答案，但今年的题目变了。

**量化视点**：
这叫**过拟合 (Overfitting)**。策略只是记住了历史数据的噪声，而没有掌握真正的市场规律。优秀的量化策略应当是逻辑简单、普适性强的，而不是对历史数据的生搬硬套。

---

## 1.5 核心数据格式 (OHLCV)
回到技术层面，量化交易中最基础的数据单元是 K 线（Bar），通常包含以下字段：
- **Open**: 开盘价
- **High**: 最高价
- **Low**: 最低价
- **Close**: 收盘价
- **Volume**: 成交量

---

## 2. 环境准备与框架安装

### 2.1 安装 AKQuant

确保你的电脑上安装了 Python 3.10 或以上版本。打开终端（Terminal 或 CMD），输入以下命令：

```bash
pip install akquant
```

### 2.2 验证安装

创建一个名为 `check_env.py` 的文件，输入以下代码并运行：

```python
import akquant
print(f"AKQuant version: {akquant.__version__}")
print("安装成功！")
```

---

## 3. 实战演练：开发第一个策略（双均线交叉）

我们将实现一个经典的**双均线策略 (Dual Moving Average)**。
- **买入信号**：短期均线（如 5日线）上穿长期均线（如 20日线） -> 金叉。
- **卖出信号**：短期均线下穿长期均线 -> 死叉。

### 3.1 完整代码示例
创建一个名为 `first_strategy.py` 的文件：

```python
import pandas as pd
import numpy as np
from akquant import Strategy, run_backtest

class DualMovingAverageStrategy(Strategy):
    def __init__(self):
        # 定义策略参数：短期窗口5，长期窗口20
        self.short_window = 5
        self.long_window = 20

    def on_bar(self, bar):
        # 获取历史收盘价数据
        # history_data 返回的是一个 DataFrame
        hist = self.history_data(n=self.long_window + 1)

        # 如果数据不足，无法计算均线，直接返回
        if len(hist) < self.long_window:
            return

        # 计算短期和长期均线
        closes = hist['close'].values
        ma_short = np.mean(closes[-self.short_window:])
        ma_long = np.mean(closes[-self.long_window:])

        # 获取上一时刻的均线值（用于判断交叉）
        prev_ma_short = np.mean(closes[-self.short_window-1 : -1])
        prev_ma_long = np.mean(closes[-self.long_window-1 : -1])

        # 获取当前持仓
        position = self.get_position(bar.symbol)

        # 交易逻辑
        # 1. 金叉：短期均线上穿长期均线，且当前无持仓 -> 买入
        if prev_ma_short <= prev_ma_long and ma_short > ma_long:
            if position == 0:
                self.buy(bar.symbol, 100) # 买入100股
                print(f"[{bar.datetime}] 金叉买入 {bar.symbol} @ {bar.close:.2f}")

        # 2. 死叉：短期均线下穿长期均线，且持有仓位 -> 卖出
        elif prev_ma_short >= prev_ma_long and ma_short < ma_long:
            if position > 0:
                self.sell(bar.symbol, 100) # 卖出100股
                print(f"[{bar.datetime}] 死叉卖出 {bar.symbol} @ {bar.close:.2f}")

# ------------------------------
# 准备测试数据并运行
# ------------------------------
if __name__ == "__main__":
    # 生成模拟数据
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    price = 100 + np.cumsum(np.random.randn(len(dates))) # 随机游走价格

    df = pd.DataFrame({
        "date": dates,
        "open": price, "high": price + 1, "low": price - 1, "close": price,
        "volume": 10000,
        "symbol": "DEMO"
    })

    # 运行回测
    print("开始回测...")
    result = run_backtest(
        strategy_class=DualMovingAverageStrategy,
        data=df,
        initial_capital=10000.0 # 初始资金 1万
    )

    # 打印简要结果
    print("\n回测结束！")
    print(f"最终权益: {result.final_value:.2f}")
    print(f"总收益率: {result.total_return * 100:.2f}%")
```

### 3.2 进阶：如何获取真实数据？
上面的例子使用了模拟数据。在实战中，你需要获取真实的股票数据。
我们推荐使用 **AKShare**，它是目前最流行的开源中文金融数据接口库，数据覆盖面极广。

**示例：使用 AKShare 获取平安银行 (000001) 数据**

```bash
pip install akshare
```

```python
import akshare as ak
import pandas as pd

# 下载 A 股历史数据 (平安银行)
# adjust="qfq" 表示使用前复权数据，这在回测中非常重要
# period="daily" 表示日线数据
df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20230101", end_date="20231231", adjust="qfq")

# 数据清洗：将 AKShare 的中文列名转换为 AKQuant 需要的英文标准格式
df = df.rename(columns={
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume"
})

# 格式转换：确保 date 列是 datetime 类型
df['date'] = pd.to_datetime(df['date'])
df['symbol'] = "000001" # 添加代码列

# 筛选需要的列（可选，AKQuant 会自动忽略多余列）
df = df[["date", "open", "high", "low", "close", "volume", "symbol"]]

# 直接传入 run_backtest
result = run_backtest(
    strategy_class=DualMovingAverageStrategy,
    data=df,
    initial_capital=10000.0
)
```

---

## 4. 读懂回测报告：核心指标解读

运行回测后，AKQuant 会输出一系列指标。以下是核心指标的含义：

| 指标名称 | 英文 | 含义与解读 |
| :--- | :--- | :--- |
| **总收益率** | Total Return | 策略期末相对于期初赚了百分之多少。如果是负数，说明亏损。 |
| **年化收益率** | Annualized Return | 假设策略运行一年能赚多少。方便不同时长的策略进行比较。 |
| **最大回撤** | Max Drawdown | 策略从历史最高点跌下来的最大幅度。**衡量风险的重要指标**。例如 -20% 意味着你可能在最坏情况下亏损20%。 |
| **夏普比率** | Sharpe Ratio | 衡量"性价比"。即每承担一单位风险，能获得多少超额收益。通常 >1 为良好，>2 为优秀。 |
| **胜率** | Win Rate | 盈利交易次数占总交易次数的比例。高胜率不一定代表高收益（可能赚小钱亏大钱）。 |

---

## 5. 常见错误排查与调试技巧

新手在开发过程中常遇到以下问题，请对照检查：

### 5.1 数据不足 (Data Not Enough)
- **现象**：程序报错 `IndexError` 或均线计算结果为 `NaN`。
- **原因**：计算20日均线至少需要20条历史数据。在回测刚开始的前几天，数据累积不足。
- **解决**：在 `on_bar` 开头添加检查：
    ```python
    if len(self.history_data(n=20)) < 20:
        return
    ```

### 5.2 前视偏差 (Look-ahead Bias)
- **现象**：回测收益率高得离谱（例如年化 1000%）。
- **原因**：在计算信号时使用了**未来**的数据。例如在 `on_bar` 处理今天的数据时，却取到了明天的收盘价。
- **解决**：确保只使用当前 `bar` 及之前的数据。AKQuant 的 `history_data` 默认是安全的。

### 5.3 交易未执行
- **现象**：日志显示发出了买单，但持仓没有变化。
- **原因**：可能是资金不足（Cash不够买1手）或未达到最小交易单位。
- **解决**：打印 `self.cash` 查看可用资金；检查 `self.buy` 的数量是否合理。

---

## 6. 下一步学习建议

恭喜你完成了入门教程！想要成为更专业的量化交易员，你可以按照以下路径继续修炼：

### 6.1 技能树点亮顺序

1.  **Python 进阶**：熟练掌握 `pandas` 和 `numpy`。量化交易 80% 的时间在处理数据，这两大库是处理数据的神兵利器。
2.  **经典策略研究**：
    - **海龟交易法则 (Turtle Trading)**：学习趋势跟踪系统的完整构建。
    - **网格交易 (Grid Trading)**：学习震荡市的自动化套利。
    - **多因子模型 (Multi-Factor)**：学习机构主流的选股逻辑。
3.  **风险管理**：深入理解凯利公式 (Kelly Criterion)、波动率控制等资金管理方法。
4.  **机器学习**：尝试用机器学习模型（如随机森林、LSTM）来预测价格或波动率（参考 [ML Guide](../ml_guide.md)）。

### 6.2 推荐资源

- **书籍**：《打开量化投资的黑箱》、《海龟交易法则》、《Python for Finance》。
- **实战**：多看 [Examples](../examples.md) 中的代码，尝试修改参数，观察结果变化。

希望这份指南能帮你顺利开启量化之旅！更多进阶功能请参考 [API 文档](../api.md)。
