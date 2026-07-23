# 量化交易新手指南

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

> 想要深入了解所有回测指标的含义（如索提诺比率、胜率等），请参考 **[回测结果详解](analysis.md)**。

### 第四幕：刻舟求剑的陷阱 (过拟合)
**故事**：
阿K 学会了编程，他试图找到一条完美的曲线。他不断修改参数，终于凑出了一个在过去一年收益率 500% 的策略！
他兴奋地开启实盘，结果一周就亏了 10%。
原来，他的策略代码里隐含着：“如果在2023年1月5日，就买入”。这就像背下了去年考试的答案，但今年的题目变了。

**量化视点**：
这叫**过拟合 (Overfitting)**。策略只是记住了历史数据的噪声，而没有掌握真正的市场规律。优秀的量化策略应当是逻辑简单、普适性强的，而不是对历史数据的生搬硬套。

### 第五幕：真实的谎言 (撮合机制)
**故事**：
阿K 的策略里写着：“如果股价跌到 10.00 元就买入”。
某天，股价最低跌到了 9.98 元，收盘 10.50 元。阿K 以为自己肯定在 10.00 元买到了，赚了 5%。
但实盘成交单显示，他是在 9.98 元才成交的，或者根本没成交。为什么？

**量化视点**：
这就是**撮合逻辑 (Matching Logic)** 的重要性。
在回测中，我们需要尽可能模拟真实的交易所行为，而不仅仅是“见价成交”。

- **穿透检查 (Penetration Check)**：
    - 如果你挂买单 10.00 元，只有当市场价格**低于或等于** 10.00 元时（即 Low <= 10.00），你的订单才可能成交。
    - 如果当天最低价是 10.01 元，即使只差一分钱，你的限价单也不会成交。
- **价格优化 (Price Improvement)**：
    - 如果你挂买单 10.00 元，但市场开盘直接低开在 9.50 元。
    - 真实的交易所会给你“更优”的价格——即以 9.50 元成交，而不是 10.00 元。这叫**价格改善**。
- **Bar 内撮合 (Intra-bar Matching)**：
    - 如果你挂的是止损单（Stop Order），触发价是 10.00 元。

### 第六幕：隐形的陷阱 (数据对齐与公司行为)
**故事**：
阿K 发现他的策略在某些股票上表现异常。

- 有时候股票突然停牌了几天，策略以为数据丢失，直接报错停止运行。
- 有时候股票价格突然从 100 元变成了 50 元，策略以为大跌，疯狂卖出。结果发现是“10送10”的拆股，实际上资产并没有缩水。

**量化视点**：
这就是**数据清洗 (Data Cleaning)** 和 **公司行为 (Corporate Actions)** 的重要性。

- **数据对齐 (Data Alignment)**：
    - 现实中，股票可能因为各种原因停牌（Suspension）。
    - AKQuant 采用 **Late Fill** 策略：如果某天数据缺失，我们会用**前一天的收盘价**填充，但成交量为 0。这样策略就能感知到“今天没有交易”，而不是“数据错了”。
- **公司行为 (Corporate Actions)**：
    - **拆股 (Split)**：如 1 股变 2 股，价格减半。如果不处理，回测会显示巨大的亏损。AKQuant 支持自动处理拆股，调整持仓数量和成本。
    - **分红 (Dividend)**：公司发钱了。如果不处理，回测会漏掉这部分收益。AKQuant 支持自动处理分红，增加现金余额。

---

## 2. 你的武器库：AKQuant 核心模块

现在，阿K 已经从一名小白进化成了拥有系统思维的量化交易员。他需要一套强大的工具来实现他的想法。
这就是 **AKQuant** 登场的时候。

AKQuant 就像是一个为你量身定制的**数字化交易所 + 交易员团队**。

### 核心模块一览

1. **DataFeed (数据投喂员)**
   - **职责**：负责把历史数据（CSV、数据库）或者实时行情（CTP、WebSocket）源源不断地喂给策略。
   - **比喻**：就像是你雇了一个专职盯盘员，每秒钟向你汇报：“现在价格是 10.50，成交量 100手”。
   - **支持**：支持股票、期货、**加密货币 (Crypto)**、**外汇 (Forex)** 等多种资产。

2. **Strategy (策略大脑)**
   - **职责**：这是你的核心逻辑。你告诉它：“如果 MA5 上穿 MA20，就买入”。
   - **比喻**：这是你的交易员替身，严格执行你的指令，不知疲倦。

3. **Engine (撮合引擎/交易所)**
   - **职责**：模拟真实的交易所环境。它接收你的订单，根据当前行情判断能否成交，以什么价格成交。
   - **比喻**：这是你私人的“纳斯达克”或“上交所”。它极其逼真，考虑了滑点、手续费、**停牌**、**分红拆股**等细节。

4. **Portfolio (资产账户)**
   - **职责**：记录你的钱和货。现在的现金是多少？持仓市值是多少？盈亏比例是多少？
   - **比喻**：这是你的财务总监，每一笔账都算得清清楚楚。

5. **RiskManager (风控官)**
   - **职责**：在你下单前进行检查。比如：“你的仓位太重了，拒绝开仓！”或者“亏损超过 10% 了，强制平仓！”
   - **比喻**：这是你的保命符，防止你因为一时冲动而破产。
   - **支持**：
     - **单标的持仓上限**：防止单只股票仓位过重。
     - **行业集中度限制**：防止过度押注某一行业。
     - **总杠杆率熔断**：防止总敞口过大，爆仓风险。

6. **Analyzer (分析师)**
   - **职责**：回测结束后，生成详细的图表和报告。告诉你策略表现如何，哪里需要改进。
   - **比喻**：这是你的绩效评估专家，用数据说话。

### 2.4 支持的资产类型

AKQuant 框架设计之初就考虑了多资产支持，目前涵盖以下主流交易品种：

| 资产类型 | 代码 (AssetType) | 特点 | 交易规则 |
| :--- | :--- | :--- | :--- |
| **股票** | `Stock` | A股/美股/港股 | T+1 (A股) / T+0 (美股/港股)，有印花税 |
| **基金** | `Fund` | ETF/LOF | 类似股票，费率较低，免印花税 |
| **期货** | `Futures` | 商品/股指期货 | T+0，双向交易，保证金制度 |
| **期权** | `Option` | 股票/股指期权 | 高杠杆，非线性收益 |
| **加密货币** | `Crypto` | Bitcoin/Ethereum | **7x24小时**，T+0，支持小数位 (fractional) 交易 |
| **外汇** | `Forex` | EURUSD/JPY | **24小时**，T+0，高杠杆，点差 (Spread) 交易 |

### 2.5 环境准备与安装

确保你的电脑上安装了 Python 3.10 或以上版本。

```bash
pip install akquant
```

验证安装：
```python
import akquant
print(f"AKQuant version: {akquant.__version__}")
```

### 2.6 理解 AKQuant 的时间与时区

很多新手第一次接触 AKQuant 时，会对“为什么日志里看到的是北京时间，而 `timestamp_iso` 却是 `Z` 结尾的 UTC 字符串”感到困惑。这个现象不是 bug，而是框架刻意设计出来的两层时间语义：

1. **内部权威时间统一为 UTC**
   - 引擎内部推进时间、跨标的排序、Python/Rust 之间传递事件时，统一使用 **UTC 纳秒时间戳**。
   - 这能避免多市场、多时区场景下出现“看起来一样，实际不是同一时刻”的歧义。

2. **结构化字段统一为 UTC ISO 8601**
   - `event_time_iso`
   - `bar.timestamp_iso`
   - `tick.timestamp_iso`
   - `trade.timestamp_iso`
   - `order.created_at_iso` / `order.updated_at_iso`
   - 这些字段都表示同一个含义：**UTC 时间的标准字符串表示**。

3. **显示层才做本地化转换**
   - 你在策略里看到的 `self.now`、`self.to_local_time(...)`、`self.format_time(...)`，以及很多面向阅读的日志前缀，才会按策略配置的 `timezone` 转成本地时间。
   - 换句话说，**结构化数据负责“准确”**，**显示层负责“好读”**。
   - 同一个 `timezone` 也会影响绩效统计里的“按日”口径，例如 `daily_returns`、波动率、Sharpe / Sortino / VaR / CVaR 等，都会按该时区的**本地自然日**聚合，而不是按 UTC 自然日硬切。

例如，A 股的北京时间 `2023-01-01 09:31:00+08:00`，对应的 UTC 时间其实是 `2023-01-01T01:31:00Z`。因此：

- `bar.timestamp` 可能是一个 UTC 纳秒整数
- `bar.timestamp_iso` 会是 `2023-01-01T01:31:00Z`
- `self.format_time(bar.timestamp)` 才可能显示为 `2023-01-01 09:31:00`

你可以把它理解成一句原则：

> **AKQuant 用 UTC 保存事实，用本地时区展示给人看。**

这样设计的好处是：

- 回测、实盘、日志、导出 JSON 在跨语言和跨市场下语义一致
- 不会再让一个字符串字段同时承担“排序、存储、展示”三种职责
- 当你需要面向人类阅读时，仍然可以在策略层自由选择北京时间、美东时间或其他时区
- 当你查看日收益和风险指标时，它们也会与策略配置的本地时区保持一致，不会出现“显示是本地时间、统计却按 UTC 日切”的错位

如果你准备处理多市场数据，强烈建议继续阅读 [时区处理指南](../advanced/timezone.md)，建立“内部 UTC、显示层转换”的统一心智模型。

---

## 3. 快速上手：你的第一个策略

我们将实现一个经典的**双均线策略 (Dual Moving Average)**。

- **买入信号**：短期均线（如 5日线）上穿长期均线（如 20日线） -> 金叉。
- **卖出信号**：短期均线下穿长期均线 -> 死叉。

### 3.1 完整代码示例
创建一个名为 `first_strategy.py` 的文件：

```python
import numpy as np
import pandas as pd
from akquant import Strategy, run_backtest, Bar


class DualMovingAverageStrategy(Strategy):
    def __init__(self):
        # 定义策略参数：短期窗口5，长期窗口20
        self.short_window = 5
        self.long_window = 20
        self.warmup_period = self.long_window + 1

    def on_bar(self, bar: Bar):
        # 获取历史收盘价数据
        # history_data 返回的是一个 DataFrame
        hist = self.get_history(count=self.long_window + 1, field="close")

        # 如果数据不足，无法计算均线，直接返回
        if len(hist) < self.long_window + 1:
            return

        # 计算短期和长期均线
        closes = hist
        ma_short = np.mean(closes[-self.short_window:])
        ma_long = np.mean(closes[-self.long_window:])

        # 获取上一时刻的均线值（用于判断交叉）
        prev_ma_short = np.mean(closes[-self.short_window - 1: -1])
        prev_ma_long = np.mean(closes[-self.long_window - 1: -1])

        # 获取当前持仓
        position = self.get_position(bar.symbol)

        # 交易逻辑
        # 1. 金叉：短期均线上穿长期均线，且当前无持仓 -> 买入
        if prev_ma_short <= prev_ma_long and ma_short > ma_long:
            if position == 0:
                self.buy(bar.symbol, 100)  # 买入100股
                print(f"[{bar.timestamp_iso}] 金叉买入 {bar.symbol} @ {bar.close:.2f}")

        # 2. 死叉：短期均线下穿长期均线，且持有仓位 -> 卖出
        elif prev_ma_short >= prev_ma_long and ma_short < ma_long:
            if position > 0:
                self.sell(bar.symbol, 100)  # 卖出100股
                print(f"[{bar.timestamp_iso}] 死叉卖出 {bar.symbol} @ {bar.close:.2f}")

# ------------------------------
# 准备测试数据并运行
# ------------------------------
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(1024)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    price = 100 + np.cumsum(np.random.randn(len(dates)))  # 随机游走价格

    df = pd.DataFrame({
        "date": dates,
        "open": price, "high": price + 1, "low": price - 1, "close": price,
        "volume": 10000,
        "symbol": "DEMO"
    })

    # 运行回测
    print("开始回测...")
    result = run_backtest(
        strategy=DualMovingAverageStrategy,
        data=df,
        initial_cash=10000.0,  # 初始资金 1万
        warmup_period=21
    )

    # 打印简要结果
    print("\n回测结束！")
    print(f"绩效指标: {result.metrics_df}")
```

### 3.2 进阶策略：均值回归 (Mean Reversion)

除了趋势跟踪（如双均线），另一种常见的策略是**均值回归**。
它的核心思想是：价格总是围绕价值上下波动，涨多了会跌，跌多了会涨。

**策略逻辑 (布林带策略 Bollinger Bands)**：

- **中轨**：20日均线
- **上轨**：中轨 + 2倍标准差
- **下轨**：中轨 - 2倍标准差
- **买入**：价格跌破下轨（超跌），且开始反弹。
- **卖出**：价格突破上轨（超买），或回归中轨。

```python
class BollingerStrategy(Strategy):
    def __init__(self):
        self.window = 20
        self.std_dev = 2

    def on_bar(self, bar):
        # 获取足够的历史数据
        hist = self.get_history(count=self.window + 1)
        if len(hist) < self.window:
            return

        closes = hist
        # 计算布林带
        ma = np.mean(closes[-self.window:])
        std = np.std(closes[-self.window:])
        upper = ma + self.std_dev * std
        lower = ma - self.std_dev * std

        current_price = bar.close
        position = self.get_position(bar.symbol)

        # 策略逻辑：
        # 1. 价格跌破下轨，买入（视为超跌）
        if current_price < lower and position == 0:
            self.buy(bar.symbol, 100)
            print(f"[{bar.timestamp_iso}] 超跌买入 {bar.symbol} @ {current_price:.2f}")

        # 2. 价格回归中轨或突破上轨，卖出（止盈）
        elif (current_price > ma or current_price > upper) and position > 0:
            self.sell(bar.symbol, 100)
            print(f"[{bar.timestamp_iso}] 回归卖出 {bar.symbol} @ {current_price:.2f}")
```

### 3.3 实战：获取真实数据

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
    strategy=DualMovingAverageStrategy,
    data=df,
    initial_cash=10000.0
)
```

### 3.4 策略可视化 (Visualization)

回测结束后，光看数字是不够的。我们需要直观地看到各项指标。
AKQuant 提供了内置的绘图功能。

```python
# 在 run_backtest 后添加

# 绘制资金曲线
result.viz.report(show=True)
```

### 3.5 参数调优 (Parameter Optimization)

你可能会问：“为什么均线是 5日和 20日？改成 10日和 30日会不会更好？”
这就是**参数调优**要做的事。我们可以遍历不同的参数组合，找到历史表现最好的一组。

```python
from akquant import run_grid_search

# 1. 调整策略类以接收参数
class OptimizedDualMA(DualMovingAverageStrategy):
    def __init__(self, short_window=5, long_window=20):
        self.short_window = short_window
        self.long_window = long_window

# 2. 定义参数网格
param_grid = {
    "short_window": range(3, 10),  # 3 到 9
    "long_window": range(15, 30)   # 15 到 29
}

# 3. 运行网格搜索
# AKQuant 会自动组合所有参数，并利用多核 CPU 并行计算
results = run_grid_search(
    strategy=OptimizedDualMA,
    param_grid=param_grid,
    data=df,
    initial_cash=10000.0,
    sort_by="sharpe_ratio",  # 按夏普比率排序
    max_workers=4            # 并行进程数
)

# 4. 获取最佳结果
# results 是一个 DataFrame，包含参数和回测指标
print("优化结果前5名：")
print(results.head())

best_params = results.iloc[0]
print(f"\n最佳参数: short={best_params['short_window']}, long={best_params['long_window']}")
print(f"最佳夏普: {best_params['sharpe_ratio']:.4f}")
```
> **注意**：参数调优容易导致**过拟合**，请务必在样本外数据（Out-of-Sample）上进行验证。

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
    if len(self.get_history(count=20)) < 20:
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

### 5.4 忽视复权 (Ignoring Adjustment)

- **现象**：股票发生拆股或分红时，价格突然跳水，导致策略误判为暴跌卖出。
- **原因**：使用的是**不复权**数据。例如 10元 拆成 5元，价格腰斩但价值没变。
- **解决**：**必须使用前复权 (qfq) 数据**进行回测。
    - 前复权：以当前价格为基准，修正历史价格，保持价格连续性。
    - AKShare 获取数据时务必指定 `adjust="qfq"`。

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
4.  **机器学习**：尝试用机器学习模型（如随机森林、LSTM）来预测价格或波动率（参考 [ML Guide](../advanced/ml.md)）。

### 6.2 推荐资源

- **书籍**：《打开量化投资的黑箱》、《海龟交易法则》、《Python for Finance》。
- **实战**：多看 [Examples](examples.md) 中的代码，尝试修改参数，观察结果变化。

希望这份指南能帮你顺利开启量化之旅！更多进阶功能请参考 [API 文档](../reference/api.md)。
