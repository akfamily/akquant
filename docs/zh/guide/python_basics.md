# 金融 Python 极简入门 (Python for Finance 101)

很多量化新手看到代码就头疼。别担心，在 AKQuant 中，你不需要成为计算机专家，只需要掌握**最核心的 20% 语法**，就能覆盖 80% 的策略开发需求。

本指南将完全从**金融交易**的视角，带你快速上手 Python。

---

## 1. 变量与数据类型：定义你的资产

在 Python 中，变量就像是一个个贴了标签的盒子，用来存数据。

### 基础变量
*   **整数 (int)**：用于表示手数、交易量。
*   **浮点数 (float)**：用于表示价格、收益率。
*   **字符串 (str)**：用于表示股票代码。

```python
# 定义变量
symbol = "000001"      # 股票代码 (字符串)
price = 10.5           # 当前价格 (浮点数)
volume = 100           # 买入数量 (整数)
is_holding = True      # 是否持仓 (布尔值)

# 打印出来看看
print(f"正在交易 {symbol}，价格：{price}")
```

### 列表 (List)：一串价格序列
想象你在看 K 线图，过去5天的收盘价就是一个列表。

```python
# 最近5天的收盘价
closes = [10.1, 10.2, 10.0, 10.5, 10.8]

# 访问数据
print(closes[0])   # 获取第1天价格 (Python从0开始计数！) -> 10.1
print(closes[-1])  # 获取最后1天价格 (最新价) -> 10.8
print(closes[:3])  # 获取前3天价格 -> [10.1, 10.2, 10.0]

# 计算均价
avg_price = sum(closes) / len(closes)
print(f"5日均价: {avg_price}")
```

### 字典 (Dictionary)：你的持仓组合
字典就像一个账本，左边是名字（Key），右边是数值（Value）。

```python
# 当前持仓：股票代码 -> 持股数
portfolio = {
    "AAPL": 100,
    "TSLA": 50,
    "GOOG": 0
}

# 查询持仓
print(portfolio["AAPL"]) # -> 100

# 更新持仓
portfolio["TSLA"] += 10  # 加仓10股
```

---

## 2. 逻辑控制：交易的决策核心

量化策略的本质就是一堆 `if...else`：如果发生什么，就做什么。

### 条件判断 (If/Else)
这是策略的“扳机”。

```python
ma5 = 10.5  # 短期均线
ma20 = 10.0 # 长期均线
cash = 5000 # 可用资金

# 金叉策略逻辑
if ma5 > ma20:
    print("信号：金叉出现！")
    if cash > 1000:
        print("资金充足，执行买入！")
    else:
        print("资金不足，无法买入。")
else:
    print("无信号，继续观望。")
```

### 循环 (Loop)
回测就是在一个大循环里，一天一天地遍历历史数据。

```python
# 模拟回测：遍历每一天的价格
prices = [10, 11, 9, 12, 13]

for p in prices:
    if p > 12:
        print(f"价格 {p} 突破 12元，触发止盈！")
    else:
        print(f"价格 {p} 正常波动")
```

---

## 3. Pandas 速成：量化神器

在 AKQuant 中，99% 的数据（如 `history_data`）都是以 `DataFrame` 的形式存在的。你可以把它想象成一个**超级 Excel 表格**。

### DataFrame 结构
一个 DataFrame 包含行（Index，通常是时间）和列（Columns，如 Open/Close）。

```python
import pandas as pd

# 模拟一个 OHLCV 表格
data = {
    "close": [10, 11, 12, 11, 13],
    "volume": [100, 150, 200, 120, 300]
}
df = pd.DataFrame(data)

# 1. 获取整列数据 (Series)
print(df["close"])

# 2. 计算技术指标 (向量化运算，极快)
# 计算收盘价的平均值
ma_price = df["close"].mean()

# 3. 获取最新一行数据
last_bar = df.iloc[-1]
print(f"最新收盘价: {last_bar['close']}")
```

---

## 4. 类与对象：读懂策略模板

在写策略时，你会看到 `class MyStrategy(Strategy):`。这是什么意思？

*   **Class (类)**：策略的**图纸**。定义了这个策略长什么样，有什么功能。
*   **Object (对象)**：根据图纸造出来的**实例**。回测运行时，系统会创建一个策略实例。
*   **self**：代表**策略实例自己**。
    *   `self.buy()`: 我要买。
    *   `self.cash`: 我的现金。
    *   `self.position`: 我的持仓。

```python
class MyStrategy:
    def __init__(self, initial_cash):
        # 初始化：策略启动时执行一次
        self.cash = initial_cash
        self.name = "均线策略"

    def on_bar(self, price):
        # 每根K线走完时执行一次
        if price < 10:
            self.buy()

    def buy(self):
        self.cash -= 10
        print(f"{self.name} 执行买入，剩余资金: {self.cash}")

# 使用策略
strategy = MyStrategy(1000) # 创建实例，初始资金1000
strategy.on_bar(9)          # 价格跌到9，触发买入
```

---

## 5. 总结：AKQuant 常用语速查

在 AKQuant 写策略时，你最常用的就是下面这几句：

| 场景 | Python 代码 | 含义 |
| :--- | :--- | :--- |
| **获取数据** | `hist = self.history_data(n=20)` | 拿过去20根K线数据 (DataFrame) |
| **计算均线** | `ma = hist['close'].mean()` | 计算收盘价平均值 |
| **取最新价** | `current_price = bar.close` | 获取当前K线的收盘价 |
| **判断信号** | `if ma_short > ma_long:` | 如果短期均线大于长期均线 |
| **查持仓** | `pos = self.get_position(bar.symbol)` | 我现在持有多少股？ |
| **下单** | `self.buy(bar.symbol, 100)` | 买入 100 股 |

掌握了这些，你就可以开始写出你的第一个量化策略了！去 [Quant Guide](quant_guide.md) 试试吧。
