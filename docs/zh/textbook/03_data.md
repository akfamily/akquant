# 第 3 章：金融数据获取与处理

> ⏱️ 预计阅读 ~20 分钟 ｜ 🎯 难度 ★★★☆☆（进阶）

## 学习目标

- 理解 OHLCV、复权、频率转换与时间序列字段的核心语义。
- 掌握量化数据清洗、标准化、存储与最小 ETL 流程。
- 建立离线研究与实时数据流之间的统一数据口径意识。

## 前置知识

- 熟悉第 2 章中的 Pandas 基础操作。
- 了解 A 股日线、分钟线与交易日历的基本概念。

## 本章实践入口

- 主示例：[examples/textbook/ch03_data.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch03_data.py)
- 进阶示例：[examples/37_feed_replay_alignment_demo.py](https://github.com/akfamily/akquant/blob/main/examples/37_feed_replay_alignment_demo.py)
- 对应指南：[数据指南](../guide/data.md)

## 快速运行与验收

```bash
python examples/textbook/ch03_data.py
```

验收要点：

1. 脚本可成功拉取并处理一段历史行情数据。
2. 输出中能看到数据行数、时间范围或字段结构等基本信息。
3. 数据可被后续回测脚本直接复用。

## 3.1 AKShare：量化投资的开源数据基石

在量化投资中，数据质量决定了策略的上限 (Garbage In, Garbage Out)。对于中国市场，[AKShare](https://akshare.akfamily.xyz/) 是目前最流行的开源金融数据接口库。它提供了从股票、期货、期权、基金到宏观经济的全维度数据。

### 3.1.1 安装与验证

在第 1 章中我们已经安装了 `akshare`。可以通过以下命令验证版本：

```python
import akshare as ak
print(ak.__version__)
```

## 3.2 金融时间序列数据 (OHLCV)

量化回测中最基础的数据单元是 **K线 (Candlestick)**，通常包含以下字段，简称为 **OHLCV**：

*   **Open**：开盘价
*   **High**：最高价
*   **Low**：最低价
*   **Close**：收盘价
*   **Volume**：成交量

### 3.2.1 复权 (Adjustment)

股票价格会受到**分红**、**配股**、**拆细**等除权除息行为的影响，导致价格出现断层。为了保证回测的连续性，必须对价格进行**复权**处理。复权方式通常有三种，差别在于以哪个时点为基准重新对齐价格序列。**前复权 (Forward Adjustment)** 以当前价格为基准，向前推算历史价格；由于它保留了当前的真实价格水平，方便计算买入股数，因此**回测推荐使用前复权**。**后复权 (Backward Adjustment)** 则反向操作，以历史上市首日价格为基准向后推算当前价格，更适合计算长周期的收益率。至于**不复权 (No Adjustment)**，它保留的是原始价格，除权日会出现巨大的价格跳空，因此**严禁直接用于策略回测**。

#### 复权因子计算 (Adjustment Factor)

前复权价格的计算公式如下：

$$ P_{adj} = P_{raw} \times \frac{P_{today}}{P_{ex-right}} $$

其中 $P_{adj}$ 是复权后价格，$P_{raw}$ 是原始价格。对于分红（每股分红 $D$），除权价 $P_{ex-right} = P_{close} - D$。这意味着历史价格会相应**调低**，使得收益率曲线平滑连接。

### 3.2.2 数据频率 (Data Frequency)

量化回测通常使用不同频率的数据，它们沿着“聚合粒度由细到粗”的方向排列。最细的是 **Tick 数据**，即逐笔成交数据，包含每一笔成交的时间、价格、量；它的数据量极大，适合高频策略 (HFT)。向上聚合一层便是 **Bar 数据 (OHLCV)**，它把一段时间内的 Tick 聚合为一个数据点（如 1分钟 Bar、日线 Bar），是最常用的格式。再粗一层的 **Daily 数据** 即日线数据，包含开高低收及成交量，适合中低频策略（如趋势跟踪、多因子选股）。

`AKQuant` 核心引擎基于 Bar 数据驱动，支持任意周期的 Bar（1分钟、5分钟、日线等）。

在 `akshare` 中获取前复权数据非常简单：

```python
import akshare as ak

# 获取浦发银行 (600000) 的日线数据，前复权
df = ak.stock_zh_a_hist(symbol="600000", period="daily", start_date="20200101", end_date="20231231", adjust="qfq")
print(df.head())
```

## 3.3 数据治理与 ETL 流程

在金融工程中，数据被视为核心资产。构建高质量的数据库需要严格遵循 **ETL (Extract, Transform, Load)** 流程。

### 3.3.1 数据清洗 (Data Cleaning)

原始数据通常包含噪音、缺失值甚至错误，常见的数据治理问题大体可分为两类：一类是看得见的脏数据，另一类是更隐蔽、会系统性扭曲回测结论的偏差。

在脏数据这一侧，最常见的是 **缺失值 (Missing Data)**，它往往源于停牌、数据源故障或非交易日，处理上可采用前向填充 (Forward Fill)、插值法或直接剔除。其次是 **异常值 (Outliers)**，多由乌龙指或数据录入错误造成，可使用 MAD (绝对中位差) 或 3$\sigma$ 原则识别并修正。

更值得警惕的是两类偏差。一是 **幸存者偏差 (Survivorship Bias)**：如果在回测中只包含当前存在的股票，而忽略了历史上已退市的股票，会导致回测结果虚高（因为退市股票通常表现很差），其对策是必须维护包含所有历史退市股票的“全集数据库”。二是 **前视偏差 (Look-ahead Bias)**：即在 $T$ 时刻做决策时，使用了 $T+1$ 时刻才能获得的数据（如使用当天的收盘价来决定当天的开盘买入），其对策是严格的时间戳对齐，使用 Point-in-Time (PIT) 数据库。

### 3.3.2 数据存储 (Storage)

对于高频或海量数据，CSV 并非最佳选择，应改用更高效的二进制格式，且选型可随数据规模与应用层级递进。日常研究中最实用的是 **Parquet / Feather**，它采用列式存储，读取速度快、压缩率高，且 Pandas 完美支持；面向大规模数值矩阵存储则可选用 **HDF5**；而当应用进入机构级场景时，则会动用 **KDB+ / DolphinDB** 这类专业的时序数据库 (Time Series Database)。

!!! tip "超出内存的数据集：out-of-core 流式回测"
    当数据大到无法一次性放进内存（例如全市场多年分钟线）时，AKQuant 支持**有界内存流式回测**：先用 `akquant.write_canonical_parquet(源, "market.parquet")` 把任意来源规范化为按时间排序的 Parquet，再用 `akquant.DataFeed.from_parquet("market.parquet")` 分块流式喂给 `run_backtest`——回测**峰值内存与数据总量无关**。详见《数据准备与加载指南 · 2.6》。

### 3.3.3 标准化字段定义

为了适配 `AKQuant` 引擎，所有数据必须被映射到以下标准字段：

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `date` | `pd.Timestamp` | 交易日期/时间 |
| `symbol` | `str` | 标的代码 (如 `sh600000`) |
| `open` | `float` | 开盘价 |
| `high` | `float` | 最高价 |
| `low` | `float` | 最低价 |
| `close` | `float` | 收盘价 |
| `volume` | `float` | 成交量 |

### 3.3.4 ETL 脚本示例

下面的代码演示了完整的 ETL 流程：从 AKShare 提取数据，清洗为标准格式，并保存为 Parquet 文件。

创建文件 `examples/textbook/ch03_data.py`：

```python
--8<-- "examples/textbook/ch03_data.py"
```

### 运行结果

```bash
python examples/textbook/ch03_data.py
```

你将在控制台看到数据清洗前后的对比，并在 `data/` 目录下找到生成的 `.parquet` 文件。

## 3.4 数据库设计 (Database Design)

随着数据量的增长，单纯的文件存储（CSV/Parquet）将难以满足查询需求。我们需要引入专业的数据库。

### 3.4.1 关系型数据库 (Relational DB)

这类数据库以 **PostgreSQL、MySQL** 为代表，适用于资产基础信息（如股票代码、上市日期、行业分类）与交易账户信息（如资金流水、订单记录）；其特点是支持复杂关联查询 (JOIN)，且事务一致性 (ACID) 强。

### 3.4.2 时序数据库 (Time-Series DB)

这类数据库以 **ClickHouse、InfluxDB、DolphinDB** 为代表，适用于行情数据 (Tick/Bar) 与高频因子数据。它之所以契合时序场景，源于三方面特点：**写入快**，每秒可写入百万级数据点；**压缩率高**，得益于列式存储与针对时间序列优化的压缩算法（如 Delta Encoding）；以及**聚合快**，计算“某股票过去一年的平均成交量”仅需毫秒级。

**ClickHouse 建表示例**：
```sql
CREATE TABLE kline_1m (
    date Date,
    datetime DateTime,
    symbol String,
    open Float32,
    high Float32,
    low Float32,
    close Float32,
    volume Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (symbol, datetime);
```

## 3.5 特征存储 (Feature Store)

在机器学习项目中，特征工程往往是最耗时的。为了避免重复计算，我们需要构建**特征存储 (Feature Store)**，它通常由训练侧与推理侧两套存储构成。**离线存储 (Offline Store)** 存储历史特征（如过去 10 年的 5日均线），用于模型训练，通常基于数仓 (Hive) 或对象存储 (S3)；与之对应的 **在线存储 (Online Store)** 则存储最新特征（如当天的 5日均线），用于实盘预测，通常基于 Redis 以满足低延迟读取。无论走哪条路径，都必须保证**一致性**，即训练和推理使用完全相同的特征计算逻辑。

## 3.6 实时数据流 (Real-time Stream)

在实盘交易中，我们需要处理实时推送的数据流，这通常涉及传输与缓冲两个环节。在传输环节，**WebSocket** 建立持久连接、由服务端主动推送数据，比轮询 (Polling) HTTP 接口效率高得多。在缓冲环节，**消息队列 (Kafka/RabbitMQ)** 在数据源和策略引擎之间引入缓冲层，防止行情爆发时（如开盘瞬间）数据积压导致系统崩溃。

`AKQuant` 的实盘网关模块内置了 WebSocket 客户端，并自动处理断线重连和心跳保活。

---

## 本章小结

### 必须掌握

- 数据质量决定策略上限，清洗、对齐与口径统一是回测前置条件。
- OHLCV、复权、停牌处理是中国市场数据工程中的高频问题。

### 理解即可

- 从离线存储到实时流式处理，本质上都在解决统一字段与时间语义问题。

### 实践提醒

- 每次入库前先检查时区、重复时间戳、缺失值与复权方式。

## 主线推进

回到那条贯穿全书的最小多均线 / 趋势策略：第 1 章里它还只是用一段直接拉取的行情跑通了第一个回测，数据口径是临时的。本章把它向前推进了关键一步——为这条主线建立了**可复现的数据地基**。从现在起，主线策略所消费的 K 线不再是随手获取的原始数据，而是经过统一流程处理的标准件：用 AKShare 以前复权 (`adjust="qfq"`) 拉取，按本章 3.3.3 的标准字段 (`date`/`symbol`/`open`/`high`/`low`/`close`/`volume`) 映射，做过缺失值与时间戳对齐的最小清洗，并落盘为 Parquet 便于反复读取。这样一来，后续第 4、5 章把策略改造成事件驱动版本时，引擎拿到的就是干净、对齐、口径一致的数据，双均线信号的每一次金叉死叉都可被精确复现，而不必再担心复权跳空或前视偏差悄悄污染结论。

## 延伸阅读

**经典著作**

- Marcos López de Prado. *Advances in Financial Machine Learning*, Wiley, 2018 —— 系统讨论金融数据结构、样本去重与回测中的前视/幸存者偏差，可对照本章 3.3.1（数据清洗与偏差）。
- Yves Hilpisch. *Python for Finance: Mastering Data-Driven Finance*（第 2 版），O'Reilly, 2018 —— 用 Pandas 处理金融时间序列、复权与重采样的实操参考，对照本章 3.2（OHLCV 与频率）。
- Wes McKinney. *Python for Data Analysis*（第 3 版），O'Reilly, 2022 —— Pandas 作者所著，讲透时间序列、缺失值与列式存储 (Parquet) 读写，对照本章 3.3（清洗与存储）。
- Martin Kleppmann. *Designing Data-Intensive Applications*, O'Reilly, 2017 —— 关系型与时序存储、批处理与流处理的权威综述，对照本章 3.4 与 3.6（数据库设计与实时流）。

**官方文档与工具**

- [AKShare 官方文档](https://akshare.akfamily.xyz/) —— 本章 `stock_zh_a_hist` 等行情接口与复权参数 `adjust` 的权威说明，对照 3.1 与 3.2。
- [Apache Parquet 官方文档](https://parquet.apache.org/docs/) —— 列式存储格式说明，对照本章 3.3.2（数据存储）。
- [ClickHouse 官方文档](https://clickhouse.com/docs) —— 时序数据库建表与 MergeTree 引擎说明，对照本章 3.4.2。
- [pandas 官方文档：时间序列与重采样](https://pandas.pydata.org/docs/user_guide/timeseries.html) —— 频率转换与时间戳对齐参考，对照本章 3.2.2。

**本书相关**

- [数据指南](../guide/data.md) —— 本章主示例 `examples/textbook/ch03_data.py` 的配套说明，承接 3.3.4 的 ETL 流程。
- [第 1 章：量化投资概述与环境搭建](01_foundations.md) —— 本章数据口径中的 UTC / 本地时区心智模型在第 1 章已建立，对照 3.3.1 的时间戳对齐与本章小结。

## 课后练习

### 基础题

1. 下载同一标的的日线与分钟线，比较字段与时间粒度差异。

### 应用题

1. 对一段数据分别做前复权与不复权，观察价格序列变化。

### 综合题

1. 编写一个最小校验脚本，自动检查缺失值与重复时间戳。

??? note "参考答案要点（先独立思考再展开）"

    **基础题**：日线与分钟线字段一致（OHLCV），但分钟线时间粒度更细、行数大得多，且需处理交易时段与午休缺口。

    **应用题**：不复权在除权日会出现价格跳空断层；前复权平滑连接历史，适合回测计算收益率与买入股数。

    **综合题**：用 `df.isna().sum()` 查缺失、`df["date"].duplicated().any()` 查重复时间戳，再结合交易日历对齐校验。

## 常见错误与排查

1. 时间列错位：确认时区和交易日历是否一致。
2. 价格跳变异常：检查是否遗漏复权处理。
3. 回测读不到数据：核对本地缓存路径与字段命名是否与引擎约定一致。
