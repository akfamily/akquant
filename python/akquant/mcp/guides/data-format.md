# 数据格式

在 MCP 里用 `prepare_data` 准备数据，拿到 `dataset_id` 后交给 `run_backtest`；数据会被规范成 `date`（带时区）、`open`、`high`、`low`、`close`、`volume`、`symbol` 七列。

## 两种来源

本地文件（CSV 或 Parquet，路径必须在项目根目录或 AKTools 数据目录内）：

```json
{"source": "file", "path": "data/600000.csv", "symbol": "600000", "timezone": "Asia/Shanghai"}
```

AKQuant 的 MCP 不联网取数。真实行情用 AKTools（AKShare 的 MCP 服务）分两步准备；没有接入 AKTools 时，请用户先把数据存成 CSV 或 Parquet 放进项目目录：

1. 调用 AKTools 的 `save_data`，例如接口 `stock_zh_a_hist_sina`、参数 `{"symbol": "sh600519", "start_date": "20240101", "end_date": "20241231", "adjust": "qfq"}`。它把完整数据存成 CSV，返回文件路径 `path` 和各列的单位说明。
2. 把这个 `path` 原样传给 `prepare_data`，并且必须传 `symbol`（文件名是“接口名加参数哈希”，不能当标的名；不传会报错）：

```json
{"source": "file", "path": "<save_data 返回的 path>", "symbol": "600519"}
```

取数时优先选新浪财经、腾讯的接口：成交量单位是“股”。东方财富行情接口的中文列名也能识别，但成交量单位是“手”（100 股），`prepare_data` 不做换算；`save_data` 返回的列说明里写有单位。

合成数据（几何随机游走；只用来验证策略能跑通，不要看收益）：

```json
{"source": "synthetic", "symbols": ["SYN001", "SYN002"], "freq": "1d", "seed": 0}
```

## 文件里的列

- 时间列按 `date`、`datetime`、`time`、`timestamp`、`日期`、`时间` 的顺序查找；8 位整数（如 `20240102`）按 `YYYYMMDD` 解析。
- 必须有 `open`、`high`、`low`、`close` 四列；没有 `volume` 时补 0；没有 `symbol` 时，取参数 `symbol`，再没有就用文件名。
- 中文列名也认：`开盘`、`最高`、`最低`、`收盘`、`成交量`、`股票代码`；标的列还认 `code`、`ticker`，成交量还认 `vol`。同一字段的英文列和别名列都在时用英文列。
- 时间不带时区时，按 `timezone`（默认 `Asia/Shanghai`）本地化，并在 `issues` 里注明。
- 时间或 OHLC 有缺失的行会被丢弃；“标的＋时间”重复的行会在 `issues` 里报出，但不会去重。
- 多个标的用长表：每行一个“标的＋时间”。

部分 AKShare 接口（如东方财富的行情接口）的成交量单位是“手”，不是“股”。如果要按成交量做限制或计算，先自己换算后再存成文件。

## 自己构造 DataFrame

直接写脚本调用 `run_backtest(data=df)` 时，用同样的列，时间要显式 `tz_localize`：

```python
import pandas as pd

raw = pd.DataFrame(
    {
        "date": ["2024-01-02", "2024-01-03"],
        "open": [10.0, 10.2],
        "high": [10.3, 10.4],
        "low": [9.9, 10.1],
        "close": [10.2, 10.3],
        "volume": [120000, 98000],
        "symbol": ["600000", "600000"],
    }
)
raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize("Asia/Shanghai")
```
