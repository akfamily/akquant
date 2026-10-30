# 数据格式

在 MCP 里用 `prepare_data` 准备数据，拿到 `dataset_id` 后交给 `run_backtest`；数据会被规范成 `date`（带时区）、`open`、`high`、`low`、`close`、`volume`、`symbol` 七列。

## 三种来源

本地文件（CSV 或 Parquet，路径必须在项目根目录内）：

```json
{"source": "file", "path": "data/600000.csv", "symbol": "600000", "timezone": "Asia/Shanghai"}
```

AKShare（一期只支持 A 股个股日线，代码为 6 位）：

```json
{"source": "akshare", "symbol": "600000", "start": "2023-01-01", "end": "2024-12-31", "adjust": "qfq"}
```

合成数据（几何随机游走；只用来验证策略能跑通，不要看收益）：

```json
{"source": "synthetic", "symbols": ["SYN001", "SYN002"], "freq": "1d", "seed": 0}
```

## 文件里的列

- 时间列按 `date`、`datetime`、`time`、`timestamp`、`日期`、`时间` 的顺序查找；8 位整数（如 `20240102`）按 `YYYYMMDD` 解析。
- 必须有 `open`、`high`、`low`、`close` 四列；没有 `volume` 时补 0；没有 `symbol` 时，取参数 `symbol`，再没有就用文件名。
- 时间不带时区时，按 `timezone`（默认 `Asia/Shanghai`）本地化，并在 `issues` 里注明。
- 时间或 OHLC 有缺失的行会被丢弃；“标的＋时间”重复的行会在 `issues` 里报出，但不会去重。
- 多个标的用长表：每行一个“标的＋时间”。

部分 AKShare 接口的成交量单位是“手”，不是“股”。如果要按成交量做限制或计算，先自己换算。

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
