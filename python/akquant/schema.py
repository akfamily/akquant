"""规范数据 schema 与列名/时区约定的单一定义源.

本模块是数据归一化的唯一常量来源, 收敛此前散落在
``utils``/``feed_adapter``/``data`` 中重复的列名候选表与时区默认值.

对应设计: docs/zh/meta/columnar-rfc.md 第 3、15 节.
"""

from typing import Dict, List, Tuple

import pyarrow as pa

# 默认输入时区: naive(无时区)时间被视为该时区, 再统一转 UTC.
# 此前重复定义于 feed_adapter.DEFAULT_INPUT_TIMEZONE 与 data.py.
DEFAULT_INPUT_TIMEZONE = "Asia/Shanghai"

# 规范字段 -> 输入列名候选(按优先级). 合并自:
#   utils.df_to_arrays.targets 与 utils.load_bar_from_df.required_map.
# 解析时先精确匹配, 再大小写不敏感回退(见 normalize.resolve_columns).
COLUMN_ALIASES: Dict[str, List[str]] = {
    "timestamp": ["日期", "date", "datetime", "time", "timestamp"],
    "open": ["开盘", "open"],
    "high": ["最高", "high"],
    "low": ["最低", "low"],
    "close": ["收盘", "close"],
    "volume": ["成交量", "volume", "vol"],
    "symbol": ["股票代码", "symbol", "code", "ticker"],
}

# 构造 Bar 所必需的字段(不含 symbol; symbol 可由参数提供).
REQUIRED_FIELDS: Tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

# OHLCV 数值列(统一转 float64).
OHLCV_FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")

# 规范 Arrow schema(RFC 第 3 节). A 期仅用 OHLCV 子集;
# Tick/Quote/OrderBook 等多类型留待 E 期. extra 数值列在归一化后作为
# float64 追加列透传, 不在此固定 schema 中枚举.
BAR_SCHEMA: pa.Schema = pa.schema(
    [
        ("ts_event", pa.timestamp("ns", tz="UTC")),
        ("instrument_id", pa.dictionary(pa.int32(), pa.string())),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
    ]
)
