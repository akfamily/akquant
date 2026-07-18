"""归一化漏斗单测: 覆盖行为并集清单与多源输入(RFC A 期 §15.4/§15.6)."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from akquant import Bar
from akquant.normalize import (
    arrow_to_legacy_arrays,
    coerce_to_pandas,
    dataframe_to_arrays,
    dataframe_to_bars,
    normalize,
    to_frame,
)


def _sample_en() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2023-01-03", "2023-01-04"],
            "open": [10.0, 11.0],
            "high": [12.0, 12.5],
            "low": [9.5, 10.5],
            "close": [11.0, 12.0],
            "volume": [100, 200],
        }
    )


def _sample_cn() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": ["2023-01-03", "2023-01-04"],
            "开盘": [10.0, 11.0],
            "最高": [12.0, 12.5],
            "最低": [9.5, 10.5],
            "收盘": [11.0, 12.0],
            "成交量": [100, 200],
            "股票代码": ["600000", "600000"],
        }
    )


def test_naive_timestamp_localized_to_shanghai_utc() -> None:
    """Naive 时间按 Asia/Shanghai 本地化后转 UTC 纳秒."""
    ts, *_ = dataframe_to_arrays(_sample_en(), symbol="X")
    expected = np.array(
        [
            pd.Timestamp("2023-01-03", tz="Asia/Shanghai").value,
            pd.Timestamp("2023-01-04", tz="Asia/Shanghai").value,
        ],
        dtype="int64",
    )
    np.testing.assert_array_equal(ts, expected)


def test_chinese_columns_resolved() -> None:
    """中文列名(AKShare 风格)可解析, 且 股票代码 触发多标的."""
    ts, o, h, low, c, v, symbol, symbols, extra = dataframe_to_arrays(_sample_cn())
    assert symbol is None
    assert symbols == ["600000", "600000"]
    np.testing.assert_array_equal(o, np.array([10.0, 11.0]))
    np.testing.assert_array_equal(v, np.array([100.0, 200.0]))


def test_datetime_index_as_time_source() -> None:
    """无时间列但有 DatetimeIndex 时, 走 __index__ 路径."""
    df = _sample_en().drop(columns=["date"])
    df.index = pd.DatetimeIndex(["2023-01-03", "2023-01-04"])
    ts, *_ = dataframe_to_arrays(df, symbol="X")
    expected = pd.Timestamp("2023-01-03", tz="Asia/Shanghai").value
    assert ts[0] == expected


def test_multi_symbol_from_symbol_column() -> None:
    """Symbol 列存在时产出多标的列表."""
    df = _sample_en()
    df["symbol"] = ["AAA", "BBB"]
    *_, symbol, symbols, _extra = dataframe_to_arrays(df)
    assert symbol is None
    assert symbols == ["AAA", "BBB"]


def test_extra_numeric_columns_passthrough() -> None:
    """非 OHLCV 的数值列作为 extra 透传."""
    df = _sample_en()
    df["amount"] = [1000.0, 2000.0]
    df["turnover"] = [0.1, 0.2]
    *_, extra = dataframe_to_arrays(df, symbol="X")
    assert extra is not None
    assert set(extra) == {"amount", "turnover"}
    np.testing.assert_array_equal(extra["amount"], np.array([1000.0, 2000.0]))


def test_empty_frame_returns_empty_arrays() -> None:
    """空 DataFrame 返回空数组且保留 symbol 参数."""
    ts, o, *_rest, symbol, symbols, extra = dataframe_to_arrays(
        pd.DataFrame(), symbol="X"
    )
    assert len(ts) == 0 and len(o) == 0
    assert symbol == "X" and symbols is None and extra is None


def test_dataframe_to_bars_roundtrip() -> None:
    """DataFrame 转 Bar 列表, 字段与 symbol 正确."""
    bars = dataframe_to_bars(_sample_en(), symbol="X")
    assert len(bars) == 2
    assert isinstance(bars[0], Bar)
    assert bars[0].close == 11.0
    assert bars[0].symbol == "X"
    # 正常纳秒时间戳不应触发秒级自动修正
    assert bars[0].timestamp > 1_000_000_000_000_000_000


def test_dataframe_to_bars_seconds_autofix_branch() -> None:
    """秒级时间戳(转 UTC 后 < 1e10)应被 ×1e9 放大, 保留旧 load_bar_from_df 语义.

    取 UTC 落在 epoch 后 5 秒的时间(SH 08:00:05), 使 ns=5e9 触发自动修正而不溢出.
    """
    df = pd.DataFrame(
        {
            "date": ["1970-01-01 08:00:05", "1970-01-01 08:00:06"],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )
    bars = dataframe_to_bars(df, symbol="X")
    assert bars[0].timestamp == 5_000_000_000 * 1_000_000_000


def test_list_of_bars_to_frame() -> None:
    """list[Bar] 经 to_frame 还原为 DataFrame."""
    bars = dataframe_to_bars(_sample_en(), symbol="X")
    frame = to_frame(bars)
    assert list(frame.columns) >= ["open", "high", "low", "close", "volume"]
    assert len(frame) == 2


def test_normalize_returns_arrow_table() -> None:
    """Normalize 产出规范 Arrow Table, ts_event 为 UTC 纳秒."""
    tbl = normalize(_sample_en(), symbol="X")
    assert isinstance(tbl, pa.Table)
    assert tbl.schema.field("ts_event").type == pa.timestamp("ns", tz="UTC")
    for col in ("open", "high", "low", "close", "volume", "instrument_id"):
        assert col in tbl.column_names


def test_arrow_bridge_roundtrip_matches_arrays() -> None:
    """Normalize -> arrow_to_legacy_arrays 应与 dataframe_to_arrays 数值一致."""
    df = _sample_en()
    ts, o, h, low, c, v, *_ = dataframe_to_arrays(df, symbol="X")
    tbl = normalize(df, symbol="X")
    bts, bo, bh, blow, bc, bv, *_ = arrow_to_legacy_arrays(tbl)
    np.testing.assert_array_equal(ts, bts)
    np.testing.assert_array_equal(o, bo)
    np.testing.assert_array_equal(c, bc)
    np.testing.assert_array_equal(v, bv)


def test_pyarrow_input_equivalent_to_pandas() -> None:
    """pyarrow.Table 输入经 coerce 后与 pandas 数值等价."""
    df = _sample_en()
    tbl = pa.Table.from_pandas(df)
    coerced = coerce_to_pandas(tbl)
    assert isinstance(coerced, pd.DataFrame)
    a = dataframe_to_arrays(df, symbol="X")
    b = dataframe_to_arrays(coerced, symbol="X")
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[4], b[4])  # closes


def test_polars_input_equivalent_to_pandas() -> None:
    """polars.DataFrame 输入经 coerce 后与 pandas 数值等价(issue #298)."""
    pl = pytest.importorskip("polars")
    df = _sample_en()
    pldf = pl.from_pandas(df)
    coerced = coerce_to_pandas(pldf)
    assert isinstance(coerced, pd.DataFrame)
    a = dataframe_to_arrays(df, symbol="X")
    b = dataframe_to_arrays(coerced, symbol="X")
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[4], b[4])


def test_coerce_leaves_other_types_untouched() -> None:
    """coerce_to_pandas 不改动 pandas/其它类型(原样返回)."""
    df = _sample_en()
    assert coerce_to_pandas(df) is df
    marker = ["not", "a", "frame"]
    assert coerce_to_pandas(marker) is marker
