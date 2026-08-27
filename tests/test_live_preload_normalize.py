"""预热数据的规范化与分级校验(纯函数, 不碰 runner 状态)."""

from typing import Any

import pandas as pd
import pytest
from akquant.live._preload import normalize_preload_history


def _frame(symbol: str = "600000.SH", rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-08-20", periods=rows, freq="D"),
            "symbol": [symbol] * rows,
            "open": [10.0] * rows,
            "high": [11.0] * rows,
            "low": [9.0] * rows,
            "close": [10.5 + i for i in range(rows)],
            "volume": [100.0] * rows,
        }
    )


_FUTURE_NS = 4_000_000_000_000_000_000  # 远晚于任何测试数据


def test_dataframe_dict_and_bar_list_are_equivalent() -> None:
    """三种输入形态产出同样的 bars."""
    frame = _frame()
    allowed = {"600000.SH"}

    from_df = normalize_preload_history(frame, allowed, _FUTURE_NS)
    assert from_df is not None
    from_dict = normalize_preload_history({"600000.SH": frame}, allowed, _FUTURE_NS)
    from_bars = normalize_preload_history(from_df.bars, allowed, _FUTURE_NS)

    assert from_dict is not None and from_bars is not None
    closes = [[b.close for b in r.bars] for r in (from_df, from_dict, from_bars)]
    assert closes[0] == closes[1] == closes[2]
    assert from_df.depth == 3


def test_missing_required_column_raises() -> None:
    """必需列缺失 -> fail-fast, 且点名缺哪列."""
    frame = _frame().drop(columns=["close"])
    with pytest.raises(ValueError) as excinfo:
        normalize_preload_history(frame, {"600000.SH"}, _FUTURE_NS)
    message = str(excinfo.value)
    # 断言缺失字段列表本身, 确保 _require_columns 真正在工作
    assert "缺少必需列" in message and "close(" in message


def test_no_symbol_matches_raises_and_mentions_the_leading_zero_trap() -> None:
    """一个标的都没匹配上 -> fail-fast, 并提示前导 0 这个坑."""
    with pytest.raises(ValueError) as excinfo:
        normalize_preload_history(_frame("600000.SH"), {"000001.SZ"}, _FUTURE_NS)
    message = str(excinfo.value)
    assert "dtype" in message and "前导 0" in message


def test_partially_unmatched_symbols_are_dropped_with_warning(caplog: Any) -> None:
    """部分标的没匹配上 -> 裁掉 + 点名告警, 不阻断."""
    frame = pd.concat([_frame("600000.SH"), _frame("999999.SH")], ignore_index=True)
    with caplog.at_level("WARNING", logger="akquant.live.preload"):
        result = normalize_preload_history(frame, {"600000.SH"}, _FUTURE_NS)

    assert result is not None
    assert {b.symbol for b in result.bars} == {"600000.SH"}
    assert [r for r in caplog.records if "999999.SH" in r.getMessage()] != []


def test_future_rows_are_clipped_with_warning(caplog: Any) -> None:
    """晚于会话启动时刻的行被裁掉 + 告警(防策略看到未来)."""
    frame = _frame(rows=3)
    cutoff = int(pd.Timestamp("2026-08-21").value)  # 只保留前两行
    with caplog.at_level("WARNING", logger="akquant.live.preload"):
        result = normalize_preload_history(frame, {"600000.SH"}, cutoff)

    assert result is not None
    assert len(result.bars) == 2
    assert [r for r in caplog.records if "晚于会话启动时刻" in r.getMessage()] != []


def test_empty_container_warns_but_does_not_raise(caplog: Any) -> None:
    """空容器 -> 一条告警后跳过, 不 raise(多半是上游查历史返回空)."""
    with caplog.at_level("WARNING", logger="akquant.live.preload"):
        result = normalize_preload_history([], {"600000.SH"}, _FUTURE_NS)

    assert result is not None
    assert result.bars == []
    assert [r for r in caplog.records if "空数据" in r.getMessage()] != []


def test_none_is_a_silent_noop(caplog: Any) -> None:
    """None -> 返回 None 且一条日志都不打(行为与不传完全一致)."""
    with caplog.at_level("DEBUG", logger="akquant.live.preload"):
        assert normalize_preload_history(None, {"600000.SH"}, _FUTURE_NS) is None
    assert caplog.records == []


def test_depth_is_per_symbol_max_not_total_rows() -> None:
    """Depth 取各标的行数的最大值, 不是总行数(否则多标的会抬高一个数量级)."""
    frame = pd.concat(
        [_frame("600000.SH", 3), _frame("000001.SZ", 5)], ignore_index=True
    )
    result = normalize_preload_history(frame, {"600000.SH", "000001.SZ"}, _FUTURE_NS)

    assert result is not None
    assert len(result.bars) == 8
    assert result.depth == 5


def test_bars_are_sorted_by_symbol_then_timestamp() -> None:
    """缓冲不排序, 推入顺序即历史顺序 -> 规范化必须保证升序."""
    frame = _frame(rows=3).iloc[::-1]  # 倒序输入
    result = normalize_preload_history(frame, {"600000.SH"}, _FUTURE_NS)

    assert result is not None
    timestamps = [int(b.timestamp) for b in result.bars]
    assert timestamps == sorted(timestamps)


def test_dict_form_without_symbol_column_uses_dict_key() -> None:
    """Dict 形态 + 帧内无 symbol 列 -> 用 dict 的 key 补齐(Critical 1)."""
    frame_no_symbol = _frame().drop(columns=["symbol"])
    result = normalize_preload_history(
        {"600000.SH": frame_no_symbol}, {"600000.SH"}, _FUTURE_NS
    )

    assert result is not None
    assert len(result.bars) == 3
    assert all(b.symbol == "600000.SH" for b in result.bars)
    assert len(result.frames) == 1
    assert "600000.SH" in result.frames


def test_chinese_column_names_are_supported() -> None:
    """中文列名被支持(open -> 开盘, close -> 收盘等)."""
    frame = pd.DataFrame(
        {
            "日期": pd.date_range("2026-08-20", periods=2, freq="D"),
            "股票代码": ["600000.SH"] * 2,
            "开盘": [10.0, 10.0],
            "最高": [11.0, 11.0],
            "最低": [9.0, 9.0],
            "收盘": [10.5, 11.5],
            "成交量": [100.0, 100.0],
        }
    )
    result = normalize_preload_history(frame, {"600000.SH"}, _FUTURE_NS)

    assert result is not None
    assert len(result.bars) == 2
    assert result.bars[0].close == 10.5


def test_frames_index_is_utc_nanoseconds() -> None:
    """Frames 的索引是 UTC 纳秒(Important 4)."""
    frame = _frame(rows=2)
    result = normalize_preload_history(frame, {"600000.SH"}, _FUTURE_NS)

    assert result is not None
    assert len(result.frames) == 1
    df = result.frames["600000.SH"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None or str(df.index.tz) == "UTC"
    # 验证行数
    assert len(df) == 2
    # 验证列
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}


def test_last_timestamp_ns_contains_symbol_end_times() -> None:
    """Last_timestamp_ns 记录每个 symbol 的末行时刻(Important 4)."""
    frame = pd.concat(
        [_frame("600000.SH", 2), _frame("000001.SZ", 3)], ignore_index=True
    )
    result = normalize_preload_history(frame, {"600000.SH", "000001.SZ"}, _FUTURE_NS)

    assert result is not None
    assert len(result.last_timestamp_ns) == 2
    assert "600000.SH" in result.last_timestamp_ns
    assert "000001.SZ" in result.last_timestamp_ns
    # 600000.SH 的末行时间应该是第 2 行
    # 000001.SZ 的末行时间应该是第 3 行
    sh_end = result.last_timestamp_ns["600000.SH"]
    sz_end = result.last_timestamp_ns["000001.SZ"]
    assert sz_end > sh_end  # SZ 的末戳应该更晚


def test_multi_symbol_sorting_by_symbol_then_timestamp() -> None:
    """多标的交错乱序 -> 按 (symbol, timestamp) 排序(Important 5).

    必须让「按纯 timestamp 排」与「按 (symbol, timestamp) 排」产生不同结果,
    否则无法验证 symbol 分组确实在起作用.
    设计: 时间戳充分交错, 使得只按 timestamp 会严重打乱 symbol 分组.
    """
    # 喂入: ZZZZ(23), ZZZZ(21), AAAA(22), AAAA(20)
    # 按纯 timestamp 排: 20, 21, 22, 23 → AAAA, ZZZZ, AAAA, ZZZZ
    # 按 (symbol, ts) 排: AAAA(20), AAAA(22), ZZZZ(21), ZZZZ(23)
    # 两者完全不同！
    frame = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2026-08-23"),  # ZZZZ 第一行
                pd.Timestamp("2026-08-21"),  # ZZZZ 第二行
                pd.Timestamp("2026-08-22"),  # AAAA 第一行
                pd.Timestamp("2026-08-20"),  # AAAA 第二行
            ],
            "symbol": ["ZZZZ.SH", "ZZZZ.SH", "AAAA.SZ", "AAAA.SZ"],
            "open": [10.0] * 4,
            "high": [11.0] * 4,
            "low": [9.0] * 4,
            "close": [10.5, 11.5, 12.5, 13.5],
            "volume": [100.0] * 4,
        }
    )
    result = normalize_preload_history(frame, {"AAAA.SZ", "ZZZZ.SH"}, _FUTURE_NS)

    assert result is not None
    # 验证排序: 必须先按 symbol, 再按 timestamp
    # 按纯 timestamp 排会得到混合的 AAAA/ZZZZ/AAAA/ZZZZ, 会失败
    symbols = [str(b.symbol) for b in result.bars]
    assert symbols == ["AAAA.SZ", "AAAA.SZ", "ZZZZ.SH", "ZZZZ.SH"]
    # 每个 symbol 内部的时间戳应该升序
    aaaa_bars = [b for b in result.bars if b.symbol == "AAAA.SZ"]
    zzzz_bars = [b for b in result.bars if b.symbol == "ZZZZ.SH"]
    aaaa_ts = [int(b.timestamp) for b in aaaa_bars]
    zzzz_ts = [int(b.timestamp) for b in zzzz_bars]
    assert aaaa_ts == sorted(aaaa_ts)
    assert zzzz_ts == sorted(zzzz_ts)


def test_second_precision_list_bars_not_mutated() -> None:
    """list[Bar] 形态传入秒级戳, 数据不被改写(只告警, 不修改)."""
    from akquant import Bar

    # 创建秒级戳 list，用独特的 symbol
    src_bars = [
        Bar(
            symbol="TEST_NOMUT.SH",
            timestamp=1692547200,  # 秒级, < 1e10
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.5,
            volume=100.0,
        ),
        Bar(
            symbol="TEST_NOMUT.SH",
            timestamp=1692633600,
            open=10.0,
            high=11.0,
            low=9.0,
            close=11.5,
            volume=100.0,
        ),
    ]
    original_ts = [b.timestamp for b in src_bars]

    result = normalize_preload_history(src_bars, {"TEST_NOMUT.SH"}, _FUTURE_NS)

    assert result is not None
    assert len(result.bars) == 2
    # 最重要：验证调用方传入的 src 没被改写(只告警, 不改数据)
    # 这验证了新问题 2 的修复：不就地改调用方的数据
    assert src_bars[0].timestamp == original_ts[0]
    assert src_bars[1].timestamp == original_ts[1]
    # 结果中的 bars 应该指向同一对象(list 的浅拷贝)
    assert result.bars[0] is src_bars[0]
    assert result.bars[1] is src_bars[1]
    # 验证其他字段也没被改
    assert src_bars[0].open == 10.0
