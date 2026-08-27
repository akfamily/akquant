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
    assert "close" in str(excinfo.value)


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
