"""symbols 语义: 传了就只跑这些标的, 不传则沿用「数据即订阅」.

设计见 docs/superpowers/specs/2026-08-14-symbols-filtering-design.md
"""

from typing import Any

import akquant as aq
import pytest


class _RecordingStrategy(aq.Strategy):
    """记录每个标的的 on_bar 触发次数."""

    def on_start(self) -> None:
        """初始化命中表."""
        self.hits: dict[str, int] = {}

    def on_bar(self, bar: aq.Bar) -> None:
        """累计该标的的触发次数."""
        self.hits[bar.symbol] = self.hits.get(bar.symbol, 0) + 1


def _bars() -> list[aq.Bar]:
    """X 与 Y 各 3 根 bar."""
    out = []
    for i in range(3):
        for symbol, base in (("X", 10.0), ("Y", 100.0)):
            out.append(
                aq.Bar(
                    timestamp=f"2025-01-{2 + i:02d}",  # type: ignore[arg-type]
                    symbol=symbol,
                    open=base + i,
                    high=base + i,
                    low=base + i,
                    close=base + i,
                    volume=100.0,
                )
            )
    return out


def _feed() -> Any:
    """同样的数据装进 DataFeed —— 走 Rust 层过滤这条路径."""
    feed = aq.DataFeed()
    for bar in _bars():
        feed.add_bar(bar)
    feed.sort()
    return feed


def _run(data: Any, **kwargs: Any) -> dict[str, int]:
    strategy = _RecordingStrategy()
    aq.run_backtest(strategy=strategy, data=data, initial_cash=100000, **kwargs)
    return strategy.hits


def test_explicit_symbols_filters_list_of_bars() -> None:
    """传了 symbols=['X'] 时 Y 不得参与(List[Bar] 形态)."""
    assert _run(_bars(), symbols=["X"]) == {"X": 3}


def test_explicit_symbols_filters_data_feed() -> None:
    """DataFeed 形态同样被过滤 —— 这条只能靠 Rust 层白名单."""
    assert _run(_feed(), symbols=["X"]) == {"X": 3}


def test_omitting_symbols_keeps_data_as_subscription() -> None:
    """不传 symbols 时沿用「数据即订阅」, 两个标的都跑(回归底线)."""
    assert _run(_bars()) == {"X": 3, "Y": 3}


def test_empty_symbols_list_is_rejected() -> None:
    """显式传空列表是参数错误, 必须报错而非静默跑出空回测."""
    with pytest.raises(ValueError, match="symbols"):
        _run(_bars(), symbols=[])


def _bars_with_benchmark_symbol() -> list[aq.Bar]:
    """标的代码分别为 BENCHMARK 与 OTHER, 各 3 根 bar.

    BENCHMARK 同时是本文件内部代表"未指定标的"的哨兵字面量, 也是这里用作
    真实标的代码的测试数据 —— 二者刚好撞了同一个字符串, 这正是要验证的场景。
    """
    out = []
    for i in range(3):
        for symbol, base in (("BENCHMARK", 10.0), ("OTHER", 100.0)):
            out.append(
                aq.Bar(
                    timestamp=f"2025-01-{2 + i:02d}",  # type: ignore[arg-type]
                    symbol=symbol,
                    open=base + i,
                    high=base + i,
                    low=base + i,
                    close=base + i,
                    volume=100.0,
                )
            )
    return out


def test_explicit_symbols_literal_benchmark_still_filters() -> None:
    """symbols=["BENCHMARK"] 显式传入时必须真的过滤掉 "OTHER".

    回归护栏: 曾经的实现把"显式传入的 BENCHMARK"错误折算成"未显式传入",
    导致这种情况下过滤永远不生效、"OTHER" 会被静默一并跑出。
    """
    assert _run(_bars_with_benchmark_symbol(), symbols=["BENCHMARK"]) == {
        "BENCHMARK": 3
    }


def test_all_input_forms_agree_under_same_symbols() -> None:
    """同一份数据、同一个 symbols, 各输入形态的命中集合必须一致.

    这是本次变更的核心保证: 此前 DataFrame 形态与 DataFeed 形态在
    symbols 上的行为并不一致, 而不一致本身就是要修的缺陷。
    """
    import pandas as pd

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [f"2025-01-{2 + i:02d}" for i in range(3) for _ in range(2)]
            ),
            "symbol": ["X", "Y"] * 3,
            "open": [10.0, 100.0] * 3,
            "high": [10.0, 100.0] * 3,
            "low": [10.0, 100.0] * 3,
            "close": [10.0, 100.0] * 3,
            "volume": [100.0, 100.0] * 3,
        }
    )
    expected = {"X": 3}
    assert _run(_bars(), symbols=["X"]) == expected
    assert _run(_feed(), symbols=["X"]) == expected
    assert _run(frame, symbols=["X"]) == expected
    assert _run({"X": frame[frame["symbol"] == "X"]}, symbols=["X"]) == expected


def test_filtered_symbols_are_logged_once_in_summary(caplog: Any) -> None:
    """被过滤掉的标的只发一条汇总日志, 不逐个刷屏.

    传全市场数据只关心几个标的是本变更的主要动机场景, 逐标的告警会淹没输出。
    """
    with caplog.at_level("INFO"):
        _run(_bars(), symbols=["X"])
    filtered_lines = [r for r in caplog.records if "过滤" in r.getMessage()]
    assert len(filtered_lines) == 1
    assert "1" in filtered_lines[0].getMessage()


def test_dataframe_and_dict_forms_also_log_filtered_summary(caplog: Any) -> None:
    """DataFrame 与 dict 形态的既有过滤同样要发汇总日志(fix round 1, finding a).

    此前只有新增的 List[Bar] 段会写 filtered_out_symbols, DataFrame/dict 两段
    的既有过滤代码从未写入这个集合 —— 导致"传全市场数据只关心几个标的"这个
    主要动机场景下, 用 DataFrame 或 dict(DataFrame 是最常见的输入形态)传入
    时完全没有汇总日志, 只有 List[Bar] 才有。
    """
    import pandas as pd

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [f"2025-01-{2 + i:02d}" for i in range(3) for _ in range(2)]
            ),
            "symbol": ["X", "Y"] * 3,
            "open": [10.0, 100.0] * 3,
            "high": [10.0, 100.0] * 3,
            "low": [10.0, 100.0] * 3,
            "close": [10.0, 100.0] * 3,
            "volume": [100.0, 100.0] * 3,
        }
    )

    with caplog.at_level("INFO"):
        _run(frame, symbols=["X"])
    df_lines = [r for r in caplog.records if "过滤" in r.getMessage()]
    assert len(df_lines) == 1
    assert "Y" in df_lines[0].getMessage()

    caplog.clear()

    data_dict = {
        "X": frame[frame["symbol"] == "X"],
        "Y": frame[frame["symbol"] == "Y"],
    }
    with caplog.at_level("INFO"):
        _run(data_dict, symbols=["X"])
    dict_lines = [r for r in caplog.records if "过滤" in r.getMessage()]
    assert len(dict_lines) == 1
    assert "Y" in dict_lines[0].getMessage()


class _LeakyAdapter:
    """故意在响应里混入未被请求的标的, 模拟违反 DataFeedAdapter 契约的实现.

    仅需一个可调用的 `load` 属性即满足 `_is_data_feed_adapter` 的 duck-type
    判据(hasattr(value, "load") and callable(...)), 无需继承任何基类。
    """

    def load(self, request: Any) -> Any:
        """无视 request.symbol, 总是额外搭售一份 'LEAK' 标的的数据."""
        import pandas as pd

        rows = []
        for symbol, base in ((str(request.symbol), 10.0), ("LEAK", 999.0)):
            for i in range(3):
                rows.append(
                    {
                        "timestamp": pd.Timestamp(f"2025-01-{2 + i:02d}"),
                        "symbol": symbol,
                        "open": base + i,
                        "high": base + i,
                        "low": base + i,
                        "close": base + i,
                        "volume": 100.0,
                    }
                )
        return pd.DataFrame(rows).set_index("timestamp")


def test_adapter_leaked_symbol_does_not_pollute_whitelist(caplog: Any) -> None:
    """Adapter 违反契约返回未请求的标的时, 该标的不得进白名单(fix round 1, finding b).

    调用点会把 adapter 返回的 data_map 里每个 key 无条件 append 进 symbols,
    而 symbols 正是随后设进 Rust 引擎 set_symbol_whitelist 的同一个列表 ——
    不做处理的话, 过滤会对这个泄漏进来的标的静默失效(白名单本身被污染)。
    还要求发一条 warning(而不是混进 filtered_out_symbols 那条 INFO 汇总),
    因为这是 adapter 违反契约, 不是用户主动排除。
    """
    with caplog.at_level("WARNING"):
        hits = _run(_LeakyAdapter(), symbols=["X"])
    assert "LEAK" not in hits
    assert hits == {"X": 3}
    warning_lines = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("LEAK" in r.getMessage() for r in warning_lines)
    assert not any("过滤" in r.getMessage() for r in warning_lines)
