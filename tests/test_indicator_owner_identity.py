"""指标定义按 owner 区分身份: ``(owner_strategy_id, indicator_key)``.

主策略与指标脚本用同名键 ``ema`` 时, 定义不能互相吞掉; 图上 series 键只在
同名冲突时才加 ``owner:`` 前缀(见 spec 2026-10-20-indicator-owner-identity §2.1)。
"""

from __future__ import annotations

from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar, Indicator, Strategy, merge_results, run_backtest
from akquant.indicator_identity import display_label, series_keys

SYMBOL = "OWN_ID"


def _md(start: str = "2024-01-01", n: int = 60) -> pd.DataFrame:
    closes = [10.0 + (i % 7) * 0.5 + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq="D", tz="UTC"),
            "symbol": SYMBOL,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": 100.0,
        }
    )


class MainEma(Strategy):
    """主策略: 主图红色 EMA5."""

    def on_start(self) -> None:
        """声明主图 ema."""
        self.ema = self.I(aq.EMA(5), name="ema", plot=True, pane=0, color="red")

    def on_bar(self, bar: Bar) -> None:
        """不交易."""


class Sub(Indicator):
    """指标脚本: 同名 ema(EMA50), 蓝色, 默认附图."""

    def on_start(self) -> None:
        """声明同名 ema."""
        self.ema = self.I(aq.EMA(50), name="ema", color="blue")


def _run(start: str = "2024-01-01", n: int = 60) -> Any:
    return run_backtest(
        strategy=MainEma,
        data=_md(start, n),
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
        indicators=[Sub],
    )


# ---------------------------------------------------------------- series_keys


def test_series_keys_no_conflict_keeps_plain_key() -> None:
    """无冲突: series 键就是 indicator_key."""
    defs = [
        {"owner_strategy_id": "_default", "indicator_key": "ema"},
        {"owner_strategy_id": "sub", "indicator_key": "rsi"},
    ]
    assert series_keys(defs) == {("_default", "ema"): "ema", ("sub", "rsi"): "rsi"}


def test_series_keys_conflict_prefixes_owner() -> None:
    """同一 key 被多个 owner 使用: 全部加 owner 前缀, 其余键不变."""
    defs = [
        {"owner_strategy_id": "_default", "indicator_key": "ema"},
        {"owner_strategy_id": "sub", "indicator_key": "ema"},
        {"owner_strategy_id": "sub", "indicator_key": "rsi"},
    ]
    assert series_keys(defs) == {
        ("_default", "ema"): "_default:ema",
        ("sub", "ema"): "sub:ema",
        ("sub", "rsi"): "rsi",
    }


def test_series_keys_missing_owner_defaults() -> None:
    """缺失/空 owner 按 _default 处理, 与显式 _default 视为同一 owner."""
    defs: list[dict[str, Any]] = [
        {"indicator_key": "ema"},
        {"owner_strategy_id": "", "indicator_key": "ema"},
        {"owner_strategy_id": None, "indicator_key": "macd"},
        {"owner_strategy_id": "sub", "indicator_key": "macd"},
    ]
    assert series_keys(defs) == {
        ("_default", "ema"): "ema",
        ("_default", "macd"): "_default:macd",
        ("sub", "macd"): "sub:macd",
    }


def test_display_label_rule() -> None:
    """未写 label 且冲突 -> series 键; 写了 label 或无冲突 -> 原 display_name."""
    plain = {"owner_strategy_id": "sub", "indicator_key": "ema", "display_name": "ema"}
    assert display_label(plain, "sub:ema") == "sub:ema"
    assert display_label(plain, "ema") == "ema"
    labeled = {
        "owner_strategy_id": "sub",
        "indicator_key": "ema",
        "display_name": "EMA 50",
    }
    assert display_label(labeled, "sub:ema") == "EMA 50"
    missing = {"owner_strategy_id": "sub", "indicator_key": "ema"}
    assert display_label(missing, "sub:ema") == "sub:ema"


# ------------------------------------------------------------ 端到端: 双 ema


def test_dual_ema_definitions_keep_both_owners() -> None:
    """主策略与指标脚本同名 ema: 定义 2 行, pane/color 各自正确, owner 在第一列."""
    result = _run()
    defs = result.indicator_definitions
    assert list(defs.columns)[0] == "owner_strategy_id"
    assert len(defs) == 2
    rows = [
        (r.owner_strategy_id, r.indicator_key, int(r.pane), r.color)
        for r in defs.itertuples()
    ]
    assert rows == [("_default", "ema", 0, "red"), ("sub", "ema", 1, "blue")]
    payload_defs = result.indicator_outputs["definitions"]
    assert all("owner_strategy_id" in d for d in payload_defs)
    assert series_keys(payload_defs) == {
        ("_default", "ema"): "_default:ema",
        ("sub", "ema"): "sub:ema",
    }
    # 数据点本来就完整, 两个 owner 各一份
    points = result.indicator_df()
    assert set(points["owner_strategy_id"]) == {"_default", "sub"}


# --------------------------------------------------------- merge_results 跨段


def test_merge_results_dedups_definitions_by_owner_and_key() -> None:
    """跨段合并: 同 (owner, key) 只留一行, 不同 owner 的同名键各留一行."""
    first = _run("2024-01-01", 60)
    second = _run("2024-03-01", 60)
    merged = merge_results(first, second)
    defs = merged.indicator_outputs["definitions"]
    identities = sorted((d["owner_strategy_id"], d["indicator_key"]) for d in defs)
    assert identities == [("_default", "ema"), ("sub", "ema")]


# ------------------------------------------------- IndicatorRecorder 定义合并


def test_recorder_fills_empty_fields_only_within_same_owner() -> None:
    """同 (owner, key) 重复 record 补齐空字段; owner A 的 record 不补 owner B 的定义."""
    from akquant.indicator_recording import IndicatorRecorder

    recorder = IndicatorRecorder()
    ts = pd.Timestamp("2024-01-01 10:00").value
    recorder.record(
        name="ema", value=1.0, symbol="X", timestamp=ts, owner_strategy_id="a"
    )
    recorder.record(
        name="ema", value=2.0, symbol="X", timestamp=ts, owner_strategy_id="b"
    )
    recorder.record(
        name="ema",
        value=1.5,
        symbol="X",
        timestamp=ts + 1,
        owner_strategy_id="a",
        color="red",
        scale_group="price",
    )
    defs = {
        (d["owner_strategy_id"], d["indicator_key"]): d
        for d in recorder.build_payload()["definitions"]
    }
    assert set(defs) == {("a", "ema"), ("b", "ema")}
    assert defs[("a", "ema")]["color"] == "red"
    assert defs[("a", "ema")]["scale_group"] == "price"
    assert not defs[("b", "ema")]["color"]
    assert not defs[("b", "ema")]["scale_group"]


# ------------------------------------------------------------ 出口: 双 ema


def _last_values(result: Any) -> dict[str, float]:
    points = result.indicator_df().dropna(subset=["value"])
    points = points.sort_values("timestamp")
    return {
        str(owner): float(group["value"].iloc[-1])
        for owner, group in points.groupby("owner_strategy_id")
    }


def test_lwc_series_split_by_owner_with_prefixed_series_key() -> None:
    """LWC: 两个 owner 的同名 ema 各一条 series, 键加前缀, pane/末值各自正确."""
    from akquant.lwc import lwc_pane_index, to_lwc_indicator_series

    result = _run()
    outputs = result.indicator_outputs
    series = to_lwc_indicator_series(
        outputs["definitions"], outputs["points"], SYMBOL, intraday=False
    )
    assert len(series) == 2
    by_key = {s["series_key"]: s for s in series}
    assert set(by_key) == {"_default:ema", "sub:ema"}
    main, sub = by_key["_default:ema"], by_key["sub:ema"]
    assert main["owner_strategy_id"] == "_default"
    assert sub["owner_strategy_id"] == "sub"
    assert main["indicator_key"] == sub["indicator_key"] == "ema"
    assert main["pane"] == lwc_pane_index(0)
    assert sub["pane"] == lwc_pane_index(1)
    assert (main["color"], sub["color"]) == ("red", "blue")
    assert main["display_name"] == "_default:ema"
    last = _last_values(result)
    assert main["points"][-1]["value"] == last["_default"]
    assert sub["points"][-1]["value"] == last["sub"]
    assert last["_default"] != last["sub"]


def test_lwc_series_key_plain_when_single_owner() -> None:
    """单 owner: series_key 就是 indicator_key, 显示名不变."""
    from akquant.lwc import to_lwc_indicator_series

    result = run_backtest(
        strategy=MainEma,
        data=_md(),
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
    )
    outputs = result.indicator_outputs
    series = to_lwc_indicator_series(
        outputs["definitions"], outputs["points"], SYMBOL, intraday=False
    )
    assert [(s["series_key"], s["display_name"]) for s in series] == [("ema", "ema")]
    assert series[0]["owner_strategy_id"] == "_default"


def test_d3kline_names_distinct_and_points_per_owner() -> None:
    """d3kline: 两个 owner 的同名 ema 产出两个不同的 name, 点位各取各的."""
    from akquant.chart import (
        group_points_by_identity,
        to_d3kline_options,
        to_raw_panes,
    )

    result = _run()
    outputs = result.indicator_outputs
    defs = outputs["definitions"]
    points = group_points_by_identity(outputs["points"])
    options = to_d3kline_options(defs, points)
    series = {s["name"]: s for o in options for s in o["dataList"]}
    assert set(series) == {"_default:ema", "sub:ema"}
    last = _last_values(result)
    main_values = [p["value"] for p in series["_default:ema"]["data"]]
    sub_values = [p["value"] for p in series["sub:ema"]["data"]]
    assert main_values[-1] == last["_default"]
    assert sub_values[-1] == last["sub"]

    panes = to_raw_panes(defs, points)
    raw = {i["series_key"]: i for p in panes for i in p["indicators"]}
    assert set(raw) == {"_default:ema", "sub:ema"}
    assert raw["sub:ema"]["owner_strategy_id"] == "sub"
    assert raw["sub:ema"]["points"][-1]["value"] == last["sub"]


def test_plot_indicators_trace_names_distinct() -> None:
    """plotly: 按 (owner, key) 关联, 每个 owner 一条 trace, 名称两两不同."""
    import pytest

    pytest.importorskip("plotly")
    from akquant.plot import plot_indicators

    result = _run()
    fig = plot_indicators(result, show=False)
    assert fig is not None
    names = [trace.name for trace in fig.data]
    assert len(names) == 2
    assert sorted(names) == ["_default:ema", "sub:ema"]
    last = _last_values(result)
    by_name = {trace.name: trace for trace in fig.data}
    assert float(by_name["sub:ema"].y[-1]) == last["sub"]
    assert float(by_name["_default:ema"].y[-1]) == last["_default"]


class LabeledMain(Strategy):
    """主策略: 同名 ema 且写了 label="EMA"."""

    def on_start(self) -> None:
        """声明带 label 的主图 ema."""
        self.ema = self.I(aq.EMA(5), name="ema", plot=True, pane=0, label="EMA")

    def on_bar(self, bar: Bar) -> None:
        """不交易."""


class LabeledSub(Indicator):
    """指标脚本: 同名 ema, 同样 label="EMA"."""

    def on_start(self) -> None:
        """声明带同名 label 的 ema."""
        self.ema = self.I(aq.EMA(50), name="ema", label="EMA")


def test_plot_trace_names_distinct_when_both_owners_share_label() -> None:
    """冲突且两个 owner 都写了 label="EMA": label 遮住前缀, 追加 <owner> 区分."""
    import pytest

    pytest.importorskip("plotly")
    from akquant.plot import plot_indicators

    result = run_backtest(
        strategy=LabeledMain,
        data=_md(),
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
        indicators=[LabeledSub],
    )
    fig = plot_indicators(result, show=False)
    assert fig is not None
    names = sorted(trace.name for trace in fig.data)
    assert names == ["EMA <_default>", "EMA <labeled_sub>"]
