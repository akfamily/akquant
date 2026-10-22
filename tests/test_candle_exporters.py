"""K 线形态指标(render_type="candle")在各图上出口的表现.

见 spec ``docs/superpowers/specs/2026-10-21-candle-render-type-design.md`` C6:

- LWC: 同一 ``(owner, group)`` 的四个分量合成一条 ``Candlestick`` series;
- d3kline: 降级为只输出 close 分量的一条 line, 名字为组的 series 键;
- raw: 保留四个分量, 带 ``group``/``role``/``group_series_key``;
- plotly: ``go.Candlestick`` 轨迹;
- 组的 series 键: candle 组 ``(owner, group)`` 与普通线 ``(owner, key)`` 同一个
  命名空间, 冲突时加 ``owner:`` 前缀。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Indicator, Strategy, run_backtest
from akquant.chart import group_points_by_identity, to_d3kline_options, to_raw_panes
from akquant.indicator_identity import chart_series_keys
from akquant.lwc._indicators import to_lwc_indicator_series, to_lwc_update
from akquant.lwc._payload import build_review_payload
from akquant.lwc._template import render_review_html

SYMBOL = "CNDL"
OHLC = ("open", "high", "low", "close")
N = 12


def _md(n: int = N) -> pd.DataFrame:
    closes = [10.0 + (i % 5) * 0.5 + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "symbol": SYMBOL,
            "open": [c - 0.2 for c in closes],
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": 100.0,
        }
    )


def _expected() -> List[tuple[float, float, float, float]]:
    df = _md()
    ref = aq.HeikinAshi()
    rows = []
    for o, h, low, c in zip(df["open"], df["high"], df["low"], df["close"]):
        row = ref.update(o, h, low, c)
        assert row is not None
        rows.append(row)
    return rows


class HA(Indicator, title="平均K线", overlay=True, replaces_price=True):
    """主图平均 K 线(不写 label), 外加一条非 candle 的 EMA 作对照."""

    def on_start(self) -> None:
        """声明 candle 四分量与对照线."""
        self.ha = self.I(
            aq.HeikinAshi(),
            name="ha",
            input_mode="ohlc",
            outputs=OHLC,
            render_type="candle",
            color="#26a69a",
        )
        self.ema = self.I(aq.EMA(3), name="ema")


class HAPlain(Indicator, id="ha_plain"):
    """另一个 owner 的普通线, 名字恰好也叫 ha."""

    def on_start(self) -> None:
        """声明普通线 ha."""
        self.line = self.I(aq.SMA(3), name="ha")


class Quiet(Strategy):
    """不交易的主策略."""

    def on_bar(self, bar: Bar) -> None:
        """不交易."""


def _run(indicators: Any = (HA,)) -> tuple[Any, List[Dict[str, Any]]]:
    events: List[Any] = []
    result = run_backtest(
        strategy=Quiet,
        data=_md(),
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
        indicators=list(indicators),
        on_event=events.append,
    )
    messages = []
    for event in events:
        if aq.is_indicator_stream_event(event):
            message = aq.to_indicator_message(event)
            if message is not None:
                messages.append(message)
    return result, messages


def _defs_points(result: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    outputs = result.indicator_outputs
    return list(outputs["definitions"]), list(outputs["points"])


def _lwc(result: Any) -> List[Dict[str, Any]]:
    defs, points = _defs_points(result)
    return to_lwc_indicator_series(defs, points, SYMBOL, intraday=False)


def _main_d3_series(result: Any) -> Dict[str, Dict[str, Any]]:
    defs, points = _defs_points(result)
    options = to_d3kline_options(defs, group_points_by_identity(points))
    main = [o for o in options if o["isMain"]]
    assert len(main) == 1
    return {s["name"]: s for s in main[0]["dataList"]}


# ------------------------------------------------------------------ 键口径


def test_chart_series_keys_groups_share_namespace_with_lines() -> None:
    """组名与其他 owner 的同名普通线冲突时两边都加前缀; 分量口径不变."""
    defs = [
        {
            "owner_strategy_id": "a",
            "indicator_key": f"ha.{role}",
            "render_type": "candle",
            "group": "ha",
            "role": role,
        }
        for role in OHLC
    ] + [
        {"owner_strategy_id": "b", "indicator_key": "ha", "render_type": "line"},
        {"owner_strategy_id": "a", "indicator_key": "ema", "render_type": "line"},
    ]
    keys = chart_series_keys(defs)
    assert keys.groups == {("a", "ha"): "a:ha"}
    assert keys.lines == {("b", "ha"): "b:ha", ("a", "ema"): "ema"}
    # 分量口径与实时流一致: ha.open 只有一个 owner, 不加前缀
    assert keys.components[("a", "ha.open")] == "ha.open"
    assert keys.components[("b", "ha")] == "ha"


def test_chart_series_keys_candle_without_group_is_a_line() -> None:
    """缺 group 的 candle 行(旧存储)按普通线处理, 不崩."""
    keys = chart_series_keys(
        [{"owner_strategy_id": "a", "indicator_key": "x", "render_type": "candle"}]
    )
    assert keys.lines == {("a", "x"): "x"} and keys.groups == {}


# ------------------------------------------------------------------ LWC


def test_lwc_one_candlestick_per_group_with_heikin_ashi_values() -> None:
    """一组只出一条 Candlestick, 四个值等于 HeikinAshi 手算结果."""
    result, _ = _run()
    series = _lwc(result)
    candles = [s for s in series if s["series_type"] == "Candlestick"]
    assert len(candles) == 1
    candle = candles[0]
    assert (candle["series_key"], candle["group"], candle["render_type"]) == (
        "ha",
        "ha",
        "candle",
    )
    assert candle["color"] == "#26a69a"
    assert not any(str(s["indicator_key"]).startswith("ha.") for s in series)
    expected = _expected()
    assert len(candle["points"]) == N
    for point, row in zip(candle["points"], expected):
        assert [point[r] for r in OHLC] == pytest.approx(list(row))
    times = [p["time"] for p in candle["points"]]
    assert times == sorted(times) and len(set(times)) == N
    assert [s["series_type"] for s in series if s["indicator_key"] == "ema"] == ["Line"]


def test_lwc_incomplete_group_skips_that_timestamp() -> None:
    """某个时刻缺一个分量: 跳过该时刻, 其余时刻照常."""
    result, _ = _run()
    defs, points = _defs_points(result)
    dropped_ts = points[0]["timestamp_ms"] + 3 * 86_400_000
    points = [
        p
        for p in points
        if not (p["indicator_key"] == "ha.high" and p["timestamp_ms"] == dropped_ts)
    ]
    series = to_lwc_indicator_series(defs, points, SYMBOL, intraday=False)
    (candle,) = [s for s in series if s["series_type"] == "Candlestick"]
    assert len(candle["points"]) == N - 1
    assert pd.Timestamp(dropped_ts, unit="ms").strftime("%Y-%m-%d") not in {
        p["time"] for p in candle["points"]
    }


def test_lwc_group_missing_a_role_definition_does_not_crash() -> None:
    """定义里缺一个分量: 该组不出 series, 也不报错."""
    result, _ = _run()
    defs, points = _defs_points(result)
    defs = [d for d in defs if d["indicator_key"] != "ha.low"]
    series = to_lwc_indicator_series(defs, points, SYMBOL, intraday=False)
    assert all(s["series_type"] != "Candlestick" for s in series)
    assert [s["indicator_key"] for s in series] == ["ema"]


def test_to_lwc_update_carries_group_and_role() -> None:
    """流里的 candle 分量点: 带 group/role, series_type 为 Candlestick."""
    _, messages = _run()
    updates = [to_lwc_update(m) for m in messages if m["type"] == "point"]
    candle = [u for u in updates if u is not None and u["group"] == "ha"]
    assert {u["role"] for u in candle} == set(OHLC)
    assert all(u["series_type"] == "Candlestick" for u in candle)
    ema = [u for u in updates if u is not None and u["indicator_key"] == "ema"]
    assert ema and all((u["group"], u["role"]) == ("", "") for u in ema)


def test_review_html_embeds_candlestick_series() -> None:
    """复盘 payload 带 Candlestick series, HTML 渲染成功."""
    result, _ = _run()
    payload = build_review_payload(result, _md().set_index("timestamp"))
    (segment,) = payload["symbols"]
    kinds = [s["series_type"] for s in segment["indicators"]]
    assert kinds.count("Candlestick") == 1
    html = render_review_html(payload, title="t", intraday=False)
    assert '"series_type":"Candlestick"' in html
    assert "downColor:CANDLE_HOLLOW" in html  # 模板里的 Candlestick 分支


def test_review_hide_price_reaches_page_config(tmp_path: Any) -> None:
    """viz.review(hide_price=) 写进页面配置; 默认 False."""
    result, _ = _run()
    md = _md().set_index("timestamp")
    shown = result.viz.review(md, filename=str(tmp_path / "shown.html"))
    hidden = result.viz.review(
        md, filename=str(tmp_path / "hidden.html"), hide_price=True
    )
    assert '"hide_price":false' in open(shown, encoding="utf-8").read()
    assert '"hide_price":true' in open(hidden, encoding="utf-8").read()


def test_review_price_toggle_is_wired() -> None:
    """「隐藏 K 线」按钮真的挂了点击、draw() 里按 candle 指标刷新显隐."""
    result, _ = _run()
    payload = build_review_payload(result, _md().set_index("timestamp"))
    html = render_review_html(payload, title="t", intraday=False, hide_price=True)
    assert 'id="price-toggle"' in html
    assert "priceBtn.addEventListener('click'" in html
    assert "hasCandleInd = (s.indicators" in html  # draw() 内按 candle 指标定显隐
    # 主题切换也走 priceOptions, 否则切主题会把已隐藏的 K 线重新着色
    theme_fn = html[
        html.index("function applyTheme()") : html.index("function colorVol")
    ]
    assert "candle.applyOptions(priceOptions())" in theme_fn


# ------------------------------------------------------------------ d3kline / raw


def test_d3kline_only_close_line_named_by_group_key() -> None:
    """d3kline 降级: 一组只剩 close 一条 line, 名字为组的 series 键."""
    result, _ = _run()
    by_name = _main_d3_series(result)
    assert set(by_name) == {"ha", "ema"}
    ha = by_name["ha"]
    assert ha["type"] == "line" and ha["render_type"] == "candle"
    assert ha["label"] == "ha"
    assert ha["style"] == {"color": "#26a69a"}
    assert [d["value"] for d in ha["data"]] == pytest.approx(
        [row[3] for row in _expected()]
    )


def test_raw_panes_keep_components_with_group_role() -> None:
    """原始 pane 保留四个分量, 带 group/role/group_series_key; 非 candle 为空串."""
    result, _ = _run()
    defs, points = _defs_points(result)
    panes = to_raw_panes(defs, group_points_by_identity(points))
    entries = {e["indicator_key"]: e for p in panes for e in p["indicators"]}
    for role in OHLC:
        entry = entries[f"ha.{role}"]
        assert (entry["group"], entry["role"], entry["group_series_key"]) == (
            "ha",
            role,
            "ha",
        )
        assert entry["series_key"] == f"ha.{role}"
        assert len(entry["points"]) == N
    ema = entries["ema"]
    assert (ema["group"], ema["role"], ema["group_series_key"]) == ("", "", "")


# ------------------------------------------------------------------ plotly


def test_plotly_candlestick_trace() -> None:
    """plot_indicators: candle 组画成一个 go.Candlestick."""
    pytest.importorskip("plotly")
    from akquant.plot.indicator import plot_indicators

    result, _ = _run()
    fig = plot_indicators(result, show=False)
    assert fig is not None
    candles = [t for t in fig.data if t.type == "candlestick"]
    assert len(candles) == 1
    expected = _expected()
    for idx, field in enumerate(OHLC):
        assert list(getattr(candles[0], field)) == pytest.approx(
            [row[idx] for row in expected]
        )
    assert candles[0].name == "ha"
    # 分量不再作为单独的线出现
    assert not any(str(t.name).startswith("ha.") for t in fig.data)


# ------------------------------------------------------------------ 多 owner


def test_multi_owner_same_group_prefixed_in_every_exporter() -> None:
    """两个脚本都叫 ha: 各出口的组名一律为 owner:ha."""
    pytest.importorskip("plotly")
    from akquant.plot.indicator import plot_indicators

    class HA2(HA, id="ha_two"):
        pass

    result, _ = _run(indicators=(HA, HA2))
    want = {"ha:ha", "ha_two:ha"}
    candles = [s for s in _lwc(result) if s["series_type"] == "Candlestick"]
    assert {s["series_key"] for s in candles} == want
    assert want <= set(_main_d3_series(result))
    defs, points = _defs_points(result)
    raw = [
        e
        for p in to_raw_panes(defs, group_points_by_identity(points))
        for e in p["indicators"]
        if e["group"]
    ]
    assert {e["group_series_key"] for e in raw} == want
    fig = plot_indicators(result, show=False)
    assert fig is not None
    names = {t.name for t in fig.data if t.type == "candlestick"}
    assert names == want


def test_group_conflicts_with_other_owners_plain_line() -> None:
    """组名 ha 与另一 owner 的普通线 ha 冲突: 两边都加前缀."""
    result, _ = _run(indicators=(HA, HAPlain))
    series = {s["series_key"]: s for s in _lwc(result)}
    assert series["ha:ha"]["series_type"] == "Candlestick"
    assert series["ha_plain:ha"]["series_type"] == "Line"
    defs, points = _defs_points(result)
    options = to_d3kline_options(defs, group_points_by_identity(points))
    by_name = {s["name"]: s for o in options for s in o["dataList"]}
    assert {"ha:ha", "ha_plain:ha"} <= set(by_name)
    assert by_name["ha:ha"]["render_type"] == "candle"
    assert "render_type" not in by_name["ha_plain:ha"]


# ------------------------------------------------------------------ 配色(fix round 1)


def _candle_js_branch() -> str:
    from akquant.lwc._template import _APP_JS

    start = _APP_JS.index("if(ind.series_type==='Candlestick'){")
    return _APP_JS[start : _APP_JS.index("}", _APP_JS.index("title:", start))]


def test_lwc_candle_colours_never_use_theme_price_colours() -> None:
    """LWC: candle 单色 c(用户色或指标色板), 涨实心、跌空心; 绝不用主题涨跌色."""
    from akquant.lwc._template import INDICATOR_PALETTE
    from akquant.plot.utils import THEMES

    branch = _candle_js_branch()
    assert "T()" not in branch and "t.up" not in branch and "t.down" not in branch
    assert "downColor:CANDLE_HOLLOW" in branch and "upColor:c" in branch
    assert "borderDownColor:c" in branch and "wickDownColor:c" in branch
    price_colours = {
        THEMES[name][k].lower() for name in THEMES for k in ("up_color", "down_color")
    }
    palette = {c.lower() for colours in INDICATOR_PALETTE.values() for c in colours}
    assert not palette & price_colours

    result, _ = _run()
    payload = build_review_payload(result, _md().set_index("timestamp"))
    (candle,) = [
        s
        for s in payload["symbols"][0]["indicators"]
        if s["series_type"] == "Candlestick"
    ]
    assert candle["color"] == "#26a69a"
    assert candle["color"].lower() not in price_colours
    html = render_review_html(payload, title="t", intraday=False)
    assert "var CANDLE_HOLLOW = 'rgba(0,0,0,0)';" in html


def test_plotly_candle_hollow_down_not_theme_colour() -> None:
    """Plotly: increasing/decreasing 都显式设色, 下跌空心, 线色不是主题跌色."""
    pytest.importorskip("plotly")
    from akquant.plot.indicator import plot_indicators
    from akquant.plot.utils import THEMES

    result, _ = _run()
    for theme in ("light", "dark"):
        fig = plot_indicators(result, show=False, theme=theme)
        assert fig is not None
        (trace,) = [t for t in fig.data if t.type == "candlestick"]
        assert trace.decreasing.fillcolor == "rgba(0,0,0,0)"
        assert trace.decreasing.line.color == trace.increasing.line.color == "#26a69a"
        assert trace.increasing.fillcolor == "#26a69a"
        down = THEMES[theme]["down_color"].lower()
        up = THEMES[theme]["up_color"].lower()
        assert trace.decreasing.line.color.lower() not in {up, down}


def test_plotly_candle_without_colour_uses_indicator_palette() -> None:
    """未写 color(含 NaN/空串)的 candle 取指标色板, 不落到主题涨跌色."""
    pytest.importorskip("plotly")
    from akquant.lwc._template import INDICATOR_PALETTE
    from akquant.plot.indicator import _color_or_none, plot_indicators

    assert _color_or_none(float("nan")) is None
    assert _color_or_none("  ") is None and _color_or_none(None) is None
    assert _color_or_none(" #fff ") == "#fff"

    result, _ = _run()
    defs = [
        {**d, "color": float("nan") if d["group"] else d["color"]}
        for d in result.indicator_outputs["definitions"]
    ]
    result.indicator_outputs = {**result.indicator_outputs, "definitions": defs}
    result.__dict__.pop("indicator_definitions", None)
    fig = plot_indicators(result, show=False)
    assert fig is not None
    (trace,) = [t for t in fig.data if t.type == "candlestick"]
    assert trace.increasing.line.color == INDICATOR_PALETTE["light"][0]
    assert trace.decreasing.fillcolor == "rgba(0,0,0,0)"


# ------------------------------------------------------------------ 同 owner 撞名


def test_same_owner_plain_line_and_candle_group_disambiguated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """同一 owner 的普通线 ha 与 candle 组 ha: 组保持 ha, 普通线为 ha:line, 告警一次."""
    from akquant import indicator_identity

    owner = "same_owner_case"
    defs = [
        {
            "owner_strategy_id": owner,
            "indicator_key": f"ha.{role}",
            "render_type": "candle",
            "group": "ha",
            "role": role,
        }
        for role in OHLC
    ] + [{"owner_strategy_id": owner, "indicator_key": "ha", "render_type": "line"}]
    indicator_identity._warned_same_owner.discard((owner, "ha"))
    with caplog.at_level(logging.WARNING, logger="akquant.indicator_identity"):
        first = chart_series_keys(defs)
        second = chart_series_keys(defs)
    assert first.groups == {(owner, "ha"): "ha"}
    assert first.lines == {(owner, "ha"): "ha:line"}
    assert second == first
    hits = [r for r in caplog.records if "ha:line" in r.getMessage()]
    assert len(hits) == 1


def test_to_lwc_update_normalises_role() -> None:
    """to_lwc_update 的 role 归一为小写去空白, 与其他出口一致."""
    message = {
        "type": "point",
        "symbol": SYMBOL,
        "indicator": {
            "indicator_key": "ha.open",
            "render_type": "candle",
            "group": " ha ",
            "role": " Open ",
            "timestamp_ms": 1_704_067_200_000,
            "value": 1.0,
        },
    }
    out = to_lwc_update(message)
    assert out is not None
    assert (out["group"], out["role"], out["series_type"]) == (
        "ha",
        "open",
        "Candlestick",
    )
