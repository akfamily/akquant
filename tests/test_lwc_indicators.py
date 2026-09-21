"""``akquant.lwc`` 指标渲染: 适配层 / payload / 模板 / 增量消息.

设计见 docs/superpowers/specs/2026-09-21-lwc-indicators-design.md。适配层是
``chart/d3kline.py`` 的镜像: 同一份指标契约(7 值 render_type、整数 pane、
timestamp_ms)面向第二个具体前端(lightweight-charts v5)。
"""

from __future__ import annotations

import math
from typing import Any

from akquant.lwc._indicators import (
    LWC_SERIES_TYPE,
    lwc_pane_index,
    to_lwc_indicator_series,
    to_lwc_update,
)

_RENDER_TYPES = ("line", "area", "bar", "column", "histogram", "scatter", "signal")


def _def(key: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "indicator_key": key,
        "display_name": key.upper(),
        "pane": 0,
        "render_type": "line",
        "color": None,
        "reference_lines": [],
    }
    base.update(over)
    return base


def _pt(symbol: str, ts_ms: int, value: Any, key: str = "sma") -> dict[str, Any]:
    return {
        "indicator_key": key,
        "symbol": symbol,
        "timestamp_ms": ts_ms,
        "timestamp": ts_ms * 1_000_000,
        "value": value,
        "warmup": False,
    }


_DAY1 = 1_704_067_200_000  # 2024-01-01 00:00 UTC, ms
_DAY = 86_400_000


def test_every_render_type_maps_to_an_lwc_series_kind() -> None:
    """7 值 render_type 枚举必须全部有映射, 不能有值静默落到默认分支."""
    assert set(LWC_SERIES_TYPE) == set(_RENDER_TYPES)
    assert LWC_SERIES_TYPE["line"] == "Line"
    assert LWC_SERIES_TYPE["area"] == "Area"
    for bar_like in ("bar", "column", "histogram"):
        assert LWC_SERIES_TYPE[bar_like] == "Histogram"
    # scatter 用 Line 系列关掉连线只画点; signal 走 K 线 marker, 不建 series
    assert LWC_SERIES_TYPE["scatter"] == "Line"
    assert LWC_SERIES_TYPE["signal"] == "Marker"


def test_pane_index_offsets_sub_panes_past_volume() -> None:
    """Akquant pane 0 叠主图; pane k>=1 落到 LWC pane k+1 —— pane 1 已被成交量占用."""
    assert lwc_pane_index(0) == 0
    assert lwc_pane_index(1) == 2
    assert lwc_pane_index(4) == 5
    assert lwc_pane_index("2") == 3  # sqlite/JSON 常给字符串
    assert lwc_pane_index(None) == 0


def test_series_filters_by_symbol_and_aligns_daily_time() -> None:
    """只取本 symbol 的点; 日频用 'YYYY-MM-DD', 与 K 线 candles 同一套时间值."""
    defs = [_def("sma")]
    pts = [
        _pt("AAA", _DAY1, 1.0),
        _pt("BBB", _DAY1, 9.0),
        _pt("AAA", _DAY1 + _DAY, 2.0),
    ]

    out = to_lwc_indicator_series(defs, pts, symbol="AAA", intraday=False)

    assert len(out) == 1
    s = out[0]
    assert s["indicator_key"] == "sma"
    assert s["series_type"] == "Line"
    assert s["pane"] == 0
    assert s["points"] == [
        {"time": "2024-01-01", "value": 1.0},
        {"time": "2024-01-02", "value": 2.0},
    ]


def test_series_uses_utc_seconds_when_intraday() -> None:
    """日内用 UTCTimestamp 秒(整数), 而非毫秒或字符串."""
    ts = _DAY1 + 9 * 3_600_000 + 30 * 60_000
    out = to_lwc_indicator_series(
        [_def("sma")], [_pt("A", ts, 1.5)], "A", intraday=True
    )

    assert out[0]["points"] == [{"time": ts // 1000, "value": 1.5}]


def test_series_drops_none_and_nan_values() -> None:
    """预热期 None/NaN 不进 series —— LWC 收到 NaN 会画断线或报错."""
    pts = [
        _pt("A", _DAY1, None),
        _pt("A", _DAY1 + _DAY, math.nan),
        _pt("A", _DAY1 + 2 * _DAY, 3.0),
    ]
    out = to_lwc_indicator_series([_def("sma")], pts, "A", intraday=False)

    assert out[0]["points"] == [{"time": "2024-01-03", "value": 3.0}]


def test_series_dedups_by_time_keeping_last_and_sorts() -> None:
    """同一 time 多个点留最后一个; 输出按 time 严格递增 —— LWC 硬要求."""
    pts = [
        _pt("A", _DAY1 + _DAY, 2.0),
        _pt("A", _DAY1, 1.0),
        _pt("A", _DAY1, 1.5),  # 同一天, 后来者覆盖
    ]
    out = to_lwc_indicator_series([_def("sma")], pts, "A", intraday=False)

    assert out[0]["points"] == [
        {"time": "2024-01-01", "value": 1.5},
        {"time": "2024-01-02", "value": 2.0},
    ]


def test_series_passes_reference_lines_and_color_through() -> None:
    """reference_lines 透传为 price_lines, color 原样带出, 供前端建 priceLine."""
    defs = [
        _def(
            "rsi",
            pane=1,
            color="#e91e63",
            reference_lines=[
                {"value": 70.0, "label": "超买", "color": "#ef4444"},
                {"value": 30.0, "label": "超卖", "color": ""},
            ],
        )
    ]
    out = to_lwc_indicator_series(
        defs, [_pt("A", _DAY1, 50.0, key="rsi")], "A", intraday=False
    )

    s = out[0]
    assert s["pane"] == 2
    assert s["color"] == "#e91e63"
    assert s["display_name"] == "RSI"
    assert s["price_lines"] == [
        {"value": 70.0, "label": "超买", "color": "#ef4444"},
        {"value": 30.0, "label": "超卖", "color": None},
    ]


def test_series_omits_indicators_with_no_points_for_symbol() -> None:
    """该 symbol 一个点都没有的指标不输出空 series, 免得前端建一堆空副图."""
    defs = [_def("sma"), _def("rsi", pane=1)]
    out = to_lwc_indicator_series(defs, [_pt("A", _DAY1, 1.0)], "A", intraday=False)

    assert [s["indicator_key"] for s in out] == ["sma"]


def test_to_lwc_update_converts_bridge_point_message() -> None:
    """吃 to_indicator_message() 的 point 消息, 产出可直接 series.update() 的增量.

    ``confirmed`` 原样带出: L2 落地后临时点与确认点同 time, LWC update() 按
    time 覆盖, 前端零改动。
    """
    message = {
        "channel": "indicator",
        "type": "point",
        "symbol": "A",
        "indicator": {
            "indicator_key": "ema_fast",
            "display_name": "EMA10",
            "pane": 0,
            "render_type": "line",
            "symbol": "A",
            "timestamp": 1_704_067_200_000_000_000,
            "timestamp_ms": 1_704_067_200_000,
            "value": 12.5,
            "warmup": False,
            "confirmed": True,
            "scale_group": "",
            "meta": {},
        },
    }

    out = to_lwc_update(message)

    assert out == {
        "symbol": "A",
        "indicator_key": "ema_fast",
        "display_name": "EMA10",
        "pane": 0,
        "series_type": "Line",
        "color": None,
        "confirmed": True,
        "point": {"time": 1_704_067_200, "value": 12.5},
    }


def test_to_lwc_update_ignores_non_point_and_unready_values() -> None:
    """Snapshot 消息与未就绪(None/NaN)点返回 None, 调用方直接跳过."""
    assert to_lwc_update({"type": "snapshot", "snapshot": {}}) is None
    nan_msg = {
        "type": "point",
        "symbol": "A",
        "indicator": {"indicator_key": "x", "timestamp_ms": _DAY1, "value": math.nan},
    }
    assert to_lwc_update(nan_msg) is None


def test_to_lwc_update_defaults_confirmed_true_when_absent() -> None:
    """早期版本/第三方 sink 的消息没有 confirmed 字段, 默认当已确认."""
    msg = {
        "type": "point",
        "symbol": "A",
        "indicator": {"indicator_key": "x", "timestamp_ms": _DAY1, "value": 1.0},
    }
    out = to_lwc_update(msg)
    assert out is not None and out["confirmed"] is True


# --------------------------------------------------------------------------
# payload / 模板 / 端到端
# --------------------------------------------------------------------------
import pandas as pd  # noqa: E402
from akquant.lwc._payload import build_review_payload  # noqa: E402
from akquant.lwc._template import render_review_html  # noqa: E402


class _FakeResult:
    """最小 result 桩: trades_df + indicator_outputs."""

    def __init__(self, indicator_outputs: dict[str, Any] | None = None) -> None:
        self._outputs = indicator_outputs

    @property
    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def indicator_outputs(self) -> dict[str, Any]:
        return self._outputs or {}


def _daily_md(symbol: str, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(n)
            ],
            "open": [10.0 + i for i in range(n)],
            "high": [10.5 + i for i in range(n)],
            "low": [9.5 + i for i in range(n)],
            "close": [10.2 + i for i in range(n)],
            "volume": [1000.0] * n,
            "symbol": [symbol] * n,
        }
    )


def _outputs() -> dict[str, Any]:
    return {
        "definitions": [_def("sma"), _def("rsi", pane=1, color="#e91e63")],
        "instances": [],
        "points": [
            _pt("A", _DAY1, 10.0),
            _pt("A", _DAY1 + _DAY, 11.0),
            _pt("A", _DAY1, 55.0, key="rsi"),
            _pt("B", _DAY1, 99.0),  # 另一标的, 不该混进 A
        ],
    }


def test_payload_attaches_indicators_per_symbol() -> None:
    """每个 symbol 段带上自己的 indicators, 与 candles 同一套时间值."""
    payload = build_review_payload(
        _FakeResult(_outputs()), {"A": _daily_md("A"), "B": _daily_md("B")}
    )

    by_sym = {s["symbol"]: s for s in payload["symbols"]}
    a_keys = [i["indicator_key"] for i in by_sym["A"]["indicators"]]
    assert a_keys == ["sma", "rsi"]  # 按 (pane, key) 排: sma pane0, rsi pane2
    assert (
        by_sym["A"]["indicators"][0]["points"][0]["time"]
        == by_sym["A"]["candles"][0]["time"]
    )
    assert [i["indicator_key"] for i in by_sym["B"]["indicators"]] == ["sma"]
    assert by_sym["B"]["indicators"][0]["points"] == [
        {"time": "2024-01-01", "value": 99.0}
    ]


def test_payload_include_indicators_false_yields_empty_lists() -> None:
    """关掉后每段仍有 indicators 键(空列表), 前端不必判 undefined."""
    payload = build_review_payload(
        _FakeResult(_outputs()), {"A": _daily_md("A")}, include_indicators=False
    )
    assert payload["symbols"][0]["indicators"] == []


def test_payload_tolerates_result_without_indicator_outputs() -> None:
    """旧 result 桩 / 第三方对象没有 indicator_outputs 时不崩, 给空列表."""

    class _Bare:
        trades_df = pd.DataFrame()

    payload = build_review_payload(_Bare(), {"A": _daily_md("A")})
    assert payload["symbols"][0]["indicators"] == []


def test_template_embeds_indicator_series_and_price_lines() -> None:
    """渲染出的 HTML 内联 payload 含 indicators, 且 APP_JS 会建 series/priceLine."""
    payload = build_review_payload(_FakeResult(_outputs()), {"A": _daily_md("A")})
    html_text = render_review_html(payload, title="t", intraday=False)

    assert '"indicators":[' in html_text
    assert '"series_type":"Line"' in html_text
    # 前端逻辑必须真的消费这些字段, 否则 payload 白带
    for token in (
        "indicators",
        "series_type",
        "price_lines",
        "createPriceLine",
        "removeSeries",
    ):
        assert token in html_text, token
    # 光有函数定义不够, draw() 必须真的调用它 —— 否则指标 payload 白带
    assert "curSignals = drawIndicators(s)" in html_text
    assert "recolorIndicators()" in html_text


def test_template_escapes_indicator_display_name_xss() -> None:
    """display_name 来自用户, 经 _safe_json 转义后不能闭合 script."""
    outputs = _outputs()
    outputs["definitions"][0]["display_name"] = "</script><script>alert(1)</script>"
    payload = build_review_payload(_FakeResult(outputs), {"A": _daily_md("A")})
    html_text = render_review_html(payload, title="t", intraday=False)

    assert "<script>alert(1)</script>" not in html_text
    # 证明指标确实被嵌入了(只是被转义), 而不是压根没进 payload
    assert r"\u003c/script\u003e\u003cscript\u003ealert(1)" in html_text


def test_end_to_end_backtest_with_declared_indicators_renders_review(
    tmp_path: Any,
) -> None:
    """真跑 run_backtest + self.I(plot 参数) → viz.review() 产出含指标的 HTML."""
    import akquant as aq
    from akquant import Bar, Strategy, run_backtest

    md = _daily_md("E2E", n=6)

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), name="sma2", pane=0, color="#3f51b5")
            self.rsi = self.I(aq.RSI(2), name="rsi2", pane=1)

        def on_bar(self, bar: Bar) -> None:
            pass

    result = run_backtest(
        strategy=Probe,
        data=md,
        symbols=["E2E"],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
    )
    out = result.viz.review(md, filename=str(tmp_path / "r.html"))
    text = open(out, encoding="utf-8").read()

    assert '"indicator_key":"sma2"' in text
    assert '"indicator_key":"rsi2"' in text
    assert '"pane":2' in text  # rsi 的 akquant pane 1 → LWC pane 2

    out2 = result.viz.review(
        md, filename=str(tmp_path / "r2.html"), include_indicators=False
    )
    assert '"indicator_key":"sma2"' not in open(out2, encoding="utf-8").read()
