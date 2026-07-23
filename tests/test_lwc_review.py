"""``akquant.lwc`` 交易复盘的 payload / 渲染 / 安全性测试.

覆盖 RFC(``docs/zh/meta/viz-namespace-and-lwc-review-rfc.md``):P1 —— payload
复用行情 normalizer、日频 vs 日内时间格式、买卖 marker 对齐、XSS 转义、
离线自包含(无 CDN、内联 LWC);P2 —— 主题无关 payload、页内明暗切换、
时间戳去重、日内大数据量压测。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from akquant.lwc._payload import (
    _bar_time,
    _is_intraday,
    build_review_payload,
)
from akquant.lwc._template import _safe_json, render_review_html
from akquant.plot.utils import THEMES


class _FakeResult:
    """最小 result 桩:仅提供 trades_df."""

    def __init__(self, trades: pd.DataFrame) -> None:
        self._trades = trades

    @property
    def trades_df(self) -> pd.DataFrame:
        """返回预置的成交表."""
        return self._trades


def _daily_md(symbol: str = "TEST", n: int = 5) -> pd.DataFrame:
    """构造 n 天日频行情(中文列名,验证 normalizer 复用)."""
    return pd.DataFrame(
        {
            "日期": [pd.Timestamp(f"2024-01-0{i + 1}") for i in range(n)],
            "开盘": [10.0 + i for i in range(n)],
            "最高": [10.5 + i for i in range(n)],
            "最低": [9.5 + i for i in range(n)],
            "收盘": [10.2 + i for i in range(n)],
            "成交量": [1000.0 + i for i in range(n)],
            "股票代码": [symbol] * n,
        }
    )


def _one_trade(symbol: str = "TEST") -> pd.DataFrame:
    """一笔多头成交(entry day1 / exit day4)."""
    return pd.DataFrame(
        {
            "symbol": [symbol],
            "side": ["long"],
            "entry_time": [pd.Timestamp("2024-01-01 10:00:00")],
            "exit_time": [pd.Timestamp("2024-01-04 10:00:00")],
            "entry_price": [10.2],
            "exit_price": [13.2],
        }
    )


def test_is_intraday_detection() -> None:
    """含时分的索引判为日内,纯日期判为日频."""
    daily = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
    intra = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01 09:30"), pd.Timestamp("2024-01-01 09:31")]
    )
    assert _is_intraday(daily) is False
    assert _is_intraday(intra) is True


def test_bar_time_daily_is_date_string() -> None:
    """日频时间值应为 'YYYY-MM-DD' BusinessDay 字符串."""
    assert _bar_time(pd.Timestamp("2024-01-03 10:00"), intraday=False) == "2024-01-03"


def test_bar_time_intraday_is_utc_seconds() -> None:
    """日内时间值应为 UTCTimestamp 秒(整数)."""
    ts = pd.Timestamp("2024-01-03 09:30:00")
    assert _bar_time(ts, intraday=True) == int(ts.value // 1_000_000_000)


def test_payload_reuses_normalizer_chinese_columns() -> None:
    """中文列名行情应被 normalizer 识别,产出规范 candles."""
    payload = build_review_payload(_FakeResult(_one_trade()), _daily_md())
    assert len(payload["symbols"]) == 1
    series = payload["symbols"][0]
    assert series["symbol"] == "TEST"
    assert len(series["candles"]) == 5
    first = series["candles"][0]
    assert set(first) == {"time", "open", "high", "low", "close"}
    assert first["time"] == "2024-01-01"


def test_payload_volume_theme_agnostic_up_flag() -> None:
    """量柱主题无关:只带 up 方向布尔,不烘焙颜色(前端按主题上色)."""
    payload = build_review_payload(_FakeResult(_one_trade()), _daily_md())
    vol = payload["symbols"][0]["volume"]
    assert len(vol) == 5
    assert all(set(v) == {"time", "value", "up"} for v in vol)
    assert all(isinstance(v["up"], bool) for v in vol)


def test_payload_markers_buy_then_sell() -> None:
    """多头成交产出买(belowBar/arrowUp)后卖(aboveBar/arrowDown),按时间升序."""
    payload = build_review_payload(_FakeResult(_one_trade()), _daily_md())
    markers = payload["symbols"][0]["markers"]
    assert len(markers) == 2
    buy, sell = markers
    assert buy["buy"] is True
    assert buy["position"] == "belowBar" and buy["shape"] == "arrowUp"
    assert sell["buy"] is False
    assert sell["position"] == "aboveBar" and sell["shape"] == "arrowDown"
    assert buy["time"] <= sell["time"]
    # 主题无关:marker 不烘焙 color
    assert "color" not in buy and "color" not in sell


def test_payload_short_trade_inverts_markers() -> None:
    """空头:入场=卖、出场=买(方向相反)."""
    tr = _one_trade()
    tr.loc[0, "side"] = "short"
    payload = build_review_payload(_FakeResult(tr), _daily_md())
    entry_marker = payload["symbols"][0]["markers"][0]
    assert entry_marker["shape"] == "arrowDown"
    assert entry_marker["position"] == "aboveBar"


def test_payload_empty_market_data_raises() -> None:
    """无有效行情应抛 ValueError."""
    with pytest.raises(ValueError):
        build_review_payload(_FakeResult(_one_trade()), pd.DataFrame())


def test_payload_no_trades_still_renders_candles() -> None:
    """无成交时仍产出 K 线,marker 为空."""
    payload = build_review_payload(_FakeResult(pd.DataFrame()), _daily_md())
    assert len(payload["symbols"][0]["candles"]) == 5
    assert payload["symbols"][0]["markers"] == []


def test_payload_multi_symbol_dict() -> None:
    """字典行情应为每个标的各产出一段序列."""
    md = {"AAA": _daily_md("AAA"), "BBB": _daily_md("BBB")}
    payload = build_review_payload(_FakeResult(_one_trade("AAA")), md)
    syms = {s["symbol"] for s in payload["symbols"]}
    assert syms == {"AAA", "BBB"}


def test_safe_json_escapes_script_breakout() -> None:
    """_safe_json 应把 < > & 转义,防止 </script> 提前闭合."""
    out = _safe_json({"x": "</script><script>alert(1)</script>"})
    assert "</script>" not in out
    assert "\\u003c" in out


def test_render_escapes_title_xss() -> None:
    """标题中的 HTML 应被转义,不produce 可执行 script."""
    payload = build_review_payload(_FakeResult(_one_trade()), _daily_md())
    html_text = render_review_html(
        payload,
        title="<script>alert(1)</script>",
        themes=THEMES,
        initial_theme="light",
        intraday=False,
    )
    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;" in html_text


def test_render_is_offline_selfcontained() -> None:
    """渲染结果应内联 LWC、无外链 CDN."""
    payload = build_review_payload(_FakeResult(_one_trade()), _daily_md())
    html_text = render_review_html(
        payload, title="t", themes=THEMES, initial_theme="dark", intraday=False
    )
    assert "LightweightCharts" in html_text
    lowered = html_text.lower()
    # 无 CDN、无脚本/样式外链(SVG 的 xmlns 命名空间 URL 不算网络抓取)
    assert "unpkg" not in lowered
    assert "cdn.jsdelivr" not in lowered and "cdn.plot" not in lowered
    assert 'src="http' not in lowered
    assert 'href="http' not in lowered


def test_plot_kline_review_writes_file(tmp_path: Path) -> None:
    """plot_kline_review 应写出 HTML 并返回其路径."""
    from akquant.lwc import plot_kline_review

    out = tmp_path / "review.html"
    path = plot_kline_review(_FakeResult(_one_trade()), _daily_md(), filename=str(out))
    assert out.exists()
    assert path == str(out.resolve())
    assert "LightweightCharts" in out.read_text(encoding="utf-8")


def test_plot_kline_review_none_market_data_raises() -> None:
    """market_data 为 None 应抛 ValueError."""
    from akquant.lwc import plot_kline_review

    with pytest.raises(ValueError):
        plot_kline_review(_FakeResult(_one_trade()), None)  # type: ignore[arg-type]


def test_initial_symbol_index_resolved(tmp_path: Path) -> None:
    """initial_symbol 命中时,payload 的初始下标应指向它."""
    from akquant.lwc import plot_kline_review

    md = {"AAA": _daily_md("AAA"), "BBB": _daily_md("BBB")}
    out = tmp_path / "r.html"
    plot_kline_review(
        _FakeResult(_one_trade("AAA")),
        md,
        symbols=["AAA", "BBB"],
        initial_symbol="BBB",
        filename=str(out),
    )
    text = out.read_text(encoding="utf-8")
    assert '"initial_symbol_index":1' in text


# --- P2:主题无关 payload / 明暗切换 / 去重 / 大数据量 ---


def test_payload_is_theme_agnostic() -> None:
    """Payload 不再烘焙颜色:量柱带 up 布尔、marker 带 buy 布尔,均无 color 键."""
    payload = build_review_payload(_FakeResult(_one_trade()), _daily_md())
    series = payload["symbols"][0]
    for v in series["volume"]:
        assert "color" not in v and isinstance(v["up"], bool)
    for m in series["markers"]:
        assert "color" not in m and isinstance(m["buy"], bool)


def test_render_inlines_both_themes_and_toggle() -> None:
    """渲染应内联明暗两套调色板 + 页内切换按钮."""
    payload = build_review_payload(_FakeResult(_one_trade()), _daily_md())
    text = render_review_html(payload, title="t", intraday=False)
    assert '"light"' in text and '"dark"' in text
    assert 'id="theme-toggle"' in text  # 切换按钮
    # 两套主题的背景色都应出现在内联 JSON 中
    assert THEMES["light"]["bg_color"].lower() in text.lower()
    assert THEMES["dark"]["bg_color"].lower() in text.lower()


def test_render_respects_initial_theme() -> None:
    """initial_theme=dark 时,初始主题键应为 dark."""
    payload = build_review_payload(_FakeResult(_one_trade()), _daily_md())
    text = render_review_html(payload, title="t", intraday=False, initial_theme="dark")
    assert '"initial_theme":"dark"' in text


def test_payload_dedupes_duplicate_timestamps() -> None:
    """重复时间戳应去重(LWC 要求严格升序唯一),保留最后一条."""
    md = _daily_md(n=3)
    dup = pd.concat([md, md.iloc[[2]]], ignore_index=True)  # 追加重复末行
    payload = build_review_payload(_FakeResult(pd.DataFrame()), dup)
    candles = payload["symbols"][0]["candles"]
    times = [c["time"] for c in candles]
    assert times == sorted(set(times)) and len(times) == 3


def _intraday_md(symbol: str, n: int) -> pd.DataFrame:
    """构造 n 根 1 分钟 K 线(日内),用于大数据量压测."""
    idx = pd.date_range("2024-01-01 09:30:00", periods=n, freq="min")
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.01, n),
            "high": close + np.abs(rng.normal(0, 0.03, n)),
            "low": close - np.abs(rng.normal(0, 0.03, n)),
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
            "symbol": symbol,
        },
        index=idx,
    )


def test_intraday_large_volume_payload() -> None:
    """日内大数据量(6万根)payload:整数秒时间、严格升序唯一、构建高效."""
    n = 60_000
    md = _intraday_md("BIG", n)
    trades = pd.DataFrame(
        {
            "symbol": ["BIG"],
            "side": ["long"],
            "entry_time": [md.index[10]],
            "exit_time": [md.index[n - 10]],
            "entry_price": [100.0],
            "exit_price": [101.0],
        }
    )
    t0 = time.perf_counter()
    payload = build_review_payload(_FakeResult(trades), md)
    elapsed = time.perf_counter() - t0

    series = payload["symbols"][0]
    candles = series["candles"]
    assert len(candles) == n
    times = [c["time"] for c in candles]
    # 日内时间为整数 UTC 秒
    assert all(isinstance(t, int) for t in times)
    # 严格递增且唯一(LWC 硬性要求)
    assert all(times[i] < times[i + 1] for i in range(len(times) - 1))
    # marker 命中已有 bar 时间
    bar_set = set(times)
    assert series["markers"]
    assert all(m["time"] in bar_set for m in series["markers"])
    # 向量化后 6 万根构建应远快于 iterrows;给宽松上限防 CI 抖动
    assert elapsed < 5.0, f"payload 构建过慢: {elapsed:.2f}s"
