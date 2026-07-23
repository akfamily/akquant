"""LWC 交易复盘入口:``plot_kline_review``."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from ..plot._market_data import extract_symbol_market_data
from ..plot.utils import THEMES
from ._payload import _is_intraday, build_review_payload
from ._template import render_review_html


def _detect_intraday(
    market_data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
    symbols: list[str],
) -> bool:
    """探测任一标的行情是否为日内频率."""
    for sym in symbols:
        df = extract_symbol_market_data(market_data, sym)
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            if _is_intraday(df.index):
                return True
    return False


def plot_kline_review(
    result: Any,
    market_data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
    *,
    symbols: Optional[list[str]] = None,
    title: str = "AKQuant 交易复盘",
    filename: str = "akquant_review.html",
    theme: str = "light",
    initial_symbol: Optional[str] = None,
    show: bool = False,
) -> str:
    """生成离线自包含的 LWC 交互式 K 线买卖点复盘 HTML.

    :param result: BacktestResult(提供 ``trades_df``).
    :param market_data: 单个 DataFrame 或 ``{symbol: df}`` 行情;必填.
    :param symbols: 可选,限定并排序要复盘的标的;默认全部可用标的.
    :param title: 报告标题(将被 HTML 转义).
    :param filename: 输出 HTML 路径.
    :param theme: ``"light"`` 或 ``"dark"``.
    :param initial_symbol: 初始展示的标的;缺省为首个.
    :param show: 是否在浏览器中打开.
    :return: 写出的 HTML 文件绝对路径.
    :raises ValueError: ``market_data`` 为空,或无有效行情可复盘.
    """
    if market_data is None:
        raise ValueError("plot_kline_review 需要 market_data(K 线复盘的行情来源)。")
    initial_theme = theme if theme in THEMES else "light"
    payload = build_review_payload(result, market_data, symbols=symbols)
    rendered_symbols = [s["symbol"] for s in payload["symbols"]]
    initial_index = 0
    if initial_symbol is not None:
        target = str(initial_symbol).strip()
        if target in rendered_symbols:
            initial_index = rendered_symbols.index(target)
    intraday = _detect_intraday(market_data, rendered_symbols)
    html_text = render_review_html(
        payload,
        title=title,
        themes=THEMES,
        initial_theme=initial_theme,
        intraday=intraday,
        initial_symbol_index=initial_index,
    )
    out_path = Path(filename).resolve()
    out_path.write_text(html_text, encoding="utf-8")
    if show:
        webbrowser.open(out_path.as_uri())
    return str(out_path)
