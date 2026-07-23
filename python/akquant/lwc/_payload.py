"""LWC 交易复盘 payload 构建.

把 ``BacktestResult`` + 行情数据转成 lightweight-charts v5 可直接消费的
JSON 结构:每个标的一段 candles / volume / markers。时间轴按数据自动选择
日频(``'YYYY-MM-DD'`` BusinessDay 字符串)或日内(``UTCTimestamp`` 秒)。

复用 ``plot._market_data`` 的规范化逻辑,不重复造轮子。
"""

from __future__ import annotations

from typing import Any, Optional, Union, cast

import pandas as pd

from ..plot._market_data import extract_symbol_market_data

# lightweight-charts UTCTimestamp 以秒计;pandas Timestamp.value 是纳秒。
_NS_PER_SEC = 1_000_000_000


def _is_intraday(index: pd.DatetimeIndex) -> bool:
    """判断索引是否含日内(非零时分秒)时间点."""
    if len(index) == 0:
        return False
    normalized = index.normalize()
    return bool((index != normalized).any())


def _bar_time(ts: pd.Timestamp, intraday: bool) -> Union[str, int]:
    """把单个时间戳转为 LWC 时间值(日频字符串 / 日内 UTC 秒)."""
    if intraday:
        return int(ts.value // _NS_PER_SEC)
    return ts.strftime("%Y-%m-%d")


def _bar_times(index: pd.DatetimeIndex, intraday: bool) -> list[Union[str, int]]:
    """向量化把整个索引转为 LWC 时间值列表(日频字符串 / 日内 UTC 秒).

    注意:``DatetimeIndex`` 单位可能是 us/ms/ns(规范化后常为 ``datetime64[us]``),
    先统一到 ns 再换算成秒,避免按 ns 假设导致的量纲错误。
    """
    if intraday:
        ns = index.as_unit("ns").asi8
        return [int(s) for s in (ns // _NS_PER_SEC).tolist()]
    return index.strftime("%Y-%m-%d").tolist()


def _build_candles_and_volume(
    df: pd.DataFrame, intraday: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """由规范化行情构建 candlestick 与 volume 数据点(主题无关,向量化).

    颜色不在此烘焙:volume 仅带 ``up`` 方向布尔,由前端按当前主题上色,
    以支持页内明暗切换。大数据量(日内数万根)下用 numpy 数组而非 iterrows。
    """
    assert isinstance(df.index, pd.DatetimeIndex)
    times = _bar_times(df.index, intraday)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low_v = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    candles = [
        {"time": t, "open": oi, "high": hi, "low": li, "close": ci}
        for t, oi, hi, li, ci in zip(
            times, o.tolist(), h.tolist(), low_v.tolist(), c.tolist()
        )
    ]
    volume: list[dict[str, Any]] = []
    if "volume" in df.columns:
        vol = df["volume"].to_numpy(dtype=float)
        up_flags = (c >= o).tolist()
        volume = [
            {"time": t, "value": vi, "up": bool(u)}
            for t, vi, u in zip(times, vol.tolist(), up_flags)
            if vi == vi  # 过滤 NaN(NaN != NaN)
        ]
    return candles, volume


def _align_tz(ts: pd.Timestamp, bar_index: pd.DatetimeIndex) -> pd.Timestamp:
    """对齐时区,使成交时间可与 bar 索引比较.

    引擎的成交时间常为 tz-aware(UTC),而规范化行情索引多为 tz-naive。
    统一到 bar 索引的时区语义:索引 naive 则去掉 tz(先转到索引 tz 再抹除),
    索引 aware 则把 naive 的 ts 本地化到该 tz。
    """
    index_tz = getattr(bar_index, "tz", None)
    if ts.tz is not None and index_tz is None:
        return cast(
            pd.Timestamp, ts.tz_convert(None) if ts.tzinfo else ts.tz_localize(None)
        )
    if ts.tz is None and index_tz is not None:
        return cast(pd.Timestamp, ts.tz_localize(index_tz))
    return ts


def _snap_time(ts: pd.Timestamp, bar_index: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """把交易时间对齐到最近的已有 bar 时间(LWC 要求 marker 时间命中数据点).

    取 <= ts 的最后一根 bar;若 ts 早于首个 bar,则退回首个 bar。
    索引为空时返回 ``None``。
    """
    if len(bar_index) == 0:
        return None
    ts = _align_tz(ts, bar_index)
    pos = bar_index.searchsorted(ts, side="right") - 1
    if pos < 0:
        pos = 0
    return cast(pd.Timestamp, bar_index[pos])


def _build_markers(
    trades: pd.DataFrame,
    symbol: str,
    bar_index: pd.DatetimeIndex,
    intraday: bool,
) -> list[dict[str, Any]]:
    """由某标的的成交对构建买卖 marker(时间对齐到 bar,按时间升序).

    主题无关:仅带 ``buy`` 布尔与 position/shape,颜色由前端按主题上色。
    多头:入场=买(belowBar/arrowUp),出场=卖(aboveBar/arrowDown);
    空头方向相反。marker 时间对齐到最近 bar,避免 LWC 静默丢弃。
    """
    if trades.empty or len(bar_index) == 0:
        return []
    sym_trades = trades[trades["symbol"].astype(str).str.strip() == str(symbol).strip()]
    raw: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for _, tr in sym_trades.iterrows():
        side = str(tr.get("side", "long")).strip().lower()
        is_long = side != "short"
        for kind in ("entry", "exit"):
            ts = pd.to_datetime(tr.get(f"{kind}_time"), errors="coerce")
            if pd.isna(ts):
                continue
            snapped = _snap_time(ts, bar_index)
            if snapped is None:
                continue
            is_buy = (kind == "entry") if is_long else (kind == "exit")
            price = tr.get(f"{kind}_price")
            marker = {
                "time": _bar_time(snapped, intraday),
                "buy": bool(is_buy),
                "position": "belowBar" if is_buy else "aboveBar",
                "shape": "arrowUp" if is_buy else "arrowDown",
                "text": ("买" if is_buy else "卖")
                + (f" @{float(price):.2f}" if pd.notna(price) else ""),
            }
            raw.append((snapped, marker))
    raw.sort(key=lambda x: x[0])
    return [m for _, m in raw]


def _resolve_symbols(
    market_data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
    trades: pd.DataFrame,
    symbols: Optional[list[str]],
) -> list[str]:
    """决定要渲染哪些标的:显式指定 > 行情字典键 > 成交表标的."""
    if symbols:
        return [str(s).strip() for s in symbols]
    if isinstance(market_data, dict):
        return [str(k).strip() for k in market_data.keys()]
    if not trades.empty and "symbol" in trades.columns:
        return [str(s) for s in trades["symbol"].astype(str).str.strip().unique()]
    # 单表且无成交:优先用行情自带 symbol 列,否则退回单一合成标签
    if isinstance(market_data, pd.DataFrame) and not market_data.empty:
        for col in market_data.columns:
            if str(col).lower() in ("symbol", "code", "ticker", "ts_code", "股票代码"):
                uniq = market_data[col].astype(str).str.strip().unique()
                if len(uniq) > 0:
                    return [str(s) for s in uniq]
        return ["行情"]
    return []


def build_review_payload(
    result: Any,
    market_data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
    symbols: Optional[list[str]] = None,
) -> dict[str, Any]:
    """构建 LWC 复盘 payload(主题无关:颜色由前端按主题上色).

    :param result: BacktestResult(需提供 ``trades_df``).
    :param market_data: 单表或 ``{symbol: df}`` 行情.
    :param symbols: 可选,限定渲染的标的;默认取全部可用标的.
    :return: ``{"symbols": [{"symbol","candles","volume","markers"}, ...]}``.
    :raises ValueError: 无任何标的产出有效 K 线数据时.
    """
    trades = result.trades_df if hasattr(result, "trades_df") else pd.DataFrame()
    series: list[dict[str, Any]] = []
    for sym in _resolve_symbols(market_data, trades, symbols):
        df = extract_symbol_market_data(market_data, sym)
        if df.empty:
            continue
        index = df.index
        assert isinstance(index, pd.DatetimeIndex)
        # LWC 要求时间严格递增且唯一:去重(留最后一条)后再排序
        if index.has_duplicates:
            df = df[~index.duplicated(keep="last")]
            index = df.index
            assert isinstance(index, pd.DatetimeIndex)
        intraday = _is_intraday(index)
        candles, volume = _build_candles_and_volume(df, intraday)
        if not candles:
            continue
        markers = _build_markers(trades, sym, index, intraday)
        series.append(
            {
                "symbol": sym,
                "candles": candles,
                "volume": volume,
                "markers": markers,
            }
        )
    if not series:
        raise ValueError(
            "无法构建复盘数据:未找到任何标的的有效 OHLC 行情。"
            "请检查 market_data 是否包含目标标的且列名可识别。"
        )
    return {"symbols": series}
