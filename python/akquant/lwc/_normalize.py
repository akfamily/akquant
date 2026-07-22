"""Normalization helpers for the lightweight-charts report payloads.

This module turns :class:`~akquant.backtest.result.BacktestResult` fields and
user-supplied OHLCV frames into plain JSON-serializable dictionaries consumed
by the browser-side TradingView Lightweight Charts application. It is shared
by the static HTML report (``report.py``) and the interactive review server
(``server.py``) so both produce identical payloads.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# A-share convention: red for up/buy, green for down/sell.
UP_COLOR = "#ef5350"
DOWN_COLOR = "#26a69a"
_VOLUME_UP = "rgba(239, 83, 80, 0.45)"
_VOLUME_DOWN = "rgba(38, 166, 154, 0.45)"

_DATE_COLUMNS = (
    "date",
    "datetime",
    "timestamp",
    "time",
    "日期",
    "时间",
    "交易日期",
)
_COLUMN_ALIASES = {
    "open": ("open", "open_price", "开盘", "开盘价"),
    "high": ("high", "high_price", "最高", "最高价"),
    "low": ("low", "low_price", "最低", "最低价"),
    "close": ("close", "close_price", "收盘", "收盘价"),
    "volume": ("volume", "vol", "成交量"),
}
_SYMBOL_COLUMNS = ("symbol", "instrument", "code", "代码", "证券代码")


def _resolve_column(df: pd.DataFrame, aliases: Tuple[str, ...]) -> Optional[str]:
    """Find the first matching column name (case-insensitive fallback)."""
    for name in aliases:
        if name in df.columns:
            return name
    lowered = {str(c).lower(): c for c in df.columns}
    for name in aliases:
        col = lowered.get(name.lower())
        if col is not None:
            return col
    return None


def _extract_times(df: pd.DataFrame) -> pd.Series:
    """Extract bar timestamps as tz-naive UTC datetimes."""
    col = _resolve_column(df, _DATE_COLUMNS)
    if col is not None:
        times = pd.to_datetime(df[col])
    elif isinstance(df.index, pd.DatetimeIndex):
        times = pd.Series(pd.to_datetime(df.index))
    else:
        raise ValueError(
            "market data frame has no date/datetime column nor DatetimeIndex"
        )
    if getattr(times.dt, "tz", None) is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    return times.reset_index(drop=True)


def _time_token(ts: pd.Timestamp, intraday: bool) -> Any:
    """Convert a timestamp into a Lightweight Charts time token."""
    if intraday:
        return int(ts.value // 10**9)
    return ts.strftime("%Y-%m-%d")


def normalize_market_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert a user OHLCV frame into candle/volume series lists.

    :param df: DataFrame with OHLC(V) columns (English or Chinese names) and
        either a date-like column or a DatetimeIndex.
    :return: Dict with ``intraday`` flag, ``candles`` and ``volumes`` lists
        ready for ``series.setData``.
    :raises ValueError: If a required column cannot be located.
    """
    times = _extract_times(df)
    columns: Dict[str, Optional[str]] = {}
    for key, aliases in _COLUMN_ALIASES.items():
        col = _resolve_column(df, aliases)
        if col is None and key != "volume":
            raise ValueError("market data frame is missing a '%s' column" % key)
        columns[key] = col

    n = len(times)
    intraday = bool((times.dt.normalize() != times).any()) if n else False

    out = pd.DataFrame({"ts": times})
    for key in ("open", "high", "low", "close"):
        out[key] = pd.to_numeric(df[columns[key]].to_numpy(), errors="coerce")
    if columns["volume"] is not None:
        out["volume"] = (
            pd.to_numeric(df[columns["volume"]], errors="coerce").fillna(0.0).to_numpy()
        )
    else:
        out["volume"] = 0.0
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["token"] = out["ts"].map(lambda t: _time_token(t, intraday))
    out = out.drop_duplicates(subset="token", keep="last").sort_values("token")

    candles: List[Dict[str, Any]] = []
    volumes: List[Dict[str, Any]] = []
    for row in out.itertuples():
        token = row.token
        if intraday:
            token = int(token)
        candles.append(
            {
                "time": token,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
            }
        )
        volumes.append(
            {
                "time": token,
                "value": float(row.volume),
                "color": _VOLUME_UP if row.close >= row.open else _VOLUME_DOWN,
            }
        )
    return {"intraday": intraday, "candles": candles, "volumes": volumes}


def extract_trades_by_symbol(result: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Group closed-trade records from ``result.trades_df`` by symbol.

    :param result: A ``BacktestResult``-like object exposing ``trades_df``.
    :return: Mapping of symbol to a list of trade dictionaries. Timestamps are
        kept as pandas Timestamps under ``entry_ts``/``exit_ts`` and converted
        to chart tokens later (per payload, since intraday differs by symbol).
    """
    df = getattr(result, "trades_df", None)
    if df is None or len(df) == 0 or "symbol" not in df.columns:
        return {}
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("entry_time")

    def _f(row: Any, name: str) -> Optional[float]:
        value = getattr(row, name, None)
        if value is None or pd.isna(value):
            return None
        return float(value)

    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for symbol, grp in df.groupby("symbol"):
        records: List[Dict[str, Any]] = []
        for t in grp.itertuples():
            entry_ts = pd.Timestamp(t.entry_time).tz_localize(None)
            exit_ts = pd.Timestamp(t.exit_time).tz_localize(None)
            records.append(
                {
                    "side": str(getattr(t, "side", "long")).lower(),
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "entry_label": entry_ts.strftime("%Y-%m-%d %H:%M"),
                    "exit_label": exit_ts.strftime("%Y-%m-%d %H:%M"),
                    "entry_price": _f(t, "entry_price"),
                    "exit_price": _f(t, "exit_price"),
                    "quantity": _f(t, "quantity"),
                    "pnl": _f(t, "pnl"),
                    "net_pnl": _f(t, "net_pnl"),
                    "return_pct": _f(t, "return_pct"),
                }
            )
        by_symbol[str(symbol)] = records
    return by_symbol


def build_symbol_payload(
    symbol: str,
    df: pd.DataFrame,
    trades: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the full per-symbol payload: candles, volumes, markers, trades.

    :param symbol: Symbol code (used only for error messages).
    :param df: OHLCV frame for this symbol.
    :param trades: Trade records from :func:`extract_trades_by_symbol`.
    :return: JSON-serializable payload dict.
    """
    payload = normalize_market_data(df)
    intraday = payload["intraday"]
    bar_times = {c["time"] for c in payload["candles"]}

    markers: List[Dict[str, Any]] = []
    trade_records: List[Dict[str, Any]] = []
    for tr in trades or []:
        long_side = tr["side"] != "short"
        entry_token = _time_token(tr["entry_ts"], intraday)
        exit_token = _time_token(tr["exit_ts"], intraday)
        if entry_token in bar_times:
            markers.append(
                {
                    "time": entry_token,
                    "position": "belowBar" if long_side else "aboveBar",
                    "color": UP_COLOR if long_side else DOWN_COLOR,
                    "shape": "arrowUp" if long_side else "arrowDown",
                    "text": "B" if long_side else "S",
                }
            )
        if exit_token in bar_times:
            markers.append(
                {
                    "time": exit_token,
                    "position": "aboveBar" if long_side else "belowBar",
                    "color": DOWN_COLOR if long_side else UP_COLOR,
                    "shape": "arrowDown" if long_side else "arrowUp",
                    "text": "S" if long_side else "B",
                }
            )
        record = {k: v for k, v in tr.items() if k not in ("entry_ts", "exit_ts")}
        record["entry_time"] = entry_token
        record["exit_time"] = exit_token
        trade_records.append(record)

    markers.sort(key=lambda m: m["time"])
    payload["markers"] = markers
    payload["trades"] = trade_records
    return payload


def extract_equity(result: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract daily equity curve and drawdown series from a result object.

    :param result: A ``BacktestResult``-like object.
    :return: Tuple of (equity list, drawdown list) with business-day tokens.
    """
    series = None
    for attr in ("equity_curve_daily", "equity_curve"):
        candidate = getattr(result, attr, None)
        if candidate is not None and len(candidate) > 0:
            series = candidate
            break
    if series is None:
        return [], []
    s = pd.Series(series, dtype="float64").copy()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s.index = idx
    s = s[~s.index.duplicated(keep="last")].sort_index()
    daily = s.resample("D").last().ffill().dropna()
    drawdown = daily / daily.cummax() - 1.0
    equity = [
        {"time": t.strftime("%Y-%m-%d"), "value": float(v)} for t, v in daily.items()
    ]
    dd = [
        {"time": t.strftime("%Y-%m-%d"), "value": float(v)} for t, v in drawdown.items()
    ]
    return equity, dd


_PCT100_KIND = "pct100"  # value already expressed in percent
_PCT_KIND = "pct"  # decimal fraction, multiply by 100
_NUM_KIND = "num"
_MONEY_KIND = "money"
_INT_KIND = "int"

_METRIC_DEFS = (
    ("metrics", "total_return_pct", "总收益率", _PCT100_KIND),
    ("metrics", "annualized_return", "年化收益率", _PCT_KIND),
    ("metrics", "max_drawdown_pct", "最大回撤", _PCT100_KIND),
    ("metrics", "sharpe_ratio", "夏普比率", _NUM_KIND),
    ("metrics", "sortino_ratio", "索提诺比率", _NUM_KIND),
    ("metrics", "calmar_ratio", "卡玛比率", _NUM_KIND),
    ("metrics", "volatility", "年化波动率", _PCT_KIND),
    ("metrics", "win_rate", "胜率", _PCT100_KIND),
    ("metrics", "exposure_time_pct", "持仓时间占比", _PCT100_KIND),
    ("trade_metrics", "total_closed_trades", "平仓交易数", _INT_KIND),
    ("trade_metrics", "profit_factor", "盈亏比", _NUM_KIND),
    ("trade_metrics", "net_pnl", "净利润", _MONEY_KIND),
    ("trade_metrics", "total_commission", "总佣金", _MONEY_KIND),
    ("trade_metrics", "avg_return_pct", "平均单笔收益率", _PCT100_KIND),
    ("trade_metrics", "largest_win", "最大单笔盈利", _MONEY_KIND),
    ("trade_metrics", "largest_loss", "最大单笔亏损", _MONEY_KIND),
    ("trade_metrics", "sqn", "SQN", _NUM_KIND),
)


def _format_metric(value: Any, kind: str) -> Optional[str]:
    """Format a metric value for display, tolerating bad inputs."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    if kind == _PCT100_KIND:
        return f"{v:.2f}%"
    if kind == _PCT_KIND:
        return f"{v * 100:.2f}%"
    if kind == _MONEY_KIND:
        return f"{v:,.0f}"
    if kind == _INT_KIND:
        return f"{int(v):d}"
    return f"{v:.2f}"


def extract_metrics(result: Any) -> List[Dict[str, str]]:
    """Build a display-ready metric card list from a result object.

    :param result: A ``BacktestResult``-like object exposing ``metrics`` and
        ``trade_metrics`` attribute bags.
    :return: List of ``{"label", "value"}`` dicts; missing/NaN fields skipped.
    """
    items: List[Dict[str, str]] = []
    for owner, attr, label, kind in _METRIC_DEFS:
        bag = getattr(result, owner, None)
        value = getattr(bag, attr, None) if bag is not None else None
        if value is None:
            continue
        text = _format_metric(value, kind)
        if text is not None:
            items.append({"label": label, "value": text})
    return items


def coerce_market_data(
    market_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
    fallback_symbols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Normalize the ``market_data`` argument into a symbol-to-frame mapping.

    Accepts a ``{symbol: frame}`` dict, a single frame (optionally carrying a
    symbol column, in which case it is split or attributed), or ``None``.

    :param market_data: User-supplied market data.
    :param fallback_symbols: Symbols used to attribute a bare single frame.
    :return: Mapping of symbol code to OHLCV frame.
    """
    if market_data is None:
        return {}
    if isinstance(market_data, dict):
        return {str(k): v for k, v in market_data.items()}
    df = market_data
    sym_col = _resolve_column(df, _SYMBOL_COLUMNS)
    if sym_col is not None:
        unique = df[sym_col].dropna().unique()
        if len(unique) > 1:
            return {str(k): g for k, g in df.groupby(sym_col)}
        if len(unique) == 1:
            return {str(unique[0]): df}
    symbol = fallback_symbols[0] if fallback_symbols else "BENCHMARK"
    warnings.warn(
        "market_data has no symbol column; attributing all rows to %r. "
        "Pass a {symbol: frame} dict to review multiple symbols." % symbol
    )
    return {symbol: df}


def pick_initial_symbol(
    payloads: Dict[str, Any],
    trades_by_symbol: Dict[str, List[Dict[str, Any]]],
    preferred: Optional[str] = None,
) -> Optional[str]:
    """Choose the symbol shown when the page loads.

    Prefers ``preferred``, then the most-traded symbol that has chart data,
    then the first available payload, then the most-traded symbol overall.

    :param payloads: Symbol payloads already embedded in the page.
    :param trades_by_symbol: Trade records grouped by symbol.
    :param preferred: Explicit user choice.
    :return: Symbol code or ``None`` when nothing is available.
    """
    if preferred:
        return preferred
    ranked = sorted(trades_by_symbol.items(), key=lambda kv: len(kv[1]), reverse=True)
    for symbol, _ in ranked:
        if symbol in payloads:
            return symbol
    if payloads:
        return sorted(payloads)[0]
    if ranked:
        return ranked[0][0]
    return None
