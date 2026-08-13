from typing import Any, Optional, cast

import numpy as np
import pandas as pd

#: tick 序列可用的字段。tick 没有 open/high/low, 请求它们会静默返回退化 OHLC
#: (price 冒充 high), 故显式拒绝。
_TICK_ALLOWED_FIELDS = frozenset({"price", "close", "volume"})


def _validate_field_for_freq(field: str, freq: Optional[str]) -> None:
    """Tick 粒度下拒绝 bar 专有字段.

    :param field: 字段名(已小写)
    :param freq: 粒度, 'tick' / 'bar' / None
    :raises ValueError: tick 粒度下请求了 open/high/low
    """
    if freq == "tick" and field not in _TICK_ALLOWED_FIELDS:
        raise ValueError(
            f"freq='tick' 时不支持 field={field!r}: tick 没有 open/high/low, "
            f"可用字段为 {sorted(_TICK_ALLOWED_FIELDS)}"
        )


def _resolve_history_cutoff(strategy: Any) -> Optional[int]:
    """Return a history cutoff for day-boundary phases."""
    phase = getattr(strategy, "_framework_phase", None)
    if phase not in {"pre_open", "before_trading"}:
        return None
    cutoff = getattr(strategy, "_framework_history_cutoff_ns", None)
    if cutoff is None:
        return None
    return int(cutoff)


def set_history_depth(strategy: Any, depth: int) -> None:
    """设置历史数据回溯长度."""
    strategy._history_depth = depth


def set_rolling_window(strategy: Any, train_window: int, step: int) -> None:
    """设置滚动训练窗口参数."""
    strategy._rolling_train_window = train_window
    strategy._rolling_step = step
    if strategy._history_depth < train_window:
        strategy._history_depth = train_window


def get_history(
    strategy: Any,
    count: int,
    symbol: Optional[str] = None,
    field: str = "close",
    freq: Optional[str] = None,
) -> np.ndarray:
    """获取历史数据 (类似 Zipline data.history).

    :param freq: 粒度, ``'tick'`` / ``'bar'`` / ``None``。``None`` 时若该 symbol
        同时存在两条序列会报错, 要求显式指定(不静默选一条)。
    """
    if strategy._history_depth == 0:
        raise RuntimeError(
            "History tracking is not enabled. Call set_history_depth() first."
        )

    if strategy.ctx is None:
        raise RuntimeError("Context not ready")

    symbol = strategy._resolve_symbol(symbol)
    normalized_field = field.lower()
    _validate_field_for_freq(normalized_field, freq)
    history_cutoff = _resolve_history_cutoff(strategy)
    arr = strategy.ctx.history(symbol, normalized_field, count, history_cutoff, freq)

    if arr is None:
        return cast(np.ndarray, np.full(count, np.nan))

    if len(arr) < count:
        padding = np.full(count - len(arr), np.nan)
        return cast(np.ndarray, np.concatenate((padding, arr)))

    return cast(np.ndarray, arr)


def get_history_multi(
    strategy: Any,
    count: int,
    symbol: Optional[str] = None,
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
    freq: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """批量获取多个字段的历史数据 (单次跨界).

    行为与逐字段调用 :func:`get_history` 完全一致(相同的左侧 NaN 填充与
    截断语义),但只锁一次 Rust 缓冲、只跨一次 FFI 边界。返回按 ``fields``
    顺序建键的 ``{field: np.ndarray}``。

    :param freq: 粒度, ``'tick'`` / ``'bar'`` / ``None``。``None`` 时若该 symbol
        同时存在两条序列会报错, 要求显式指定(不静默选一条)。
    """
    if strategy._history_depth == 0:
        raise RuntimeError(
            "History tracking is not enabled. Call set_history_depth() first."
        )

    if strategy.ctx is None:
        raise RuntimeError("Context not ready")

    symbol = strategy._resolve_symbol(symbol)
    normalized_fields = [field.lower() for field in fields]
    for normalized_field in normalized_fields:
        _validate_field_for_freq(normalized_field, freq)
    history_cutoff = _resolve_history_cutoff(strategy)
    raw = strategy.ctx.history_multi(
        symbol, normalized_fields, count, history_cutoff, freq
    )

    out: dict[str, np.ndarray] = {}
    for field in normalized_fields:
        arr = None if raw is None else raw.get(field)
        if arr is None:
            out[field] = np.full(count, np.nan)
        elif len(arr) < count:
            padding = np.full(count - len(arr), np.nan)
            out[field] = np.concatenate((padding, arr))
        else:
            out[field] = arr
    return out


def get_history_df(
    strategy: Any, count: int, symbol: Optional[str] = None
) -> pd.DataFrame:
    """获取历史数据 DataFrame (Open, High, Low, Close, Volume)."""
    symbol = strategy._resolve_symbol(symbol)
    data = get_history_multi(
        strategy, count, symbol, ("open", "high", "low", "close", "volume")
    )
    return pd.DataFrame(data)


def get_rolling_data(
    strategy: Any, length: Optional[int] = None, symbol: Optional[str] = None
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """获取滚动训练数据."""
    if length is None:
        length = strategy._rolling_train_window

    if length <= 0:
        raise ValueError("Invalid rolling window length")

    df = get_history_df(strategy, length, symbol)
    return df, None
