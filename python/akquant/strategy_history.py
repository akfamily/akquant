from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from .log import build_log_extra, get_logger

logger = get_logger("strategy")

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
            f"可用字段为 {sorted(_TICK_ALLOWED_FIELDS)}; "
            "如需成交价序列请改用 get_history(freq='tick', field='price') 或 "
            "field='close'"
        )


#: 从当前回调名推断历史粒度。只有 on_bar / on_tick 这两个行情回调能确定意图:
#: 在 on_bar 里取历史显然是要 bar, 在 on_tick 里显然是要 tick。
_CALLBACK_FREQ = {"on_bar": "bar", "on_tick": "tick"}


def _infer_freq(strategy: Any, freq: Optional[str]) -> Optional[str]:
    """`freq` 省略时按当前所处的回调推断粒度.

    双流(同一 symbol 同时存在 bar 与 tick 两条历史序列)下省略 ``freq`` 会让
    Rust 侧 ``resolve_use_tick_history`` 抛歧义错误, 要求显式指定。该规则对
    「策略同时挂 on_bar 与 on_tick」是必要的, 但**对只挂 on_bar 的策略是误伤**:
    tick 序列由 ``HistoryBuffer::update_tick`` **无条件**写入(与策略是否覆写
    ``on_tick`` 无关, 见 src/pipeline/stages/data.rs), 于是用户只要订阅了 tick
    流(实盘 ``emit_ticks=True``, 回测 ``run_backtest(data=[Tick,...], freq=...)``)
    就会中招 —— 他从不读 tick, 却被要求在一个"他不知道存在"的维度上做选择,
    且报错直接中止会话。

    在 ``on_bar`` / ``on_tick`` 回调内, 调用点的意图毫无歧义, 故这里按当前回调
    定档。**回调之外(on_timer / on_before_trading / 用户自建线程等)仍返回原值**,
    让既有的歧义报错生效 —— 那些位置确实推断不出该取哪条。

    与 ``freq`` 已显式传入时不做任何干预(显式优先), 单流场景也完全不受影响
    (单流下 ``freq=None`` 本就直接命中唯一存在的那条序列, 推断出的值与它一致)。

    :param strategy: 策略实例
    :param freq: 用户传入的粒度, ``None`` 表示未指定
    :return: 推断后的粒度; 无法推断时原样返回 ``freq``
    """
    if freq is not None:
        return freq
    callback = getattr(strategy, "_framework_current_callback", None)
    if not isinstance(callback, str):
        return freq
    return _CALLBACK_FREQ.get(callback, freq)


def _resolve_history_cutoff(strategy: Any) -> Optional[int]:
    """Return a history cutoff for day-boundary phases."""
    phase = getattr(strategy, "_framework_phase", None)
    if phase not in {"pre_open", "before_trading"}:
        return None
    cutoff = getattr(strategy, "_framework_history_cutoff_ns", None)
    if cutoff is None:
        return None
    return int(cutoff)


def _log_missing_history_symbol(strategy: Any, symbol: str, field: str) -> None:
    """未登记 symbol 的全 NaN 历史留痕: 首次 WARNING 点名, 之后同 key 降 DEBUG.

    只应在 ``arr is None``(Rust 侧 history 缓冲对该 symbol 完全没有记录)时调用
    —— 这通常是配置错误(标的没进 ``instruments_config``/``symbols``, 或代码写
    错了标的), 值得告警并点名。**绝不能**用在"有数据但不够长"(预热不足)的
    分支: 那是每根 bar 都会触发的正常语义, 告警会刷屏。

    去重集合挂在 strategy 实例上惰性建, 与 ``gateway/broker_event_bridge.py``
    的 ``_log_foreign_symbol`` 同一套防刷屏模式。

    :param strategy: 策略实例, 用于挂载去重集合
    :param symbol: 未登记的标的代码
    :param field: 请求的字段名(已小写)
    """
    warned = getattr(strategy, "_warned_missing_history_symbols", None)
    if warned is None:
        warned = set()
        strategy._warned_missing_history_symbols = warned
    key = (symbol, field)
    first_time = key not in warned
    warned.add(key)
    log = logger.warning if first_time else logger.debug
    log(
        "get_history(symbol=%s, field=%s) 无历史记录, 返回全 NaN: 该 symbol 在历史"
        "缓冲中无任何记录, 通常意味着它没有被登记/订阅(检查 instruments_config/"
        "symbols 配置或标的代码是否写错), 而不是数据源当天没数据",
        symbol,
        field,
        extra=build_log_extra(phase="strategy", symbol=symbol),
    )


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
    # 字段校验用**用户显式传入的** freq, 不用推断值: 推断只为在双流下选对容器,
    # 不应改变字段合法性。纯 tick 单流下 `get_history(field='open')` 一直是
    # "静默返回退化 OHLC", 若这里拿推断出的 'tick' 去校验, 那些既有调用会突然
    # 开始报错(该退化行为的取舍见 _TICK_ALLOWED_FIELDS 注释, 属独立议题)。
    _validate_field_for_freq(normalized_field, freq)
    resolved_freq = _infer_freq(strategy, freq)
    history_cutoff = _resolve_history_cutoff(strategy)
    arr = strategy.ctx.history(
        symbol, normalized_field, count, history_cutoff, resolved_freq
    )

    if arr is None:
        _log_missing_history_symbol(strategy, symbol, normalized_field)
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
    # 同 get_history: 字段校验用显式 freq, 容器选择用推断后的 freq。
    for normalized_field in normalized_fields:
        _validate_field_for_freq(normalized_field, freq)
    resolved_freq = _infer_freq(strategy, freq)
    history_cutoff = _resolve_history_cutoff(strategy)
    raw = strategy.ctx.history_multi(
        symbol, normalized_fields, count, history_cutoff, resolved_freq
    )

    out: dict[str, np.ndarray] = {}
    for field in normalized_fields:
        arr = None if raw is None else raw.get(field)
        if arr is None:
            _log_missing_history_symbol(strategy, symbol, field)
            out[field] = np.full(count, np.nan)
        elif len(arr) < count:
            padding = np.full(count - len(arr), np.nan)
            out[field] = np.concatenate((padding, arr))
        else:
            out[field] = arr
    return out


def get_history_df(
    strategy: Any,
    count: int,
    symbol: Optional[str] = None,
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """获取历史数据 DataFrame (Open, High, Low, Close, Volume).

    :param freq: 粒度, ``'tick'`` / ``'bar'`` / ``None``。``None`` 时若该 symbol
        同时存在两条序列会报错, 要求显式指定(不静默选一条)。默认取的
        open/high/low/close/volume 五个字段在 ``freq='tick'`` 下必然报错
        (tick 没有 OHLC), 此时请改用 :func:`get_history` 配合
        ``field='price'`` 或 ``'close'`` 取成交价序列。
    """
    symbol = strategy._resolve_symbol(symbol)
    data = get_history_multi(
        strategy, count, symbol, ("open", "high", "low", "close", "volume"), freq
    )
    return pd.DataFrame(data)


def get_rolling_data(
    strategy: Any,
    length: Optional[int] = None,
    symbol: Optional[str] = None,
    freq: Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """获取滚动训练数据.

    :param freq: 粒度, ``'tick'`` / ``'bar'`` / ``None``, 透传给
        :func:`get_history_df`; 语义与限制同上。
    """
    if length is None:
        length = strategy._rolling_train_window

    if length <= 0:
        raise ValueError("Invalid rolling window length")

    df = get_history_df(strategy, length, symbol, freq)
    return df, None
