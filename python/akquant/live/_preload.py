"""实盘历史预热的数据规范化与分级校验.

设计见 ``docs/superpowers/specs/2026-08-27-live-history-preload-design.md``。
这里只做纯函数式的转换与校验, 不碰 ``LiveRunner`` 的任何状态 —— 灌入时序、
``_history_depth`` 联动、重叠告警都在 ``_runner.py`` 里。

校验分级(设计 §5): 结构错 fail-fast, 数据问题裁剪 + 点名告警。分界线是
「几乎必然是配置错」还是「运行期可能合法发生」。
"""

from dataclasses import dataclass, field
from typing import Union

import pandas as pd

from ..akquant import Bar, from_arrays
from ..gateway.symbol_match import normalize_symbol_for_match
from ..log import build_log_extra, get_logger
from ..normalize import dataframe_to_arrays

logger = get_logger("live.preload")

PreloadInput = Union[pd.DataFrame, dict[str, pd.DataFrame], list[Bar]]

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class PreloadResult:
    """规范化后的预热数据与诊断信息."""

    bars: list[Bar] = field(default_factory=list)
    #: 供增量指标 bootstrap 复用(它要的是 {symbol: DataFrame})
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: per-symbol 最大行数, 不是总行数(设计 §4.2)
    depth: int = 0
    #: 每个 symbol 的预热末戳, 供实盘首根重叠告警(设计 §4.4)
    last_timestamp_ns: dict[str, int] = field(default_factory=dict)


def _require_columns(frame: pd.DataFrame) -> None:
    """检查必需列是否存在."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"preload_history 缺少必需列: {missing}; "
            f"需要 date/symbol/{'/'.join(_REQUIRED_COLUMNS)} 七列"
            "(与 examples/70_csv_multi_symbol_import_demo.py 的格式一致)"
        )


def _to_bars(preload: PreloadInput) -> list[Bar]:
    """三种输入形态统一成 list[Bar](复用唯一的归一化实现)."""
    if isinstance(preload, pd.DataFrame):
        _require_columns(preload)
        arrays = dataframe_to_arrays(preload)
        return from_arrays(*arrays)
    if isinstance(preload, dict):
        collected: list[Bar] = []
        for symbol, frame in preload.items():
            if frame is None or frame.empty:
                logger.warning(
                    "preload_history[%s] 为空, 跳过该标的",
                    symbol,
                    extra=build_log_extra(phase="live", symbol=str(symbol)),
                )
                continue
            _require_columns(frame)
            arrays = dataframe_to_arrays(frame)
            # 如果没有从 DataFrame 中解析到 symbol，使用传入的 symbol 参数
            if arrays[6] is None and arrays[7] is None:
                arrays = (*arrays[:6], str(symbol), None, arrays[8])
            collected.extend(from_arrays(*arrays))
        return collected
    return list(preload)


def _filter_symbols(bars: list[Bar], allowed_symbols: set[str]) -> list[Bar]:
    """按挂载标的过滤; 一个都没匹配上时 fail-fast(几乎必然是配置错)."""
    if not allowed_symbols:
        return bars
    kept = [b for b in bars if normalize_symbol_for_match(b.symbol) in allowed_symbols]
    if not kept:
        seen = sorted({str(b.symbol) for b in bars})[:5]
        raise ValueError(
            "preload_history 里没有任何标的匹配 run_live(instruments=...): "
            f"预热数据里的标的形如 {seen}, 挂载的是 {sorted(allowed_symbols)[:5]}。"
            "常见原因是 pandas.read_csv 把纯数字 symbol 推断成整数丢了前导 0"
            "(如 '002202' -> 2202), 读文件时请显式 dtype={'symbol': str}"
        )
    dropped = {
        normalize_symbol_for_match(b.symbol)
        for b in bars
        if normalize_symbol_for_match(b.symbol) not in allowed_symbols
    }
    for symbol in sorted(dropped):
        logger.warning(
            "preload_history 里的标的 %s 不在挂载列表内, 已忽略",
            symbol,
            extra=build_log_extra(phase="live", symbol=symbol),
        )
    return kept


def _clip_future(bars: list[Bar], session_start_ns: int) -> list[Bar]:
    """裁掉晚于会话启动时刻的行, 防止策略"看到未来"."""
    kept = [b for b in bars if int(b.timestamp) <= session_start_ns]
    dropped_count = len(bars) - len(kept)
    if dropped_count:
        symbols = sorted(
            {str(b.symbol) for b in bars if int(b.timestamp) > session_start_ns}
        )
        logger.warning(
            "preload_history 有 %s 行晚于会话启动时刻, 已裁掉(标的: %s)",
            dropped_count,
            symbols,
            extra=build_log_extra(phase="live"),
        )
    return kept


def _build_result(bars: list[Bar]) -> PreloadResult:
    """按 symbol 聚合出 frames / depth / 末戳."""
    frames: dict[str, pd.DataFrame] = {}
    last_ts: dict[str, int] = {}
    rows_by_symbol: dict[str, list[Bar]] = {}
    for bar in bars:
        rows_by_symbol.setdefault(str(bar.symbol), []).append(bar)
    for symbol, symbol_bars in rows_by_symbol.items():
        frames[symbol] = pd.DataFrame(
            {
                "open": [float(b.open) for b in symbol_bars],
                "high": [float(b.high) for b in symbol_bars],
                "low": [float(b.low) for b in symbol_bars],
                "close": [float(b.close) for b in symbol_bars],
                "volume": [float(b.volume) for b in symbol_bars],
            },
            index=pd.to_datetime([int(b.timestamp) for b in symbol_bars], unit="ns"),
        )
        last_ts[symbol] = int(symbol_bars[-1].timestamp)
    depth = max((len(v) for v in rows_by_symbol.values()), default=0)
    return PreloadResult(
        bars=bars, frames=frames, depth=depth, last_timestamp_ns=last_ts
    )


def normalize_preload_history(
    preload: PreloadInput | None,
    allowed_symbols: set[str],
    session_start_ns: int,
) -> PreloadResult | None:
    """把三种输入形态规范化成按 (symbol, timestamp) 升序的 bars.

    :param preload: 调用方传入的历史数据; ``None`` 表示不预热。
    :param allowed_symbols: 已归一化的挂载标的集合; 空集表示不做标的过滤。
    :param session_start_ns: 会话启动时刻(纳秒), 晚于它的行会被裁掉。
    :return: 规范化结果; ``preload`` 为 ``None`` 时返回 ``None``。
    :raises ValueError: 必需列缺失, 或传了数据但一个标的都没匹配上。
    """
    if preload is None:
        return None

    bars = _to_bars(preload)
    if not bars:
        logger.warning(
            "preload_history 传入了空数据, 跳过预热"
            "(若上游查历史返回空属正常, 否则检查取数条件)",
            extra=build_log_extra(phase="live"),
        )
        return PreloadResult()

    bars = _filter_symbols(bars, allowed_symbols)
    bars = _clip_future(bars, session_start_ns)
    bars.sort(key=lambda b: (b.symbol, b.timestamp))
    return _build_result(bars)
