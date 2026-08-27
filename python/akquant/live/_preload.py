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
from ..normalize import dataframe_to_arrays, resolve_columns
from ..schema import COLUMN_ALIASES

logger = get_logger("live.preload")

PreloadInput = Union[pd.DataFrame, dict[str, pd.DataFrame], list[Bar]]

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# 合理纳秒时间戳的下限. 1e17 ns ≈ 1973-03。选它而不是更贴近"1970 年附近"
# 的 1e15(那只能拦住秒/毫秒级误当纳秒解析的情形, 拦不住微秒级——微秒级
# 当前时间戳约 1.756e15, 落在 1e15 之上, 会被 1e15 静默放过, 而它对应的
# 真实时刻其实是 1970-01-21): 1e17 能同时覆盖秒/毫秒/微秒三种单位被误当
# 纳秒解析的情形, 距任何真实市场数据(1990 年 ≈ 6.3e17 ns)仍有 6 倍余量,
# 不会误伤真实交易日期。不套用仓库里已有的 "< 1e10 视为秒级" 阈值
# (normalize.py 两处 + Rust Bar 构造器用的 1e12): 那个阈值只为了区分
# "秒还是纳秒"服务于 ×1e9 修正, 再抄一份只会让下次改阈值必漏一处; 而且它
# 对负数/1970 年附近的问题值表达力不够(负值本就远小于 1e10, 但真正想拦的
# 正是它)。这里只做"落在合理区间之外"的存在性判断, 不猜测原始单位, 也不
# 做任何修正。
_MIN_SANE_TIMESTAMP_NS = 100_000_000_000_000_000


def _warn_invalid_timestamps(bars: list[Bar]) -> None:
    """点名时间戳落在合理纳秒区间之外的 bar(只告警, 不改数据).

    在 ``_to_bars()`` 之后统一调用, 覆盖 DataFrame / dict / list[Bar] 三条
    输入路径 —— DataFrame/dict 路径可能因把秒/毫秒/微秒级整数误当纳秒解析
    而产出极小甚至负的时间戳(实测过 ``pd.to_datetime`` 对秒级 int 列的默认
    解释会产出负数纳秒戳); list[Bar] 路径的调用方也可能直接传入秒/毫秒/
    微秒级戳。这里不做任何 ×1e9/×1e6/×1e3 之类的修正(那是 ``normalize.py``
    的职责, 与回测同源), 只负责让问题在日志里可见, 避免静默产出错误年代的
    数据。

    去重用调用内局部集合(不是模块级全局): 同一个 symbol 在本次调用里第一次
    命中打 WARNING, 之后降级 DEBUG; 下一次调用(如下一次 ``run_live``)会
    重新从 WARNING 开始, 不会被上一次调用的状态压制。
    """
    warned: set[str] = set()
    for bar in bars:
        if int(bar.timestamp) >= _MIN_SANE_TIMESTAMP_NS:
            continue
        symbol = str(bar.symbol)
        extra = build_log_extra(phase="live", symbol=symbol)
        if symbol in warned:
            logger.debug(
                "preload_history[%s] 时间戳 %s 落在合理纳秒区间之外"
                "(< %s), 本次预热中已告警过, 不再重复",
                symbol,
                bar.timestamp,
                _MIN_SANE_TIMESTAMP_NS,
                extra=extra,
            )
            continue
        warned.add(symbol)
        logger.warning(
            "preload_history[%s] 时间戳 %s 落在合理纳秒区间之外"
            "(< %s, 疑似秒/毫秒/微秒级整数被误当纳秒解析), 未做任何修正"
            "(与回测同源), 请检查取数/转换逻辑",
            symbol,
            bar.timestamp,
            _MIN_SANE_TIMESTAMP_NS,
            extra=extra,
        )


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
    """检查必需列是否存在(支持别名: 英文/中文/大小写变体)."""
    resolved = resolve_columns(frame)
    missing_fields = [f for f in _REQUIRED_COLUMNS if f not in resolved]
    if missing_fields:
        # 点名缺失字段及其候选列名
        candidates = {f: COLUMN_ALIASES.get(f, [f]) for f in missing_fields}
        candidates_str = ", ".join(
            f"{f}({'/'.join(cands)})" for f, cands in candidates.items()
        )
        raise ValueError(
            f"preload_history 缺少必需列: {candidates_str}。"
            "需要含有时间、开盘/收盘/最高/最低/成交量、标的代码，"
            "与 examples/70_csv_multi_symbol_import_demo.py 的格式一致"
        )


def _to_bars(preload: PreloadInput) -> list[Bar]:
    """三种输入形态统一成 list[Bar](复用唯一的归一化实现)."""
    if isinstance(preload, pd.DataFrame):
        _require_columns(preload)
        ts, o, h, lo, c, v, symbol_val, symbols_list, extra = dataframe_to_arrays(
            preload
        )
        bars = from_arrays(ts, o, h, lo, c, v, symbol_val, symbols_list, extra)
        return bars
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
            ts, o, h, lo, c, v, symbol_val, symbols_list, extra = dataframe_to_arrays(
                frame
            )
            # 帧内没有 per-row symbol 列时，用 dict 的 key
            if symbols_list is None:
                symbol_val = str(symbol)
            bars = from_arrays(ts, o, h, lo, c, v, symbol_val, symbols_list, extra)
            collected.extend(bars)
        return collected
    # list[Bar] 形态：直接透传, 时间戳合理性检查统一在
    # normalize_preload_history() 里的 _warn_invalid_timestamps() 做
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

    _warn_invalid_timestamps(bars)
    bars = _filter_symbols(bars, allowed_symbols)
    bars = _clip_future(bars, session_start_ns)
    bars.sort(key=lambda b: (b.symbol, b.timestamp))
    return _build_result(bars)
