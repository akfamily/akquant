"""
参数优化模块 (Parameter Optimization).

提供类似 Backtrader optstrategy 的网格搜索功能.
"""

import itertools
import json
import logging
import multiprocessing
import pickle
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as datetime_time
from logging.handlers import QueueHandler, QueueListener
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from tqdm import tqdm  # type: ignore

from .backtest import run_backtest
from .log import get_logger
from .params_adapter import validate_strategy_params
from .strategy import Strategy

_WORKER_LOG_QUEUE: Any = None
OptimizationData = Union[pd.DataFrame, Dict[str, pd.DataFrame]]
logger = get_logger("optimize")

# run_grid_search / 进程池专用的关键字参数：这些键在 run_backtest 中没有对应形参，
# 一旦透传给 run_backtest 会落入 strategy_kwargs，被 strict_strategy_params 校验拒绝。
# run_walk_forward 的样本外验证 (run_backtest) 调用前必须过滤掉它们。
_GRID_SEARCH_ONLY_KWARGS = frozenset(
    {
        "max_workers",
        "db_path",
        "forward_worker_logs",
        "return_df",
    }
)


def _normalize_backtest_symbol_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(kwargs)
    has_symbol = "symbol" in normalized
    has_symbols = "symbols" in normalized
    if has_symbol and has_symbols:
        raise ValueError("pass only one of `symbol` or `symbols`")
    if has_symbol:
        normalized["symbols"] = normalized.pop("symbol")
    return normalized


def _normalize_symbol_values(symbols: Any) -> list[str]:
    """标准化 symbols 参数."""
    if symbols is None:
        return []
    if isinstance(symbols, str):
        normalized = [symbols]
    elif isinstance(symbols, (list, tuple, set)):
        normalized = [str(item) for item in symbols]
    else:
        raise TypeError("symbols must be a string, list, tuple, or set")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in normalized:
        value = str(item).strip()
        if not value:
            raise ValueError("symbols cannot contain empty values")
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def _infer_symbols_from_data(data: Any) -> list[str]:
    """从优化输入数据中推断 symbols."""
    if isinstance(data, dict):
        return [str(symbol).strip() for symbol in data.keys() if str(symbol).strip()]
    if isinstance(data, pd.DataFrame) and "symbol" in data.columns:
        symbol_series = data["symbol"].dropna().astype(str).str.strip()
        return [symbol for symbol in symbol_series.unique().tolist() if symbol]
    return []


def _resolve_optimization_backtest_kwargs(
    data: Any,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """解析优化入口的 symbols 参数并做数据一致性校验."""
    normalized = _normalize_backtest_symbol_kwargs(kwargs)
    requested_symbols = _normalize_symbol_values(normalized.get("symbols"))
    inferred_symbols = _infer_symbols_from_data(data)
    available_symbols = set(inferred_symbols)

    if not requested_symbols:
        if inferred_symbols:
            normalized["symbols"] = inferred_symbols
        return normalized

    if available_symbols:
        missing_symbols = [
            symbol for symbol in requested_symbols if symbol not in available_symbols
        ]
        if missing_symbols:
            raise ValueError(
                "Requested symbols are not available in optimization data: "
                f"{missing_symbols}"
            )

    normalized["symbols"] = requested_symbols
    return normalized


def _ensure_dataframe_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DataFrame 使用 DatetimeIndex 并按时间排序."""
    prepared = df
    if not isinstance(prepared.index, pd.DatetimeIndex):
        for column in ["date", "timestamp", "datetime", "Date", "Timestamp"]:
            if column in prepared.columns:
                prepared = prepared.set_index(column)
                break
        prepared = prepared.copy()
        prepared.index = pd.to_datetime(prepared.index)
    elif not prepared.index.is_monotonic_increasing:
        prepared = prepared.copy()

    if not prepared.index.is_monotonic_increasing:
        prepared = prepared.sort_index()
    return cast(pd.DataFrame, prepared)


def _filter_optimization_data_by_symbols(
    data: OptimizationData,
    symbols: Sequence[str],
) -> OptimizationData:
    """按 symbols 过滤优化数据."""
    if not symbols:
        return data

    symbol_set = set(symbols)
    if isinstance(data, pd.DataFrame):
        if "symbol" not in data.columns:
            return data
        filtered = data[data["symbol"].astype(str).isin(symbol_set)]
        return cast(pd.DataFrame, filtered.copy())

    filtered_map: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol in data:
            filtered_map[symbol] = data[symbol]
    return filtered_map


def _prepare_optimization_data(data: OptimizationData) -> OptimizationData:
    """标准化优化数据的时间索引."""
    if isinstance(data, pd.DataFrame):
        return _ensure_dataframe_time_index(data)

    prepared: dict[str, pd.DataFrame] = {}
    for symbol, df in data.items():
        prepared[str(symbol)] = _ensure_dataframe_time_index(df)
    return prepared


def _build_optimization_timeline(data: OptimizationData) -> pd.DatetimeIndex:
    """提取优化切窗使用的统一时间轴."""
    if isinstance(data, pd.DataFrame):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(
                "Optimization data must use DatetimeIndex after preparation"
            )
        return cast(pd.DatetimeIndex, data.index.unique().sort_values())

    timeline = pd.DatetimeIndex([])
    for df in data.values():
        if df.empty:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "Optimization data must use DatetimeIndex after preparation"
            )
        timeline = cast(pd.DatetimeIndex, timeline.union(df.index.unique()))
    return cast(pd.DatetimeIndex, timeline.sort_values())


def _slice_dataframe_by_time(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """根据时间窗口切片 DataFrame."""
    mask = df.index >= start_time
    if end_time is not None:
        mask = mask & (df.index < end_time)
    return cast(pd.DataFrame, df.loc[mask].copy())


def _slice_optimization_data(
    data: OptimizationData,
    start_time: pd.Timestamp,
    end_time: Optional[pd.Timestamp],
) -> OptimizationData:
    """根据统一时间窗口切片优化数据."""
    if isinstance(data, pd.DataFrame):
        return _slice_dataframe_by_time(data, start_time, end_time)

    sliced: dict[str, pd.DataFrame] = {}
    for symbol, df in data.items():
        window_df = _slice_dataframe_by_time(df, start_time, end_time)
        if not window_df.empty:
            sliced[symbol] = window_df
    return sliced


@dataclass
class OptimizationResult:
    """
    单个优化结果.

    :param params: 参数组合
    :param metrics: 性能指标字典
    :param duration: 回测耗时 (秒)
    :param error: 错误信息 (可选)
    """

    params: Dict[str, Any]
    metrics: Dict[str, Any]
    duration: float = 0.0
    error: Optional[str] = None

    def __repr__(self) -> str:
        """Return string representation."""
        if self.error:
            return f"OptimizationResult(params={self.params}, error={self.error})"
        return f"OptimizationResult(params={self.params}, metrics={self.metrics})"


def _run_backtest_safe(
    strategy_cls: Type[Strategy],
    kwargs: Dict[str, Any],
    result_container: Dict[str, Any],
) -> None:
    """Run backtest in a thread and store result/exception."""
    try:
        kwargs = _normalize_backtest_symbol_kwargs(kwargs)
        # 运行回测
        # 注意：show_progress 在并行时最好关掉
        kwargs["show_progress"] = False
        result = run_backtest(strategy=strategy_cls, **kwargs)
        metrics_df = result.metrics_df

        if "Backtest" in metrics_df.columns:
            metrics = cast(Dict[str, Any], metrics_df["Backtest"].to_dict())
        else:
            metrics = cast(Dict[str, Any], metrics_df.iloc[:, 0].to_dict())

        result_container["metrics"] = metrics
    except Exception as e:
        result_container["error"] = str(e)


def _run_single_backtest(args: Dict[str, Any]) -> OptimizationResult:
    """
    运行单个回测任务 (Internal).

    args 包含:
    - strategy_cls: 策略类
    - params: 当前参数组合
    - backtest_kwargs: run_backtest 的其他参数 (data, cash, etc.)
    - warmup_calc: 动态预热期计算函数 (可选)
    - timeout: 超时时间 (秒, 可选)

    :param args: 任务参数字典
    :return: 优化结果
    """
    strategy_cls = args["strategy_cls"]
    params = args["params"]
    backtest_kwargs = args["backtest_kwargs"]
    warmup_calc = args.get("warmup_calc")
    timeout = args.get("timeout")

    # 将参数合并到 kwargs 中传给 strategy
    kwargs = backtest_kwargs.copy()
    kwargs.update(params)

    # 动态计算 warmup_period
    if warmup_calc:
        try:
            dynamic_warmup = warmup_calc(params)
            base_warmup = kwargs.get("warmup_period", 0)
            kwargs["warmup_period"] = max(base_warmup, dynamic_warmup)
        except Exception as exc:
            logger.warning(
                "Failed to calculate dynamic warmup period: %s",
                exc,
            )

    start_time = time.time()
    metrics: Dict[str, Any] = {}
    error_msg: Optional[str] = None

    if timeout:
        # 使用线程运行回测，支持超时
        result_container: Dict[str, Any] = {}
        t = threading.Thread(
            target=_run_backtest_safe,
            args=(strategy_cls, kwargs, result_container),
            daemon=True,
        )
        t.start()
        t.join(timeout)

        if t.is_alive():
            # 超时
            error_msg = f"Timeout after {timeout} seconds"
            metrics = {"error": error_msg}
            # 设置默认 bad metrics 以便后续排序不报错
            metrics["sharpe_ratio"] = -999.0
            metrics["total_return"] = -999.0
            # 注意：无法强制杀死线程，但如果使用了 maxtasksperchild=1，
            # 当前进程会在任务结束后退出，从而清理线程。
        else:
            # 正常结束
            if "error" in result_container:
                error_msg = result_container["error"]
                metrics = {"error": error_msg}
                metrics["sharpe_ratio"] = -999.0
                metrics["total_return"] = -999.0
            else:
                metrics = result_container.get("metrics", {})

    else:
        # 直接运行
        try:
            kwargs["show_progress"] = False
            result = run_backtest(strategy=strategy_cls, **kwargs)
            metrics_df = result.metrics_df
            if "Backtest" in metrics_df.columns:
                metrics = cast(Dict[str, Any], metrics_df["Backtest"].to_dict())
            else:
                metrics = cast(Dict[str, Any], metrics_df.iloc[:, 0].to_dict())
        except Exception as e:
            error_msg = str(e)
            metrics = {"error": error_msg}
            metrics["sharpe_ratio"] = -999.0
            metrics["total_return"] = -999.0

    duration = time.time() - start_time

    return OptimizationResult(
        params=params, metrics=metrics, duration=duration, error=error_msg
    )


def _init_worker_logging(log_queue: Any) -> None:
    """Initialize worker logger with queue handler."""
    global _WORKER_LOG_QUEUE
    _WORKER_LOG_QUEUE = log_queue
    if log_queue is None:
        return
    logger = get_logger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(QueueHandler(log_queue))


class JSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder for numpy types."""

    def default(self, obj: Any) -> Any:
        """Encode object."""
        if obj is pd.NaT:
            return None
        if isinstance(obj, pd.Timestamp):
            if pd.isna(obj):
                return None
            return obj.isoformat()
        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()
        if isinstance(obj, (datetime, date, datetime_time)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _assert_parallel_pickleable(
    strategy: Type[Strategy],
    backtest_kwargs: Mapping[str, Any],
    warmup_calc: Optional[Any],
) -> None:
    """
    Validate critical multiprocessing inputs are pickle-serializable.

    :param strategy: 策略类
    :param backtest_kwargs: run_backtest kwargs (不包含 data)
    :param warmup_calc: 动态预热函数
    :raises TypeError: 当关键输入无法序列化时抛出
    """
    strategy_module = getattr(strategy, "__module__", "")
    if strategy_module == "__main__":
        raise TypeError(
            "Parallel optimization requires strategy class importable from module, "
            "but got strategy defined in __main__. "
            "Please move strategy class to a module file."
        )

    checks: List[tuple[str, Any]] = [("strategy", strategy)]
    if warmup_calc is not None:
        checks.append(("warmup_calc", warmup_calc))

    sensitive_keys = (
        "fill_policy",
        "on_event",
        "initialize",
        "on_start",
        "on_stop",
        "on_tick",
        "on_order",
        "on_trade",
        "on_timer",
    )
    for key in sensitive_keys:
        if key in backtest_kwargs:
            checks.append((f"kwargs['{key}']", backtest_kwargs[key]))

    for label, obj in checks:
        try:
            pickle.dumps(obj)
        except Exception as e:
            raise TypeError(
                "run_grid_search with max_workers>1 requires pickle-serializable "
                f"arguments, but {label} failed: {e}. "
                "Tips: use fill_policy dict and avoid lambda/local callbacks, "
                "and ensure strategy class is defined in importable module."
            ) from e


def _validate_strategy_param_grid_keys(
    strategy: Type[Strategy], param_grid: Mapping[str, Sequence[Any]]
) -> None:
    """按 __param_model__ 校验网格键名/类型/约束（越界即报错）."""
    model = getattr(strategy, "__param_model__", None)
    if model is None:
        return
    field_names = set(model.model_fields)
    unknown = sorted(str(k) for k in param_grid if str(k) not in field_names)
    if unknown:
        raise TypeError(
            "Unknown strategy param(s) in param_grid: "
            f"{', '.join(unknown)}. Strategy={strategy.__module__}.{strategy.__name__}"
        )
    # 逐参数逐候选值校验（含类型/ge/le/choices），失败信息包含字段名
    for key, values in param_grid.items():
        for value in values:
            validate_strategy_params(strategy, {str(key): value})


def _save_result_to_db(
    db_path: str, strategy_name: str, result: OptimizationResult
) -> None:
    """Save a single result to SQLite."""
    try:
        import sqlite3

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Serialize
            params_json = json.dumps(result.params, sort_keys=True, cls=JSONEncoder)
            metrics_json = json.dumps(result.metrics, cls=JSONEncoder)

            cursor.execute(
                """
                INSERT OR IGNORE INTO optimization_results
                (strategy_name, params_json, metrics_json, duration, error)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    strategy_name,
                    params_json,
                    metrics_json,
                    result.duration,
                    result.error,
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning("Failed to save result to DB at %s: %s", db_path, e)


def run_grid_search(
    strategy: Type[Strategy],
    param_grid: Mapping[str, Sequence[Any]],
    data: Any = None,
    max_workers: Optional[int] = None,
    sort_by: Union[str, List[str]] = "sharpe_ratio",
    ascending: Union[bool, List[bool]] = False,
    return_df: bool = True,
    warmup_calc: Optional[Any] = None,
    constraint: Optional[Any] = None,
    result_filter: Optional[Any] = None,
    timeout: Optional[float] = None,
    max_tasks_per_child: Optional[int] = None,
    db_path: Optional[str] = None,
    forward_worker_logs: bool = False,
    **kwargs: Any,
) -> Union[pd.DataFrame, List[OptimizationResult]]:
    """
    运行参数优化 (Grid Search).

    :param strategy: 策略类
    :param param_grid: 参数网格，例如 {'period': [10, 20], 'factor': [0.5, 1.0]}
    :param data: 回测数据 (DataFrame, Dict[str, DataFrame], or List[Bar])
    :param max_workers: 并行进程数，默认 CPU 核心数
    :param sort_by: 结果排序指标 (默认: "sharpe_ratio")，支持单字段或多字段列表
    :param ascending: 排序方向 (默认: False, 即降序)，支持单值或多值列表
    :param return_df: 是否返回 DataFrame 格式 (默认: True)
    :param warmup_calc: 动态计算预热期的函数，接收 params 字典，返回 int (默认: None)
    :param constraint: 参数约束函数，接收 params 字典，返回 bool。True 表示保留，
                       False 表示过滤 (默认: None)
    :param result_filter: 结果筛选函数，接收 metrics 字典，返回 bool。True 表示保留，
                          False 表示过滤 (默认: None)
    :param timeout: 单次任务超时时间 (秒, 默认: None)。如果设置，
                    建议也设置 max_tasks_per_child=1 以清理超时线程。
    :param max_tasks_per_child: Worker 进程执行多少个任务后重启 (默认: None)。
                                设置 1 可以避免内存泄漏或超时线程残留。
    :param db_path: SQLite 数据库路径 (可选)。如果提供，将支持断点续传和增量保存。
    :param forward_worker_logs: 并行时是否将子进程策略日志回传主进程 (默认: False)
    :param kwargs: 传递给 run_backtest 的其他参数 (symbol, cash, etc.)
    :return: 优化结果 (DataFrame 或 List[OptimizationResult])
    """
    backtest_kwargs = _resolve_optimization_backtest_kwargs(data, dict(kwargs))
    backtest_kwargs.setdefault("strict_strategy_params", True)
    if (
        "execution_mode" in backtest_kwargs
        or "timer_execution_policy" in backtest_kwargs
    ):
        raise ValueError(
            "run_grid_search no longer accepts execution_mode/timer_execution_policy; "
            "please use fill_policy"
        )
    strict_strategy_params = bool(backtest_kwargs.get("strict_strategy_params", False))
    if strict_strategy_params:
        _validate_strategy_param_grid_keys(strategy, param_grid)

    # 1. 生成参数组合
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 1.5 应用约束过滤
    if constraint:
        original_count = len(param_combinations)
        param_combinations = [p for p in param_combinations if constraint(p)]
        filtered_count = len(param_combinations)
        if original_count != filtered_count:
            logger.info(
                "Constraint filtered %s combinations (%s -> %s)",
                original_count - filtered_count,
                original_count,
                filtered_count,
            )

    # 1.6 断点续传 (如果有 db_path)
    existing_results = []
    if db_path:
        try:
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                # 检查表是否存在
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS optimization_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT,
                        params_json TEXT UNIQUE,
                        metrics_json TEXT,
                        duration REAL,
                        error TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.commit()

                # 读取已有的结果
                cursor.execute(
                    "SELECT params_json, metrics_json, duration, error "
                    "FROM optimization_results WHERE strategy_name = ?",
                    (strategy.__name__,),
                )
                rows = cursor.fetchall()

                existing_params_set = set()
                for row in rows:
                    p_json, m_json, dur, err = row
                    try:
                        # 尝试解析 JSON
                        p = json.loads(p_json)
                        # 将 params 转为 tuple sorted items 以便比较
                        # (因为 list 不可哈希, dict 也不可哈希)
                        # 这里我们简单使用 json string 作为 key
                        # (假设 json 序列化是确定性的)
                        # 为了更健壮，我们应该重新序列化一遍 param_combinations
                        # 中的 param 来比较
                        existing_params_set.add(p_json)

                        m = json.loads(m_json)
                        existing_results.append(
                            OptimizationResult(
                                params=p, metrics=m, duration=dur, error=err
                            )
                        )
                    except Exception:
                        continue

                if existing_results:
                    logger.info(
                        "Found %s existing results in DB. Resuming...",
                        len(existing_results),
                    )

                    # 过滤已完成的任务
                    # 注意：需要确保 param_combinations 的 json 序列化格式与 DB 中一致
                    # 简单起见，我们对 param_combinations 中的每个 param 进行同样的
                    # json dumps
                    new_combinations = []
                    skipped_count = 0
                    for p in param_combinations:
                        # 使用 sort_keys=True 确保顺序一致
                        p_str = json.dumps(p, sort_keys=True, cls=JSONEncoder)
                        if p_str in existing_params_set:
                            skipped_count += 1
                        else:
                            new_combinations.append(p)

                    param_combinations = new_combinations
                    logger.info(
                        "Skipped %s completed tasks. Remaining: %s",
                        skipped_count,
                        len(param_combinations),
                    )

        except Exception as e:
            logger.warning("Failed to access SQLite DB at %s: %s", db_path, e)

    total_combinations = len(param_combinations)

    # 3. 并行执行 (如果有剩余任务)
    new_results = []
    if total_combinations > 0:
        logger.info(
            "Running optimization for %s parameter combinations...",
            total_combinations,
        )

        # 2. 准备任务
        tasks = []
        for params in param_combinations:
            tasks.append(
                {
                    "strategy_cls": strategy,
                    "params": params,
                    "backtest_kwargs": {"data": data, **backtest_kwargs},
                    "warmup_calc": warmup_calc,
                    "timeout": timeout,
                }
            )

        # 如果 max_workers 为 None，默认使用 os.cpu_count()
        if max_workers is None:
            max_workers = multiprocessing.cpu_count() or 1

        # 如果只有一个任务或 worker=1，直接运行
        # (除非设置了 timeout，需要线程支持，仍走单线程逻辑)
        if max_workers == 1 or total_combinations == 1:
            for task in tqdm(tasks, desc="Optimizing"):
                result = _run_single_backtest(task)
                new_results.append(result)
                # 单线程模式下也可以实时写入 DB
                if db_path:
                    _save_result_to_db(db_path, strategy.__name__, result)
        else:
            # 使用 multiprocessing.Pool
            # 如果设置了 timeout，且未指定 max_tasks_per_child，建议设为 1 以清理线程
            if timeout is not None and max_tasks_per_child is None:
                max_tasks_per_child = 1
            _assert_parallel_pickleable(strategy, backtest_kwargs, warmup_calc)
            listener: Optional[QueueListener] = None
            log_queue: Any = None
            pool_initializer: Optional[Any] = None
            pool_init_args: tuple[Any, ...] = ()
            worker_log_forwarding_active = False
            if forward_worker_logs:
                root_logger = get_logger()
                active_handlers = [
                    handler
                    for handler in root_logger.handlers
                    if not isinstance(handler, logging.NullHandler)
                ]
                if active_handlers:
                    log_queue = multiprocessing.Queue()
                    listener = QueueListener(
                        log_queue, *active_handlers, respect_handler_level=True
                    )
                    listener.start()
                    pool_initializer = _init_worker_logging
                    pool_init_args = (log_queue,)
                    worker_log_forwarding_active = True
                else:
                    logger.warning(
                        "forward_worker_logs=True but no active logger handler "
                        "found in main process."
                    )
            if not worker_log_forwarding_active and not forward_worker_logs:
                logger.warning(
                    "max_workers>1 uses subprocess workers. Strategy self.log() "
                    "output may not be visible in the main process console."
                )
            try:
                with multiprocessing.Pool(
                    processes=max_workers,
                    maxtasksperchild=max_tasks_per_child,
                    initializer=pool_initializer,
                    initargs=pool_init_args,
                ) as pool:
                    iterator = pool.imap(_run_single_backtest, tasks)
                    try:
                        for result in tqdm(
                            iterator, total=total_combinations, desc="Optimizing"
                        ):
                            new_results.append(result)
                            if db_path:
                                _save_result_to_db(db_path, strategy.__name__, result)
                    except Exception as e:
                        logger.error(
                            "Error during optimization (Worker Crash/OOM?): %s",
                            e,
                        )
                        pass
            finally:
                if listener is not None:
                    listener.stop()
                if log_queue is not None:
                    log_queue.close()
                    log_queue.join_thread()
    else:
        logger.info("All tasks completed. Returning existing results.")

    # 合并结果
    results = existing_results + new_results

    # 4. 结果筛选
    if result_filter:
        original_count = len(results)
        results = [r for r in results if result_filter(r.metrics)]
        filtered_count = len(results)
        if original_count != filtered_count:
            logger.info(
                "Result filter removed %s combinations (%s -> %s)",
                original_count - filtered_count,
                original_count,
                filtered_count,
            )

    # 5. 排序结果
    # 确保 sort_by 字段存在，否则给默认值
    if isinstance(sort_by, list):
        # 多字段排序
        # Python 的 sort 是稳定的，可以多次排序来实现多键排序
        # 必须反向遍历 sort_by 列表
        # 如果 ascending 是列表，则对应每个键；如果是单个值，则统一应用
        if isinstance(ascending, bool):
            asc_list = [ascending] * len(sort_by)
        else:
            asc_list = ascending
            if len(asc_list) != len(sort_by):
                raise ValueError("Length of ascending list must match sort_by list")

        for key, asc in zip(reversed(sort_by), reversed(asc_list)):
            results.sort(
                key=lambda x: x.metrics.get(key, -float("inf")), reverse=not asc
            )
    else:
        # 单字段排序
        results.sort(
            key=lambda x: x.metrics.get(sort_by, -float("inf")), reverse=not ascending
        )

    if return_df:
        data_list = []
        for r in results:
            row = r.params.copy()
            row.update(r.metrics)
            row["_duration"] = r.duration
            data_list.append(row)
        return pd.DataFrame(data_list)

    return results


def run_walk_forward(
    strategy: Type[Strategy],
    param_grid: Mapping[str, Sequence[Any]],
    data: OptimizationData,
    train_period: int,
    test_period: int,
    metric: Union[str, List[str]] = "sharpe_ratio",
    ascending: Union[bool, List[bool]] = False,
    initial_cash: float = 100_000.0,
    warmup_period: int = 0,
    warmup_calc: Optional[Any] = None,
    constraint: Optional[Any] = None,
    result_filter: Optional[Any] = None,
    compounding: bool = False,
    timeout: Optional[float] = None,
    max_tasks_per_child: Optional[int] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    执行 Walk-Forward Optimization (WFO).

    将数据切分为多个 "训练集+测试集" 片段，滚动优化参数并验证。

    :param strategy: 策略类
    :param param_grid: 参数网格
    :param data: 回测数据 (支持 DataFrame 或 Dict[str, DataFrame])
    :param train_period: 训练窗口长度 (Bar数量)
    :param test_period: 测试窗口长度 (Bar数量)
    :param metric: 优化目标指标 (默认: "sharpe_ratio")，支持多字段排序列表。
                   对应 run_grid_search 的 sort_by 参数。
    :param ascending: 排序方向 (默认: False, 即降序)，支持单值或多值列表。
                      对应 run_grid_search 的 ascending 参数。
    :param initial_cash: 初始资金 (默认: 100,000.0)
    :param warmup_period: 基础预热长度 (Bar数量)
    :param warmup_calc: 动态预热计算函数 (可选)
    :param constraint: 参数约束函数 (可选)
    :param result_filter: 结果筛选函数 (可选)
    :param compounding: 是否使用复利拼接结果 (True=复利, False=累加盈亏, 默认: False)
    :param timeout: 单次优化任务超时时间 (秒)
    :param max_tasks_per_child: Worker 重启频率
    :param kwargs: 透传给 run_grid_search 和 run_backtest 的其他参数
    :return: 包含拼接后资金曲线的 DataFrame
    """
    kwargs = _resolve_optimization_backtest_kwargs(data, kwargs)
    requested_symbols = _normalize_symbol_values(kwargs.get("symbols"))
    prepared_data = _prepare_optimization_data(data)
    prepared_data = _filter_optimization_data_by_symbols(
        prepared_data,
        requested_symbols,
    )
    timeline = _build_optimization_timeline(prepared_data)
    total_len = len(timeline)
    if total_len < train_period + test_period:
        raise ValueError(
            f"Data length ({total_len}) is too short for "
            f"train ({train_period}) + test ({test_period})."
        )

    logger.info(
        "Starting Walk-Forward Optimization: Train=%s, Test=%s, Total Bars=%s",
        train_period,
        test_period,
        total_len,
    )

    oos_results = []
    current_capital = initial_cash

    # 滚动窗口循环
    # Step size is test_period
    for i in range(0, total_len - train_period - test_period + 1, test_period):
        train_start_idx = i
        train_end_idx = i + train_period
        oos_start_idx = train_end_idx
        oos_end_idx = min(oos_start_idx + test_period, total_len)

        train_start_time = timeline[train_start_idx]
        train_end_exclusive = (
            timeline[train_end_idx] if train_end_idx < total_len else None
        )
        train_end_time = timeline[train_end_idx - 1]
        oos_start_time = timeline[oos_start_idx]
        oos_end_exclusive = timeline[oos_end_idx] if oos_end_idx < total_len else None
        oos_end_time = timeline[oos_end_idx - 1]
        train_data = _slice_optimization_data(
            prepared_data,
            train_start_time,
            train_end_exclusive,
        )

        logger.info(
            "=== Window %s: Train [%s - %s] ===",
            i // test_period + 1,
            train_start_time,
            train_end_time,
        )

        # 2. 样本内优化 (Optimization)
        opt_results = run_grid_search(
            strategy=strategy,
            param_grid=param_grid,
            data=train_data,
            sort_by=metric,
            ascending=ascending,
            return_df=True,
            warmup_calc=warmup_calc,
            constraint=constraint,
            result_filter=result_filter,
            initial_cash=initial_cash,
            timeout=timeout,
            max_tasks_per_child=max_tasks_per_child,
            **kwargs,
        )

        if isinstance(opt_results, list) or opt_results.empty:
            logger.warning(
                "Optimization failed or returned no results. Skipping window."
            )
            continue

        # 获取最佳参数
        best_row = opt_results.iloc[0]
        best_params = {k: best_row[k] for k in param_grid.keys()}

        # 显示排序指标的值
        metric_str = ""
        if isinstance(metric, list):
            metric_str = ", ".join([f"{m}={best_row.get(m, 0):.4f}" for m in metric])
        else:
            metric_str = f"{metric}={best_row.get(metric, 0):.4f}"

        logger.info("Best Params: %s (%s)", best_params, metric_str)

        # 计算实际需要的预热期
        current_warmup = warmup_period
        if warmup_calc:
            try:
                current_warmup = max(current_warmup, warmup_calc(best_params))
            except Exception:
                pass

        slice_start_idx = max(0, oos_start_idx - current_warmup)
        slice_start_time = timeline[slice_start_idx]
        test_data_with_warmup = _slice_optimization_data(
            prepared_data,
            slice_start_time,
            oos_end_exclusive,
        )

        # 4. 样本外验证 (Backtest)
        # 使用最佳参数运行回测
        # 注意：这里我们使用一个新的 initial_cash 进行回测，后续再拼接
        backtest_kwargs = {
            k: v for k, v in kwargs.items() if k not in _GRID_SEARCH_ONLY_KWARGS
        }
        backtest_kwargs.update(best_params)
        backtest_kwargs["initial_cash"] = initial_cash
        backtest_kwargs["warmup_period"] = current_warmup

        logger.info(
            "Test [%s - %s] (Warmup: %s)",
            oos_start_time,
            oos_end_time,
            current_warmup,
        )

        bt_result = run_backtest(
            strategy=strategy, data=test_data_with_warmup, **backtest_kwargs
        )

        # 5. 提取并拼接结果
        equity_curve = bt_result.equity_curve

        if equity_curve.empty:
            logger.warning("Empty equity curve in OOS.")
            continue

        # 处理时区不匹配问题
        idx = equity_curve.index
        # Cast to DatetimeIndex to access .tz
        dt_idx = (
            cast(pd.DatetimeIndex, idx) if isinstance(idx, pd.DatetimeIndex) else None
        )

        if (
            dt_idx is not None
            and dt_idx.tz is not None
            and oos_start_time.tzinfo is None
        ):
            # 如果结果有时区但原始数据没有，假设原始数据是本地时间并本地化
            # 或者将结果转换为 naive (不太推荐，可能丢失信息)
            # 这里尝试将 oos_start_time 本地化到结果的时区
            try:
                oos_start_time = oos_start_time.tz_localize(dt_idx.tz)
            except Exception:
                # 如果失败 (例如可能是 UTC)，尝试转为 naive 进行比较
                equity_curve = equity_curve.tz_localize(None)
        elif (
            dt_idx is None or dt_idx.tz is None
        ) and oos_start_time.tzinfo is not None:
            equity_curve = equity_curve.tz_localize(oos_start_time.tzinfo)

        # 过滤时间段
        valid_equity = equity_curve[equity_curve.index >= oos_start_time]
        if valid_equity.empty:
            logger.warning("No equity data in valid OOS period.")
            continue

        # 拼接逻辑
        if compounding:
            # 复利模式：计算收益率并累乘
            # 收益率 = (当前净值 - 上一刻净值) / 上一刻净值
            # 但这里我们是基于一段独立的 equity curve
            # 计算该段的收益率序列
            returns = valid_equity.pct_change().fillna(0)
            # 第一个点的收益率需要相对于"入场资金"计算
            # 入场资金 = initial_cash (因为回测是重置的)
            # valid_equity.iloc[0] 相对于 initial_cash 的收益
            first_ret = (valid_equity.iloc[0] - initial_cash) / initial_cash
            returns.iloc[0] = first_ret

            # 将收益率记录下来，最后统一计算？
            # 或者直接计算调整后的净值
            # 累积净值 = current_capital * (1 + returns).cumprod()
            # 这种方式每一段的起点是上一段的终点

            # 简单做法：将收益率序列存起来，最后统一 cumprod
            # 但我们需要返回 DataFrame，最好包含 params
            segment_df = pd.DataFrame({"return": returns})
            segment_df["equity"] = (
                current_capital * (1 + segment_df["return"]).cumprod()
            )
            current_capital = segment_df["equity"].iloc[-1]

        else:
            # 累加模式 (默认)：计算 PnL 并累加
            # PnL = 当前净值 - 初始资金
            pnl = valid_equity - initial_cash

            # 调整后的净值 = 上一段结束资金 + 当前段PnL
            adjusted_equity = current_capital + pnl
            segment_df = pd.DataFrame({"equity": adjusted_equity})
            current_capital = adjusted_equity.iloc[-1]

        # 添加元数据
        segment_df["train_start"] = train_start_time
        segment_df["train_end"] = train_end_time
        for k, v in best_params.items():
            segment_df[k] = v

        oos_results.append(segment_df)

    if not oos_results:
        logger.warning("Walk-Forward Optimization produced no results.")
        return pd.DataFrame()

    # 6. 合并所有片段
    final_df = pd.concat(oos_results)

    # 填补空缺 (如果时间不连续) ? WFO 通常是连续的
    return final_df
