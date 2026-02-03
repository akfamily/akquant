"""
参数优化模块 (Parameter Optimization).

提供类似 Backtrader optstrategy 的网格搜索功能.
"""

import itertools
import multiprocessing
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union, cast

import pandas as pd
from tqdm import tqdm  # type: ignore

from .backtest import run_backtest
from .strategy import Strategy


@dataclass
class OptimizationResult:
    """
    单个优化结果.

    :param params: 参数组合
    :param metrics: 性能指标字典
    :param duration: 回测耗时 (秒)
    """

    params: Dict[str, Any]
    metrics: Dict[str, Any]
    duration: float = 0.0

    def __repr__(self) -> str:
        """Return string representation."""
        return f"OptimizationResult(params={self.params}, metrics={self.metrics})"


def _run_single_backtest(args: Dict[str, Any]) -> OptimizationResult:
    """
    运行单个回测任务 (Internal).

    args 包含:
    - strategy_cls: 策略类
    - params: 当前参数组合
    - backtest_kwargs: run_backtest 的其他参数 (data, cash, etc.)

    :param args: 任务参数字典
    :return: 优化结果
    """
    strategy_cls = args["strategy_cls"]
    params = args["params"]
    backtest_kwargs = args["backtest_kwargs"]

    # 将参数合并到 kwargs 中传给 strategy
    # 注意：run_backtest 会将 kwargs 传给 strategy(**kwargs)
    # 我们需要避免修改原始 backtest_kwargs
    kwargs = backtest_kwargs.copy()
    kwargs.update(params)

    start_time = time.time()

    # 运行回测
    # 注意：show_progress 在并行时最好关掉
    kwargs["show_progress"] = False

    try:
        result = run_backtest(strategy=strategy_cls, **kwargs)
        metrics_df = result.metrics_df
        # 提取第一行作为字典
        metrics = cast(Dict[str, Any], metrics_df.iloc[0].to_dict())
    except Exception as e:
        metrics = {"error": str(e)}
        # Set default bad metrics
        metrics["sharpe_ratio"] = -999.0
        metrics["total_return"] = -999.0

    duration = time.time() - start_time

    return OptimizationResult(params=params, metrics=metrics, duration=duration)


def run_optimization(
    strategy: Type[Strategy],
    param_grid: Dict[str, List[Any]],
    data: Any = None,
    max_workers: Optional[int] = None,
    sort_by: str = "sharpe_ratio",
    ascending: bool = False,
    return_df: bool = True,
    **kwargs: Any,
) -> Union[pd.DataFrame, List[OptimizationResult]]:
    """
    运行参数优化 (Grid Search).

    :param strategy: 策略类
    :param param_grid: 参数网格，例如 {'period': [10, 20], 'factor': [0.5, 1.0]}
    :param data: 回测数据 (DataFrame, Dict[str, DataFrame], or List[Bar])
    :param max_workers: 并行进程数，默认 CPU 核心数
    :param sort_by: 结果排序指标 (默认: "sharpe_ratio")
    :param ascending: 排序方向 (默认: False, 即降序)
    :param return_df: 是否返回 DataFrame 格式 (默认: True)
    :param kwargs: 传递给 run_backtest 的其他参数 (symbol, cash, etc.)
    :return: 优化结果 (DataFrame 或 List[OptimizationResult])
    """
    # 1. 生成参数组合
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total_combinations = len(param_combinations)
    print(f"Running optimization for {total_combinations} parameter combinations...")

    # 2. 准备任务
    tasks = []
    for params in param_combinations:
        tasks.append(
            {
                "strategy_cls": strategy,
                "params": params,
                "backtest_kwargs": {"data": data, **kwargs},
            }
        )

    # 3. 并行执行
    results = []

    # 如果 max_workers 为 None，默认使用 os.cpu_count()
    if max_workers is None:
        max_workers = multiprocessing.cpu_count() or 1

    # 如果只有一个任务或 worker=1，直接运行
    if max_workers == 1 or total_combinations == 1:
        for task in tqdm(tasks, desc="Optimizing"):
            results.append(_run_single_backtest(task))
    else:
        # 使用 multiprocessing.Pool
        # 注意：这里需要确保 _run_single_backtest 是可 pickle 的
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_run_single_backtest, tasks),
                    total=total_combinations,
                    desc="Optimizing",
                )
            )

    # 4. 排序结果
    # 确保 sort_by 字段存在，否则给默认值
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
