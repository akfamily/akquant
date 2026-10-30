"""回测子进程入口: ``python -m akquant.mcp.worker <run_dir>``.

读 ``<run_dir>/request.json``, 跑回测, 把明细写成 Parquet, 最后原子写
``summary.json``(不含 run_id / elapsed_s / log_tail, 由主进程补)。
stdout/stderr 已被主进程重定向到 ``output.log``。不导入 mcp SDK。
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

import pandas as pd

from .service.errors import describe_exception
from .service.paths import write_json_atomic
from .service.records import frame_to_records, to_jsonable

#: 直接沿用 result.metrics_df 的索引名, 不另起名字
METRIC_KEYS = (
    "total_return_pct",
    "annualized_return",
    "max_drawdown_pct",
    "sharpe_ratio",
    "win_rate",
    "closed_trade_count",
    "end_market_value",
    "total_commission",
)
FIRST_TRADES = 10
MAX_WARNINGS = 20
MAX_MESSAGE_CHARS = 500


def _metrics(result: Any) -> Dict[str, Any]:
    values = result.metrics_df["value"]
    return {
        k: to_jsonable(values[k]) if k in values.index else None for k in METRIC_KEYS
    }


def _save_table(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _save_artifacts(result: Any, run_dir: Path) -> List[str]:
    problems: List[str] = []
    builders = {
        "trades": lambda: result.trades_df,
        "orders": lambda: result.orders_df,
        "equity": lambda: (
            result.equity_curve_daily.rename("equity").to_frame().reset_index()
        ),
    }
    for name, build in builders.items():
        try:
            _save_table(build(), run_dir / f"{name}.parquet")
        except Exception as exc:  # 明细保存失败不影响 summary
            problems.append(f"{name} 明细保存失败: {exc}")
    return problems


class _WarningCollector:
    """替换 showwarning: 照常写 stderr(进 output.log), 并去重记录前 N 条."""

    def __init__(self) -> None:
        self.messages: Dict[str, None] = {}

    def __call__(
        self,
        message: Any,
        category: Any,
        filename: str,
        lineno: int,
        file: Any = None,
        line: Optional[str] = None,
    ) -> None:
        text = warnings.formatwarning(message, category, filename, lineno, line)
        sys.stderr.write(text)
        if len(self.messages) < MAX_WARNINGS:
            key = f"{category.__name__}: {message}"
            if len(key) > MAX_MESSAGE_CHARS:
                key = key[:MAX_MESSAGE_CHARS] + "…"
            self.messages[key] = None


def _prepare_environment(request: Dict[str, Any]) -> None:
    """切到项目根解析相对路径; 策略所在目录只追加到 sys.path 末尾."""
    root = request.get("root")
    if root:
        os.chdir(root)
    folder = str(Path(request["strategy_path"]).parent)
    if folder not in sys.path:
        sys.path.append(folder)


def _build_summary(result: Any, run_dir: Path, messages: List[str]) -> Dict[str, Any]:
    try:
        problems = _save_artifacts(result, run_dir)
        return {
            "status": "ok",
            "metrics": _metrics(result),
            "first_trades": frame_to_records(result.trades_df.head(FIRST_TRADES)),
            "warnings": messages + problems,
        }
    except Exception as exc:
        traceback.print_exc()
        error = describe_exception(exc, "")
        error["phase"] = "engine"
        return {"status": "error", "error": error, "warnings": messages}


def run(run_dir: Path) -> Dict[str, Any]:
    """执行一次回测并写 summary.json, 返回写入的内容."""
    import akquant as aq

    raw = (run_dir / "request.json").read_text(encoding="utf-8")
    request = cast(Dict[str, Any], json.loads(raw))
    _prepare_environment(request)
    data = pd.read_parquet(request["dataset_path"])
    kwargs: Dict[str, Any] = dict(request.get("config") or {})
    if request.get("strategy_params"):
        kwargs["strategy_params"] = request["strategy_params"]
    if request.get("strategy_attr"):
        kwargs["strategy_loader_options"] = {"strategy_attr": request["strategy_attr"]}
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    collector = _WarningCollector()
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        warnings.showwarning = collector  # type: ignore[assignment]
        try:
            result = aq.run_backtest(
                data=data,
                strategy_source=request["strategy_path"],
                show_progress=False,
                **kwargs,
            )
        except (Exception, SystemExit) as exc:
            traceback.print_exc()
            error = describe_exception(exc, request["strategy_path"])
    messages = list(collector.messages)
    summary: Dict[str, Any]
    if error is not None or result is None:
        summary = {"status": "error", "error": error, "warnings": messages}
    else:
        summary = _build_summary(result, run_dir, messages)
    write_json_atomic(run_dir / "summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m akquant.mcp.worker <run_dir>", file=sys.stderr)
        return 2
    run(Path(args[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
