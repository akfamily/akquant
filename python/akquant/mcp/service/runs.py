"""run_backtest 调度(子进程 + 超时) 与 get_run_details 分页读取.

每次回测一个新的 ``python -m akquant.mcp.worker`` 子进程: 子进程隔离了死循环、
Rust panic、``sys.exit`` 与状态残留。
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd

from . import ServiceError
from .data import dataset_file
from .paths import (
    PathArg,
    ServiceConfig,
    prune_runs,
    resolve_in_root,
    write_json_atomic,
)
from .records import frame_to_records

ALLOWED_CONFIG = frozenset(
    {
        "initial_cash",
        "commission_rate",
        "stamp_tax_rate",
        "min_commission",
        "slippage",
        "t_plus_one",
        "lot_size",
        "timezone",
        "symbols",
        "start_time",
        "end_time",
        "broker_profile",
        "warmup_period",
    }
)
SECTIONS = ("trades", "orders", "equity", "logs")
TIMEOUT_MIN, TIMEOUT_MAX = 1, 1800
KILL_GRACE_S = 5
LOG_TAIL = 30
MAX_LINE_CHARS = 500
_RUN_LOCK = threading.Lock()
MAX_LIMIT = 1000
_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def _read_lines(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _clip(text: str) -> str:
    if len(text) > MAX_LINE_CHARS:
        return text[:MAX_LINE_CHARS] + "…"
    return text


def _crash_error(returncode: Optional[int], last_line: str = "") -> Dict[str, Any]:
    suffix = f": {last_line}" if last_line else ""
    return {
        "phase": "engine",
        "type": "WorkerCrashed",
        "message": f"回测子进程异常退出(exit code {returncode}), 未生成结果{suffix}",
        "user_frames": [],
        "innermost_framework_frame": None,
        "hint": None,
    }


def _spawn(cfg: ServiceConfig, run_dir: Path, timeout_s: int) -> Tuple[bool, int]:
    """启动 worker 并等待; 返回 (是否超时, 退出码)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    cmd = [sys.executable, "-m", "akquant.mcp.worker", str(run_dir)]
    with open(run_dir / "output.log", "wb") as log:
        # stdin 必须是 DEVNULL: 本进程的 stdin 是 MCP 协议通道
        proc = subprocess.Popen(
            cmd,
            cwd=str(run_dir),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            proc.wait(timeout=timeout_s)
            return False, proc.returncode
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return True, proc.returncode


def run_backtest(
    cfg: ServiceConfig,
    path: PathArg,
    dataset_id: str,
    config: Optional[Dict[str, Any]] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    strategy_attr: Optional[str] = None,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    """在子进程里跑一次回测, 返回结构化摘要(同时写入 summary.json)."""
    strategy = resolve_in_root(cfg, path)
    if not strategy.is_file():
        raise ServiceError(f"策略文件不存在: {path}")
    data_path = dataset_file(cfg, dataset_id)
    options = dict(config or {})
    unknown = sorted(set(options) - ALLOWED_CONFIG)
    if unknown:
        allowed = sorted(ALLOWED_CONFIG)
        raise ServiceError(f"config 不支持字段 {unknown}; 允许: {allowed}")
    if not TIMEOUT_MIN <= timeout_s <= TIMEOUT_MAX:
        raise ServiceError(f"timeout_s 取值范围 {TIMEOUT_MIN}~{TIMEOUT_MAX}")
    run_id = _new_run_id()
    run_dir = cfg.runs_dir / run_id
    with _RUN_LOCK:
        prune_runs(cfg, cfg.max_runs - 1)
        run_dir.mkdir(parents=True)
    request = {
        "root": str(cfg.root),
        "strategy_path": str(strategy),
        "dataset_path": str(data_path),
        "config": options,
        "strategy_params": strategy_params,
        "strategy_attr": strategy_attr,
    }
    write_json_atomic(run_dir / "request.json", request)
    started = time.monotonic()
    timed_out, returncode = _spawn(cfg, run_dir, timeout_s)
    elapsed = round(time.monotonic() - started, 3)
    summary_path = run_dir / "summary.json"
    lines = [t for t in _read_lines(run_dir / "output.log") if t.strip()]
    body: Dict[str, Any]
    if timed_out:
        body = {
            "status": "timeout",
            "message": f"回测超过 {timeout_s} 秒未结束, 子进程已终止",
        }
    elif summary_path.is_file():
        raw = summary_path.read_text(encoding="utf-8")
        body = cast(Dict[str, Any], json.loads(raw))
    else:
        last = lines[-1].strip() if lines else ""
        body = {
            "status": "error",
            "error": _crash_error(returncode, last),
            "warnings": [],
        }
    summary = {
        "run_id": run_id,
        **body,
        "elapsed_s": elapsed,
        "log_tail": [_clip(t) for t in lines[-LOG_TAIL:]],
    }
    write_json_atomic(summary_path, summary)
    return summary


def get_run_details(
    cfg: ServiceConfig,
    run_id: str,
    section: str,
    offset: int = 0,
    limit: int = 100,
) -> Dict[str, Any]:
    """分页读取一次回测的成交 / 委托 / 日频权益 / 日志."""
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ServiceError(f"run_id 格式不对: {run_id!r}")
    if section not in SECTIONS:
        raise ServiceError(f"section 只能是 {', '.join(SECTIONS)} 之一")
    if offset < 0:
        raise ServiceError("offset 不能为负")
    if not 1 <= limit <= MAX_LIMIT:
        raise ServiceError(f"limit 取值范围 1~{MAX_LIMIT}")
    run_dir = cfg.runs_dir / run_id
    if not run_dir.is_dir():
        raise ServiceError(f"找不到 run_id: {run_id}(可能已被清理)")
    rows: List[Dict[str, Any]]
    if section == "logs":
        lines = _read_lines(run_dir / "output.log")
        total = len(lines)
        window = lines[offset : offset + limit]
        rows = [{"line": offset + i + 1, "text": t} for i, t in enumerate(window)]
    else:
        file = run_dir / f"{section}.parquet"
        if not file.is_file():
            raise ServiceError(
                f"这次回测没有 {section} 明细(回测失败或保存失败, "
                "见 run_backtest 返回的 warnings)"
            )
        df = pd.read_parquet(file)
        total = len(df)
        rows = frame_to_records(df.iloc[offset : offset + limit])
    return {
        "run_id": run_id,
        "section": section,
        "total": total,
        "offset": offset,
        "rows": rows,
    }
