"""根目录、工作目录、路径边界、原子写与 run 目录清理."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from . import ServiceError

#: 覆盖工作目录的环境变量
HOME_ENV = "AKQUANT_MCP_HOME"
#: AKTools(AKShare 的 MCP 服务)存放 save_data 文件的目录, 与 AKTools 读同一个
#: 环境变量; 未设置时两边的默认值都是 ~/.aktools/data
AKTOOLS_DATA_ENV = "AKTOOLS_DATA_DIR"
DEFAULT_MAX_RUNS = 50

PathArg = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class ServiceConfig:
    """一次 MCP 服务会话的目录配置."""

    root: Path
    home: Path
    max_runs: int = DEFAULT_MAX_RUNS
    #: 项目根目录之外唯一允许读数据文件的目录; None 表示不放行
    aktools_dir: Optional[Path] = None

    @property
    def datasets_dir(self) -> Path:
        """数据缓存目录."""
        return self.home / "datasets"

    @property
    def runs_dir(self) -> Path:
        """回测结果目录."""
        return self.home / "runs"


def make_config(
    root: Optional[PathArg] = None,
    home: Optional[PathArg] = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    aktools_dir: Optional[PathArg] = None,
) -> ServiceConfig:
    """构造配置.

    root 默认取当前目录; home 依次取参数、环境变量 ``AKQUANT_MCP_HOME``、
    ``<root>/.akquant-mcp``; aktools_dir 依次取参数、环境变量
    ``AKTOOLS_DATA_DIR``、``~/.aktools/data``, 目录不存在也可以(AKTools 第一次
    保存数据时才创建)。
    """
    if max_runs < 1:
        raise ServiceError("max_runs 必须 >= 1")
    root_path = Path(root if root is not None else os.getcwd()).resolve()
    if not root_path.is_dir():
        raise ServiceError(f"项目根目录不存在: {root_path}")
    home_raw = home if home is not None else os.environ.get(HOME_ENV)
    if home_raw:
        home_path = Path(home_raw).resolve()
    else:
        home_path = root_path / ".akquant-mcp"
    aktools_raw = (
        aktools_dir
        if aktools_dir is not None
        else os.environ.get(AKTOOLS_DATA_ENV) or Path.home() / ".aktools" / "data"
    )
    return ServiceConfig(
        root=root_path,
        home=home_path,
        max_runs=max_runs,
        aktools_dir=Path(aktools_raw).expanduser().resolve(),
    )


def resolve_in_root(cfg: ServiceConfig, path: PathArg) -> Path:
    """把路径解析成绝对路径, 并要求它位于项目根目录内.

    相对路径按 root 解析; 符号链接先解开再判断, 指向根目录外同样拒绝。
    这是防误读的边界, 不是安全沙箱。
    """
    raw = Path(path)
    candidate = raw if raw.is_absolute() else cfg.root / raw
    resolved = candidate.resolve()
    if not resolved.is_relative_to(cfg.root):
        raise ServiceError(f"路径不在项目根目录 {cfg.root} 内: {path}")
    return resolved


def resolve_data_file(cfg: ServiceConfig, path: PathArg) -> Tuple[Path, bool]:
    """解析 prepare_data 的数据文件路径, 返回 (绝对路径, 是否在 AKTools 数据目录内).

    除项目根目录外, 还允许 AKTools 的数据目录: AKTools 的 save_data 把文件写在
    那里, 路径不由 AI 指定。只用于数据文件, 策略等其余路径仍走
    :func:`resolve_in_root`。
    """
    raw = Path(path)
    resolved = (raw if raw.is_absolute() else cfg.root / raw).resolve()
    if resolved.is_relative_to(cfg.root):
        return resolved, False
    if cfg.aktools_dir is not None and resolved.is_relative_to(cfg.aktools_dir):
        return resolved, True
    allowed = f"项目根目录 {cfg.root}"
    if cfg.aktools_dir is not None:
        allowed += f" 或 AKTools 数据目录 {cfg.aktools_dir}"
    raise ServiceError(f"数据文件必须位于{allowed} 内: {path}")


def write_json_atomic(path: Path, payload: Any) -> None:
    """先写临时文件再 rename, 读方不会看到写了一半的 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def prune_runs(cfg: ServiceConfig, keep: int) -> None:
    """删除最旧的 run, 只保留最近 ``keep`` 个.

    按目录 mtime 排序、目录名兜底: run_id 只精确到秒, 同一秒内的两个 run
    按名字(随机后缀)排序不代表先后。
    """
    if not cfg.runs_dir.is_dir():
        return
    stamped = []
    for p in cfg.runs_dir.iterdir():
        try:
            if p.is_dir():
                stamped.append((p.stat().st_mtime_ns, p.name, p))
        except FileNotFoundError:  # 并发清理时目录刚好消失
            continue
    runs = [p for _, _, p in sorted(stamped, key=lambda t: (t[0], t[1]))]
    for old in runs[: max(len(runs) - keep, 0)]:
        shutil.rmtree(old, ignore_errors=True)
