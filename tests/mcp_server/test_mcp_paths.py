"""akquant.mcp.service.paths: 配置、路径边界、原子写与 run 清理."""

import json
import os
from pathlib import Path

import pytest
from akquant.mcp.service import ServiceError
from akquant.mcp.service.paths import (
    HOME_ENV,
    ServiceConfig,
    make_config,
    prune_runs,
    resolve_in_root,
    write_json_atomic,
)


def test_default_home_under_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不传 home 且无环境变量时, 工作目录是 <root>/.akquant-mcp."""
    monkeypatch.delenv(HOME_ENV, raising=False)
    cfg = make_config(root=tmp_path)
    assert cfg.root == tmp_path.resolve()
    assert cfg.home == tmp_path.resolve() / ".akquant-mcp"
    assert cfg.runs_dir == cfg.home / "runs"
    assert cfg.datasets_dir == cfg.home / "datasets"
    assert cfg.max_runs == 50


def test_env_overrides_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """环境变量 AKQUANT_MCP_HOME 覆盖默认工作目录."""
    other = tmp_path / "elsewhere"
    monkeypatch.setenv(HOME_ENV, str(other))
    assert make_config(root=tmp_path).home == other.resolve()


def test_invalid_root_and_max_runs(tmp_path: Path) -> None:
    """根目录不存在、max_runs < 1 都报 ServiceError."""
    with pytest.raises(ServiceError):
        make_config(root=tmp_path / "missing")
    with pytest.raises(ServiceError):
        make_config(root=tmp_path, max_runs=0)


def test_resolve_relative_inside_root(tmp_path: Path) -> None:
    """相对路径按 root 解析."""
    cfg = make_config(root=tmp_path, home=tmp_path / ".h")
    (tmp_path / "s.py").write_text("", encoding="utf-8")
    assert resolve_in_root(cfg, "s.py") == (tmp_path / "s.py").resolve()


def test_resolve_rejects_escape(tmp_path: Path) -> None:
    """../ 与根目录外的绝对路径都被拒绝."""
    root = tmp_path / "proj"
    root.mkdir()
    cfg = make_config(root=root, home=root / ".h")
    with pytest.raises(ServiceError, match="项目根目录"):
        resolve_in_root(cfg, "../x.py")
    with pytest.raises(ServiceError):
        resolve_in_root(cfg, tmp_path / "x.py")


def test_resolve_rejects_symlink_out_of_root(tmp_path: Path) -> None:
    """根目录内指向根目录外的符号链接被拒绝."""
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")
    link = root / "link.py"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("当前系统不允许创建符号链接")
    cfg = make_config(root=root, home=root / ".h")
    with pytest.raises(ServiceError):
        resolve_in_root(cfg, "link.py")


def test_prune_runs_keeps_newest(tmp_path: Path) -> None:
    """按 mtime 删除最旧的, 只留 keep 个; 名字顺序与先后无关."""
    cfg = make_config(root=tmp_path, home=tmp_path / ".h")
    names = [
        "20260101-000000-cccccc",
        "20260101-000000-aaaaaa",
        "20260101-000000-bbbbbb",
    ]
    for i, name in enumerate(names):
        run_dir = cfg.runs_dir / name
        run_dir.mkdir(parents=True)
        stamp = 10**18 + i * 10**9
        os.utime(run_dir, ns=(stamp, stamp))
    prune_runs(cfg, keep=2)
    assert sorted(p.name for p in cfg.runs_dir.iterdir()) == sorted(names[1:])
    prune_runs(cfg, keep=0)
    assert list(cfg.runs_dir.iterdir()) == []


def test_write_json_atomic_roundtrip(tmp_path: Path) -> None:
    """写入后可读回, 且不留临时文件."""
    target = tmp_path / "a" / "b.json"
    write_json_atomic(target, {"中文": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"中文": 1}
    assert not target.with_name("b.json.tmp").exists()


def test_prune_tolerates_vanished_dir(
    cfg: ServiceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """排序时某个 run 目录刚好消失(stat 抛 FileNotFoundError): 跳过它."""
    for name in ("a", "b", "c"):
        (cfg.runs_dir / name).mkdir(parents=True)
    real_stat = Path.stat

    def fake_stat(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == "b":
            raise FileNotFoundError(str(self))
        return real_stat(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", fake_stat)
    prune_runs(cfg, 1)
    monkeypatch.undo()
    assert len([p for p in cfg.runs_dir.iterdir()]) <= 2
