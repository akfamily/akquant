"""python_plain 加载器: 源文件内容变化后应重新加载, 不复用旧模块."""

import os
import pickle
import sys
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from akquant import run_backtest
from akquant.strategy_loader import resolve_strategy_input


def _src(version: str) -> str:
    return "\n".join(
        [
            "from akquant.strategy import Strategy",
            "",
            "class ReloadProbe(Strategy):",
            f"    VERSION = {version!r}",
            "",
            "    def on_bar(self, bar):",
            "        pass",
        ]
    )


def _rewrite_same_stat(path: Path, text: str) -> None:
    """改写文件内容但保持 mtime(同秒、等长改动, 或编辑器保留 mtime 的场景)."""
    st = path.stat()
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def _load(path: Path) -> Any:
    return resolve_strategy_input(
        strategy_source=str(path), strategy_loader="python_plain"
    )


def test_changed_content_reloads(tmp_path: Path) -> None:
    """改写文件后再加载应拿到新版本."""
    path = tmp_path / "reload_a.py"
    path.write_text(_src("A"), encoding="utf-8")
    first = _load(path)
    assert first.VERSION == "A"
    _rewrite_same_stat(path, _src("B"))
    second = _load(path)
    assert second.VERSION == "B"
    assert second.__module__ == first.__module__


def test_unchanged_content_reuses_same_class(tmp_path: Path) -> None:
    """内容不变时返回同一个类对象."""
    path = tmp_path / "reload_b.py"
    path.write_text(_src("A"), encoding="utf-8")
    assert _load(path) is _load(path)


def test_reloaded_class_pickles_to_new_class(tmp_path: Path) -> None:
    """重新加载后的类可 pickle, 且 loads 得到的是新类."""
    path = tmp_path / "reload_c.py"
    path.write_text(_src("A"), encoding="utf-8")
    _load(path)
    _rewrite_same_stat(path, _src("B"))
    new_cls = _load(path)
    restored = pickle.loads(pickle.dumps(new_cls))
    assert restored is new_cls
    assert restored.VERSION == "B"


def test_reload_failure_keeps_previous_module(tmp_path: Path) -> None:
    """重新加载时 exec 失败: 抛原异常, sys.modules 仍是上一版模块."""
    path = tmp_path / "reload_d.py"
    path.write_text(_src("B"), encoding="utf-8")
    cls = _load(path)
    module = sys.modules[cls.__module__]
    path.write_text("def broken(:\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        _load(path)
    assert sys.modules[cls.__module__] is module
    assert module.ReloadProbe is cls


def test_first_load_failure_leaves_no_module(tmp_path: Path) -> None:
    """首次加载失败不留下空壳模块."""
    from akquant.strategy_loader import _encode_module_name

    path = tmp_path / "reload_e.py"
    path.write_text("def broken(:\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        _load(path)
    assert _encode_module_name(str(path)) not in sys.modules


def test_cached_module_without_digest_is_reloaded(tmp_path: Path) -> None:
    """缓存模块没有摘要属性(非本加载器登记)时视为需要重新加载."""
    from akquant.strategy_loader import _DIGEST_ATTR

    path = tmp_path / "reload_f.py"
    path.write_text(_src("A"), encoding="utf-8")
    first = _load(path)
    delattr(sys.modules[first.__module__], _DIGEST_ATTR)
    assert _load(path) is not first


def test_run_backtest_picks_up_edited_strategy(tmp_path: Path) -> None:
    """同进程两次 run_backtest(strategy_source=...) 之间改文件, 第二次用新代码."""
    path = tmp_path / "reload_g.py"
    body = "\n".join(
        [
            "from akquant.strategy import Strategy",
            "",
            "class Marker(Strategy):",
            "    def on_bar(self, bar):",
            "        self.marker = {tag!r}",
        ]
    )
    idx = pd.date_range("2023-01-02", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.0,
            "volume": 1000.0,
            "symbol": "RELOAD",
        },
        index=idx,
    )
    df["date"] = idx
    marks = []
    for tag in ("A", "B"):
        if path.exists():
            _rewrite_same_stat(path, body.format(tag=tag))
        else:
            path.write_text(body.format(tag=tag), encoding="utf-8")
        result = run_backtest(
            data=df,
            strategy_source=str(path),
            strategy_loader="python_plain",
            symbols="RELOAD",
            show_progress=False,
        )
        marks.append(getattr(result.strategy, "marker", None))
    assert marks == ["A", "B"]


def _grid_df() -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.0,
            "volume": 1000.0,
            "symbol": "GRID",
        },
        index=idx,
    )
    df["date"] = idx
    return df


def test_grid_search_workers_run_reloaded_version(tmp_path: Path) -> None:
    """Spawn 的 worker 执行的是改写后的 B, 而非 A 遗留的 .pyc.

    A 先跑一轮网格搜索, worker 会写下 .pyc; 再用保持 mtime 的等长改写换成 B,
    B 在 on_start 里往标记文件追加版本号。worker 若读旧字节码, 标记会是 A。
    """
    from akquant.optimize import run_grid_search

    marker = tmp_path / "marker.txt"
    path = tmp_path / "reload_h.py"
    body = "\n".join(
        [
            "from akquant import IntParam",
            "from akquant.strategy import Strategy",
            "",
            "class GridProbe(Strategy):",
            "    period = IntParam(1)",
            "",
            "    def on_start(self):",
            "        with open({marker!r}, 'a') as f:",
            "            f.write({tag!r})",
            "",
            "    def on_bar(self, bar):",
            "        pass",
        ]
    )

    def run(tag: str) -> None:
        text = body.format(marker=str(marker), tag=tag)
        if path.exists():
            _rewrite_same_stat(path, text)
        else:
            path.write_text(text, encoding="utf-8")
        run_grid_search(
            _load(path),
            {"period": [1, 2]},
            data=_grid_df(),
            symbols="GRID",
            max_workers=2,
            return_df=False,
        )

    run("A")
    assert set(marker.read_text()) == {"A"}
    marker.unlink()
    run("B")
    assert set(marker.read_text()) == {"B"}


def test_concurrent_reload_is_thread_safe(tmp_path: Path) -> None:
    """多线程同时加载已改动的文件: 都应拿到完整的新类."""
    path = tmp_path / "reload_i.py"
    path.write_text(_src("A"), encoding="utf-8")
    _load(path)
    _rewrite_same_stat(path, _src("B"))
    results: list[Any] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def work() -> None:
        try:
            barrier.wait()
            results.append(_load(path))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 4
    assert all(cls.VERSION == "B" and cls is results[0] for cls in results)
