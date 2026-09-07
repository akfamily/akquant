"""python_plain 动态加载策略的多进程(pickle)回归测试.

背景: 加载器给动态模块起名 ``akquant_user_strategy_<uuid>`` 且不登记进
``sys.modules``, 导致父进程 ``pickle.dumps`` 当场失败 —— 即
``strategy_source`` + ``python_plain`` 加载的策略跑 ``max_workers>1`` 的
网格搜索必然报错(且报的是 pickle 的天书信息)。

修法: 稳定可逆的模块名 + 登记 sys.modules + 在 **akquant 导入期**安装
``sys.meta_path`` finder。时机是关键 —— spawn 的 worker 在 ``call_queue.get()``
里就要 unpickle, 早于任何用户代码。
"""

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import akquant.strategy_loader as loader_mod
import pytest
from akquant.strategy_loader import resolve_strategy_input

STRATEGY_SRC = "\n".join(
    [
        "from akquant.strategy import Strategy",
        "",
        "class PickleProbeStrategy(Strategy):",
        "    def __init__(self):",
        "        self.calls = 0",
        "",
        "    def on_bar(self, bar):",
        "        self.calls += 1",
    ]
)


def _write_strategy(tmp_path: Path, name: str = "pickle_probe.py") -> Path:
    """落一个最小策略源文件."""
    path = tmp_path / name
    path.write_text(STRATEGY_SRC, encoding="utf-8")
    return path


def _load(path: Path) -> type:
    """按 python_plain 加载."""
    return resolve_strategy_input(
        strategy_source=str(path),
        strategy_loader="python_plain",
    )  # type: ignore[return-value]


def child_roundtrip(cls: type) -> str:
    """Worker 侧: 收到类并实例化 —— 证明 unpickle 真的可用, 而非只是没报错.

    必须是模块级函数, 否则自身无法被 pickle 引用。
    """
    instance = cls()
    return f"{cls.__name__}:{type(instance).__name__}"


def test_dynamic_strategy_class_is_picklable(tmp_path: Path) -> None:
    """父进程应能 pickle 动态加载的策略类."""
    cls = _load(_write_strategy(tmp_path))
    blob = pickle.dumps(cls)
    assert len(blob) > 0


def test_dynamic_strategy_survives_pickle_roundtrip(tmp_path: Path) -> None:
    """同进程内 loads 应还原为同一个类对象."""
    cls = _load(_write_strategy(tmp_path))
    restored = pickle.loads(pickle.dumps(cls))
    assert restored is cls


def test_same_source_file_reuses_module_name(tmp_path: Path) -> None:
    """同一文件重复加载应得同一模块名(可命中 sys.modules 缓存)."""
    path = _write_strategy(tmp_path)
    first = _load(path)
    second = _load(path)
    assert first.__module__ == second.__module__


def test_module_name_is_registered_in_sys_modules(tmp_path: Path) -> None:
    """动态模块必须登记进 sys.modules, 否则 pickle 无法按名引用."""
    cls = _load(_write_strategy(tmp_path))
    assert cls.__module__ in sys.modules


def test_distinct_files_get_distinct_module_names(tmp_path: Path) -> None:
    """不同文件不得撞名, 否则后加载的会覆盖前一个."""
    first = _load(_write_strategy(tmp_path, "probe_a.py"))
    second = _load(_write_strategy(tmp_path, "probe_b.py"))
    assert first.__module__ != second.__module__


def test_dynamic_strategy_restores_in_process_pool(tmp_path: Path) -> None:
    """真实回归: ProcessPoolExecutor 的 worker 应能还原并实例化该类.

    这是本次修复的核心断言 —— finder 若不在 akquant 导入期安装, worker
    会在反序列化调用队列时 ModuleNotFoundError 并令整个池 BrokenProcessPool。
    """
    cls = _load(_write_strategy(tmp_path))
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = [executor.submit(child_roundtrip, cls) for _ in range(3)]
        got = [future.result(timeout=120) for future in results]
    assert got == ["PickleProbeStrategy:PickleProbeStrategy"] * 3


def test_finder_is_installed_on_import() -> None:
    """Finder 必须随 import akquant 装好, 而非等到首次加载策略."""
    assert any(
        isinstance(finder, loader_mod._DynamicStrategyFinder)
        for finder in sys.meta_path
    )


def test_finder_reports_missing_source_clearly(tmp_path: Path) -> None:
    """源文件被删后, 重新导入该模块应给出可理解的错误而非崩溃."""
    path = _write_strategy(tmp_path)
    cls = _load(path)
    module_name = cls.__module__
    del sys.modules[module_name]
    path.unlink()

    import importlib

    with pytest.raises(ImportError) as exc:
        importlib.import_module(module_name)
    assert "策略源文件" in str(exc.value) or "not found" in str(exc.value).lower()


def test_module_name_roundtrips_to_source_path(tmp_path: Path) -> None:
    """模块名须可逆解回源文件路径 —— 子进程正是靠这一点重新加载."""
    path = _write_strategy(tmp_path)
    cls = _load(path)
    decoded = loader_mod._decode_module_name(cls.__module__)
    assert decoded is not None
    assert Path(decoded).resolve() == path.resolve()


def test_module_name_is_a_valid_identifier(tmp_path: Path) -> None:
    """编码结果必须是合法标识符, 否则 import 机制无法处理该模块名."""
    cls = _load(_write_strategy(tmp_path))
    assert cls.__module__.isidentifier()


@pytest.mark.parametrize(
    "relative",
    [
        "中文目录/中文策略.py",
        "有 空 格/带 空格 策略.py",
        "混合中文 mixed_v2/趋势线网格-策略.py",
    ],
)
def test_non_ascii_and_spaced_paths_roundtrip(tmp_path: Path, relative: str) -> None:
    """中文/空格路径必须可编码可解回.

    锁住 base32(UTF-8) 这一取舍: 若有人把编码"优化"成 ASCII-only 或按
    文件名截断, 中文路径的策略会在多进程下找不回源文件 —— 而中文路径在
    本项目的目标用户里是常态, 不是边缘情况。
    """
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(STRATEGY_SRC, encoding="utf-8")

    cls = _load(target)
    assert cls.__module__.isidentifier()
    decoded = loader_mod._decode_module_name(cls.__module__)
    assert decoded is not None
    assert Path(decoded) == target.resolve()
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_deep_path_module_name_still_works(tmp_path: Path) -> None:
    """深路径产生的长模块名不得被截断或降级.

    实测模块名长到 20 万字符仍可 exec / pickle / 跨进程还原, 且
    ``__pycache__`` 的文件名取自**源文件路径**而非模块名(故不受 255
    字符文件名上限约束)。因此本实现刻意**不设**长度上限 —— 截断会让名字
    不可逆, 反而破坏 worker 找回源文件的唯一依据。
    """
    deep = tmp_path
    while len(str(deep)) < 180:
        deep = deep / "nested_directory_segment"
    deep.mkdir(parents=True, exist_ok=True)
    target = deep / "deep_strategy.py"
    target.write_text(STRATEGY_SRC, encoding="utf-8")

    cls = _load(target)
    # 名字确实很长 —— 若被截断, 下面的路径往返会失败
    assert len(cls.__module__) > 200
    decoded = loader_mod._decode_module_name(cls.__module__)
    assert decoded is not None
    assert Path(decoded) == target.resolve()
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_bytecode_cache_name_is_derived_from_source_not_module(
    tmp_path: Path,
) -> None:
    """字节码缓存名取自源文件, 与模块名长度无关.

    这是"无需长度上限"结论的依据: 若 ``.pyc`` 名字取自模块名, 长名字会撞
    Windows 的 255 字符文件名上限。
    """
    path = _write_strategy(tmp_path)
    cls = _load(path)
    module = sys.modules[cls.__module__]
    cached = getattr(module.__spec__, "cached", None)
    assert cached is not None
    cache_name = Path(cached).name
    assert cache_name.startswith(path.stem)
    assert len(cache_name) < 255
    assert cls.__module__[:40] not in cache_name
