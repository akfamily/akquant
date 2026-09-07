import base64
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, cast

from .akquant import Bar
from .log import get_logger
from .strategy import Strategy

logger = get_logger("strategy_loader")

StrategyLike = Union[type[Strategy], Strategy, Callable[[Any, Bar], None]]
StrategySource = Union[str, bytes, os.PathLike[str]]
StrategyLoader = Callable[[StrategySource, Dict[str, Any]], StrategyLike]

ENTRY_POINT_GROUP = "akquant.strategy_loaders"

_LOADERS: Dict[str, StrategyLoader] = {}
_PLUGINS_LOADED = False

# 动态加载的策略模块名前缀. 名字里编码了源文件路径, 因此可逆 ——
# 这是多进程(spawn)下 worker 能重新找回该模块的唯一依据。
_MODULE_PREFIX = "akquant_user_strategy_"


def _encode_module_name(source_path: str) -> str:
    """把源文件路径编码进模块名(稳定且可逆).

    必须稳定: 用 uuid 起名会让同一文件每次加载都换名字, ``pickle`` 无从按名
    引用; 必须可逆: spawn 出来的 worker 只拿到模块名, 得据此找回源文件。

    用 base32 而非 base64/hex: base32 的字符集(A-Z2-7)转小写后仍是合法标识符
    字符, 不含 base64 的 ``+/=``; 相比 hex 又短约四成。
    """
    resolved = str(Path(source_path).resolve())
    encoded = base64.b32encode(resolved.encode("utf-8")).decode("ascii")
    return _MODULE_PREFIX + encoded.rstrip("=").lower()


def _decode_module_name(module_name: str) -> Optional[str]:
    """从模块名解回源文件路径; 非本前缀或无法解码时返回 None."""
    if not module_name.startswith(_MODULE_PREFIX):
        return None
    payload = module_name[len(_MODULE_PREFIX) :].upper()
    payload += "=" * (-len(payload) % 8)
    try:
        return base64.b32decode(payload).decode("utf-8")
    except Exception:  # noqa: BLE001 — 名字不是我们编的, 交回给标准导入流程
        return None


class _DynamicStrategyFinder(importlib.abc.MetaPathFinder):
    """让动态加载的策略模块能在**新解释器进程**中按名找回.

    多进程网格搜索(``optimize.py`` 的 ``ProcessPoolExecutor``)会 pickle 策略类,
    而 pickle 只存 ``module.qualname``。Windows/macOS 默认 spawn, worker 的
    ``sys.modules`` 全新且不继承父进程, 反序列化时必须能重新导入那个模块 ——
    本 finder 就是这条退路: 从模块名解出源文件路径, 重新建 spec。

    安全边界: 只对 ``_MODULE_PREFIX`` 前缀的模块名生效, 且仅在有人真的导入
    该名字时触发(即 unpickle 我们自己编码过的模块), 不会扩大任意文件的可导入面。
    """

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        """按可逆模块名重建 spec; 非本前缀返回 None 交回标准流程."""
        source_path = _decode_module_name(fullname)
        if source_path is None:
            return None
        if not Path(source_path).is_file():
            # 明确报错而非返回 None: 后者只会得到 "No module named <139字符乱码>",
            # 在 worker 里还会连带整个进程池 BrokenProcessPool, 无从排查。
            raise ImportError(
                f"策略源文件已不存在, 无法重新加载动态策略模块: {source_path}",
                name=fullname,
            )
        return importlib.util.spec_from_file_location(fullname, source_path)


def _install_dynamic_strategy_finder() -> None:
    """在 **import akquant** 期安装 finder(幂等).

    时机是关键: spawn 的 worker 在 ``call_queue.get()`` 里就要反序列化任务,
    早于任何用户代码或策略加载调用。因此 finder 必须随框架导入就绪, 不能等到
    首次调用加载器时才装。
    """
    if any(isinstance(finder, _DynamicStrategyFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _DynamicStrategyFinder())


_install_dynamic_strategy_finder()


def _is_strategy_like(value: Any) -> bool:
    if isinstance(value, type) and issubclass(value, Strategy):
        return True
    if isinstance(value, Strategy):
        return True
    if callable(value):
        return True
    return False


def register_strategy_loader(name: str, loader: StrategyLoader) -> None:
    """Register a strategy loader by name."""
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("strategy loader name cannot be empty")
    if not callable(loader):
        raise TypeError("strategy loader must be callable")
    _LOADERS[normalized] = loader


def register_plugin_strategy_loaders() -> None:
    """发现并注册 ``akquant.strategy_loaders`` 插件(幂等、失败隔离).

    与 ``gateway/brokers/plugins.py`` / ``signal/registry.py`` 同一范式:
    单个插件加载失败只记警告, 不拖垮其余插件与调用方。

    发现是**懒的** —— 由 :func:`get_strategy_loader` 首次按名查表时触发,
    而非 ``import akquant`` 期: 插件代码属第三方, 不应在导入框架时无条件执行,
    也不该拖慢 ``import akquant``。

    插件仓的声明方式::

        [project.entry-points."akquant.strategy_loaders"]
        my_loader = "my_pkg.loader:register"
    """
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 — 发现阶段失败也不能拖垮调用方
        logger.warning("发现策略加载器插件失败, 已跳过", exc_info=True)
        return
    for entry in eps:
        try:
            entry.load()()
        except Exception:  # noqa: BLE001 — 单插件失败不拖垮其余
            logger.warning(
                "加载策略加载器插件 %r 失败, 已跳过", entry.name, exc_info=True
            )


def get_strategy_loader(name: str) -> StrategyLoader:
    """Resolve a registered strategy loader."""
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("strategy loader name cannot be empty")
    if normalized not in _LOADERS:
        # 插件提供的加载器要到这一刻才发现: 用户只需在 pyproject 里装上插件包,
        # 不必在脚本里 import 它。
        register_plugin_strategy_loaders()
    if normalized not in _LOADERS:
        available = ", ".join(sorted(_LOADERS)) or "(无)"
        raise ValueError(
            f"unknown strategy_loader: {normalized}; available: {available}"
        )
    return _LOADERS[normalized]


def _load_python_plain(source: StrategySource, options: Dict[str, Any]) -> StrategyLike:
    if isinstance(source, bytes):
        raise TypeError("python_plain loader does not accept bytes source")
    module_path = os.fspath(source)
    # 模块名按源文件路径编码(稳定可逆)并登记进 sys.modules: 两者缺一,
    # pickle 就无法按名引用该类, 多进程网格搜索会在父进程 dumps 时直接失败。
    module_name = _encode_module_name(module_path)
    cached = sys.modules.get(module_name)
    if cached is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"failed to create module spec from: {module_path}")
        module = importlib.util.module_from_spec(spec)
        # 先登记再 exec: 模块体内若 import 自身(或触发循环导入)才能拿到半成品
        # 而非无限递归; exec 失败则回滚, 不留下空壳模块。
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    else:
        module = cached

    attr_name_raw = options.get("strategy_attr")
    attr_name = str(attr_name_raw).strip() if attr_name_raw is not None else ""
    if attr_name:
        picked = getattr(module, attr_name, None)
        if picked is not None:
            if _is_strategy_like(picked):
                return cast(StrategyLike, picked)
            raise TypeError(f"module attr '{attr_name}' is not a valid strategy input")

    candidates = [
        obj
        for obj in module.__dict__.values()
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy
    ]
    if len(candidates) == 1:
        return cast(StrategyLike, candidates[0])
    if not candidates:
        raise ValueError(
            "no Strategy subclass found in module; "
            "provide strategy_attr in loader options"
        )
    raise ValueError(
        "multiple Strategy subclasses found; provide strategy_attr in loader options"
    )


def _load_encrypted_external(
    source: StrategySource, options: Dict[str, Any]
) -> StrategyLike:
    callback = options.get("decrypt_and_load")
    if not callable(callback):
        raise ValueError(
            "encrypted_external loader requires callable option: decrypt_and_load"
        )
    loaded = callback(source, dict(options))
    if not _is_strategy_like(loaded):
        raise TypeError("decrypt_and_load must return Strategy type/instance/callable")
    return cast(StrategyLike, loaded)


def resolve_strategy_input(
    strategy: Optional[StrategyLike] = None,
    strategy_source: Optional[StrategySource] = None,
    strategy_loader: Optional[str] = None,
    strategy_loader_options: Optional[Dict[str, Any]] = None,
) -> StrategyLike:
    """Resolve strategy-like input from direct strategy or source + loader."""
    if strategy is not None:
        return strategy
    if strategy_source is None:
        raise ValueError("Strategy must be provided.")
    if strategy_loader is not None and not isinstance(strategy_loader, str):
        raise TypeError("strategy_loader must be str when provided")
    if strategy_loader_options is not None and not isinstance(
        strategy_loader_options, dict
    ):
        raise TypeError("strategy_loader_options must be dict when provided")
    loader_name = strategy_loader.strip() if strategy_loader else "python_plain"
    options: Dict[str, Any] = dict(strategy_loader_options or {})
    loader = get_strategy_loader(loader_name)
    loaded = loader(strategy_source, options)
    if not _is_strategy_like(loaded):
        raise TypeError("resolved strategy must be Strategy type/instance/callable")
    return loaded


register_strategy_loader("python_plain", _load_python_plain)
register_strategy_loader("encrypted_external", _load_encrypted_external)


__all__ = [
    "StrategyLike",
    "StrategySource",
    "StrategyLoader",
    "ENTRY_POINT_GROUP",
    "register_strategy_loader",
    "register_plugin_strategy_loaders",
    "get_strategy_loader",
    "resolve_strategy_input",
]
