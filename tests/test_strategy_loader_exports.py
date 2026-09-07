"""策略加载器 API 的顶层导出一致性测试.

``register_strategy_loader`` / ``resolve_strategy_input`` 早已在 ``akquant``
顶层导出, 但 ``get_strategy_loader`` 没有 —— 同一组 API 一半在顶层、一半
只在子模块, 使用者无从判断该从哪里导入。本文件锁住这组 API 的完整性。
"""

import akquant
import akquant.strategy_loader as loader_mod

# 这组 API 应当同进同出: 注册、解析、按名取, 是同一个使用场景的三个动作。
LOADER_API = (
    "register_strategy_loader",
    "resolve_strategy_input",
    "get_strategy_loader",
    "register_plugin_strategy_loaders",
)


def test_loader_api_is_importable_from_top_level() -> None:
    """整组加载器 API 都应能从 akquant 顶层取到."""
    missing = [name for name in LOADER_API if not hasattr(akquant, name)]
    assert missing == [], f"顶层缺少: {missing}"


def test_loader_api_is_declared_in_all() -> None:
    """整组 API 都应出现在 __all__ 里.

    仅 ``hasattr`` 通过不够: 不进 ``__all__`` 就不算公开契约,
    ``from akquant import *`` 取不到, 文档工具也不会收录。
    """
    missing = [name for name in LOADER_API if name not in akquant.__all__]
    assert missing == [], f"__all__ 缺少: {missing}"


def test_top_level_names_are_the_submodule_objects() -> None:
    """顶层名字必须与子模块是同一对象, 不能是副本或包装."""
    for name in LOADER_API:
        assert getattr(akquant, name) is getattr(loader_mod, name), name


def test_all_has_no_duplicate_loader_entries() -> None:
    """__all__ 里不得重复登记同一名字(两处分支各加一次会导致重复)."""
    for name in LOADER_API:
        assert akquant.__all__.count(name) <= 1, f"{name} 在 __all__ 中重复"


def test_entry_point_group_stays_out_of_top_level() -> None:
    """组名**不**应提升到顶层 —— 与 broker / signal source 保持一致.

    两个既有插件机制都只在自己的模块里定义 ``ENTRY_POINT_GROUP``, 顶层不暴露。
    把这一个提上去反而破坏对称性; 插件作者照文档写 pyproject 即可, 不需要
    在运行期读组名。
    """
    assert not hasattr(akquant, "ENTRY_POINT_GROUP")
    assert not hasattr(akquant, "STRATEGY_LOADER_ENTRY_POINT_GROUP")
