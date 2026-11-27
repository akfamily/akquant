"""运行信息 ``run_info``: 由运行入口的调用方传入, 策略经 ``self.run_info`` 只读访问.

框架不解释其中的键, 只做三件事: 在入口处校验取值是简单数据, 深拷贝, 冻结成只读结构。

冻结一直做到最里层: 字典变成只读映射 :class:`ReadOnlyMapping`(包着框架私有的新字典
的 ``types.MappingProxyType``), 列表与元组变成 ``tuple``。原因有二: 多个槽位与指标
脚本共用同一份对象, 只冻顶层的话, 一个槽位改了嵌套字典, 其余槽位读到的也跟着变;
调用方的原始对象不会被引用, 之后再改它也不影响策略。

只收内置的 ``str``/``int``/``float``/``bool``/``None`` 本身(不收子类, 子类可以带可变
属性, 等于留了改动的口子), 以及由它们组成的 ``list``/``tuple``/映射; 键必须是 ``str``;
不收 NaN 与无穷大。这样内容既能 pickle 到网格搜索的子进程, 也能 JSON 序列化。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Dict, Iterator, Optional, Set


class ReadOnlyMapping(Mapping[str, Any]):
    """只读映射: 不能增删改键, 也不能替换内部数据.

    与直接用 ``types.MappingProxyType`` 的区别只在于可以 pickle(按内容重建): 策略把
    ``self.run_info`` 或其中的嵌套字典存进自己的属性后, 存档与多进程仍能序列化它。
    框架自己的 ``run_info`` 属性不入存档(见 ``Strategy.__getstate__``)。
    """

    __slots__ = ("_proxy",)
    _proxy: "MappingProxyType[str, Any]"

    def __init__(self, data: Mapping[str, Any]) -> None:
        """包一份 ``data`` 的浅拷贝; 值须已冻结(由 :func:`freeze_run_info` 保证)."""
        object.__setattr__(self, "_proxy", MappingProxyType(dict(data)))

    def __getitem__(self, key: str) -> Any:
        """按键取值."""
        return self._proxy[key]

    def __iter__(self) -> Iterator[str]:
        """按插入顺序迭代键."""
        return iter(self._proxy)

    def __len__(self) -> int:
        """键的个数."""
        return len(self._proxy)

    def __setattr__(self, name: str, value: Any) -> None:
        """禁止改属性(包括替换内部数据)."""
        raise AttributeError("run_info 是只读的")

    def __delattr__(self, name: str) -> None:
        """禁止删属性."""
        raise AttributeError("run_info 是只读的")

    def __reduce__(self) -> Any:
        """pickle/deepcopy 时按内容重建."""
        return (type(self), (dict(self._proxy),))

    def __repr__(self) -> str:
        """显示为普通字典的样子."""
        return repr(dict(self._proxy))


#: 没传 run_info 时、以及策略交给引擎之前读到的空映射
EMPTY_RUN_INFO: Mapping[str, Any] = ReadOnlyMapping({})

_SCALAR_TYPES = (str, int, float, bool, type(None))
_ALLOWED = "只接受 str/int/float/bool/None 以及由它们组成的 list/tuple/dict"


def _freeze(value: Any, path: str, active: Set[int]) -> Any:
    if type(value) in _SCALAR_TYPES:
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"{path} 不接受 NaN 或无穷大, 收到 {value!r}")
        return value
    if isinstance(value, (Mapping, list, tuple)):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{path} 含循环引用")
        active.add(marker)
        try:
            if isinstance(value, Mapping):
                frozen: Dict[str, Any] = {}
                for key, item in value.items():
                    if type(key) is not str:
                        raise TypeError(
                            f"{path} 的键必须是字符串, 收到 {key!r} "
                            f"({type(key).__name__})"
                        )
                    frozen[key] = _freeze(item, f"{path}[{key!r}]", active)
                return ReadOnlyMapping(frozen)
            return tuple(
                _freeze(item, f"{path}[{index}]", active)
                for index, item in enumerate(value)
            )
        finally:
            active.discard(marker)
    raise TypeError(
        f"{path} {_ALLOWED}, 收到 {type(value).__module__}."
        f"{type(value).__qualname__}: {value!r}"
    )


def freeze_run_info(run_info: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """校验并深拷贝 ``run_info``, 返回只读映射; ``None`` 返回空映射.

    :param run_info: 调用方传入的映射, 或 ``None``
    :return: 只读映射(嵌套字典同样只读, 列表变为 tuple)
    :raises TypeError: 不是映射、键不是字符串, 或含不支持的类型
    :raises ValueError: 含 NaN/无穷大, 或含循环引用
    """
    if run_info is None:
        return EMPTY_RUN_INFO
    if not isinstance(run_info, Mapping):
        raise TypeError(
            f"run_info 须为键为字符串的映射(如 dict), 收到 {type(run_info).__name__}"
        )
    frozen = _freeze(run_info, "run_info", set())
    return frozen if frozen else EMPTY_RUN_INFO


def reject_framework_context_keys(context: Optional[Mapping[str, Any]]) -> None:
    """``context`` 的键会被设成策略属性: 不允许借它改写框架的内部状态.

    以 ``_framework_`` 开头的属性是框架自己的(运行信息、数据周期等), 由运行入口
    写入。放行的话, 能决定 ``context`` 内容的一方(例如把用户参数转成 ``context``
    的宿主程序的用户)就能顶掉调用方传入的 ``run_info``。

    :param context: 运行入口收到的 ``context``, 或 ``None``
    :raises ValueError: 含以 ``_framework_`` 开头的键
    """
    if not context:
        return
    bad = sorted(str(key) for key in context if str(key).startswith("_framework_"))
    if bad:
        raise ValueError(
            f"context 不接受以 _framework_ 开头的键(框架内部状态): {bad}; "
            "运行信息请用 run_info= 传入"
        )


def run_info_to_plain(run_info: Mapping[str, Any]) -> Dict[str, Any]:
    """把冻结后的 ``run_info`` 还原成普通 dict/list, 用于跨进程传给子任务.

    :param run_info: :func:`freeze_run_info` 的返回值
    :return: 内容相同的普通字典与列表
    """

    def thaw(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        return value

    return {key: thaw(item) for key, item in run_info.items()}


def plain_run_info(run_info: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """校验后返回普通字典副本(``None`` 原样返回), 供要再转发给其他入口的地方用.

    :param run_info: 调用方传入的映射, 或 ``None``
    :return: 校验通过的普通字典副本, 或 ``None``
    """
    if run_info is None:
        return None
    return run_info_to_plain(freeze_run_info(run_info))
