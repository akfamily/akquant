"""指标身份与图上 series 键的单一来源.

指标定义的身份是 ``(owner_strategy_id, indicator_key)``: 主策略(owner
``"_default"``)与指标脚本(owner 为其 snake_case id)可以使用同名键而互不覆盖。

图上的 series 键只在冲突时加前缀: 某个 ``indicator_key`` 只属于一个 owner 时,
series 键就是 ``indicator_key``; 被多个 owner 使用时为 ``"{owner}:{key}"``。
LWC、d3kline、plotly 等所有出口都必须经由 :func:`series_keys` 取键, 不得各自拼接。

本模块是纯函数、零依赖(不导入 ``akquant`` 其他模块), 以便任何出口都能安全导入。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

DEFAULT_OWNER = "_default"

__all__ = ["DEFAULT_OWNER", "display_label", "identity_of", "series_keys"]


def _text(value: Any, default: str = "") -> str:
    # 与 indicator_recording._normalize_text 同口径: None/空白 -> default
    text = str(value or "").strip()
    return text or default


def identity_of(definition: Mapping[str, Any]) -> tuple[str, str]:
    """返回定义(或点/实例行)的身份 ``(owner_strategy_id, indicator_key)``.

    缺失或为空的 owner 按 ``"_default"`` 处理。
    """
    owner = _text(definition.get("owner_strategy_id"), default=DEFAULT_OWNER)
    key = _text(definition.get("indicator_key"))
    return owner, key


def series_keys(defs: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """(owner, key) -> 图上的 series 键: key 只属于一个 owner 时就是 key, 否则 "owner:key".

    :param defs: 指标定义行, 字段同 ``BacktestResult.indicator_definitions`` 的列;
        只用到 ``owner_strategy_id`` 与 ``indicator_key``。缺失 owner 按
        ``"_default"`` 处理, 重复的 ``(owner, key)`` 只算一次。
    :return: 以 ``(owner, key)`` 为键的 series 键映射。
    """
    identities: list[tuple[str, str]] = []
    owners_by_key: dict[str, set[str]] = {}
    for definition in defs:
        identity = identity_of(definition)
        if identity in owners_by_key.get(identity[1], set()):
            continue
        identities.append(identity)
        owners_by_key.setdefault(identity[1], set()).add(identity[0])
    return {
        (owner, key): (key if len(owners_by_key[key]) == 1 else f"{owner}:{key}")
        for owner, key in identities
    }


def display_label(definition: Mapping[str, Any], series_key: str) -> str:
    """图例显示名.

    用户没写 label(``display_name`` 缺失、为空或等于 ``indicator_key``)时返回
    ``series_key``——无冲突时它就是 ``indicator_key``, 冲突时是 ``"owner:key"``;
    用户写了 label 时原样返回该 label。
    """
    key = _text(definition.get("indicator_key"))
    name = _text(definition.get("display_name"), default=key)
    if name == key:
        return series_key
    return name
