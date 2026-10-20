"""指标身份与图上 series 键的单一来源.

指标定义的身份是 ``(owner_strategy_id, indicator_key)``: 主策略(owner
``"_default"``)与指标脚本(owner 为其 snake_case id)可以使用同名键而互不覆盖。

图上的 series 键只在冲突时加前缀: 某个 ``indicator_key`` 只属于一个 owner 时,
series 键就是 ``indicator_key``; 被多个 owner 使用时为 ``"{owner}:{key}"``。
LWC、d3kline、plotly 等所有出口都必须经由 :func:`series_keys` 取键, 不得各自拼接。

``render_type="candle"`` 的四个分量(``ha.open`` 等, 带 ``group``/``role``)在图上
合成一个 K 线 series。图上出口用 :func:`chart_series_keys`: candle 组以
``(owner, group)``、非 candle 定义以 ``(owner, indicator_key)`` 参与**同一个**命名
空间——另一个 owner 有一条普通线也叫 ``ha`` 时, 双方都带 ``owner:`` 前缀。实时流的
分量 ``series_key`` 仍按分量键(:func:`series_keys`)计算, 消费方按
``(owner_strategy_id, group, symbol, timestamp)`` 把四个分量拼成一根 K 线。

本模块是纯函数、零依赖(不导入 ``akquant`` 其他模块), 以便任何出口都能安全导入。
"""

from __future__ import annotations

import logging
from typing import Any, Collection, Iterable, Mapping, NamedTuple

DEFAULT_OWNER = "_default"

#: 同一 owner 的普通线与 candle 组同名时, 普通线图上键追加的后缀
LINE_SUFFIX = ":line"

_logger = logging.getLogger(__name__)
#: 已告警过的同 owner 撞名 (owner, name), 每个只告警一次
_warned_same_owner: set[tuple[str, str]] = set()

__all__ = [
    "ChartSeriesKeys",
    "DEFAULT_OWNER",
    "LINE_SUFFIX",
    "candle_group_of",
    "chart_series_keys",
    "display_label",
    "group_display_label",
    "identity_of",
    "resolve_series_key",
    "series_keys",
]


def _text(value: Any, default: str = "") -> str:
    # 与 indicator_recording._normalize_text 同口径: None/空白 -> default;
    # 另把 float NaN 视为缺失: DataFrame.to_dict("records") 会把缺列补成 NaN,
    # 而 NaN 为真值, str(NaN or "") 会得到 "nan"
    if value is None or (isinstance(value, float) and value != value):
        return default
    text = str(value).strip()
    return text or default


def identity_of(definition: Mapping[str, Any]) -> tuple[str, str]:
    """返回定义(或点/实例行)的身份 ``(owner_strategy_id, indicator_key)``.

    缺失或为空的 owner 按 ``"_default"`` 处理。
    """
    owner = _text(definition.get("owner_strategy_id"), default=DEFAULT_OWNER)
    key = _text(definition.get("indicator_key"))
    return owner, key


def resolve_series_key(
    owner: str, key: str, owners_by_key: Mapping[str, Collection[str]]
) -> str:
    """按登记表计算单个 ``(owner, key)`` 的 series 键.

    历史出口 :func:`series_keys` 与实时流的录制器/sink 共用本函数, 保证两边
    规则一致: ``key`` 在登记表里至多一个 owner 时就是 ``key``, 否则为
    ``"{owner}:{key}"``。

    :param owner: 已归一的 owner(缺失按 ``"_default"``, 见 :func:`identity_of`)
    :param key: 已归一的 ``indicator_key``
    :param owners_by_key: ``indicator_key`` -> 使用它的 owner 集合
    :return: 图上的 series 键
    """
    if len(owners_by_key.get(key, ())) <= 1:
        return key
    return f"{owner}:{key}"


def series_keys(defs: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """(owner, key) -> 图上的 series 键.

    key 只属于一个 owner 时就是 key, 否则为 ``"owner:key"``。

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
        (owner, key): resolve_series_key(owner, key, owners_by_key)
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


def candle_group_of(definition: Mapping[str, Any]) -> str:
    """返回 candle 分量所属的组名; 不是 candle 分量时返回 ``""``.

    只有 ``render_type == "candle"`` 且 ``group`` 非空的行才算 candle 分量;
    缺 ``group`` 的 candle 行(如旧版第三方存储)按普通线处理。
    """
    render_type = _text(definition.get("render_type")).lower()
    if render_type != "candle":
        return ""
    return _text(definition.get("group"))


class ChartSeriesKeys(NamedTuple):
    """图上出口的 series 键, 见 :func:`chart_series_keys`."""

    #: 非 candle 定义: ``(owner, indicator_key)`` -> 图上的 series 键
    lines: dict[tuple[str, str], str]
    #: candle 组: ``(owner, group)`` -> 图上的 series 键
    groups: dict[tuple[str, str], str]
    #: 全部定义按分量键算的 series 键(= :func:`series_keys`, 与实时流一致)
    components: dict[tuple[str, str], str]


def chart_series_keys(defs: Iterable[Mapping[str, Any]]) -> ChartSeriesKeys:
    """计算图上出口(LWC / d3kline / raw / plotly)共用的 series 键.

    candle 组以 ``(owner, group)``、非 candle 定义以 ``(owner, indicator_key)``
    进入同一个命名空间, 再按 :func:`resolve_series_key` 的规则(只有一个 owner
    用这个名字时不加前缀, 否则 ``"owner:name"``)取键。

    同一 owner 既有普通线 ``ha``(如直接 ``record_indicator("ha")``)又有 candle
    组 ``ha`` 时, 两者按上面的规则会得到同一个键; 此时 candle 组保持原键, 普通线
    的键追加 :data:`LINE_SUFFIX`(``"ha:line"``), 并对每个 ``(owner, name)`` 只打
    一次 warning。

    :param defs: 指标定义行(字段同 ``BacktestResult.indicator_definitions``)。
    :return: :class:`ChartSeriesKeys`。
    """
    defs = list(defs)
    owners_by_name: dict[str, set[str]] = {}
    line_ids: list[tuple[str, str]] = []
    group_ids: list[tuple[str, str]] = []
    for definition in defs:
        owner, key = identity_of(definition)
        group = candle_group_of(definition)
        name = group or key
        if not name:
            continue
        target = group_ids if group else line_ids
        if (owner, name) not in target:
            target.append((owner, name))
        owners_by_name.setdefault(name, set()).add(owner)
    group_set = set(group_ids)
    lines: dict[tuple[str, str], str] = {}
    for ident in line_ids:
        key = resolve_series_key(ident[0], ident[1], owners_by_name)
        if ident in group_set:
            key = f"{key}{LINE_SUFFIX}"
            if ident not in _warned_same_owner:
                _warned_same_owner.add(ident)
                _logger.warning(
                    "owner %r 的普通线与 candle 组同名 %r, 图上普通线改名为 %r",
                    ident[0],
                    ident[1],
                    key,
                )
        lines[ident] = key
    return ChartSeriesKeys(
        lines=lines,
        groups={
            ident: resolve_series_key(ident[0], ident[1], owners_by_name)
            for ident in group_ids
        },
        components=series_keys(defs),
    )


def group_display_label(definition: Mapping[str, Any], series_key: str) -> str:
    """返回 candle 组的图例显示名.

    用户没写 label(``display_name`` 缺失、为空、等于组名或分量键)时返回组的
    ``series_key``, 否则原样返回该 label。
    """
    key = _text(definition.get("indicator_key"))
    group = _text(definition.get("group"))
    name = _text(definition.get("display_name"))
    if not name or name in {key, group}:
        return series_key
    return name
