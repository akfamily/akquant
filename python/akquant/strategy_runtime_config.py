# -*- coding: utf-8 -*-
"""``StrategyRuntimeConfig`` 的单一权威定义, 连同归一化、冲突比对与下发.

配置类此前住在 ``strategy.py``, 写入侧 helper 住在 ``backtest/engine.py``, 读取侧
默认值又在 ``strategy_framework_hooks.py`` 里手写了第三份 —— 三处分散的直接后果是
``_RUNTIME_DEFAULTS`` 漏了 ``indicator_mode`` 却没人发现。现在类和写入侧都归本模块,
读取侧从 ``dataclass`` 派生默认值, 漂移在结构上不再可能。

**依赖方向是本模块存在的前提**: 本模块运行时不导入任何本包模块(``Strategy`` 只做
类型注解, 走 ``TYPE_CHECKING``)。因此 ``strategy.py`` 与 ``strategy_framework_hooks.py``
都能顶层导入它而不成环 —— 而反过来把 helper 放进 hooks 是不行的: ``strategy.py:51``
顶层导入 hooks, 而 ``fields()`` / ``isinstance`` / ``**value`` 都是运行时真需要
``StrategyRuntimeConfig`` 的用法, 不是 ``TYPE_CHECKING`` 能绕开的。

``StrategyRuntimeConfig`` 由 ``strategy.py`` 与顶层 ``akquant`` 重新导出, 对外的
``from akquant import StrategyRuntimeConfig`` 写法不受影响。
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Union, cast

if TYPE_CHECKING:
    from .strategy import Strategy


@dataclass
class StrategyRuntimeConfig:
    """策略运行时行为配置."""

    enable_precise_day_boundary_hooks: bool = False
    portfolio_update_eps: float = 0.0
    error_mode: Literal["raise", "continue", "legacy"] = "raise"
    re_raise_on_error: bool = True
    indicator_mode: Literal["incremental", "precompute"] = "precompute"

    def __post_init__(self) -> None:
        """校验并标准化配置."""
        self.portfolio_update_eps = float(self.portfolio_update_eps)
        if self.portfolio_update_eps < 0.0:
            raise ValueError("portfolio_update_eps must be >= 0")
        mode = str(self.error_mode).strip().lower()
        if mode not in {"raise", "continue", "legacy"}:
            raise ValueError("error_mode must be one of: raise, continue, legacy")
        self.error_mode = cast(Literal["raise", "continue", "legacy"], mode)
        indicator_mode = str(self.indicator_mode).strip().lower()
        if indicator_mode not in {"incremental", "precompute"}:
            raise ValueError("indicator_mode must be one of: incremental, precompute")
        self.indicator_mode = cast(Literal["incremental", "precompute"], indicator_mode)
        self.enable_precise_day_boundary_hooks = bool(
            self.enable_precise_day_boundary_hooks
        )
        self.re_raise_on_error = bool(self.re_raise_on_error)


RUNTIME_CONFIG_FIELDS = {f.name for f in fields(StrategyRuntimeConfig)}

RUNTIME_CONFIG_DEFAULTS: Dict[str, Any] = {
    f.name: f.default for f in fields(StrategyRuntimeConfig)
}
"""逐字段默认值, 供读取侧兜底。从 ``dataclass`` 派生而非手写 —— 手写那份曾漏掉
``indicator_mode``, 新增字段时同样的漏项会再次发生。"""


def coerce_strategy_runtime_config(
    value: Union[StrategyRuntimeConfig, Dict[str, Any]],
) -> StrategyRuntimeConfig:
    """把配置对象或字典归一化成一个全新的 ``StrategyRuntimeConfig``.

    传入对象时刻意复制而非直接返回: 下发会写到多个策略实例上, 共享同一个可变
    dataclass 会让其中一个实例的后续改动串到其他实例。
    """
    if isinstance(value, StrategyRuntimeConfig):
        return StrategyRuntimeConfig(
            enable_precise_day_boundary_hooks=value.enable_precise_day_boundary_hooks,
            portfolio_update_eps=value.portfolio_update_eps,
            error_mode=value.error_mode,
            re_raise_on_error=value.re_raise_on_error,
            indicator_mode=value.indicator_mode,
        )
    if isinstance(value, dict):
        unknown_fields = sorted(set(value.keys()) - RUNTIME_CONFIG_FIELDS)
        if unknown_fields:
            allowed = ", ".join(sorted(RUNTIME_CONFIG_FIELDS))
            unknown = ", ".join(unknown_fields)
            raise ValueError(
                "strategy_runtime_config contains unknown fields: "
                f"{unknown}. Allowed fields: {allowed}"
            )
        try:
            return StrategyRuntimeConfig(**value)
        except ValueError as exc:
            raise ValueError(f"invalid strategy_runtime_config: {exc}") from None
    raise TypeError(
        "strategy_runtime_config must be StrategyRuntimeConfig or Dict[str, Any]"
    )


def runtime_config_conflicts(
    current: StrategyRuntimeConfig, incoming: StrategyRuntimeConfig
) -> List[str]:
    """列出入口传入值与策略自设值的逐字段差异, 形如 ``key: before -> after``."""
    conflicts: List[str] = []
    for key in sorted(RUNTIME_CONFIG_FIELDS):
        before = getattr(current, key)
        after = getattr(incoming, key)
        if before != after:
            conflicts.append(f"{key}: {before} -> {after}")
    return conflicts


def apply_strategy_runtime_config(
    strategy_instance: "Strategy",
    incoming: Union[StrategyRuntimeConfig, Dict[str, Any]],
    runtime_config_override: bool,
    logger: Any,
) -> None:
    """把入口传入的运行时配置下发到策略实例, 冲突时按 override 决定取舍并告警.

    告警按 ``(override, 冲突文本)`` 去重: 同一实例被反复下发同一份配置时只警告
    一次, 否则多槽位会话或重复调用会刷屏。
    """
    cfg = coerce_strategy_runtime_config(incoming)
    current = strategy_instance.runtime_config
    conflicts = runtime_config_conflicts(current, cfg)
    if conflicts:
        conflict_text = "; ".join(conflicts)
        warning_key = f"{runtime_config_override}|{conflict_text}"
        warned_keys = getattr(strategy_instance, "_runtime_config_warning_keys", None)
        if not isinstance(warned_keys, set):
            warned_keys = set()
            setattr(strategy_instance, "_runtime_config_warning_keys", warned_keys)
        should_log = warning_key not in warned_keys
        warned_keys.add(warning_key)
        if runtime_config_override:
            if should_log:
                logger.warning(
                    "strategy_runtime_config overrides strategy runtime_config: "
                    f"{conflict_text}"
                )
        else:
            if should_log:
                logger.warning(
                    "strategy_runtime_config is ignored because "
                    f"runtime_config_override=False: {conflict_text}"
                )
            return
    strategy_instance.runtime_config = cfg
