import json
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

from .chart import (
    bar_end_ns,
    normalize_group_role,
    normalize_pane_index,
    normalize_reference_lines,
    normalize_render_type,
    normalize_scale_group,
    timestamp_ms_from_ns,
    timestamp_to_ms_and_ns,
)
from .chart import (
    normalize_meta_json as _normalize_meta_json,
)
from .indicator_identity import DEFAULT_OWNER, identity_of, resolve_series_key


@runtime_checkable
class IndicatorSink(Protocol):
    """Public protocol for objects that collect indicator points during a run.

    Pass an implementation to ``run_backtest(indicator_recorder=...)`` to route
    every ``Strategy.record_indicator`` call into your own collector instead of
    the built-in :class:`IndicatorRecorder`. This is the supported extension
    point for chart backends and frontends: implement ``record`` to capture
    points and ``build_payload`` to return the definitions/instances/points
    payload attached to :class:`BacktestResult.indicator_outputs`.

    The built-in :class:`IndicatorRecorder` satisfies this protocol.

    A sink may additionally implement the **optional**
    ``declare(owner_strategy_id, indicator_key)`` method (1.5): the framework
    calls it, when present, for every ``plot=True`` ``Strategy.I()`` declaration
    so the sink can compute stream ``series_key`` values before the first point.
    It is detected with ``getattr`` and deliberately not part of this protocol,
    so sinks written against 1.4 remain valid.
    """

    def record(
        self,
        *,
        name: str,
        value: Any,
        symbol: str,
        timestamp: Any,
        owner_strategy_id: str,
        display_name: Optional[str] = ...,
        pane: int = ...,
        render_type: str = ...,
        unit: Optional[str] = ...,
        precision: Optional[int] = ...,
        color: Optional[str] = ...,
        meta: Optional[Dict[str, Any]] = ...,
        reference_lines: Optional[list[Dict[str, Any]]] = ...,
        scale_group: Optional[str] = ...,
        warmup: bool = ...,
        confirmed: bool = ...,
        phase: str = ...,
        group: str = ...,
        role: str = ...,
        interval_start: Optional[int] = ...,
    ) -> None:
        """Record one indicator point.

        ``phase`` is ``"preload"`` for points replayed from
        ``run_live(preload_history=)`` and ``"run"`` otherwise (1.4). Sinks
        written against the 1.3 protocol may omit it; the framework detects
        that and simply does not pass it.

        ``interval_start`` (bar 区间起点, UTC 纳秒, 可选): 点由 tick 合成的终点戳
        bar 触发时由框架传入, sink 据此把对外毫秒时间取整到区间终点(整分钟);
        不接受该参数的 sink 收不到, 行为与此前一致。

        ``group``/``role`` (1.6) identify one OHLC component of a
        ``render_type="candle"`` declaration: ``group`` is the ``Strategy.I``
        name and ``role`` is ``open``/``high``/``low``/``close``; both are
        ``""`` for every other render type. Sinks written against the 1.5
        protocol may omit them; the framework then does not pass them and
        downgrades the candle components to ``render_type="line"`` for that
        sink (four separate lines), since a candle point without its group
        cannot be assembled into a bar.
        """
        ...

    def build_payload(self) -> Dict[str, list[Dict[str, Any]]]:
        """Return the row-oriented definitions/instances/points payload."""
        ...

    def flush_stream_snapshot(self) -> None:
        """Emit any buffered stream snapshot for the current callback cycle."""
        ...

    def set_stream_emitter(
        self,
        stream_emitter: Optional[
            Callable[[str, Optional[str], str, Dict[str, str]], None]
        ],
    ) -> None:
        """Attach or replace the runtime stream emitter."""
        ...


def _normalize_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def all_items_confirmed(items: list[Dict[str, Any]]) -> str:
    """快照级 confirmed: 任一 item 是临时值则整份快照标 false.

    回测 recorder 与实盘 sink 共用, 两边的快照口径必须严格一致。
    """
    return "true" if all(bool(i.get("confirmed", True)) for i in items) else "false"


def snapshot_phase(items: list[Dict[str, Any]]) -> str:
    """快照级 phase: 全部 item 都来自预热回放才标 preload, 否则 run.

    回测 recorder 与实盘 sink 共用, 两边的快照口径必须严格一致。
    """
    if items and all(i.get("phase") == "preload" for i in items):
        return "preload"
    return "run"


def _normalize_timestamp_ns(timestamp: Any) -> int:
    _, timestamp_ns = timestamp_to_ms_and_ns(timestamp)
    return timestamp_ns


class SeriesKeyRegistry:
    """会话级登记表: 实时流消息的 ``series_key`` 按它在发出时计算.

    回测录制器与实盘 sink 共用, 两边口径必须一致。登记来源: 挂载时补登已有的
    ``plot=True`` 声明、挂载后 ``self.I`` 的声明(经 ``declare``)、每次
    ``record``。命名规则只来自 :func:`akquant.indicator_identity.resolve_series_key`。

    某个 key 已经以不带前缀的名字发出过、之后才有第二个 owner 用它时(运行中途
    撞名), 该 key 的所有 owner 从此刻起都改用 ``owner:key``, 已发出的点不追溯;
    每个 key 只告警一次。
    """

    def __init__(self) -> None:
        """初始化空登记表."""
        self._owners_by_key: Dict[str, set[str]] = {}
        # 已经以不带前缀的名字发出过流消息的 key
        self._emitted_plain: set[str] = set()
        self._warned: set[str] = set()

    def register(self, owner_strategy_id: Any, indicator_key: Any) -> None:
        """登记一个 ``(owner, key)``; owner 缺失/空白/NaN 按 ``"_default"``."""
        owner, key = identity_of(
            {"owner_strategy_id": owner_strategy_id, "indicator_key": indicator_key}
        )
        if not key:
            return
        owners = self._owners_by_key.setdefault(key, set())
        if owner in owners:
            return
        owners.add(owner)
        if len(owners) > 1 and key in self._emitted_plain and key not in self._warned:
            self._warned.add(key)
            from .log import get_logger

            get_logger("indicator").warning(
                "指标键 %r 在运行中途被第二个 owner(%s)使用: 从此刻起实时流的 "
                "series_key 改为 'owner:%s'(涉及 %s), 此前已发出的点仍是 %r, "
                "不追溯修改。在 __init__/on_start 里用 self.I(plot=True) 声明, "
                "或改用不同的 name, 可避免这一切换",
                key,
                owner,
                key,
                ", ".join(sorted(owners)),
                key,
            )

    def series_key(self, owner_strategy_id: Any, indicator_key: Any) -> str:
        """返回流消息此刻应带的 ``series_key``(并记下不带前缀的发出)."""
        owner, key = identity_of(
            {"owner_strategy_id": owner_strategy_id, "indicator_key": indicator_key}
        )
        resolved = resolve_series_key(owner, key, self._owners_by_key)
        if resolved == key:
            self._emitted_plain.add(key)
        return resolved


def declare_plotted_indicators(
    strategy: Any, recorder: Any, names: Optional[list[str]] = None
) -> None:
    """把策略的 ``plot=True`` 声明登记到 sink(可选的 ``declare`` 方法).

    挂载录制器时不传 ``names``, 补登全部已有声明 —— ``__init__`` 里的 ``self.I``
    早于挂载, 只能在这里补登; ``self.I`` 在挂载后声明时只传它自己的名字。
    sink 没实现 ``declare``、或策略没有声明表(测试替身)时什么都不做。
    """
    declare = getattr(recorder, "declare", None)
    if not callable(declare):
        return
    declarations = getattr(strategy, "_incremental_indicators", None)
    if not isinstance(declarations, dict):
        return
    owner = getattr(strategy, "_owner_strategy_id", None) or DEFAULT_OWNER
    selected = list(declarations) if names is None else names
    for name in selected:
        declaration = declarations.get(name)
        if declaration is None or not getattr(declaration, "plot", False):
            continue
        for key, _, _ in declaration.report_items(name, None):
            declare(owner, key)


def stamp_snapshot_items(
    registry: SeriesKeyRegistry, owner_strategy_id: str, items: list[Dict[str, Any]]
) -> None:
    """快照发出时给每个 item 补上 ``owner_strategy_id`` 与此刻的 ``series_key``.

    回测 recorder 与实盘 sink 共用, 两边的快照口径必须严格一致。
    """
    for item in items:
        item["owner_strategy_id"] = owner_strategy_id
        item["series_key"] = registry.series_key(
            owner_strategy_id, item.get("indicator_key")
        )


class IndicatorRecorder:
    """Collect normalized indicator metadata and point values during a run."""

    def __init__(
        self,
        stream_emitter: Optional[
            Callable[[str, Optional[str], str, Dict[str, str]], None]
        ] = None,
    ) -> None:
        """Initialize recorder state and optional stream emitter."""
        # 定义身份是 (owner_strategy_id, indicator_key): 主策略与指标脚本可同名
        self._definitions: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._instances: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        self._points: list[Dict[str, Any]] = []
        self._stream_emitter = stream_emitter
        # 键末位是快照的毫秒时间(按 bar 区间终点取整), 与纳秒戳分开存
        self._pending_snapshots: Dict[
            Tuple[str, str, int, int], list[Dict[str, Any]]
        ] = {}
        self._series_registry = SeriesKeyRegistry()

    def declare(self, owner_strategy_id: str, indicator_key: str) -> None:
        """登记一个将要上报的 ``(owner, key)``, 供流消息提前算出 series_key."""
        self._series_registry.register(owner_strategy_id, indicator_key)

    @property
    def has_data(self) -> bool:
        """Return whether any indicator point has been recorded."""
        return bool(self._points)

    def set_stream_emitter(
        self,
        stream_emitter: Optional[
            Callable[[str, Optional[str], str, Dict[str, str]], None]
        ],
    ) -> None:
        """Attach or replace the runtime stream emitter."""
        self._stream_emitter = stream_emitter

    def record(
        self,
        *,
        name: str,
        value: Any,
        symbol: str,
        timestamp: Any,
        owner_strategy_id: str,
        display_name: Optional[str] = None,
        pane: int = 0,
        render_type: str = "line",
        unit: Optional[str] = None,
        precision: Optional[int] = None,
        color: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        reference_lines: Optional[list[Dict[str, Any]]] = None,
        scale_group: Optional[str] = None,
        warmup: bool = False,
        confirmed: bool = True,
        phase: str = "run",
        group: str = "",
        role: str = "",
        interval_start: Optional[int] = None,
    ) -> None:
        """Record one indicator definition, instance, and point update.

        ``confirmed=False`` 的临时点**不进** ``_points``(DataFrame / export 只含
        确认值), 只发流事件 —— 与实盘 sink "只发不累积" 的口径一致。
        """
        indicator_key = _normalize_text(name)
        if not indicator_key:
            raise ValueError("indicator name cannot be empty")

        symbol_text = _normalize_text(symbol, default="_unknown")
        strategy_id = _normalize_text(owner_strategy_id, default="_default")
        pane_index = normalize_pane_index(pane)
        render_type_text = normalize_render_type(render_type)
        group_text, role_text = normalize_group_role(render_type_text, group, role)
        display_name_text = _normalize_text(display_name, default=indicator_key)
        unit_text = _normalize_text(unit)
        color_text = _normalize_text(color)
        meta_json = _normalize_meta_json(meta)
        reference_lines_norm = normalize_reference_lines(reference_lines)
        scale_group_norm = normalize_scale_group(scale_group)
        timestamp_ns = _normalize_timestamp_ns(timestamp)
        timestamp_ms = timestamp_ms_from_ns(bar_end_ns(timestamp_ns, interval_start))
        phase_text = _normalize_text(phase, default="run")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = float("nan")

        definition_key = (strategy_id, indicator_key)
        definition = self._definitions.get(definition_key)
        if definition is None:
            self._definitions[definition_key] = {
                "owner_strategy_id": strategy_id,
                "indicator_key": indicator_key,
                "display_name": display_name_text,
                "pane": pane_index,
                "render_type": render_type_text,
                "unit": unit_text,
                "precision": precision,
                "color": color_text,
                "reference_lines": reference_lines_norm,
                "scale_group": scale_group_norm,
                "group": group_text,
                "role": role_text,
            }
        else:
            if not definition.get("display_name"):
                definition["display_name"] = display_name_text
            if definition.get("pane") is None:
                definition["pane"] = pane_index
            if not definition.get("render_type"):
                definition["render_type"] = render_type_text
            if not definition.get("unit"):
                definition["unit"] = unit_text
            if definition.get("precision") is None and precision is not None:
                definition["precision"] = precision
            if not definition.get("color"):
                definition["color"] = color_text
            if not definition.get("reference_lines") and reference_lines_norm:
                definition["reference_lines"] = reference_lines_norm
            if not definition.get("scale_group") and scale_group_norm:
                definition["scale_group"] = scale_group_norm
            if not definition.get("group") and group_text:
                definition["group"] = group_text
                definition["role"] = role_text

        instance_key = (strategy_id, symbol_text, indicator_key, meta_json)
        instance = self._instances.get(instance_key)
        if instance is None:
            instance = {
                "instance_id": f"ins_{len(self._instances) + 1}",
                "owner_strategy_id": strategy_id,
                "symbol": symbol_text,
                "indicator_key": indicator_key,
                "meta_json": meta_json,
            }
            self._instances[instance_key] = instance

        if confirmed:
            self._points.append(
                {
                    "instance_id": instance["instance_id"],
                    "owner_strategy_id": strategy_id,
                    "symbol": symbol_text,
                    "indicator_key": indicator_key,
                    "timestamp": timestamp_ns,
                    "timestamp_ms": timestamp_ms,
                    "value": numeric_value,
                    "warmup": bool(warmup),
                }
            )
        self._series_registry.register(strategy_id, indicator_key)
        if self._stream_emitter is not None:
            self._stream_emitter(
                "indicator_point",
                None if symbol_text == "_unknown" else symbol_text,
                "info",
                {
                    "owner_strategy_id": strategy_id,
                    "indicator_key": indicator_key,
                    "series_key": self._series_registry.series_key(
                        strategy_id, indicator_key
                    ),
                    "display_name": display_name_text,
                    "pane": str(pane_index),
                    "render_type": render_type_text,
                    "symbol": symbol_text,
                    "timestamp": str(timestamp_ns),
                    "timestamp_ms": str(timestamp_ms),
                    "value": repr(numeric_value),
                    "warmup": str(bool(warmup)).lower(),
                    "confirmed": str(bool(confirmed)).lower(),
                    "phase": phase_text,
                    "scale_group": scale_group_norm,
                    "group": group_text,
                    "role": role_text,
                    "meta_json": meta_json,
                },
            )
            snapshot_key = (strategy_id, symbol_text, timestamp_ns, timestamp_ms)
            snapshot_items = self._pending_snapshots.setdefault(snapshot_key, [])
            snapshot_items.append(
                {
                    "indicator_key": indicator_key,
                    "display_name": display_name_text,
                    "pane": pane_index,
                    "render_type": render_type_text,
                    "value": numeric_value,
                    "warmup": bool(warmup),
                    "confirmed": bool(confirmed),
                    "phase": phase_text,
                    "group": group_text,
                    "role": role_text,
                    "meta_json": meta_json,
                }
            )

    def flush_stream_snapshot(self) -> None:
        """Emit pending indicator snapshots for the latest callback cycle."""
        if self._stream_emitter is None or not self._pending_snapshots:
            self._pending_snapshots.clear()
            return
        for (strategy_id, symbol_text, timestamp_ns, timestamp_ms), items in list(
            self._pending_snapshots.items()
        ):
            stamp_snapshot_items(self._series_registry, strategy_id, items)
            self._stream_emitter(
                "indicator_snapshot",
                None if symbol_text == "_unknown" else symbol_text,
                "info",
                {
                    "owner_strategy_id": strategy_id,
                    "symbol": symbol_text,
                    "timestamp": str(timestamp_ns),
                    "timestamp_ms": str(timestamp_ms),
                    "indicator_count": str(len(items)),
                    "confirmed": all_items_confirmed(items),
                    "phase": snapshot_phase(items),
                    "items_json": json.dumps(
                        items,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                },
            )
        self._pending_snapshots.clear()

    def build_payload(self) -> Dict[str, list[Dict[str, Any]]]:
        """Return stable row-oriented outputs for result access and export."""
        definitions = [
            self._definitions[key] for key in sorted(self._definitions.keys())
        ]
        instances = sorted(
            self._instances.values(),
            key=lambda row: (
                str(row.get("owner_strategy_id", "")),
                str(row.get("symbol", "")),
                str(row.get("indicator_key", "")),
                str(row.get("instance_id", "")),
            ),
        )
        points = sorted(
            self._points,
            key=lambda row: (
                str(row.get("owner_strategy_id", "")),
                str(row.get("symbol", "")),
                str(row.get("indicator_key", "")),
                int(row.get("timestamp", 0)),
            ),
        )
        return {
            "definitions": definitions,
            "instances": instances,
            "points": points,
        }
